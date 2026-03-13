import torch
import os
import cv2

import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import Optional

from torchvision import transforms as vt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from utils.util import get_prompt_template
from utils.viz import draw_overall, draw_overlaid

import datasets.vggsound.eval_utils as vggsound_eval
import datasets.VGGSS.eval_utils as vggss_eval
import datasets.VGGSS.extend_eval_utils as exvggss_eval
import datasets.Flickr.eval_utils as flickr_eval
import datasets.Flickr.extend_eval_utils as exflickr_eval
import datasets.AVSBench.eval_utils as avsbench_eval
import datasets.AVATAR.eval_utils as avatar_eval
from datasets.silence_and_noise.silence_and_noise import get_silence_noise_audios

from typing import List, Optional, Tuple, Dict
from importlib import import_module
from contextlib import nullcontext

import torch.nn.functional as F

import wandb
import sys

@torch.no_grad()
def eval_vggsound_validation(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int],
    tensorboard_path: Optional[str] = None,
    rank = 0,
    wandb_run: Optional[wandb.Run] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    use_amp = False
):
    '''
    Evaluate provided model on VGG-Sound validation dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        val_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        loss and other things
    '''
    gt_resolution = (args.ground_truth_resolution, args.ground_truth_resolution)

    autocast_fn = nullcontext
    if use_amp:
        autocast_fn = autocast

    loss_dict = {}
    total_loss_per_epopch = 0.0
    loss_add_count = 0.0

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    # test_split = val_dataloader.dataset.split

    loss_per_epoch_dict = {loss_name: 0.0 for loss_name in args.loss}

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        val_dataloader.dataset[0]['audios'].shape,
        args.san,
        real_san_audio_path,
        val_dataloader.dataset.SAMPLE_RATE,
        val_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    if use_cuda and neg_audios != None:
        for key in neg_audios.keys():
            neg_audios[key] = neg_audios[key].half()

    san_dict = {'san': args.san, 'san_real': args.san_real, **neg_audios}

    pbar = tqdm(val_dataloader, desc=f"Validation Epoch [{epoch}/{args.epoch}]", disable=(rank != 0))

    for step, data in enumerate(pbar):
        images, audios, name = data['images'], data['audios'], data['ids']
        noisy_audios = data['noisy_audios']

        if use_cuda:
            images = images.half()

        audio_embeddings = {}

        with autocast_fn():

            placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
            placeholder_tokens = placeholder_tokens.repeat((val_dataloader.batch_size, 1))
            audio_driven_embedding = model.encode_audio(
                audios.to(model.device),
                placeholder_tokens,
                text_pos_at_prompt,
                prompt_length
            )

            if use_cuda:
                audio_driven_embedding = audio_driven_embedding.half()

            audio_embeddings['pred_emb'] = audio_driven_embedding

            if 'diff_san_l' in args.loss:
                audio_driven_embedding_noisy = model.encode_audio(
                    noisy_audios.to(model.device),
                    placeholder_tokens,
                    text_pos_at_prompt,
                    prompt_length
                )

                if use_cuda:
                    audio_driven_embedding_noisy = audio_driven_embedding_noisy.half()

                audio_embeddings['pred_emb_noisy'] = audio_driven_embedding_noisy

            if 'silence_l' in args.loss and args.san:
                audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            if 'noise_l' in args.loss and args.san:
                audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

            # contains also SaN outputs if set
            out_dict = model.forward_for_validation(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

            loss_args = {**out_dict, **san_dict, **audio_embeddings}

            for j, loss_name in enumerate(args.loss):
                loss_dict[loss_name] = getattr(import_module('utils.loss'), loss_name)(**loss_args) * args.loss_w[j]
                loss_per_epoch_dict[loss_name] += loss_dict[loss_name].item()
            loss = torch.sum(torch.stack(list(loss_dict.values())))

        # Visual results
        for j in range(val_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1].detach()
            seg_image = ((1 - seg.squeeze().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

            if step < 2 and wandb_run and rank == 0:
                heatmap_image = cv2.applyColorMap(((seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                original_image = Image.open(os.path.join(val_dataloader.dataset.image_path, name[j] + '.jpg')).resize(gt_resolution)
                overlaid_image = cv2.addWeighted(np.array(original_image), 0.5, heatmap_image, 0.5, 0)

                wandb_run.log({
                    f'images/val_overlaid/{name[j]}.jpg': wandb.Image(overlaid_image),
                    'trainer/epoch': epoch
                })

        total_loss_per_epopch += loss.item()
        loss_add_count += 1.0

        avr_loss = total_loss_per_epopch / loss_add_count

        if rank == 0:
            pbar.set_description(f"Validation Epoch {epoch}, Loss = {round(avr_loss, 5)}")

            if wandb_run:
                val_step = (epoch * len(val_dataloader)) + step

                wandb_run.log({
                    **{f'validation_losses/step/{key}': val for key, val in loss_dict.items()},
                    **{f'validation_losses/avr/{key}': val / loss_add_count for key, val in loss_per_epoch_dict.items()},
                    'validation/step/loss': loss.item(),
                    'validation/avr/loss': avr_loss,
                    'trainer/val_step': val_step
                })

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    del out_dict, neg_audios, audio_embeddings, loss_args, loss, san_dict, loss_dict

    return float(total_loss_per_epopch / loss_add_count)


@torch.no_grad()
def eval_vggsound_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> Dict[str, float]:
    '''
    Evaluate provided model on VGGS (VGG-Sound) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        result_dict (Dict): Best AUC value (threshold optimized)

    Notes:
        The evaluation includes threshold optimization for VGG-SS.
    '''

    gt_resolution = (args.ground_truth_resolution, args.ground_truth_resolution)

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators = [vggsound_eval.Evaluator() for i in range(len(thrs))]

    if snr != None:
        # Special case: a snr is passed, therefore silence and noise are not computed.
        # VGGSound does not have original (std) evaluation, so there is nothing to do here
        return {}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate VGGS dataset ({test_split})...")):
        images, audios = data['images'], data['audios']
        labels, name = data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

        audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(**out_dict, thr=thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(gt_resolution)

            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
            heatmap_image = Image.fromarray(seg_image)

            if 'sil_heatmap' in out_dict and 'noise_heatmap' in out_dict:
                seg = out_dict['sil_heatmap'][j:j+1]
                seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
                heatmap_image_silence = Image.fromarray(seg_image)

                seg = out_dict['noise_heatmap'][j:j+1]
                seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
                heatmap_image_noise = Image.fromarray(seg_image)

                draw_overall(result_dir, original_image, heatmap_image, heatmap_image_silence, heatmap_image_noise, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)
    heatmap_sil_min_values = np.array(heatmap_sil_min_values)
    heatmap_sil_max_values = np.array(heatmap_sil_max_values)
    heatmap_noise_min_values = np.array(heatmap_noise_min_values)
    heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    epoch = 0 if epoch == 'best' else epoch

    # Final result
    best_AUC_silence = [0.0, 0.0]
    best_AUC_noise = [0.0, 0.0]

    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
        msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/silence/{test_split}({thr})', silence_metrics, epoch)
            best_AUC_silence = [silence_metrics['AUC_N'], thr] if best_AUC_silence[0] < silence_metrics['AUC_N'] else best_AUC_silence

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
        msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/noise/{test_split}({thr})', noise_metrics, epoch)
            best_AUC_noise = [noise_metrics['AUC_N'], thr] if best_AUC_noise[0] < noise_metrics['AUC_N'] else best_AUC_noise

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    result_dict = {'epoch': epoch, 'best_AUC_silence': best_AUC_silence, 'best_AUC_noise': best_AUC_noise}

    return result_dict

@torch.no_grad()
def eval_vggss_get_thresholds(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    rank = 0
) -> Dict[str, float]:

    if snr != None:
        return {}

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    for step, data in enumerate(tqdm(test_dataloader, desc=f"[{epoch}] Evaluating thresholds in VGG-SS dataset ({test_split})...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

        audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)
    heatmap_sil_min_values = np.array(heatmap_sil_min_values)
    heatmap_sil_max_values = np.array(heatmap_sil_max_values)
    heatmap_noise_min_values = np.array(heatmap_noise_min_values)
    heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    epoch = 0 if epoch == 'best' else epoch

    if tensorboard_path is not None and epoch is not None:
        numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
        os.makedirs(numpy_path, exist_ok=True)
        heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min.pkl.{rank}'))
        heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max.pkl.{rank}'))
        heatmap_sil_min_values.dump(os.path.join(numpy_path, test_split + f'_sil_min.pkl.{rank}'))
        heatmap_sil_max_values.dump(os.path.join(numpy_path, test_split + f'_sil_max.pkl.{rank}'))
        heatmap_noise_min_values.dump(os.path.join(numpy_path, test_split + f'_noise_min.pkl.{rank}'))
        heatmap_noise_max_values.dump(os.path.join(numpy_path, test_split + f'_noise_max.pkl.{rank}'))

    max_negatives = [heatmap_sil_max_values, heatmap_noise_max_values]
    max_negatives_separate = [np.percentile(heatmap_sil_max_values, 75), np.percentile(heatmap_noise_max_values, 75)]

    return_thresholds = {
        'max_neg': np.mean(heatmap_pos_max_values),
        'max_neg_plus_10': np.mean(heatmap_pos_max_values) * 1.1,
        'max_q2_pos': np.percentile(heatmap_pos_max_values, 25),
        'max_q3_all': np.percentile(max_negatives, 75),
        'max_q3_separate': np.max(max_negatives_separate)
    }

    print(return_thresholds)

    return return_thresholds

@torch.no_grad()
def eval_vggss_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> Dict[str, float]:
    '''
    Evaluate provided model on VGG-SS (VGG Sound Source) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        result_dict (Dict): Best AUC value (threshold optimized)

    Notes:
        The evaluation includes threshold optimization for VGG-SS.
    '''
    gt_resolution = (args.ground_truth_resolution, args.ground_truth_resolution)

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators = [vggss_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate VGG-SS dataset ({test_split})...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        if snr == None:
            audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        if snr != None:
            heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(**out_dict, target=bboxes, thr=thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(gt_resolution)
            gt_image = vt.ToPILImage()(bboxes[j]).resize(gt_resolution).point(lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize(gt_resolution)
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize(gt_resolution).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # only these two because non-snr are already computed by eval_vggss_get_thresholds
    if snr != None:
        heatmap_pos_min_values = np.array(heatmap_pos_min_values)
        heatmap_pos_max_values = np.array(heatmap_pos_max_values)

        if tensorboard_path is not None and epoch is not None:
            numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
            os.makedirs(numpy_path, exist_ok=True)
            heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min_snr{snr}.pkl'))
            heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max_snr{snr}.pkl'))

    epoch = 0 if epoch == 'best' else epoch

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    best_AUC = [0.0, 0.0]
    best_AUC_silence = [0.0, 0.0]
    best_AUC_noise = [0.0, 0.0]

    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/{test_split}({thr})', std_metrics, epoch)

        best_AUC = [std_metrics['AUC'], thr] if best_AUC[0] < std_metrics['AUC'] else best_AUC

        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/{test_split}({thr})', silence_metrics, epoch)
                best_AUC_silence = [silence_metrics['AUC_N'], thr] if best_AUC_silence[0] < silence_metrics['AUC_N'] else best_AUC_silence

            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/{test_split}({thr})', noise_metrics, epoch)
                best_AUC_noise = [noise_metrics['AUC_N'], thr] if best_AUC_noise[0] < noise_metrics['AUC_N'] else best_AUC_noise

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    result_dict = {'epoch': epoch, 'best_AUC': best_AUC, 'best_AUC_silence': best_AUC_silence, 'best_AUC_noise': best_AUC_noise}

    return result_dict


@torch.no_grad()
def eval_avsbench_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    gt_resolution = (args.ground_truth_resolution, args.ground_truth_resolution)

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.setting

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators = [avsbench_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate AVSBench dataset ({test_split})...")):
        images, audios, gts, labels, name = data['images'], data['audios'], data['gts'], data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        if snr == None:
            audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()

        if snr == None:
            heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(**out_dict, target=gts.to(model.device), thr=thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.png')).resize(gt_resolution)
            gt_image = Image.open(os.path.join(test_dataloader.dataset.gt_path, name[j] + '.png')).resize(gt_resolution).point(
                lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize(gt_resolution)
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize(gt_resolution).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)

    if snr == None:
        heatmap_sil_min_values = np.array(heatmap_sil_min_values)
        heatmap_sil_max_values = np.array(heatmap_sil_max_values)
        heatmap_noise_min_values = np.array(heatmap_noise_min_values)
        heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    if tensorboard_path is not None and epoch is not None:
        numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
        os.makedirs(numpy_path, exist_ok=True)
        heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))
        heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))

        if snr == None:
            heatmap_sil_min_values.dump(os.path.join(numpy_path, test_split + '_sil_min.pkl'))
            heatmap_sil_max_values.dump(os.path.join(numpy_path, test_split + '_sil_max.pkl'))
            heatmap_noise_min_values.dump(os.path.join(numpy_path, test_split + '_noise_min.pkl'))
            heatmap_noise_max_values.dump(os.path.join(numpy_path, test_split + '_noise_max.pkl'))

    epoch = 0 if epoch == 'best' else epoch

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += f'{std_metrics["mIoU"]=}, {std_metrics["Fmeasure"]=}\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/avs/{test_split}({thr})', std_metrics, epoch)

        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/avs/{test_split}({thr})', silence_metrics, epoch)

            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/avs/{test_split}({thr})', noise_metrics, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_flickr_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    gt_resolution = (args.ground_truth_resolution, args.ground_truth_resolution)

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators = [flickr_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Flickr dataset...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        if snr == None:
            audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()

        if snr == None:
            heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(**out_dict, target=bboxes, thr=thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = (out_dict['heatmap'][j:j+1])
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(gt_resolution)
            gt_image = vt.ToPILImage()(bboxes[j]).resize(gt_resolution).point(lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize(gt_resolution)
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize(gt_resolution).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)

    if snr == None:
        heatmap_sil_min_values = np.array(heatmap_sil_min_values)
        heatmap_sil_max_values = np.array(heatmap_sil_max_values)
        heatmap_noise_min_values = np.array(heatmap_noise_min_values)
        heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    if tensorboard_path is not None and epoch is not None:
        numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
        os.makedirs(numpy_path, exist_ok=True)
        heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))
        heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))

        if snr == None:
            heatmap_sil_min_values.dump(os.path.join(numpy_path, test_split + '_sil_min.pkl'))
            heatmap_sil_max_values.dump(os.path.join(numpy_path, test_split + '_sil_max.pkl'))
            heatmap_noise_min_values.dump(os.path.join(numpy_path, test_split + '_noise_min.pkl'))
            heatmap_noise_max_values.dump(os.path.join(numpy_path, test_split + '_noise_max.pkl'))

    epoch = 0 if epoch == 'best' else epoch

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result (aggressive)
    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/{test_split}({thr})', std_metrics, epoch)

        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/{test_split}({thr})', silence_metrics, epoch)

            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/{test_split}({thr})', noise_metrics, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_exvggss_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators = [exvggss_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate Extend VGG-SS dataset ({test_split})...")):
        images, audios, bboxes,  = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        if snr == None:
            audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()

        if snr == None:
            heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Calculate confidence value for extended dataset
        v_f = model.encode_masked_vision(images.to(model.device), audio_driven_embedding)[0]
        ind = torch.arange(test_dataloader.batch_size).to(images.device)
        confs = torch.cosine_similarity(v_f[ind, ind, :], audio_driven_embedding)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(**out_dict, gt=bboxes, label=labels, conf=confs, name=name, thr=thr)

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)

    if snr == None:
        heatmap_sil_min_values = np.array(heatmap_sil_min_values)
        heatmap_sil_max_values = np.array(heatmap_sil_max_values)
        heatmap_noise_min_values = np.array(heatmap_noise_min_values)
        heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    if tensorboard_path is not None and epoch is not None:
        numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
        os.makedirs(numpy_path, exist_ok=True)
        heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))
        heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))

        if snr == None:
            heatmap_sil_min_values.dump(os.path.join(numpy_path, test_split + '_sil_min.pkl'))
            heatmap_sil_max_values.dump(os.path.join(numpy_path, test_split + '_sil_max.pkl'))
            heatmap_noise_min_values.dump(os.path.join(numpy_path, test_split + '_noise_min.pkl'))
            heatmap_noise_max_values.dump(os.path.join(numpy_path, test_split + '_noise_max.pkl'))

    epoch = 0 if epoch == 'best' else epoch

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/{test_split}({thr})', std_metrics, epoch)


        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/{test_split}({thr})', silence_metrics, epoch)

            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/{test_split}({thr})', noise_metrics, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_exflickr_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators = [exflickr_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate Extend Flickr dataset ({test_split})...")):
        images, audios, bboxes,  = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        if snr == None:
            audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()

        if snr == None:
            heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Calculate confidence value for extended dataset
        v_f = model.encode_masked_vision(images.to(model.device), audio_driven_embedding)[0]
        ind = torch.arange(test_dataloader.batch_size).to(images.device)
        confs = torch.cosine_similarity(v_f[ind, ind, :], audio_driven_embedding)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(**out_dict, gt=bboxes, label=labels, conf=confs, name=name, thr=thr)

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)

    if snr == None:
        heatmap_sil_min_values = np.array(heatmap_sil_min_values)
        heatmap_sil_max_values = np.array(heatmap_sil_max_values)
        heatmap_noise_min_values = np.array(heatmap_noise_min_values)
        heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    if tensorboard_path is not None and epoch is not None:
        numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
        os.makedirs(numpy_path, exist_ok=True)
        heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))
        heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))

        if snr == None:
            heatmap_sil_min_values.dump(os.path.join(numpy_path, test_split + '_sil_min.pkl'))
            heatmap_sil_max_values.dump(os.path.join(numpy_path, test_split + '_sil_max.pkl'))
            heatmap_noise_min_values.dump(os.path.join(numpy_path, test_split + '_noise_min.pkl'))
            heatmap_noise_max_values.dump(os.path.join(numpy_path, test_split + '_noise_max.pkl'))

    epoch = 0 if epoch == 'best' else epoch

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/{test_split}({thr})', std_metrics, epoch)

        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/{test_split}({thr})', silence_metrics, epoch)

            msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/{test_split}({thr})', noise_metrics, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

def avatar_collate_fn(batch):
    """
    Custom collate function to handle mixed tensor/list data.
    """
    # Initialize the output dictionary
    batched_data = {}

    # 1. Stack tensors (Images and Audios)
    # We extract the list of images/audios from the batch and stack them
    batched_data['images'] = torch.stack([item['images'] for item in batch])
    batched_data['audios'] = torch.stack([item['audios'] for item in batch])

    # 2. Keep metadata as lists (Ground Truths and IDs)
    # We do NOT stack these; we just return a list of dictionaries/strings
    batched_data['gts'] = [item['gts'] for item in batch]
    batched_data['ids'] = [item['ids'] for item in batch]

    return batched_data

@torch.no_grad()
def eval_avatar_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    data_path_dict: dict = {},
    use_cuda = False,
    snr = None,
    add_thresholds = {}
) -> None:
    '''
    Evaluate provided  model on AVATAR test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    gt_resolution = (args.ground_truth_resolution, args.ground_truth_resolution)

    def convert_ann_to_mask(ann: List, height: int, width: int):
        mask = np.zeros((height, width), dtype=np.uint8)
        if 'segmentation' in ann:
            poly = ann["segmentation"]

            for p in poly:
                p = np.array(p).reshape(-1, 2).astype(int)
                cv2.fillPoly(mask, [p], 1)
        return mask

    def convert_bb_to_mask(gt: List, height: int, width: int):
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []
        for ann in gt['annotations']:
            if 'bbox' in ann:
                bboxes.append(ann["bbox"])

        if not bboxes:
            return mask

        # Calculate average box: [x, y, w, h]
        final_bbox = np.mean(bboxes, axis=0).astype(int)
        x, y, w, h = final_bbox

        # fill the rectangle: (top-left) to (bottom-right)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 1, thickness=-1)

        return mask

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(model,
        test_dataloader.dataset[0]['audios'].shape,
        True,
        real_san_audio_path,
        test_dataloader.dataset.SAMPLE_RATE,
        test_dataloader.dataset.set_length,
        use_cuda=use_cuda
    )

    san_dict = {'san': True, 'san_real': args.san_real, **neg_audios}

    heatmap_pos_min_values = []
    heatmap_pos_max_values = []
    heatmap_sil_min_values = []
    heatmap_sil_max_values = []
    heatmap_noise_min_values = []
    heatmap_noise_max_values = []

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95] + list(add_thresholds.values())
    evaluators_seg = [avatar_eval.Evaluator() for i in range(len(thrs))]
    evaluators_bb = [avatar_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate AVATAR dataset ({test_split})...")):
        images, audios, gts, name = data['images'], data['audios'], data['gts'], data['ids']

        audio_embeddings = {}

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        audio_embeddings['pred_emb'] = audio_driven_embedding

        if snr == None:
            audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)

            audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

        # Localization result
        out_dict = model(images.to(model.device), resolution=args.ground_truth_resolution, **audio_embeddings)

        # Add info for boxplots and threshold evaluation
        heatmap_pos_min_values += torch.amin(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()
        heatmap_pos_max_values += torch.amax(out_dict['heatmap'], dim=(1, 2)).detach().cpu().tolist()

        if snr == None:
            heatmap_sil_min_values += torch.amin(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_sil_max_values += torch.amax(out_dict['sil_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_min_values += torch.amin(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()
            heatmap_noise_max_values += torch.amax(out_dict['noise_heatmap'], dim=(1, 2)).detach().cpu().tolist()

        # Evaluation for all thresholds
        target = torch.zeros_like(out_dict['heatmap'])
        labels = []
        for b, gt in enumerate(gts):
            mask = torch.zeros((gt['original_height'], gt['original_width']))
            label = []
            for ann in gt['annotations']:
                mask += torch.tensor(convert_ann_to_mask(ann, gt['original_height'], gt['original_width']))
                label.append(ann['audio_visual_category'])
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=gt_resolution, mode='bilinear', align_corners=False).squeeze()
            target[b] = mask >= 1
            labels.append(label)

        target_bb = torch.zeros_like(out_dict['heatmap'])
        for b, gt in enumerate(gts):
            mask = torch.tensor(convert_bb_to_mask(gt, gt['original_height'], gt['original_width']))
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=gt_resolution, mode='bilinear', align_corners=False).squeeze()
            target_bb[b] = mask

        for i, thr in enumerate(thrs):
            evaluators_seg[i].evaluate_batch(**out_dict, target=target.to(model.device), thr=thr)
            evaluators_bb[i].evaluate_batch(**out_dict, target=target_bb.to(model.device), thr=thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            heatmap = out_dict['heatmap'][j:j+1]
            heatmap_np = ((1 - heatmap.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
            heatmap_image = Image.fromarray(heatmap_np)

            os.makedirs(f'{result_dir}/heatmap/{name[j].split("/")[0]}', exist_ok=True)
            os.makedirs(f'{result_dir}/overall/{name[j].split("/")[0]}', exist_ok=True)
            os.makedirs(f'{result_dir}/overlaid/{name[j].split("/")[0]}', exist_ok=True)
            heatmap_image.save(f'{result_dir}/heatmap/{name[j]}.jpg')

            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(gt_resolution)
            gt_image = Image.fromarray(((1 - target_bb[j].squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)).resize(gt_resolution)
            seg_image = heatmap_image.resize(gt_resolution).point(lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    heatmap_pos_min_values = np.array(heatmap_pos_min_values)
    heatmap_pos_max_values = np.array(heatmap_pos_max_values)

    if snr == None:
        heatmap_sil_min_values = np.array(heatmap_sil_min_values)
        heatmap_sil_max_values = np.array(heatmap_sil_max_values)
        heatmap_noise_min_values = np.array(heatmap_noise_min_values)
        heatmap_noise_max_values = np.array(heatmap_noise_max_values)

    if tensorboard_path is not None and epoch is not None:
        numpy_path = tensorboard_path.replace('tensorboard', 'numpy')
        os.makedirs(numpy_path, exist_ok=True)
        heatmap_pos_min_values.dump(os.path.join(numpy_path, test_split + f'_pos_min_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))
        heatmap_pos_max_values.dump(os.path.join(numpy_path, test_split + f'_pos_max_snr{"_snr" + str(snr) if snr != None else ""}.pkl'))

        if snr == None:
            heatmap_sil_min_values.dump(os.path.join(numpy_path, test_split + '_sil_min.pkl'))
            heatmap_sil_max_values.dump(os.path.join(numpy_path, test_split + '_sil_max.pkl'))
            heatmap_noise_min_values.dump(os.path.join(numpy_path, test_split + '_noise_min.pkl'))
            heatmap_noise_max_values.dump(os.path.join(numpy_path, test_split + '_noise_max.pkl'))

    epoch = 0 if epoch == 'best' else epoch

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        std_metrics, silence_metrics, noise_metrics = evaluators_seg[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split}_seg with thr = {thr})\n'
        msg += f'{std_metrics["mIoU"]=}, {std_metrics["Fmeasure"]=}\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/avatar/{test_split}_seg({thr})', std_metrics, epoch)

        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split}_seg with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/avatar/{test_split}_seg({thr})', silence_metrics, epoch)

            msg += f'{model.__class__.__name__} ({test_split}_seg with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/avatar/{test_split}_seg({thr})', noise_metrics, epoch)

        std_metrics, silence_metrics, noise_metrics = evaluators_bb[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split}_bb with thr = {thr})\n'
        msg += f'{std_metrics["mIoU"]=}, {std_metrics["Fmeasure"]=}\n'
        msg += f'{std_metrics["cIoU_ap50"]=}, {std_metrics["AUC"]=}, {std_metrics["cIoU_hat"]=}\n'

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/std{"_snr" + str(snr) if snr != None else ""}/avatar/{test_split}_bb({thr})', std_metrics, epoch)

        if snr == None:
            msg += f'{model.__class__.__name__} ({test_split}_bb with thr = {thr} evaluated with Silence)\n'
            msg += f'{silence_metrics["pIA_ap50"]=}, {silence_metrics["AUC_N"]=}, {silence_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/silence/avatar/{test_split}_bb({thr})', silence_metrics, epoch)

            msg += f'{model.__class__.__name__} ({test_split}_bb with thr = {thr} evaluated with Noise)\n'
            msg += f'{noise_metrics["pIA_ap50"]=}, {noise_metrics["AUC_N"]=}, {noise_metrics["pIA_hat"]=}\n'
            if tensorboard_path is not None and epoch is not None:
                writer.add_scalars(f'test/noise/avatar/{test_split}_bb({thr})', noise_metrics, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()
