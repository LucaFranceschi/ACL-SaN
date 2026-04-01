import os
import sys
import time
import datetime
import yaml
import shutil
import argparse
import gc
import re
import wandb

import numpy as np
from tqdm import tqdm
from importlib import import_module
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.util import get_prompt_template, fix_seed, seed_worker
from datasets.vggsound.VGGSound_Dataset import VGGSoundDataset
from datasets.silence_and_noise.silence_and_noise import get_silence_noise_audios
from utils.eval import eval_vggsound_validation


def main(model_name, model_path, exp_name, train_config_name, data_path_dict, save_path, recover_from = None):
    """
    Main function for training an image compression model.

    Args:
        model_name (str): The name of the compression model, corresponding to the model config file in './config/model'.
        exp_name (str): The postfix for saving the experiment.
        train_config_name (str): The name of the training configuration, corresponding to the files in './config/train'.
        data_path_dict (dict): The directory for dataset.
        save_path (str): The directory where training results will be saved.

    Returns:
        None
    """

    if USE_DDP:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=9000))
        global rank
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        print(f'World size: {world_size}') if rank == 0 else None

    device = torch.device('cuda', torch.cuda.current_device()) if USE_CUDA else torch.device('cpu')
    print(f'Device: {device} is used\n')

    model_exp_name = f'{model_name}_{exp_name}' if exp_name != "" else model_name

    ''' Set logging dir '''
    tensorboard_path = os.path.join(save_path, 'Train_record', model_exp_name, "tensorboard")

    ''' Get train configure '''
    train_conf_file = f'./config/train/{train_config_name}.yaml'
    with open(train_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config['common'])
        args.optim = config['optim_conf'][config['optimizer']]
        if rank == 0:
            print(vars(args))

    wandb_run = None
    if rank == 0 and WANDB_LOGGING:
        wandb_run = wandb.init(project=os.getenv('WANDB_PROJECT_NAME'), entity=os.getenv('WANDB_ENTITY_TEAM'), config=vars(args))
        wandb.define_metric("train/*", step_metric="trainer/train_step")
        wandb.define_metric("train_losses/*", step_metric="trainer/train_step")
        wandb.define_metric("validation/*", step_metric="trainer/val_step")
        wandb.define_metric("validation_losses/*", step_metric="trainer/val_step")
        wandb.define_metric("images/val_overlaid/*", step_metric="trainer/epoch")

    ''' Fix random seed'''
    fix_seed(args.seed)

    ''' Tensorboard '''
    writer = SummaryWriter(tensorboard_path)
    print(f"\nSave dir: {os.path.join(save_path, 'Train_record', model_exp_name)}\n") if rank == 0 else None

    ''' Get model '''
    model_conf_file = f'./config/model/{model_name}.yaml'
    model = getattr(import_module('modules.models'), config['model'])(model_conf_file, device, model_path)
    if rank == 0:
        print(f"Model '{model.__class__.__name__}' with configure file '{model_name}' is loaded")
        print(f"Loaded model details: {vars(model.args.model)}\n")

    print(args.train_data)

    ''' Get dataloader '''
    # Get Train Dataloader (VGGSS)
    subset = ''
    train_dataset = VGGSoundDataset(data_path_dict['vggsound'], f'vggsound_train{subset}', is_train=True,
                                    input_resolution=args.input_resolution, noise_transform_train=args.san_added_noise_tr, set_length=3)

    validation_dataset = VGGSoundDataset(data_path_dict['vggsound'], f'vggsound_val{subset}', is_train=False,
                                    input_resolution=args.input_resolution, set_length=3)

    ''' Create DistributedSampler '''
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if USE_DDP else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                                   num_workers=args.num_workers, pin_memory=False, drop_last=True,
                                                   worker_init_fn=seed_worker)

    sampler_validation = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=False) if USE_DDP else None

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, sampler=sampler_validation,
                                                   num_workers=args.num_workers, pin_memory=False, drop_last=True,
                                                   worker_init_fn=seed_worker, shuffle=False)

    ''' Optimizer '''
    module_path, module_name = args.optim.pop('module_path'), args.optim.pop('module_name')
    optimizer = getattr(import_module(module_path), module_name)(model.parameters(), **args.optim)

    ''' Scheduler '''
    scheduler = None
    if config['scheduler']:
        print(f"Scheduler: {config['scheduler']}")
        args.sched = config['sched_conf'][config['scheduler']]
        module_path, module_name = args.sched.pop('module_path'), args.sched.pop('module_name')
        scheduler = getattr(import_module(module_path), module_name)(optimizer,
                                                                     T_max=args.epoch * len(train_dataloader),
                                                                     eta_min=args.sched['eta_ratio'] * args.optim['lr'])

    ''' Autocast '''
    if config['amp']:
        if rank == 0:
            print('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    ''' Make distributed data parallel module '''
    model = DistributedDataParallel(model, device_ids=[device], output_device=device) if USE_DDP else model
    module = model.module if isinstance(model, DistributedDataParallel) else model

    if recover_from != None:
        module.load(recover_from)
        recovered_epoch = int(re.search(r'Param_(\d+).pth', recover_from).group(1))

    validation_loss_list = []
    train_loss_list = []

    real_san_audio_path = data_path_dict['san'] if args.san_real else None

    neg_audios = get_silence_noise_audios(module,
        train_dataset[0]['audios'].shape,
        args.san,
        real_san_audio_path,
        train_dataset.SAMPLE_RATE,
        train_dataset.set_length,
        use_cuda=USE_CUDA
    )

    if USE_CUDA and neg_audios != None:
        for key in neg_audios.keys():
            neg_audios[key] = neg_audios[key].half()

    san_dict = {'san': args.san, 'san_real': args.san_real, **neg_audios}

    ''' Train Loop '''
    for epoch in range(args.epoch):
        module.train(True)

        total_loss_per_epopch = 0.0
        loss_add_count = 0.0

        loss_dict = {}
        loss_per_epoch_dict = {loss_name: 0.0 for loss_name in args.loss}

        train_dataloader.dataset.epoch = epoch
        train_dataloader.dataset.audio_transform.step(0, epoch, args.san_added_noise_schedule_k)

        pbar = tqdm(train_dataloader, desc=f"Train Epoch [{epoch}/{args.epoch}]", disable=(rank != 0))
        sampler.set_epoch(epoch) if USE_DDP else None

        if recover_from != None and epoch <= recovered_epoch:
            continue

        for step, data in enumerate(pbar):
            images, audios, labels = data['images'], data['audios'], data['labels']
            noisy_audios = data['noisy_audios']

            if USE_CUDA:
                images = images.half()

            prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

            audio_embeddings = {}

            with autocast_fn():
                # Train step
                placeholder_tokens = module.get_placeholder_token(prompt_template.replace('{}', ''))
                placeholder_tokens = placeholder_tokens.repeat((train_dataloader.batch_size, 1))
                audio_driven_embedding = module.encode_audio(audios.to(module.device), placeholder_tokens,
                                                             text_pos_at_prompt, prompt_length)
                if USE_CUDA:
                    audio_driven_embedding = audio_driven_embedding.half()

                audio_embeddings['pred_emb'] = audio_driven_embedding

                if 'diff_san_l' in args.loss:
                    audio_driven_embedding_noisy = module.encode_audio(noisy_audios.to(module.device), placeholder_tokens,
                                                             text_pos_at_prompt, prompt_length)
                    if USE_CUDA:
                        audio_driven_embedding_noisy = audio_driven_embedding_noisy.half()

                    audio_embeddings['pred_emb_noisy'] = audio_driven_embedding_noisy

                if args.san:
                    audio_embeddings['pred_emb_silence'] = san_dict['pred_emb_san'][0].unsqueeze(0)
                    audio_embeddings['pred_emb_noise'] = san_dict['pred_emb_san'][1].unsqueeze(0)

                if args.san_real:
                    audio_embeddings['pred_emb_real_san'] = san_dict['pred_emb_real_san']

                # contains also SaN outputs if set
                out_dict = module(images.to(module.device), resolution=args.input_resolution, **audio_embeddings)

                loss_args = {**out_dict, **audio_embeddings}

                for j, loss_name in enumerate(args.loss):
                    loss_dict[loss_name] = getattr(import_module('utils.loss'), loss_name)(**loss_args) * args.loss_w[j]
                    loss_per_epoch_dict[loss_name] += loss_dict[loss_name].item()
                loss = torch.sum(torch.stack(list(loss_dict.values())))

            total_loss_per_epopch += loss.item()
            loss_add_count += 1.0
            optimizer.zero_grad()

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if scheduler is not None:
                scheduler.step()

            avr_loss = total_loss_per_epopch / loss_add_count

            if rank == 0:
                pbar.set_description(f"Training Epoch {epoch}, Loss = {round(avr_loss, 5)}")

                if wandb_run:
                    train_step = (epoch * len(train_dataloader)) + step

                    wandb_run.log({
                        **{f'train_losses/step/{key}': val for key, val in loss_dict.items()},
                        **{f'train_losses/avr/{key}': val / loss_add_count for key, val in loss_per_epoch_dict.items()},
                        'train/step/loss': loss.item(),
                        'train/avr/loss': avr_loss,
                        'trainer/train_step': train_step
                    })

                # print(gc.get_stats())

        if rank == 0:
            train_loss_list.append(float(avr_loss))

        if USE_DDP:
            dist.barrier()

        module.train(False)

        viz_dir_template = os.path.join(save_path, 'Visual_results', '{}', model_exp_name, f'epoch{epoch}')

        sampler_validation.set_epoch(epoch) if USE_DDP else None
        avr_loss_val = eval_vggsound_validation(
            module,
            validation_dataloader,
            args,
            viz_dir_template.format('vggsound_val'),
            epoch,
            tensorboard_path=tensorboard_path,
            rank=rank,
            wandb_run=wandb_run,
            data_path_dict=data_path_dict,
            use_cuda=USE_CUDA,
            use_amp=config['amp']
        )

        validation_loss_list.append(avr_loss_val)

        if rank == 0:
            save_dir = os.path.join(save_path, 'Train_record', model_exp_name, f'Param_{str(epoch)}.pth')
            module.save(save_dir)

        if USE_DDP:
            dist.barrier()

        if USE_CUDA:
            torch.cuda.empty_cache()

        gc.collect()

    writer.close()

    if rank == 0:
        with open(os.path.join(save_path, 'Train_record', model_exp_name, 'train_losses.pkl'), 'wb') as f:
            np.array(train_loss_list).dump(f)

        with open(os.path.join(save_path, 'Train_record', model_exp_name, 'validation_losses.pkl'), 'wb') as f:
            np.array(validation_loss_list).dump(f)

    dist.destroy_process_group() if USE_DDP else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Use model config file name')
    parser.add_argument('--model_path', type=str, default='', help='Use model save path')
    parser.add_argument('--train_config', type=str, default='', help='Use train config file name')
    parser.add_argument('--exp_name', type=str, default='', help='postfix for save experiment')
    parser.add_argument('--save_path', type=str, default='', help='Save path for model and results')
    parser.add_argument('--vggss_path', type=str, default='', help='VGGSS dataset directory')
    parser.add_argument('--flickr_path', type=str, default='', help='Flickr dataset directory')
    parser.add_argument('--avs_path', type=str, default='', help='AVSBench dataset directory')
    parser.add_argument('--vggsound_path', type=str, default='', help='VGGSound dataset directory')
    parser.add_argument('--san_path', type=str, default='', help='Silence and noise data directory')
    parser.add_argument('--local_rank', type=str, default='', help='Rank for distributed train')
    parser.add_argument('--recover_from', type=str, default=None, help='Path to weights for recover after crash')
    parser.add_argument('--wandb_logging', action='store_true', help='Login to wandb and log losses and experiments')

    args = parser.parse_args()

    WANDB_LOGGING = args.wandb_logging

    data_path = {'vggss': args.vggss_path,
                 'flickr': args.flickr_path,
                 'avs': args.avs_path,
                 'vggsound': args.vggsound_path,
                 'san': args.san_path}

    USE_CUDA = torch.cuda.is_available()

    # Check the number of GPUs for training
    NUM_GPUS = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    USE_DDP = True if NUM_GPUS > 1 else False

    rank = 0 if not USE_DDP else None

    if rank == 0 and WANDB_LOGGING:
        wandb.login(os.getenv('WANDB_API_KEY'))

    # Run example
    main(args.model_name, args.model_path, args.exp_name, args.train_config, data_path, args.save_path, args.recover_from)
