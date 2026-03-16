import torch
import os

import yaml
import argparse

from tqdm import tqdm
from utils.util import get_prompt_template, fix_seed, seed_worker
from datasets.VGGSS.VGGSS_Dataset import VGGSSDataset, ExtendVGGSSDataset
from datasets.Flickr.Flickr_Dataset import FlickrDataset, ExtendFlickrDataset
from datasets.AVSBench.AVSBench_Dataset import AVSBenchDataset
from datasets.vggsound.VGGSound_Dataset import VGGSoundDataset
from datasets.AVATAR.AVATAR_Dataset import AVATARDataset
from importlib import import_module
from utils.eval import *

import numpy as np

import re

from datasets.silence_and_noise.silence_and_noise import get_silence_noise_audios

@torch.no_grad()
def main(model_name, model_path, train_config_name, data_path_dict, save_path):
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f'Device: {device} is used\n')
    print(f'Testing {train_config_name} and storing results in {save_path}')

    ''' Get train configure '''
    train_conf_file = f'./config/train/{train_config_name}.yaml'
    with open(train_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config['common'])
        args.optim = config['optim_conf'][config['optimizer']]

    ''' Fix random seed'''
    fix_seed(args.seed)

    model_exp_name = f'{model_name}_{train_config_name}' if train_config_name != "" else model_name
    match = re.search(r'Param_(.*).pth', data_path_dict['model_weights'])
    if match:
        epoch = match.group(1)

    print(f'Testing epoch {epoch}')

    ''' Set logging dir '''
    tensorboard_path = os.path.join(save_path, 'Test_record_noisy', model_exp_name, "tensorboard", f'epoch{epoch}')

    viz_dir_template = os.path.join(save_path, 'Visual_results_test_noisy', '{}', model_exp_name, f'epoch{epoch}')

    ''' Get model '''
    model_conf_file = f'./config/model/{model_name}.yaml'
    model = getattr(import_module('modules.models'), config['model'])(model_conf_file, device, model_path)
    print(f"Model '{model.__class__.__name__}' with configure file '{model_name}' is loaded")
    print(f"Loaded model details: {vars(model.args.model)}\n")

    ''' Make distributed data parallel module '''
    module = model
    module.load(data_path_dict['model_weights'])

    module.train(False)

    thresholds = {}

    for snr in [None, 20, 10, 5]:
        # Get Test Dataloader (VGGSound)
        test_dataset = VGGSoundDataset(data_path_dict['vggsound'], f'vggsound_test', is_train=False,
            input_resolution=args.ground_truth_resolution, set_length=3, eval_snr=snr)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=False, drop_last=True, shuffle=False)

        # Get Test Dataloader (VGGSS)
        vggss_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_test', is_train=False,
                                    input_resolution=args.input_resolution, eval_snr=snr)
        vggss_dataloader = torch.utils.data.DataLoader(vggss_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                    pin_memory=False, drop_last=True)

        # Get Test Dataloader (Flickr)
        flickr_dataset = FlickrDataset(data_path_dict['flickr'], 'flickr_test', is_train=False,
                                    input_resolution=args.input_resolution, eval_snr=snr)
        flickr_dataloader = torch.utils.data.DataLoader(flickr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                        pin_memory=False, drop_last=True)

        # Get Test Dataloader (Extended VGGSS)
        exvggss_dataset = ExtendVGGSSDataset(data_path_dict['vggss'], input_resolution=args.input_resolution, eval_snr=snr)
        exvggss_dataloader = torch.utils.data.DataLoader(exvggss_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                        pin_memory=False, drop_last=True)

        # Get Test Dataloader (Extended Flickr)
        exflickr_dataset = ExtendFlickrDataset(data_path_dict['flickr'], input_resolution=args.input_resolution, eval_snr=snr)
        exflickr_dataloader = torch.utils.data.DataLoader(exflickr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                        pin_memory=False, drop_last=True)

        # Get Test Dataloader (AVS)
        avss4_dataset = AVSBenchDataset(data_path_dict['avs'], 'avs1_s4_test', is_train=False,
                                        input_resolution=args.input_resolution, eval_snr=snr)
        avss4_dataloader = torch.utils.data.DataLoader(avss4_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                    pin_memory=False, drop_last=True)

        avsms3_dataset = AVSBenchDataset(data_path_dict['avs'], 'avs1_ms3_test', is_train=False,
                                        input_resolution=args.input_resolution, eval_snr=snr)
        avsms3_dataloader = torch.utils.data.DataLoader(avsms3_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                        pin_memory=False, drop_last=True)

        avatar_dataset = AVATARDataset(data_path_dict['avatar'], 'avatar_one', is_train=False, input_resolution=args.input_resolution, eval_snr=snr)
        avatar_dataloader = torch.utils.data.DataLoader(avatar_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                                        pin_memory=False, drop_last=True, collate_fn=avatar_collate_fn)

        if snr == None:
            thresholds = eval_vggss_get_thresholds(module, vggss_dataloader, args, epoch, tensorboard_path, data_path_dict, USE_CUDA)

        eval_flickr_agg(module, flickr_dataloader, args, viz_dir_template.format('flickr'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_exflickr_agg(module, exflickr_dataloader, args, viz_dir_template.format('exflickr'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_avsbench_agg(module, avsms3_dataloader, args, viz_dir_template.format('ms3'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_vggss_agg(module, vggss_dataloader, args, viz_dir_template.format('vggss'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_vggsound_agg(module, test_dataloader, args, viz_dir_template.format('vggsound_test'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_exvggss_agg(module, exvggss_dataloader, args, viz_dir_template.format('exvggss'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_avsbench_agg(module, avss4_dataloader, args, viz_dir_template.format('s4'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)
        eval_avatar_agg(module, avatar_dataloader, args, viz_dir_template.format('avatar'), epoch,
            tensorboard_path, data_path_dict, USE_CUDA, snr=snr, add_thresholds=thresholds)

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Use model config file name')
    parser.add_argument('--model_path', type=str, default='', help='Use model save path')
    parser.add_argument('--model_weights', type=str, default='', help='Path for model weights')
    parser.add_argument('--train_config', type=str, default='', help='Use train config file name')
    parser.add_argument('--save_path', type=str, default='', help='Save path for results')
    parser.add_argument('--vggss_path', type=str, default='', help='VGGSS dataset directory')
    parser.add_argument('--flickr_path', type=str, default='', help='Flickr dataset directory')
    parser.add_argument('--avs_path', type=str, default='', help='AVSBench dataset directory')
    parser.add_argument('--vggsound_path', type=str, default='', help='VGGSound dataset directory')
    parser.add_argument('--avatar_path', type=str, default='', help='AVATAR dataset directory')
    parser.add_argument('--san_path', type=str, default='', help='Silence and noise data directory')

    args = parser.parse_args()

    data_path = {'vggss': args.vggss_path,
                 'flickr': args.flickr_path,
                 'avs': args.avs_path,
                 'vggsound': args.vggsound_path,
                 'avatar': args.avatar_path,
                 'san': args.san_path,
                 'model_weights': args.model_weights}

    USE_CUDA = torch.cuda.is_available()

    # Check the number of GPUs for training
    NUM_GPUS = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))

    if NUM_GPUS == 1:
        main(args.model_name, args.model_path, args.train_config, data_path, args.save_path)

    exit(1)
