import torch
from torch.utils.data import Dataset

import numpy as np
import torchaudio
from torchvision import transforms as vt
from PIL import Image
import os
import csv
import json
from typing import Dict, List, Optional, Union

from utils.util import AddRandomNoise, RandomApply, get_key

import random

class AVATARDataset(Dataset):
    def __init__(self, data_path: str, split: str, is_train: bool = True, set_length: int = 8,
                 input_resolution: int = 224, hard_aug: bool = False, noise_transform_train: bool = False,
                 eval_snr = None):
        """
        Initialize VGG-Sound Dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Dataset split (Use csv file name in metadata directory).
            is_train (bool, optional): Whether it's a training set. Default is True.
            set_length (int, optional): Duration of input audio. Default is 8.
            input_resolution (int, optional): Resolution of input images. Default is 224.
            hard_aug (bool, optional): Not used.
        """
        super(AVATARDataset, self).__init__()

        self.epoch = 0

        self.SAMPLE_RATE = 16000
        self.split = split
        self.set_length = set_length
        self.metadata_dir = os.path.join(data_path, 'metadata')
        metadata_files = []
        for dirname in os.listdir(self.metadata_dir):
            if not os.path.isdir(os.path.join(self.metadata_dir, dirname)):
                continue
            for filename in os.listdir(os.path.join(self.metadata_dir, dirname)):
                if filename.endswith('.json'):
                    metadata_files.append(os.path.join(dirname, filename.split('.json')[0]))
                    if self.split == 'avatar_one':
                        break # only get one from each directory
        metadata_files = set(metadata_files)

        ''' Audio files '''
        self.audio_path = os.path.join(data_path, 'audio')
        audio_files = set([file.split('/')[0] for file in metadata_files if file.split('/')[0] + '.wav' in os.listdir(self.audio_path)])

        ''' Image files '''
        self.image_path = os.path.join(data_path, 'frames')
        image_files = []
        for dirname in os.listdir(self.image_path):
            if not os.path.isdir(os.path.join(self.image_path, dirname)):
                continue
            for filename in os.listdir(os.path.join(self.image_path, dirname)):
                if filename.endswith('.jpg'):
                    image_files.append(os.path.join(dirname, filename.split('.jpg')[0]))
        image_files = set(image_files)

        ''' Ground truth '''
        self.ground_truths = {}
        for file in metadata_files:
            gt = json.load(open(os.path.join(self.metadata_dir, file + '.json')))
            self.ground_truths[file] = {k: gt[k] for k in ('original_width', 'original_height', 'annotations')}

        off_screen_files = []
        for key, val in self.ground_truths.items():
            if val['annotations'][0]['task'] == "Off-Screen":
                off_screen_files.append(key)

        for file in off_screen_files:
            del self.ground_truths[file]

        ''' Available files'''
        subset = []
        for file in metadata_files:
            if file.split('/')[0] in audio_files:
                subset.append(file)

        self.file_list = set(subset).intersection(image_files)
        self.file_list = self.file_list - set(off_screen_files)
        print(len(self.file_list), len(set(off_screen_files)))
        self.file_list = sorted(self.file_list)
        # print(f'Intersection of {len(audio_files)}a, {len(image_files)}i and {len(subset)}l is {len(self.file_list)}')

        with open(os.path.join(self.metadata_dir, 'avatar_broad_classes.json')) as fp:
            self.broad_classes_dict = json.load(fp)

        for k, v in self.broad_classes_dict.items():
            self.broad_classes_dict[k] = list(set(self.file_list).intersection(['/'.join(['_'.join(f.split('_')[:-1]), f.split('_')[-1]]) for f in v]))

        self.broad_classes_dict = {k: v for k, v in self.broad_classes_dict.items() if len(v) != 0}

        ''' Transform '''
        if is_train:
            # since at.AddNoise is not a thing in torchaudio 0.13.0
            if noise_transform_train:
                self.audio_transform = RandomApply([
                    AddRandomNoise()
                ], 0.5)
            else:
                self.audio_transform = RandomApply([AddRandomNoise()], -1.0) # nothing essentially

            self.image_transform = vt.Compose([
                vt.Resize((int(input_resolution * 1.1), int(input_resolution * 1.1)), vt.InterpolationMode.BICUBIC),
                vt.ToTensor(),
                vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
                vt.RandomCrop((input_resolution, input_resolution)),
                vt.RandomHorizontalFlip(),
            ])
            if hard_aug:
                self.image_transform = vt.Compose([
                    vt.RandomResizedCrop((input_resolution, input_resolution)),
                    vt.RandomApply([vt.GaussianBlur(5, [.1, 2.])], p=0.8),
                    # vt.RandomApply([vt.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    vt.RandomGrayscale(p=0.2),
                    vt.ToTensor(),
                    vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
                    vt.RandomHorizontalFlip(),
                ])
        else:
            self.audio_transform = RandomApply([AddRandomNoise()], -1.0) # nothing essentially
            self.image_transform = vt.Compose([
                vt.Resize((input_resolution, input_resolution), vt.InterpolationMode.BICUBIC),
                vt.ToTensor(),
                vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
            ])

        self.is_train = is_train
        self.use_image = True
        if input_resolution is None:
            self.use_image = False

        self.eval_noise_tr = None
        if eval_snr != None:
            self.eval_noise_tr = AddRandomNoise(snr=eval_snr)

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.file_list)

    def get_audio(self, item: int, file_id = None) -> torch.Tensor:
        """
        Get audio data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            torch.Tensor: Audio data.
        """
        if item != None or file_id == None:
            audio_file, sr = torchaudio.load(os.path.join(self.audio_path, self.file_list[item].split('/')[0] + '.wav'))
        else:
            audio_file, sr = torchaudio.load(os.path.join(self.audio_path, file_id.split('/')[0] + '.wav'))

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            audio_file = resampler(audio_file)

        if audio_file.shape[0] > 1:
            audio_file = audio_file.mean(dim=0)

        audio_file = audio_file.squeeze(0)

        # slicing or padding based on set_length
        # slicing
        if audio_file.shape[0] > (self.SAMPLE_RATE * self.set_length):
            audio_file = audio_file[:self.SAMPLE_RATE * self.set_length]
        # zero padding
        if audio_file.shape[0] < (self.SAMPLE_RATE * self.set_length):
            pad_len = (self.SAMPLE_RATE * self.set_length) - audio_file.shape[0]
            pad_val = torch.zeros(pad_len)
            audio_file = torch.cat((audio_file, pad_val), dim=0)

        return audio_file

    def get_image(self, item: int) -> Image.Image:
        """
        Get image data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            Image.Image: Image data.
        """
        image_file = Image.open(os.path.join(self.image_path, self.file_list[item] + '.jpg'))
        return image_file

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str, str]]:
        """
        Get item from the dataset.

        Args:
            item (int): Index of the item.

        Returns:
            Dict[str, Union[torch.Tensor, torch.Tensor, Optinal[torch.Tensor], str, str]]: Data example
        """

        if item >= len(self.file_list):
            raise IndexError(f"Index {item} out of range for dataset of size {len(self.file_list)}")

        file_id = self.file_list[item]

        ''' Load data '''
        audio_file = self.get_audio(item) if self.set_length != 0 else None
        image_file = self.get_image(item) if self.use_image else None

        annotations = self.ground_truths[self.file_list[item]]

        ''' Transform '''
        if self.eval_noise_tr == None:
            audio = self.audio_transform(audio_file) if self.set_length != 0 else None
        else:
            audio = self.eval_noise_tr(audio_file) if self.set_length != 0 else None
        image = self.image_transform(image_file) if self.use_image else None

        all_classes = set(self.broad_classes_dict.keys())
        same_class = get_key(self.broad_classes_dict, file_id)

        if same_class == None:
            print(file_id+".wav")

        all_classes -= same_class

        random_class = random.choice(sorted(all_classes))
        random_file = random.choice(self.broad_classes_dict[random_class])

        offscreen_audios = self.get_audio(None, random_file) if self.set_length != 0 else None

        out = {'images': image, 'audios': audio, 'gts': annotations, 'ids': file_id, 'offscreen_audios': offscreen_audios}
        out = {key: value for key, value in out.items() if value is not None}
        return out
