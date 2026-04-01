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

from utils.util import AddRandomNoise, RandomApply

class VGGSoundDataset(Dataset):
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
        super(VGGSoundDataset, self).__init__()

        self.epoch = 0

        self.SAMPLE_RATE = 16000
        self.split = split
        self.set_length = set_length
        self.csv_dir = os.path.join(data_path, 'metadata', split + '.csv')

        ''' Audio files '''
        self.audio_path = os.path.join(data_path, 'audio')
        audio_files = set([fn.split('.wav')[0] for fn in os.listdir(self.audio_path) if fn.endswith('.wav')])

        ''' Image files '''
        self.image_path = os.path.join(data_path, 'frames')
        image_files = set([fn.split('.jpg')[0] for fn in os.listdir(self.image_path) if fn.endswith('.jpg')])

        ''' Ground truth (Text label) '''
        self.label_dict = {item[3]: item[1] for item in csv.reader(open(self.csv_dir))}

        ''' Available files'''
        subset = set([item[3] for item in csv.reader(open(self.csv_dir))])
        self.file_list = list(audio_files.intersection(image_files).intersection(subset))
        self.file_list = sorted(self.file_list)
        # print(f'Intersection of {len(audio_files)}a, {len(image_files)}i and {len(subset)}l is {len(self.file_list)}')

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

    def get_audio(self, item: int) -> torch.Tensor:
        """
        Get audio data for a given item.

        Args:
            item (int): Index of the item.

        Returns:
            torch.Tensor: Audio data.
        """
        audio_file, sr = torchaudio.load(os.path.join(self.audio_path, self.file_list[item] + '.wav'))

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
        label = self.label_dict[self.file_list[item]].replace('_', ' ')

        ''' Transform '''
        if self.eval_noise_tr == None:
            audio = self.audio_transform(audio_file) if self.set_length != 0 else None
        else:
            audio = self.eval_noise_tr(audio_file) if self.set_length != 0 else None
        image = self.image_transform(image_file) if self.use_image else None

        noisy_audios = self.audio_transform(audio_file, force=True) if self.set_length != 0 else None

        out = {'images': image, 'audios': audio, 'noisy_audios': noisy_audios, 'labels': label, 'ids': file_id}
        out = {key: value for key, value in out.items() if value is not None}
        return out
