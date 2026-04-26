import torch

import numpy as np
import random
import os
from typing import Tuple, Optional


def get_prompt_template(mode: str = 'default') -> Tuple[str, int, int]:
    '''
    Generate a prompt template based on the specified mode.

    Args:
        mode (str, optional): The mode for selecting the prompt template. Default is 'default'.

    Returns:
        Tuple[str, int, int]: A tuple containing the generated prompt template, the position of the placeholder '{}',
                             and the length of the prompt.

    Notes:
        If the mode is 'random', a random prompt template is chosen from a predefined list.
    '''
    prompt_template = 'A photo of {}'

    if mode == 'random':
        prompt_templates = [
            'a photo of a {}', 'a photograph of a {}', 'an image of a {}', '{}',
            'a cropped photo of a {}', 'a good photo of a {}', 'a photo of one {}',
            'a bad photo of a {}', 'a photo of the {}', 'a photo of {}', 'a blurry photo of a {}',
            'a picture of a {}', 'a photo of a scene where {}'
        ]
        prompt_template = random.choice(prompt_templates)

    # Calculate prompt length and text position
    prompt_length = 1 + len(prompt_template.split(' ')) + 1 - 1  # eos, sos => 1 + 1, {} => -1
    text_pos_at_prompt = 1 + prompt_template.split(' ').index('{}')

    return prompt_template, text_pos_at_prompt, prompt_length


# Reproducibility
def fix_seed(seed: int = 0) -> None:
    '''
    Set seeds for random number generators to ensure reproducibility.

    Args:
        seed (int, optional): The seed value. Default is 0.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id: int) -> None:
    '''
    Set a seed for a worker process to ensure reproducibility in PyTorch DataLoader.

    Args:
        worker_id (int): The ID of the worker process.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

'''
Backported from TorchAudio (torchaudio.functional.add_noise)
Source: https://github.com/pytorch/audio/blob/e284e58c83f69c95a7f4a8a7d402f6c27ef56f5d/src/torchaudio/functional/functional.py#L2317

Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
Licensed under the BSD 2-Clause License.
Reason: Version compatibility for torchaudio==0.13.0
'''
def add_noise(
    waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r'''Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    '''

    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    if lengths is not None:
        mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
            -1
        )  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)

'''
Modified/Backported from TorchAudio (torchaudio.transforms)
Source: https://github.com/pytorch/audio/blob/e284e58c83f69c95a7f4a8a7d402f6c27ef56f5d/src/torchaudio/transforms/_transforms.py#L2058

Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
Licensed under the BSD 2-Clause License.
Reason: Version compatibility for torchaudio==0.13.0
'''
class AddRandomNoise(torch.nn.Module):
    r'''Scales and adds noise to waveform per signal-to-noise ratio.
    See :meth:`torchaudio.functional.add_noise` for more details.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript
    '''

    def __init__(self, lengths: Optional[torch.Tensor] = None, snr: float = None):
        r'''
        Args:
            snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
            lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``,
        '''
        super().__init__()

        if snr != None:
            self.snr = torch.Tensor([snr])
        else:
            self.snr = torch.Tensor([1000.0]) # high value --> no noise initially
        self.epoch = 0
        self.lengths = lengths

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
            with shape `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all
            elements in ``waveform`` and ``noise`` are treated as valid. (Default: ``None``)

        Returns:
            torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
            (same shape as ``waveform``).
        '''
        # slightly changed
        waveform = waveform.unsqueeze(0)
        noise = torch.clip(torch.randn(waveform.shape), min=-1., max=1.)
        noisy_waveform = add_noise(waveform, noise, self.snr, self.lengths)
        return noisy_waveform.squeeze(0)

    def step(self, epoch, k=5):
        self.epoch = epoch
        self.snr = torch.Tensor([np.exp(-epoch/k + k) + k]) # negative exponential profile

'''
Useful implementation for randomly applying transforms :)
Source: https://github.com/Spijkervet/torchaudio-augmentations/blob/891b3b6e19551c211e7cdab36376c7e67e9d199c/torchaudio_augmentations/apply.py#L34

Copyright (c) 2021 Janne Spijkervet
Licensed under the MIT License (to my best knowledge)
'''
class RandomApply(torch.nn.Module):
    '''Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    '''
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, tensor, force=False):
        if not force:
            if self.p < torch.rand(1):
                return tensor
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

    def step(self, idx, *args):
        if self.transforms:
            if len(self.transforms) > idx:
                self.transforms[idx].step(*args)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

'''
If applying more audio transforms with different probability each will need also this
https://github.com/Spijkervet/torchaudio-augmentations/blob/891b3b6e19551c211e7cdab36376c7e67e9d199c/torchaudio_augmentations/compose.py#L4
'''

def remove_diagonal(x: torch.Tensor) -> torch.Tensor:
    B, _, H, W = x.shape

    # For each row i, collect column indices [0..i-1, i+1..B-1]
    idx = torch.arange(B, device=x.device)
    # Build [B, B-1] index matrix of column indices to keep
    col_idx = torch.stack([
        torch.cat([idx[:i], idx[i+1:]])
        for i in range(B)
    ])  # [B, B-1]

    # Expand for gathering over H and W
    col_idx = col_idx[:, :, None, None].expand(B, B - 1, H, W)  # [B, B-1, H, W]

    return x.gather(1, col_idx)

def get_key(dct, val):
    list_keys = list(dct.keys())
    list_classes = []
    for key in list_keys:
        values = dct[key]
        if val in values:
            list_classes.append(key)

    return set(list_classes)
