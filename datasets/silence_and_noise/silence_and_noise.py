import os
import torch, torchaudio
from utils.util import get_prompt_template

def process_audio(audio_file, SAMPLE_RATE = 16000, set_length: int = 8):
    if audio_file.shape[0] > 1:
        audio_file = audio_file.mean(dim=0)

    audio_file = audio_file.squeeze(0)

    # slicing or padding based on set_length
    # slicing
    if audio_file.shape[0] > (SAMPLE_RATE * set_length):
        audio_file = audio_file[:SAMPLE_RATE * set_length]
    # zero padding
    if audio_file.shape[0] < (SAMPLE_RATE * set_length):
        pad_len = (SAMPLE_RATE * set_length) - audio_file.shape[0]
        pad_val = torch.zeros(pad_len)
        audio_file = torch.cat((audio_file, pad_val), dim=0)

    return audio_file

def get_real_noise_audios(real_san_audios_path, SAMPLE_RATE = 16000, set_length: int = 8) -> torch.Tensor:
    audio_paths = os.listdir(real_san_audios_path)
    audio_files = []

    for audio_path in audio_paths:
        audio_file, sr = torchaudio.load(os.path.join(real_san_audios_path, audio_path))

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            audio_file = resampler(audio_file)

        audio_files.append(process_audio(audio_file, SAMPLE_RATE, set_length))

    return torch.stack(audio_files, dim=0)

def get_silence_noise_audios(module, audio_size, san_active = False, real_san_audios_path = None,
                             SAMPLE_RATE = 16000, set_length: int = 8, use_cuda = False) -> dict[str, torch.Tensor]:
    '''
    Generates embeddings for negative audios given if san_active or san_real_active,
    concatenating along the first dimension all the audios given.
    '''
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()
    placeholder_tokens = module.get_placeholder_token(prompt_template.replace('{}', ''))

    negative_audios_emb = {}
    if san_active:
        negative_audios = torch.cat(
            (
                torch.zeros(audio_size).unsqueeze(0),
                torch.clip(torch.randn(audio_size), min=-1., max=1.).unsqueeze(0)
            ),
            dim=0
        )

        audio_driven_embedding = module.encode_audio(
            negative_audios.to(module.device),
            placeholder_tokens.repeat((negative_audios.shape[0], 1)),
            text_pos_at_prompt,
            prompt_length
        ).detach()

        if use_cuda:
            audio_driven_embedding = audio_driven_embedding.half()

        negative_audios_emb['pred_emb_san'] = audio_driven_embedding

    if real_san_audios_path:
        negative_audios = get_real_noise_audios(real_san_audios_path, SAMPLE_RATE, set_length)

        audio_driven_embedding = module.encode_audio(
            negative_audios.to(module.device),
            placeholder_tokens.repeat((negative_audios.shape[0], 1)),
            text_pos_at_prompt,
            prompt_length
        ).detach()

        if use_cuda:
            audio_driven_embedding = audio_driven_embedding.half()

        negative_audios_emb['pred_emb_real_san'] = audio_driven_embedding

    return negative_audios_emb
