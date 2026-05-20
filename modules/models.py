import torch
import torch.nn.functional as F
from torch import nn

import yaml
import argparse

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.AudioToken.embedder import FGAEmbedder
from modules.CLIPSeg.clipseg_for_audio import CLIPSeg
from modules.mask_utils import ImageMasker, FeatureMasker, TrainableADCLSigmoid
from transformers import AutoTokenizer
from torch.utils.checkpoint import checkpoint

from transformers.models.clipseg.modeling_clipseg import _expand_mask

from utils.util import remove_diagonal

class ACL(nn.Module):
    def __init__(self, conf_file: str, device: str, model_path: str):
        """
        Audio-Grounded Contrastive Learning (ACL) model.

        Args:
            conf_file (str): Path to the configuration file.
            device (str): Device to move the model to.
        """
        super(ACL, self).__init__()

        # Get configuration
        with open(conf_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.args = argparse.Namespace()
            self.args.model = argparse.Namespace(**config['model'])
            self.args.clip_embedding_dim = config['clip_conf'][self.args.model.clip]['embedding_dim']
            self.args.clip_name = config['clip_conf'][self.args.model.clip]['name']
            self.pretrain = argparse.Namespace(**config['pretrain'])
            self.args.audio_proj = argparse.Namespace(**config['fga_conf'][self.args.model.audio_proj])

        # Init audio encoder
        checkpoint = torch.load(self.pretrain.audio_backbone)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.audio_backbone = BEATs(cfg)

        # Text Tokenizer for placeholder prompt
        # self.tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        local_model_path = model_path + "/clipseg-rd64-refined-local"
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)

        # Init audio projection layer
        self.audio_proj = FGAEmbedder(input_size=self.args.audio_proj.input_size * 3,
                                      output_size=self.args.audio_proj.output_size)

        # Init audio-visual grounder (Grounder: CLIPSeg)
        # self.av_grounder = CLIPSeg.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.av_grounder = CLIPSeg.from_pretrained(local_model_path)

        # Init maskers
        self.masker_i = ImageMasker(10.0, 14.0, 1.0)
        self.masker_f = FeatureMasker(0.5, 0.07)

        # Load weights
        self.audio_backbone.load_state_dict(checkpoint['model'])
        self.audio_backbone.predictor = None

        if self.pretrain.audio_proj is not None:
            self.audio_proj.load_state_dict(torch.load(self.pretrain.audio_embedder))

        # Set device
        self.device = device
        self.audio_backbone.to(device=self.device)
        self.av_grounder.to(device=self.device)
        self.audio_proj.to(device=self.device)
        self.masker_i.to(self.device)
        self.masker_f.to(self.device)

    def get_placeholder_token(self, prompt_text: str):
        """
        Get placeholder token from prompt text

        Args:
            prompt_text (str): prompt text without '{}'

        Returns:
            CLIPTokenizerFast result with prompt text
        """
        placeholder_token = self.tokenizer(prompt_text, return_tensors="pt").data['input_ids']
        placeholder_token = F.pad(placeholder_token, (0, 77 - placeholder_token.shape[-1])).to(self.device)
        return placeholder_token

    def train(self, bool: bool = True):
        """
        Set the module in training mode.

        Args:
            bool (bool): If True, set the module in training mode.
        """
        super().train(bool)
        self.av_grounder.requires_grad_(False)
        self.audio_backbone.requires_grad_(False)

    def encode_audio(self, audio: torch.Tensor, placeholder_token: torch.Tensor, pos: int,
                     prompt_size: int) -> torch.Tensor:
        """
        Encode audio input into audio-driven embedding (Audio-Driven Embedder)

        Args:
            audio (torch.Tensor): Input audio tensor.
            placeholder_token (torch.Tensor): Placeholder token for CLIP Text encoder.
            pos (int): Position of audio token.
            prompt_size (int): Size of the placeholder prompt.

        Returns:
            torch.Tensor: Audio-driven embeddings.
        """
        audio_feat = self.audio_backbone.extract_features(audio)[1]
        audio_token_emb = self.audio_proj(audio_feat).unsqueeze(1)
        audio_driven_embedding = self.av_grounder.encode_audio(placeholder_token, audio_token_emb, pos,
                                                               prompt_size + audio_token_emb.shape[1])

        return audio_driven_embedding

    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode visual input and generate visual embeddings.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Visual embeddings.
        """
        vision_outputs = self.av_grounder.clip.vision_model(pixel_values=image,
                                                            output_attentions=None,
                                                            output_hidden_states=True,
                                                            return_dict=True)
        pooled_output = self.av_grounder.clip.visual_projection(vision_outputs[1])

        return pooled_output

    def _forward_decoder(self, image: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        # step 1: forward the query images through the frozen CLIP vision encoder
        vision_outputs = self.av_grounder.clip.vision_model(pixel_values=image,
                                                            output_attentions=None,
                                                            output_hidden_states=True,
                                                            return_dict=True)
        hidden_states = vision_outputs.hidden_states
        # we add +1 here as the hidden states also include the initial embeddings
        activations = [hidden_states[i + 1] for i in self.av_grounder.extract_layers]

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        # Audio injected embedding from input argument

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.av_grounder.decoder(
            activations,
            embedding,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        return decoder_outputs.logits

    def forward_decoder(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224) -> torch.Tensor:
        """
        Forward pass of audio-visual grounder

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            resolution (int): Resolution of the output.
            ignore_indices (list): List of indices to ignore.

        Returns:
            torch.Tensor: Logits from the decoder.
        """
        logits = checkpoint(self._forward_decoder, image, embedding, use_reentrant=False) # [B, h, w]

        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(1)
        else:
            logits = logits.unsqueeze(1)

        B, c, h, w = image.shape
        if (h, w) != (resolution, resolution):
            logits = F.interpolate(logits, resolution, mode='bicubic')

        return logits # [B, 1, h, w]

    def forward_module(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224,
                       force_comb: bool = False) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            resolution (int): Resolution of the output tensor.
            force_comb (bool): If True, force to get logits with all combination audio and image.

        Returns:
            torch.Tensor: Logits from the decoder.
        """
        # N image, 1 embedding case -> [B_i, h, w]
        if embedding.shape[0] != image.shape[0] and embedding.shape[0] == 1:
            embeddings = embedding.repeat(image.shape[0], 1)
            logits = self.forward_decoder(image, embeddings, resolution)

        # N image, M embedding case -> [B_i, B_e, h, w]
        elif embedding.shape[0] != image.shape[0] and embedding.shape[0] != 1 and image.shape[0] != 1 or force_comb:
            logit_list = []
            for i in range(embedding.shape[0]):
                embeddings = embedding[i].unsqueeze(0).repeat(image.shape[0], 1)
                logit_list.append(self.forward_decoder(image, embeddings, resolution))
            logits = torch.cat(logit_list, dim=1)

        # N image, N embedding or 1 image, N embedding -> [B_e, h, w]
        else:
            logits = self.forward_decoder(image, embedding, resolution)

        return logits # [B_i, B_e, h, w] or [B, h, w] depending on force_comb or other things

    def forward_module_eval(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224,
                       force_comb: bool = False) -> dict[str, torch.Tensor]:
        '''
        Same spirit as forward_module but returns more things for evaluation purposes.
        '''
        B, c, h, w = image.shape # [B, 3, 352, 352]
        if embedding.shape[0] != image.shape[0] and embedding.shape[0] == 1:
            raise NotImplementedError('forward_module_eval is not meant to be used during training!')
        elif embedding.shape[0] != image.shape[0] and embedding.shape[0] != 1 and image.shape[0] != 1 or force_comb:
            raise NotImplementedError('forward_module_eval is not meant to be used during training!')
        # N image, N embedding or 1 image, N embedding

        embedding = F.normalize(embedding) # [B, C]

        v_d = self.av_grounder.get_pixels(image) # v^D: [B, C, h, w] // [16, 512, 22, 22]
        v_d_sim = torch.einsum('bchw,bc->bhw', F.normalize(v_d), embedding).unsqueeze(1) # cosine similarity --> range [-1, 1] // [16, 1, 22, 22]
        v_d_seg = (v_d_sim + 1) / 2 # rescaled to range [0, 1]

        seg_logit = self.forward_decoder(image, embedding, h) # M^G: [B, 1, h, w] // [16, 1, 352, 352]
        image_mask = self.masker_i(seg_logit, infer=True) # this is the "heatmap", basically a sigmoid of seg_logit // [16, 1, 352, 352]
        # i believe this image_mask has range [0, 1]

        v_i_bp = self.av_grounder.get_pixels(image * image_mask) # v^i before pooling: [B, c, h, w]
        v_i_sim = torch.einsum('bchw,bc->bhw', F.normalize(v_i_bp), embedding) # cosine similarity --> range [-1, 1]
        v_i_seg = (v_i_sim + 1) / 2 # rescaled to range [0, 1]

        masked_vision_outputs_pooled = checkpoint(self._vision_impl, image * image_mask, use_reentrant=False)
        masked_image_emb = self.av_grounder.clip.visual_projection(masked_vision_outputs_pooled)
        v_f_sim = (torch.einsum('bc,bc->b', F.normalize(masked_image_emb), embedding) + 1) / 2 # cosine sim + rescaling to [0, 1]
        v_f_sim = v_f_sim.unsqueeze(1) # to make min/max operations the same otherwise [B] --> [1]

        # this is because we need h resolution for the get_pixels, but need resolution for the rest of the evaluation
        if image_mask.shape[2] != resolution:
            image_mask = F.interpolate(image_mask, resolution)

        if v_d_seg.shape[2] != resolution:
            v_d_seg = F.interpolate(v_d_seg, resolution)

        # these are the values that I will have to store min/max values for boxplots
        return {
            'v_d_seg': v_d_seg,
            'm_i_seg': image_mask,
            'v_i_seg': v_i_seg,
            'v_f_sim': v_f_sim
        }

    def _vision_impl(self, pixel_values):
        """
        Helper to run just the vision model.
        This allows us to checkpoint the massive ViT pass.
        """
        # We only need the pooled output (index 1) usually, but your code uses hidden states too.
        # For encode_masked_vision, you only use [1] (pooled).
        outputs = self.av_grounder.clip.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=False,
            return_dict=False
        )
        return outputs[1]

    def encode_masked_vision(self, image: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Encode masked visual feature both image-level and feature-level.

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float, float]: Feature masked embeddings, masked image embeddings, positive area, negative area.
        """
        B, c, h, w = image.shape
        maskclip_feat = self.av_grounder.get_pixels(image)  # v^D: [B, c, h, w]
        clipseg_mask = self.forward_module(image, embedding, h, force_comb=True)  # M^G: [B, B, h, w]

        # Area
        area_matrix = self.masker_i(clipseg_mask).mean((2, 3))
        positive_area = area_matrix.diagonal().mean()
        negative_area = area_matrix.mean() - positive_area / B

        # Feature level masker
        feature_mask = F.interpolate(self.masker_f(clipseg_mask), maskclip_feat.shape[2])
        norm_feat_mask = feature_mask.sum(dim=(-2,-1)).clamp(1e-6).unsqueeze(-1) # [B, N, 1]
        feature_masked_emb = torch.einsum('bchw,bnhw->bnc', maskclip_feat, feature_mask) / norm_feat_mask

        if B == embedding.shape[0]:
            # Image level masker
            ind = torch.arange(B).to(image.device) # POSSIBLE BUG HERE TRY WITH MIN OF B AND N
            image_mask = self.masker_i(clipseg_mask[ind, ind].unsqueeze(1))  # Positive pair only

            # step 1: forward the query images through the frozen CLIP vision encoder
            masked_vision_outputs_pooled = checkpoint(self._vision_impl, image * image_mask, use_reentrant=False)
            # masked_vision_outputs = self._vision_impl(image * image_mask)

            masked_image_emb = self.av_grounder.clip.visual_projection(masked_vision_outputs_pooled)
        else:
            masked_image_emb = []
            for n in range(embedding.shape[0]):
                image_mask = self.masker_i(clipseg_mask[:, n].unsqueeze(1))
                masked_vision_outputs_pooled = checkpoint(self._vision_impl, image * image_mask, use_reentrant=False)
                masked_image_emb.append(self.av_grounder.clip.visual_projection(masked_vision_outputs_pooled))
            masked_image_emb = torch.cat(masked_image_emb, dim=1)

        return feature_masked_emb, masked_image_emb, positive_area, negative_area

    def forward(self, image: torch.Tensor, pred_emb: torch.Tensor, resolution: int = 224, **kwargs) -> dict:
        """
        Forward pass of ACL model.

        Args:
            image (torch.Tensor): Input image tensor.
            pred_emb (torch.Tensor): Condition pred_emb tensor for grounder.
            resolution (int): Resolution of the output tensor.

        Returns:
            dict: Output dictionary containing relevant tensors.
        """
        if self.training:
            # basically forward for silence audio
            pred_emb_sil = kwargs.get('pred_emb_silence', None)
            out_dict_sil = {}
            if pred_emb_sil != None:
                sil_v_f, sil_v_i, sil_p_area, sil_n_area = self.encode_masked_vision(image, pred_emb_sil)
                out_dict_sil = {'sil_v_f': sil_v_f, 'sil_v_i': sil_v_i, 'sil_p_area': sil_p_area, 'sil_n_area': sil_n_area}

            # basically forward for noise audio (only gaussian noise)
            pred_emb_noise = kwargs.get('pred_emb_noise', None)
            out_dict_noise = {}
            if pred_emb_noise != None:
                noise_v_f, noise_v_i, noise_p_area, noise_n_area = self.encode_masked_vision(image, pred_emb_noise)
                out_dict_noise = {'noise_v_f': noise_v_f, 'noise_v_i': noise_v_i, 'noise_p_area': noise_p_area, 'noise_n_area': noise_n_area}

            # forward for real san audios
            pred_emb_real_san = kwargs.get('pred_emb_real_san', None)
            out_dict_real_san = {}
            if pred_emb_real_san != None:
                rsan_v_f, rsan_v_i, rsan_p_area, rsan_n_area = self.encode_masked_vision(image, pred_emb_real_san)
                out_dict_real_san = {'rsan_v_f': rsan_v_f, 'rsan_v_i': rsan_v_i, 'rsan_p_area': rsan_p_area, 'rsan_n_area': rsan_n_area}

            # forward for noisy audio (original + noise)
            pred_emb_noisy = kwargs.get('pred_emb_noisy', None)
            out_dict_noisy = {}
            if pred_emb_noisy != None:
                noisy_v_f, noisy_v_i, noisy_p_area, noisy_n_area = self.encode_masked_vision(image, pred_emb_noisy)
                out_dict_noisy = {'noisy_v_f': noisy_v_f, 'noisy_v_i': noisy_v_i, 'noisy_p_area': noisy_p_area, 'noisy_n_area': noisy_n_area}

            # finally forward for original audios
            v_f, v_i, p_area, n_area = self.encode_masked_vision(image, pred_emb)
            out_dict = {'v_f': v_f, 'v_i': v_i, 'p_area': p_area, 'n_area': n_area, **out_dict_noisy, **out_dict_sil, **out_dict_noise, **out_dict_real_san}

        else:
            out_dict = {}
            out_dict['positive'] = self.forward_module_eval(image, pred_emb, resolution)

            pred_emb_sil = kwargs.get('pred_emb_silence', None)
            if pred_emb_sil != None:
                out_dict['silence'] = self.forward_module_eval(image, pred_emb_sil.repeat(pred_emb.shape[0], 1), resolution)

            pred_emb_noise = kwargs.get('pred_emb_noise', None)
            if pred_emb_noise != None:
                out_dict['noise'] = self.forward_module_eval(image, pred_emb_noise.repeat(pred_emb.shape[0], 1), resolution)

            pred_emb_offscreen = kwargs.get('pred_emb_offscreen', None)
            if pred_emb_offscreen != None:
                out_dict['offscreen'] = self.forward_module_eval(image, pred_emb_offscreen, resolution)

        return out_dict

    def forward_for_validation(self, image: torch.Tensor, pred_emb: torch.Tensor, resolution: int = 224, **kwargs) -> dict:
        """
        Forward pass of ACL model especifically for the validation step during training.

        Args:
            image (torch.Tensor): Input image tensor.
            pred_emb (torch.Tensor): Condition pred_emb tensor for grounder.
            resolution (int): Resolution of the output tensor.

        Returns:
            dict: Output dictionary containing relevant tensors.
        """
        # basically forward for silence audio
        pred_emb_sil = kwargs.get('pred_emb_silence', None)
        out_dict_sil = {}
        if pred_emb_sil != None:
            sil_v_f, sil_v_i, sil_p_area, sil_n_area = self.encode_masked_vision(image, pred_emb_sil)
            out_dict_sil = {'sil_v_f': sil_v_f, 'sil_v_i': sil_v_i, 'sil_p_area': sil_p_area, 'sil_n_area': sil_n_area}

        # basically forward for noise audio (only gaussian noise)
        pred_emb_noise = kwargs.get('pred_emb_noise', None)
        out_dict_noise = {}
        if pred_emb_noise != None:
            noise_v_f, noise_v_i, noise_p_area, noise_n_area = self.encode_masked_vision(image, pred_emb_noise)
            out_dict_noise = {'noise_v_f': noise_v_f, 'noise_v_i': noise_v_i, 'noise_p_area': noise_p_area, 'noise_n_area': noise_n_area}

        # forward for noisy audio (original + noise)
        pred_emb_noisy = kwargs.get('pred_emb_noisy', None)
        out_dict_noisy = {}
        if pred_emb_noisy != None:
            noisy_v_f, noisy_v_i, noisy_p_area, noisy_n_area = self.encode_masked_vision(image, pred_emb_noisy)
            out_dict_noisy = {'noisy_v_f': noisy_v_f, 'noisy_v_i': noisy_v_i, 'noisy_p_area': noisy_p_area, 'noisy_n_area': noisy_n_area}

        # forward for real san audios
        pred_emb_real_san = kwargs.get('pred_emb_real_san', None)
        out_dict_real_san = {}
        if pred_emb_real_san != None:
            rsan_v_f, rsan_v_i, rsan_p_area, rsan_n_area = self.encode_masked_vision(image, pred_emb_real_san)
            out_dict_real_san = {'rsan_v_f': rsan_v_f, 'rsan_v_i': rsan_v_i, 'rsan_p_area': rsan_p_area, 'rsan_n_area': rsan_n_area}

        # finally forward for original audios
        v_f, v_i, p_area, n_area = self.encode_masked_vision(image, pred_emb)
        out_dict = {'v_f': v_f, 'v_i': v_i, 'p_area': p_area, 'n_area': n_area, **out_dict_noisy, **out_dict_sil, **out_dict_noise, **out_dict_real_san}

        seg_logit = self.forward_module(image, pred_emb, resolution)
        heatmap = self.masker_i(seg_logit, infer=True)

        out_dict = {**out_dict, 'heatmap': heatmap}

        return out_dict

    def save(self, model_dir: str):
        """
        Save model parameters to a file. (Only trainable parts)

        Args:
            model_dir (str): Directory to save the model.
        """
        ckp = {'audio_proj': self.audio_proj.state_dict(), 'masker_i': self.masker_i.state_dict()}
        torch.save(ckp, model_dir)

    def load(self, model_dir: str):
        """
        Load model parameters from a file. (Only trainable parts)

        Args:
            model_dir (str): Directory to load the model from.
        """
        ckp = torch.load(model_dir, map_location=self.device)
        self.audio_proj.load_state_dict(ckp['audio_proj'])
        self.masker_i.load_state_dict(ckp['masker_i'])

class ADCL(ACL):
    '''
    Audio-DeGrounded Contrastive Learning (ACL) model. Removes the CLIPSeg decoder step to evaluate
    how much does it affect the final ACL model.
    '''
    def __init__(self, conf_file, device, model_path):
        super().__init__(conf_file, device, model_path)

        self.m = TrainableADCLSigmoid(
            self.args.model.epsilon,
            self.args.model.epsilon2,
            self.args.model.tau,
            self.args.model.trainable_sigmoid
        )
        self.m.to(self.device)

    def audio_visual_sim(self, v_d: torch.Tensor, embedding: torch.Tensor, resolution: int = 224) -> torch.Tensor:
        """
        Forward pass of audio-visual grounder

        Args:
            v_d (torch.Tensor): Input v_d tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            resolution (int): Resolution of the output.
            ignore_indices (list): List of indices to ignore.

        Returns:
            torch.Tensor: Logits from the decoder.
        """

        logits = torch.einsum('bchw,bc->bhw', F.normalize(v_d, dim=1), F.normalize(embedding, dim=1)) # [B, h, w]

        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(1)
        else:
            logits = logits.unsqueeze(1)

        B, c, h, w = v_d.shape
        if (h, w) != (resolution, resolution):
            logits = F.interpolate(logits, resolution, mode='bicubic')

        return logits # [B, 1, h, w]

    def forward_module(self, v_d: torch.Tensor, embedding: torch.Tensor, resolution: int = 224,
                       force_comb: bool = False) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            v_d (torch.Tensor): Input v_d tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            force_comb (bool): If True, force to get logits with all combination audio and v_d.

        Returns:
            torch.Tensor: Logits from the decoder.
        """
        # N v_d, 1 embedding case -> [B_i, h, w]
        if embedding.shape[0] != v_d.shape[0] and embedding.shape[0] == 1:
            embeddings = embedding.repeat(v_d.shape[0], 1)
            logits = self.audio_visual_sim(v_d, embeddings, resolution)

        # N v_d, M embedding case -> [B_i, B_e, h, w]
        elif embedding.shape[0] != v_d.shape[0] and embedding.shape[0] != 1 and v_d.shape[0] != 1 or force_comb:
            logit_list = []
            for i in range(embedding.shape[0]):
                embeddings = embedding[i].unsqueeze(0).repeat(v_d.shape[0], 1)
                logit_list.append(self.audio_visual_sim(v_d, embeddings, resolution))
            logits = torch.cat(logit_list, dim=1)

        # N v_d, N embedding or 1 v_d, N embedding -> [B_e, h, w]
        else:
            logits = self.audio_visual_sim(v_d, embedding, resolution)

        return logits # [B_i, B_e, h, w] or [B, h, w] depending on force_comb or other things

    def encode_masked_vision(self, image: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Following procedure in:
        https://github.com/jinxiang-liu/SSL-TIE/blob/c49e6a94e4ed63bba864ba01e9138451ca1cc801/models/model.py#L57
        """
        trimap = self.args.model.trimap

        B, c, H, W = image.shape
        v_d = self.av_grounder.get_pixels(image) # v^D: [B, c, h, w]

        # similarity for (soft) positives
        sim_i_i = self.audio_visual_sim(v_d, embedding, H) # A: [B, 1, H, W]

        # similarity for (soft) negatives
        sim_i_j = self.forward_module(v_d, embedding, H, force_comb=True)  # A0: [B, B, H, W]
        sim_i_j = remove_diagonal(sim_i_j) # [B, B-1, H, W] removed positive (diagonal elements)

        Pos_mask = self.m(sim_i_i, True)
        Pos_mask_i_j = self.m(sim_i_j, True)

        if trimap:
            Pos2 = self.m(sim_i_i, False)
            Neg_mask = 1 - Pos2
        else:
            Neg_mask = 1 - Pos_mask

        sim = (Pos_mask * sim_i_i).view(*sim_i_i.shape[:2],-1).sum(-1) / (Pos_mask.view(*Pos_mask.shape[:2],-1).sum(-1))                # easy positives [B, 1]
        sim1 = (Pos_mask_i_j * sim_i_j).view(*sim_i_j.shape[:2],-1).sum(-1) / (Pos_mask_i_j.view(*Pos_mask_i_j.shape[:2],-1).sum(-1))   # easy negatives [B, B-1]
        sim2 = (Neg_mask * sim_i_i).view(*sim_i_i.shape[:2],-1).sum(-1) / (Neg_mask.view(*Neg_mask.shape[:2],-1).sum(-1))               # hard negatives [B, 1]

        logits = torch.cat((sim, sim1, sim2), 1) # / self.temperature # done in loss

        # Area
        positive_area = Pos_mask.mean()
        negative_area = Neg_mask.mean()

        return logits, None, positive_area, negative_area

    def forward_for_validation(self, image: torch.Tensor, pred_emb: torch.Tensor, resolution: int = 224, **kwargs) -> dict:
        # basically forward for silence audio
        pred_emb_sil = kwargs.get('pred_emb_silence', None)
        out_dict_sil = {}
        if pred_emb_sil != None:
            sil_v_f, sil_v_i, sil_p_area, sil_n_area = self.encode_masked_vision(image, pred_emb_sil.repeat(pred_emb.shape[0], 1))
            out_dict_sil = {'sil_v_f': sil_v_f, 'sil_v_i': sil_v_i, 'sil_p_area': sil_p_area, 'sil_n_area': sil_n_area}

        # basically forward for noise audio (only gaussian noise)
        pred_emb_noise = kwargs.get('pred_emb_noise', None)
        out_dict_noise = {}
        if pred_emb_noise != None:
            noise_v_f, noise_v_i, noise_p_area, noise_n_area = self.encode_masked_vision(image, pred_emb_noise.repeat(pred_emb.shape[0], 1))
            out_dict_noise = {'noise_v_f': noise_v_f, 'noise_v_i': noise_v_i, 'noise_p_area': noise_p_area, 'noise_n_area': noise_n_area}

        # forward for noisy audio (original + noise)
        pred_emb_noisy = kwargs.get('pred_emb_noisy', None)
        out_dict_noisy = {}
        if pred_emb_noisy != None:
            noisy_v_f, noisy_v_i, noisy_p_area, noisy_n_area = self.encode_masked_vision(image, pred_emb_noisy)
            out_dict_noisy = {'noisy_v_f': noisy_v_f, 'noisy_v_i': noisy_v_i, 'noisy_p_area': noisy_p_area, 'noisy_n_area': noisy_n_area}

        # finally forward for original audios
        v_f, v_i, p_area, n_area = self.encode_masked_vision(image, pred_emb)
        out_dict = {'v_f': v_f, 'v_i': v_i, 'p_area': p_area, 'n_area': n_area, **out_dict_noisy, **out_dict_sil, **out_dict_noise}

        v_d = self.av_grounder.get_pixels(image) # v^D: [B, c, h, w]
        seg_logit = self.forward_module(v_d, pred_emb, resolution)
        heatmap = self.m(seg_logit, True)

        out_dict = {**out_dict, 'heatmap': heatmap}

        return out_dict

    def forward_module_eval(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224,
                       force_comb: bool = False) -> dict[str, torch.Tensor]:
        B, c, H, W = image.shape
        v_d = self.av_grounder.get_pixels(image) # v^D: [B, c, h, w]

        # similarity for (soft) positives
        sim_i_i = self.audio_visual_sim(v_d, embedding, resolution) # A: [B, 1, H, W]
        Pos_mask = self.m(sim_i_i, True)

        v_d_seg = (sim_i_i + 1) / 2 # rescale to range [0, 1] from [-1, 1]

        return {
            'v_d_seg': v_d_seg,
            'm_i_seg': Pos_mask
        }

    def save(self, model_dir: str):
        """
        Save model parameters to a file. (Only trainable parts)

        Args:
            model_dir (str): Directory to save the model.
        """
        ckp = {
            'audio_proj': self.audio_proj.state_dict(),
            'masker_i': self.masker_i.state_dict(),
            'm': self.m.state_dict(),
        }
        torch.save(ckp, model_dir)

    def load(self, model_dir: str):
        """
        Load model parameters from a file. (Only trainable parts)

        Args:
            model_dir (str): Directory to load the model from.
        """
        ckp = torch.load(model_dir, map_location=self.device)
        self.audio_proj.load_state_dict(ckp['audio_proj'])
        self.masker_i.load_state_dict(ckp['masker_i'])
        self.m.load_state_dict(ckp['m'])
