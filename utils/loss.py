import torch
import torch.nn.functional as F
from typing import Optional

def infonce(pred: torch.Tensor, target: torch.Tensor, beta: float = 1/0.07, **kwargs) -> torch.Tensor:
    '''
    Compute the InfoNCE (Noise Contrastive Estimation) loss.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.
        beta (float, optional): Temperature parameter. Default is 1/0.07.

    Returns:
        torch.Tensor: InfoNCE loss.
    '''
    B = pred.shape[0]
    logits = torch.einsum('nc,mc->nm', F.normalize(pred), F.normalize(target)) * beta
    labels = torch.arange(B).long().to(pred.device)
    loss = F.cross_entropy(logits, labels)

    return loss


def area_reg(p_area: torch.Tensor, n_area: torch.Tensor, p_thr: float = 0.4, n_thr: float = 0.0,
             **kwargs) -> torch.Tensor:
    '''
    Compute the area regularization loss.

    Args:
        p_area (torch.Tensor): Positive area tensor.
        n_area (torch.Tensor): Negative area tensor.
        p_thr (float, optional): Expected positive area. Default is 0.4.
        n_thr (float, optional): Expected negative area. Default is 0.0.

    Returns:
        torch.Tensor: Area regularization loss.
    '''
    loss = torch.abs(p_area - p_thr) + torch.abs(n_area - n_thr)
    return loss


def acl_i(v_i: torch.Tensor, pred_emb: torch.Tensor, beta: float = 1 / 0.07, **kwargs) -> torch.Tensor:
    '''
    Compute the image-level audio-grounded contrastive learning (ACL_I) loss.

    Args:
        v_i (torch.Tensor): Image-level audio-grounded visual embedding tensor.
        pred_emb (torch.Tensor): Audio-driven embedding tensor.
        beta (float, optional): Temperature parameter. Default is 1/0.07.

    Returns:
        torch.Tensor: Image-level ACL loss
    '''
    loss = 0.5 * (infonce(pred_emb, v_i, beta=beta) + infonce(v_i, pred_emb, beta=beta))

    return loss

def acl_f(v_f: torch.Tensor, pred_emb: torch.Tensor, beta: float = 1 / 0.07, **kwargs) -> torch.Tensor:
    '''
    Compute the feature-level audio-grounded contrastive learning (ACL_F) loss.

    Args:
        v_f (torch.Tensor): Feature-level audio-grounded visual embedding tensor.
        pred_emb (torch.Tensor): Audio-driven embedding tensor.
        beta (float, optional): Temperature parameter. Default is 1/0.07.

    Returns:
        torch.Tensor: Feature-level ACL loss
    '''
    B, _, C = v_f.size()

    full_v_f = [v_f] # [B, N=B, C]
    full_v_f.append(kwargs.get('sil_v_f', None)) # [B, N=1, C]
    full_v_f.append(kwargs.get('noise_v_f', None)) # [B, N=1, C]
    full_v_f.append(kwargs.get('rsan_v_f', None)) # [B, N=5 or so, C]
    full_v_f = [val for val in full_v_f if val is not None]
    full_v_f = torch.cat(full_v_f, dim=1) # [B, N', C]

    full_pred_emb = [pred_emb] # [B, C]
    full_pred_emb.append(kwargs.get('pred_emb_silence', None)) # [N=1+1, C]
    full_pred_emb.append(kwargs.get('pred_emb_noise', None)) # [N=1+1, C]
    full_pred_emb.append(kwargs.get('pred_emb_real_san', None)) # [N=5 or so, C]
    full_pred_emb = [val for val in full_pred_emb if val is not None]
    full_pred_emb = torch.cat(full_pred_emb, dim=0) # [N', C]

    logits = torch.einsum('bnc,nc->bn', F.normalize(full_v_f, dim=2), F.normalize(full_pred_emb, dim=1))

    labels = torch.eye(logits.shape[0], logits.shape[1], device=pred_emb.device)

    loss = 0.5 * (F.cross_entropy(logits * beta, labels) + F.cross_entropy(logits.T * beta, labels.T))

    return loss

def silence_l(noise_n_area: torch.Tensor, noise_p_area: torch.Tensor, v_f: torch.Tensor, n_thr: float = 0.0, **kwargs) -> torch.Tensor:
    loss = torch.zeros([]).to(v_f.device) # using here v_f only to get device since it should be guaranteed to exist
    loss = torch.abs(noise_n_area + noise_p_area - n_thr)
    return loss

def noise_l(sil_n_area: torch.Tensor, sil_p_area: torch.Tensor, v_f: torch.Tensor, n_thr: float = 0.0, **kwargs) -> torch.Tensor:
    loss = torch.zeros([]).to(v_f.device)
    loss = torch.abs(sil_n_area + sil_p_area - n_thr)
    return loss

def diff_san_l(v_f: torch.Tensor, pred_emb: torch.Tensor, noisy_v_f: torch.Tensor, pred_emb_noisy: torch.Tensor, **kwargs) -> torch.Tensor:
    logits = torch.einsum('bnc,bc->bn', F.normalize(v_f, dim=2), F.normalize(pred_emb, dim=1))
    logits_noisy = torch.einsum('bnc,bc->bn', F.normalize(noisy_v_f, dim=2), F.normalize(pred_emb_noisy, dim=1))

    return F.mse_loss(logits_noisy, logits)

def adcl_f(v_f: torch.Tensor, pred_emb: torch.Tensor, beta: float = 1 / 0.07, **kwargs) -> torch.Tensor:
    B, _ = v_f.size() # remember v_f is logits and has shape [B, B+1]

    logits = [v_f]
    logits.append(kwargs.get('sil_v_f', None))
    logits.append(kwargs.get('noise_v_f', None))
    logits = [val for val in logits if val is not None]

    logits = torch.cat(logits, dim=1)

    labels = torch.zeros_like(logits, device=v_f.device)
    labels[:, 0] = 1

    loss = 0.5 * (F.cross_entropy(logits * beta, labels) + F.cross_entropy(logits.T * beta, labels.T))

    return loss