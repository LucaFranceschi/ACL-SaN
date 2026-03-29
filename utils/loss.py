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

    logits = torch.einsum('bnc,bc->bn', F.normalize(v_f, dim=2), F.normalize(pred_emb, dim=1))

    neg_audios = []
    neg_audios.append(kwargs.get('pred_emb_san', None))
    neg_audios.append(kwargs.get('pred_emb_real_san', None))
    neg_audios = [val for val in neg_audios if val is not None]
    if len(neg_audios) > 0:
        neg_audios = torch.cat(neg_audios, dim=0)

        # neg_audios has shape [K, C]
        # b is already broadcasted in the uncommented version
        # neg_audios = neg_audios.unsqueeze(1).repeat(1, B, 1)
        # neg_sim = torch.einsum('bnc,kbc->bkn', F.normalize(v_f, dim=2), neg_audios)
        neg_sim = torch.einsum('bnc,kc->bkn', F.normalize(v_f, dim=2), F.normalize(neg_audios, dim=1)).mean(dim=2)

        logits = torch.cat((logits, neg_sim), dim=1)

    labels = torch.eye(logits.shape[0], logits.shape[1], device=pred_emb.device)

    loss = 0.5 * (F.cross_entropy(logits * beta, labels) + F.cross_entropy(logits.T * beta, labels.T))

    return loss

def silence_l(noise_n_area: torch.Tensor, v_f: torch.Tensor, n_thr: float = 0.0, san = False, **kwargs) -> torch.Tensor:
    loss = torch.zeros([]).to(v_f.device) # using here v_f only to get device since it should be guaranteed to exist
    if san:
        loss = torch.abs(noise_n_area - n_thr)
    return loss

def noise_l(sil_n_area: torch.Tensor, v_f: torch.Tensor, n_thr: float = 0.0, san = False, **kwargs) -> torch.Tensor:
    loss = torch.zeros([]).to(v_f.device)
    if san:
        loss = torch.abs(sil_n_area - n_thr)
    return loss

def diff_san_l(v_f: torch.Tensor, pred_emb: torch.Tensor, noisy_v_f: torch.Tensor, pred_emb_noisy: torch.Tensor, **kwargs) -> torch.Tensor:
    logits = torch.einsum('bnc,bc->bn', F.normalize(v_f, dim=2), F.normalize(pred_emb, dim=1))
    logits_noisy = torch.einsum('bnc,bc->bn', F.normalize(noisy_v_f, dim=2), F.normalize(pred_emb_noisy, dim=1))

    return F.mse_loss(logits_noisy, logits)
