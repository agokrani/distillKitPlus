"""
Decoupled Knowledge Distillation
https://arxiv.org/abs/2203.08679

This implementation is based on https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py
"""
import torch
from torch.nn import functional as F
from transformers import Trainer
from .base import DistilLoss


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def dkd_loss(
    logits_student_in,
    logits_teacher_in,
    target,
    mask,
    alpha=1.0,
    beta=5,
    temperature=1.0,
    logit_stand=False,
):
    logits_student_flat = logits_student_in.view(-1, logits_student_in.size(-1))
    logits_teacher_flat = logits_teacher_in.view(-1, logits_teacher_in.size(-1))
    target_flat = target.flatten()
    mask_flat = mask.flatten().bool()

    valid_logits_student = logits_student_flat[mask_flat]
    valid_logits_teacher = logits_teacher_flat[mask_flat]
    valid_target = target_flat[mask_flat]
    num_valid_tokens = mask_flat.sum()

    if num_valid_tokens == 0:
        return torch.tensor(0.0, device=logits_student_in.device, requires_grad=True)

    logits_student = normalize(valid_logits_student) if logit_stand else valid_logits_student
    logits_teacher = normalize(valid_logits_teacher) if logit_stand else valid_logits_teacher

    gt_mask = _get_gt_mask(logits_student, valid_target)
    other_mask = _get_other_mask(logits_student, valid_target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='sum')
    tckd_loss = tckd_loss * (temperature**2) / num_valid_tokens

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
    nckd_loss = nckd_loss * (temperature**2) / num_valid_tokens

    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(DistilLoss):
    """
    Implementation of DKD for Language Modeling
    paper: https://arxiv.org/abs/2203.08679
    code: https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, temperature: float = 1.0, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        trainer_instance: Trainer,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch,
        **kwargs,
    ) -> torch.Tensor:
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("DKD loss requires 'labels' in batch dict.")

        valid_token_mask = (labels != self.ignore_index)
        logit_seq_len = logits.size(1)
        label_seq_len = labels.size(1)
        if logit_seq_len != label_seq_len:
            if logit_seq_len > label_seq_len:
                logits = logits[:, :label_seq_len, :]
                teacher_logits = teacher_logits[:, :label_seq_len, :]
            else:
                raise ValueError(f"Logit sequence length ({logit_seq_len}) < Label sequence length ({label_seq_len}). Check model/data.")
            valid_token_mask = (labels != self.ignore_index)

        return dkd_loss(
            logits_student_in=logits,
            logits_teacher_in=teacher_logits,
            target=labels,
            mask=valid_token_mask,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        ) 