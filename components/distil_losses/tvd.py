"""
This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L55)
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def tvd(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    student_probs: Optional[torch.Tensor] = None,
    teacher_probs: Optional[torch.Tensor] = None,
):
    if student_probs is None:
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(
        torch.abs(teacher_probs - student_probs), inf_mask, 0
    )
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


class TVD(DistilLoss):
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return tvd(logits, teacher_logits, mask) 