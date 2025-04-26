"""
This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L14)
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def reverse_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    student_probs: Optional[torch.Tensor] = None,
    teacher_logprobs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if student_probs is None:
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    if teacher_logprobs is None:
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


class ReverseKL(DistilLoss):
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return reverse_kl(logits, teacher_logits, mask) 