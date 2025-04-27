"""
This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L4)
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    teacher_probs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits / temperature, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1)) / torch.sum(mask.view(-1))
    distil_loss = distil_loss * (temperature ** 2)
    return distil_loss


class ForwardKL(DistilLoss):
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        return forward_kl(logits, teacher_logits, mask, temperature=temperature) 