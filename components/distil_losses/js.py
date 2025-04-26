"""
This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L32)
"""
import torch
from torch.nn import functional as F
from .base import DistilLoss


class JS(DistilLoss):
    def __init__(self, teacher_weight: float = 0.1):
        super().__init__()
        self.teacher_weight = teacher_weight

    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        mixed_probs = (
            self.teacher_weight * teacher_probs
            + (1 - self.teacher_weight) * student_probs
        )

        teacher_logprobs = torch.log(teacher_probs)
        student_logprobs = torch.log(student_probs)
        mixed_logprobs = torch.log(mixed_probs)

        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
        # reverse kl
        prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = (
            (1 - self.teacher_weight)
            * -torch.sum(x * mask.view(-1), dim=0)
            / torch.sum(mask.view(-1), dim=0)
        )
        # forward kl
        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss += (
            (self.teacher_weight)
            * -torch.sum(x * mask.view(-1), dim=0)
            / torch.sum(mask.view(-1), dim=0)
        )
        return distil_loss 