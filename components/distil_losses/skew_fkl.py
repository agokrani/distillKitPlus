"""
DistiLLM: Towards Streamlined Distillation for Large Language Models
https://arxiv.org/abs/2402.03898

This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L66)
"""
import torch
from torch.nn import functional as F
from .base import DistilLoss


class SkewForwardKL(DistilLoss):
    def __init__(self, skew_beta: float = 0.1):
        super().__init__()
        self.skew_beta = skew_beta

    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits / temperature, dim=-1, dtype=torch.float32)
        mixed_probs = (
            self.skew_beta * teacher_probs
            + (1 - self.skew_beta) * student_probs
        )
        mixed_logprobs = torch.log(mixed_probs + 1e-7)
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.sum(x * mask.view(-1)) / torch.sum(mask.view(-1))
        distil_loss = distil_loss * (temperature ** 2)
        return distil_loss 