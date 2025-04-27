"""
Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models
https://arxiv.org/abs/2404.02657
"""
import torch
from torch.nn import functional as F
from .base import DistilLoss


class AdaptiveKL(DistilLoss):
    """
    https://arxiv.org/abs/2404.02657
    """

    def __init__(self, head_threshold: float = 0.5):
        super().__init__()
        # indicates mu in the paper
        self.head_threshold = head_threshold

    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        # 1. compute weights
        sorted_logits, sorted_indices = torch.sort(teacher_logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        head_mask = cumulative_probs >= self.head_threshold
        gap_distance = torch.abs(teacher_probs - student_probs)
        head_gap = (head_mask * gap_distance).sum(dim=-1).view(-1)
        tail_gap = (~head_mask * gap_distance).sum(dim=-1).view(-1)
        denominator = head_gap + tail_gap
        # Add a small epsilon for numerical stability
        epsilon = torch.finfo(denominator.dtype).eps # Get smallest representable positive value
        fkl_weight = head_gap / (denominator + epsilon)
        rkl_weight = tail_gap / (denominator + epsilon)

        # 2. forward kl
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(
            teacher_probs * student_logprobs, torch.isinf(logits), 0
        )
        fkl = torch.sum(prod_probs, dim=-1).view(-1)

        # 3. reverse kl
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
        prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        rkl = torch.sum(prod_probs, dim=-1).view(-1)

        # weighting
        distil_loss = fkl_weight * fkl + rkl_weight * rkl
        distil_loss = -torch.sum(distil_loss * mask.view(-1), dim=0) / torch.sum(
            mask.view(-1), dim=0
        )
        return distil_loss 