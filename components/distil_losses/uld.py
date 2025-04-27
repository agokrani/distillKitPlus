from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer

from .base import DistilLoss

class ULD(DistilLoss):
    """
    Universal LogitDistillation (ULD).
    Compares the L1 distance between the top-k probabilities of student and teacher.
    """
    def __init__(self, temperature: float = 2.0, k: int = 100, ignore_index: int = -100):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.ignore_index = ignore_index

    def forward(
        self,
        trainer_instance: Trainer,
        logits: Tensor,
        teacher_logits: Tensor,
        mask: Tensor,
        batch: Dict[str, Any],
        **kwargs,
    ) -> Tensor:
        """
        Compute the Universal LogitDistillation (ULD) loss (vectorized v2).

        Args:
            trainer_instance (Trainer): The HuggingFace Trainer instance.
            logits (Tensor): Student logits, shape (B, seq_length, vocab_size_student).
            teacher_logits (Tensor): Teacher logits, shape (B, seq_length, vocab_size_teacher).
            mask (Tensor): Attention mask (may not be directly used here but part of signature).
            batch (dict): Dictionary containing batch data. Must include:
                - "labels": Tensor of student labels, shape (B, seq_length).
                - "teacher_labels": Tensor of teacher labels, shape (B, seq_length).
            **kwargs: Additional arguments (ignored).

        Returns:
            Tensor: Mean ULD loss across the batch.
        """
        student_labels = batch.get("labels", None)
        if student_labels is None:
            raise ValueError(
                "ULD loss requires 'labels' in batch dict for student answer positions"
            )
        teacher_labels = batch.get("teacher_labels", None)
        if teacher_labels is None:
            raise ValueError(
                "ULD loss requires 'teacher_labels' in batch dict for teacher answer positions"
            )

        B, seq_len, vocab_student = logits.shape
        _, _, vocab_teacher = teacher_logits.shape
        device = logits.device

        # Compute boolean masks indicating valid tokens.
        student_valid = student_labels != self.ignore_index  # shape: (B, seq_len)
        teacher_valid = teacher_labels != self.ignore_index  # shape: (B, seq_len)

        # Compute answer lengths: number of valid tokens per sample.
        student_answer_length = student_valid.sum(dim=1)  # shape: (B,)
        teacher_answer_length = teacher_valid.sum(dim=1)  # shape: (B,)

        # Determine the valid (overlap) length per sample.
        valid_length = torch.min(
            student_answer_length, teacher_answer_length
        )  # shape: (B,)

        # --- Compute start indices ---
        student_start = student_valid.int().argmax(dim=1)  # shape: (B,)
        teacher_start = teacher_valid.int().argmax(dim=1)  # shape: (B,)
        # --- End start index computation ---

        # Apply softmax with temperature scaling to FULL logits
        student_probs_full = F.softmax(
            logits / self.temperature, dim=-1
        )  # (B, seq_len, V_s)
        teacher_probs_full = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )  # (B, seq_len, V_t)

        # Determine the maximum length needed for gathering
        max_valid = (
            valid_length.max().item() if B > 0 else 0
        )

        if max_valid == 0:
            return torch.tensor(
                0.0, device=device, requires_grad=logits.requires_grad
            )

        # Create relative index vector
        rel_idx = torch.arange(max_valid, device=device).unsqueeze(0) # (1, max_valid)

        # Compute absolute indices to gather
        student_indices = student_start.unsqueeze(1) + rel_idx # (B, max_valid)
        teacher_indices = teacher_start.unsqueeze(1) + rel_idx # (B, max_valid)

        # Clamp indices
        student_indices = student_indices.clamp(min=0, max=seq_len - 1)
        teacher_indices = teacher_indices.clamp(min=0, max=seq_len - 1)

        # Gather probability spans
        student_span_probs = torch.gather(
            student_probs_full,
            1,
            student_indices.unsqueeze(-1).expand(B, max_valid, vocab_student),
        ) # (B, max_valid, V_s)
        teacher_span_probs = torch.gather(
            teacher_probs_full,
            1,
            teacher_indices.unsqueeze(-1).expand(B, max_valid, vocab_teacher),
        ) # (B, max_valid, V_t)

        # Sort probabilities along the vocabulary dimension.
        k_student = min(self.k, vocab_student)
        k_teacher = min(self.k, vocab_teacher)
        sorted_student_probs = torch.topk(student_span_probs, k=k_student, dim=-1, largest=True).values
        sorted_teacher_probs = torch.topk(teacher_span_probs, k=k_teacher, dim=-1, largest=True).values

        # Pad the smaller tensor if k values or vocab sizes differ
        if k_student != k_teacher:
            if k_student < k_teacher:
                pad_size = k_teacher - k_student
                sorted_student_probs = F.pad(sorted_student_probs, (0, pad_size), value=0.0)
            else:
                pad_size = k_student - k_teacher
                sorted_teacher_probs = F.pad(sorted_teacher_probs, (0, pad_size), value=0.0)

        # Compute element-wise L1 difference over the top-k.
        l1_diff = torch.abs(sorted_student_probs - sorted_teacher_probs).sum(dim=-1) # (B, max_valid)

        # Create valid position mask
        valid_mask = (rel_idx < valid_length.unsqueeze(1)).float() # (B, max_valid)

        # Compute per-sample loss
        clamped_valid_length = torch.clamp(valid_length.float(), min=1.0)
        sample_loss = (l1_diff * valid_mask).sum(dim=1) / clamped_valid_length

        # Zero out loss for samples with no valid overlap
        sample_loss = torch.where(
            valid_length == 0, torch.zeros_like(sample_loss), sample_loss
        )

        # Average loss over the batch
        distillation_loss = sample_loss.mean()

        return distillation_loss 