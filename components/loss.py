from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor


def forward_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
    inputs: Dict[str, Any] = None,
) -> Tensor:
    """
    Compute the forward KL divergence loss for knowledge distillation.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Temperature parameter for softening distributions
        inputs: Dictionary of input data
    Returns:
        KL divergence loss tensor
    """
    # Scale logits by temperature
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature

    # Handle vocabulary size mismatch
    if teacher_logits_scaled.size(-1) > student_logits_scaled.size(-1):
        # Truncate teacher vocab if larger than student's
        teacher_logits_scaled = teacher_logits_scaled[
            ..., : student_logits_scaled.size(-1)
        ]
    elif teacher_logits_scaled.size(-1) < student_logits_scaled.size(-1):
        # Pad teacher logits with -inf if smaller than student's
        padding_size = student_logits_scaled.size(-1) - teacher_logits_scaled.size(-1)
        padding = torch.full(
            (
                teacher_logits_scaled.size(0),
                teacher_logits_scaled.size(1),
                padding_size,
            ),
            float("-inf"),
            device=teacher_logits_scaled.device,
        )
        teacher_logits_scaled = torch.cat([teacher_logits_scaled, padding], dim=-1)

    # Compute KL divergence loss
    kd_loss = (
        F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction="batchmean",
        )
        * (temperature**2)
        / student_logits.size(1)
    )

    return kd_loss


def uld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    k: int = 100,
    inputs: dict = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """
    Compute the Universal LogitDistillation (ULD) loss (vectorized v2).
    Applies softmax to full logits, then gathers relevant probability spans.

    Args:
        student_logits (Tensor): shape (B, seq_length, vocab_size_student)
        teacher_logits (Tensor): shape (B, seq_length, vocab_size_teacher)
        temperature (float): Temperature scaling for softening distributions.
        inputs (dict): Must contain:
            - "labels": Tensor of student labels, shape (B, seq_length)
            - "teacher_labels": Tensor of teacher labels, shape (B, seq_length)
        ignore_index (int): value used to denote masked tokens. Default is -100.
        **kwargs: Additional arguments.

    Returns:
        Tensor: Mean ULD loss across the batch.
    """
    if inputs is None:
        raise ValueError("`inputs` must be provided and include label information.")

    student_labels = inputs.get("labels", None)
    if student_labels is None:
        raise ValueError(
            "ULD loss requires 'labels' in inputs for student answer positions"
        )
    teacher_labels = inputs.get("teacher_labels", None)
    if teacher_labels is None:
        raise ValueError(
            "ULD loss requires 'teacher_labels' in inputs for teacher answer positions"
        )

    B, seq_len, vocab_student = student_logits.shape
    _, _, vocab_teacher = teacher_logits.shape
    device = student_logits.device

    # Compute boolean masks indicating valid tokens.
    student_valid = student_labels != ignore_index  # shape: (B, seq_len)
    teacher_valid = teacher_labels != ignore_index  # shape: (B, seq_len)

    # Compute answer lengths: number of valid tokens per sample.
    student_answer_length = student_valid.sum(dim=1)  # shape: (B,)
    teacher_answer_length = teacher_valid.sum(dim=1)  # shape: (B,)

    # Determine the valid (overlap) length per sample.
    valid_length = torch.min(
        student_answer_length, teacher_answer_length
    )  # shape: (B,)

    # --- Compute start indices (FIXED) ---
    # Find the index of the first valid token. Handles all-padding case (returns 0).
    student_start = student_valid.int().argmax(dim=1)  # shape: (B,)
    teacher_start = teacher_valid.int().argmax(dim=1)  # shape: (B,)
    # --- End start index computation ---

    # Apply softmax with temperature scaling to FULL logits
    student_probs_full = F.softmax(
        student_logits / temperature, dim=-1
    )  # (B, seq_len, V_s)
    teacher_probs_full = F.softmax(
        teacher_logits / temperature, dim=-1
    )  # (B, seq_len, V_t)

    # Determine the maximum length needed for gathering (based on min lengths)
    max_valid = (
        valid_length.max().item() if B > 0 else 0
    )  # maximum valid span length over the batch

    if max_valid == 0:  # Handle batch with no overlapping valid tokens
        return torch.tensor(
            0.0, device=device, requires_grad=student_logits.requires_grad
        )

    # Create a relative index vector for the maximum valid span length.
    rel_idx = torch.arange(max_valid, device=device).unsqueeze(
        0
    )  # (1, max_valid) -> expand to (B, max_valid) later

    # Compute the absolute indices to gather for each sample.
    student_indices = student_start.unsqueeze(1) + rel_idx  # shape: (B, max_valid)
    teacher_indices = teacher_start.unsqueeze(1) + rel_idx  # shape: (B, max_valid)

    # Clamp indices to prevent out-of-bounds errors (important!)
    student_indices = student_indices.clamp(min=0, max=seq_len - 1)
    teacher_indices = teacher_indices.clamp(min=0, max=seq_len - 1)

    # Gather probability spans using the computed indices
    # Indices need shape (B, max_valid, 1) to gather along dim 1
    student_span_probs = torch.gather(
        student_probs_full,
        1,
        student_indices.unsqueeze(-1).expand(B, max_valid, vocab_student),
    )
    teacher_span_probs = torch.gather(
        teacher_probs_full,
        1,
        teacher_indices.unsqueeze(-1).expand(B, max_valid, vocab_teacher),
    )
    # Shape of gathered tensors: (B, max_valid, V)

    # Sort probabilities along the vocabulary dimension.
    sorted_student = torch.topk(student_span_probs, k=k, dim=-1, largest=True).values
    sorted_teacher = torch.topk(teacher_span_probs, k=k, dim=-1, largest=True).values
    # sorted_student = torch.sort(student_span_probs, dim=-1, descending=True)[0]
    # sorted_teacher = torch.sort(teacher_span_probs, dim=-1, descending=True)[0]

    # Handle vocabulary size mismatch: pad the smaller tensor along the vocab dimension.
    # vocab_gap = vocab_student - vocab_teacher
    # if vocab_gap > 0:
    #    sorted_teacher = F.pad(sorted_teacher, (0, vocab_gap), value=0.0)
    # elif vocab_gap < 0:
    #    sorted_student = F.pad(sorted_student, (0, -vocab_gap), value=0.0)

    if k > vocab_teacher:
        sorted_teacher = F.pad(sorted_teacher, (0, k - vocab_teacher), value=0.0)
        print(
            f"Warning: k ({k}) is larger than vocab_teacher ({vocab_teacher}). Padding with zeros. k might be too large and could lead to OOM errors."
        )
    elif k > vocab_student:
        sorted_student = F.pad(sorted_student, (0, k - vocab_student), value=0.0)
        print(
            f"Warning: k ({k}) is larger than vocab_student ({vocab_student}). Padding with zeros. k might be too large and could lead to OOM errors."
        )

    # Compute element-wise L1 difference over the vocabulary dimension.
    l1_diff = torch.abs(sorted_student - sorted_teacher).sum(
        dim=-1
    )  # shape: (B, max_valid)

    # Create a mask that selects only valid positions per sample (up to valid_length).
    # Use the same rel_idx tensor (now shape (1, max_valid))
    valid_mask = (rel_idx < valid_length.unsqueeze(1)).float()  # shape: (B, max_valid)

    # Compute per-sample loss as the mean L1 difference over the valid positions.
    # Avoid division by zero for samples where valid_length is 0
    clamped_valid_length = torch.clamp(valid_length.float(), min=1.0)
    sample_loss = (l1_diff * valid_mask).sum(dim=1) / clamped_valid_length

    # Handle cases where original valid_length was 0 - loss should be 0
    sample_loss = torch.where(
        valid_length == 0, torch.zeros_like(sample_loss), sample_loss
    )

    # Average the loss over the batch.
    distillation_loss = sample_loss.mean()

    return distillation_loss


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    original_loss: Tensor,
    inputs: Dict[str, Any] = None,
    loss_type: str = "fkl",
    k: int = 100,
    alpha: float = 0.1,
    temperature: float = 2.0,
    **kwargs,
) -> Tensor:
    """
    Compute the distillation loss combining a specific divergence loss and original loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        original_loss: Original task loss
        inputs: Dictionary of input data
        loss_type: Type of distillation loss to use ("fkl", "kld", "uld", etc.)
        alpha: Weight for the distillation loss
        temperature: Temperature parameter for softening distributions
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Combined loss tensor
    """
    if loss_type == "kld" or loss_type == "fkl":
        kd_loss = forward_kl(student_logits, teacher_logits, temperature, inputs)
    elif loss_type == "uld":
        # For ULD, we may have different parameters
        kd_loss = uld_loss(
            student_logits, teacher_logits, temperature, k, inputs, **kwargs
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # Combine distillation loss with original task loss
    return alpha * kd_loss + (1 - alpha) * original_loss
