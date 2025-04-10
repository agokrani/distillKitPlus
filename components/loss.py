from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor

def forward_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    """
    Compute the forward KL divergence loss for knowledge distillation.
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Temperature parameter for softening distributions
        
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
            ..., 
            :student_logits_scaled.size(-1)
        ]
    elif teacher_logits_scaled.size(-1) < student_logits_scaled.size(-1):
        # Pad teacher logits with -inf if smaller than student's
        padding_size = (
            student_logits_scaled.size(-1) - teacher_logits_scaled.size(-1)
        )
        padding = torch.full(
            (
                teacher_logits_scaled.size(0),
                teacher_logits_scaled.size(1),
                padding_size
            ),
            float('-inf'),
            device=teacher_logits_scaled.device
        )
        teacher_logits_scaled = torch.cat(
            [teacher_logits_scaled, padding],
            dim=-1
        )

    # Compute KL divergence loss
    kd_loss = F.kl_div(
        F.log_softmax(student_logits_scaled, dim=-1),
        F.softmax(teacher_logits_scaled, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2) / student_logits.size(1)
    
    return kd_loss

def uld_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
    **kwargs
) -> Tensor:
    """
    Compute the Universal LogitDistillation (ULD) loss for knowledge distillation.
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Temperature parameter for softening distributions
        **kwargs: Additional arguments including 'labels' for identifying answer positions
        
    Returns:
        ULD loss tensor
    """
    # Get labels to find answer positions
    labels = kwargs.get('labels', None)
    if labels is None:
        raise ValueError("ULD loss requires 'labels' in kwargs to identify answer positions")
    
    # Create attention mask from labels (1 where labels != -100, 0 elsewhere)
    attention_mask = (labels != -100).float()
    
    # Find the start indices for each sequence in the batch
    # Get indices of first non-masked position in each sequence
    batch_size = student_logits.size(0)
    seq_length = student_logits.size(1)
    
    # Get start indices efficiently using nonzero and groupby
    # nonzero_indices = attention_mask.nonzero(as_tuple=True)
    # start_indices = torch.zeros(batch_size, dtype=torch.long, device=student_logits.device)
    
    # # Find the first nonzero index for each batch item
    # for batch_idx, pos_idx in zip(nonzero_indices[0], nonzero_indices[1]):
    #     if attention_mask[batch_idx].sum() > 0 and start_indices[batch_idx] == 0:
    #         start_indices[batch_idx] = pos_idx
    
    # Apply softmax with temperature to student and teacher logits
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Extract answer spans using gathered indices and attention mask
    # Use the attention mask to zero out non-answer positions
    masked_student_probs = student_probs * attention_mask.unsqueeze(-1)
    masked_teacher_probs = teacher_probs * attention_mask.unsqueeze(-1)
    
    # Sort probabilities in descending order
    sorted_student_probs = torch.sort(masked_student_probs, dim=-1, descending=True)[0]
    sorted_teacher_probs = torch.sort(masked_teacher_probs, dim=-1, descending=True)[0]
    
    # Handle vocabulary size mismatch
    vocab_size_student = sorted_student_probs.size(-1)
    vocab_size_teacher = sorted_teacher_probs.size(-1)
    vocab_size_gap = vocab_size_student - vocab_size_teacher
    
    if vocab_size_gap > 0:
        # Teacher vocab is smaller, pad teacher
        sorted_teacher_probs = F.pad(sorted_teacher_probs, (0, vocab_size_gap), value=0.0)
    elif vocab_size_gap < 0:
        # Student vocab is smaller, pad student
        sorted_student_probs = F.pad(sorted_student_probs, (0, -vocab_size_gap), value=0.0)
    
    # Compute the ULD loss (L1 distance between sorted probability distributions)
    l1_loss = torch.abs(sorted_student_probs - sorted_teacher_probs).sum(-1)
    
    # Normalize by attention mask (only consider loss at answer positions)
    # Avoid division by zero
    attention_sum = attention_mask.sum(-1)
    attention_sum = torch.clamp(attention_sum, min=1.0)  # Avoid division by zero
    
    # For each sequence in batch, mean over valid positions only
    l1_loss = (l1_loss * attention_mask).sum(-1) / attention_sum
    
    # Mean over batch
    distillation_loss = l1_loss.mean()
    print(f"distillation_loss: {distillation_loss}")
    return distillation_loss

def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    original_loss: Tensor,
    loss_type: str = "fkl",
    alpha: float = 0.1,
    temperature: float = 2.0,
    **kwargs
) -> Tensor:
    """
    Compute the distillation loss combining a specific divergence loss and original loss.
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        original_loss: Original task loss
        loss_type: Type of distillation loss to use ("fkl", "kld", "uld", etc.)
        alpha: Weight for the distillation loss
        temperature: Temperature parameter for softening distributions
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Combined loss tensor
    """
    if loss_type == "kld" or loss_type == "fkl":
        kd_loss = forward_kl(student_logits, teacher_logits, temperature)
    elif loss_type == "uld":
        # For ULD, we may have different parameters
        kd_loss = uld_loss(
            student_logits, 
            teacher_logits, 
            temperature,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
        
    # Combine distillation loss with original task loss
    return alpha * kd_loss + (1 - alpha) * original_loss