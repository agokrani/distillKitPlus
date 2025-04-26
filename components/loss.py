import logging
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer # Use Trainer for type hinting

# Import new distillation loss classes
from .distil_losses import (
    ForwardKL,
    ReverseKL,
    TVD,
    JS,
    AdaptiveKL,
    SkewForwardKL,
    SkewReverseKL,
    CTKD,
    DKD,
    TAID,
    ULD, # Added ULD
    DistilLoss,
)


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    original_loss: Tensor,
    trainer_instance: Optional[Trainer] = None, # Changed from lightning_module
    inputs: Dict[str, Any] = None,
    loss_type: str = "fkl",
    k: int = 100, # Parameter specific to ULD, passed via kwargs now
    alpha: float = 0.1,
    temperature: float = 2.0, # Common param, pass via kwargs if needed by loss
    distil_loss_weight: float = 1.0, # Added weight specifically for the kd_loss component
    # Allow passing constructor args for specific loss classes via kwargs
    **kwargs,
) -> Tensor:
    """
    Compute the distillation loss combining a specific divergence loss and original loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        original_loss: Original task loss (e.g., CrossEntropy)
        trainer_instance: The Hugging Face Trainer instance (needed for some losses like CTKD, TAID, DKD).
        inputs: Dictionary of input data, potentially containing 'mask', 'labels', 'teacher_labels', etc.
        loss_type: Type of distillation loss to use. Options include:
                   "fkl", "rkl", "tvd", "js", "adaptive_kl",
                   "skew_fkl", "skew_rkl", "ctkd", "dkd", "taid", "uld".
                   Defaults to "fkl".
        k: Parameter for ULD loss (passed via kwargs).
        alpha: Weight for combining original_loss and kd_loss (combined = (1-alpha)*orig + alpha*distil).
        temperature: Temperature parameter (passed via kwargs if needed by the specific loss).
        distil_loss_weight: The weight applied *only* to the distillation component (kd_loss).
        **kwargs: Additional arguments passed to the constructor of the chosen DistilLoss class
                  (e.g., t_start, beta, head_threshold, and now k, temperature if needed).

    Returns:
        Combined loss tensor: (1 - alpha) * original_loss + alpha * distil_loss_weight * kd_loss_component
    """
    kd_loss_component = 0.0
    loss_instance = None

    # Extract common inputs expected by TAID-style losses
    mask = inputs.get("mask", inputs.get("attention_mask", None)) # Try both keys
    batch = inputs # Assuming the whole inputs dict can serve as the 'batch' argument

    if mask is None:
         logging.warning("Mask/attention_mask not found in inputs, some distillation losses might fail or behave unexpectedly.")
         # Consider creating a default mask if critical and appropriate for the use case
         # mask = torch.ones_like(student_logits[..., 0], dtype=torch.bool, device=student_logits.device)

    # Prepare constructor args, ensuring k and temp are passed if present
    loss_kwargs = kwargs.copy()
    if 'k' not in loss_kwargs:
        loss_kwargs['k'] = k
    if 'temperature' not in loss_kwargs:
        loss_kwargs['temperature'] = temperature

    # Instantiate and compute the distillation component based on loss_type
    if loss_type == "fkl":
        loss_instance = ForwardKL(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "rkl":
        loss_instance = ReverseKL(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "tvd":
        loss_instance = TVD(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "js":
        loss_instance = JS(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "adaptive_kl":
        loss_instance = AdaptiveKL(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "skew_fkl":
        loss_instance = SkewForwardKL(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "skew_rkl":
        loss_instance = SkewReverseKL(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "ctkd":
        if trainer_instance is None:
            raise ValueError("CTKD loss requires the trainer_instance to be passed.")
        loss_instance = CTKD(**loss_kwargs)
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "dkd":
        if trainer_instance is None:
            raise ValueError("DKD loss requires the trainer_instance to be passed.")
        loss_instance = DKD(**loss_kwargs) # DKD constructor takes temperature
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    elif loss_type == "taid":
        if trainer_instance is None:
            raise ValueError("TAID loss requires the trainer_instance to be passed.")
        loss_instance = TAID(**loss_kwargs)
        loss_dict = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
        # TAID returns a dict, extract the main loss
        kd_loss_component = loss_dict.get("distil_loss", 0.0)
        # TODO: Log other values from loss_dict if needed (e.g., tiki_t)
    elif loss_type == "uld":
        # Use the new ULD class
        loss_instance = ULD(**loss_kwargs) # ULD constructor takes k and temperature
        kd_loss_component = loss_instance(trainer_instance=trainer_instance, logits=student_logits, teacher_logits=teacher_logits, mask=mask, batch=batch, **loss_kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


    # Combine original loss and distillation loss component
    # Using alpha to interpolate between original and distillation loss
    # Using distil_loss_weight to scale only the distillation component
    combined_loss = (1 - alpha) * original_loss + alpha * distil_loss_weight * kd_loss_component

    # TODO: Log kd_loss_component, original_loss, combined_loss?

    return combined_loss
