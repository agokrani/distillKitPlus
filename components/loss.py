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
    alpha: float = 0.1,
    distil_loss_weight: float = 1.0,
    # Pass specific constructor/forward args via kwargs
    **kwargs,
) -> Tensor:
    """
    Compute the distillation loss combining a specific divergence loss and original loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        original_loss: Original task loss (e.g., CrossEntropy)
        trainer_instance: The Hugging Face Trainer instance (needed for some losses like CTKD, TAID, DKD, ULD).
        inputs: Dictionary of input data, potentially containing 'mask', 'labels', 'teacher_labels', etc.
        loss_type: Type of distillation loss to use. Options include:
                   "fkl", "rkl", "tvd", "js", "adaptive_kl",
                   "skew_fkl", "skew_rkl", "ctkd", "dkd", "taid", "uld".
                   Defaults to "fkl".
        alpha: Weight for combining original_loss and kd_loss (combined = (1-alpha)*orig + alpha*distil).
        distil_loss_weight: The weight applied *only* to the distillation component (kd_loss).
        **kwargs: Additional arguments passed to the constructor OR forward method of the
                  chosen DistilLoss class (e.g., temperature, k, t_start, beta).
                  The function will try to pass relevant args to the correct place.

    Returns:
        Combined loss tensor: (1 - alpha) * original_loss + alpha * distil_loss_weight * kd_loss_component
    """
    kd_loss_component = 0.0
    loss_instance = None

    # Extract common inputs expected by forward methods
    mask = inputs.get("mask", inputs.get("attention_mask", None))
    batch = inputs # Pass the whole inputs dict as 'batch'

    if mask is None:
         logging.warning("Mask/attention_mask not found in inputs, some distillation losses might fail or behave unexpectedly.")
         # mask = torch.ones_like(student_logits[..., 0], dtype=torch.bool, device=student_logits.device) # Default if needed

    # Separate kwargs for constructor and forward based on common patterns
    # Constructor args often define behavior (e.g., thresholds, modes)
    # Forward args often relate to the specific calculation (e.g., temperature)
    constructor_kwargs = {}
    forward_kwargs = {}
    common_forward_args = {'temperature'} # Use a set for faster lookup
    known_constructor_args = { # Args specific to certain loss types' __init__
        # Add args based on kwargs passed from demo_losses.py
        'js': {'teacher_weight'}, # JS takes teacher_weight
        'adaptive_kl': {'head_threshold'}, # AdaptiveKL takes head_threshold
        'skew_fkl': {'skew_beta'}, # Use skew_beta passed from demo
        'skew_rkl': {'skew_beta'}, # Use skew_beta passed from demo
        'ctkd': {'lambda_max', 'lambda_min', 'num_loops', 'temp_start', 'temp_end'}, # Correct CTKD constructor args
        'dkd': {'temperature', 'dkd_alpha', 'dkd_beta'}, # DKD takes temp in constructor
        'taid': {'taid_t_start', 'taid_t_end', 'taid_alpha', 'taid_beta', 'taid_disable_adaptive'}, # Use explicit names from demo
        'uld': ['k', 'temperature'], # ULD takes k and temp in constructor
    }

    for key, value in kwargs.items():
        # Check if it's a known constructor arg for the current loss type
        is_constructor_arg = False
        if loss_type in known_constructor_args and key in known_constructor_args[loss_type]:
            is_constructor_arg = True
        # Special handling: DKD and ULD can take temperature in constructor, others take it in forward
        elif key == 'temperature' and loss_type in ['dkd', 'uld']:
            is_constructor_arg = True

        if is_constructor_arg:
            constructor_kwargs[key] = value
        # Check if it's a common forward arg (and NOT a constructor arg for this loss)
        elif key in common_forward_args:
            forward_kwargs[key] = value
        else:
            # Specifically ignore the incorrectly passed kwargs from demo_losses.py for JS/AdaptiveKL
            if not ((loss_type == 'js' and key == 'js_beta') or 
                    (loss_type == 'adaptive_kl' and key == 'adaptive_kl_threshold')):
                logging.debug(f"Ignoring unknown kwarg '{key}' for loss type '{loss_type}'")


    # Instantiate loss based on type, passing ONLY constructor args
    loss_classes = {
        "fkl": ForwardKL, "rkl": ReverseKL, "tvd": TVD, "js": JS,
        "adaptive_kl": AdaptiveKL, "skew_fkl": SkewForwardKL, "skew_rkl": SkewReverseKL,
        "sfkl": SkewForwardKL, "srkl": SkewReverseKL, # Add sfkl/srkl mappings
        "ctkd": CTKD, "dkd": DKD, "taid": TAID, "uld": ULD
    }

    if loss_type not in loss_classes:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Validate required trainer_instance for specific losses (including sfkl/srkl mapped names)
    mapped_loss_type = loss_type # Store original name for error message if needed
    if loss_type == "sfkl": loss_type_for_check = "skew_fkl"
    elif loss_type == "srkl": loss_type_for_check = "skew_rkl"
    else: loss_type_for_check = loss_type

    if loss_type_for_check in ["ctkd", "dkd", "taid", "uld"] and trainer_instance is None:
         raise ValueError(f"{mapped_loss_type.upper()} loss requires the trainer_instance to be passed.")

    loss_instance = loss_classes[loss_type](**constructor_kwargs)

    # Prepare arguments for the forward call
    # All forward methods seem to follow the pattern:
    # (self, trainer_instance, logits, teacher_logits, mask, batch, **kwargs)
    # based on base.py and the previous structure. ULD errored on missing 'lightning_module'
    # let's assume 'trainer_instance' maps to 'lightning_module' if needed.
    forward_call_args = {
        "trainer_instance": trainer_instance,
        "logits": student_logits,
        "teacher_logits": teacher_logits,
        "mask": mask,
        "batch": batch,
        **forward_kwargs # Pass args like temperature here
    }

    # Call the forward method
    output = loss_instance(**forward_call_args)

    # Handle output (some return dict, some return tensor)
    if isinstance(output, dict):
        kd_loss_component = output.get("distil_loss", 0.0)
        # TODO: Log other values from loss_dict if needed
    elif isinstance(output, Tensor):
        kd_loss_component = output
    else:
        logging.error(f"Unexpected output type from {loss_type} loss: {type(output)}")
        kd_loss_component = torch.tensor(0.0, device=student_logits.device)


    # Combine original loss and distillation loss component
    # Alpha is used here, outside the loss call
    combined_loss = (1 - alpha) * original_loss + alpha * distil_loss_weight * kd_loss_component

    # TODO: Log kd_loss_component, original_loss, combined_loss?

    return combined_loss
