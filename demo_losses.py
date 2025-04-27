import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerState
from components.loss import compute_distillation_loss
from types import SimpleNamespace
import logging
import random # Import random for seeding
import numpy as np # Import numpy for seeding

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Seeding ---
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
logger.info(f"Set seed to {SEED}")

def run_demo():
    # --- Configuration ---
    model_name = "distilgpt2" # Small causal LM for faster demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 16

    # List of loss types to test - aligned with TAID script (excluding ULD, adding SFT handling placeholder, using sfkl/srkl)
    loss_types = [
        "sft", # Simple Fine-tuning (no distillation) - Added
        "fkl", "rkl", "tvd", "js", "adaptive_kl",
        "sfkl", "srkl", "ctkd", "dkd", "taid" # Changed skew names, removed uld
    ]

    logger.info(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # Use the same model as teacher for simplicity in this demo
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    teacher_model.eval()

    vocab_size = student_model.config.vocab_size

    # --- Create Dummy Data ---
    logger.info("Creating dummy data...")
    # Ensure input_ids are within vocab range
    # Note: Using student_model.config.vocab_size ensures consistency if tokenizer size differs
    dummy_input_ids = torch.randint(0, student_model.config.vocab_size, (batch_size, seq_len), device=device)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    # Create labels, randomly mask some tokens (KEEPING THIS METHOD FOR COMPARISON)
    dummy_labels = dummy_input_ids.clone()
    mask_indices = torch.rand(dummy_labels.shape, device=device) < 0.15 # Mask ~15%
    dummy_labels[mask_indices] = -100
    # For ULD, teacher_labels are needed - REMOVED as ULD is removed
    # dummy_teacher_labels = dummy_labels.clone() # Use same labels for demo

    inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
        "labels": dummy_labels,
        # "teacher_labels": dummy_teacher_labels, # No longer needed
        # Add 'mask' key as well, pointing to attention_mask, as loss.py checks for both
        "mask": dummy_attention_mask
    }

    # --- Create Mock Trainer State ---
    # Needed for TAID, DKD, CTKD
    # Ensure attributes match TAID script's mock_trainer if needed
    mock_trainer_state = TrainerState(
        epoch=1.0, # Consistent epoch (TAID uses 1)
        global_step=100, # Consistent global step
        max_steps=1000 # Consistent max steps (TAID uses estimated_stepping_batches=1000)
        # Add other state attributes if needed by future losses
    )
    # Mock trainer instance needs a .state attribute
    mock_trainer_instance = SimpleNamespace(state=mock_trainer_state)

    # --- Test Each Loss Function ---
    logger.info("Testing loss functions...")
    successful_losses = []
    failed_losses = {}

    for loss_type in loss_types:
        logger.info(f"-- Testing loss_type: {loss_type} --")
        try:
            # 1. Get student outputs (including original loss)
            student_outputs = student_model(input_ids=inputs["input_ids"],
                                          attention_mask=inputs["attention_mask"],
                                          labels=inputs["labels"])
            student_logits = student_outputs.logits
            original_loss = student_outputs.loss
            if original_loss is None:
                 # This can happen if all labels are -100, add a small check
                 if (inputs["labels"] == -100).all():
                     logger.warning("All labels are masked (-100). Original loss calculation skipped by model. Using dummy 0.0")
                     original_loss = torch.tensor(0.0, device=device, requires_grad=True) # Still need grad if alpha != 1
                 else:
                    logger.error("Original loss is None despite valid labels. Cannot proceed.")
                    raise ValueError("Original loss is None from the model.")
            elif torch.isnan(original_loss).any() or torch.isinf(original_loss).any():
                 logger.error(f"Original loss is NaN or Inf: {original_loss.item()}. Cannot proceed.")
                 raise ValueError("Original loss is NaN or Inf.")

            # Handle SFT separately (just uses original loss)
            if loss_type == "sft":
                logger.info(f"[sft] SUCCESS: Using Original LM Loss = {original_loss.item():.4f}")
                successful_losses.append(loss_type)
                # Check grad requirement for SFT
                if not original_loss.requires_grad:
                    logger.warning("Original loss for SFT does not require grad.")
                continue # Skip KD calculation for SFT

            # 2. Get teacher logits
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"])
                teacher_logits = teacher_outputs.logits

            # 3. Align sequence lengths (just in case, though same model here)
            teacher_len = teacher_logits.size(1)
            student_len = student_logits.size(1)
            if teacher_len > student_len:
                teacher_logits = teacher_logits[:, :student_len, :]
            elif teacher_len < student_len:
                 padding_needed = student_len - teacher_len
                 padding_tuple = (0, 0, 0, padding_needed, 0, 0)
                 teacher_logits = F.pad(teacher_logits, padding_tuple, mode='constant', value=0)

            # 4. Define loss-specific kwargs (add more here if needed for defaults)
            # Ensure these match the defaults used in TAID's get_loss_fn or explicitly set in taid_style_demo's mock_args
            distil_kwargs = {
                 'js_beta': 0.5, # Default from TAID
                 'adaptive_kl_threshold': 0.5, # Default from TAID
                 'skew_beta': 0.5, # Default from TAID (for sfkl/srkl)
                 # TAID specific args (provide defaults even if not used by all losses in compute_distillation_loss)
                 'taid_t_start': 0.3,
                 'taid_t_end': 0.7,
                 'taid_alpha': 0.9,
                 'taid_beta': 0.1,
                 'taid_disable_adaptive': False,
            }

            # Common forward args (like temperature for KL-based losses) might be passed directly
            # Or handled within compute_distillation_loss based on **kwargs
            common_forward_kwargs = {
                'temperature': 2.0, # Consistent temperature
            }

            # Combine kwargs for the main call
            combined_kwargs = {**common_forward_kwargs, **distil_kwargs}

            # 5. Call the central loss function
            combined_loss = compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                original_loss=original_loss,
                # Pass the mock trainer instance, which ULD/TAID/DKD/CTKD might need
                trainer_instance=mock_trainer_instance,
                inputs=inputs,
                loss_type=loss_type,
                alpha=0.5, # Weight for combining original and KD loss (Consistent with distil_ratio=0.5)
                distil_loss_weight=1.0, # Weight for the KD component itself (Keep at 1.0 for direct comparison)
                # Pass specific constructor/forward args via kwargs
                **combined_kwargs
            )

            # 6. Check if loss requires grad and is valid
            if torch.isnan(combined_loss).any() or torch.isinf(combined_loss).any():
                 raise RuntimeError(f"Combined loss is NaN or Inf: {combined_loss.item()}")
            if not combined_loss.requires_grad:
                 # Original loss requires grad, and student logits were used, so combined should require grad unless alpha=0
                 if alpha > 0:
                     logger.warning(f"Combined loss for {loss_type} does not require grad unexpectedly.")
                 # raise RuntimeError("Combined loss does not require grad.")

            # Log success *before* appending to avoid issues if append fails (unlikely but safe)
            logger.info(f"[{loss_type}] SUCCESS: Combined Loss = {combined_loss.item():.4f}")

            successful_losses.append(loss_type)

        except Exception as e:
            logger.error(f"[{loss_type}] FAILED: {e}", exc_info=False) # Set exc_info=True for full traceback
            failed_losses[loss_type] = str(e)

    # --- Summary ---
    logger.info("\n--- Demo Summary ---")
    logger.info(f"Successfully computed losses for: {successful_losses}")
    if failed_losses:
        logger.warning(f"Failed to compute losses for: {list(failed_losses.keys())}")
        for loss, error in failed_losses.items():
            logger.warning(f"  - {loss}: {error}")
    logger.info("--------------------\n")

if __name__ == "__main__":
    run_demo() 