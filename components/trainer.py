from typing import Dict, Any, Optional, Tuple, Union
import torch
from trl import SFTTrainer
from transformers import PreTrainedModel, TrainerState
from torch import Tensor
import torch.nn.functional as F

from components.loss import compute_distillation_loss


class LogitsTrainer(SFTTrainer):
    """
    Custom trainer for knowledge distillation with logits.
    Extends SFTTrainer to support both direct and pre-computed logits distillation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize trainer with distillation parameters from config"""
        self.temperature = kwargs.pop("temperature", 2.0)
        self.alpha = kwargs.pop("alpha", 0.1)
        self.loss_type = kwargs.pop("loss_type", "fkl")
        # only used for uld loss
        self.k = kwargs.pop("k", 100)
        # Store arbitrary kwargs for specific loss functions
        self.distil_kwargs = kwargs.pop("distil_kwargs", {})
        self.student_temperature = kwargs.pop("student_temperature", self.temperature)
        self.teacher_temperature = kwargs.pop("teacher_temperature", self.temperature)
        self.skip_eos = kwargs.pop("skip_eos", False)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        """
        Compute the distillation loss combining KL divergence and original loss.

        Args:
            model: The student model
            inputs: Dictionary containing input tensors
            return_outputs: Whether to return model outputs along with loss

        Returns:
            Loss tensor or tuple of (loss tensor, model outputs)
        """
        # Move inputs to model device
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        # Store the current inputs for accessing labels in _compute_distillation_loss
        # Create a shallow copy to avoid modifying the original
        self.current_inputs = {k: v for k, v in inputs.items()}

        # Extract teacher logits if present
        teacher_logits = inputs.pop("logits") if "logits" in inputs else None

        # Get actual model (unwrap from DataParallel if needed)
        student_model = model.module if hasattr(model, "module") else model

        # Get student outputs
        # Ensure labels are passed if they exist in inputs, required for original loss calculation
        student_outputs = student_model(**inputs)

        # Get teacher logits either from inputs or by computing them
        if teacher_logits is None:
            teacher_logits = self._compute_teacher_logits(inputs)

        # Handle sequence length mismatch
        teacher_logits = self._align_sequence_length(
            teacher_logits, student_outputs.logits
        )

        # Compute combined loss
        custom_loss = self._compute_distillation_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_logits,
            original_loss=student_outputs.loss, # Use the loss returned by the model
            inputs=inputs, # Pass original inputs (might contain labels, mask etc.)
            distil_kwargs=self.distil_kwargs # Pass the specific loss kwargs
        )

        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def _compute_teacher_logits(self, inputs: Dict[str, Any]) -> Tensor:
        """
        Compute logits using teacher model when pre-computed logits aren't available.

        Args:
            inputs: Input tensors for the teacher model

        Returns:
            Teacher logits tensor
        """
        assert hasattr(self, "teacher_model"), (
            "Teacher model required for distillation without precomputed logits."
        )

        # Ensure teacher is on correct device
        self.teacher_model = self.teacher_model.to(self.model.device)
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )
        # Check if we have teacher-specific inputs
        teacher_inputs = {}
        has_teacher_inputs = False

        # Create teacher-specific inputs dictionary if any teacher inputs exist
        if "teacher_input_ids" in self.current_inputs:
            has_teacher_inputs = True
            teacher_inputs["input_ids"] = self.current_inputs["teacher_input_ids"]

        if "teacher_attention_mask" in self.current_inputs:
            has_teacher_inputs = True
            teacher_inputs["attention_mask"] = self.current_inputs[
                "teacher_attention_mask"
            ]

        if "teacher_labels" in self.current_inputs:
            has_teacher_inputs = True
            teacher_inputs["labels"] = self.current_inputs["teacher_labels"]

        # Use teacher-specific inputs if available, otherwise use regular inputs
        # Important: Don't pass labels to teacher model if it's only for inference
        final_input_keys = list(final_inputs.keys())
        for key in final_input_keys:
            if key == "labels":
                 del final_inputs[key]

        final_inputs = teacher_inputs if has_teacher_inputs else {k:v for k,v in inputs.items() if k != 'labels'} # Don't pass labels

        with torch.no_grad():
            teacher_outputs = teacher_model(**final_inputs)

        return teacher_outputs.logits

    def _align_sequence_length(
        self, teacher_logits: Tensor, student_logits: Tensor
    ) -> Tensor:
        """
        Align teacher and student sequence lengths.

        Args:
            teacher_logits: Logits from teacher model
            student_logits: Logits from student model

        Returns:
            Aligned teacher logits
        """
        teacher_len = teacher_logits.size(1)
        student_len = student_logits.size(1)

        if teacher_len > student_len:
            # Truncate teacher logits if longer than student's
            return teacher_logits[:, :student_len, :]
        elif teacher_len < student_len:
            # Pad teacher logits if shorter than student's
            padding_needed = student_len - teacher_len
            # Pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            # We pad dim 1 (sequence length) at the end: (0, 0 for dim 2, 0, padding_needed for dim 1, 0, 0 for dim 0)
            padding_tuple = (0, 0, 0, padding_needed, 0, 0)
            # Use -inf for padding logits typically, but 0 might be okay if softmax handles it. Check loss impl.
            padded_teacher_logits = F.pad(
                teacher_logits, padding_tuple, mode='constant', value=0
            )
            return padded_teacher_logits
        return teacher_logits

    def _compute_distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        original_loss: Tensor,
        inputs: Dict[str, Any],
        distil_kwargs: Dict[str, Any], # Accept distil_kwargs dict
    ) -> Tensor:
        """
        Compute the distillation loss combining KL divergence and original loss.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            original_loss: Original task loss
            inputs: Dictionary of input data (contains mask, labels etc)
            distil_kwargs: Dictionary of specific arguments for the loss constructor

        Returns:
            Combined loss tensor
        """

        return compute_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            original_loss=original_loss,
            trainer_instance=self, # Pass the trainer instance
            inputs=inputs,
            loss_type=self.loss_type,
            k=self.k,
            alpha=self.alpha,
            temperature=self.temperature,
            distil_loss_weight=distil_kwargs.get("distil_loss_weight", 1.0), # Get weight from kwargs or default
            **distil_kwargs, # Pass the rest of the specific kwargs
        )
