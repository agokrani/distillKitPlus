"""
Curriculum Temperature for Knowledge Distillation
https://arxiv.org/abs/2211.16231

This implementation is based on https://github.com/zhengli97/CTKD
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer
from .base import DistilLoss


# copied from https://github.com/zhengli97/CTKD/blob/master/models/temp_global.py#L21
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class CTKD(DistilLoss):
    """
    Implementation of CTKD for Language Modeling
    """

    def __init__(
        self,
        lambda_max: float = 1,
        lambda_min: float = 0,
        num_loops: int = 10,
        temp_start: float = 1,
        temp_end: float = 20,
        **kwargs,
    ):
        super().__init__()
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.num_loops = num_loops
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.global_temperature = nn.Parameter(torch.ones([], dtype=torch.float32))
        # In their experiments, Global-T is used as default
        self.grl = GradientReversal()

    def get_value(self, epoch):
        if epoch < 0:
            epoch = 0
        if epoch >= self.num_loops:
            epoch = self.num_loops
        value = (math.cos(epoch * math.pi / self.num_loops) + 1.0) * 0.5
        value = value * (self.lambda_max - self.lambda_min) + self.lambda_min
        return value

    def forward(
        self,
        trainer_instance: Trainer,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch,
        **kwargs,
    ) -> torch.Tensor:
        # Ensure global_temperature is on the correct device
        if self.global_temperature.device != logits.device:
            self.global_temperature = self.global_temperature.to(logits.device)

        # Use trainer state for epoch
        # Note: HF Trainer uses float for epoch, original CTKD assumes integer
        epoch = int(trainer_instance.state.epoch) + 1 # Use integer part

        lambda_ = self.get_value(epoch)
        temp = self.grl(self.global_temperature, lambda_)
        temp = self.temp_start + self.temp_end * torch.sigmoid(temp)

        # Flatten inputs for KLDiv
        mask_flat = mask.view(-1)
        logits_flat = logits.view(-1, logits.size(-1))
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

        # forward kl
        teacher_probs = F.softmax(teacher_logits_flat / temp, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(logits_flat / temp, dim=-1, dtype=torch.float32)

        # --- Match TAID's manual forward KL calculation --- 
        inf_mask = torch.isinf(logits_flat)
        # Calculate -sum(P * log Q)
        prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1) # Sum over vocab dim
        # Apply mask and calculate mean loss over non-masked tokens
        distil_loss = -torch.sum(x * mask_flat) / torch.sum(mask_flat) # Changed dim=0 to remove, sum over all
        # --- End TAID matching section ---

        distil_loss = distil_loss * temp * temp

        # TODO: Log temperature?
        return distil_loss 