import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Tuple, Dict
from .base import DistilLoss
from .fkl import forward_kl


class TAID(DistilLoss):
    def __init__(
        self,
        t_start: float = 0.4,
        t_end: float = 1.0,
        alpha: float = 5e-4,
        beta: float = 0.99,
        disable_adaptive: bool = False,
        **kwargs,
    ):
        super().__init__()
        # validation
        assert 0.0 <= t_start < 1.0
        assert 0.0 < t_end <= 1.0
        assert 0.0 <= alpha <= 1.0

        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.beta = beta
        self.disable_adaptive = disable_adaptive
        self.register_buffer(
            "t", torch.tensor(t_start, dtype=torch.float32)
        )
        self.register_buffer(
            "prev_loss", torch.tensor(float("inf"), dtype=torch.float32)
        )
        self.register_buffer(
            "momentum", torch.zeros([], dtype=torch.float32)
        )

    def update_t(
        self, loss: torch.Tensor, trainer_instance: Trainer
    ) -> torch.Tensor:
        # Ensure buffers are on the correct device (needed if model moves device)
        if self.t.device != loss.device:
            self.t = self.t.to(loss.device)
            self.prev_loss = self.prev_loss.to(loss.device)
            self.momentum = self.momentum.to(loss.device)

        global_step = trainer_instance.state.global_step
        # Use max_steps for num_train_steps if available, otherwise estimate (might be less accurate)
        num_train_steps = trainer_instance.state.max_steps if trainer_instance.state.max_steps > 0 else 1e6 # Avoid division by zero

        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return torch.tensor(0.0, device=loss.device) # Return zero delta on first step
        # Calculate relative change rate
        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        # Update momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        # Calculate adaptive delta
        adaptive_delta = torch.sigmoid(self.momentum)
        # Update t (ensure monotonic increase)
        progress = min(1.0, global_step / num_train_steps) # Ensure progress doesn't exceed 1
        t_target = self.t_start + (self.t_end - self.t_start) * progress
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        t = (
            min(self.t_end, max(t_target, self.t + delta_t))
            if not self.disable_adaptive
            else t_target
        )
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.t.device, dtype=self.t.dtype)
        self.t = t
        self.prev_loss = loss
        return delta_t

    def compute_loss(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ):
        # Ensure t is on the correct device
        if self.t.device != logits.device:
            self.t = self.t.to(logits.device)

        # Apply temperature to p_t calculation
        p_t_logits = (1 - self.t) * logits.detach() + self.t * teacher_logits
        p_t = F.softmax(p_t_logits / temperature, dim=-1, dtype=torch.float32)
        
        # Pass temperature to forward_kl
        distil_loss = forward_kl(
            logits=logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
            teacher_probs=p_t,
        )
        return distil_loss

    def forward(
        self,
        trainer_instance: Trainer,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch: Dict,
        temperature: float = 1.0,
        **kwargs,
    ) -> Dict:
        # compute kd loss, passing temperature
        loss = self.compute_loss(logits, teacher_logits, mask, temperature=temperature)

        # update t
        delta_t = self.update_t(
            loss.detach().clone(),
            trainer_instance=trainer_instance
        )

        loss_dict = {
            "distil_loss": loss,
            "taid_t": self.t.item() if self.t.numel() == 1 else self.t,
            "delta_t": delta_t,
        }
        return loss_dict
 