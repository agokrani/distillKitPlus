from typing import Union, Dict
import torch
from torch import nn
from transformers import Trainer


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        trainer_instance: Trainer,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch: Dict[str, Dict[str, torch.Tensor]],
        **kwargs,
    ) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError 