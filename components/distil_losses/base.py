from typing import Union, Dict
import torch
from torch import nn
from lightning.pytorch import LightningModule


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module: LightningModule,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        batch: Dict[str, Dict[str, torch.Tensor]],
        **kwargs,
    ) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError 