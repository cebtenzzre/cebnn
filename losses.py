# From https://github.com/dougbrion/pytorch-classification-uncertainty/blob/06fb2f6/losses.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from typing import Callable
    from torch import Tensor

_DEF_ANNEAL_STEP = 10


def kl_divergence(alpha: Tensor) -> Tensor:
    beta = torch.ones((1, alpha.shape[1]), dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = (torch.lgamma(S_alpha)
           - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True))
    lnB_uni = (torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
               - torch.lgamma(S_beta))

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = (torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True)
          + lnB + lnB_uni)
    return kl


def loglikelihood_loss(alpha: Tensor, target: Tensor) -> Tensor:
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (target - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def torch_loss(func: Callable[[Tensor], Tensor], alpha: Tensor, target: Tensor) -> Tensor:
    S = torch.sum(alpha, dim=1, keepdim=True)
    return torch.sum(target * (func(S) - func(alpha)), dim=1, keepdim=True)


class EDL_Loss(nn.Module):
    def __init__(self, func: Callable[[Tensor, Tensor], Tensor], *, annealing_step: int = _DEF_ANNEAL_STEP) -> None:
        super().__init__()
        self.func = func
        self.annealing_step = annealing_step

    def forward(self, alpha: Tensor, target: Tensor, epoch: int) -> Tensor:
        target = F.one_hot(target.to(torch.int64), num_classes=2).to(torch.float32)
        assert alpha.device == target.device
        assert alpha.shape == target.shape
        assert target.ndim == 3
        # Combine the sample and label dimensions; 2 samples of 3 labels each is effectively 6 samples
        alpha = alpha.view(-1, alpha.shape[-1])
        target = target.view(-1, target.shape[-1])
        A = self.func(alpha, target)
        annealing_coef = torch.min(torch.tensor(1.0), torch.tensor(epoch / self.annealing_step))
        kl_alpha = (alpha - 1) * (1 - target) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha)
        return torch.mean(A + kl_div)


class EDL_MSE_Loss(EDL_Loss):
    def __init__(self, *, annealing_step: int = _DEF_ANNEAL_STEP) -> None:
        super().__init__(loglikelihood_loss, annealing_step=annealing_step)


class EDL_Log_Loss(EDL_Loss):
    def __init__(self, *, annealing_step: int = _DEF_ANNEAL_STEP) -> None:
        super().__init__(partial(torch_loss, torch.log), annealing_step=annealing_step)


class EDL_Digamma_Loss(EDL_Loss):
    def __init__(self, *, annealing_step: int = _DEF_ANNEAL_STEP) -> None:
        super().__init__(partial(torch_loss, torch.digamma), annealing_step=annealing_step)
