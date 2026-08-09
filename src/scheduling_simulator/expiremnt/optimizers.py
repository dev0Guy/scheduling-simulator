"""Muon optimizer: orthogonalized momentum for matrix parameters.

For 2D weight matrices (Linear, attention projections), Muon applies
Newton-Schulz orthogonalization to the momentum buffer before the
update step. This equalizes the gradient signal across all directions,
which is particularly effective for attention layers where gradient
magnitudes vary across heads. Non-matrix parameters (biases, LayerNorm)
use standard AdamW.

Reference: Keller Jordan, "Muon: An optimizer for neural network
training" (2024). Newton-Schulz coefficients from the Muon repository.
"""

from __future__ import annotations

import torch as th
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: th.Tensor, steps: int = 5) -> th.Tensor:
    """Orthogonalize matrix G using 5-step Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7752, 2.0315)
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: Momentum + Orthogonalization.

    Matrix parameters (ndim == 2, both dims > 1) get orthogonalized
    momentum updates. All other parameters fall back to AdamW.

    :param lr: Learning rate for Muon parameters (default 0.02).
    :param momentum: Momentum factor (default 0.95).
    :param nesterov: Use Nesterov momentum (default True).
    :param ns_steps: Newton-Schulz iteration steps (default 5).
    :param adamw_lr: Learning rate for non-matrix AdamW parameters.
    :param adamw_betas: AdamW beta1, beta2.
    :param adamw_eps: AdamW epsilon.
    :param weight_decay: Decoupled weight decay applied to all params.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas=(0.9, 0.99),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adamw_lr=adamw_lr,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @th.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with th.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']
            adamw_lr = group['adamw_lr']
            beta1, beta2 = group['adamw_betas']
            adamw_eps = group['adamw_eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                is_matrix = p.ndim == 2 and min(p.shape) > 1

                if is_matrix:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = th.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    update = _zeropower_via_newtonschulz5(
                        buf + g if nesterov else buf, steps=ns_steps
                    )
                    update *= max(1, p.size(0) / p.size(1)) ** 0.5
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = th.zeros_like(g)
                        state['exp_avg_sq'] = th.zeros_like(g)
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias_c1 = 1 - beta1 ** state['step']
                    bias_c2 = 1 - beta2 ** state['step']
                    step_lr = adamw_lr
                    denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(adamw_eps)
                    p.mul_(1 - adamw_lr * wd)
                    p.addcdiv_(exp_avg / bias_c1, denom, value=-step_lr)

        return loss
