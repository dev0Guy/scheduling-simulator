"""Permutation-invariant autoencoder that folds K machines into the view of ONE machine and back.

A machine is its free-capacity grid (n_resource x n_time, free / capacity in [0, 1]). The encoder is a
DeepSets network: a shared per-machine MLP, then mean- and max-pooling over the machines (so the result cannot
depend on their order), then a head that outputs one machine-shaped grid, the "view". That grid is the only
thing that passes through the bottleneck, and it has exactly the format of a real machine, so the 1-machine PPO
model can be given the view instead of a machine. The decoder maps the view back to K unordered machine
grids; because the machines have no order, the reconstruction loss is the Chamfer distance between the decoded
set and the real set (order-free). An auxiliary prior pulls the view towards the elementwise MAX of the free
grids ("some machine in the group has this much room"), which keeps the view meaningful for the fit test.
"""
import torch as th
from torch import nn


class SetMachineAE(nn.Module):
    def __init__(self, k_machines: int, n_resource: int = 3, n_time: int = 10, hidden: int = 128, at_least_max: bool = False):
        super().__init__()
        # at_least_max: the view is the elementwise max of the free grids PLUS a learned non-negative surplus, so it never
        # claims less room than the roomiest machine of the group (a job that fits some machine always fits the view)
        self.k, self.shape, self.at_least_max = k_machines, (n_resource, n_time), at_least_max
        d = n_resource * n_time
        self.phi = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.dec = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, k_machines * d))

    def encode(self, free: th.Tensor) -> th.Tensor:
        """free: (B, K, R, T) in [0, 1] -> view (B, R, T) in [0, 1]. Any K and any machine order give the same kind of view."""
        h = self.phi(free.flatten(2))                              # (B, K, hidden)
        pooled = th.cat([h.mean(1), h.max(1).values], dim=-1)
        raw = self.rho(pooled).reshape(-1, *self.shape)
        if self.at_least_max:
            mx = free.max(1).values
            return mx + (1.0 - mx) * th.sigmoid(raw - 4.0)         # starts at the plain max, the surplus is learned
        return th.sigmoid(raw)

    def decode(self, view: th.Tensor) -> th.Tensor:
        """view (B, R, T) -> K unordered machine grids (B, K, R, T)."""
        return th.sigmoid(self.dec(view.flatten(1))).reshape(-1, self.k, *self.shape)

    def forward(self, free):
        view = self.encode(free)
        return view, self.decode(view)


def chamfer(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    """Order-free distance between two sets of grids (B, K, R, T): mean squared error to the nearest member, both ways."""
    d = th.cdist(a.flatten(2), b.flatten(2)) ** 2 / a[0, 0].numel()          # (B, Ka, Kb)
    return d.min(2).values.mean() + d.min(1).values.mean()
