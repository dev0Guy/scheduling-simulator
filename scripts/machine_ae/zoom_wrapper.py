"""Gym wrapper: a policy that handles `group` machines (default 2) schedules a cluster of ANY size by zooming.

The wrapper takes the original environment and a set autoencoder trained for `group` machines. The real M machines
are padded with empty machines (capacity 0, usage 0, so nothing fits) up to group**L and folded into a tree: the AE
encodes every pair of machines into ONE virtual machine, and again, until only `group` virtual machines are left
(8 machines -> 4 -> 2). The agent's observation always has exactly `group` machines, and its action space is
that of a `group`-machine cluster: 0 = skip, 1 + machine * n_jobs + job.

An agent step is one level of the zoom, not a step of the real environment:
  * the agent picks (job, machine) among the machines currently shown;
  * if that machine is virtual, the observation CHANGES to its `group` children (zoom in one level: reward 0, the
    real environment does not move) and the agent chooses again, now among those children;
  * if it is a real machine, the action (this job on this machine) is executed in the original environment, the
    real reward / done come back, the tree is rebuilt from the NEW real state and the zoom starts again at the top;
  * skip at the TOP level is a real skip (the agent decided to wait), executed immediately;
  * skip below the top is a dead end ("nothing in here fits"): the wrapper BACKTRACKS one level, the virtual machine
    just left is blanked out (capacity 0, so nothing fits it) for the rest of this decision, and the agent chooses again
    among the remaining machines of the upper level. Blanked machines are restored after the next real step.
The job is chosen again at every level; the job of the last level is the one started.
info carries: zoom_level (levels left to the real machines), real_step (True when the real environment moved).
"""
import os
import sys

import gymnasium as gym
import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SetMachineAE  # noqa: E402


class ZoomMachineWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, ae: SetMachineAE, exact_masks: bool = False):
        super().__init__(env)
        self.exact_masks = exact_masks
        self.ae, self.g = ae.eval(), ae.k
        self.n_jobs = int(env.observation_space["status"].shape[0])
        self.m_real = int(env.observation_space["machines_usage"].shape[0])
        self.levels = 0                                        # number of encode rounds, group ** (levels + 1) >= machines
        while self.g ** (self.levels + 1) < self.m_real:
            self.levels += 1
        self.n_leaf = self.g ** (self.levels + 1)
        space = dict(env.observation_space.spaces)
        for key in ("machines_usage", "machines_capacity"):
            box = space[key]
            space[key] = gym.spaces.Box(box.low[:1].repeat(self.g, 0), box.high[:1].repeat(self.g, 0), dtype=box.dtype)
        self.observation_space = gym.spaces.Dict(space)
        self.action_space = gym.spaces.Discrete(1 + self.g * self.n_jobs)
        self._obs = self._tree = self._nodes = self._level = None
        self._path, self._dead = [], set()           # path: (level, nodes) above the current view; dead: (level, node) blanked this decision

    # ---- tree ---------------------------------------------------------------------------------------------------
    def _build(self, obs):
        """levels[0] = real machines padded to n_leaf; levels[l] = encoded pairs of levels[l-1]. Grids are (n, R, T)."""
        cap = obs["machines_capacity"].astype(np.float32)
        free = np.zeros((self.n_leaf,) + cap.shape[1:], np.float32)
        free[: self.m_real] = (cap - obs["machines_usage"]) / 255.0
        capacity = np.zeros_like(free)
        capacity[: self.m_real] = cap
        tree = [(free, capacity)]
        for _ in range(self.levels):
            f, c = tree[-1]
            groups = th.tensor(f.reshape(-1, self.g, *f.shape[1:]))
            with th.no_grad():
                view = self.ae.encode(groups).numpy()          # one virtual machine per group, order-free
            full = np.full((len(view),) + c.shape[1:], 255.0, np.float32)
            tree.append((np.round(view * 255.0) / 255.0, full))
        self._tree = tree

    def _show(self):
        """Observation of the `group` machines currently on screen: same jobs, machines = the current nodes."""
        f, c = self._tree[self._level]
        idx = self._nodes
        obs = dict(self._obs)
        cap, use = np.round(c[idx]), np.round(c[idx] - f[idx] * 255.0).clip(0, 255)
        for slot, node in enumerate(idx):
            if (self._level, node) in self._dead:                  # backtracked out of this machine: nothing fits it
                cap[slot], use[slot] = 0, 0
        obs["machines_capacity"] = cap.astype(self._obs["machines_capacity"].dtype)
        obs["machines_usage"] = use.astype(self._obs["machines_usage"].dtype)
        return obs

    def _top(self):
        self._level, self._nodes = self.levels, list(range(self.g))
        self._path, self._dead = [], set()

    def action_masks(self) -> np.ndarray:
        """Exact legality for the machines on screen: job j on a virtual machine is legal iff j is pending and fits at
        least one REAL machine below it (a real machine: the job fits it now). Skip is always legal. With these masks the
        agent can never walk into a dead end, so use them with exact_masks=True (skip is then always a real skip)."""
        from scheduling_simulator.scheduler.abc_scheduler import Scheduler
        fits = np.zeros((self.n_leaf, self.n_jobs), bool)
        for j, m in Scheduler.options(self._obs):
            fits[m, j] = True
        span = self.g ** self._level                           # real machines under one node of the current level
        mask = np.zeros(self.action_space.n, bool)
        mask[0] = True
        for slot, node in enumerate(self._nodes):
            mask[1 + slot * self.n_jobs: 1 + (slot + 1) * self.n_jobs] = fits[node * span:(node + 1) * span].any(0)
        return mask

    # ---- gym ----------------------------------------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = obs
        self._build(obs)
        self._top()
        return self._show(), info

    def step(self, action):
        action = int(action)
        if action == 0 and self._path and not self.exact_masks:                             # dead end below the top: go back up, blank the machine we left
            self._level, self._nodes, left = self._path.pop()
            self._dead.add((self._level, left))
            return self._show(), 0.0, False, False, {"zoom_level": self._level, "real_step": False, "backtrack": True}
        if action != 0:
            m, j = divmod(action - 1, self.n_jobs)
            node = self._nodes[m]
            if self._level > 0:                                # zoom in: look at the children of the chosen virtual machine
                self._path.append((self._level, self._nodes, node))
                self._level -= 1
                self._nodes = [self.g * node + c for c in range(self.g)]
                return self._show(), 0.0, False, False, {"zoom_level": self._level, "real_step": False}
            action = 1 + node * self.n_jobs + j if node < self.m_real else 0     # real machine (a padding node cannot fit anything)
        obs, reward, term, trunc, info = self.env.step(action)
        self._obs = obs
        self._build(obs)                                       # the real state changed: rebuild the whole tree
        self._top()
        info = dict(info, zoom_level=self._level, real_step=True)
        return self._show(), reward, term, trunc, info
