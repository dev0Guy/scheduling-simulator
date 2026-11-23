"""Online heuristic schedulers besides SJF. All pick among the currently valid (job, machine) options (Scheduler.options);
no valid option -> skip. Deterministic: ties broken by the first option (lowest job index, then lowest machine index)."""
import numpy as np

from scheduling_simulator.scheduler.abc_scheduler import Scheduler


def _free_total(obs, m):
    return int((obs["machines_capacity"][m] - obs["machines_usage"][m]).sum())


class FifoBestFitScheduler(Scheduler):
    """Earliest arrival first (FIFO); the machine is the tightest one (least free capacity) where it fits (best fit)."""

    def select(self, obs):
        opts = self.options(obs)
        if not opts:
            return True, 0, 0
        arr = obs["arrival"]
        j = min({o[0] for o in opts}, key=lambda x: (arr[x], x))
        m = min((o[1] for o in opts if o[0] == j), key=lambda x: (_free_total(obs, x), x))
        return False, j, m


class SjfBestFitScheduler(Scheduler):
    """Shortest job first (as SJF), but on the tightest machine that fits instead of the first one."""

    def select(self, obs):
        opts = self.options(obs)
        if not opts:
            return True, 0, 0
        size = obs["size"]
        j = min({o[0] for o in opts}, key=lambda x: (size[x], x))
        m = min((o[1] for o in opts if o[0] == j), key=lambda x: (_free_total(obs, x), x))
        return False, j, m


class TetrisScheduler(Scheduler):
    """Tetris (Grandl et al., SIGCOMM 2014) packing score: dot product of the job's demand vector and the machine's free
    resource vector (per-resource peak demand / free capacity at the current tick, normalised by the capacity), combined with
    shortest-job preference: score = alignment + eps * (1 / size). Highest score wins."""

    def __init__(self, eps: float = 0.5):
        self.eps = eps

    def select(self, obs):
        opts = self.options(obs)
        if not opts:
            return True, 0, 0
        cap = obs["machines_capacity"].astype(np.float64)[:, :, 0].max(axis=0)       # (R,) capacity per resource
        free_now = (obs["machines_capacity"] - obs["machines_usage"]).astype(np.float64)[:, :, 0]   # (M, R)
        jobs = obs["jobs_usage"].astype(np.float64).max(axis=2)                      # (J, R) peak demand
        size = obs["size"]
        best, best_s = None, -1e18
        for j, m in opts:
            s = float(((jobs[j] / cap) * (free_now[m] / cap)).sum()) + self.eps / max(int(size[j]), 1)
            if s > best_s:
                best, best_s = (j, m), s
        return False, *best
