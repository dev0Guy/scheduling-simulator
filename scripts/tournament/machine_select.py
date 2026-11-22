"""Choice of the machines shown to a model trained on fewer machines than the environment has.

For a slice of jobs: a machine is a candidate if some PENDING job of the slice fits on it (free capacity - usage >= 0 on
every resource and tick, the environment's own test). The k candidates with the LEAST total free capacity (best fit)
are shown. Fully deterministic, never random: if fewer than k candidates exist, the remaining places are filled with the
non-fitting machines that have the MOST total free capacity (the closest to being able to hold a job), ties by machine
index. `rng` is accepted for backward compatibility and unused. Returned in ascending machine index so the model always
sees a fixed order."""
import os
import numpy as np

PENDING = 1


def select_machines(machines_usage, machines_capacity, jobs_usage, status, k, rng):
    n_m = machines_usage.shape[0]
    if n_m <= k:
        return list(range(n_m))
    mode = os.environ.get("SELECT_MODE", "tight")      # diagnostics: first = machines 0..k-1, loose = LEAST tight fitting
    if mode == "first":
        return list(range(k))
    free = machines_capacity.astype(np.int64) - machines_usage.astype(np.int64)          # (M, R, T)
    pend = [j for j in range(len(status)) if status[j] == PENDING]
    fits = np.zeros(n_m, bool)
    for j in pend:
        fits |= ((free - jobs_usage[j][None]).min(axis=(1, 2)) >= 0)
    total = free.sum(axis=(1, 2))
    order = np.argsort(-total if mode == "loose" else total, kind="stable")
    cand = [m for m in order if fits[m]][:k]
    rest = sorted((m for m in range(n_m) if m not in cand), key=lambda m: (-total[m], m))
    extra = rest[:k - len(cand)]
    return sorted(int(m) for m in cand + extra)
