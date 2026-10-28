"""Exact optimal value of ANY state of one episode (one machine, per-resource usage).

Event-driven memoised search (same model as scripts/solvers/optimal_solver_multires.py, which was
validated against brute force and against the real environment): at every event (arrival or
completion) every capacity-feasible subset of the pending jobs is started, then the clock jumps to
the next event. The memo is shared by all queries of one episode, so a rollout of an arbitrary policy
only pays for states that were not reached before.

`remaining_penalty(obs)` is the optimal total of the per-tick penalty (number of pending + running
jobs, the "flow-time" part of the environment reward) that is still to be paid from the state of
`obs`; the optimal value of the state under the cleaned reward is minus this number.
"""
import numpy as np

from scheduling_simulator.core.job import JobStatus


def _as_vec(x):
    if isinstance(x, (int, float)):
        return (int(round(x)),)
    return tuple(int(round(v)) for v in x)


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _fits(used, add, capacity):
    return all(x + y <= capacity for x, y in zip(used, add))


def _feasible_subsets(pending, usages, used0, capacity):
    out = []

    def rec(i, chosen, used):
        if i == len(pending):
            out.append(tuple(chosen))
            return
        rec(i + 1, chosen, used)
        j = pending[i]
        if _fits(used, usages[j], capacity):
            chosen.append(j)
            rec(i + 1, chosen, _add(used, usages[j]))
            chosen.pop()

    rec(0, [], used0)
    return out


class OptimalStateSolver:

    def __init__(self, sizes, usages, capacity, arrivals):
        self.n = len(sizes)
        self.p = [int(round(s)) for s in sizes]
        self.u = [_as_vec(x) for x in usages]
        self.arr = [int(round(a)) for a in arrivals]
        self.cap = capacity
        self.zero = tuple(0 for _ in self.u[0])
        self.full = (1 << self.n) - 1
        self.memo = {}

    @classmethod
    def from_obs(cls, obs, capacity):
        return cls(np.asarray(obs["size"]).tolist(), np.asarray(obs["jobs_usage"])[:, :, 0].tolist(),
                   capacity, np.asarray(obs["arrival"]).tolist())

    def _cost(self, t, done, active):
        """Minimal sum of completion times (absolute) of all unfinished jobs."""
        if done == self.full:
            return 0
        key = (t, done, active)
        hit = self.memo.get(key, -1)
        if hit != -1:
            return hit
        n, p, u, arr = self.n, self.p, self.u, self.arr
        busy = {j for j, _ in active}
        pending = [j for j in range(n) if not (done >> j) & 1 and j not in busy and arr[j] <= t]
        used0 = self.zero
        for j, _ in active:
            used0 = _add(used0, u[j])
        future = [arr[j] for j in range(n) if not (done >> j) & 1 and j not in busy and arr[j] > t]
        best = None
        for S in _feasible_subsets(pending, u, used0, self.cap):
            new_active = tuple(sorted(active + tuple((j, t + p[j]) for j in S)))
            events = [f for _, f in new_active] + future
            if not events:
                continue
            next_t = min(events)
            finishing = [j for j, f in new_active if f == next_t]
            remaining = tuple((j, f) for j, f in new_active if f != next_t)
            new_done = done
            for j in finishing:
                new_done |= 1 << j
            sub = self._cost(next_t, new_done, remaining)
            if sub is None:
                continue
            cost = next_t * len(finishing) + sub
            if best is None or cost < best:
                best = cost
        self.memo[key] = best
        return best

    def remaining_penalty(self, obs):
        """Optimal remaining sum of (pending + running) per tick, from the state in `obs`."""
        status = [int(x) for x in obs["status"]]
        t = int(round(float(np.asarray(obs["time"]).reshape(-1)[0])))
        ttl = [int(round(float(x))) for x in obs["ttl"]]
        done, active = 0, []
        arrived_unfinished, future_arrivals = 0, 0
        for j, s in enumerate(status):
            if s == int(JobStatus.COMPLETED):
                done |= 1 << j
            else:
                if s == int(JobStatus.RUNNING):
                    active.append((j, t + ttl[j]))
                if s == int(JobStatus.NOT_CREATED):
                    future_arrivals += self.arr[j]
                else:
                    arrived_unfinished += 1
        if done == self.full:
            return 0
        cost = self._cost(t, done, tuple(sorted(active)))
        # job j is counted once per tick transition into times arrival .. completion - 1
        return cost - arrived_unfinished * (t + 1) - future_arrivals
