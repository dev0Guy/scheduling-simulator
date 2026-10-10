"""
Exact optimal solver for the concurrent-packing task with PER-RESOURCE usage:
one machine with capacity C on each of R resources, n jobs (arrival r_j,
duration p_j, usage vector u_j in R^R, constant over the job's run),
non-preemptive, several jobs may run at once while, on EVERY resource, the
sum of their usage stays <= C. Minimizes total completion time
(equivalently total flow time).

This is optimal_solver_packing.py generalised from one usage number per job
to a usage vector per job (a job may be given as an int, which is treated as
a 1-resource vector, so the scalar problems are a special case).

Environment facts this models (checked against cluster.pyx / the wrapper):
  * allocations do not advance time, only a skip does, so any number of
    jobs may start in the same tick;
  * a job started at t occupies the machine during [t, t+p) and completes
    at t+p;
  * a job fits iff the usage of the jobs active at t, plus its own, is <= C
    (jobs only end later, so this is the binding check for its whole run).

Method: event-driven memoized search. Decisions are only worth making at
job arrivals and completions. At each event, try every capacity-feasible
subset of pending jobs to start now (including the empty subset), jump to
the next event, and keep the best.
"""

import itertools
import sys

sys.setrecursionlimit(10000)


def _as_vec(x):
    if isinstance(x, (int, float)):
        return (int(round(x)),)
    return tuple(int(round(v)) for v in x)


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _fits(used, add, capacity):
    return all(x + y <= capacity for x, y in zip(used, add))


def _feasible_subsets(pending, usages, used0, capacity):
    """All subsets of `pending` that fit next to `used0` on every resource (incl. empty)."""
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


def solve(arrivals, sizes, usages, capacity):
    """Returns (total_completion_time, start_times: list[int])."""
    n = len(arrivals)
    arr = [int(round(a)) for a in arrivals]
    p = [int(round(s)) for s in sizes]
    u = [_as_vec(x) for x in usages]
    zero = tuple(0 for _ in u[0])
    full = (1 << n) - 1

    memo = {}

    def dfs(t, done, active):
        # active: tuple of (job, finish_time), sorted by job
        if done == full:
            return 0, ()
        key = (t, done, active)
        hit = memo.get(key)
        if hit is not None:
            return hit

        busy = {j for j, _ in active}
        pending = [
            j for j in range(n)
            if not (done >> j) & 1 and j not in busy and arr[j] <= t
        ]
        used0 = zero
        for j, _ in active:
            used0 = _add(used0, u[j])
        future_arrivals = [
            arr[j] for j in range(n)
            if not (done >> j) & 1 and j not in busy and arr[j] > t
        ]

        best_cost, best_choice = None, None
        for S in _feasible_subsets(pending, u, used0, capacity):
            new_active = tuple(sorted(active + tuple((j, t + p[j]) for j in S)))
            events = [f for _, f in new_active] + future_arrivals
            if not events:
                continue
            next_t = min(events)
            finishing = [j for j, f in new_active if f == next_t]
            remaining = tuple((j, f) for j, f in new_active if f != next_t)
            new_done = done
            for j in finishing:
                new_done |= 1 << j
            sub, _ = dfs(next_t, new_done, remaining)
            if sub is None:
                continue
            cost = next_t * len(finishing) + sub
            if best_cost is None or cost < best_cost:
                best_cost, best_choice = cost, (S, next_t, new_done, remaining)

        memo[key] = (best_cost, best_choice)
        return memo[key]

    total, _ = dfs(0, 0, ())

    starts = [None] * n
    t, done, active = 0, 0, ()
    while done != full:
        _, choice = memo[(t, done, active)] if (t, done, active) in memo else dfs(t, done, active)
        S, next_t, new_done, remaining = choice
        for j in S:
            starts[j] = t
        t, done, active = next_t, new_done, remaining
    return total, starts


def brute_force(arrivals, sizes, usages, capacity, horizon=20):
    """Tick-by-tick exhaustive search (no event compression), tiny n only."""
    n = len(arrivals)
    arr = [int(a) for a in arrivals]
    p = [int(s) for s in sizes]
    u = [_as_vec(x) for x in usages]
    zero = tuple(0 for _ in u[0])
    memo = {}

    def rec(t, started):  # started: tuple of start time or -1 per job
        if t > horizon:
            return None
        finished = [s >= 0 and s + p[j] <= t for j, s in enumerate(started)]
        if all(finished):
            return 0
        key = (t, started)
        if key in memo:
            return memo[key]
        active = [j for j, s in enumerate(started) if s >= 0 and s + p[j] > t]
        used0 = zero
        for j in active:
            used0 = _add(used0, u[j])
        pending = [j for j in range(n) if started[j] < 0 and arr[j] <= t]
        best = None
        for r in range(len(pending) + 1):
            for S in itertools.combinations(pending, r):
                used = used0
                for j in S:
                    used = _add(used, u[j])
                if any(x > capacity for x in used):
                    continue
                st = list(started)
                for j in S:
                    st[j] = t
                done_now = sum(
                    (st[j] + p[j]) for j in range(n)
                    if st[j] >= 0 and st[j] + p[j] == t + 1
                )
                sub = rec(t + 1, tuple(st))
                if sub is None:
                    continue
                if best is None or done_now + sub < best:
                    best = done_now + sub
        memo[key] = best
        return best

    return rec(0, tuple([-1] * n))


if __name__ == "__main__":
    import random

    random.seed(0)
    for trial in range(300):
        n = random.randint(1, 4)
        arrivals = [random.randint(0, 3) for _ in range(n)]
        sizes = [random.randint(1, 3) for _ in range(n)]
        # new environment: resources 0 and 1 drawn independently, last resource unused
        usages = [
            (random.choice([64, 128, 255]), random.choice([64, 128, 255]), 0)
            for _ in range(n)
        ]
        cost, starts = solve(arrivals, sizes, usages, 255)
        bf = brute_force(arrivals, sizes, usages, 255)
        if cost != bf:
            print("MISMATCH", arrivals, sizes, usages, cost, bf)
            raise SystemExit(1)
        recomputed = sum(starts[j] + sizes[j] for j in range(n))
        if recomputed != cost:
            print("START-TIME MISMATCH", arrivals, sizes, usages, starts, cost)
            raise SystemExit(1)
    print("300/300 trials match brute force (cost and reconstructed starts).")
