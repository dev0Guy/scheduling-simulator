"""
Exact optimal solver for 1 | r_j | sum C_j (single machine, release dates,
non-preemptive, minimize total completion time).

Bitmask DP over subsets of jobs (n=10 -> 2^10=1024 states). For each
subset S of already-completed jobs, we track the Pareto frontier of
(finish_time, total_completion_cost) pairs reachable by some ordering of
S -- a pair (f1, c1) dominates (f2, c2) if f1 <= f2 and c1 <= c2, since a
state that finishes earlier AND cheaper is strictly at least as good for
every possible continuation. We only keep non-dominated pairs per subset.

Transition: from a non-dominated (finish_time, cost) state for subset S,
for every job j not in S, start j at max(finish_time, arrival[j]), which
completes at new_finish = start + size[j], adding new_finish to the cost
(completion time contributes directly to sum C_j). This produces a
candidate state for S | {j}.

Once the DP over all subsets completes, the full-set frontier's minimum
cost is the optimal sum of completion times, and backtracking (via
explicit parent pointers) recovers the optimal job order.
"""

import itertools


class _State:
    __slots__ = ("finish", "cost", "parent_mask", "parent_index", "job")

    def __init__(self, finish, cost, parent_mask, parent_index, job):
        self.finish = finish
        self.cost = cost
        self.parent_mask = parent_mask
        self.parent_index = parent_index
        self.job = job


def solve_optimal_order(arrivals, sizes):
    """
    arrivals, sizes: same-length sequences of per-job ints.
    Returns (optimal_order: list[int], total_completion_time: int).
    """
    n = len(arrivals)
    arrivals = [int(round(a)) for a in arrivals]
    sizes = [int(round(s)) for s in sizes]

    full_mask = (1 << n) - 1

    frontier = {0: [_State(0, 0, None, None, None)]}

    masks_by_popcount = sorted(range(1 << n), key=lambda m: bin(m).count("1"))

    for mask in masks_by_popcount:
        if mask not in frontier:
            continue
        states = frontier[mask]

        for idx, s in enumerate(states):
            for j in range(n):
                if mask & (1 << j):
                    continue

                start = max(s.finish, arrivals[j])
                new_finish = start + sizes[j]
                new_cost = s.cost + new_finish
                new_mask = mask | (1 << j)

                existing = frontier.setdefault(new_mask, [])

                dominated = False
                kept = []
                for other in existing:
                    if other.finish <= new_finish and other.cost <= new_cost:
                        dominated = True
                        kept.append(other)
                        continue
                    if new_finish <= other.finish and new_cost <= other.cost:
                        continue  # candidate strictly dominates `other`, drop it
                    kept.append(other)

                if not dominated:
                    kept.append(_State(new_finish, new_cost, mask, idx, j))

                frontier[new_mask] = kept

    best = min(frontier[full_mask], key=lambda s: s.cost)
    total_cost = best.cost

    order = []
    mask = full_mask
    state = best
    while state.parent_mask is not None:
        order.append(state.job)
        parent_mask = state.parent_mask
        parent_index = state.parent_index
        state = frontier[parent_mask][parent_index]
        mask = parent_mask

    order.reverse()
    return order, total_cost


def brute_force_order(arrivals, sizes):
    """Reference brute-force search for small n, used only for validation."""
    n = len(arrivals)
    best_cost = None
    best_order = None
    for perm in itertools.permutations(range(n)):
        t = 0
        cost = 0
        for j in perm:
            t = max(t, arrivals[j]) + sizes[j]
            cost += t
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_order = perm
    return list(best_order), best_cost


if __name__ == "__main__":
    import random

    random.seed(0)
    for trial in range(300):
        n = random.randint(1, 7)
        arrivals = [random.randint(0, 10) for _ in range(n)]
        sizes = [random.randint(1, 10) for _ in range(n)]

        dp_order, dp_cost = solve_optimal_order(arrivals, sizes)
        bf_order, bf_cost = brute_force_order(arrivals, sizes)

        if dp_cost != bf_cost:
            print(f"MISMATCH trial={trial} arrivals={arrivals} sizes={sizes} dp={dp_cost} bf={bf_cost}")
            raise SystemExit(1)

        # sanity: dp_order itself must actually achieve dp_cost
        t = 0
        recomputed = 0
        for j in dp_order:
            t = max(t, arrivals[j]) + sizes[j]
            recomputed += t
        if recomputed != dp_cost:
            print(f"ORDER MISMATCH trial={trial} dp_order={dp_order} recomputed={recomputed} expected={dp_cost}")
            raise SystemExit(1)

    print("All 300 validation trials matched brute force (cost and reconstructed order).")
