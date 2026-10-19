"""Exact optimal solver for M identical machines, per-resource usage vectors.

Generalises optimal_solver_multires.py (one machine) to `n_machines` machines of the
same capacity on every resource. A job runs on one machine, non-preemptive, several
jobs may share a machine while the sum of usage stays <= capacity on EVERY resource.
Objective: total completion time (equivalently total flow time), which is what the
wrapper reward measures.

Event-driven memoized search. State = (time, set of finished jobs, per-machine tuple
of (job, finish_time)); machines are interchangeable, so the state is canonicalised by
sorting the machines. At each event every feasible assignment of pending jobs to
machines (including "start nothing") is tried and the clock jumps to the next event.

`MultiMachineSolver.action(obs)` returns the optimal ENVIRONMENT action from any state
of an episode (0 = skip, 1 + machine * n_jobs + job = start `job` on `machine`).
"""
import sys

import numpy as np

sys.setrecursionlimit(10000)


class MultiMachineSolver:

    def __init__(self, sizes, usages, capacity, arrivals, n_machines):
        self.n = len(sizes)
        self.m = n_machines
        self.p = [int(round(s)) for s in sizes]
        self.u = [tuple(int(round(v)) for v in x) for x in usages]
        self.arr = [int(round(a)) for a in arrivals]
        self.cap = capacity
        self.R = len(self.u[0])
        self.full = (1 << self.n) - 1
        self.memo = {}
        self.place = {}
        self._plan = None

    # --- helpers -------------------------------------------------------------
    def _used(self, active):
        used = [0] * self.R
        for j, _ in active:
            for r in range(self.R):
                used[r] += self.u[j][r]
        return used

    def _assignments(self, pending, actives):
        """All feasible ways to start pending jobs: list of tuples of (job, machine)."""
        used = [self._used(a) for a in actives]
        assigned = [[] for _ in range(self.m)]
        out = []

        def rec(i, chosen):
            if i == len(pending):
                out.append(tuple(chosen))
                return
            rec(i + 1, chosen)                       # job i not started now
            j = pending[i]
            tried = set()
            for k in range(self.m):
                key = (actives[k], tuple(assigned[k]))
                if key in tried:                     # interchangeable machine
                    continue
                tried.add(key)
                if all(used[k][r] + self.u[j][r] <= self.cap for r in range(self.R)):
                    for r in range(self.R):
                        used[k][r] += self.u[j][r]
                    assigned[k].append(j)
                    chosen.append((j, k))
                    rec(i + 1, chosen)
                    chosen.pop()
                    assigned[k].pop()
                    for r in range(self.R):
                        used[k][r] -= self.u[j][r]

        rec(0, [])
        return out

    def _after(self, t, done, actives, assignment):
        """State right after starting `assignment` at t and jumping to the next event."""
        new = [list(a) for a in actives]
        for j, k in assignment:
            new[k].append((j, t + self.p[j]))
        busy = {j for a in new for j, _ in a}
        future = [self.arr[j] for j in range(self.n)
                  if not (done >> j) & 1 and j not in busy and self.arr[j] > t]
        finishes = [f for a in new for _, f in a]
        events = finishes + future
        if not events:
            return None
        next_t = min(events)
        new_done, n_fin = done, 0
        rem = []
        for a in new:
            keep = []
            for j, f in a:
                if f == next_t:
                    new_done |= 1 << j
                    n_fin += 1
                else:
                    keep.append((j, f))
            rem.append(tuple(sorted(keep)))
        return next_t, new_done, tuple(sorted(rem)), n_fin

    def _pending(self, t, done, actives):
        busy = {j for a in actives for j, _ in a}
        return [j for j in range(self.n)
                if not (done >> j) & 1 and j not in busy and self.arr[j] <= t]

    def dfs(self, t, done, actives):
        if done == self.full:
            return 0
        key = (t, done, actives)
        hit = self.memo.get(key, -1)
        if hit != -1:
            return hit
        best = None
        for asg in self._assignments(self._pending(t, done, actives), actives):
            nxt = self._after(t, done, actives, asg)
            if nxt is None:
                continue
            next_t, nd, na, n_fin = nxt
            sub = self.dfs(next_t, nd, na)
            if sub is None:
                continue
            cost = next_t * n_fin + sub
            if best is None or cost < best:
                best = cost
        self.memo[key] = best
        return best

    # --- A* with a speed-M SRPT lower bound --------------------------------------
    def _lower_bound(self, t, done, actives):
        """Lower bound on the sum of completion times of all unfinished jobs.

        Running jobs finish exactly at their finish time (counted exactly, and left out
        of the relaxations: dropping jobs can only lower the bound of the others).
        For the others the bound is the larger of
          * trivial: every job needs its own length, max(arrival, t) + size;
          * per resource r: a job needs size * usage_r / capacity units of that
            resource's capacity, M units are available per tick, so the sum of
            completion times is at least that of a preemptive single server of speed
            M in SRPT order (optimal on one machine). Time is counted in ticks of 1/M
            so all numbers are integers, completions are rounded up."""
        import heapq
        M = self.m
        total = 0
        rest = []
        for a in actives:
            for j, f in a:
                total += f
        busy = {j for a in actives for j, _ in a}
        for j in range(self.n):
            if not (done >> j) & 1 and j not in busy:
                rest.append((max(self.arr[j], t), j))
        trivial = sum(r + self.p[j] for r, j in rest)
        best = trivial
        for r in range(self.R):
            jobs = sorted((M * rel, self.p[j] * self.u[j][r], j) for rel, j in rest)
            # work is size * usage / capacity; scale ticks by capacity as well
            k, now, heap, acc = 0, M * t * self.cap, [], 0
            jobs = [(rel * self.cap, w, j) for rel, w, j in jobs]
            while k < len(jobs) or heap:
                if not heap:
                    now = max(now, jobs[k][0])
                while k < len(jobs) and jobs[k][0] <= now:
                    heapq.heappush(heap, (jobs[k][1], jobs[k][2]))
                    k += 1
                rem, j = heapq.heappop(heap)
                nxt = jobs[k][0] if k < len(jobs) else None
                if nxt is not None and now + rem > nxt:
                    heapq.heappush(heap, (rem - (nxt - now), j))
                    now = nxt
                    continue
                now += rem
                acc += -(-now // (M * self.cap))
            best = max(best, acc)
        return total + best

    def _step(self, t, done, actives, assignment):
        """Like _after but keeps machine labels (no sorting of machines)."""
        new = [list(a) for a in actives]
        for j, k in assignment:
            new[k].append((j, t + self.p[j]))
        busy = {j for a in new for j, _ in a}
        future = [self.arr[j] for j in range(self.n)
                  if not (done >> j) & 1 and j not in busy and self.arr[j] > t]
        events = [f for a in new for _, f in a] + future
        if not events:
            return None
        next_t = min(events)
        new_done, n_fin, rem = done, 0, []
        for a in new:
            keep = []
            for j, f in a:
                if f == next_t:
                    new_done |= 1 << j
                    n_fin += 1
                else:
                    keep.append((j, f))
            rem.append(tuple(sorted(keep)))
        return next_t, new_done, tuple(rem), n_fin

    def plan(self, t, done, actives):
        """Optimal remaining plan from a (labelled) state: (cost, {time: [(job, machine)]})."""
        import heapq
        nodes = [(None, ())]                       # (parent index, assignment taken at parent's time)
        times = [t]
        h0 = self._lower_bound(t, done, actives)
        heap = [(h0, 0, 0, t, done, actives, 0)]  # f, tie, g, t, done, actives, node idx
        best_g = {}
        tie = 0
        while heap:
            f, _, g, tt, dd, aa, idx = heapq.heappop(heap)
            if dd == self.full:
                path, i = [], idx
                while nodes[i][0] is not None:
                    path.append((times[nodes[i][0]], nodes[i][1]))
                    i = nodes[i][0]
                plan = {}
                for time_, asg in reversed(path):
                    if asg:
                        plan[time_] = list(asg)
                return g, plan
            key = (tt, dd, tuple(sorted(aa)))
            if best_g.get(key, 1 << 60) < g:
                continue
            for asg in self._assignments(self._pending(tt, dd, aa), aa):
                nxt = self._step(tt, dd, aa, asg)
                if nxt is None:
                    continue
                nt, nd, na, n_fin = nxt
                ng = g + nt * n_fin
                nkey = (nt, nd, tuple(sorted(na)))
                if best_g.get(nkey, 1 << 60) <= ng:
                    continue
                best_g[nkey] = ng
                nodes.append((idx, asg)); times.append(nt)
                tie += 1
                heapq.heappush(heap, (ng + self._lower_bound(nt, nd, na), tie, ng, nt, nd, na, len(nodes) - 1))
        return None, {}

    # --- public ----------------------------------------------------------------
    def reset_plan(self):
        self._plan, self._expected, self._pending_check = None, None, None

    def action(self, obs):
        """Optimal environment action for the current state. Call `executed(a, success)`
        afterwards with the action that was actually executed."""
        from scheduling_simulator.core.job import JobStatus
        status = [int(x) for x in obs["status"]]
        t = int(round(float(np.asarray(obs["time"]).reshape(-1)[0])))
        if getattr(self, "_plan", None) is None:
            ttl = [int(round(float(x))) for x in obs["ttl"]]
            done = 0
            actives = [[] for _ in range(self.m)]
            for j, s in enumerate(status):
                if s == int(JobStatus.COMPLETED):
                    done |= 1 << j
                elif s == int(JobStatus.RUNNING):
                    actives[self.place[j]].append((j, t + ttl[j]))
            _, self._plan = self.plan(t, done, tuple(tuple(sorted(a)) for a in actives))
        todo = [(j, k) for j, k in self._plan.get(t, []) if status[j] == int(JobStatus.PENDING)]
        self._last = 0 if not todo else 1 + todo[0][1] * self.n + todo[0][0]
        return self._last

    def executed(self, action, success):
        """Tell the solver what was really executed; invalidates the plan on a deviation."""
        if action != getattr(self, "_last", None):
            self._plan = None
        if action > 0 and success:
            self.place[(action - 1) % self.n] = (action - 1) // self.n
