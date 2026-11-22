"""Environment that trains the tournament decision itself (end-to-end), for a pool of N jobs and a model with a
window of `slots` (10) job slots.

One *real* scheduling decision of the N-job environment is a knock-out tournament (same procedure as
scripts/tournament/tournament_eval.py): all unfinished jobs are cut into slices of `slots`, the model sees one slice
at a time (real machine state + the slice's jobs, empty slots padded as COMPLETED / zeros), the job it picks wins
the slice, the winners are sliced again, and so on until one slice is left; the pick in that last slice is the
real action (a job on a machine, or skip) and is applied to the real environment. Every model pick is one step of
this environment; only the last pick of a tournament gets the (real) reward, all earlier picks get 0.
Slices without any legal start are not shown (they have no winner), and if no start is legal at all the
environment skips by itself until one is.

The observation carries a flag `final_slice`; the policy (policy/tournament_masked.py) masks skip in non-final slices,
as the evaluation never lets a slice winner be 'skip'. (If a skip still arrives there it is resolved as the first legal job.)
"""
from typing import Any

import gymnasium as gym
import numpy as np

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper

_JOB_KEYS = ("ttl", "arrival", "wait_time", "scheduled_at", "finished_at", "size")
COMPLETED = int(JobStatus.COMPLETED)


class TournamentEnv(gym.Env):
    metadata = {"render_modes": ["none"]}

    def __init__(self, config: dict, slots: int = 10, steps_per_job: int = 11) -> None:
        super().__init__()
        self.N = int(config["n_jobs"])
        self.M = int(config["n_machines"])
        self.slots = slots
        self.steps_per_job = steps_per_job
        self.render_mode = "none"
        self._config = dict(config)
        window_cfg = dict(config, n_jobs=slots)
        template = SchedulingEnviorment(window_cfg, render_mode="none")        # spaces of the 10-slot model
        self.observation_space = gym.spaces.Dict({**template.observation_space.spaces,
                                                  "final_slice": gym.spaces.Box(0.0, 1.0, (1,), np.float32)})
        self.action_space = template.action_space
        self._inner: FailureSkipTimeWrapper | None = None
        self.n_picks = 0

    # ---- views ---------------------------------------------------------------------------------------------
    def _slice(self, obs: dict, jobs: list[int]) -> dict:
        k, s = len(jobs), self.slots
        out = {"machines_usage": obs["machines_usage"], "machines_capacity": obs["machines_capacity"],
               "time": obs["time"], "action_success": int(obs["action_success"])}
        ju = np.zeros((s,) + obs["jobs_usage"].shape[1:], obs["jobs_usage"].dtype)
        st = np.full(s, COMPLETED, obs["status"].dtype)
        if k:
            ju[:k] = obs["jobs_usage"][jobs]
            st[:k] = obs["status"][jobs]
        out["jobs_usage"], out["status"] = ju, st
        for key in _JOB_KEYS:
            v = np.zeros(s, obs[key].dtype)
            if k:
                v[:k] = obs[key][jobs]
            out[key] = v
        return out

    def _legal_jobs(self) -> set[int]:
        mask = self._inner.unwrapped.action_masks()
        return {j for m in range(self.M) for j in range(self.N) if mask[1 + m * self.N + j]}

    # ---- tournament bookkeeping ----------------------------------------------------------------------------
    def _start_tournament(self) -> None:
        """Called when the real environment needs a decision (some start is legal)."""
        self._legal = self._legal_jobs()
        cands = [j for j in range(self.N) if self._obs["status"][j] != COMPLETED] if self.N > self.slots else list(range(self.N))
        self._start_round(cands)

    def _start_round(self, cands: list[int]) -> None:
        chunks = [cands[i:i + self.slots] for i in range(0, len(cands), self.slots)]
        self._final = len(chunks) == 1
        self._queue = [c for c in chunks if any(j in self._legal for j in c)]     # no legal job: no winner, not shown
        self._winners: list[int] = []
        self._current = self._queue.pop(0)

    def _present(self) -> dict:
        out = self._slice(self._obs, self._current)
        out["final_slice"] = np.array([float(self._final)], np.float32)     # skip is a legal pick only in the last slice
        return out

    def _auto_skip_until_decision(self) -> tuple[float, bool, bool]:
        """Skip time while no start is legal. Returns (reward collected, terminated, truncated)."""
        total = 0.0
        while not self._inner.unwrapped.action_masks()[1:].any():
            self._obs, r, te, tr, self._info = self._inner.step(0)
            total += float(r)
            if te or tr:
                return total, te, tr
        return total, False, False

    # ---- gym API -------------------------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        if self._inner is not None:
            self._inner.close()
        self._inner = FailureSkipTimeWrapper(SchedulingEnviorment(self._config, render_mode="none"),
                                             max_time=self.steps_per_job * self.N)
        self._obs, info = self._inner.reset(seed=seed)
        self._carry, te, tr = self._auto_skip_until_decision()
        assert not (te or tr), "episode ended before the first decision"
        self._start_tournament()
        return self._present(), dict(info)

    def step(self, action: int):
        action = int(action)
        legal_here = [j for j in self._current if j in self._legal]
        slot, m = ((action - 1) % self.slots, (action - 1) // self.slots) if action else (None, 0)
        if not self._final:
            # winner of this slice (a skip is no valid pick here: first legal job)
            j = self._current[slot] if slot is not None and slot < len(self._current) and self._current[slot] in self._legal else legal_here[0]
            self._winners.append(j)
            if self._queue:
                self._current = self._queue.pop(0)
            else:
                self._start_round(self._winners)
            return self._present(), 0.0, False, False, {"final_pick": False}
        # final slice: a real action
        if slot is None or slot >= len(self._current):
            inner_action = 0
        else:
            inner_action = 1 + m * self.N + self._current[slot]
        self._obs, r, te, tr, self._info = self._inner.step(inner_action)
        reward = self._carry + float(r)
        self._carry = 0.0
        if not (te or tr):
            self._carry, te, tr = self._auto_skip_until_decision()
            if te or tr:
                reward += self._carry
                self._carry = 0.0
        if te or tr:
            end = self._slice(self._obs, [])
            end["final_slice"] = np.ones(1, np.float32)
            return end, reward, bool(te), bool(tr), dict(self._info, final_pick=True)
        self._start_tournament()
        return self._present(), reward, False, False, {"final_pick": True}

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
