"""Training environment for the tournament form: a fixed window of `slots` job slots, of which only 1..slots hold real jobs.

In the tournament evaluation (scripts/tournament/tournament_eval.py) the model never sees a whole large job pool;
it sees slices of at most `slots` jobs, the last slice of a round is padded with empty slots, and the winners are
sliced again. So the model has to decide well when the window holds anything from one job up to a full window.
This environment trains exactly that: every episode draws k real jobs (k uniform in [min_jobs, max_jobs]), runs the
ordinary environment with k jobs, and shows the policy a `slots`-slot observation whose padding slots look exactly
like the tournament's padding (status COMPLETED, usage / size / wait / ttl / arrival 0), so the policy masks them as
not startable and the attention ignores them.

The step limit of an episode is 11 steps per real job (the same rule as the other models).
The job count of an episode is a function of its seed (evaluation: 42 + episode), so a model and the SJF baseline
can be compared on identical episodes.
"""
from typing import Any

import gymnasium as gym
import numpy as np

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper

_JOB_KEYS = ("ttl", "arrival", "wait_time", "scheduled_at", "finished_at", "size")


class VariableJobsEnv(gym.Env):
    metadata = {"render_modes": ["none"]}

    def __init__(self, config: dict, min_jobs: int = 1, max_jobs: int | None = None, steps_per_job: int = 11) -> None:
        super().__init__()
        self._config = dict(config)
        self.slots = int(config["n_jobs"])
        self.min_jobs = min_jobs
        self.max_jobs = max_jobs or self.slots
        assert 1 <= self.min_jobs <= self.max_jobs <= self.slots
        self.steps_per_job = steps_per_job
        self.n_machines = int(config["n_machines"])
        self.render_mode = "none"
        template = SchedulingEnviorment(self._config, render_mode="none")      # full-window spaces
        self.observation_space = template.observation_space
        self.action_space = template.action_space
        self._inner: FailureSkipTimeWrapper | None = None
        self.k = self.slots

    # ---- helpers -------------------------------------------------------------------------------------------
    def _pad(self, obs: dict) -> dict:
        k, s = self.k, self.slots
        out = {"machines_usage": obs["machines_usage"], "machines_capacity": obs["machines_capacity"],
               "time": obs["time"], "action_success": int(obs["action_success"])}
        ju = np.zeros((s,) + obs["jobs_usage"].shape[1:], obs["jobs_usage"].dtype)
        ju[:k] = obs["jobs_usage"]
        out["jobs_usage"] = ju
        st = np.full(s, int(JobStatus.COMPLETED), obs["status"].dtype)
        st[:k] = obs["status"]
        out["status"] = st
        for key in _JOB_KEYS:
            v = np.zeros(s, obs[key].dtype)
            v[:k] = obs[key]
            out[key] = v
        return out

    def _inner_action(self, a: int) -> int:
        if a == 0:
            return 0
        m, j = (a - 1) // self.slots, (a - 1) % self.slots
        if j >= self.k:                       # an empty slot was picked: counts as a skip (the policy masks it)
            return 0
        return 1 + m * self.k + j

    # ---- gym API -------------------------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            rng = np.random.default_rng(seed)
            inner_seed = seed
        else:
            rng = self.np_random
            inner_seed = int(rng.integers(0, 2 ** 31 - 1))
        self.k = int(rng.integers(self.min_jobs, self.max_jobs + 1))
        cfg = dict(self._config, n_jobs=self.k)
        self._inner = FailureSkipTimeWrapper(SchedulingEnviorment(cfg, render_mode="none"),
                                             max_time=self.steps_per_job * self.k)
        obs, info = self._inner.reset(seed=inner_seed)
        info = dict(info, n_real_jobs=self.k)
        return self._pad(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._inner.step(self._inner_action(int(action)))
        return self._pad(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        inner = self._inner.unwrapped.action_masks()
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = inner[0]
        for m in range(self.n_machines):
            mask[1 + m * self.slots:1 + m * self.slots + self.k] = inner[1 + m * self.k:1 + (m + 1) * self.k]
        return mask

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
