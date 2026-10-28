from typing import Any, SupportsFloat, TYPE_CHECKING

import numpy as np
from gymnasium import Wrapper

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


class FailureSkipTimeWrapper(Wrapper['ObservationDict', int, 'ObservationDict', int]):
    """Caps an episode at `max_time` steps and auto-skips time whenever
    the underlying allocation attempt fails, so a bad policy doesn't
    burn its whole step budget on invalid actions.
    """

    def __init__(self, env: SchedulingEnviorment, max_time: int = 500, skip_penalty_divisor: float = 1.0,
                 skip_reward_mode: str = "count"):
        super().__init__(env)
        self._max_time = max_time
        # skip reward = -(pending + running jobs) / skip_penalty_divisor  (1.0 = the original reward;
        # max_time gives -(waiting + running jobs) / max_ticks)
        self._skip_div = float(skip_penalty_divisor)
        # "count":    skip reward = -(number of pending + running jobs) / divisor        (original)
        # "wait_sum": skip reward = -(sum of wait_time of the pending + running jobs) / divisor
        #             (wait_time grows while a job is pending and freezes when it starts)
        assert skip_reward_mode in ("count", "wait_sum")
        self._skip_mode = skip_reward_mode
        self._time_counter = 0

    @staticmethod
    def _mask_non_pending_usage(observation: 'ObservationDict') -> 'ObservationDict':
        # observation['jobs_usage'][np.where(observation['status'] != JobStatus.PENDING)] = 256
        return observation

    @staticmethod
    def _skip_time_reward(observation: 'ObservationDict') -> float:
        return sum(
            -10.0 / max(float(observation['size'][idx]), 1e-6)
            for idx, status in enumerate(observation['status'])
            if status in (JobStatus.PENDING, JobStatus.RUNNING)
        )

    @staticmethod
    def _stupid_reward(observation: 'ObservationDict') -> float:
        return sum(
            -1.0
            for _, status in enumerate(observation['status'])
            if status in (JobStatus.PENDING, JobStatus.RUNNING)
        )

    def _skip_penalty(self, observation: 'ObservationDict') -> float:
        if self._skip_mode == "count":
            return self._stupid_reward(observation)
        active = (observation['status'] == JobStatus.PENDING) | (observation['status'] == JobStatus.RUNNING)
        return -float(np.sum(observation['wait_time'][active]))

    @staticmethod
    def _completion_time_reward(observation: 'ObservationDict') -> float:
        active_mask = (
            (observation['status'] == JobStatus.PENDING) |
            (observation['status'] == JobStatus.RUNNING)
        )
        if not np.any(active_mask):
            return 0.0  # nothing active this step — no penalty to assign
        wait_time = observation['wait_time'][active_mask]
        ttl = observation['ttl'][active_mask]
        return -float(np.mean(wait_time + ttl))

    def _terminal_override(self, observation: 'ObservationDict') -> tuple[bool, float] | None:
        """Returns (done, reward) if a wrapper-level terminal condition
        is hit this step, else None (caller should use its own reward)."""
        if np.all(observation['status'] == JobStatus.COMPLETED) and len(observation['status']) > 0:
            return True, 100.0
        if self._time_counter >= self._max_time:
            return True, -100.0
        return None

    def step(self, action: int) -> tuple['ObservationDict', SupportsFloat, bool, bool, dict[str, Any]]:
        observation, _, terminated, truncated, info = self.env.step(action)
        observation = self._mask_non_pending_usage(observation)
        self._time_counter += 1

        skip_time, m_idx, j_idx = self.env.unwrapped._cluster.action_to_value(action)
        has_allocation_failed = not observation['action_success']

        override = self._terminal_override(observation)
        if override is not None:
            done, reward = override
            return observation, reward, done, False, info

        if skip_time:
            return observation, self._skip_penalty(observation) / self._skip_div, terminated, False, info

        if has_allocation_failed:
            # Allocation attempt failed: burn one tick of simulated time
            # instead of letting the policy retry indefinitely on the
            # same (invalid) step.
            observation, _, terminated, truncated, info = self.env.step(0)
            observation = self._mask_non_pending_usage(observation)
            observation['action_success'] = False
            self._time_counter += 1

            override = self._terminal_override(observation)
            if override is not None:
                done, reward = override
                return observation, reward, done, False, info

            invalid_action_penalty = -10
            return observation, invalid_action_penalty + self._skip_penalty(observation) / self._skip_div, terminated, False, info

        return observation, 1.0, terminated, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', dict[str, Any]]:
        observation, extra = self.env.reset(seed=seed, options=options)
        self._time_counter = 0
        return observation, extra
