from typing import Any, SupportsFloat,TYPE_CHECKING
from gymnasium import Wrapper
from gymnasium.core import RenderFrame
from scheduling_simulator.core.job import JobStatus
import numpy as np

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment

if TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig
    from scheduling_simulator.core.cluster import ObservationDict

class FailureSkipTimeWrapper(Wrapper['ObservationDict', int, 'ObservationDict', int]):

    def __init__(self, env: SchedulingEnviorment, max_time: int = 500):
        super().__init__(env)
        self._max_time = max_time
        self._time_counter = 0

    def foward_time_untill_max_tick(self, observation: 'ObservationDict') -> 'ObservationDict':
        for _ in range(self._max_time - self._time_counter):
            observation, *_ = super().step(0)
        return observation

    def completion_time_reward(self, observation: 'ObservationDict') -> float:
        return - (observation['wait_time'] + observation['ttl']).sum()

    def step(self, action: int) -> tuple['ObservationDict', SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, trunced, info = super().step(action)
        self._time_counter += 1

        if np.all(observation['status'] == JobStatus.COMPLETED):
            return observation, self.completion_time_reward(observation), True, trunced, info

        if self._time_counter >= self._max_time or not observation['action_success']:
            terminated = True
            observation = self.foward_time_untill_max_tick(observation)
            reward = self.completion_time_reward(observation)


        return observation, reward, terminated, trunced, info


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', dict[str, Any]]:
        observation, _extra  = super().reset(seed=seed, options=options)
        self._time_counter = 0
        return observation, _extra
