from typing_extensions import Callable
from scheduling_simulator.core.cluster import Observation, Cluster
from scheduling_simulator.core.render import Renderer
from scheduling_simulator.core.creator import generate_cluster_python
from typing import Literal, Any, Optional, TYPE_CHECKING
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

if TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig
    from scheduling_simulator.core.cluster import ObservationDict

Information = dict
RewardFunction = Callable[[Observation, Optional[Observation]], float]

def defualt_reward_function(current_observation: Observation, prev_observation: Optional[Observation]) -> float:
    return -1

class SchedulingEnviorment(gym.Env['ObservationDict', int]):
    _config: 'ClusterGenerationConfig'
    _renderer: Renderer
    _last_observation: Optional[Observation]
    _cluster: Cluster
    _rewarder: RewardFunction

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
        reward_function: RewardFunction = defualt_reward_function,
        render_mode: Literal['human', 'rgb_array'] = 'human',
    ) -> None:
        super().__init__()
        self._config = config
        self._reward_function = reward_function
        self._renderer = Renderer(render_mode == 'human')

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', Information]:
        self._cluster = generate_cluster_python(self._config, np.random.default_rng(seed))
        self._last_observation = self._cluster.get_observation()
        return self._last_observation.to_dict(), {}

    def step(self, action: int) -> tuple['ObservationDict', float, bool, bool, Information]:
        previous_observation = self._last_observation
        self._last_observation = self._cluster.step(action)
        reward = self._reward_function(self._last_observation, previous_observation)
        terminated = self._cluster.has_all_jobs_been_completed()

        return self._last_observation.to_dict(), reward, terminated, False, {}

    def render(self) -> Optional[RenderFrame]:
        if self._last_observation is None:
            return None

        return self._renderer.render(self._last_observation)

    def close(self) -> None:
        self._renderer.close()
