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
ClusterCreator = Callable[['ClusterGenerationConfig', np.random.Generator], Cluster]

def defualt_reward_function(current_observation: Observation, prev_observation: Optional[Observation]) -> float:
    return -1

def generate_deep_rm_cluster(config: 'ClusterGenerationConfig', random: np.random.Generator) -> Cluster:
    return generate_cluster_python(config, random)


class SchedulingEnviorment(gym.Env['ObservationDict', int]):
    _config: 'ClusterGenerationConfig'
    _renderer: Renderer
    _creator: ClusterCreator
    _last_observation: Optional[Observation]
    _cluster: Cluster
    _rewarder: RewardFunction

    metadata = {'render_modes': ['rgb_array', 'huamn']}

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
        reward_function: RewardFunction = defualt_reward_function,
        creator: ClusterCreator = generate_deep_rm_cluster,
        render_mode: Literal['human', 'rgb_array'] = 'human',
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._config = config
        self._reward_function = reward_function
        self._renderer = Renderer(self.render_mode == 'human')
        self._creator = creator
        self.observation_space = gym.spaces.Dict(self._create_observation_space())
        n_actions = 1 + (self._config['n_jobs'] * self._config['n_machines'])
        self.action_space = gym.spaces.Discrete(n_actions)

    def _create_observation_space(self) -> dict:
        n_jobs = self._config['n_jobs']
        n_machines = self._config['n_machines']
        n_resources = self._config['n_resource']
        n_time = self._config['n_time']
        return {
            'machines_usage': gym.spaces.Box(low=0, high=255, shape=(n_machines, n_resources, n_time), dtype=np.int32),
            'machines_capacity': gym.spaces.Box(low=0, high=255, shape=(n_machines, n_resources, n_time), dtype=np.int32),
            'jobs_usage': gym.spaces.Box(low=0, high=255, shape=(n_jobs, n_resources, n_time), dtype=np.int32),
            'status': gym.spaces.Box(low=0, high=5, shape=(n_jobs,), dtype=np.int32),
            'ttl': gym.spaces.Box(low=0, high=np.inf, shape=(n_jobs,), dtype=np.float32),
            'arrival': gym.spaces.Box(low=0, high=n_time, shape=(n_jobs,), dtype=np.float32),
            'wait_time': gym.spaces.Box(low=0, high=np.inf, shape=(n_jobs,), dtype=np.float32),
            'scheduled_at': gym.spaces.Box(low=0, high=np.inf, shape=(n_jobs,), dtype=np.float32),
            'finished_at': gym.spaces.Box(low=0, high=np.inf, shape=(n_jobs,), dtype=np.float32),
            'size': gym.spaces.Box(low=0, high=np.inf, shape=(n_jobs,), dtype=np.float32),
            'time':  gym.spaces.Box(high=np.inf,low=0, shape=(1,), dtype=np.float32),
            'action_success': gym.spaces.Discrete(2)
        }

    def _cast(self, observation: 'Observation') -> 'ObservationDict':
        observation_dict = observation.to_dict()
        return {
            'machines_usage': observation_dict['machines_usage'],
            'machines_capacity': observation_dict['machines_capacity'],
            'jobs_usage':observation_dict['jobs_usage'],
            'status': observation_dict['status'],
            'ttl': observation_dict['ttl'].astype(np.float32),
            'arrival': observation_dict['arrival'].astype(np.float32),
            'wait_time': observation_dict['wait_time'].astype(np.float32),
            'scheduled_at': observation_dict['scheduled_at'].astype(np.float32),
            'finished_at': observation_dict['finished_at'].astype(np.float32),
            'size': observation_dict['size'].astype(np.float32),
            'time': np.array([observation_dict['time']], dtype=np.float32),
            'action_success': int(observation_dict['action_success'])
        }


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', Information]:
        self._cluster = self._creator(self._config, np.random.default_rng(seed))
        self._last_observation = self._cluster.get_observation()
        return self._cast(self._last_observation), {}

    def step(self, action: int) -> tuple['ObservationDict', float, bool, bool, Information]:
        previous_observation = self._last_observation
        self._last_observation = self._cluster.step(action)
        reward = self._reward_function(self._last_observation, previous_observation)
        terminated = self._cluster.has_all_jobs_been_completed()

        return self._cast(self._last_observation), reward, terminated, False, {}

    def render(self) -> Optional[RenderFrame]:
        if self._last_observation is None:
            return None

        return self._renderer.render(self._last_observation)

    def close(self) -> None:
        self._renderer.close()
