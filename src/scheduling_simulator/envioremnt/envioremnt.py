from typing_extensions import Callable
from scheduling_simulator.core.cluster import Observation, Cluster
from scheduling_simulator.core.creator import generate_cluster_python
from typing import Literal, Any, Optional, TYPE_CHECKING
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

if TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig
    from scheduling_simulator.core.cluster import ObservationDict
    from scheduling_simulator.core.render import Renderer

Information = dict
RewardFunction = Callable[[Observation, Optional[Observation]], float]
ClusterCreator = Callable[['ClusterGenerationConfig', np.random.Generator], Cluster]

def flow_time_reward(current_observation: Observation, prev_observation: Optional[Observation]) -> float:
    """Dense per-step reward aligned with total flow time.

    Time-skip steps cost -active_jobs (the actual incremental flow time
    of one simulator tick). Allocation steps cost -pending_jobs, which
    penalizes leaving jobs waiting without advancing time. Since
    allocations are zero-time, their direct flow-time cost is zero, but
    charging -pending_jobs gives the policy an immediate gradient:
    scheduling a pending job reduces the step cost from -pending to
    -(pending-1), and the value function learns which allocations
    minimize future time-skip costs.
    """
    if prev_observation is None:
        return 0.0

    previous = prev_observation.to_dict()
    elapsed = current_observation.to_dict()['time'] - previous['time']

    if elapsed > 0:
        active = np.count_nonzero(
            (previous['status'] == 1) | (previous['status'] == 2)
        )
        return -float(active)
    else:
        pending = np.count_nonzero(previous['status'] == 1)
        return -float(pending)


def generate_deep_rm_cluster(config: 'ClusterGenerationConfig', random: np.random.Generator) -> Cluster:
    return generate_cluster_python(config, random)


class SchedulingEnviorment(gym.Env['ObservationDict', int]):
    _config: 'ClusterGenerationConfig'
    _renderer: Optional['Renderer']
    _creator: ClusterCreator
    _last_observation: Optional[Observation]
    _last_observation_dict: Optional['ObservationDict']
    _cluster: Cluster
    _reward_function: RewardFunction

    metadata = {'render_modes': ['rgb_array', 'human']}

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
        reward_function: RewardFunction = flow_time_reward,
        creator: ClusterCreator = generate_deep_rm_cluster,
        render_mode: Literal['human', 'rgb_array'] = 'human',
        max_n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._config = config
        self._reward_function = reward_function
        self._renderer = None
        self._last_observation_dict = None
        self._n_actual = 0
        self._creator = creator
        self._max_n_jobs = max_n_jobs or config['n_jobs']
        self.observation_space = gym.spaces.Dict(self._create_observation_space())
        n_actions = 1 + (self._max_n_jobs * self._config['n_machines'])
        self.action_space = gym.spaces.Discrete(n_actions)

    def _create_observation_space(self) -> dict:
        n_jobs = self._max_n_jobs
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
        d = observation.to_dict()
        n_actual = d['status'].shape[0]
        n_pad = self._max_n_jobs - n_actual

        def pad_1d(arr: np.ndarray) -> np.ndarray:
            out = np.zeros(self._max_n_jobs, dtype=np.float32)
            out[:n_actual] = arr.astype(np.float32)
            return out

        def pad_3d(arr: np.ndarray) -> np.ndarray:
            out = np.zeros((self._max_n_jobs, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            out[:n_actual] = arr
            return out

        status = d['status']
        jobs_usage = d['jobs_usage']
        if n_pad > 0:
            status = np.concatenate([status, np.full(n_pad, 3, dtype=np.int32)])
            jobs_usage = pad_3d(jobs_usage)

        return {
            'machines_usage': d['machines_usage'],
            'machines_capacity': d['machines_capacity'],
            'jobs_usage': jobs_usage,
            'status': status,
            'ttl': pad_1d(d['ttl']),
            'arrival': pad_1d(d['arrival']),
            'wait_time': pad_1d(d['wait_time']),
            'scheduled_at': pad_1d(d['scheduled_at']),
            'finished_at': pad_1d(d['finished_at']),
            'size': pad_1d(d['size']),
            'time': np.array([d['time']], dtype=np.float32),
            'action_success': int(d['action_success']),
        }


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', Information]:
        super().reset(seed=seed)
        self._cluster = self._creator(self._config, self.np_random)
        self._last_observation = self._cluster.get_observation()
        self._n_actual = self._last_observation.to_dict()['status'].shape[0]
        result = self._cast(self._last_observation)
        self._last_observation_dict = result
        return result, {}

    def step(self, action: int) -> tuple['ObservationDict', float, bool, bool, Information]:
        # Translate from padded action space to cluster's encoding.
        # The action space is sized by max_n_jobs, but the cluster
        # decodes using the actual job count.
        if action == 0:
            cluster_action = 0
        else:
            padded_job = (action - 1) % self._max_n_jobs
            machine_idx = (action - 1) // self._max_n_jobs
            if padded_job >= self._n_actual:
                cluster_action = 0  # padded slot, treat as skip
            else:
                cluster_action = 1 + machine_idx * self._n_actual + padded_job

        previous_observation = self._last_observation
        self._last_observation = self._cluster.step(cluster_action)
        result = self._cast(self._last_observation)
        self._last_observation_dict = result
        reward = self._reward_function(self._last_observation, previous_observation)
        terminated = self._cluster.has_all_jobs_been_completed()

        return result, reward, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        obs = self._last_observation_dict
        n_actual = self._n_actual
        n_machines = self._config['n_machines']
        pending = obs['status'][:n_actual] == 1
        fits = np.all(
            obs['machines_usage'][:, None, :, :]
            + obs['jobs_usage'][:n_actual][None, :, :, :]
            <= obs['machines_capacity'][:, None, :, :],
            axis=(2, 3),
        )
        mask = np.zeros((n_machines, self._max_n_jobs), dtype=bool)
        mask[:, :n_actual] = fits & pending[None, :]
        return np.concatenate(([True], mask.reshape(-1)))

    def render(self) -> Optional[RenderFrame]:
        if self._last_observation is None:
            return None

        if self._renderer is None:
            from scheduling_simulator.core.render import Renderer
            self._renderer = Renderer(self.render_mode == 'human')
        return self._renderer.render(self._last_observation)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
