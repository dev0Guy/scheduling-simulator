import typing as tp

from stable_baselines3.dqn import DQN
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from stable_baselines3.common.env_checker import check_env

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig

config: 'ClusterGenerationConfig' ={
    'n_machines': 1,
    'n_jobs': 2,
    'n_resource': 1,
    'n_time': 2,
    'max_capacity': 255
}
env = SchedulingEnviorment(config, render_mode='rgb_array')
check_env(env)

model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10_000,
    learning_starts=100,
    batch_size=32,
    gamma=0.99,
    train_freq=1,
    target_update_interval=250,
    verbose=1,
)

model.learn(total_timesteps=20_000)
# TODO: Add all wandb with metric save
