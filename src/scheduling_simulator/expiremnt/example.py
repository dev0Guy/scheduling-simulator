from time import sleep
import typing as tp

from stable_baselines3.dqn import DQN
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from stable_baselines3.common.env_checker import check_env

from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.expiremnt.train_runner import TrainExperimentRunner

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig

config: 'ClusterGenerationConfig' = {
    'n_machines': 1,
    'n_jobs': 4,
    'n_resource': 1,
    'n_time': 10,
    'max_capacity': 255
}
runner = TrainExperimentRunner(config)
print(runner)
runner.run()
# env = SchedulingEnviorment(config, render_mode='human')
# env = FailureSkipTimeWrapper(env, max_time=500)
# check_env(env)

# model = DQN(
#     "MultiInputPolicy",
#     env,
#     learning_rate=1e-3,
#     buffer_size=10_000,
#     learning_starts=100,
#     batch_size=32,
#     gamma=0.99,
#     train_freq=1,
#     target_update_interval=250,
#     verbose=1,
# )

# model.learn(total_timesteps=4_000)
# obs, info = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     sleep(0.2)
#     if terminated or truncated:
#         break
# # TODO: Add all wandb with metric save
