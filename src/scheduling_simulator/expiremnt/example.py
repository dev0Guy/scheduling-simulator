# TODO: Add Render where selected action is with color
# TODO: add metric of number of completed job
import typing as tp

from stable_baselines3.dqn import DQN
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from stable_baselines3.common.env_checker import check_env

from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.expiremnt.feature_extractor.features_extractor import SchedulingFeaturesExtractor
from scheduling_simulator.expiremnt.train_runner import ExperimentRunner

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig

config: 'ClusterGenerationConfig' = {
    'n_machines': 1,
    'n_jobs': 10,
    'n_resource': 1,
    'n_time': 1,
    'max_capacity': 255
}
runner = ExperimentRunner(
    config,
    train_steps=200_000,
    evalution_steps=500,
    policy_kwargs=dict(
        # features_extractor_class=SchedulingFeaturesExtractor,
        # features_extractor_kwargs=dict(cnn_out_dim=64, mlp_out_dim=64),
        net_arch=[512, 1024, 512, 128, 32],
    ),
    max_time=20
)
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
