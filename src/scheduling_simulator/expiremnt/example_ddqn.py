"""Train the scheduling network with Double DQN (instead of PPO).

Same environment, network flags, number of environments, 2M steps and
5000-episode evaluation as example.py, so the result is directly
comparable with the PPO runs and the SJF baseline.
"""
import typing as tp

from scheduling_simulator.expiremnt.runners.ddqn_runner import DDQNRunner

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig


config: "ClusterGenerationConfig" = {
    "n_machines": 1,
    "n_jobs": 10,
    "n_resource": 3,
    "n_time": 10,
    "max_capacity": 255,
}

runner = DDQNRunner(
    config,
    train_steps=2_000_000,
    evalution_steps=5000,
    policy_kwargs={
        "embedding_dim": 256,
        "n_machines": config["n_machines"],
        "max_time": 110.0,
        "per_resource_fit": True,
        "pair_features": True,
        "dynamic_machine": True,
    },
    max_time=110,
    run_with_wandb=True,
    eval_every_steps=12_500,   # callback calls, 8 envs each -> every 100k steps
    n_envs=8,
    learning_rate=1e-4,
    lr_decay=True,
    n_eval_episodes=50,
)

runner.run()
