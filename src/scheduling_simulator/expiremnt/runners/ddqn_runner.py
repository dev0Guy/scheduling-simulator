"""Double DQN version of the PPO experiment.

Same network as the PPO policy (feature extractor + `SchedulingActionHead`),
but the 11 outputs are read as Q-values, one per action, and trained with
Double DQN:

    target = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))

The online network picks the next action and the target network scores it,
which removes the over-estimation of the plain DQN max.

Run with `expiremnt/example_ddqn.py`. Nothing here changes the PPO code.
"""
import os

import numpy as np
import torch as th
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch.nn import functional as F
from wandb.integration.sb3 import WandbCallback

from scheduling_simulator.expiremnt.callbacks.scheduler_callbacks import (
    CustomMetricsCallback,
)
from scheduling_simulator.expiremnt.feature_extractor.new_features import (
    StatusSizeWaitTtlCapacityFeaturesExtractor,
)
from scheduling_simulator.expiremnt.policy.new_sched import SchedulingActionHead
from scheduling_simulator.expiremnt.runners.train_runner import (
    ExperimentRunner,
    _linear_schedule,
    get_auto_device,
)


class SchedulingDQNPolicy(DQNPolicy):
    """DQN policy whose Q-network is the scheduling feature extractor plus
    the scheduling action head (one value per action)."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        embedding_dim=256,
        max_time=250.0,
        n_machines=1,
        per_resource_fit=False,
        pair_features=False,
        dynamic_machine=False,
        **kwargs,
    ):
        self.scheduling_n_jobs = int(observation_space["status"].shape[0])
        self.scheduling_n_machines = n_machines
        self.scheduling_embedding_dim = embedding_dim
        self.scheduling_dynamic_machine = dynamic_machine
        self.scheduling_pair_feat_dim = (
            6 + int(observation_space["jobs_usage"].shape[1])
            if pair_features
            else 0
        )
        kwargs.pop("net_arch", None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            features_extractor_class=StatusSizeWaitTtlCapacityFeaturesExtractor,
            features_extractor_kwargs={
                "status_embed_dim": 16,
                "embedding_dim": embedding_dim,
                "attention_hidden_dim": 32,
                "max_time": max_time,
                "per_resource_fit": per_resource_fit,
                "pair_features": pair_features,
                "dynamic_machine": dynamic_machine,
            },
            **kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Online and target networks each get their own feature extractor.
        q_network = super().make_q_net()
        q_network.q_net = SchedulingActionHead(
            n_jobs=self.scheduling_n_jobs,
            n_machines=self.scheduling_n_machines,
            embedding_dim=self.scheduling_embedding_dim,
            pair_feat_dim=self.scheduling_pair_feat_dim,
            dynamic_machine=self.scheduling_dynamic_machine,
        ).to(self.device)
        return q_network


class DoubleDQN(DQN):
    """Stable-Baselines3 DQN with the Double-DQN target.

    `reward_scale` only rescales the TD target: rewards here reach -100 or
    +100, which a Huber loss would learn very slowly. The greedy action does
    not depend on the scale.
    """

    def __init__(self, *args, reward_scale: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_scale = reward_scale

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                # Double DQN: the online net chooses, the target net evaluates.
                next_actions = self.q_net(replay_data.next_observations).argmax(
                    dim=1, keepdim=True
                )
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    dim=1,
                    index=next_actions,
                )
                target_q_values = (
                    replay_data.rewards * self.reward_scale
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = th.gather(
                self.q_net(replay_data.observations),
                dim=1,
                index=replay_data.actions.long(),
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/mean_q", float(current_q_values.mean().item()))


class DDQNRunner(ExperimentRunner):
    """ExperimentRunner that trains Double DQN instead of PPO.

    Evaluation, the SJF comparison and the W&B metrics are inherited, so
    results are directly comparable with the PPO runs.
    """

    def __init__(
        self,
        *args,
        buffer_size: int = 300_000,
        learning_starts: int = 20_000,
        batch_size: int = 128,
        train_freq: int = 4,
        gradient_steps: int = 2,
        target_update_interval: int = 5_000,
        exploration_fraction: float = 0.3,
        exploration_final_eps: float = 0.02,
        reward_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.reward_scale = reward_scale
        if self.run_with_wandb:
            self._run.tags = tuple(self._run.tags or ()) + ("ddqn",)
            wandb.config.update({"algo": "DoubleDQN"}, allow_val_change=True)

    def _train(self, env):
        model = DoubleDQN(
            SchedulingDQNPolicy,
            env,
            learning_rate=(
                _linear_schedule(self.learning_rate)
                if self.lr_decay
                else self.learning_rate
            ),
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=1.0,
            gamma=0.995,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=self.exploration_final_eps,
            max_grad_norm=10.0,
            reward_scale=self.reward_scale,
            verbose=1,
            tensorboard_log=f"runs/{self.run_id}",
            policy_kwargs=self.policy_kwargs,
            device=get_auto_device(),
        )

        eval_env = self.generate_enviroemnt(f"videos/{self.run_id}/eval", with_video=False)
        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=f"models/{self.run_id}",
                log_path=f"models/{self.run_id}",
                eval_freq=self.eval_every_steps,
                n_eval_episodes=self.n_eval_episodes,
            )
        ]
        if self.run_with_wandb:
            callbacks.append(
                WandbCallback(
                    gradient_save_freq=50_000,
                    model_save_path=f"models/{self.run_id}",
                    verbose=2,
                )
            )
            callbacks.append(CustomMetricsCallback())

        model.learn(self.train_steps, callback=CallbackList(callbacks))
        model.save(f"models/{self.run_id}/final_model")
        if self.run_with_wandb:
            wandb.save(f"models/{self.run_id}/final_model.zip")
        eval_env.close()

        best_path = f"models/{self.run_id}/best_model.zip"
        if self.use_best_model and os.path.exists(best_path):
            print(f"Final evaluation uses best checkpoint: {best_path}")
            model = DoubleDQN.load(best_path, device=get_auto_device())
        return model
