import os
import typing as tp

import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.envioremnt.wrappers.sjf_relative_reward_wrapper import SJFRelativeRewardWrapper
from scheduling_simulator.envioremnt.wrappers.clean_scaled_reward_wrapper import CleanScaledRewardWrapper
from scheduling_simulator.envioremnt.wrappers.optimal_gap_reward_wrapper import OptimalGapRewardWrapper

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig


class ExposeActionMasks(gym.Wrapper):
    """Outermost wrapper: gymnasium 1.x wrappers do not forward attributes, and sb3-contrib's
    MaskablePPO calls `env.env_method("action_masks")` on the top-level environment."""

    def action_masks(self):
        return self.env.unwrapped.action_masks()


def generate_scheduling_env(
    config: 'ClusterGenerationConfig',
    max_time: int,
    path: str,
    with_video: bool = False,
    n_env: int = 4,
    sjf_relative: bool = False,
    gamma: float = 0.995,
    maskable: bool = False,
    reward_scale: float = 1.0,
    drop_constant_rewards: bool = False,
    optimal_gap_reward: bool = False,
    normalize_skip_penalty: bool = False,
    skip_reward_mode: str | None = None,
):
    """Single source of truth for building the eval/train environment stack.

    Used by BOTH ExperimentRunner and RandomBaselineRunner so a trained
    model and the random baseline are always evaluated under identical
    conditions: same wrapper (FailureSkipTimeWrapper, not TimeLimit),
    same max_time semantics, same video recording setup.
    """
    render_mode = 'rgb_array' if with_video else 'none'

    def make_env():
        _env = SchedulingEnviorment(config, render_mode=render_mode)
        # normalize_skip_penalty: skip reward = -(waiting + running jobs) / max_time instead of -(waiting + running)
        _env = FailureSkipTimeWrapper(
            _env, max_time=max_time,
            skip_penalty_divisor=float(max_time) if normalize_skip_penalty else 1.0,
            skip_reward_mode=skip_reward_mode or "count")
        if sjf_relative:
            # training only: reward = r + gamma*V_SJF(s') - V_SJF(s) (see the wrapper)
            _env = SJFRelativeRewardWrapper(_env, gamma=gamma, max_time=max_time)
        _env = Monitor(_env)
        if optimal_gap_reward:
            # training only: reward = Q*(s, a) - V*(s), the gap to the optimal action (exact solver)
            _env = OptimalGapRewardWrapper(_env, capacity=config["max_capacity"])
        elif reward_scale != 1.0 or drop_constant_rewards:
            # training only; outside Monitor so the logged episode reward stays the original one
            _env = CleanScaledRewardWrapper(_env, scale=reward_scale, drop_constants=drop_constant_rewards)
        return ExposeActionMasks(_env) if maskable else _env

    envs = DummyVecEnv([make_env for _ in range(n_env)])
    if with_video:
        envs = VecVideoRecorder(
            envs,
            path,
            record_video_trigger=lambda x: x == 0,
            video_length=max_time,
        )
    return envs
