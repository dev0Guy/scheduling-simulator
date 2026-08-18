import typing as tp

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig


def generate_scheduling_env(
    config: 'ClusterGenerationConfig',
    max_time: int,
    path: str,
    with_video: bool = False,
    n_env: int = 4,
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
        _env = FailureSkipTimeWrapper(_env, max_time=max_time)
        return Monitor(_env)

    envs = DummyVecEnv([make_env for _ in range(n_env)])
    if with_video:
        envs = VecVideoRecorder(
            envs,
            path,
            record_video_trigger=lambda x: x == 0,
            video_length=500,
        )
    return envs
