import argparse
import copy
import json
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.expiremnt.policy import SchedulingPolicy, get_auto_device

PolicyName = Literal['random', 'shortest', 'learned']
MAX_EPISODE_STEPS = 500

MAX_N_JOBS = 48

DEFAULT_CONFIG = {
    'n_machines': 2,
    'n_jobs': 32,
    'n_resource': 2,
    'n_time': 20,
    'max_capacity': 255,
}

TRAIN_JOB_COUNTS = [20, 24, 28, 32, 36]
VAL_JOB_COUNTS = [40]
TEST_JOB_COUNTS = [24, 32, 36, 40, 44]


def make_environment(
    config: dict,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    max_n_jobs: int = MAX_N_JOBS,
):
    return Monitor(
        gym.wrappers.TimeLimit(
            SchedulingEnviorment(
                config, render_mode='rgb_array', max_n_jobs=max_n_jobs,
            ),
            max_episode_steps=max_episode_steps,
        )
    )


def _make_env_fn(config: dict, idx: int, randomize_seed: bool = False):
    rng = np.random.default_rng(idx)
    def _fn():
        if randomize_seed:
            n_jobs = int(rng.choice(TRAIN_JOB_COUNTS))
            train_config = {**config, 'n_jobs': n_jobs}
            env = make_environment(train_config)
            env.reset(seed=int(rng.randint(0, 2**31)))
        else:
            env = make_environment(config)
            env.reset(seed=idx)
        return env
    return _fn


def train_model(
    config: dict,
    total_timesteps: int,
    seed: int,
) -> tuple[MaskablePPO, dict[str, float | int]]:
    environment = DummyVecEnv([_make_env_fn(config, 0, randomize_seed=True)])
    # LR: 3e-4 for first half (progress_remaining > 0.5), then decay to 1e-5
    def lr_schedule(progress_remaining: float) -> float:
        if progress_remaining > 0.5:
            return 3e-4
        factor = progress_remaining / 0.5
        return 1e-5 + (3e-4 - 1e-5) * factor

    # Clip range: 0.2 for first half, then decay to 0.1
    def clip_schedule(progress_remaining: float) -> float:
        if progress_remaining > 0.5:
            return 0.2
        factor = progress_remaining / 0.5
        return 0.1 + (0.2 - 0.1) * factor

    model = MaskablePPO(
        SchedulingPolicy,
        environment,
        learning_rate=lr_schedule,
        clip_range=clip_schedule,
        n_steps=512,
        batch_size=64,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=0.02,
        n_epochs=4,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
        device=get_auto_device(),
    )
    checkpoint_interval = min(5_000, total_timesteps)
    best_parameters = None
    best_validation_flow = np.inf
    best_timestep = 0
    patience = 3
    patience_counter = 0
    validation_seeds = range(20_000, 20_128)

    while model.num_timesteps < total_timesteps:
        training_steps = min(
            checkpoint_interval, total_timesteps - model.num_timesteps
        )
        model.learn(training_steps, reset_num_timesteps=False)
        timestep = model.num_timesteps
        val_config = {**config, 'n_jobs': VAL_JOB_COUNTS[0]}
        validation_flow, _ = evaluate('learned', model, val_config, validation_seeds)
        validation_flow = validation_flow.mean()
        if validation_flow < best_validation_flow:
            best_validation_flow = validation_flow
            best_timestep = timestep
            best_parameters = copy.deepcopy(model.get_parameters())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.set_parameters(best_parameters)
    return model, {
        'best_timestep': best_timestep,
        'best_validation_flow': float(best_validation_flow),
        'validation_episodes': len(validation_seeds),
        'actual_timesteps': model.num_timesteps,
    }


def evaluate(
    policy: PolicyName,
    model: MaskablePPO,
    config: dict,
    seeds: range,
    max_n_jobs: int = MAX_N_JOBS,
) -> tuple[np.ndarray, np.ndarray]:
    flow_times = []
    completions = []
    n_jobs = config['n_jobs']

    for seed in seeds:
        environment = make_environment(config, max_n_jobs=max_n_jobs)
        observation, _ = environment.reset(seed=seed)
        rng = np.random.default_rng(seed)

        while True:
            mask = get_action_masks(environment)
            allocations = np.flatnonzero(mask[1:]) + 1
            if policy == 'random':
                action = int(rng.choice(allocations)) if allocations.size else 0
            elif policy == 'shortest':
                job_indices = (allocations - 1) % n_jobs
                action = (
                    int(allocations[np.argmin(observation['size'][job_indices])])
                    if allocations.size
                    else 0
                )
            else:
                action = int(
                    model.predict(
                        observation,
                        deterministic=False,
                        action_masks=mask,
                    )[0]
                )

            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                break

        completed = bool(terminated and not truncated)
        flow_time = float((observation['finished_at'] - observation['arrival']).sum())
        if not completed:
            flow_time = float(MAX_EPISODE_STEPS * n_jobs)
        flow_times.append(flow_time)
        completions.append(completed)
        environment.close()

    return (
        np.asarray(flow_times, dtype=np.float64),
        np.asarray(completions, dtype=np.bool_),
    )


def summarize(values: np.ndarray, completions: np.ndarray) -> dict[str, float]:
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'completion_rate': float(completions.mean()),
    }


def run_benchmark(total_timesteps: int, episodes: int, seed: int) -> dict:
    model, checkpoint = train_model(DEFAULT_CONFIG, total_timesteps, seed)
    evaluation_seeds = range(10_000, 10_000 + episodes)

    # Evaluate generalization across job counts
    generalization = {}
    for n_jobs in TEST_JOB_COUNTS:
        test_config = {**DEFAULT_CONFIG, 'n_jobs': n_jobs}
        returns = {
            policy: evaluate(policy, model, test_config, evaluation_seeds)
            for policy in ('random', 'shortest', 'learned')
        }
        generalization[n_jobs] = {
            policy: summarize(values, completions)
            for policy, (values, completions) in returns.items()
        }

    results = {
        'config': DEFAULT_CONFIG,
        'training': {
            'requested_timesteps': total_timesteps,
            'seed': seed,
            'train_job_counts': TRAIN_JOB_COUNTS,
            'val_job_counts': VAL_JOB_COUNTS,
            'test_job_counts': TEST_JOB_COUNTS,
            **checkpoint,
        },
        'evaluation': {
            'episodes': episodes,
            'seed_start': evaluation_seeds.start,
            'generalization': generalization,
        },
    }
    model.get_env().close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    results = run_benchmark(args.timesteps, args.episodes, args.seed)
    serialized = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.write_text(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    main()
