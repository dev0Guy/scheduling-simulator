import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.expiremnt.policy import SchedulingPolicy


def test_actor_is_equivariant_to_job_order() -> None:
    config = {
        'n_machines': 1,
        'n_jobs': 2,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    env = SchedulingEnviorment(config, render_mode='rgb_array')
    observation, _ = env.reset(seed=0)
    model = MaskablePPO(
        SchedulingPolicy,
        env,
        n_steps=8,
        batch_size=4,
        seed=0,
    )

    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    original = model.policy.get_distribution(observation_tensor).distribution.probs

    swapped = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }
    job_fields = [
        'jobs_usage',
        'status',
        'ttl',
        'arrival',
        'wait_time',
        'scheduled_at',
        'finished_at',
        'size',
    ]
    for field in job_fields:
        swapped[field] = swapped[field][::-1].copy()

    swapped_tensor, _ = model.policy.obs_to_tensor(swapped)
    permuted = model.policy.get_distribution(swapped_tensor).distribution.probs
    env.close()

    np.testing.assert_allclose(
        original.detach().numpy()[:, [0, 2, 1]],
        permuted.detach().numpy(),
        atol=1e-6,
    )


def test_policy_learns_through_masked_rollout() -> None:
    config = {
        'n_machines': 1,
        'n_jobs': 2,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    env = SchedulingEnviorment(config, render_mode='rgb_array')
    model = MaskablePPO(
        SchedulingPolicy,
        env,
        n_steps=8,
        batch_size=4,
        seed=0,
    )

    before = model.policy.action_net.query.weight.detach().clone()
    model.learn(16)
    after = model.policy.action_net.query.weight.detach()
    env.close()

    assert th.isfinite(after).all()
    assert not th.equal(before, after)
