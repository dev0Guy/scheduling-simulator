from typing import Tuple, TYPE_CHECKING
from hypothesis.control import assume
from scheduling_simulator.core.cluster import Cluster
from scheduling_simulator.core import Job, Machine
from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment, flow_time_reward
from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


from tests.test_core.strategies import cluster_strategies
from tests.test_core.test_cluster import does_cluster_can_run_all_jobs, extract_observation_from_cluster_for_filter, get_possible_allocation_foreach_job
from tests.test_envioremnt.strategies import scheduling_enviorment_strategy

def assert_tick_untill_job_arrival_time_with_correct_status(env: SchedulingEnviorment, observation: 'ObservationDict', job_idx: int) -> 'ObservationDict':
    time_untill_arrival = int(observation['arrival'][job_idx] - observation['time'])
    assert observation['status'][job_idx] == JobStatus.NOT_CREATED
    for _ in range(time_untill_arrival):
        assert observation['status'][job_idx] == JobStatus.NOT_CREATED
        assert np.all(observation['machines_usage'] <= observation['machines_capacity'])
        observation, *_ = env.step(0)
        env.render()
    assert observation['status'][job_idx] == JobStatus.PENDING
    return observation

def assert_job_allocation_with_correct_usage_and_status(env: SchedulingEnviorment, observation: 'ObservationDict', machine_idx: int, job_idx: int) -> 'ObservationDict':
    original_usage = observation['machines_usage'][machine_idx].copy()
    action = env._cluster.allocation_to_action(machine_idx, job_idx)
    observation, *_ = env.step(action)
    env.render()

    assert (observation['status'][job_idx] == JobStatus.RUNNING)
    np.testing.assert_equal(observation['machines_usage'][machine_idx], original_usage + observation['jobs_usage'][job_idx])
    assert np.all(observation['machines_usage'] <= observation['machines_capacity'])
    return observation

def assert_run_job_untill_completion_with_correct_status(env: SchedulingEnviorment, observation: 'ObservationDict', job_idx: int) -> 'ObservationDict':
    ttl = int(observation['ttl'][job_idx])
    for _ in range(ttl):
        observation, *_ = env.step(0)
        env.render()

    assert(observation['status'][job_idx] == JobStatus.COMPLETED)
    return observation

def does_cluster_jobs_max_usage_is_half_of_all_machines(observation: 'ObservationDict') -> bool:
    compatible = get_possible_allocation_foreach_job(observation)
    return bool(compatible.any(axis=0).all())

def get_machine_for_job(observation: 'ObservationDict', job_idx: int):
    compatible = get_possible_allocation_foreach_job(observation)
    return [
        np.flatnonzero(compatible[:, j]) for j in range(compatible.shape[1])
    ][job_idx][0]

def fits_in_half_capacity_foreach_job(observation: 'ObservationDict') -> bool:
    return bool(np.all(np.all(
        observation["jobs_usage"][None, :, :, :]
        < (observation["machines_capacity"][:, None, :, :] // 2),
        axis=(2, 3),
    )))

@given(scheduling_enviorment_strategy(n_jobs=10, n_machines=5))
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_cluster_deep_rm_select_one_job_untill_completion(env: SchedulingEnviorment) -> None:
    observation, _ = env.reset()
    env.render()

    not_created = observation["status"] == JobStatus.NOT_CREATED
    compatible = get_possible_allocation_foreach_job(observation)
    runnable = compatible.any(axis=0)

    candidates = np.flatnonzero(not_created & runnable)
    assert candidates.size > 0, "no NOT_CREATED job is schedulable in this generated cluster"
    job_idx = candidates[0]

    observation = assert_tick_untill_job_arrival_time_with_correct_status(env, observation, job_idx)

    machine_idx = get_machine_for_job(observation, job_idx)
    original_usage =observation['machines_usage'][machine_idx].copy()

    assert_job_allocation_with_correct_usage_and_status(env, observation, machine_idx, job_idx)

    observation = assert_run_job_untill_completion_with_correct_status(env, observation, job_idx)
    assert np.all(original_usage == observation['machines_usage'][machine_idx])

@given(
    scheduling_enviorment_strategy(
        n_jobs=10,
        n_machines=5,
        filter_funcion=extract_observation_from_cluster_for_filter(does_cluster_can_run_all_jobs)
    )
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], deadline=None)
def test_cluster_deep_rm_run_serial_completion(env: SchedulingEnviorment) -> None:
    observation, _ = env.reset()
    env.render()
    arrival_time = observation["arrival"]
    for job_idx in np.argsort(arrival_time):
        if observation['status'][job_idx] == JobStatus.NOT_CREATED:
            observation = assert_tick_untill_job_arrival_time_with_correct_status(env, observation, job_idx)

        machine_idx = get_machine_for_job(observation, job_idx)

        assert_job_allocation_with_correct_usage_and_status(env, observation, machine_idx, job_idx)

        observation = assert_run_job_untill_completion_with_correct_status(env, observation, job_idx)

    assert np.all(observation['machines_usage'] == 0)


@given(
    scheduling_enviorment_strategy(
        n_jobs=10,
        n_machines=5,
        min_capacity=100,
        max_capacity=255,
        max_job_usage=50,
        min_job_usage=1,
        filter_funcion=extract_observation_from_cluster_for_filter(fits_in_half_capacity_foreach_job)
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_cluster_half_job_to_machine_run_all_jobs_untill_completion(env: SchedulingEnviorment) -> None:
    observation, _ = env.reset()
    env.render()
    n_jobs = len(observation['status'])
    n_machines = observation['machines_usage'].shape[0]
    assume(n_jobs / 2 <= n_machines)

    for job_idx in np.argsort(observation["arrival"]):
        if observation['status'][job_idx] == JobStatus.NOT_CREATED:
            observation = assert_tick_untill_job_arrival_time_with_correct_status(env, observation, job_idx)

        machine_idx = job_idx // 2

        observation = assert_job_allocation_with_correct_usage_and_status(env, observation, machine_idx, job_idx)

    max_ttl = int(np.max(observation['ttl']))
    for _ in range(max_ttl):
        observation, *_ = env.step(0)
        env.render()

    assert np.all(observation['ttl'] == 0)
    assert np.all(observation['status'] == JobStatus.COMPLETED)
    assert np.all(observation['machines_usage'] == 0)


def test_reward_receives_distinct_observation_snapshots() -> None:
    transitions = []

    def reward(current, previous):
        transitions.append((current, previous))
        return 0.0

    config = {
        'n_machines': 1,
        'n_jobs': 1,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    env = SchedulingEnviorment(config, reward_function=reward, render_mode='rgb_array')
    env.reset(seed=0)
    env.step(0)
    env.close()

    current, previous = transitions[0]
    assert current is not previous
    assert previous.to_dict()['time'] == 0
    assert current.to_dict()['time'] == 1


def test_flow_time_reward_distinguishes_allocations_from_time_skips() -> None:
    config = {
        'n_machines': 1,
        'n_jobs': 2,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    def creator(config, random):
        capacity = np.full((1, 10), 255, dtype=np.int32)
        usage = np.zeros((1, 10), dtype=np.int32)
        usage[:, :2] = 100
        return Cluster(
            [Machine(capacity)],
            [Job(usage.copy(), arrival_time=0, size=2),
             Job(usage.copy(), arrival_time=0, size=2)],
        )

    env = SchedulingEnviorment(
        config,
        reward_function=flow_time_reward,
        creator=creator,
        render_mode='rgb_array',
    )
    observation, _ = env.reset(seed=0)

    # Allocation step: 2 pending jobs -> reward = -2
    allocation = env._cluster.allocation_to_action(0, 0)
    observation, alloc_reward, *_ = env.step(allocation)
    assert alloc_reward == -2.0

    # Second allocation: 1 pending job -> reward = -1
    observation, alloc2_reward, *_ = env.step(env._cluster.allocation_to_action(0, 1))
    assert alloc2_reward == -1.0

    # Time skip: 2 active jobs (both running) -> reward = -2
    observation, tick_reward, *_ = env.step(0)
    assert tick_reward == -2.0

    env.close()


def test_action_mask_contains_only_pending_jobs_that_fit() -> None:
    config = {
        'n_machines': 1,
        'n_jobs': 2,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    def creator(config, random):
        capacity = np.full((1, 10), 255, dtype=np.int32)
        first_usage = np.zeros((1, 10), dtype=np.int32)
        first_usage[:, :2] = 150
        second_usage = np.zeros((1, 10), dtype=np.int32)
        second_usage[:, :2] = 100
        return Cluster(
            [Machine(capacity)],
            [
                Job(first_usage, arrival_time=0, size=2),
                Job(second_usage, arrival_time=0, size=2),
            ],
        )

    env = SchedulingEnviorment(config, creator=creator, render_mode='rgb_array')
    env.reset(seed=0)

    initial_mask = env.action_masks()
    assert initial_mask.tolist() == [True, True, True]

    env.step(1)
    observation = env._last_observation.to_dict()
    expected_second_job = bool(np.all(
        observation['machines_usage'][0] + observation['jobs_usage'][1]
        <= observation['machines_capacity'][0]
    ))
    assert env.action_masks().tolist() == [True, False, expected_second_job]
    env.close()


def test_rgb_array_environment_does_not_open_a_display() -> None:
    config = {
        'n_machines': 1,
        'n_jobs': 1,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    env = SchedulingEnviorment(config, render_mode='rgb_array')
    env.reset(seed=0)
    frame = env.render()
    env.close()

    assert frame.shape == (700, 1400, 3)


def test_reset_seed_reproduces_the_episode_sequence() -> None:
    config = {
        'n_machines': 1,
        'n_jobs': 4,
        'n_resource': 1,
        'n_time': 10,
        'max_capacity': 255,
    }
    first = SchedulingEnviorment(config, render_mode='rgb_array')
    second = SchedulingEnviorment(config, render_mode='rgb_array')

    first_initial, _ = first.reset(seed=123)
    first_next, _ = first.reset()
    second_initial, _ = second.reset(seed=123)
    second_next, _ = second.reset()
    first.close()
    second.close()

    np.testing.assert_array_equal(first_initial['jobs_usage'], second_initial['jobs_usage'])
    np.testing.assert_array_equal(first_next['jobs_usage'], second_next['jobs_usage'])
    assert not np.array_equal(first_initial['jobs_usage'], first_next['jobs_usage'])
