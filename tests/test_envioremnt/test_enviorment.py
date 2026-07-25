from typing import Tuple, TYPE_CHECKING
from hypothesis.control import assume
from scheduling_simulator.core.cluster import Cluster
from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


from tests.test_core.strategies import cluster_strategies
from tests.test_core.test_cluster import does_cluster_can_run_all_jobs, extract_observation_from_cluster_for_filter, get_possible_allocation_foreach_job
from tests.test_envioremnt.strategies import scheduling_enviorment_strategy

def assert_tick_untill_job_arrival_time_with_correct_status(env: SchedulingEnviorment, observation: 'ObservationDict', job_idx: int) -> 'ObservationDict':
    time_untill_arrival = observation['arrival'][job_idx] - observation['time']
    assert observation['status'][job_idx] == JobStatus.NOT_CREATED
    for _ in range(time_untill_arrival):
        assert observation['status'][job_idx] == JobStatus.NOT_CREATED
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
    return observation

def assert_run_job_untill_completion_with_correct_status(env: SchedulingEnviorment, observation: 'ObservationDict', job_idx: int) -> 'ObservationDict':
    for _ in range(observation['ttl'][job_idx]):
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

    max_ttl = np.max(observation['ttl'])
    for _ in range(max_ttl):
        observation, *_ = env.step(0)
        env.render()

    assert np.all(observation['ttl'] == 0)
    assert np.all(observation['status'] == JobStatus.COMPLETED)
    assert np.all(observation['machines_usage'] == 0)
