from ast import Call

from scheduling_simulator.core import Cluster
from hypothesis import given, settings, HealthCheck, strategies as st
from scheduling_simulator.core.job import JobStatus
from strategies import cluster_strategies
from typing import Callable
import numpy as np


def does_cluster_has_job_with_status(*, status: JobStatus) -> Callable[[Cluster], bool]:
    def inner(cluster: Cluster) -> bool:
        observation = cluster.get_observation().to_dict()
        return bool(np.any(observation["status"] == status))

    return inner


def get_possible_allocation_foreach_job(cluster: Cluster) -> np.ndarray:
    observation = cluster.get_observation().to_dict()
    return np.all(
        observation["machines_usage"][:, None, :, :]
        >= observation["jobs_usage"][None, :, :, :],
        axis=(2, 3),
    )


def has_scheduled_job_with(status: JobStatus, scheduble: bool = True) -> Callable[[Cluster], bool]:

    def inner(cluster: Cluster) -> bool:
        obs = cluster.get_observation().to_dict()
        compatible = get_possible_allocation_foreach_job(cluster)

        has_status = obs["status"] == status
        runnable = compatible.any(axis=0)
        if scheduble:
             return bool(np.any(has_status & runnable))

        return bool(np.any(has_status & ~runnable))

    return inner

def does_cluster_can_run_all_jobs(cluster: Cluster) -> bool:
    compatible = get_possible_allocation_foreach_job(cluster)
    return bool(compatible.any(axis=0).all())


def assert_changed_to_pending_in(observation: dict) -> None:
    time = observation["time"]
    for status, arrival in zip(observation["status"], observation["arrival"]):
        already_started = status == 1 and arrival <= time
        not_started_yet = status == 0 and arrival != time
        assert already_started or not_started_yet


def foward_time_by(
    cluster: Cluster, count: int, assert_func: Callable[[dict], None]
) -> dict:
    observation = cluster.observation.to_dict()
    for _ in range(count):
        observation = cluster.step(0).to_dict()
        assert_func(observation)
    return observation


def assert_allocation_has_enogh_space(cluster: Cluster, before_status: JobStatus, after_status: JobStatus) -> int:
    observation = cluster.get_observation().to_dict()
    compatible = get_possible_allocation_foreach_job(cluster)
    has_status =  observation["status"] == before_status
    job_idx = np.flatnonzero(compatible.any(axis=0) & has_status)[0]
    assert observation["status"][job_idx] == before_status

    machine_idx = [
        np.flatnonzero(compatible[:, j])
        for j in range(compatible.shape[1])
    ][job_idx][0]

    action = cluster.allocation_to_action(machine_idx, job_idx)
    observation = cluster.step(action).to_dict()
    assert observation["status"][job_idx] == after_status
    return job_idx


@given(cluster=cluster_strategies(), row=st.data())
def test_action_conversion(cluster: Cluster, row: st.DataObject) -> None:
    observation = cluster.get_observation().to_dict()
    n_machines = observation["machines_usage"].shape[0]
    n_jobs = observation["ttl"].shape[0]

    machine_idx = row.draw(st.integers(0, n_machines - 1))
    job_idx = row.draw(st.integers(0, n_jobs - 1))
    action = cluster.allocation_to_action(machine_idx, job_idx)
    after = cluster.action_to_value(action)
    assert after == (0, machine_idx, job_idx)


@given(cluster_strategies())
@settings(deadline=None)
def test_cluster_foward_time_untill_all_jobs_pending(cluster: Cluster) -> None:
    observation = cluster.get_observation().to_dict()
    max_job_arrival_time = np.max(observation["arrival"])
    assert_changed_to_pending_in(observation)
    foward_time_by(cluster, max_job_arrival_time, assert_changed_to_pending_in)


@given(
    cluster_strategies(n_machines=2, n_jobs=8, max_resources=3, max_time=10)
    .filter(does_cluster_has_job_with_status(status=JobStatus.NOT_CREATED))
    .filter(does_cluster_can_run_all_jobs)
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_single_not_created_job_untill_completion(cluster: Cluster) -> None:
    observation = cluster.get_observation().to_dict()
    job_not_created_idx = np.where(observation["status"] == JobStatus.NOT_CREATED)[0][0]

    observation = foward_time_by(
        cluster,
        observation["arrival"][job_not_created_idx],
        lambda obs: obs["status"][job_not_created_idx] == 0,
    )
    observation = cluster.get_observation().to_dict()
    assert observation["status"][job_not_created_idx] == 1

    compatible = get_possible_allocation_foreach_job(cluster)

    selected_machine = [
        np.flatnonzero(compatible[:, j]) for j in range(compatible.shape[1])
    ][job_not_created_idx][0]

    action = cluster.allocation_to_action(selected_machine, job_not_created_idx)
    observation = cluster.step(action).to_dict()
    assert observation["status"][job_not_created_idx] == 2

    observation = foward_time_by(
        cluster,
        observation["ttl"][job_not_created_idx],
        lambda obs: obs["status"][job_not_created_idx] == 2,
    )

    assert (
        observation["status"][job_not_created_idx] == 3
        and observation["ttl"][job_not_created_idx] == 0
    )


@given(
    cluster_strategies(n_machines=2, n_jobs=8, max_resources=3, max_time=10)
    .filter(has_scheduled_job_with(JobStatus.PENDING, scheduble=False))
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_allocation_of_pending_job_with_no_space(cluster: Cluster) -> None:
    observation = cluster.get_observation().to_dict()
    compatible = get_possible_allocation_foreach_job(cluster)
    job_idx = np.flatnonzero(~compatible.any(axis=0) & observation["status"] == JobStatus.PENDING)[0]
    assert observation["status"][job_idx] == JobStatus.PENDING

    machine_idx = [
        np.flatnonzero(~compatible[:, j])
        for j in range(compatible.shape[1])
    ][job_idx][0]

    action = cluster.allocation_to_action(machine_idx, job_idx)
    observation = cluster.step(action).to_dict()

    assert observation["status"][job_idx] == JobStatus.PENDING, (
        "Selected job is not PENDING. The test requires a pending job that cannot fit on any machine."
    )

@given(
    cluster_strategies(n_machines=2, n_jobs=8, max_resources=3, max_time=10)
    .filter(has_scheduled_job_with(JobStatus.PENDING, scheduble=True))
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_allocation_of_pending_job_with_enough_space(cluster: Cluster) -> None:
    assert_allocation_has_enogh_space(cluster, JobStatus.PENDING, JobStatus.RUNNING)

@given(
    cluster_strategies(n_machines=2, n_jobs=8, max_resources=3, max_time=10, min_arrival_time=1)
    .filter(has_scheduled_job_with(JobStatus.NOT_CREATED, scheduble=True))
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
def test_allocation_of_not_created_job_with_enough_space(cluster: Cluster) -> None:
    assert_allocation_has_enogh_space(cluster, JobStatus.NOT_CREATED, JobStatus.NOT_CREATED)

@given(
    cluster_strategies(n_machines=2, n_jobs=8, max_resources=3, max_time=10)
    .filter(has_scheduled_job_with(JobStatus.PENDING, scheduble=True))
)
@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], deadline=None)
def test_allocation_of_running_job_with_enough_space(cluster: Cluster) -> None:
    job_idx = assert_allocation_has_enogh_space(cluster, JobStatus.PENDING, JobStatus.RUNNING)
    observation = cluster.get_observation().to_dict()
    ttl = observation['ttl'][job_idx]
    observartion = foward_time_by(cluster, ttl, lambda x: None)
    assert observartion['status'][job_idx] == JobStatus.COMPLETED
    return
    if ttl > 1:
        observartion = foward_time_by(cluster, 1, lambda x: None)
        assert_allocation_has_enogh_space(cluster, JobStatus.RUNNING, JobStatus.RUNNING)
        observartion = foward_time_by(cluster, ttl-1, lambda x: None)
    else:
        observartion = foward_time_by(cluster, ttl, lambda x: None)
        observation['ttl'][job_idx]
    assert observartion['status'][job_idx] == JobStatus.COMPLETED


# @given(
#     cluster_strategies(n_machines=2, n_jobs=8, max_resources=3, max_time=10)
#     .filter(has_scheduled_job_with(JobStatus.COMPLETED, scheduble=True))
# )
# @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
# def test_allocation_of_completed_job_with_enough_space(cluster: Cluster) -> None:
#     assert_allocation_failed_enogh_space(cluster, JobStatus.COMPLETED, JobStatus.COMPLETED)


# @given(cluster_strategies())
# def test_full_scheduling_with_random_scheduler(cluster: Cluster) -> None:
#     pass
