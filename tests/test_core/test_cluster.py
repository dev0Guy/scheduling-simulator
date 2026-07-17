from core import Job, Cluster
from hypothesis import given, strategies as st
from core.cluster import Observation
from strategies import cluster_strategies
import numpy as np

def assert_changed_to_pending_in(observation: dict) -> None:
    time = observation['time']
    for status, arrival in zip(observation['status'], observation['arrival']):
        already_started = status == 1 and arrival == time
        not_started_yet = status == 0 and arrival != time
        assert already_started or not_started_yet


@given(cluster_strategies())
def test_cluster_foward_time_untill_all_jobs_pending(cluster: Cluster) -> None:
    observation = cluster.get_observation().to_dict()
    max_job_arrival_time = np.max(observation['arrival'])
    assert_changed_to_pending_in(observation)
    for _ in range(max_job_arrival_time):
        observation = cluster.step(0).to_dict()
        assert_changed_to_pending_in(observation)

@given(cluster_strategies())
def test_single_not_created_job_untill_completion(cluster: Cluster) -> None:
    raise NotImplemented

@given(cluster_strategies())
def test_allocation_of_pending_job_with_no_space(cluster: Cluster) -> None:
    raise NotImplemented

@given(cluster_strategies())
def test_allocation_of_none_pending_job_with_space(cluster: Cluster) -> None:
    raise NotImplemented

@given(cluster_strategies())
def test_allocation_of_pending_job_with_enogh_space(cluster: Cluster) -> None:
    raise NotImplemented

@given(cluster_strategies())
def test_full_scheduling_with_random_scheduler(cluster: Cluster) -> None:
    raise NotImplemented
