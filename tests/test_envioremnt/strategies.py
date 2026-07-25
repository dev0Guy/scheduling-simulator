from scheduling_simulator.core.cluster import Cluster
from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from hypothesis import strategies as st
from tests.test_core.strategies import cluster_strategies
from typing import TYPE_CHECKING
import numpy as np

from tests.test_core.test_cluster import extract_observation_from_cluster_for_filter, has_scheduled_job_with


if TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig



@st.composite
def scheduling_enviorment_strategy(
    draw,
    n_jobs: int = 10,
    n_machines: int = 3,
    min_resources: int = 1,
    max_resources: int = 5,
    min_time: int = 1,
    max_time: int = 100,
    min_arrival_time: int = 0,
    max_arrival_time: int = 100,
    max_capacity: int = 255,
    max_job_usage: int = 255,
    min_job_usage: int = 1,
    min_capacity: int = 1,
    filter_funcion = extract_observation_from_cluster_for_filter(has_scheduled_job_with(JobStatus.NOT_CREATED, scheduble=True))
) -> SchedulingEnviorment:
    number_of_machines = draw(st.integers(1, n_machines))
    number_of_jobs = draw(st.integers(1, n_jobs))
    number_of_resource = draw(st.integers(min_resources, max_resources))
    number_of_time = draw(st.integers(min_time, max_time))
    render_mode = draw(st.sampled_from(['rgb_array', 'human']))

    config: 'ClusterGenerationConfig' = dict(
        n_machines=number_of_machines,
        n_jobs=number_of_jobs,
        n_resource=number_of_resource,
        n_time=number_of_time,
        max_capacity=max_capacity
    )

    # draw the actual cluster NOW, during generation — not deferred to reset() time
    cluster = draw(
        cluster_strategies(
            n_machines=number_of_machines,
            n_jobs=number_of_jobs,
            min_resources=min_resources,
            max_resources=max_resources,
            max_time=max_time,
            min_time=min_time,
            max_arrival_time=max_arrival_time,
            min_arrival_time=min_arrival_time,
            max_capacity=max_capacity,
            min_capacity=min_capacity,
            max_job_usage=max_job_usage,
            min_job_usage=min_job_usage
        ).filter(filter_funcion)
    )

    def tmp_creator(config: 'ClusterGenerationConfig', random: np.random.Generator) -> Cluster:
        return cluster

    return SchedulingEnviorment(config, render_mode=render_mode, creator=tmp_creator)
