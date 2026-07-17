import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from core import Job, Machine, Cluster
from typing import TypedDict, Tuple
from typing_extensions import Unpack


class ResourceThrowTimeStrategy(TypedDict, total=False):
    min_capacity: int
    max_capacity: int
    min_time: int
    max_time: int
    min_resources: int
    max_resources: int


DEFAULT_CONSTRAINTS: ResourceThrowTimeStrategy = {
    "min_capacity": 1,
    "max_capacity": 255,
    "min_time": 1,
    "max_time": 100,
    "min_resources": 1,
    "max_resources": 5,
}


@st.composite
def resource_throw_time_strategy(
    draw, **constraints: Unpack[ResourceThrowTimeStrategy]
) -> np.ndarray:
    size = draw(
        st.integers(
            min_value=constraints["min_time"], max_value=constraints["max_time"]
        )
    )
    rows = draw(
        st.integers(
            min_value=constraints["min_resources"],
            max_value=constraints["max_resources"],
        )
    )
    return draw(
        arrays(
            dtype=np.int32,
            shape=(rows, size),
            elements=st.integers(
                min_value=constraints["min_capacity"],
                max_value=constraints["max_capacity"],
            ),
        )
    )


@st.composite
def machine_strategies(draw, **kwargs: Unpack[ResourceThrowTimeStrategy]) -> Machine:
    constraints: ResourceThrowTimeStrategy = {**DEFAULT_CONSTRAINTS, **kwargs}
    capacity = draw(resource_throw_time_strategy(**constraints))
    return Machine(capacity)


@st.composite
def usage_for_machine(draw, machine: Machine):
    usage = np.empty_like(machine.capacity)

    for idx in np.ndindex(machine.capacity.shape):
        usage[idx] = draw(st.integers(0, int(machine.capacity[idx])))

    return usage


@st.composite
def job_strategies(
    draw,
    min_arrival_time: int = 0,
    max_arrival_time: int = 100,
    **kwargs: Unpack[ResourceThrowTimeStrategy],
) -> Job:
    constraints: ResourceThrowTimeStrategy = {**DEFAULT_CONSTRAINTS, **kwargs}
    usage = draw(resource_throw_time_strategy(**constraints))
    size = draw(
        st.integers(min_value=constraints["min_time"], max_value=usage.shape[1])
    )
    arrival_time = draw(
        st.integers(min_value=min_arrival_time, max_value=max_arrival_time)
    )

    return Job(usage=usage, arrival_time=arrival_time, size=size)


@st.composite
def cluster_strategies(
    draw,
    n_jobs: int=1,
    n_machines: int=1,
    min_arrival_time: int = 0,
    max_arrival_time: int = 100,
    **kwargs: Unpack[ResourceThrowTimeStrategy],
) -> Cluster:
    kwargs: ResourceThrowTimeStrategy = {**DEFAULT_CONSTRAINTS, **kwargs}
    number_of_resource = draw(st.integers(kwargs['min_resources'], kwargs['max_resources']))
    number_of_time = draw(st.integers(kwargs['min_time'], kwargs['max_time']))

    kwargs['min_resources'] = kwargs['max_resources'] = number_of_resource
    kwargs['min_time'] = kwargs['max_time'] = number_of_time

    ## TODO make sure the capacity is bigger than all max job usage

    jobs = [
        draw(job_strategies(
            min_arrival_time=min_arrival_time,
            max_arrival_time=max_arrival_time,
            **kwargs
        ))
        for _ in range(n_jobs)
    ]
    machines = [
        draw(machine_strategies(**kwargs))
        for _ in range(n_machines)
    ]
    return Cluster(machines, jobs)
