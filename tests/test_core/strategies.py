import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from core import Job, Machine
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
