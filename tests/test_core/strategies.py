import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from core import Job


@st.composite
def job_strategies(
    draw,
    min_size=1,
    max_size=8,
    min_rows=1,
    max_rows=6,
    min_arrival_time: int = 0,
    max_arrival_time: int = 5,
):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    usage = draw(
        arrays(
            dtype=np.int32,
            shape=(rows, size),
            elements=st.integers(min_value=0, max_value=100),
        )
    )
    arrival_time = draw(
        st.integers(min_value=min_arrival_time, max_value=max_arrival_time)
    )

    return Job(usage=usage, arrival_time=arrival_time, size=size)
