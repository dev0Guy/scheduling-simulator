import pytest
import numpy as np
from core import Job


def assert_metadata(
    job: Job,
    *,
    arrival_time: int,
    size: int,
    wait_time: int,
    scheduled_at: int,
    finished_at: int,
    status: int,
    ttl: int,
) -> None:
    assert job.metadata["arrival_time"] == arrival_time
    assert job.metadata["size"] == size
    assert job.metadata["wait_time"] == wait_time
    assert job.metadata["scheduled_at"] == scheduled_at
    assert job.metadata["finished_at"] == finished_at
    assert job.metadata["status"] == status
    assert job.metadata["ttl"] == ttl


def test_basic_example() -> None:
    usage = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
    arrival_time = 1
    size = 3
    job = Job(usage, arrival_time=arrival_time, size=size)

    assert np.all(job.usage == usage)
    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=0,
        scheduled_at=0,
        finished_at=0,
        status=0,
        ttl=3,
    )

    job.forward_time(1)
    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=0,
        scheduled_at=0,
        finished_at=0,
        status=1,
        ttl=3,
    )

    job.forward_time(2)
    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=1,
        scheduled_at=0,
        finished_at=0,
        status=1,
        ttl=3,
    )

    job.update_status(2, 2)
    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=1,
        scheduled_at=2,
        finished_at=0,
        status=2,
        ttl=3,
    )

    for tick_count in range(3):
        assert_metadata(
            job,
            arrival_time=arrival_time,
            size=size,
            wait_time=1,
            scheduled_at=2,
            finished_at=0,
            status=2,
            ttl=size - tick_count,
        )
        job.forward_time(3 + tick_count)

    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=1,
        scheduled_at=2,
        finished_at=5,
        status=3,
        ttl=0,
    )

    job.forward_time(7)
    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=1,
        scheduled_at=2,
        finished_at=5,
        status=3,
        ttl=0,
    )
