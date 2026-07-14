import copy
from typing import Optional

import numpy as np
from core import Job
from hypothesis import given, strategies as st
from strategies import job_strategies


def assert_metadata(
    job: Job,
    *,
    arrival_time: Optional[int] = None,
    size: Optional[int] = None,
    wait_time: Optional[int] = None,
    scheduled_at: Optional[int] = None,
    finished_at: Optional[int] = None,
    status: Optional[int] = None,
    ttl: Optional[int] = None,
) -> None:
    # TODO: allow none values for finished_at and scheduled_at
    assert arrival_time is None or job.metadata["arrival_time"] == arrival_time
    assert size is None or job.metadata["size"] == size
    assert wait_time is None or job.metadata["wait_time"] == wait_time
    assert scheduled_at is None or job.metadata["scheduled_at"] == scheduled_at
    assert finished_at is None or job.metadata["finished_at"] == finished_at
    assert status is None or job.metadata["status"] == status
    assert ttl is None or job.metadata["ttl"] == ttl


def assert_and_forward_job_to_pending(job: Job) -> int:
    arrival_time = job.metadata["arrival_time"]
    if arrival_time != 0:
        for _ in range(arrival_time):
            assert_metadata(
                job,
                wait_time=0,
                scheduled_at=0,
                finished_at=0,
                status=0,
                ttl=job.metadata["size"],
            )
        job.forward_time(arrival_time)
    assert_metadata(
        job,
        wait_time=0,
        scheduled_at=0,
        finished_at=0,
        status=1,
        ttl=job.metadata["size"],
    )
    return arrival_time


def assert_pending_to_completed(job: Job, current_time: int) -> int:
    arrival_time = job.metadata["arrival_time"]
    assert current_time >= arrival_time and job.metadata["status"] == 1
    wait_time = current_time - arrival_time
    job.update_status(2, current_time)
    size = job.metadata["size"]
    assert_metadata(job, status=2, scheduled_at=current_time, wait_time=wait_time)
    finish_time = current_time + size
    for tick in range(current_time + 1, finish_time):
        job.forward_time(current_time + tick)
        assert_metadata(job, status=2)
    job.forward_time(finish_time)
    assert_metadata(
        job,
        status=3,
        scheduled_at=current_time,
        wait_time=wait_time,
        finished_at=finish_time,
        ttl=0,
    )
    return finish_time


def assert_wait_time_ops_actions(job: Job, current_time: int, n_nops: int) -> int:
    assert n_nops >= 0 and job.metadata["status"] == 1
    assert_metadata(job, wait_time=0, status=1)
    tick = current_time + 1
    for wait_time in range(1, n_nops + 1):
        job.forward_time(current_time + tick)
        assert_metadata(job, wait_time=wait_time, status=1)
    assert_metadata(job, wait_time=n_nops)
    return current_time + n_nops


@given(job_strategies(min_arrival_time=2))
def test_job_not_created_to_pending(job: Job) -> None:
    assert_and_forward_job_to_pending(job)


@given(job_strategies(max_arrival_time=0))
def test_pending_to_completed(job: Job) -> None:
    assert_pending_to_completed(job, current_time=0)


@given(job_strategies(), st.integers(min_value=0, max_value=1_000))
def test_full_scenario_creation_to_completion(job: Job, wait_time: int) -> None:
    time = assert_and_forward_job_to_pending(job)
    time = assert_wait_time_ops_actions(job, time, wait_time)
    _ = assert_pending_to_completed(job, time)


def test_basic_example() -> None:
    # TODO: rewrite using the function above
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
