import copy
from typing import Optional

import numpy as np
from scheduling_simulator.core import Job
from hypothesis import given, strategies as st
from scheduling_simulator.core.job import JobStatus
from strategies import job_strategies


def assert_metadata(
    job: Job,
    *,
    arrival_time: Optional[int] = None,
    size: Optional[int] = None,
    wait_time: Optional[int] = None,
    scheduled_at: Optional[int] = None,
    finished_at: Optional[int] = None,
    status: Optional[JobStatus] = None,
    ttl: Optional[int] = None,
) -> None:
    assert arrival_time is None or job.metadata["arrival_time"] == arrival_time
    assert size is None or job.metadata["size"] == size
    assert wait_time is None or job.metadata["wait_time"] == wait_time
    assert scheduled_at is None or job.metadata["scheduled_at"] == scheduled_at
    assert finished_at is None or job.metadata["finished_at"] == finished_at
    assert status is None or job.metadata["status"] == status
    assert ttl is None or job.metadata["ttl"] == ttl


def assert_and_forward_job_to_pending(job: Job) -> int:
    arrival_time = job.metadata["arrival_time"]
    original_usage = job.usage.copy()
    if arrival_time != 0:
        for _ in range(arrival_time):
            assert_metadata(
                job,
                wait_time=0,
                scheduled_at=0,
                finished_at=0,
                status=JobStatus.NOT_CREATED,
                ttl=job.metadata["size"],
            )
        job.forward_time(arrival_time)
    assert_metadata(
        job,
        wait_time=0,
        scheduled_at=0,
        finished_at=0,
        status=JobStatus.PENDING,
        ttl=job.metadata["size"],
    )
    assert np.array_equal(job.usage, original_usage)
    return arrival_time


def assert_pending_to_completed(job: Job, current_time: int) -> int:
    original_usage = job.usage.copy()
    arrival_time = job.metadata["arrival_time"]
    assert current_time >= arrival_time and job.metadata["status"] == 1
    wait_time = current_time - arrival_time
    job.update_status(JobStatus.RUNNING, current_time)
    size = job.metadata["size"]
    assert_metadata(
        job, status=JobStatus.RUNNING, scheduled_at=current_time, wait_time=wait_time
    )
    finish_time = current_time + size
    for tick in range(current_time + 1, finish_time):
        job.forward_time(current_time + tick)
        assert_metadata(job, status=JobStatus.RUNNING)
    job.forward_time(finish_time)
    assert_metadata(
        job,
        status=JobStatus.COMPLETED,
        scheduled_at=current_time,
        wait_time=wait_time,
        finished_at=finish_time,
        ttl=0,
    )
    assert np.array_equal(job.usage, original_usage)
    return finish_time


def assert_wait_time_ops_actions(job: Job, current_time: int, n_nops: int) -> int:
    assert n_nops >= 0 and job.metadata["status"] == 1
    original_usage = job.usage.copy()
    assert_metadata(job, wait_time=0, status=JobStatus.PENDING)
    tick = current_time + 1
    for wait_time in range(1, n_nops + 1):
        job.forward_time(current_time + tick)
        assert_metadata(job, wait_time=wait_time, status=JobStatus.PENDING)
    assert_metadata(job, wait_time=n_nops)
    assert np.array_equal(job.usage, original_usage)
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
        status=JobStatus.NOT_CREATED,
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
        status=JobStatus.PENDING,
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
        status=JobStatus.PENDING,
        ttl=3,
    )

    job.update_status(JobStatus.RUNNING, 2)
    assert_metadata(
        job,
        arrival_time=arrival_time,
        size=size,
        wait_time=1,
        scheduled_at=2,
        finished_at=0,
        status=JobStatus.RUNNING,
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
            status=JobStatus.RUNNING,
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
        status=JobStatus.COMPLETED,
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
        status=JobStatus.COMPLETED,
        ttl=0,
    )
