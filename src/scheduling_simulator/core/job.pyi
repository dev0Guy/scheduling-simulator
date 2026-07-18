from hypothesis.internal.compat import TypedDict
import numpy.typing as npt
import numpy as np
from enum import IntEnum

class JobStatus(IntEnum):
    NOT_CREATED = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3

class JobMetadata(TypedDict):
    status: JobStatus
    arrival_time: int
    size: int
    ttl: int
    wait_time: int
    scheduled_at: int
    finished_at: int

class Job:
    usage: npt.NDArray[np.int32]
    metadata: JobMetadata

    def __init__(
        self,
        usage: npt.NDArray[np.int32],
        arrival_time: int,
        size: int,
    ) -> None: ...
    def forward_time(self, time: int) -> None: ...
    def update_status(self, new_status: JobStatus, time: int, /) -> None: ...
