from typing import Tuple, List
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

from scheduling_simulator.core.job import JobStatus

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


ScheduleAction = Tuple[int, int]

class Scheduler(ABC):

    @abstractmethod
    def select(self, observation: 'ObservationDict') -> tuple[bool, int, int]: ...

    def options(self, observation: 'ObservationDict') -> List[ScheduleAction]:
        machines_usage = observation["machines_usage"]
        machines_capacity = observation["machines_capacity"]
        jobs_usage = observation["jobs_usage"]
        status = observation["status"]

        remaining = machines_capacity - machines_usage
        result = remaining[None, :, :, :] - jobs_usage[:, None, :, :]
        possible = np.all(result >= 0, axis=(2, 3))

        pending_mask = status == JobStatus.PENDING
        possible &= pending_mask[:, None]

        job_idx, machine_idx = np.nonzero(possible)
        return list(zip(job_idx.tolist(), machine_idx.tolist()))
