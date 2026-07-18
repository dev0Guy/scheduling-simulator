from .machine import Machine
from .job import Job
from typing import List
import numpy as np
import numpy.typing as npt

class Observation:
    machines_usage: npt.NDArray[np.int32]
    jobs_usage: npt.NDArray[np.int32]
    status: npt.NDArray[np.int32]
    ttl: npt.NDArray[np.int32]
    arrival_time: npt.NDArray[np.int32]
    size: npt.NDArray[np.int32]
    time: int

    def to_dict(self) -> dict: ...

class Cluster:
    observation: Observation

    def __init__(self, machines: List[Machine], jobs: List[Job]) -> None: ...
    def step(self, v: int) -> Observation: ...
    def get_observation(self) -> Observation: ...
    def allocation_to_action(self, machine_idx: int, job_idx: int, /) -> int: ...
    def action_to_value(self, v: int, /) -> tuple[bool, int, int]: ...
