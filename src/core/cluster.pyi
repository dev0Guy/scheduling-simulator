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

    def to_dict(self) -> dict: ...

class Cluster:
    observation: Observation

    def __init__(self, machines: List[Machine], jobs: List[Job]) -> None: ...

    def step(self, v: int) -> Observation: ...
