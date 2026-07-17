import numpy.typing as npt
import numpy as np


class Job:
    usage: npt.NDArray[np.int32]

    def __init__(
        self,
        usage: npt.NDArray[np.int32],
        arrival_time: int,
        size: int,
    ) -> None: ...

    def forward_time(self, time: int) -> None: ...

    def update_status(self, new_status: int, time: int) -> None: ...
