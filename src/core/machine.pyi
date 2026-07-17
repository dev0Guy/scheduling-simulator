import numpy as np
import numpy.typing as npt

class Machine:
    capacity: npt.NDArray[np.int32]
    usage: npt.NDArray[np.int32]

    def __init__(self, capacity: npt.NDArray[np.int32]) -> None: ...
