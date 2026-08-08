from scheduling_simulator.scheduler.abc_scheduler import Scheduler
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict

class RandomScheduler(Scheduler):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def select(self, observation: 'ObservationDict') -> tuple[bool, int, int]:
        options = self.options(observation)

        if options is None:
            return True, 0, 0

        idx = self._rng.integers(len(options))
        return False, *options[idx]
