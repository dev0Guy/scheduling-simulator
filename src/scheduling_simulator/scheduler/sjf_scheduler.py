from scheduling_simulator.scheduler.abc_scheduler import Scheduler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


class ShortestJobScheduler(Scheduler):
    """Picks the (job, machine) pair among currently-valid options whose
    job has the smallest `size` — i.e. Shortest-Job-First. Ties are
    broken by whichever option appears first in `options()`.
    """

    def select(self, observation: 'ObservationDict') -> tuple[bool, int, int]:
        options = self.options(observation)
        if not options:
            return True, 0, 0

        sizes = observation['size']
        best_idx = min(range(len(options)), key=lambda i: sizes[options[i][0]])
        return False, *options[best_idx]
