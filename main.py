import numpy as np
from core.observation import Observation


class ClusterSimulator:
    def __init__(self, num_machines=2, num_resources=3, num_slots=4, num_jobs=3):
        self._machines_usage = np.zeros(
            (num_machines, num_resources, num_slots), dtype=np.int32
        )
        self._jobs_usage = np.zeros(
            (num_machines, num_resources, num_slots), dtype=np.int32
        )
        self._status = np.zeros(num_jobs, dtype=np.int32)
        self._arrival = np.arange(num_jobs, dtype=np.int32)
        self._length = np.full(num_jobs, 5, dtype=np.int32)
        self._tick_to_completion = np.full(num_jobs, 5, dtype=np.int32)
        self.tick = 0

    def step(self, action):
        self.tick += 1
        # ... your real scheduling logic mutates the arrays here ...

        return Observation(
            np.ascontiguousarray(self._machines_usage),
            np.ascontiguousarray(self._jobs_usage),
            np.ascontiguousarray(self._status),
            np.ascontiguousarray(self._arrival),
            np.ascontiguousarray(self._length),
            np.ascontiguousarray(self._tick_to_completion),
            np.array([self.tick], dtype=np.int32),
        )


sim = ClusterSimulator()
obs = sim.step(action=None)

print(obs.machines_free_space[0, 0, 0])  # natural 3D indexing, no manual math
print(obs.jobs_status)  # 1D memoryview, iterate/index normally
print(obs.last_action)  # plain int
# obs.as_dict()                 # escape hatch: real dict of numpy arrays, for old call sites
