import numpy as np

from core.job import Job
from core.machine import Machine
from core.cluster import Cluster
usage1 = np.array([
    [2, 1],
    [2, 1],
], dtype=np.int32)

usage2 = np.array([
    [1, 3],
    [1, 2],
], dtype=np.int32)

jobs = np.array([
    Job(usage1, arrival_time=0, size=3),
    Job(usage2, arrival_time=2, size=2),
])

machines = np.array([
    Machine(np.array([[8, 16], [8, 16]], dtype=np.int32)),
    Machine(np.array([[4, 8], [4, 8]], dtype=np.int32)),
])

cluster = Cluster(jobs, machines)

print(cluster)
print("----"*20)
print(cluster.observation.to_dict())

print(cluster.allocate(0, 0))
print(cluster.observation.to_dict())
cluster.foward_time()
cluster.foward_time()
cluster.foward_time()
print(cluster.observation.to_dict())

# # import numpy as np
# # from core.observation import Observation


# # class ClusterSimulator:
# #     def __init__(self, num_machines=2, num_resources=3, num_slots=4, num_jobs=3):
# #         self._machines_usage = np.zeros(
# #             (num_machines, num_resources, num_slots), dtype=np.int32
# #         )
# #         self._jobs_usage = np.zeros(
# #             (num_machines, num_resources, num_slots), dtype=np.int32
# #         )
# #         self._status = np.zeros(num_jobs, dtype=np.int32)
# #         self._arrival = np.arange(num_jobs, dtype=np.int32)
# #         self._length = np.full(num_jobs, 5, dtype=np.int32)
# #         self._tick_to_completion = np.full(num_jobs, 5, dtype=np.int32)
# #         self.tick = 0

# #     def step(self, action):
# #         self.tick += 1
# #         # ... your real scheduling logic mutates the arrays here ...

# #         return Observation(
# #             np.ascontiguousarray(self._machines_usage),
# #             np.ascontiguousarray(self._jobs_usage),
# #             np.ascontiguousarray(self._status),
# #             np.ascontiguousarray(self._arrival),
# #             np.ascontiguousarray(self._length),
# #             np.ascontiguousarray(self._tick_to_completion),
# #             np.array([self.tick], dtype=np.int32),
# #         )


# # sim = ClusterSimulator()
# # obs = sim.step(action=None)

# # print(obs.machines_free_space[0, 0, 0])  # natural 3D indexing, no manual math
# # print(obs.jobs_status)  # 1D memoryview, iterate/index normally
# # print(obs.last_action)  # plain int
# # # obs.as_dict()                 # escape hatch: real dict of numpy arrays, for old call sites
