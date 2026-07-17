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

jobs = [
    Job(usage1, arrival_time=0, size=3),
    Job(usage2, arrival_time=2, size=2),
]

machines = [
    Machine(np.array([[8, 16], [8, 16]], dtype=np.int32)),
    Machine(np.array([[4, 8], [4, 8]], dtype=np.int32)),
]

cluster = Cluster(machines, jobs)

print(cluster)
print("----"*20)
print(cluster.observation.to_dict())

print(cluster.step(1))
print(cluster.observation.to_dict())
cluster.step(0)
cluster.step(0)
cluster.step(0)
print(cluster.observation.to_dict())
