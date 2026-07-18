import numpy as np

from scheduling_simulator.core.job import Job
from scheduling_simulator.core.machine import Machine
from scheduling_simulator.core.cluster import Cluster
from scheduling_simulator.core.render import Renderer
from time import sleep


renderer = Renderer()

# ------------------------------------------------------------------
# Create 10 jobs
# Each job has a 3-resource × 10-time usage matrix.
# ------------------------------------------------------------------

jobs = []

for i in range(10):
    usage = np.array(
        [
            [2 + (i % 3)] * 10,
            [3 + (i % 3)] * 10,
            [4 + (i % 3)] * 10,
        ],
        dtype=np.int32,
    )
    size = int((i % 5) + 2)
    usage[:, size:] = 0
    jobs.append(
        Job(
            usage=usage,
            arrival_time=i,
            size=size,
        )
    )

# ------------------------------------------------------------------
# Create 3 identical machines
# Each machine has capacity 20 for every resource and every time slot.
# Shape = (3 resources, 10 time slots)
# ------------------------------------------------------------------

capacity = np.full((3, 10), 20, dtype=np.int32)

machines = [
    Machine(capacity.copy()),
    Machine(capacity.copy()),
    Machine(capacity.copy()),
]

cluster = Cluster(machines, jobs)

print("-" * 80)
print(cluster.observation.to_dict())

# ------------------------------------------------------------------
# Run one action per job
# ------------------------------------------------------------------

for action in range(len(jobs)):
    obs = cluster.step(action)
    renderer.render(obs)
    sleep(1)

# Final observation
obs = cluster.observation

renderer.render(obs)
sleep(5)
