from scheduling_simulator.core.cluster import Cluster
from typing import TypedDict
import numpy as np

class ClusterGenerationConfig(TypedDict):
    n_machines: int
    n_jobs: int
    n_resource: int
    n_time: int
    max_capacity: int

def generate_cluster_python(
    config: ClusterGenerationConfig,
    random: np.random.Generator = np.random.default_rng()
) -> Cluster
