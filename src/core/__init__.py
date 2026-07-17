"""fastmath: a tiny example library backed by a compiled Cython extension."""
from .job import Job
from .machine import Machine
from .cluster import Cluster, Observation

__all__ = ["Observation", "Job", "Machine", "Cluster"]
__version__ = "0.1.0"
