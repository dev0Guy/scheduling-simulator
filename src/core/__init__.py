"""fastmath: a tiny example library backed by a compiled Cython extension."""

from .observation import Observation
from .job import Job
from .machine import Machine
from .cluster import Cluster

__all__ = ["Observation", "Job", "Machine", "Cluster"]
__version__ = "0.1.0"
