"""fastmath: a tiny example library backed by a compiled Cython extension."""

from .observation import Observation
from .job import Job

__all__ = ["Observation", "Job"]
__version__ = "0.1.0"
