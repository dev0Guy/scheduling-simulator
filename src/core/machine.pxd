import numpy as np

cdef class Machine:
    cdef readonly int[:, ::1] capacity
    cdef readonly int[:, ::1] usage

    cpdef bint add_usage(self, int[:, ::1] job_usage)

    cpdef void forward_time(self, unsigned int time) noexcept
