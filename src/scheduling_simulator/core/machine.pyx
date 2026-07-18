import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound, initializedcheck

cdef class Machine:

    def __init__(self, int[:, ::1] capacity) -> None:
        self.usage = np.zeros_like(capacity)
        self.capacity = capacity

    def __repr__(self):
        return (
            f"Machine("
            f"shape=({self.capacity.shape[0]}, {self.capacity.shape[1]}), "
            f")"
        )

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    cpdef bint add_usage(self, int[:, ::1] job_usage):
        cdef:
            Py_ssize_t i, j
            Py_ssize_t rows = self.usage.shape[0]
            Py_ssize_t cols = self.usage.shape[1]
            int new_cell_value

        for i in range(rows):
            for j in range(cols):
                new_cell_value = self.usage[i, j] + job_usage[i, j]
                if new_cell_value > self.capacity[i, j]:
                    return False
                self.usage[i, j] = new_cell_value

        return True

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    cpdef void forward_time(self, unsigned int time) noexcept:
        cdef:
            Py_ssize_t i, j
            Py_ssize_t rows = self.usage.shape[0]
            Py_ssize_t cols = self.usage.shape[1]
        for i in range(rows):
            for j in range(cols - 1):
                self.usage[i, j] = self.usage[i, j + 1]
            self.usage[i, cols - 1] = 0
