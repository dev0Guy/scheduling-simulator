from .job cimport Job, JobStatus
from .machine cimport Machine
cimport numpy as cnp


cdef struct Action:
    bint skip
    unsigned int selected_machine
    unsigned int selected_job


cdef class Observation:
    cdef int[:, :, ::1] machines_usage
    cdef int[:, :, ::1] jobs_usage
    cdef int[::1] status
    cdef int[::1] ttl
    cdef int[::1] arrival_time

    cpdef dict to_dict(self)


cdef class Cluster:
    cdef Job[::1] jobs
    cdef Machine[::1] machines
    cdef int time
    cdef readonly Observation observation

    cpdef Observation step(self, unsigned int action)
    cdef Observation create_observation(self)
    cpdef void update_observation(self)
    cdef bint allocate(self, int machine_index, int job_index)
    cdef void foward_time(self)
    cdef Action action_from(self, unsigned int v)
    cpdef Observation step(self, unsigned int v)
