from .job cimport Job, JobStatus
from .machine cimport Machine
cimport numpy as cnp

cdef struct Action:
    bint skip
    unsigned int selected_machine
    unsigned int selected_job


cdef class Observation:
    cdef int[:, :, ::1] machines_usage
    cdef int[:, :, ::1] machines_capacity
    cdef int[:, :, ::1] jobs_usage
    cdef int[::1] status
    cdef int[::1] ttl
    cdef int[::1] arrival_time
    cdef int[::1] size
    cdef unsigned int time

    cpdef dict to_dict(self)


cdef class Cluster:
    cdef Machine[::1] machines
    cdef Job[::1] jobs
    cdef unsigned int time
    cdef readonly Observation observation

    cdef Observation create_observation(self)
    cpdef void update_observation(self)
    cdef bint allocate(self, int machine_index, int job_index)
    cdef void foward_time(self)
    cdef Action action_from(self, unsigned int v)
    cpdef tuple action_to_value(self, unsigned int v)
    cpdef unsigned int allocation_to_action(self, unsigned int machine_idx, unsigned int job_idx)
    cpdef Observation step(self, unsigned int v)
    cpdef Observation get_observation(self)
