from core.action cimport Action

cdef class Observation:
    cdef readonly int[:, :, ::1] machines_free_space
    cdef readonly int[:, :, ::1] jobs_usage
    cdef readonly int[::1] jobs_status
    cdef readonly int[::1] jobs_arrival_time
    cdef readonly int[::1] jobs_lengths
    cdef readonly int[::1] jobs_ttl
    cdef readonly int[::1] time
    cdef readonly Action last_action
    def __init__(
        self,
        int[:, :, ::1] machines_free_space,
        int[:, :, ::1] jobs_usage,
        int[::1] jobs_status,
        int[::1] jobs_arrival_time,
        int[::1] jobs_lengths,
        int[::1] jobs_ttl,
        int[::1] time
    ):
        self.machines_free_space = machines_free_space
        self.jobs_usage = jobs_usage
        self.jobs_status = jobs_status
        self.jobs_arrival_time = jobs_arrival_time
        self.jobs_lengths = jobs_lengths
        self.jobs_ttl = jobs_ttl
        self.time = time


    def __repr__(self):
        return (
            f"MetricResourceManagementObservation("
            f"machines.shape={self.machines_free_space.shape[0], self.machines_free_space.shape[1], self.machines_free_space.shape[2]}, "
            f"jobs.shape={self.jobs_usage.shape[0], self.jobs_usage.shape[1], self.jobs_usage.shape[2]}, "
            f"status.shape={(self.jobs_status.shape[0],)}, "
            f"time={self.time})"
        )