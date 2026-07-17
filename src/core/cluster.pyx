from .job cimport Job, JobStatus
from .machine cimport Machine
import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound, initializedcheck


cdef class Observation:
    cdef int[:, :, ::1] machines_usage
    cdef int[:, :, ::1] jobs_usage
    cdef int[::1] status
    cdef int[::1] ttl

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def __init__(self, int[:, :, ::1] machines_usage, int[:, :, ::1] jobs_usage) -> None:
        self.machines_usage = machines_usage
        self.jobs_usage = jobs_usage
        self.status = np.empty(jobs_usage.shape[0],dtype=np.int32)
        self.ttl = np.empty(jobs_usage.shape[0],dtype=np.int32)

    cpdef dict to_dict(self):
        return {
            'machines_usage': np.asarray(self.machines_usage),
            'jobs_usage': np.asarray(self.jobs_usage),
            'status': np.asarray(self.status),
            'ttl': np.asarray(self.ttl)
        }

cdef class Cluster:
    cdef Job[::1] jobs
    cdef Machine[::1] machines
    cdef int time
    cdef readonly Observation observation

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def __init__(self, Job[::1] jobs, Machine[::1] machines) -> None:
        self.jobs = jobs
        self.machines = machines
        self.time = 0
        self.observation = self.create_observation()
        self.update_observation()

    def __repr__(self):
        return (
            f"Cluster(\n"
            f"  time={self.time},\n"
            f"  jobs={list(self.jobs)},\n"
            f"  machines={list(self.machines)}\n"
            f")"
        )

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    cdef Observation create_observation(self):
        cdef:
            Py_ssize_t NJ = self.jobs.shape[0]
            Py_ssize_t NM = self.machines.shape[0]
            Py_ssize_t T = self.jobs[0].usage.shape[0]
            Py_ssize_t R = self.jobs[0].usage.shape[1]
            Py_ssize_t i, t, r

        cdef:
            cnp.ndarray[cnp.int32_t, ndim=3] machines_free_space = np.empty(
                (NM, T, R), dtype=np.int32
            )
            cnp.ndarray[cnp.int32_t, ndim=3] jobs_usage = np.empty(
                (NM, T, R), dtype=np.int32
            )

        return Observation(
            machines_free_space,
            jobs_usage
        )

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    cpdef void update_observation(self):

        cdef:
            Py_ssize_t i, t, r
            Job job
            Machine machine
            int[:, :, ::1] free = self.observation.machines_usage
            int[:, :, ::1] usage = self.observation.jobs_usage
            int[::1] status = self.observation.status
            int[::] ttl = self.observation.ttl

        # Update jobs
        for i in range(self.jobs.shape[0]):
            job = self.jobs[i]
            status[i] = job.metadata.status
            ttl[i] = job.metadata.ttl
            for t in range(job.usage.shape[0]):
                for r in range(job.usage.shape[1]):
                    usage[i, t, r] = job.usage[t, r]

        # Update machines free space
        for i in range(self.machines.shape[0]):
            machine = self.machines[i]

            for t in range(machine.capacity.shape[0]):
                for r in range(machine.capacity.shape[1]):
                    free[i, t, r] = (
                        machine.capacity[t, r]
                        - machine.usage[t, r]
                    )

    @initializedcheck(False)
    cpdef bint allocate(self, int machine_index, int job_index):
        cdef:
            Machine machine = self.machines[machine_index]
            Job job = self.jobs[job_index]
            bint machine_allocation_succsued = True

        if job.metadata.status != JobStatus.PENDING:
            return False

        machine_allocation_succsued = machine.add_usage(job.usage)

        if not machine_allocation_succsued:
            return False

        job.update_status(JobStatus.RUNNING, self.time)
        self.update_observation()

        return True

    @initializedcheck(False)
    cpdef void foward_time(self):
        cdef:
            Py_ssize_t i
            Machine machine
            Job job

        self.time += 1

        for i in range(self.jobs.shape[0]):
            job = self.jobs[i]
            job.forward_time(self.time)

        for i in range(self.machines.shape[0]):
            machine = self.machines[i]
            machine.forward_time(self.time)

        self.update_observation()

    ## TODO: make the step fucntion where instead of inside python code
