from .job cimport Job, JobStatus
from .machine cimport Machine
import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound, initializedcheck
import logging

cdef class Observation:

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def __init__(self, int[:, :, ::1] machines_usage, int [:, :, ::1] machines_capacity, int[:, :, ::1] jobs_usage) -> None:
        cdef Py_ssize_t n_jobs = jobs_usage.shape[0]

        self.status = np.empty(n_jobs, dtype=np.int32)
        self.ttl = np.empty(n_jobs, dtype=np.int32)
        self.arrival_time = np.empty(n_jobs, dtype=np.int32)
        self.size = np.empty(n_jobs, dtype=np.int32)
        self.wait_time = np.empty(n_jobs, dtype=np.int32) - 1
        self.scheduled_at = np.empty(n_jobs, dtype=np.int32) - 1
        self.finished_at = np.empty(n_jobs, dtype=np.int32) - 1

        self.time = 0
        self.action_success = True

        self.machines_usage = machines_usage
        self.jobs_usage = jobs_usage
        self.machines_capacity = machines_capacity

    cpdef dict to_dict(self):
        return {
            'machines_usage': np.asarray(self.machines_usage),
            'machines_capacity': np.asarray(self.machines_capacity),
            'wait_time': np.asarray(self.wait_time),
            'scheduled_at': np.asarray(self.scheduled_at),
            'finished_at': np.asarray(self.finished_at),
            'jobs_usage': np.asarray(self.jobs_usage),
            'status': np.asarray(self.status),
            'arrival': np.asarray(self.arrival_time),
            'ttl': np.asarray(self.ttl),
            'size': np.asarray(self.size),
            'time': self.time,
            'action_success': self.action_success
        }

cdef class Cluster:

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def __init__(self, list[Machine] machines, list[Job] jobs) -> None:
        self.machines = np.array(machines)
        self.jobs = np.array(jobs)
        self.time = 0
        self.observation = self.create_observation()
        self.update_observation()

        if (
            self.observation.machines_usage.shape[1] != self.observation.jobs_usage.shape[1] or
            self.observation.machines_usage.shape[2] != self.observation.jobs_usage.shape[2]
        ): # pragma: no cover
            raise ValueError(
                "machines_usage and jobs_usage must have the same shape"
            ) # pragma: no cover

    def __repr__(self): # pragma: no cover
        return (
            f"Cluster(\n"
            f"  time={self.time},\n"
            f"  jobs={list(self.jobs)},\n"
            f"  machines={list(self.machines)}\n"
            f")"
        ) # pragma: no cover

    cpdef Observation get_observation(self):
        return self.observation

    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    cdef Observation create_observation(self):
        cdef:
            Py_ssize_t NJ = self.jobs.shape[0]
            Py_ssize_t NM = self.machines.shape[0]
            Py_ssize_t R = self.jobs[0].usage.shape[0]
            Py_ssize_t T = self.jobs[0].usage.shape[1]
            Py_ssize_t i, r, t
            Machine machine

        cdef:
            cnp.ndarray[cnp.int32_t, ndim=3] machines_free_space = np.empty(
                (NM, R, T), dtype=np.int32
            )
            cnp.ndarray[cnp.int32_t, ndim=3] jobs_usage = np.empty(
                (NJ, R, T), dtype=np.int32
            )
            cnp.ndarray[cnp.int32_t, ndim=3] machines_capacity = np.empty(
                (NM, R, T), dtype=np.int32
            )

        cdef int id

        for idx in range(NM):
            machine = <Machine>self.machines[idx]
            machines_capacity[idx, :, :] = machine.capacity

        return Observation(
            machines_free_space,
            machines_capacity,
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
            int[:, :, ::1] machine_usage = self.observation.machines_usage
            int[:, :, ::1] job_usage = self.observation.jobs_usage
            int[::1] status = self.observation.status
            int[::1] ttl = self.observation.ttl
            int[::1] arrival_time = self.observation.arrival_time
            int[::1] wait_time = self.observation.wait_time
            int[::1] scheduled_at = self.observation.scheduled_at
            int[::1] finished_at = self.observation.finished_at

            int[::] size = self.observation.size


        self.observation.time = self.time

        # Update jobs
        for i in range(self.jobs.shape[0]):
            job = self.jobs[i]
            status[i] = job.metadata.status
            ttl[i] = job.metadata.ttl
            arrival_time[i] = job.metadata.arrival_time
            wait_time[i] = job.metadata.wait_time
            scheduled_at[i] = job.metadata.scheduled_at
            finished_at[i] = job.metadata.finished_at
            size[i] = job.metadata.size
            for r in range(job.usage.shape[0]):
                for t in range(job.usage.shape[1]):
                    job_usage[i, r, t] = job.usage[r, t]

        # # Update machines machine_usage space
        for i in range(self.machines.shape[0]):
            machine = self.machines[i]

            for r in range(machine.capacity.shape[0]):
                for t in range(machine.capacity.shape[1]):
                    machine_usage[i, r, t] = machine.usage[r, t]

    @initializedcheck(False)
    cdef bint allocate(self, int machine_index, int job_index):
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

        return True

    @initializedcheck(False)
    cdef void foward_time(self):
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

    cdef Action action_from(self, unsigned int v):
        cdef bint skip_time = v == 0
        cdef unsigned int selected_machine = (v - 1) // self.jobs.shape[0]
        cdef unsigned int selected_job = (v - 1) % self.jobs.shape[0]
        return Action(skip_time, selected_machine, selected_job)

    cpdef tuple action_to_value(self, unsigned int v):
        cdef Action action = self.action_from(v)
        return (action.skip, action.selected_machine, action.selected_job)

    cpdef unsigned int allocation_to_action(self, unsigned int machine_idx, unsigned int job_idx):
        return 1 + machine_idx * self.jobs.shape[0] + job_idx

    cpdef Observation step(self, unsigned int v):

        cdef:
            Action action = self.action_from(v)
            unsigned int selected_machine
            unsigned int selected_job

        self.observation.action_success = True

        if action.skip:
            self.foward_time()
        else:
            selected_machine = action.selected_machine
            selected_job = action.selected_job
            allocation_succsued = self.allocate(selected_machine, selected_job)
            if not allocation_succsued:
                self.observation.action_success = False
                logging.info(f"Allocation Failed, job: '{selected_job}', machine: '{selected_machine}'")

        self.update_observation()
        return self.observation

    cpdef bint has_all_jobs_been_completed(self):
        cdef bint reduced = True

        for job in self.jobs:
            reduced &= job.metadata.status == JobStatus.COMPLETED

        return reduced
