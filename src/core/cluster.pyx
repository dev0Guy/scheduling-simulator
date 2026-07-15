from .machines import Machine
from .job import Job, JobStatus


cdef class Cluster:
	cdef Job* jobs;
	cdef Machine* machines;
	cdef unsigned int time;


	cpdef bint allocate(self, size_t machine_index, size_t job_index, int time):
		cdef Machine machine = self.machines[machine_index]
		cdef Job job = self.jobs[job_index]
		
		if job.metadata.status != JobStatus.PENDING:
			return False

        if not machine.add_usage(job.usage):
            return False

        job.metadata.scheduled_at = time
		job.update_status = JobStatus.RUNNING

		return True

	cpdef unsigned int foward_time(self):
		self.time += 1 
		self.foward_running_jobs_time()
		self.foward_machines_time()
		return self.time

	cdef void foward_running_jobs_time(self):
		for job in self.jobs:
			job.forward_time(self.time)
			
	cdef void foward_machines_time(self):
		for machine in self.machines:
			machine.forward_time(self.time)