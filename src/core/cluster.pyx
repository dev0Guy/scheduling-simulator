from .machines import Machine
from .job import Job

cpdef enum AllocationStatus:
	SUCCESS
	FAILURE

cdef class Cluster:
	cdef Job* jobs;
	cdef Machine* machines;
	cdef unsigned int time;


	cpdef AllocationStatus allocate(self, size_t machine_index, size_t job_index):
		cdef Machine machine = self.machines[machine_index]
		cdef Job job = self.jobs[job_index]
		
		if job.metadata.status != JobStatus.PENDING:
			return AllocationStatus.FAILURE

		machine.usage

		machine.allocate(job)

	cpdef unsigned int foward_time(self):
		self.time += 1 
		self.foward_running_jobs_time()
		self.foward_machines_time()
		return self.time

	cdef void foward_running_jobs_time(self):
		for job in self.jobs:
			if job.metadata.status == JobStatus.RUNNING:
				job.forward_time(self.time)
			# TODO: add all possibleites
			
	cdef void foward_machines_time(self):
		for machine in self.machines:
			machine.forward_time(self.time)