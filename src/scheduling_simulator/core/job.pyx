from .job cimport Job, JobStatus, JobMetadata


cdef class Job:

	def __init__(self, int[:,::1] usage, unsigned int arrival_time, unsigned int size) -> None:
		self.usage = usage
		self.metadata = JobMetadata()
		self.metadata.arrival_time = arrival_time
		self.metadata.size = size
		self.metadata.wait_time = 0
		self.metadata.ttl = size
		self.metadata.scheduled_at = 0
		self.metadata.finished_at = 0

		if self.metadata.arrival_time == 0:
			self.metadata.status = JobStatus.PENDING
		else:
			self.metadata.status = JobStatus.NOT_CREATED

	cpdef void forward_time(self, unsigned int time) except *:
		if self.metadata.status == JobStatus.PENDING:
			self.metadata.wait_time += 1
		elif self.metadata.status == JobStatus.RUNNING:
			if self.metadata.ttl == 1:
				self.metadata.status = JobStatus.COMPLETED
				self.metadata.finished_at = time

			self.metadata.ttl -= 1

		elif self.metadata.status == JobStatus.NOT_CREATED and time == self.metadata.arrival_time:
			self.metadata.status = JobStatus.PENDING
		elif self.metadata.status == JobStatus.NOT_CREATED:
			pass
		elif self.metadata.status == JobStatus.COMPLETED:
			pass
		else:
			raise ValueError(f"Unknown job status: {self.metadata.status}") # pragma: no cover

	def __repr__(self): # pragma: no cover
		cdef str status # pragma: no cover

		if self.metadata.status == JobStatus.NOT_CREATED: # pragma: no cover
			status = "NOT_CREATED" # pragma: no cover
		elif self.metadata.status == JobStatus.PENDING: # pragma: no cover
			status = "PENDING" # pragma: no cover
		elif self.metadata.status == JobStatus.RUNNING: # pragma: no cover
			status = "RUNNING" # pragma: no cover
		elif self.metadata.status == JobStatus.COMPLETED: # pragma: no cover
			status = "COMPLETED" # pragma: no cover
		else: # pragma: no cover
			status = f"UNKNOWN({<int>self.metadata.status})" # pragma: no cover
		return (
			f"Job("
			f"shape=({self.usage.shape[0]}, {self.usage.shape[1]}), "
			f"status={status}, "
			f"arrival_time={self.metadata.arrival_time}, "
			f"size={self.metadata.size}, "
			f"ttl={self.metadata.ttl}, "
			f"wait_time={self.metadata.wait_time}, "
			f"scheduled_at={self.metadata.scheduled_at}, "
			f"finished_at={self.metadata.finished_at}"
			f")"
		)# pragma: no cover

	cpdef void update_status(self, JobStatus new_status, unsigned int time) except *:
		if self.metadata.status == JobStatus.PENDING and new_status == JobStatus.RUNNING:
			self.metadata.scheduled_at = time
			self.metadata.status = new_status
		else: # pragma: no cover
			raise ValueError(f"Invalid status transition from {self.metadata.status} to {new_status}") # pragma: no cover
