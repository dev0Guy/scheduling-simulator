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
			raise ValueError(f"Unknown job status: {self.metadata.status}")

	def __repr__(self):
		cdef str status

		if self.metadata.status == JobStatus.NOT_CREATED:
			status = "NOT_CREATED"
		elif self.metadata.status == JobStatus.PENDING:
			status = "PENDING"
		elif self.metadata.status == JobStatus.RUNNING:
			status = "RUNNING"
		elif self.metadata.status == JobStatus.COMPLETED:
			status = "COMPLETED"
		else:
			status = f"UNKNOWN({<int>self.metadata.status})"
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
		)

	cpdef void update_status(self, JobStatus new_status, unsigned int time) except *:
		if self.metadata.status == JobStatus.PENDING and new_status == JobStatus.RUNNING:
			self.metadata.scheduled_at = time
			self.metadata.status = new_status
		else:
			print("Invalid status transition from", self.metadata.status, "to", new_status)
			raise ValueError(f"Invalid status transition from {self.metadata.status} to {new_status}")
