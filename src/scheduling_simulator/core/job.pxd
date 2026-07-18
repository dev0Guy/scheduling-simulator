
cpdef enum JobStatus:
	NOT_CREATED = 0
	PENDING = 1
	RUNNING = 2
	COMPLETED = 3

cdef struct JobMetadata:
	JobStatus status
	unsigned int arrival_time
	unsigned int size
	unsigned int ttl
	unsigned int wait_time
	unsigned int scheduled_at
	unsigned int finished_at

cdef class Job:
	cdef readonly int[:,::1] usage
	cdef readonly JobMetadata metadata

	cpdef void forward_time(self, unsigned int time) except *

	cpdef void update_status(self, JobStatus new_status, unsigned int time) except *
