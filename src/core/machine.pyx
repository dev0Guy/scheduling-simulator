


cdef class Machine:
	cdef readonly int[:, ::1] capacity
	cdef readonly int[:, ::1] usage

	def __init__(self, int[:, ::1] capacity) -> None:
		self.usage = None # TODO: replace with zeros np array of the same shape as capacity
		self.capacity = capacity

	cdef add_usage(self, int[:, ::1] job_usage):
		# TODO: check if job_usage can fit in the machine's capacity
		# if not raise an exception
		# if it can fit, add job_usage to self.usage and update free_space accordingly
		self.usage += job_usage

	cpdef void forward_time(self, unsigned int time) except *:
		# TODO: implement logic to forward time for the machine, which may involve updating usage and freeing up resources as jobs complete
		pass 
