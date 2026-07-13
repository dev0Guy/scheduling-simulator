


cdef class Machine:
	cdef readonly int[:, ::1] free_space
	cdef readonly int[:, ::1] usage

	def __init__(self, int[:, ::1] free_space, int[:, ::1] usage):
		self.free_space = free_space
		self.usage = usage