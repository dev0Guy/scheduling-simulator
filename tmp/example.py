import numpy as np
from core import Job

usage = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)


job = Job(usage, arrival_time=0, size=3)
print(job)
job.forward_time(1)
print(job)
job.update_status(2, 1)
print(job)
job.forward_time(1)
print(job)
