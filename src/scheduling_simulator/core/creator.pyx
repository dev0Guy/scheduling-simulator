from .cluster cimport Cluster
from .machine cimport Machine
from .job cimport Job
import numpy as np


cdef list[Machine] max_capacity_machines(ClusterGenerationConfig config, object random):
    cdef list[Machine] machines
    capacity = np.full((config.n_resource, config.n_time), config.max_capacity, dtype=np.int32)
    machines = [Machine(capacity) for machine in range(config.n_machines)]
    return machines

cdef list[Job] half_capcity_jobs(ClusterGenerationConfig config, object random):
    cdef list[Job] jobs
    usage = np.full((config.n_resource, config.n_time), config.max_capacity // 2, dtype=np.int32)
    size = 5
    usage[:, size:] = 0
    jobs = [
        Job(usage=usage, arrival_time=0, size=size)
        for job in range(config.n_jobs)
    ]
    return jobs

# cdef deeprm_generate_single_job(ClusterGenerationConfig config, object random):
#     cdef:
#         int dominant_resource = random.integers(0, config.n_resource)
#         int resource_idx
#         int[:, ::1] usage = np.zeros((config.n_resource, config.n_time), dtype=np.int32)
#         int size = random.integers(1, config.n_time + 1)
#         int arrival_time = random.integers(0, config.n_time) # TODO: remove or change to zero
#         int dominant_low = max(1, config.max_capacity // 4)
#         int dominant_high = max(dominant_low + 1, (3 * config.max_capacity) // 4)
#         int secondary_low = max(1, config.max_capacity // 20)
#         int secondary_high = max(secondary_low + 1, config.max_capacity // 4)

#     for resource_idx in range(config.n_resource):
#         usage[resource_idx, :size] = 255
#         # if resource_idx == dominant_resource:
#         #     usage[resource_idx, :size] = random.integers(
#         #         dominant_low, dominant_high, dtype=np.int32
#         #     )
#         # else:
#         #     usage[resource_idx, :size] = random.integers(
#         #         secondary_low, secondary_high, dtype=np.int32
#         #     )

#     return Job(usage=usage, arrival_time=arrival_time, size=size)



cdef deeprm_generate_single_job_decreasing_size(ClusterGenerationConfig config, object random, unsigned int job_index, unsigned int n_jobs):
    cdef:
        int dominant_resource = random.integers(0, config.n_resource)
        int resource_idx
        int[:, ::1] usage = np.zeros((config.n_resource, config.n_time), dtype=np.int32)
        int size
        int arrival_time
        double progress
        double arrival_progress
        double size_progress
        int max_size = config.n_time
        int min_size = 1
        int low
        int high
        int utilization

    # progress still trends 0 -> 1 across job_index (so job order remains
    # a meaningful, learnable signal). arrival_time and size each get
    # their OWN independently-jittered copy of progress (rather than
    # sharing one jittered value) so the two are no longer near-perfectly
    # anti-correlated -- that near-perfect correlation is exactly what
    # let a plain SJF heuristic match/beat the trained policy for free,
    # since "arrived first" and "smallest" stopped being reliably the
    # same job.
    progress = job_index / max(1.0, <double>(n_jobs - 1))

    arrival_progress = min(1.0, max(0.0, progress + random.uniform(-0.35, 0.35)))
    size_progress = min(1.0, max(0.0, progress + random.uniform(-0.35, 0.35)))

    arrival_time = <int>(arrival_progress * (config.n_time // 2))
    # add independent small randomness to arrival_time too, so identical
    # progress values don't always land on the identical tick
    arrival_time = max(0, arrival_time + random.integers(-3, 4))  # +/- 3 tick jitter

    size = <int>(max_size - size_progress * (max_size - min_size))
    # independent randomness on size as well, so size isn't a perfectly
    # deterministic function of arrival_time either
    size = size + random.integers(-5, 6)  # +/- 5 jitter
    size = max(1, min(config.n_time, size))

    # Per-resource usage: every resource except the last one gets its own
    # value, drawn independently from the three levels, so two resources
    # can be equal or different. The last resource is never used (always
    # 0), so only the first n_resource - 1 resources limit what fits.
    possible_options = [64, 128, 255]
    for resource_idx in range(config.n_resource - 1):
        usage[resource_idx, :size] = random.choice(possible_options)
    # usage[config.n_resource - 1, :] stays 0



    return Job(usage=usage, arrival_time=arrival_time, size=size)


cdef list[Job] deeprm_capcity_jobs(ClusterGenerationConfig config, object random):
    cdef:
        unsigned int job_idx
        list[Job] jobs

    jobs = [
        deeprm_generate_single_job_decreasing_size(config, random, job_idx, config.n_jobs)
        for job_idx in range(config.n_jobs)
    ]
    return jobs


# cdef list[Job] deeprm_capcity_jobs(ClusterGenerationConfig config, object random):
#     cdef:
#         unsigned int job_idx, resource_idx,
#         Job job
#         list[Job] jobs
#         unsigned int size = 5

#     jobs = [
#         deeprm_generate_single_job(config, random)
#         for job_idx in range(config.n_jobs)
#     ]
#     return jobs


cdef Cluster generate_cluster(
    ClusterGenerationConfig config,
    object random,
    machines_creator_t machine_creator = max_capacity_machines,
    jobs_creator_t job_creator = deeprm_capcity_jobs,
):
    cdef:
        list[Machine] machines
        list[Job] jobs

    machines = machine_creator(config, random)
    jobs = job_creator(config, random)

    return Cluster(machines, jobs)



cpdef Cluster generate_cluster_python(ClusterGenerationConfig config, object random = np.random.default_rng()):
    return generate_cluster(config, random)
