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

cdef deeprm_generate_single_job(ClusterGenerationConfig config, object random):
    cdef:
        int dominant_resource = random.integers(0, config.n_resource)
        int resource_idx
        int[:, ::1] usage = np.zeros((config.n_resource, config.n_time), dtype=np.int32)
        int size = random.integers(1, config.n_time + 1)
        int arrival_time = 0#random.integers(0, config.n_time)
        int dominant_low = max(1, config.max_capacity // 4)
        int dominant_high = max(dominant_low + 1, (3 * config.max_capacity) // 4)
        int secondary_low = max(1, config.max_capacity // 20)
        int secondary_high = max(secondary_low + 1, config.max_capacity // 4)

    for resource_idx in range(config.n_resource):
        usage[resource_idx, :size] = 255
        # if resource_idx == dominant_resource:
        #     usage[resource_idx, :size] = random.integers(
        #         dominant_low, dominant_high, dtype=np.int32
        #     )
        # else:
        #     usage[resource_idx, :size] = random.integers(
        #         secondary_low, secondary_high, dtype=np.int32
        #     )

    return Job(usage=usage, arrival_time=arrival_time, size=size)


cdef list[Job] deeprm_capcity_jobs(ClusterGenerationConfig config, object random):
    cdef:
        unsigned int job_idx, resource_idx,
        Job job
        list[Job] jobs
        unsigned int size = 5

    jobs = [
        deeprm_generate_single_job(config, random)
        for job_idx in range(config.n_jobs)
    ]
    return jobs


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
