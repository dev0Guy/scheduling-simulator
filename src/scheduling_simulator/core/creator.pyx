from .cluster cimport Cluster
from .machine cimport Machine
from .job cimport Job
import os
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

    # Per-resource usage: every resource gets its own value, drawn
    # independently from the same three levels, so any resources of a job
    # can be equal or different.
    # With JOB_USAGE_MODE=random every resource of a job instead gets a uniformly random
    # integer in [1, max_capacity] (no fixed set of levels). Default: the three levels.
    possible_options = [64, 128, 255]
    random_usage = os.environ.get("JOB_USAGE_MODE") == "random"
    for resource_idx in range(config.n_resource):
        if random_usage:
            usage[resource_idx, :size] = random.integers(1, config.max_capacity + 1)
        else:
            usage[resource_idx, :size] = random.choice(possible_options)



    return Job(usage=usage, arrival_time=arrival_time, size=size)


cdef list[Job] deeprm_paper_jobs(ClusterGenerationConfig config, object random):
    """Workload of the DeepRM paper (Mao et al., HotNets 2016, section 4.1), JOB_GEN_MODE=deeprm.
    Duration: 80% uniform integer in [1, 3], 20% uniform in [10, 15] (capped at n_time).
    Each job has one dominant resource (picked uniformly at random): its demand is uniform in [0.25, 0.5] of the capacity,
    the demand of every other resource uniform in [0.05, 0.1]; constant over the job's duration.
    Arrivals: a Poisson process (in the paper a Bernoulli one, here several jobs can arrive in the same tick) whose rate is set
    so that the average load per machine and resource equals DEEPRM_LOAD (default 1.0 = 100% of the cluster capacity).
    The first job arrives at tick 0."""
    cdef:
        int idx, r, dom, size
        double cap = <double>config.max_capacity
        double load = float(os.environ.get("DEEPRM_LOAD", "1.0"))
        # demand ranges as fractions of the capacity (paper: dominant 0.25-0.5, others 0.05-0.1); DEEPRM_DEMAND_SCALE multiplies both
        double scale = float(os.environ.get("DEEPRM_DEMAND_SCALE", "1.0"))
        double dom_lo = 0.25 * scale, dom_hi = min(1.0, 0.5 * scale)
        double oth_lo = 0.05 * scale, oth_hi = min(1.0, 0.1 * scale)
        double mean_len = 0.8 * 2.0 + 0.2 * 12.5
        double mean_share = (0.5 * (dom_lo + dom_hi) + (config.n_resource - 1) * 0.5 * (oth_lo + oth_hi)) / config.n_resource
        double rate = load * config.n_machines / (mean_len * mean_share)
        double t = 0.0
        list[Job] jobs = []
        int[:, ::1] usage
        int arrival

    for idx in range(config.n_jobs):
        if idx > 0:
            t += random.exponential(1.0 / rate)
        arrival = <int>t
        if random.random() < 0.8:
            size = random.integers(1, 4)
        else:
            size = random.integers(10, 16)
        size = min(size, config.n_time)
        dom = random.integers(0, config.n_resource)
        usage = np.zeros((config.n_resource, config.n_time), dtype=np.int32)
        for r in range(config.n_resource):
            if r == dom:
                usage[r, :size] = <int>(cap * random.uniform(dom_lo, dom_hi))
            else:
                usage[r, :size] = max(1, <int>(cap * random.uniform(oth_lo, oth_hi)))
        jobs.append(Job(usage=usage, arrival_time=arrival, size=size))
    return jobs


cdef list[Job] deeprm_capcity_jobs(ClusterGenerationConfig config, object random):
    cdef:
        unsigned int job_idx
        list[Job] jobs

    if os.environ.get("JOB_GEN_MODE") == "deeprm":
        return deeprm_paper_jobs(config, random)

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
