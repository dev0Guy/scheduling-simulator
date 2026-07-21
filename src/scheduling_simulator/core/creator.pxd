from .cluster cimport Cluster


cdef struct ClusterGenerationConfig:
    unsigned int n_machines
    unsigned int n_jobs
    unsigned int n_resource
    unsigned int n_time
    unsigned int max_capacity


ctypedef list[Machine](*machines_creator_t)(ClusterGenerationConfig config, object random)
ctypedef list[Job](*jobs_creator_t)(ClusterGenerationConfig config, object random)


cdef Cluster generate_cluster(
    ClusterGenerationConfig config,
    object random,
    machines_creator_t machine_creator =*,
    jobs_creator_t job_creator =*,
)

cpdef Cluster generate_cluster_python(
    ClusterGenerationConfig config,
    object random =*
)
