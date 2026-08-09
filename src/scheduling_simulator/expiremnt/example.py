import typing as tp

from scheduling_simulator.expiremnt.train_runner import TrainExperimentRunner

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig


config: 'ClusterGenerationConfig' = {
    'n_machines': 2,
    'n_jobs': 32,
    'n_resource': 2,
    'n_time': 20,
    'max_capacity': 255,
}
runner = TrainExperimentRunner(config)
runner.run()
