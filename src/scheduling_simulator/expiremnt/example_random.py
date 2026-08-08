"""
Entry point for evaluating a pure random scheduling baseline
(no training) against the cluster scheduling simulator.
"""

from scheduling_simulator.expiremnt.scheduler_runner import RandomBaselineRunner
from typing import TYPE_CHECKING

if  TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig

def main() -> None:
    config: 'ClusterGenerationConfig' = {
        'n_machines': 1,
        'n_jobs': 10,
        'n_resource': 1,
        'n_time': 1,
        'max_capacity': 255
    }
    runner = RandomBaselineRunner(
        config=config,
        evalution_steps=100,
        max_time=250,
        seed=42,
    )
    runner.run()


if __name__ == "__main__":
    main()
