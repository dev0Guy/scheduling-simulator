"""
Entry point for evaluating a pure random scheduling baseline
(no training) against the cluster scheduling simulator.
"""

from scheduling_simulator.expiremnt.runners.random_runner import RandomBaselineRunner
from typing import TYPE_CHECKING

if  TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig

def main() -> None:
    config: 'ClusterGenerationConfig' = {
        'n_machines': 1,
        'n_jobs': 10,
        'n_resource': 3,
        'n_time': 10,
        'max_capacity': 255,
    }
    runner = RandomBaselineRunner(
        config=config,
        evalution_steps=5000,
        max_time=110,
        seed=42,
    )
    runner.run()


if __name__ == "__main__":
    main()
