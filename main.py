import numpy as np
from scheduling_simulator.core.render import Renderer
from scheduling_simulator.core.creator import generate_cluster_python
from time import sleep
from typing import TYPE_CHECKING
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
if TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig

config: 'ClusterGenerationConfig'  = {
    'n_machines': 3,
    'n_jobs': 10,
    'n_resource': 3,
    'n_time': 10,
    'max_capacity': 255
}
env = SchedulingEnviorment(config)
env.reset()
env.render()
sleep(1)
env.step(0)
env.render()
sleep(1)
env.step(1)
env.render()
sleep(2)
for _ in range(6):
    env.step(0)
    env.render()
    sleep(1)

env.step(0)
env.render()

# cluster = generate_cluster_python(config, np.random.default_rng(10))
# renderer = Renderer(True)


# obs = cluster.step(0)
# screen = renderer.render(obs)
# sleep(5)
# print(screen)
