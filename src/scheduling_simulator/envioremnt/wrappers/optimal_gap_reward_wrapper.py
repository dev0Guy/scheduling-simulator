"""Training-only reward: how much worse the chosen action is than the optimal one, from the same state.

With V*(s) = optimal value of state s under the cleaned reward (minus the optimal remaining
flow time, from the exact solver) the reward of a step is

    r' = r(s, a) + V*(s') - V*(s)  =  [ r(s, a) + V*(s') ]  -  V*(s)

The first bracket is "the chosen action's reward plus the rest of the trace played by the optimal
solver"; V*(s) is "the optimal action's reward plus the rest of the optimal trace". So r' is 0 when the
action is optimal and negative by the damage otherwise (undiscounted, so r' = Q*(s,a) - V*(s) exactly).

r is the cleaned step reward (minus the number of pending + running jobs per skipped tick; the
constant +1 per start and the +100 completion bonus are removed because they are identical for every
policy; the -100 penalty for hitting the step limit is kept). The sum of r' over an episode equals
(the cleaned return) - V*(s0) = minus the regret of the whole episode against the optimum.

Use it only on the TRAINING environment; evaluation keeps the original reward. It sits outside Monitor,
so `rollout/ep_rew_mean` still logs the original reward.
"""
import numpy as np
from gymnasium import Wrapper

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.scheduler.optimal_state_solver import OptimalStateSolver


class OptimalGapRewardWrapper(Wrapper):

    def __init__(self, env, capacity: int = 255, scale: float = 1.0):
        super().__init__(env)
        self._capacity = capacity
        self._scale = float(scale)
        self._solver = None
        self._v = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._solver = OptimalStateSolver.from_obs(obs, self._capacity)
        self._v = -float(self._solver.remaining_penalty(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = float(reward)
        done = terminated or truncated
        if done and np.all(np.asarray(obs["status"]) == int(JobStatus.COMPLETED)):
            r -= 100.0                                       # completion bonus (constant)
        elif not done and int(action) > 0 and bool(obs["action_success"]):
            r -= 1.0                                         # successful start (constant)
        v_next = -float(self._solver.remaining_penalty(obs))   # 0 when every job has finished
        shaped = r + v_next - self._v
        self._v = v_next
        info = dict(info)
        info["original_reward"] = float(reward)
        return obs, shaped / self._scale, terminated, truncated, info
