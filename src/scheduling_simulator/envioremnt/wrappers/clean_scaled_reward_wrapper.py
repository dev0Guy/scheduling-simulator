"""Training-only reward cleaning and scaling.

* drop_constants: removes the two constant terms of the wrapper reward, which are identical for every
  policy: +1 for each successful start (there are always n_jobs of them) and the +100 bonus when all
  jobs have finished. What remains is minus the number of active jobs per skipped tick, i.e. minus the
  flow time, plus the -100 penalty for hitting the step limit (kept: it punishes failure).
* scale: divides the reward by a constant so returns and value targets are O(1) instead of O(100),
  which balances the critic's loss against the actor's in the shared feature extractor.

Neither changes which policy is optimal. Use it only on the TRAINING environment; evaluation keeps the
original reward. It sits outside Monitor, so `rollout/ep_rew_mean` still logs the original reward.
"""
import numpy as np
from gymnasium import Wrapper

from scheduling_simulator.core.job import JobStatus


class CleanScaledRewardWrapper(Wrapper):

    def __init__(self, env, scale: float = 100.0, drop_constants: bool = True):
        super().__init__(env)
        self._scale = float(scale)
        self._drop = drop_constants

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = float(reward)
        if self._drop:
            if (terminated or truncated) and np.all(np.asarray(obs["status"]) == int(JobStatus.COMPLETED)):
                r -= 100.0                                   # completion bonus
            elif not (terminated or truncated) and int(action) > 0 and bool(obs["action_success"]):
                r -= 1.0                                     # successful start
        info = dict(info)
        info["original_reward"] = float(reward)
        return obs, r / self._scale, terminated, truncated, info
