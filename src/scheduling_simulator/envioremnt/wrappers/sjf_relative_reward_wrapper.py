"""SJF-relative reward (potential-based shaping).

Let V(s) be the discounted return that Shortest-Job-First would collect from
state s under the wrapper reward of `FailureSkipTimeWrapper`. The shaped reward is

    r' = r + gamma * V(s') - V(s)          (V = 0 at the end of the episode)

It is the one-step advantage of the chosen action over letting SJF decide: an
action that SJF would also take gets 0, a better action a positive reward and a
worse one a negative reward, immediately instead of many ticks later. Because it
is potential-based shaping, the best policy is the same as with the original
reward (Ng, Harada and Russell, 1999).

`sjf_return_to_go` re-implements the environment dynamics exactly (job states,
the machine's shifting usage grid, the wrapper rewards and the step limit) for a
single SJF policy, using the same tie-breaking as `ShortestJobScheduler`
(smallest size, then lowest job index, then lowest machine index).

Use it only on the TRAINING environment; evaluation must keep the original reward.
"""
import numpy as np
from gymnasium import Wrapper

from scheduling_simulator.core.job import JobStatus

_NOT_CREATED = int(JobStatus.NOT_CREATED)
_PENDING = int(JobStatus.PENDING)
_RUNNING = int(JobStatus.RUNNING)
_COMPLETED = int(JobStatus.COMPLETED)


def sjf_return_to_go(obs, steps_taken: int, max_time: int = 110, gamma: float = 0.995) -> float:
    """Discounted return SJF would collect from `obs` (wrapper reward, step limit `max_time`)."""
    status = [int(x) for x in obs["status"]]
    n = len(status)
    size = [int(round(float(x))) for x in obs["size"]]
    ttl = [int(round(float(x))) for x in obs["ttl"]]
    arrival = [int(round(float(x))) for x in obs["arrival"]]
    t = int(round(float(np.asarray(obs["time"]).reshape(-1)[0])))

    cap = np.asarray(obs["machines_capacity"], dtype=np.int64)               # (m, R, T)
    free = cap - np.asarray(obs["machines_usage"], dtype=np.int64)            # (m, R, T)
    usage = np.asarray(obs["jobs_usage"], dtype=np.int64)[:, :, 0]            # (n, R): value of the first tick
    n_machines = cap.shape[0]

    total, disc, steps = 0.0, 1.0, int(steps_taken)
    for _ in range(2000):                                                     # safety bound
        # SJF: among pending jobs that fit now, the smallest size (first in job, machine order)
        best_job, best_machine = -1, -1
        for j in range(n):
            if status[j] != _PENDING:
                continue
            for k in range(n_machines):
                if (free[k, :, : size[j]] >= usage[j][:, None]).all():
                    if best_job < 0 or size[j] < size[best_job]:
                        best_job, best_machine = j, k
                    break

        steps += 1
        if best_job >= 0:                                                     # start a job: time does not advance
            free[best_machine, :, : size[best_job]] -= usage[best_job][:, None]
            status[best_job] = _RUNNING
            reward = 1.0
        else:                                                                 # skip: one tick passes
            t += 1
            for j in range(n):
                if status[j] == _PENDING:
                    pass
                elif status[j] == _RUNNING:
                    if ttl[j] == 1:
                        status[j] = _COMPLETED
                    ttl[j] -= 1
                elif status[j] == _NOT_CREATED and t == arrival[j]:
                    status[j] = _PENDING
            free[:, :, :-1] = free[:, :, 1:].copy()
            free[:, :, -1] = cap[:, :, -1]
            reward = -float(sum(1 for s in status if s in (_PENDING, _RUNNING)))

        done = False
        if all(s == _COMPLETED for s in status):                              # wrapper terminal overrides
            reward, done = 100.0, True
        elif steps >= max_time:
            reward, done = -100.0, True

        total += disc * reward
        disc *= gamma
        if done:
            break
    return total


class SJFRelativeRewardWrapper(Wrapper):
    """Wrap a `FailureSkipTimeWrapper` and replace its reward by r + gamma*V(s') - V(s)."""

    def __init__(self, env, gamma: float = 0.995, max_time: int = 110):
        super().__init__(env)
        self._gamma = gamma
        self._max_time = max_time
        self._v = 0.0

    def _value(self, obs) -> float:
        return sjf_return_to_go(obs, self.env._time_counter, self._max_time, self._gamma)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._v = self._value(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        v_next = 0.0 if done else self._value(obs)
        shaped = float(reward) + self._gamma * v_next - self._v
        self._v = v_next
        info = dict(info)
        info["original_reward"] = float(reward)
        return obs, shaped, terminated, truncated, info
