# 3-resource environment: optimal vs PPO vs SJF

Job usage: all three resources of each job drawn independently from 64 / 128 / 255 (none zero).
5000 deterministic episodes (seed 42 + episode), the same job sets for all three.
Margin vs SJF = (X − SJF) / |SJF|. For wait and completion time, a negative margin means faster than SJF.

| | Optimal (exact search) | PPO `ob8g1j1u` | SJF `f4vkih6o` |
|---|---|---|---|
| Mean reward | −113.44 | −129.10 | −118.57 |
| Mean wait time | 16.40 | 17.97 | 16.92 |
| Mean completion time | 22.63 | 24.19 | 23.14 |
| Reward margin vs SJF | +4.32% | −8.88% | 0% |
| Wait-time margin vs SJF | −3.03% | +6.20% | 0% |
| Completion-time margin vs SJF | −2.22% | +4.54% | 0% |
| Episodes equal to the optimal reward | 100% | 45.14% (2257) | 55.44% (2772) |
| Failed starts per episode | 0 | 0.0032 | 0 |

| Pair | Wins / losses / ties (reward per episode) |
|---|---|
| PPO vs SJF | 601 / 2207 / 2192 |
| PPO vs optimal | 0 / 2743 / 2257 |
| SJF vs optimal | 0 / 2228 / 2772 |

The optimal is 5.13 reward points per episode better than SJF (95% CI 4.89 to 5.36) and 15.65 better than PPO (95% CI 15.08 to 16.23).

Optimal solver: `scripts/solvers/optimal_solver_multires.py` (300/300 random small instances with three independent resources match brute force), replayed through the real environment by `scripts/solvers/run_optimal_multires.py` (0 failed starts, 0 episodes where the predicted flow time differed from the measured one).
