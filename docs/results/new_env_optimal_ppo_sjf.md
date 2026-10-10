# New environment: optimal vs PPO vs SJF

Job usage: resources 0 and 1 drawn independently from 64 / 128 / 255, last resource always 0.
5000 deterministic episodes (seed 42 + episode), the same job sets for all three.
Margin vs SJF = (X − SJF) / |SJF|. For wait and completion time, a negative margin means faster than SJF.

| | Optimal (exact search) | PPO `vy9asra5` | SJF `xd83r7lf` |
|---|---|---|---|
| Mean reward | −93.92 | −98.84 | −101.25 |
| Mean wait time | 14.44 | 14.93 | 15.17 |
| Mean completion time | 20.68 | 21.17 | 21.41 |
| Reward margin vs SJF | +7.24% | +2.37% | 0% |
| Wait-time margin vs SJF | −4.83% | −1.60% | 0% |
| Completion-time margin vs SJF | −3.42% | −1.13% | 0% |
| Episodes equal to the optimal reward | 100% | 38.86% (1943) | 37.38% (1869) |
| Failed starts per episode | 0 | 0.0018 | 0 |

| Pair | Wins / losses / ties (reward per episode) |
|---|---|
| PPO vs SJF | 1329 / 824 / 2847 |
| PPO vs optimal | 0 / 3057 / 1943 |
| SJF vs optimal | 0 / 3131 / 1869 |

PPO closes 32.8% of the reward gap between SJF and the optimal. The optimal is 4.93 reward points per episode better than PPO (95% CI 4.73 to 5.13).

Optimal solver: `scripts/solvers/optimal_solver_multires.py` (per-resource usage; 300/300 random small instances match brute force), replayed through the real environment by `scripts/solvers/run_optimal_multires.py` (0 failed starts, 0 episodes where the predicted flow time differed from the measured one).
