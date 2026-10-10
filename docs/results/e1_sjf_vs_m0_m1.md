# E1: SJF vs M0 and M1

E1 environment (every job uses 255). 5000 deterministic episodes with the same job sets for all three.
M0 and M1 are the rebuilt versions of `np7aenua` and `g231w4jg`.
Margin = (PPO − SJF) / |SJF| on the mean of each metric. For wait and completion time, a negative margin means PPO is faster than SJF.
Ties, PPO wins and SJF wins compare the reward in each episode.

| Measure | SJF | M0 | M1 |
|---|---|---|---|
| Run | `mkl0zwp7` | `6b195fqa` | `khz1l51q` |
| Mean reward | −133.15 | −131.84 | −133.35 |
| Mean wait time | 18.39 | 18.26 | 18.24 |
| Mean completion time | 24.60 | 24.47 | 24.45 |
| Ties | n/a | 78.10% (3905) | 81.14% (4057) |
| PPO wins | n/a | 15.52% (776) | 9.60% (480) |
| SJF wins | n/a | 6.38% (319) | 9.26% (463) |
| Mean reward margin | 0% | +0.98% | −0.15% |
| Wait-time margin | 0% | −0.71% | −0.80% |
| Completion-time margin | 0% | −0.53% | −0.60% |
