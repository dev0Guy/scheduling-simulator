# Masked PPO vs SJF across cluster sizes

Masked PPO (in-policy mask), original reward, random per-resource usage (`JOB_USAGE_MODE=random`), 3 resources,
seed 3, 2M steps, final model, 5000 evaluation episodes per cell (job sets seeded 42+episode, same as SJF).
Script: `example_masked_scale.py` (env vars `N_MACHINES`, `N_JOBS`); SJF: `scripts/analysis/sjf_baseline.py`.
Each cell: wins / ties / losses vs SJF by reward, mean reward (SJF in brackets), gain over SJF (95% CI).

| machines \ jobs | 10 | 20 | 30 | 40* | 50* |
|---|---|---|---|---|---|
| 1 | 1236/3061/703; -92.85 (-94.94); +2.09±0.26 | 1942/1730/1328; -589.8 (-593.4); +3.61±0.56 | 2526/705/1769; -1377.2 (-1383.9); +6.64±1.17 | 449/62/461; -2467.7 (-2462.6); -5.1±4.1 (972 eps) | 96/0/250; -3879.2 (-3809.8); -69.4±12.4 (346 eps) |
| 2 | 1396/2900/704; -3.36 (-4.92); +1.56±0.15 | 1785/1896/1319; -252.2 (-253.8); +1.64±0.32 | 2309/894/1797; -643.3 (-646.2); +2.94±0.57 | training | training |
| 3 | 1926/2216/858; +25.02 (+23.64); +1.38±0.11 | 2307/1083/1610; -142.8 (-144.9); +2.04±0.29 | 2333/915/1752; -402.5 (-405.0); +2.42±0.40 | training | training |
| 4 | 757/4019/224; +37.51 (+37.19); +0.32±0.04 | 2532/1114/1354; -89.4 (-92.2); +2.80±0.22 | 2025/1291/1684; -287.6 (-288.3); +0.72±0.26 | training | training |

\* 40/50 jobs: partial evaluation at the time of writing (episode counts shown); 2-4 machine runs were still training.

Mean completion time, PPO (SJF): 10 jobs: 20.570 (20.779), 11.622 (11.778), 8.784 (8.922), 7.535 (7.567);
20 jobs: 35.775 (35.956), 18.894 (18.976), 13.425 (13.527), 10.753 (10.893);
30 jobs: 50.528 (50.749), 26.062 (26.160), 18.038 (18.118), 14.206 (14.230) (rows = 1..4 machines).

Also: sb3-contrib MaskablePPO, 1 machine, 10 jobs, 1.5M steps: -93.01 vs SJF -94.94, optimal -86.79; gain +1.93±0.26,
closes 23.7% of the SJF-to-optimal gap (in-policy mask: ~25%).

Caveats: one seed per cell; no optimal comparison beyond 1 machine x 10 jobs (exact solver too slow).
