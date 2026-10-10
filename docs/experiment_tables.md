# Experiment tables: runs from Oct 2-3

All runs: 1 machine, 10 jobs, 3 resources, 10-tick usage window, capacity 255, episode cap 110 steps.
Results are 5000 deterministic episodes with `seed = 42 + episode`, so each run sees the same job sets as its SJF baseline.
**Margin vs SJF** = (run − SJF) / |SJF| on mean reward; positive means better than SJF.

## 1. Environment job-creation settings

In both settings `arrival` and `size` come from separately jittered copies of a job-order "progress" value (size 1-10 ticks, arrivals in the first ~10 ticks), and a job's usage is one constant value applied to every resource and every tick of its run.

| ID | Job usage | Machine behavior | SJF baseline |
|---|---|---|---|
| **E1** | always 255 | one job at a time | `mkl0zwp7` (run Sep 30) |
| **E4** | one of **63, 128, 255**, chosen at random per job | 63+63+128 and 63+128 fit; 128+128 does not; 255 blocks the machine | `rjzcixv2` |

## 2. Model architectures

PPO settings for all models: learning rate 7e-6, 2048 steps per rollout, batch 64, 5 epochs, γ = λ = 0.995, clip 0.2, value coef 0.5, grad clip 0.5, entropy coef annealed 0.02 → 0.002, embedding size 256, 2M steps.

| ID | Per-job inputs | Cross-job mechanism | Actor scorer |
|---|---|---|---|
| **M0** | status (16-d), size, wait, ttl, arrival (20-d) | attention pooling only | static machine embedding; `[global, machine, job]` → 768 → 128 → 128 → 1 |
| **M5** | M0 + job `max(usage)` / 255 (21-d) | attention pooling only | M0 scorer + job usage + machine `min(capacity − usage)` / 255 (770 inputs) |
| **M6** | M0 inputs, 20-d, no usage | attention pooling + **fit attention**: job `max(usage)/255` (query) vs machine `min(capacity − usage)/255` (key, value), sigmoid weights, 64-d output joined to each job summary and projected to 256 | M0 scorer, 768 inputs |
| **M7** | same as M6 | M6 + **job self-attention** (4 heads, finished jobs masked) after the fit step, before pooling | M0 scorer, 768 inputs |
| **M8** | same as M7 | M7 with the fit output **multiplying** the job summary per channel (`summary × (1 + tanh(Linear(fit)))`) instead of concatenate + project | M0 scorer, 768 inputs |
| **M9** (current) | same as M7 (concat + project, rolled back from M8) | M7 + **feasibility mask** in the fit attention: job-machine pairs with job usage > machine room get weight 0 | M0 scorer, 768 inputs |
| **M10** | same as M9 | M9 trained with a higher entropy coefficient: annealed 0.05 → 0.01 (was 0.02 → 0.002), to slow the early collapse of exploration | M0 scorer, 768 inputs |

Critic for all: `256 → 256 → 256 → 1` on the global embedding. Skip logit: `256 → 64 → 1`.

## 3. Runs and results

### E1: every job uses 255

| Run ID | Model | Mean reward | Margin vs SJF | Wait | Completion |
|---|---|---|---|---|---|
| `mkl0zwp7` | SJF baseline | −133.15 | 0% | 18.39 | 24.60 |
| `6b195fqa` | M0 (rebuilt `np7aenua`) | −131.84 | +0.98% | 18.26 | 24.47 |

The optimal for E1 (exact DP, `scripts/solvers/run_optimal.py`) is −131.08 / +1.56% / 18.19 / 24.39. It was run Oct 1.

### E4: usage 63 / 128 / 255

| Run ID | Model | Mean reward | Margin vs SJF | Wait | Completion |
|---|---|---|---|---|---|
| (script) | **Optimal, exact search** (`scripts/solvers/run_optimal_packing.py`) | −44.41 | +11.9% | 9.49 | 15.73 |
| `rjzcixv2` | SJF baseline | −50.43 | 0% | 10.09 | 16.33 |
| `cf2zhwti` | M0 (cannot see usage) | −133.44 | −164.6% | 18.39 | 24.63 |
| `z4sayojp` | M5 (job usage + machine free as raw scorer inputs) | −73.53 | −45.81% | 12.38 | 18.62 |
| `iusicvre` | M6 (fit attention, usage not in encoder) | −69.07 | −36.98% | 11.60 | 17.83 |
| `emt46quc` | **M7** (M6 + job self-attention) | −58.94 | −16.87% | 10.83 | 17.07 |
| `bv1hk6lb` | **M8** (M7, fit output multiplies job summary) | −63.20 | −25.3% | 11.36 | 17.60 |
| `etudt4qo` | **M9** (M7 + feasibility mask in fit attention) | −49.28 | +2.28% | 9.97 | 16.20 |
| `ne5qv1d2` | **M10** (M9 + entropy coef 0.05 → 0.01) | −49.27 | +2.31% | 9.96 | 16.20 |

PPO wins / SJF wins / ties against `rjzcixv2` over 5000 episodes: `cf2zhwti` 56 / 4837 / 107, `z4sayojp` 266 / 4528 / 206, `iusicvre` 497 / 4125 / 378, `emt46quc` 855 / 3105 / 1040, `bv1hk6lb` 483 / 3841 / 676, `etudt4qo` 800 / 662 / 3538, `ne5qv1d2` 788 / 730 / 3482.

## 4. Notes

- The optimal solver for E4 allows several jobs to share the machine. It matches a brute-force search on 150 small instances, and its predicted wait times match the environment's on every one of the 5000 episodes.
- `cf2zhwti` behaves like the one-job-at-a-time policy because it never sees usage: its numbers on E4 are almost identical to M0's on E1.
- A crashed attempt, `iw2n4ky5` (disk full during evaluation), has no results; `6b195fqa` is its rerun.
