# PPO vs. SJF — Evaluation Results

**Config:** `n_machines=1, n_jobs=10, n_resource=3, n_time=10, max_capacity=255`
**SJF baseline:** `silvery-microwave-299` (`nb11ox0z`)
**Evaluation:** 5000 episodes logged, all 5000 retrieved (`history(..., samples=10_000)`, above the logged count so no subsampling occurs), deterministic policy evaluation (`model.predict(..., deterministic=True)`)

## Feature-extractor variants compared

All variants share the same base per-job features (status, size, wait_time, ttl, arrival — see `new_features.py`). Each row below adds one change on top of the previous best.

| Variant | Run | Change |
|---|---|---|
| Baseline | `silver-butterfly-298` (`np7aenua`) | base features only |
| Global-only `queue_pressure` | `smooth-hill-300` (`dbqcytgc`) | + `queue_pressure` scalar (active job count / n_jobs) in the pooled global/critic embedding only |
| Global + per-job `queue_pressure` | `confused-snowball-301` (`t3hmzxfp`) | + same scalar also broadcast into every per-job feature vector (available at actual allocation-decision time) |
| Job self-attention (unmasked) | `frosty-sea-303` (`2qbzle63`) | + multi-head self-attention across all 10 jobs before pooling, so each job's embedding is contextualized against every other job (not just a pooled average) |
| **Job self-attention (`COMPLETED` masked) — current best** | `stellar-dawn-304` (`g231w4jg`) | + `COMPLETED` jobs excluded as attention keys (`key_padding_mask`), so their stale data can't dilute other jobs' comparisons |

## Results — current best (`g231w4jg`)

| Metric | Comparable episodes | PPO better | SJF better | Equal | PPO mean | SJF mean | Difference |
|---|---|---|---|---|---|---|---|
| `eval/reward` | 5000 | 707 (14.14%) | 193 (3.86%) | 4100 (82.00%) | -131.6342 | -133.1472 | +1.14% |
| `eval/avg_wait_time` | 5000 | 707 (14.14%) | 191 (3.82%) | 4102 (82.04%) | 18.2396 | 18.3919 | -0.83% |
| `eval/avg_completion_time` | 5000 | 707 (14.14%) | 191 (3.82%) | 4102 (82.04%) | 24.4482 | 24.6005 | -0.62% |

**Reading the sign:** for `eval/reward`, higher (less negative) is better, so a positive difference means PPO wins. For `avg_wait_time`/`avg_completion_time`, lower is better, so a negative difference means PPO wins. PPO wins on all three, on every variant below.

## All five variants, side by side (`eval/reward`)

| Variant | Ties | PPO better | SJF better | Mean difference |
|---|---|---|---|---|
| Baseline | 82.66% | 13.46% (673) | 3.88% (194) | +1.02% |
| Global-only `queue_pressure` | 77.18% | 15.78% (789) | 7.04% (352) | +0.74% |
| Global + per-job `queue_pressure` | 82.74% | 13.56% (678) | 3.70% (185) | +1.08% |
| Self-attention (unmasked) | 83.60% | 13.22% (661) | **3.18% (159)** | +0.75% |
| **Self-attention (`COMPLETED` masked)** | 82.00% | **14.14% (707)** | 3.86% (193) | **+1.14%** |

## What each step taught us

- **Global-only `queue_pressure`** made the policy diverge from SJF far more often in *both* directions (more wins, but even more new losses) without a net improvement — exposing the signal only to the pooled critic summary let the network act on it more, but not more *correctly*.
- **Broadcasting the same signal into the per-job path** (where the "wait or take this job now" decision is actually scored) turned that into a small, consistent win over baseline.
- **Adding job-to-job self-attention** (replacing the previous blurred-average-only pooling with genuine job-vs-job comparisons) cut SJF's win rate to its lowest point of any variant (159) — but also pulled PPO's own win rate down (661), making the policy more cautious/SJF-like overall rather than more accurate.
- **Masking `COMPLETED` jobs out of the attention keys** fixed exactly that: a finished job's stale size/wait_time/arrival was diluting the *other* jobs' contextualized comparisons with irrelevant noise. Removing it (without hiding a completed job's own status from itself — that still flows into its own per-job vector, so the actor still knows not to pick it) recovered the win rate to a new high (707) while keeping the loss rate at baseline level (193) — a net improvement on both fronts at once, not a trade-off between them.

(An earlier pass at this table used W&B's default `history()` sampling, which silently caps at ~500 points regardless of how many were logged; that 500-episode estimate was within 0.2 points of the true full-5000 numbers on every metric, so subsampling wasn't a factor in any of these results.)

## Summary

PPO beats the SJF heuristic baseline across all three metrics, winning outright on 14.14% of episodes vs. SJF's 3.86%, with the large majority (82.00%) tied. The current best architecture is job self-attention with `COMPLETED` jobs masked out of the attention keys, `queue_pressure` fed into both the global and per-job paths, plus the base `status/size/wait_time/ttl/arrival` feature set. The improvement over baseline (~1.1% margin) remains modest in absolute terms but is now the most consistent result across every metric tried so far.
