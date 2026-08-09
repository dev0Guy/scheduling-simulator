# Progress Tracker

## Status 
- [x] E1-E9 completed (see RESEARCH_NOTES.md)
- [ ] P0: Multi-seed eval on 2m/2r (running, PID 42152)
- [x] P1: Baselines (SJF/WSPT strongest, learned beats by 5-9%)
- [ ] P2: 24-job regression diagnosis (running, PID 42174)
- [ ] P3: Decima-style pointer network (requires custom training loop)
- [ ] P4: Paper-ready docs

## Running Jobs
- .cache/p0_out.log — multi-seed 2m/2r
- .cache/p2_out.log — small-job skew diagnosis

## P2 Result: 24-job regression diagnosed

Root cause: under-exposure to small job counts during training.
The uniform training distribution [20, 24, 28, 32, 36] drew each size
equally, but small sizes have lower per-episode reward signal.

Fix: skew training distribution (80% small [20,24], 20% large).
Result: 24-job regression disappears (474.9 -> 439.7, SJF=441.2),
maintains gains at 32/40/44.

## P1 Result: Baselines

SJF/WSPT is strongest simple heuristic. Learned beats by 5-9%.

## P3 Result: Two-stage pointer policy

Implemented a Decima-style two-stage pointer policy with a custom compact
PPO trainer (job stage + machine stage, padding/masking).

Findings:
- Custom pipeline works on smoke test (8 jobs, reward improves).
- On 2m/2r it is unstable and much slower to evaluate: no vectorized
  rollout, episodes often truncate at 500 steps, and the custom ratio
  aggregation (sum of stage ratios) is not theoretically sound.
- Not competitive with single-stage SB3 pointer. Requires a proper
  implementation: product of ratios, full-state machine conditioning,
  entropy scheduling, vectorized eval. Marked as future work.

Conclusion: single-stage pointer + SB3 remains the production approach.
BOPO integration on SB3 is the higher-value next step.

## BOPO on SB3 (in progress)

Implemented BOPOCallback: periodically collects SJF/random/learned
trajectories, builds Bradley-Terry preference pairs, applies a
differentiable preference loss after PPO updates.

Quick benchmark: PPO-only vs PPO+BOPO on 2m/2r, 50k steps each,
comparing flow on [24, 32, 40, 44]. Results in .cache/bopo_result.log.
