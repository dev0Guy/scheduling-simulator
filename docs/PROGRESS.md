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
