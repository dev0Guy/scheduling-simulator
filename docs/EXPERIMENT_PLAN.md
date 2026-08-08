# Experiment Plan

## Completed Experiments

### E1: Bug Fixes and Baseline (Phase 1)
- Fixed 9 bugs in simulator, reward, training setup
- Original DQN baseline: flow time 0.0 (completely broken)
- Status: DONE

### E2: Attention Architecture (Phase 2)
- Self-attention over jobs + cross-attention to machines
- 8-job/1-machine: learned 76.0 vs SJF 76.5 (beats SJF, near-optimal)
- Status: DONE

### E3: Multi-Machine Generalization (Phase 3)
- 2 machines, train [12-24], test [12-32]: beats SJF by 3-5%
- 3 machines, train [16-32], test [16-40]: beats SJF by 0.3-6%
- 3 machines, train [16-40], test [20-48]: beats SJF by 0.5-8.2%
- Status: DONE

### E4: Decision Analysis
- Learned policy: better load balancing, delays long jobs, arrival-aware
- Status: DONE

### E5: BOPO Preference Training
- Bradley-Terry pairwise loss, SJF vs random pairs
- Improves unseen-size generalization by ~1pp (5-7% -> 6-7%)
- Slightly hurts at training size
- Status: DONE (rough implementation, needs refinement)

## Planned Experiments

### E6: Multi-Seed Evaluation
- Run 3-5 training seeds on best config (3 machines, train [16-40])
- Report mean ± std of learned-vs-SJF gap
- Priority: HIGH (required for paper credibility)

### E7: Refined BOPO Training
- Larger preference pair collection (500+ pairs)
- More epochs (50+), larger batch size (128)
- Combine with PPO in alternating steps (not separate phases)
- Add learned-vs-learned pairs from different training stages
- Priority: MEDIUM

### E8: Architecture for Better Generalization
- Option A: Increase attention layers (2-3 stacked, residual)
- Option B: Add positional encoding for job arrival time
- Option C: Pointer network with two-stage selection (Decima-style)
- Option D: Heterogeneous attention (separate relation types for
  job-job, job-machine, machine-machine interactions)
- Priority: MEDIUM

### E9: Multi-Resource Workload
- 2+ resources per machine, creating packing trade-offs
- Jobs with different dominant resources (CPU-heavy vs memory-heavy)
- This is where SJF is clearly suboptimal and the gap should be largest
- Priority: HIGH

### E10: Scaling Study
- 4-8 machines, 48-64 jobs
- Measure how the learned-vs-SJF gap scales with problem complexity
- Priority: LOW (depends on E9 results)
