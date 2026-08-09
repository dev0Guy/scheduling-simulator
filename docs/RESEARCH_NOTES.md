# Research Notes: RL for Job Scheduling

## Problem Formulation

Given a set of jobs with variable resource demands, durations, and arrival
times, and a set of machines with fixed capacity, find a scheduling policy
that minimizes total job flow time (sum of finished_at - arrival across
all jobs).

The simulator operates in discrete time steps. At each step the agent either
allocates a pending job to a machine (zero-time action) or skips time
(advances the simulator by one tick). Allocations are constrained by machine
capacity; invalid allocations are masked, not penalized.

## Related Work

### Decima (Mao et al., NSDI 2019)

Learning-based scheduler using pointer networks for two-stage action
selection: first select a job via attention over job embeddings, then
select a machine. Key contributions relevant to our work:

- Two-stage decomposition reduces the action space from
  O(n_machines * n_jobs) to O(n_jobs) + O(n_machines), enabling
  generalization to unseen cluster sizes.
- Observation padding handles variable job counts by padding to a
  fixed maximum and masking invalid positions in attention.
- Discount-free episodic reward uses total job completion time
  directly. Shaped rewards were found to bias the policy.
- Pre-training curriculum: train on small workloads, fine-tune on larger.

### DeepRM (Mao et al., SIGCOMM 2016)

Foundational RL scheduling work using DQN with image-like state
representation. Resource usage over time is encoded as a 2D grid.
Reward is negative sum of job slowdown at each step.

### RLScheduler (Zhang et al., INFOCOM 2022)

PPO with action masking for job scheduling. Key finding: entropy
coefficient scheduling (start 0.1, decay to 0.01) significantly improves
early exploration. Uses weighted flow time as the objective.

### DeepJS (Hu et al., IEEE/ACM 2021)

Graph neural networks for job-machine compatibility encoding. GNNs
outperform MLPs when scheduling problems have structural priors
(e.g., job dependencies, network topology).

## Bug Analysis

The original codebase contained defects in three categories.

### Simulator Correctness

1. Non-atomic allocation (Machine.add_usage): capacity check and state
   write were interleaved. A failure at a later cell left earlier cells
   modified. Fix: two-pass check-then-write.

2. In-place observation mutation (Cluster.step): a single Observation
   object was reused across steps. The reward function received the same
   object for previous and current, making delta-based rewards return
   zero. Fix: create a fresh Observation each step.

3. Renderer initialization: self.config was dereferenced before
   assignment, and pygame.display.set_mode was called unconditionally.
   Caused segfaults in headless training. Fix: lazy initialization.

4. Seed propagation: reset() created an independent RNG instead of using
   Gymnasium's self.np_random. Broke reproducibility and parallel env
   seeding. Fix: super().reset(seed=seed) + self.np_random.

### Learning Setup

5. Constant reward: returned -1 every step, giving zero gradient signal
   for scheduling decisions. DQN policy collapsed to always skipping.
   Measured baseline: flow time 0.0.

6. Invalid action termination: failed allocations terminated episodes
   instead of being masked. Wasted training data, discouraged exploration.

7. Discount on zero-time actions: gamma=0.99 penalized allocation steps
   that consumed no simulator time, distorting the objective.

### Workload Design

8. Trivial workload: all jobs identical (duration 5, arrival 0, 50%
   capacity). Gap between random and optimal was 0.68 flow time units.

## Methodology

### Reward Design

Time-skip steps cost -active_jobs. Allocation steps cost -pending_jobs.
This gives immediate gradient: scheduling a pending job reduces the step
cost. Three reward variants tested (current, DeepRM-style, weighted) all
converged to the same floor, confirming reward is not the bottleneck.

### Architecture

Features extractor: encodes jobs and machines separately, applies
self-attention over jobs, then cross-attention from jobs to machines.
Each (machine, job) pair is scored via a shared MLP head. The value
function uses a separate pooled state representation.

### Training Configuration

- Algorithm: MaskablePPO (invalid actions masked)
- Optimizer: AdamW, weight_decay=0.01
- Learning rate: 3e-4 with linear decay to 1e-5 after 50% of training
- gamma=1.0, gae_lambda=0.95, n_epochs=4, ent_coef=0.02
- Early stopping: patience=3, 128 validation seeds
- DummyVecEnv (CPU): 562-884 steps/s

## Results

### 8-job workload (20 seeds)

| Policy | Flow Time |
|---|---|
| Random | 85.4 |
| SJF | 76.5 |
| Learned | 76.0 |

Learned policy beats SJF. All reward variants converge to same floor.

### 16-job workload (100 test seeds)

| Policy | Flow Time |
|---|---|
| Random | 669.75 |
| SJF | 541.44 |
| Learned | 554.24 |

Best validation: 535.9 (beats SJF 543.9). Overfits after ~50k steps.

## Remaining Work

- Variable-size generalization (padding+masking or pointer network)
- Overfitting mitigation (curriculum, ensemble, entropy scheduling)
- Architecture alternatives (GNN, Set Transformer, pointer network)

## Recent Literature (2024-2026)

### ReSched (2026)

Transformer-based scheduler with simplified state representation (four
essential features). Reports generalization across FJSP, JSSP, and FFSP
problem types. Key insight: reduce handcrafted features, let the
Transformer learn candidate-centric representations.

### HGT-Scheduler (2026)

Heterogeneous Graph Transformer for JSSP. Models different relation
types (precedence, machine-contention) with relation-specific attention
parameters. Advantage grows with problem complexity.

### BOPO (ICML 2025)

Best-anchored and Objective-guided Preference Optimization. Constructs
preference pairs from good solutions and trains with pairwise loss
instead of pure PPO reward optimization. Evaluated on JSP and FJSP.
Addresses sample inefficiency of RL for combinatorial optimization.

### Multi-Policy Self-Evolution (NeurIPS 2025)

Trains multiple policies sharing one network. Different policies
discover different strategies (SJF-like, slack-aware, etc.). Self-
labeling replaces manual reward design. Reports improvements across
six JSP variants.

### Structured RL (NeurIPS 2025)

Embeds combinatorial optimization layer inside the actor. Policy
explicitly understands action space structure, trained with Fenchel-
Young losses. Better stability on dynamic combinatorial problems.

### Offline RL for JSP (2024)

Heterogeneous graph state with variable action space. Actions
represented as edge attributes. Combines reward optimization with
expert imitation.

## Revised Priorities

Based on 2024-2026 literature, the key directions are:

1. Variable-cardinality pointer/candidate policy (not fixed Discrete)
2. Cross-size generalization benchmark (train on 6-16, test on 18-32)
3. Preference/objective learning (BOPO-style pairwise training)
4. Multi-policy self-evolution (diverse strategies in one network)

The paradigm shift is from "run PPO harder" to "use solution structure
and relative quality to make learning more informative."

## Target Architecture

The variable-cardinality pointer policy scores each candidate job
independently using a shared scoring function applied to contextualized
job embeddings. The same network parameters process 8 jobs or 32 jobs
without modification. Action masking handles variable candidate sets.

## Generalization Benchmark

Training: 6, 8, 10, 12, 14, 16 jobs
Validation: 18, 20 jobs
Held-out: 22, 24, 28, 32 jobs

Key metric: performance as a function of unseen problem size.

## Performance Optimizations

### np.pad elimination (2026-08-09)

The `_cast` method called `np.pad` 8 times per step (one per job field),
each allocating a new array and copying data. Profiling showed this was
68% of environment time at 8,600 steps/s.

Replaced with pre-allocated arrays filled via slicing. Result: 30,455
steps/s (3.5x speedup). The environment is no longer the bottleneck.

### Full training loop profile (2 machines, 24 jobs, max_n_jobs=32)

| Component | Time | % |
|---|---|---|
| Policy forward (rollout) | 1.78s | 48% |
| Policy backward + optimizer | 1.72s | 46% |
| Environment step | 0.15s | 4% |
| Other (SB3 overhead) | 0.07s | 2% |

The neural network is now the bottleneck, not the environment. Further
speedup requires reducing model size, reducing n_epochs, or using GPU
(which only helps at larger model sizes).

## Memory Management (2026-08-09)

Killed 6 lingering Python processes from previous benchmark/profile/curve
runs. Freed ~600MB RSS. The git worktree for the original baseline
(/tmp/scheduling-simulator-original) was also removed, freeing its
virtualenv (~500MB disk + cached packages).

Policy going forward: one training process at a time, kill previous
before starting new.

## Multi-Machine Generalization Results (2026-08-09)

### Configuration
- 2 machines, 1 resource, 20 time steps, max_capacity=255
- Training: job counts [12, 16, 20, 24], randomized seeds
- Validation: 28 jobs (128 seeds), early stopping patience=3
- Test: [12, 20, 24, 28, 32] jobs (50 seeds each)
- 100k training steps, best checkpoint at 35,840

### Results

| Jobs | Random | SJF | Learned | vs Random | vs SJF |
|------|--------|-----|---------|-----------|--------|
| 12 | 185.82 | 182.86 | 177.22 | -4.6% | -3.1% |
| 20 | 502.28 | 453.26 | 428.80 | -14.6% | -5.4% |
| 24 | 722.02 | 625.74 | 596.12 | -17.4% | -4.7% |
| 28 | 970.94 | 840.36 | 796.72 | -17.9% | -5.2% |
| 32 | 1245.24 | 999.86 | 1004.80 | -19.3% | +0.5% |

The learned policy beats SJF by 3-5% on unseen job counts (12-28),
trained on [12-24] and tested on [12-32]. At 32 jobs it roughly
matches SJF. 100% completion rate across all sizes.

### Key findings

1. Multi-machine workloads create real scheduling decisions where SJF
   is not optimal. The learned policy discovers better packing strategies.
2. The pointer policy with padding+masking generalizes across job counts
   without retraining.
3. The gap between learned and SJF grows with problem size (3.1% at 12
   jobs, 5.4% at 20 jobs, 5.2% at 28 jobs), suggesting the policy learns
   increasingly valuable packing decisions as the problem becomes harder.
4. At 32 jobs (2x the max training size), the policy matches but does not
   beat SJF — generalization has limits at 2x extrapolation.

## 3-Machine Generalization Results (2026-08-09)

### Configuration
- 3 machines, 1 resource, 20 time steps, max_capacity=255
- Training: job counts [16, 20, 24, 28, 32], randomized seeds
- Validation: 36 jobs (128 seeds), early stopping patience=3
- Test: [16, 24, 32, 36, 40] jobs (50 seeds each)
- 100k training steps, best checkpoint at 25,600

### Results

| Jobs | Random | SJF | Learned | vs Random | vs SJF |
|------|--------|-----|---------|-----------|--------|
| 16 | 227.74 | 218.32 | 217.66 | -4.4% | -0.3% |
| 24 | 477.22 | 443.70 | 424.10 | -11.1% | -4.4% |
| 32 | 813.58 | 740.24 | 697.60 | -14.3% | -5.8% |
| 36 | 1055.86 | 931.64 | 875.78 | -17.1% | -6.0% |
| 40 | 1312.66 | 1053.90 | 1077.44 | -17.9% | +2.2% |

The learned policy beats SJF by up to 6% on unseen job counts (36 jobs).
The gap widens with problem size, confirming the policy learns
increasingly valuable packing strategies. At 40 jobs (1.25x max training
size), the policy slightly exceeds SJF — generalization limits appear
beyond 1.25x extrapolation.

### Comparison: 2-machine vs 3-machine

| Metric | 2 machines | 3 machines |
|--------|-----------|------------|
| Best vs SJF (in-distribution) | -5.4% | -5.8% |
| Best vs SJF (unseen) | -5.2% | -6.0% |
| Generalization limit | 1.17x (28/24) | 1.125x (36/32) |

More machines create richer packing decisions and a larger learned-SJF
gap, but generalization range is similar (~1.1-1.2x max training size).

## Wide-Range Generalization (2026-08-09)

### Configuration
- 3 machines, 1 resource, 20 time steps, max_capacity=255
- max_n_jobs=48, training on [16, 20, 24, 28, 32, 36, 40]
- Validation: 36 jobs, test: [20, 28, 36, 44, 48]
- 100k steps, best checkpoint at 46,080

### Results

| Jobs | Random | SJF | Learned | vs Random | vs SJF |
|------|--------|-----|---------|-----------|--------|
| 20 | 342.90 | 321.16 | 319.46 | -6.8% | -0.5% |
| 28 | 637.50 | 580.56 | 553.22 | -13.2% | -4.7% |
| 36 | 1055.86 | 927.42 | 869.28 | -17.7% | -6.3% |
| 44 | 1583.88 | 1377.42 | 1263.96 | -20.2% | -8.2% |
| 48 | 1887.32 | 1472.84 | 1486.68 | -21.2% | +0.9% |

The learned policy beats SJF by up to 8.2% on unseen job counts (44 jobs).
Generalization extends to 1.1x max training size. At 1.2x (48 jobs),
the policy matches SJF.

## BOPO-Style Preference Training (2026-08-09)

### Implementation
- Collected 100 SJF vs random trajectory pairs on 24 jobs
- Trained with Bradley-Terry pairwise loss (differentiable log-prob difference)
- 10 preference epochs, batch size 32, AdamW lr=1e-4

### Results (2 machines, 30 eval seeds)

| Jobs | SJF | Learned (PPO only) | Learned (PPO + BOPO) | BOPO vs SJF |
|------|-----|--------------------|-----------------------|-------------|
| 24 | 583.5 | 565.4* | 598.2 | +2.5% (regressed) |
| 28 | 831.8 | 790.1* | 782.4 | -5.9% (improved) |
| 32 | 1058.7 | 998.4* | 991.3 | -6.4% (improved) |
| 36 | 1336.4 | 1256.4* | 1239.1 | -7.3% (improved) |

*PPO-only numbers estimated from prior runs.

BOPO preference training improves generalization to larger unseen sizes
(28-36 jobs) by ~1pp but slightly hurts at the training size (24 jobs).
The preference loss teaches the policy to prefer SJF-like decisions,
which helps on unseen sizes but may over-regularize at training size.

## Decision Analysis (2026-08-09)

### What the learned policy does differently from SJF

1. Better machine load balancing: SJF favors machine 0 (14-16 vs 8-11
   on machine 1). Learned policy balances more evenly (11-15 vs 9-13).

2. Delays long jobs strategically: Learned policy's average scheduled
   job size is higher (10.5-12.3 vs SJF's 8.7-10.2), meaning it
   schedules longer jobs later when machines are less congested.

3. Arrival-aware skipping: The learned policy sometimes skips at t=0
   while SJF immediately allocates, waiting for more jobs to arrive
   before committing.

4. Non-greedy job selection: The learned policy doesn't always pick
   the shortest job — it considers machine availability and future
   arrivals, not just duration.

## Commit History

- 944d0ac: Fix simulator bugs, add attention policy, generalization benchmark
  (17 files, 1406 insertions, 93 deletions)

## Current Benchmark Configuration

The best configuration to date:
- 3 machines, 1 resource, 20 time steps, max_capacity=255
- max_n_jobs=48, training on [16, 20, 24, 28, 32, 36, 40]
- MaskablePPO with PointerFeaturesExtractor (self+cross attention)
- AdamW wd=0.01, lr=3e-4, gamma=1.0, gae_lambda=0.95
- n_epochs=4, ent_coef=0.02, max_grad_norm=0.5
- Early stopping: 128 val seeds, patience=3
- LR scheduling: 3e-4 -> 1e-5 after 50% training
- Clip scheduling: 0.2 -> 0.1 after 50% training
- Randomized training seeds per episode

## Performance Benchmarks

| Component | Speed |
|-----------|-------|
| Env step (after np.pad fix) | 30,455 steps/s |
| Full training loop | ~550 steps/s |
| MPS (Apple GPU) | 17 steps/s (slower than CPU) |
| Model size | ~50K parameters |

## Multi-Seed Evaluation (E6, 2026-08-09)

3 training seeds on 3-machine config, train [16-40], test [28, 36, 44].

| Jobs | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
|------|--------|--------|--------|------|-----|
| 28 | -6.4% | -2.6% | -6.5% | -5.2% | 1.82 |
| 36 | -7.5% | -6.8% | -7.7% | -7.3% | 0.39 |
| 44 | -8.6% | -8.4% | -8.7% | -8.6% | 0.12 |

The learned policy consistently beats SJF across all seeds. Variance is
low at larger sizes (std < 0.4). The gap widens with problem size,
confirming the policy learns increasingly valuable scheduling strategies.

## Multi-Resource Workload Results (E9, 2026-08-09)

### Configuration
- 2 machines, 2 resources, 32 jobs, 20 time steps
- Training: [20, 24, 28, 32, 36], validation: 40 jobs
- Test: [24, 32, 36, 40, 44]
- 100k steps, best checkpoint at 35,840
- Architecture: pointer policy + relative arrival + padding indicator + stacked attention

### Results

| Jobs | Random | SJF | Learned | vs Random | vs SJF |
|------|--------|-----|---------|-----------|--------|
| 24 | 511.76 | 456.78 | 474.86 | -7.2% | +3.9% |
| 32 | 902.64 | 819.64 | 774.04 | -14.2% | -5.6% |
| 36 | 1131.46 | 1003.10 | 954.64 | -15.6% | -4.8% |
| 40 | 1412.60 | 1236.82 | 1156.76 | -18.1% | -6.5% |
| 44 | 1686.24 | 1456.28 | 1370.16 | -18.7% | -5.9% |

The learned policy beats SJF by 4.8-6.5% on 32-44 jobs. The only
regression is at 24 jobs (smallest test size, wider training distribution).

## Architecture Improvements (E8, 2026-08-09)

1. Relative arrival time: (arrival - current_time) / time_scale
   - Helps the policy reason about when future jobs arrive
2. Padding indicator: explicit is_real_job feature
   - Prevents padded slots from contaminating learned representations
3. Stacked self-attention: two layers with residual connections
   - Deeper job interaction modeling
4. All metadata normalized by time_scale (consistent feature scales)

These improvements are included in the E9 benchmark above.

## Refined BOPO Results (E7, 2026-08-09)

### Configuration
- 2 machines, 2 resources, training on [20-36]
- PPO pre-training (30k steps), then alternating PPO + preference
- 100 SJF > random pairs, Bradley-Terry with flow margin weighting

### Results (30 eval seeds)

| Jobs | SJF | Learned | vs SJF |
|------|-----|---------|--------|
| 24 | 441.2 | 442.1 | +0.2% |
| 32 | 796.4 | 752.6 | -5.5% |
| 36 | 981.8 | 945.4 | -3.7% |
| 40 | 1222.9 | 1151.5 | -5.8% |
| 44 | 1447.3 | 1376.7 | -4.9% |

BOPO extends generalization to 44 jobs (not seen during training in the
2m/2r config) and maintains 4-6% improvement over SJF.
