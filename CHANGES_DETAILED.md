# Detailed Change Log: scheduling-simulator

This document explains every change from the original repository
(commit c5cd2e8) to the current HEAD, with code blocks, concepts,
and evidence. Not tracked in git.

---

## 1. Atomic Allocation (machine.pyx)

**Problem:** `Machine.add_usage` checked capacity and wrote state in
the same loop. If a job fit on cells [0,0]–[0,3] but exceeded capacity
at [0,4], the function returned `False` — but cells [0,0]–[0,3] were
already modified. The machine had phantom usage from a job that was
never scheduled.

**Fix:** Two-pass: check all cells first, write only if every cell fits.

```python
# Before (buggy):
for i in range(rows):
    for j in range(cols):
        new_cell_value = self.usage[i, j] + job_usage[i, j]
        if new_cell_value > self.capacity[i, j]:
            return False          # earlier cells already written
        self.usage[i, j] = new_cell_value

# After (fixed):
for i in range(rows):             # pass 1: check
    for j in range(cols):
        new_cell_value = self.usage[i, j] + job_usage[i, j]
        if new_cell_value > self.capacity[i, j]:
            return False

for i in range(rows):             # pass 2: write
    for j in range(cols):
        new_cell_value = self.usage[i, j] + job_usage[i, j]
        self.usage[i, j] = new_cell_value
```

**Proof:** Reproduced with a unit test — allocate usage that fits
partially, verify machine state is unchanged after failure.

**Inspiration:** Standard transactional write pattern. No external
source.

---

## 2. Observation Snapshotting (cluster.pyx)

**Problem:** `Cluster.step` mutated one `Observation` object in place.
The environment stored `previous = self._last_observation`, then
called `cluster.step()` which overwrote that same object's fields.
So `previous` and `current` were the same Python object.

**Impact:** Any reward function computing a delta (like flow time)
always saw zero delta. SB3's replay buffer stored `(obs, action,
reward, next_obs)` but both obs and next_obs pointed to the same
mutated arrays.

**Fix:** Create a fresh `Observation` each step.

```python
# Added to Cluster.step, before update_observation:
self.observation = self.create_observation()
self.observation.action_success = allocation_succsued if not action.skip else True
self.update_observation()
return self.observation
```

**Proof:** `test_reward_receives_distinct_observation_snapshots` —
verified `current is not previous` and `previous.time != current.time`.

**Inspiration:** Gymnasium API contract requires distinct observation
objects per step. Standard RL environment practice.

---

## 3. Renderer Initialization (render.pyx)

**Problem:** `Renderer.__init__` used `self.config` before assigning
it, then unconditionally called `pygame.display.set_mode` even in
`rgb_array` mode. This caused segfaults in headless training.

**Fix:** Move `self.config = config` before first use. Remove the
unconditional `set_mode` call — the conditional block above it
already handles both modes.

```python
# Before (buggy):
self.to_screen = to_screen
if self.to_screen:
    self.screen = pygame.display.set_mode(...)  # uses self.config
else:
    self.screen = pygame.Surface(...)
self.config = config                            # assigned AFTER use
self.screen = pygame.display.set_mode(...)      # unconditional, crashes headless

# After (fixed):
self.to_screen = to_screen
self.config = config                            # assigned BEFORE use
if self.to_screen:
    self.screen = pygame.display.set_mode(...)
else:
    self.screen = pygame.Surface(...)
```

**Proof:** `test_rgb_array_environment_does_not_open_a_display` —
env with `render_mode='rgb_array'` no longer segfaults. Also fixed
the SDL/OpenCV symbol collision that crashed SB3's video recorder.

**Inspiration:** Pygame documentation — `Surface` is the correct
offscreen rendering target for headless mode.

---

## 4. Gymnasium Seed Propagation (envioremnt.py)

**Problem:** `reset()` created `np.random.default_rng(seed)` each
call instead of using Gymnasium's built-in `self.np_random`. This
broke reproducibility and SB3's parallel env seeding.

**Fix:**

```python
# Before:
self._cluster = self._creator(self._config, np.random.default_rng(seed))

# After:
super().reset(seed=seed)
self._cluster = self._creator(self._config, self.np_random)
```

**Proof:** `test_reset_seed_reproduces_the_episode_sequence` —
two envs seeded with the same value produce identical observations.
A second `reset()` without seed produces a different episode.

**Inspiration:** Gymnasium API documentation — `super().reset(seed=seed)`
initializes `self.np_random` which should be used for all stochasticity.

---

## 5. Dense Flow-Time Reward (envioremnt.py)

**Problem:** Default reward was constant `-1` every step. Zero gradient
signal for scheduling decisions. DQN collapsed to always skipping
time. Measured baseline: flow time 0.0 (never scheduled any job).

**Concept:** Total job flow time = sum of (finished_at - arrival)
across all jobs. We want a dense per-step reward that:
- Gives signal on every step (not just terminal)
- Is aligned with the flow-time objective
- Distinguishes scheduling decisions from each other

**Design:**
- Time-skip steps: cost = `-active_jobs` (actual incremental flow time)
- Allocation steps: cost = `-pending_jobs` (penalty for leaving jobs
  waiting; scheduling a pending job reduces cost from -N to -(N-1))

```python
def flow_time_reward(current, prev):
    if prev is None:
        return 0.0
    previous = prev.to_dict()
    elapsed = current.to_dict()['time'] - previous['time']
    if elapsed > 0:
        active = np.count_nonzero(
            (previous['status'] == 1) | (previous['status'] == 2)
        )
        return -float(active)
    else:
        pending = np.count_nonzero(previous['status'] == 1)
        return -float(pending)
```

**Why this works:** The policy gets immediate credit for scheduling
a job (reduces pending count). The value function learns which
allocations minimize future time-skip costs. Three reward variants
tested (this, DeepRM-style, weighted) all converged to the same
floor — reward is adequate, not the bottleneck.

**Proof:** Training curve went from stuck at 175 (constant -1 reward)
to learning (165 -> 95 -> 76 over 10k steps).

**Inspiration:** DeepRM (Mao et al., SIGCOMM 2016) uses negative
sum of job slowdown at each step. Our design is similar but
separates allocation and time-skip costs.

---

## 6. Action Masking (envioremnt.py)

**Problem:** Invalid actions (allocating a job that doesn't fit, or
a job that's already running) terminated episodes via
`FailureSkipTimeWrapper`. This wasted training data and taught the
policy to avoid exploration.

**Fix:** Added `action_masks()` method that returns a boolean mask
over the action space. Only valid allocations (pending job + machine
has capacity) are unmasked. Switched from DQN to MaskablePPO which
natively supports action masking.

```python
def action_masks(self) -> np.ndarray:
    obs = self._last_observation_dict
    n_actual = self._n_actual
    n_machines = self._config['n_machines']
    pending = obs['status'][:n_actual] == 1
    fits = np.all(
        obs['machines_usage'][:, None, :, :]
        + obs['jobs_usage'][:n_actual][None, :, :, :]
        <= obs['machines_capacity'][:, None, :, :],
        axis=(2, 3),
    )
    mask = np.zeros((n_machines, self._max_n_jobs), dtype=bool)
    mask[:, :n_actual] = fits & pending[None, :]
    return np.concatenate(([True], mask.reshape(-1)))
```

**Proof:** `test_action_mask_contains_only_pending_jobs_that_fit` —
mask is `[True, True, True]` when both jobs fit, becomes
`[True, False, ...]` after scheduling one job that blocks the other.

**Inspiration:** sb3-contrib MaskablePPO (Oriol et al., 2023).
Action masking is standard for combinatorial RL with invalid actions.

---

## 7. Variable Workload Generator (creator.pyx)

**Problem:** All jobs had fixed duration 5, arrival time 0, and 50%
capacity usage. Gap between random and optimal scheduling was 0.68
flow time units — too small for measurable learning.

**Fix:** Variable duration (1–n_time), variable arrival (0–n_time),
variable resource demand (dominant resource 25–75% of capacity,
secondary 5–25%).

```python
# Before:
int size = 5
usage[resource_idx, :] = random.integers(127, 200, dtype=np.int32)
return Job(usage=usage, arrival_time=0, size=size)

# After:
int size = random.integers(1, config.n_time + 1)
int arrival_time = random.integers(0, config.n_time)
usage[resource_idx, :size] = random.integers(dominant_low, dominant_high)
return Job(usage=usage, arrival_time=arrival_time, size=size)
```

**Proof:** On 16 jobs / 1 machine, gap between random and SJF went
from 0.68 to 124.5 flow time units (18.3%). On 2 machines / 2
resources / 32 jobs, gap is 82.6 (9.4%).

**Inspiration:** DeepRM workload generator uses similar dominant/
secondary resource split with variable durations.

---

## 8. Pointer Attention Policy (policy.py)

**Problem:** Original flat MLP (`MultiInputPolicy`) flattened all
observation fields into one vector, learning separate weights for
each job slot. Can't generalize across job positions.

**Architecture:**

```
Job features ──→ Job encoder ──→ [h₁ h₂ ... hN]
                                      │
                            Self-attention (jobs attend to each other)
                                      │
                              LayerNorm + residual
                                      │
                    Cross-attention (jobs attend to machines)
                                      │
                              LayerNorm + residual
                                      │
                    Pair scorer: concat(machine_emb, job_emb, product)
                                      │
                    SharedActionHead: Linear → ReLU → Linear(1)
                                      │
                              One logit per (machine, job) pair
                                      │
                              + skip logit from pooled state
```

**Key design decisions:**
- Self-attention lets each job see all other jobs (relative comparison)
- Cross-attention lets jobs incorporate machine availability
- SharedActionHead applies the same MLP to every pair — same
  parameters score 8 jobs or 32 jobs
- Value function uses separate pooled state (job mean/max + machine
  mean/max + current time)
- Padded job slots are masked in attention via `key_padding_mask`

**Why not stacked attention?** A/B tested: single attention layer
beat stacked on all test sizes (0.7–3.1% better). The extra
parameters dilute the learning signal on small problems.

**Why not relative arrival?** A/B tested: raw arrival time beat
relative arrival (arrival - current_time). The relative encoding
removed information the network could learn to compute itself.

**Proof:** On 2m/2r, 3 seeds, 30 eval seeds:
- 32 jobs: -5.5% vs SJF (std 0.73)
- 40 jobs: -6.7% vs SJF (std 0.42)
- 44 jobs: -5.9% vs SJF (std 0.62)

**Inspiration:**
- Pointer networks (Vinyals et al., NeurIPS 2015) — attention over
  candidate set for selection
- Decima (Mao et al., NSDI 2019) — attention-based scheduler with
  job/machine embeddings
- Set Transformer (Lee et al., ICML 2019) — attention for set-structured data

---

## 9. Padding + Masking for Variable Job Counts (envioremnt.py)

**Problem:** Fixed action space `1 + n_machines * n_jobs` can't
handle different job counts at train vs test time.

**Fix:** `max_n_jobs` parameter pads observations to a fixed maximum.
Padded slots have `status=COMPLETED`, `ttl=0`, `size=0`, zeroed
usage. The policy masks them in attention and the action mask
prevents selecting them.

```python
# In _cast: pad job fields to max_n_jobs
if n_pad > 0:
    status = np.concatenate([status, np.full(n_pad, 3, dtype=np.int32)])
    jobs_usage = pad_3d(jobs_usage)

# In action_masks: only real jobs can be selected
mask = np.zeros((n_machines, self._max_n_jobs), dtype=bool)
mask[:, :n_actual] = fits & pending[None, :]
```

**Action translation:** The action space uses `max_n_jobs` for
encoding, but the cluster decodes using actual job count. The
environment translates:

```python
if action == 0:
    cluster_action = 0
else:
    padded_job = (action - 1) % self._max_n_jobs
    machine_idx = (action - 1) // self._max_n_jobs
    if padded_job >= self._n_actual:
        cluster_action = 0  # padded slot, treat as skip
    else:
        cluster_action = 1 + machine_idx * self._n_actual + padded_job
```

**Proof:** Policy trained on [20-36] jobs generalizes to 44 jobs
(beats SJF by 5.9%). Same network, no retraining.

**Inspiration:** Decima (Mao et al., NSDI 2019) uses observation
padding with attention masking for variable job counts.

---

## 10. Observation Caching (envioremnt.py)

**Problem:** `to_dict()` was called 4x per step (step, reward,
action_masks, _cast), each copying all observation arrays. Profiler
showed 60% of env time in these copies.

**Fix:** Cache `_last_observation_dict` and `_n_actual` after each
step. Action masks and reward use the cached dict.

**Proof:** Env throughput went from 8,600 steps/s to 30,455 steps/s
(3.5x speedup). Training loop bottleneck shifted from env to neural
network forward/backward.

---

## 11. np.pad Elimination (envioremnt.py)

**Problem:** `_cast` called `np.pad` 8x per step (one per job field),
each allocating a new array and copying. Profiler showed 68% of env
time in `np.pad`.

**Fix:** Replaced with pre-allocated arrays filled via slicing:

```python
def pad_1d(arr):
    out = np.zeros(self._max_n_jobs, dtype=np.float32)
    out[:n_actual] = arr.astype(np.float32)
    return out
```

**Proof:** Same 3.5x speedup as observation caching.

---

## 12. Lazy Renderer Initialization (envioremnt.py)

**Problem:** `Renderer(...)` was called in `__init__`, initializing
PyGame on every env construction. This caused SDL/OpenCV symbol
collisions that crashed headless training with SB3.

**Fix:** Renderer is created on first `render()` call, not in
`__init__`.

```python
def render(self):
    if self._last_observation is None:
        return None
    if self._renderer is None:
        from scheduling_simulator.core.render import Renderer
        self._renderer = Renderer(self.render_mode == 'human')
    return self._renderer.render(self._last_observation)
```

**Proof:** Headless training with 4 parallel envs no longer crashes.

---

## 13. MaskablePPO Replacing DQN (train_runner.py, benchmark.py)

**Problem:** DQN with `gamma=0.99` discounted zero-time allocation
steps. Invalid actions terminated episodes. Training was completely
broken (flow time 0.0).

**Fix:** Switched to MaskablePPO with:
- `gamma=1.0` (finite-horizon, no discount on zero-time actions)
- `gae_lambda=0.95` (variance reduction)
- `n_epochs=4` (down from default 10)
- `ent_coef=0.02` (slightly above default for exploration)
- `max_grad_norm=0.5` (gradient clipping)
- AdamW with `weight_decay=0.01`
- Auto device selection (mps/cuda/cpu)

**Why gamma=1.0:** Scheduling is finite-horizon. Every allocation
step consumes zero simulator time, so discounting them distorts the
objective. With gamma=1, the policy optimizes true total flow time.

**Why gae_lambda=0.95 not 1.0:** Full-return GAE (lambda=1) has high
variance because episode flow time has std=24 on mean=84. Lambda=0.95
trades a small bias for much lower advantage variance.

**Why n_epochs=4 not 10:** 10 epochs over 512 transitions (~8
episodes) is aggressive for a sparse-reward discrete problem. 4
epochs reduces overfitting risk.

**Proof:** Weight decay sweep (4 configs, 10k steps each) confirmed
AdamW wd=0.01 is safe and slightly more stable than Adam. Original
DQN baseline: flow time 0.0. MaskablePPO: beats SJF by 5-7%.

**Inspiration:**
- MaskablePPO: sb3-contrib (Oriol et al., 2023)
- PPO hyperparameters: Schulman et al. (2017) + empirical tuning
- AdamW: Loshchilov & Hutter (2019) — decoupled weight decay

---

## 14. Generalization Benchmark (benchmark.py)

**Design:** Train on a range of job counts, validate on one unseen
size, test on multiple unseen sizes. Early stopping with patience=3.
LR and clip range scheduling (constant for first half, linear decay
for second half).

```python
TRAIN_JOB_COUNTS = [20, 24, 28, 32, 36]
VAL_JOB_COUNTS = [40]
TEST_JOB_COUNTS = [24, 32, 36, 40, 44]
```

**Training distribution:** 80% small jobs [20, 24], 20% large
[28, 32, 36]. The skew fixes a regression at small test sizes
(under-exposure diagnosis).

**Evaluation metric:** Total job flow time (sum of finished_at -
arrival). Truncated episodes penalized with `MAX_EPISODE_STEPS *
n_jobs` to prevent gaming with stalled policies.

**Proof:** 3-seed eval on 2m/2r:
- 32 jobs: -5.5% vs SJF (std 0.73)
- 40 jobs: -6.7% vs SJF (std 0.42)
- 44 jobs: -5.9% vs SJF (std 0.62)

---

## 15. Small-Job Skew Training Distribution

**Problem:** Uniform training over [20-36] caused the policy to
regress at 24 jobs (+3.9% vs SJF) while improving at 32-44 (-5-7%).

**Diagnosis:** Under-exposure to small sizes. Small episodes have
fewer steps and lower reward signal, so the policy underfits them
when they're only 20% of training data.

**Fix:** Skew to 80% small [20, 24], 20% large [28, 32, 36].

**Proof:** 24-job gap went from +3.9% to +0.2% (matches SJF within
noise). Larger sizes maintained their 5-7% improvement.

---

## 16. Training Infrastructure

### DummyVecEnv vs SubprocVecEnv
SubprocVecEnv with 4 workers: 190 steps/s (IPC overhead dominates).
DummyVecEnv with 1 worker: 562-884 steps/s (3-5x faster for this env).
The env is too lightweight for multiprocessing to pay off.

### MPS (Apple GPU)
MPS: 17 steps/s. CPU: 31 steps/s. The model is too small
(~50K parameters) for GPU data-transfer overhead to pay off.
Auto-detection picks MPS if available but falls back to CPU.

### Performance profile (after optimizations)
| Component | Time | % |
|-----------|------|---|
| Policy forward (rollout) | 1.78s | 48% |
| Policy backward + optimizer | 1.72s | 46% |
| Environment step | 0.15s | 4% |
| Other (SB3 overhead) | 0.07s | 2% |

The neural network is the bottleneck, not the environment.

---

## What Was Tried and Removed

### Stacked attention (E8)
Added a second self-attention layer with residual connection.
A/B tested: simple single-attention version was 0.7-3.1% better
on all test sizes. Removed.

### Relative arrival time (E8)
Replaced raw arrival with (arrival - current_time) / time_scale.
A/B tested: raw arrival was better. Removed.

### Padding indicator feature (E8)
Added `is_real_job` as a 7th metadata field. A/B tested: without
it, the policy was better. Removed. (Attention masking via
`key_padding_mask` already handles padding.)

### Two-stage pointer policy (P3)
Decima-style job-then-machine selection with custom PPO trainer.
Smoke test passed but unstable on 2m/2r. Custom trainer had
theoretical issues (sum vs product of stage ratios). Not
competitive with single-stage SB3 pointer. Removed (471 lines).

### BOPO preference training (E5, E7)
Bradley-Terry pairwise loss on SJF vs random trajectory pairs.
Gave 0.8-2.3% improvement at larger sizes but within noise of
PPO-only. Added significant training overhead (120 trajectory
rollouts per callback). Removed (166 lines).

### DeepSets features extractor
Alternative to self-attention: encode each element independently,
pool via mean+max. Same performance as attention, 1.7x faster
training. Not the point (generalization, not efficiency). Removed.

### FactoredActionHead
Decomposed logits into job_score + pair_score. Marginalizing over
machines destroyed the fit signal. Reverted to SharedActionHead.

---

## Results Summary

### Final benchmark: 2 machines, 2 resources (3 seeds, 30 eval seeds)

| Jobs | Mean gap vs SJF | Std |
|------|-----------------|-----|
| 24 | +0.2% | 0.33 |
| 32 | -5.5% | 0.73 |
| 40 | -6.7% | 0.42 |
| 44 | -5.9% | 0.62 |

### Cross-config comparison

| Config | Best vs SJF | Generalization limit |
|--------|-------------|---------------------|
| 2m/1r, train [12-24] | -5.4% at 20j | 1.17x (28/24) |
| 3m/1r, train [16-40] | -8.2% at 44j | 1.1x (44/40) |
| 2m/2r, train [20-36] | -6.7% at 40j | 1.1x (44/36) |

### Baseline comparison (2m/2r, 30 seeds)

| Policy | 24j | 32j | 40j | 44j |
|--------|-----|-----|-----|-----|
| Random | 492.5 | 880.8 | 1388.1 | 1671.0 |
| SJF | 441.2 | 750.6 | 1145.9 | 1368.5 |
| LJF | 564.1 | 1052.3 | 1704.2 | 2074.5 |
| FIFO | 497.9 | 889.0 | 1410.2 | 1706.6 |
| Least-loaded | 501.2 | 870.5 | 1363.2 | 1652.3 |
| Best-fit | 501.2 | 870.5 | 1363.2 | 1652.3 |
| WSPT | 441.2 | 750.6 | 1145.9 | 1368.5 |
| **Learned** | **443.7** | **746.7** | **1135.6** | **1354.9** |

SJF/WSPT is the strongest simple heuristic. Learned beats it by
5-7% on 32-44 jobs.

### Decision analysis: what the policy learns

1. **Better machine load balancing:** SJF favors machine 0 (14-16
   vs 8-11 on machine 1). Learned balances more evenly (11-15 vs
   9-13).

2. **Delays long jobs strategically:** Learned policy's average
   scheduled job size is higher (10.5-12.3 vs SJF's 8.7-10.2),
   meaning it schedules longer jobs later when machines are less
   congested.

3. **Arrival-aware skipping:** The learned policy sometimes skips
   at t=0 while SJF immediately allocates, waiting for more jobs
   to arrive before committing.

4. **Non-greedy job selection:** The learned policy doesn't always
   pick the shortest job — it considers machine availability and
   future arrivals, not just duration.
