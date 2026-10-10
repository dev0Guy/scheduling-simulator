# Current Architecture (M9 + optional per-resource fit)

Source files:
- `src/scheduling_simulator/expiremnt/feature_extractor/new_features.py` — `StatusSizeWaitTtlCapacityFeaturesExtractor`
- `src/scheduling_simulator/expiremnt/policy/new_sched.py` — `NewSchedulingPolicy`
- `src/scheduling_simulator/expiremnt/runners/train_runner.py` — PPO setup
- `src/scheduling_simulator/expiremnt/runners/env_factory.py` — env + wrapper stack

Reference result: M9 = wandb run `etudt4qo`, mean reward -49.28 (SJF `rjzcixv2`: -50.43, +2.28%).

## 1. Environment

- Config (example.py): `n_machines=1, n_jobs=10, n_resource=3, n_time=10, max_capacity=255`, `max_time=110`.
- Job lifecycle: NOT_CREATED -> PENDING -> RUNNING -> COMPLETED.
  - `wait_time` increments only while PENDING.
  - `ttl` starts at `size` and counts down while RUNNING.
  - Completion time = `wait_time + size`.
- Machine usage is a sliding `(n_resource, n_time)` window, shifted left every tick.
- Action space: `Discrete(1 + n_machines * n_jobs)`; `0 = skip`, `1 + machine_idx * n_jobs + job_idx = allocate`.
- Wrapper stack: `DummyVecEnv([Monitor(FailureSkipTimeWrapper(SchedulingEnviorment))])`.
- Reward (`FailureSkipTimeWrapper`):
  - skip tick: `-(# PENDING or RUNNING jobs)`
  - successful allocation: `+1.0`
  - failed allocation: penalty, and one tick is burned
  - terminal: `+100` if all jobs complete, `-100` at the time cap
  - So reward is roughly minus total flow time.
- Job generator: `deeprm_generate_single_job_decreasing_size` (size shrinks with job index with +-5 jitter, jittered arrival). Each job gets ONE random utilization in [10%, 50%] of 255, applied identically to all resources.

## 2. Observation (Dict)

Per job: `status`, `size`, `wait_time`, `ttl`, `arrival`, `usage (n_resource, n_time)`.
Per machine: `capacity`, `usage (n_resource, n_time)`.

## 3. Features extractor

Output shape: `(batch, embedding_dim * (n_jobs + 1))`, laid out as `[global, job_0 .. job_N]`. `embedding_dim = 256`.

1. **Job encoder**: input `[status_embed(16), size, wait_time, ttl, arrival]` -> MLP (Linear -> LayerNorm -> ReLU, twice) -> `embedding_dim`. Usage does NOT enter here.
2. **Fit attention (job -> machine)**
   - Per job, usage = max over time, divided by `max_capacity`, clamped to [0, 1].
   - Per machine, free = min over time of `(capacity - usage)`, divided by `max_capacity`, clamped to [0, 1].
   - `scalar_mlp` (shared) maps each into a `fit_dim=64` query / key / value.
   - A feasibility mask (`job usage <= machine free + 1e-6`) gates a sigmoid "fit attention".
   - `fit_fuse`: Linear(embedding_dim + fit_dim -> embedding_dim) + LayerNorm + ReLU.
3. **`per_resource_fit` option** (default `False`, so default behaviour equals M9; `example.py` currently sets `True`)
   - Reduces over time only (`amax` / `amin` over dim 3), keeping all `n_resource` values.
   - `scalar_mlp` input becomes `n_resource` instead of 1.
   - `can_fit = (usage <= free + 1e-6).all(dim=-1)` across resources.
   - Caveat: the current generator gives identical utilization on all 3 resources, so this adds no information until the generator produces distinct per-resource values.
4. **Job self-attention**: 4 heads, COMPLETED jobs are masked as keys (guard for rows where all jobs are completed), residual + LayerNorm.
5. **Attention pooling** (Deep-Sets style): a scorer produces weights over jobs, then a weighted sum goes into `global_head`, producing the `global` embedding.

## 4. Policy (`NewSchedulingPolicy`)

- `ActorCriticPolicy` with `net_arch={"pi": [], "vf": []}`, `ortho_init=False`, Adam optimizer.
- Extractor kwargs: `status_embed_dim=16, embedding_dim, attention_hidden_dim=32, max_time, per_resource_fit`.
- `SchedulingMlpExtractor`: the actor receives the full feature vector; the critic receives only the global embedding (first `embedding_dim`). Both nets are Identity.
- **Actor (`SchedulingActionHead`)**
  - Static `nn.Embedding(n_machines, embedding_dim)` gives machine identity.
  - Pair scorer over `[global, machine, job]` (3 * embedding_dim): Linear(->128) -> ReLU -> Linear(128) -> ReLU -> Linear(1).
  - Skip head: Linear(embedding_dim -> 64) -> ReLU -> Linear(64 -> 1).
  - Logits = `[skip, pair_scores...]`, size `1 + n_machines * n_jobs`.
- **Critic (`SchedulingValueNet`)**: embedding_dim -> 256 -> 256 -> 1.
- A pointer-network head exists in the repo but is not used by M9.

## 5. PPO hyperparameters

| Param | Value |
|---|---|
| learning rate | 7e-6 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 5 |
| gamma / gae_lambda | 0.995 / 0.995 |
| clip_range | 0.2 |
| ent_coef | linear 0.05 -> 0.01 (`EntCoefScheduler`) |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| train_steps | 2,000,000 |

Evaluation: `EvalCallback` every 50,000 steps, 10 episodes; logging via `WandbCallback`.

## 6. Known caveats

- Job usage is uniform in the current generator, so usage is redundant with `size` and the per-resource input is unlikely to help yet.
- `_mask_non_pending_usage` is a no-op, so completed and not-created jobs still show usage.
- `Machine.usage` sliding-window capacity behaviour should be double-checked.
- No not-created mask in the extractor (that was M11, E1).

## 7. Update: pair features + dynamic machine embedding (flags)

Both are OFF by default in the code; `example.py` turns them on
(`"pair_features": True, "dynamic_machine": True` in `policy_kwargs`).
Diagram: `docs/diagrams/current_architecture_v2.svg`.

**pair_features** (extractor appends `n_machines * n_jobs * 9` numbers):
for every (machine, job): `[pending, fits_now, delay/T, finish = (delay+size)/max_time,
headroom x n_resource (free_min - job_usage, signed), dominant_share, packing]`.
`delay` = first start offset s in 0..T-1 where the job fits on ALL resources for ticks
[s, s+size) in the machine's free-capacity grid (T = "never inside the window").
The actor passes them through `Linear(9 -> 64) + ReLU` and concatenates them to
`[global, machine, job]` (score_net input 3*256 + 64 = 832).

**dynamic_machine** (extractor appends `n_machines * 256` numbers): free grid
`(capacity - usage) / max_capacity` of shape (resources x ticks) -> shared per-row
encoder (Linear(10->64), ReLU, Linear(64->64), ReLU) -> concat rows -> MLP -> 256.
Last layer is zero-initialised. The actor adds it to the static `nn.Embedding` machine vector.

Feature vector layout: `[global | job_0..job_9 | pair feats | machine_dyn]`
(2816 + 90 + 256 = 3162 with the current config). The critic still reads only the first 256.
Not yet run; shapes and logic were checked by hand and by a numpy mirror of the fit/delay
computation on the example in `docs/html/usage_ideas.html`.
