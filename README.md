![Tests](https://github.com/dev0Guy/scheduling-simulator/actions/workflows/tests.yml/badge.svg?branch=main)
[![Coverage](https://codecov.io/gh/dev0Guy/scheduling-simulator/branch/main/graph/badge.svg)](https://codecov.io/gh/dev0Guy/scheduling-simulator)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/github/license/dev0Guy/scheduling-simulator)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Base Status
-----------
We've created 3 Environment for scheduling, which implement the `ClusterABC` with `JobCollection` and `MachineCollection` protocols:

---

# Current Architecture (as implemented)

> The sections below this one ("Full Architecture Writeup") describe an earlier/alternate `PointerFeaturesExtractor` + `SchedulingPolicy` design. The classes actually wired up and trained today are `StatusSizeWaitTtlCapacityFeaturesExtractor` ([new_features.py](src/scheduling_simulator/expiremnt/feature_extractor/new_features.py)) and `NewSchedulingPolicy` ([new_sched.py](src/scheduling_simulator/expiremnt/policy/new_sched.py)) — this section documents those.

## Diagram

```
                              OBSERVATION
        ┌───────────────────────────────────────────────────┐
        │ per-job (n_jobs=10): status, size, wait_time, ttl  │
        │ scalar: time                                        │
        │ per-machine: usage, capacity                        │
        │ (jobs_usage/machines_usage ALSO feed a separate,     │
        │  raw/unembedded path -- see "Fit gate" below)        │
        └───────────────────────┬───────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                                ▼
        queue_pressure = active_job          per-job feature vector
        count / n_jobs (1 scalar),           [status_emb(16) + size(1)
        broadcast to every job  ─────┐        + wait_time(1)
                                      │        + ttl(1)] = 19-dim
                                      ▼                │
                              concat (+1) ◄────────────┘
                                      │
                                      ▼
                        per-job feature vector, 20-dim
                                      │
                                      ▼
                          Job encoder (shared MLP)
                              20 → 256 → 256
                                      │
                                      ▼
                        per-job embeddings (10 × 256)
                 │
                 ▼                               (per-machine, from
       Job self-attention (multi-head,             machines_usage/
       4 heads, post-norm residual)                capacity, NOT a
       every job attends to every other             static identity
       job -- arrived or not, EXCEPT                embedding)
       jobs with status=COMPLETED, which                  │
       are masked out as attention keys                   ▼
       (key_padding_mask) so their stale        Machine encoder (MLP)
       data can't dilute other jobs'            2 → 256 → 256
       contextualized embeddings                        │
                 │                                       ▼
                 ▼                            per-machine embeddings
    job embeddings (10 × 256)                   (n_machines × 256)
                 │                                       │
        NOTE: this whole attention path never sees usage/capacity
        directly -- it only ever reasons about job identity and
        timing. See "Fit gate" below for how capacity info re-enters.
                 │                                       │
                 ▼                                       ▼
                              DUAL ATTENTION
      ┌──────────────────────────────────────────────────────────┐
      │ Machine self-attention (4 heads) -- machines attend to    │
      │ every other machine (competition for shared jobs)          │
      │                                                             │
      │ Job → Machine cross-attention: jobs attend to machine state│
      │ Machine → Job cross-attention: machines attend to job state│
      │ each: embeddings + GATE * cross_attn_output, post-norm     │
      │ GATE = learnable per-channel scale, init to ZERO            │
      │ (LayerScale -- starts as a no-op, grows only if it helps)  │
      └──────────────────────────────────────────────────────────┘
                 │                                       │
                 ▼                                       ▼
   contextualized job embeddings          contextualized machine embeddings
        (10 × 256)                               (n_machines × 256)
                 │                                       │
        ┌────────┴────────┐                              ▼
        ▼                 ▼                        mean pooling
  returned to ACTOR   Pooling attention                   │
  (per-job, unchanged)  (learned weights,                 ▼
        │                weighted sum → 1)      machine_global (256)
        │                       │                         │
        │                       ▼                         │
        │                 job_global (256) ────────────────┤
        │                       │                          │
        │                       ▼                          ▼
        │         concat [job_global, machine_global, queue_pressure] (513)
        │                       │
        │                       ▼
        │              global_head MLP: 513 → 256
        │                       │
        │                       ▼
        │              global embedding (256)
        │                       │
        ├───────────────────────┼─────────────────────┐
        ▼                       ▼                      ▼
     ACTOR                   ACTOR                   CRITIC
  [global+machine+job]      skip logit             [global(256)]
   → score_net MLP          (from global            → value MLP
   → embedding_score          embedding alone)        → V(s)
   per (machine,job) pair
   (machine embedding here is the DYNAMIC,
   dual-attention-contextualized one above,
   not a static per-machine identity vector)
        │
        ▼
   FIT GATE (per (machine,job) pair, see "Fit gate" below)
   fit_gate_net([global,machine,job]) -> gate in [0,1]
   fit = free_capacity_now[machine] - usage[job]  (raw, from
         the unembedded path, NOT from the embeddings above)
   pair_score = embedding_score + fit_scale * gate * fit
   (fit_scale: learnable scalar, init = 0 -- LayerScale-style,
    starts as a pure no-op)
        │
        └───────────────┬───────────────┘
                         ▼
              softmax over all action logits
                         │
                         ▼
                       ACTION
              (skip, or allocate job N to machine M)
```

## Per-job features and normalization

| Feature | Meaning | Normalization |
|---|---|---|
| `status` | NOT_CREATED / PENDING / RUNNING / COMPLETED | learned `nn.Embedding(n_statuses, 16)` |
| `size` | total job duration | `/ max_time` (no clamp) |
| `wait_time` | time spent PENDING so far | `/ max_time` (no clamp) |
| `ttl` | remaining runtime once RUNNING (starts at `size`, counts down) — the only signal for "how soon does the machine free up" | `/ max_time`, clamped to `[0, 1]` |
| `queue_pressure` | fraction of jobs currently PENDING or RUNNING — exactly the quantity the reward penalizes every tick | `active_job_count / n_jobs`, fed into **both** the global/critic path and broadcast into every per-job vector |

> **Removed:** `arrival` (time until arrival) was dropped as a feature — see "Random per-job utilization" below for why, and `docs/experiment_log.md` #16 for the ablation that motivated it.
>
> **`usage` is deliberately NOT in this table.** It exists (a single scalar per job, `/ max_capacity`, read from cell `[0, 0]` of the job's usage matrix — see "Random per-job utilization" below), but is never fed into `job_encoder` or any attention layer. It only reaches the policy through the separate, raw "fit gate" path — see the next section.

## Why job self-attention

Before this addition, the only cross-job mechanism was the pooling attention: one learned scorer assigns each job a scalar weight, then all job embeddings are weighted-summed into a **single shared vector**. Every job contributed to that vector, but no job ever saw any *other specific* job — only the blurred average. A job couldn't represent "I'm smaller than the one pending now, bigger than the one landing in 2 ticks," only vague global context.

The `job_self_attention` block (`nn.MultiheadAttention`, 4 heads, post-norm residual) runs *before* pooling: every job attends directly to every other job — arrived or not, since `NOT_CREATED` jobs still carry real `size`/`wait_time` — producing a genuinely contextualized embedding per job. Both the actor's per-job scoring and the pooled global/critic summary now use these contextualized embeddings instead of the raw, job-in-isolation ones.

> **Tried and reverted:** replacing the residual sum with `concat([own_embedding, attention_output])` fed through a deeper 3-layer `job_context_encoder`, plus an extra hidden layer in `score_net`, was tested (run `znx8m6mz`) on the theory that concatenation preserves more information than a residual add. It converged cleanly but to a **substantially worse** policy (mean reward margin vs. SJF fell from +1.14% to -17.51%, SJF's win rate rose from 3.86% to 41.34%) — more depth and parameters hurt here rather than helping, at this task's small scale. Reverted to the residual-sum version documented here.

**`COMPLETED` jobs are masked out as attention keys** (`key_padding_mask`), so their stale `size`/`wait_time`/`ttl` can't dilute other jobs' contextualized comparisons with irrelevant noise — a job finishing shouldn't change how two *other*, still-relevant jobs get compared to each other. This does **not** hide a completed job's own status from itself: each job's own `status` embedding still flows straight into its own per-job vector, so the actor still directly sees "this job is COMPLETED" when scoring whether to (invalidly) pick it — the mask only affects what other jobs use as comparison context, not what a job knows about itself. (Guarded against the all-jobs-completed edge case, which would otherwise produce an all-`True` mask and NaN the attention softmax.)

## Dual attention (job ↔ machine)

Adapted from **"Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning"** (Wang et al., [arXiv:2305.05119](https://arxiv.org/abs/2305.05119), a.k.a. DANIEL). That paper's Dual Attention Network alternates an *Operation Message Attention Block* (operations attend to precedence-linked operations within a job) and a *Machine Message Attention Block* (machines attend to each other via a "competition intensity" signal over shared candidate operations), then pools both into one global context feeding the actor/critic.

Our problem differs from FJSP in one key way: jobs here are single atomic tasks, not sequences of operations, so there's no precedence structure to exploit. The adaptation:

- **Job attention** (already existed before this change) already matches the paper's competition framing directly — with a shared machine pool, "which jobs are competing for machine time" is just "all currently-relevant jobs," which is exactly what job self-attention already models.
- **Machine attention (new)**: machines previously had a static, per-machine identity embedding (`nn.Embedding(n_machines, embedding_dim)`, no scheduling-state awareness at all). They now get real dynamic features — `utilization_now` (mean `usage/capacity` at the current tick) and `utilization_window` (same, averaged over the whole visible time window) — encoded through their own MLP, then attend to each other via multi-head self-attention. At `n_machines=1` this block is a structural no-op (nothing to attend to); it only does real work once `n_machines > 1`.
- **Job ↔ machine cross-attention (new)**: jobs attend to machine state and machines attend to job state, so both sides mutually inform each other before anything gets pooled or scored — previously machine state never reached the job embeddings at all, and machine choice was made purely from a fixed identity vector.
- **Global context** = pooled jobs + pooled machines + `queue_pressure`, matching the paper's `hG = avg(operations) ‖ avg(machines)`.

The actor's per-(machine, job) score now uses a genuinely state-dependent machine embedding (computed per batch element from that machine's actual current load) instead of a fixed per-machine-index vector shared across every state the policy has ever seen.

**Gated cross-attention (LayerScale).** The first version of this (run `ddsv1ozh`, tested at `n_machines=1` where the machine-attention half is a structural no-op) folded each cross-attention branch in at full strength immediately: `embeddings = LayerNorm(embeddings + cross_attn_output)`. Measured against the current best (`g231w4jg`): ties dropped 82.00%→75.74%, and *both* PPO wins (707→844) and SJF wins (193→369) rose together — the network was forced to fully absorb a not-yet-calibrated signal from step one, making the policy more decisive without being more accurate; mean reward margin fell from +1.14% to +0.97%.

Fixed by gating each branch with a learnable per-channel scale initialized to **zero** — `embeddings = LayerNorm(embeddings + gate * cross_attn_output)`, `gate = nn.Parameter(torch.zeros(embedding_dim))` — the LayerScale technique from vision transformers (Touvron et al., "Going Deeper with Image Transformers," 2021). At initialization the branch is a true no-op (training starts equivalent to the pre-dual-attention baseline), and the network only grows each gate where the cross-attention signal actually earns its keep, rather than being forced to use it uniformly everywhere from step one.

## Random per-job utilization (concurrent machine packing)

Every job used to claim **full machine capacity** on every resource (`usage[resource, :size] = max_capacity`, i.e. `255`), so a machine could only ever run one job at a time — exclusive occupation was an artifact of job generation, not a rule the scheduler enforced. `Machine.add_usage` ([machine.pyx](src/scheduling_simulator/core/machine.pyx)) already sums per-cell usage against capacity and only rejects an allocation that would overflow a cell, so nothing in the allocation logic actually required exclusivity.

`deeprm_generate_single_job_decreasing_size` ([creator.pyx](src/scheduling_simulator/core/creator.pyx)) now draws **one random utilization value per job**, uniformly in `[40%, 60%]` of `max_capacity` (narrowed from an initial `[10%, 50%]` — see below), and broadcasts it identically across every resource and every active time-slot (same flat shape as before — just no longer always `255`). Multiple jobs can now genuinely be bin-packed onto the same machine whenever their combined usage stays within capacity, turning this into a real bin-packing problem instead of pure single-tenant sequencing.

**This changed what information the policy needs.** A first PPO run trained fresh on the new generator (run `h0vwcu5e`) lost to SJF in **97.42%** of evaluation episodes (mean reward margin **-154%**) despite converging cleanly — not because training failed, but because **the feature set had no signal at all for how much of a machine a job would occupy**. SJF doesn't need that signal (it always picks the smallest pending job, regardless of machine occupancy), but a policy trying to reason about concurrent packing cannot, without it, tell the difference between a job that fits comfortably alongside what's already running and one that would overflow the machine.

This is the direct counterpart to `arrival`'s removal earlier: `arrival` was cut because it gave the policy foresight a reactive baseline structurally couldn't use, while capacity information is needed here because it's foresight a reactive baseline *doesn't need* (SJF's rule ignores occupancy) but the RL policy *cannot function without* now that the task allows concurrent packing. Three successive fixes closed most (not all) of the resulting gap — see `docs/experiment_log.md` #18-20 for the full numbers:

1. Added `usage` (and later `fit_slack = avg_free_capacity_now - usage`) directly as per-job input features. Recovered a large part of the gap (mean margin -154% → -52% → -25%), but always at a fixed weight the network had to bake into the embeddings themselves.
2. Added an explicit per-(machine, job) `fit` term (`free_capacity_now[m] - usage[j]`), concatenated straight into `score_net`'s input — closed more of the gap (-25% → -16%), but still a fixed-weight contribution regardless of situation.

## Fit gate (context-dependent capacity weighting)

The current design (see `docs/experiment_log.md` #21) separates "sequencing reasoning" from "packing reasoning" entirely, instead of mixing capacity info into the same embeddings job self-attention/pooling reason over:

- **`job_encoder` and every attention layer are capacity-blind.** `usage`/`fit_slack` were removed from the per-job input entirely (`job_input_dim` back to 20 — `status`, `size`, `wait_time`, `ttl`, `queue_pressure` only). The contextualized job/machine embeddings that reach the actor never saw a single capacity number.
- **`score_net`** scores every `(machine, job)` pair from `[global, machine, job]` — these same clean embeddings — exactly as it did before `usage` existed at all.
- **A separate `fit_gate_net`** (`Linear → ReLU → Linear → Sigmoid`, same `[global, machine, job]` input as `score_net`) outputs a per-pair gate in `[0, 1]`, learned purely from the clean embeddings.
- **The feature extractor still emits raw, unembedded `usage`/`free_capacity_now`** in its output tail (unchanged from the previous design), from which `fit[m, j] = free_capacity_now[m] - usage[j]` is computed exactly per pair.
- **Combined additively**: `pair_score = score_net([global,machine,job]) + fit_scale * gate([global,machine,job]) * fit`, where `fit_scale = nn.Parameter(torch.zeros(1))` — LayerScale-style zero init (the same pattern that fixed the earlier cross-attention regression), so training starts exactly equivalent to ignoring capacity info entirely and only grows its influence where it earns its keep.

The intent: instead of the network being forced to use a constant amount of capacity information everywhere (whether concatenated into `score_net`'s input or baked into the embeddings), it can learn *when* the fit signal should matter — e.g. weight it heavily when packing is tight and multiple jobs are competing for the same free capacity, but ignore it when a machine is nearly empty and any pending job obviously fits.

## Reward

`FailureSkipTimeWrapper` overrides the inner env's own reward function. Per tick, exactly one applies:

| Situation | Reward |
|---|---|
| All jobs completed | +100 |
| Hit `max_time` without finishing | -100 |
| Explicit skip | `-1 × count(PENDING or RUNNING jobs)` |
| Failed/invalid allocation attempt | `-10 + (-1 × active job count)` |
| Successful allocation | +1 |

This is a pure flow-time (total wait time) minimization objective — the reason Shortest-Job-First is such a strong baseline throughout the results below.

---

# Cluster Scheduling RL — Full Architecture Writeup

## 1. The problem, formalized

The task is: given a fixed number of jobs (`n_jobs`) competing for a fixed number of machines (`n_machines`), each with `n_resources` resource dimensions tracked over `n_time` future time slots, decide at every timestep whether to allocate a pending job to a machine or do nothing, so as to minimize overall wait/completion time.

This is formalized as a `gymnasium.Env`:

- **Observation**: a `Dict` space with per-job tensors (`jobs_usage`, `status`, `ttl`, `arrival`, `wait_time`, `scheduled_at`, `finished_at`, `size`), per-machine tensors (`machines_usage`, `machines_capacity`), and two scalars (`time`, `action_success`).
- **Action**: `Discrete(1 + n_machines * n_jobs)`. Action `0` is "do nothing this step." Action `a > 0` maps to `(machine_idx, job_idx)` via `machine_idx = (a-1) // n_jobs`, `job_idx = (a-1) % n_jobs` (implemented in Cython as `Cluster.action_from`), and the inverse mapping `1 + machine_idx * n_jobs + job_idx` is `Cluster.allocation_to_action`.
- **Reward**: a pluggable `RewardFunction`; currently a flow-time-style penalty (`-count of active/pending jobs` per real timestep, plus a penalty scaled by consecutive failures for invalid allocation attempts).

Everything downstream — the wrapper, the feature extractor, the policy — exists to make an RL agent tractable against this formalization.

---

## 2. Environment-level structure

### 2.1 `SchedulingEnviorment`

The core env wraps a Cython `Cluster` object that owns the actual simulation state. Each `step(action)`:

1. Decodes `action` into `(skip, machine_idx, job_idx)`.
2. If not skip, attempts `Cluster.allocate(machine_idx, job_idx)` — this can fail (e.g. insufficient capacity, job not pending) without raising, setting `observation.action_success = False`.
3. Recomputes the full observation and returns it, with `terminated = all jobs completed`.

Critically, a **failed allocation does not advance simulated time** — `elapsed = current['time'] - previous['time']` is `0` on a failed attempt. This is the fact that drives everything in the wrapper layer below.

### 2.2 `FailureSkipTimeWrapper`

Sits directly on top of `SchedulingEnviorment` and solves a specific pathology: if the policy is bad at finding valid allocations (especially early in training, or once masking is removed), it can spend its entire step budget on zero-time failed attempts and never make simulated progress, so episodes end via `TimeLimit` truncation rather than actually finishing.

The wrapper's step logic, in order:

1. Step the underlying env with the raw action.
2. Mask non-pending jobs' `jobs_usage` to a sentinel value (`256`) — a cheap way to signal "this job's resource profile is no longer relevant" to whatever consumes the observation next, since a completed/running job's usage array is not meaningful to reason about for future allocation decisions.
3. Increment an internal `_time_counter` (this is now the effective episode clock — decoupled from `Cluster`'s own `time`).
4. Check wrapper-level terminal conditions: all jobs completed → `+100` and done; `_time_counter >= max_time` → `-100` and done.
5. If the action was an explicit skip → return a shaped `_skip_time_reward` (a soft penalty proportional to `-1/size` summed over active jobs — smaller jobs waiting is penalized more per unit, encouraging clearing small jobs fast).
6. If the allocation failed → force one additional skip step (`env.step(0)`) so time actually advances, re-run the same masking/terminal checks, and return `1.2 ×` the skip-time reward (a slightly harsher penalty than a deliberate skip, since this cost was involuntary).
7. Otherwise (successful allocation) → reward `0.0` — a successful allocation is treated as reward-neutral; the model must ultimately be signaled by later steps' reduced active-job counts, not the allocation event itself.

This wrapper is what allows the pipeline to run **without action masking** — it doesn't prevent invalid actions, it bounds their cost.

---

## 3. `PointerFeaturesExtractor` — turning the observation into a feature vector

This is an SB3 `BaseFeaturesExtractor`. Its entire job is: observation dict in, one flat vector out, for the policy's downstream layers to consume. Internally it's structured as a five-stage pipeline, and every stage exists to solve a specific structural problem with the action space.

### 3.1 Why not just flatten everything into an MLP?

A flat MLP ending in `Linear(hidden, n_actions)` has to learn `n_actions = 1 + n_jobs * n_machines` independent output weight vectors. Nothing in that architecture encodes that action "(job 3, machine 0)" and action "(job 3, machine 1)" share a job, or that two actions targeting the same machine share machine-side context. Every action is scored from scratch, using the same shared hidden representation but with entirely separate final-layer weights. This scales badly (parameters grow with `n_jobs × n_machines`) and generalizes badly (nothing learned about job 3 vs. machine 0 transfers to job 3 vs. machine 1 except through the earlier shared hidden layers).

### 3.2 Stage 1 — per-entity encoders (weight-shared across jobs / across machines)

```
job_features     = concat(jobs_usage.flatten(resource×time), status_onehot, [ttl, arrival, wait_time, scheduled_at, finished_at, size])
machine_features = concat(machines_usage.flatten(resource×time), machines_capacity.flatten(resource×time))

job_emb     = job_encoder(job_features)         # same MLP applied to every job independently
machine_emb = machine_encoder(machine_features) # same MLP applied to every machine independently
```

Both encoders are small 2-layer MLPs (`Linear→ReLU→Linear`) mapping into a shared `embedding_dim` (64 by default). Because the *same* encoder weights are applied to every job (and separately, every machine), the network learns one reusable notion of "what makes a job look a certain way" rather than per-slot representations — this is the first source of parameter efficiency and the first step toward permutation-reasonable behavior (jobs in different array positions with the same underlying features get the same embedding).

`status` is one-hot encoded with `num_classes` derived dynamically from the observation space's declared bound (`status.high[0] + 1`), rather than hardcoded, so it doesn't silently break if the status enum's range changes.

### 3.3 Stage 2 — job self-attention

```
attended = LayerNorm(job_emb + MultiheadAttention(job_emb, job_emb, job_emb))
```

Standard post-norm residual self-attention block (4 heads). This lets each job's representation become *relative* rather than absolute: a job's embedding after this stage can encode "I am the smallest of the pending jobs" or "I've been waiting the longest relative to the others," which is information no per-job encoder could produce on its own since it only sees a single job's raw features.

### 3.4 Stage 3 — cross-attention onto machine state

```
cross = LayerNorm(attended + MultiheadAttention(query=attended, key=machine_emb, value=machine_emb))
```

Each (already job-contextualized) job embedding now attends onto the machine embeddings, absorbing "how do I relate to currently available capacity" before any pairwise scoring happens. Doing this once here, rather than inside the pairwise scorer, avoids redundant computation — without this stage, every one of the `n_machines × n_jobs` pairwise score computations would have to independently rediscover the same job-vs-machine relationship.

### 3.5 Stage 4 — exhaustive pairwise scoring (the "pointer" mechanism)

```
m_exp = machine_emb broadcast to (batch, n_machines, n_jobs, embedding_dim)
j_exp = cross       broadcast to (batch, n_machines, n_jobs, embedding_dim)

pair_emb = pair_scorer(concat(m_exp, j_exp, m_exp * j_exp))   # (batch, n_machines, n_jobs, embedding_dim)
```

Every possible `(machine, job)` combination gets its own embedding, computed by one shared `pair_scorer` MLP fed `[machine_embedding, job_embedding, elementwise_product]`. The elementwise product term gives the network a cheap, direct signal of alignment/compatibility between the two embeddings (if they encode similar resource-shape information, the product will highlight the overlap), without needing a full bilinear layer.

This is the actual mechanism that fixes the original structural problem: actions targeting the same job but different machines now literally share the `j_exp` half of their input tensor. Everything the network learns about "this job's shape" is reused identically across every machine it could go to — the network only has to learn the *machine-conditional adjustment* on top, not the whole scoring function from scratch per pair.

### 3.6 Stage 5 — the skip action and the value branch

```
skip_emb    = skip_encoder(concat(job_emb.mean(jobs), job_emb.max(jobs), current_time))
action_emb  = concat(skip_emb, pair_emb.flatten(machines×jobs))   # (batch, n_actions, embedding_dim)

value_state = value_encoder(concat(job_mean, job_max, machine_mean, machine_max, current_time))
```

`skip_emb` is deliberately built from a *pooled* (mean + max) summary of all jobs plus the current time, rather than any single job or pair — "do nothing" is a judgment about overall queue state, not about any specific allocation. `value_state` is a separate pooled summary (jobs and machines both) feeding the critic — the critic needs "how good is this state overall," which is a different question from "which specific action is best," so it gets its own dedicated path rather than sharing the per-action embeddings.

The extractor's final output is the concatenation `[flattened action_emb, value_state]`, of length `n_actions × embedding_dim + embedding_dim`. Note this is *not yet* scored — every action still has an `embedding_dim`-sized vector, not a scalar logit. Turning those into logits is the policy's job, not the extractor's.

---

## 4. `SchedulingPolicy` — wiring the extractor into SB3's actor-critic contract

SB3's `ActorCriticPolicy` normally flows: `features_extractor` (produces one flat vector) → `mlp_extractor` (splits/transforms into `latent_pi` and `latent_vf`) → `action_net` / `value_net` (plain `Linear` layers mapping those latents to actual logits / a scalar). Because our "flat vector" secretly contains structured per-action embeddings, three of these default pieces get overridden.

### 4.1 `SchedulingMlpExtractor` — a splitter, not an MLP

Despite the name (forced by SB3's internal contract — `_build_mlp_extractor` must produce something with `.latent_dim_pi`/`.latent_dim_vf` and `forward_actor`/`forward_critic`), this class does no computation. All the real work already happened in the features extractor.

```python
def forward_actor(self, features):
    return features[:, :n_actions * embedding_dim]     # the action-embedding chunk

def forward_critic(self, features):
    return features[:, n_actions * embedding_dim:]     # the pooled value_state chunk
```

It exists purely to satisfy SB3's expected interface while passing the two halves of the extractor's output through unchanged.

### 4.2 `PointerActionHead` — replaces the default `Linear(latent_dim_pi, n_actions)`

A plain linear action head would take the flattened action embeddings and apply one big matrix — which would immediately throw away the whole point of keeping per-action structure. Instead:

```python
action_embeddings = latent_pi.reshape(batch, n_actions, embedding_dim)
query = self.query(action_embeddings.mean(dim=1, keepdim=True))   # one pooled query over the current action set
keys  = self.key(action_embeddings)                                 # per-action key
logits = (query * keys).sum(dim=-1) * scale                         # (batch, n_actions)
```

This is a single attention-style scoring pass: rather than static fixed weights deciding "how important is this action," the query itself is derived from the current set of action embeddings, so the scoring adapts to what's actually on the table this step (e.g. if every pair embedding currently looks similar because most jobs are equally sized, the query reflects that).

### 4.3 `SchedulingValueNet` — replaces the default `Linear(latent_dim_vf, 1)`

Nothing pointer-specific here — a standard 2-hidden-layer MLP (`embedding_dim → 256 → 256 → 1`) operating on the pooled `value_state`. The critic doesn't need per-action detail, just a scalar state-value estimate, and the extractor already produced the right pooled representation for that.

### 4.4 Assembly in `SchedulingPolicy.__init__`

```python
super().__init__(
    ...,
    net_arch={'pi': [], 'vf': []},     # no default hidden layers — everything is in the custom heads above
    ortho_init=False,                    # orthogonal init assumes plain Linear layers; skip for custom heads
    features_extractor_class=PointerFeaturesExtractor,
    features_extractor_kwargs={'embedding_dim': embedding_dim},
    ...,
)
self.action_net = PointerActionHead(action_space.n, embedding_dim)
self.value_net  = SchedulingValueNet(embedding_dim)
```

`_build_mlp_extractor` is overridden to install `SchedulingMlpExtractor` rather than letting SB3 build its usual `MlpExtractor` from `net_arch` — this is what makes passing an empty `net_arch` safe, since nothing tries to actually build layers from it.

---

## 5. Training algorithm: PPO (masked or unmasked)

The policy above is algorithm-agnostic — it plugs into either `stable_baselines3.PPO` or `sb3_contrib.MaskablePPO` unchanged, since both just call `self.policy(obs)` / `self.policy(obs, action_masks=...)` and expect logits + value back.

- **With `MaskablePPO`**: `action_masks()` on the env zeroes out invalid actions' probability before sampling, guaranteeing every sampled action is valid. This removes the entire "policy wastes steps on invalid actions" problem at the source but requires `ActionMasker` wrapping and an `action_masks()` implementation kept in sync with `Cluster`'s actual validity rules.
- **With plain `PPO`** (current setup, per your earlier request to drop masking): invalid actions can be sampled freely. The `FailureSkipTimeWrapper` bounds the damage (converts a failed attempt into one skipped tick plus a penalty, rather than a fully wasted step), and the reward function's `-active_jobs × (1 + consecutive_failures)` term is meant to teach the policy to avoid repeat failures — but nothing architecturally prevents the `PointerActionHead` from assigning high logits to invalid pairs. This is a real capability gap versus the masked version, traded for less code/complexity.

Key PPO hyperparameters as currently set and why:

| Param | Value | Rationale |
|---|---|---|
| `learning_rate` | `3e-4` | Standard PPO default; fine as a starting point. |
| `n_steps` | `512` | Rollout length per env before an update; smaller than default `2048` to get more frequent updates given short-ish episodes. |
| `gamma` | `1.0` → consider `0.99` | Undiscounted return over a ~250-step horizon inflates value-target variance; this was flagged as a likely source of eval instability. |
| `ent_coef` | `0.005`–`0.02` | Must be a fixed float (SB3 doesn't support scheduled `ent_coef` natively); higher values keep exploration alive longer at the cost of eval-score noise from a still-stochastic policy. |
| `gae_lambda` | `0.95` | Standard GAE smoothing between full-Monte-Carlo and pure bootstrap advantage estimates. |

---

## 6. What this architecture buys you, and where it costs you

**Wins:**
- Parameter count scales with `n_jobs + n_machines`, not `n_jobs × n_machines`.
- Learning about one job's representation transfers automatically to every machine it could be paired with (and vice versa), because the pairwise scorer shares weights across all pairs.
- Attention stages give the network relative/contextual information (queue-position-aware, capacity-aware) that a per-entity-only encoder can't produce.

**Costs:**
- Five sub-networks instead of one flat MLP — more surface area for shape bugs (several of the tracebacks earlier in this thread trace directly to the `SchedulingMlpExtractor`/`PointerActionHead` contract with SB3 internals).
- More compute per forward pass than a comparably-sized flat MLP (attention is quadratic in sequence length, though at `n_jobs=10` this is cheap).
- The architecture is not validity-aware by construction — it buys sample efficiency and generalization, not "never proposes an invalid action." That's an orthogonal concern currently handled (partially) by the reward function and the `FailureSkipTimeWrapper`, not by the network itself.

**The honest baseline check worth running:** train a plain `MultiInputPolicy` + `PPO` on the same fixed config for the same step budget, and compare `
`. If the pointer architecture isn't clearly ahead, the added complexity here isn't paying for itself yet on this problem size (`n_jobs=10, n_machines=1` is small enough that a flat MLP's parameter inefficiency may not matter much in practice).

---

## 7. PPO vs. SJF baseline — evaluation results

Config: `n_machines=1, n_jobs=10, n_resource=3, n_time=10, max_capacity=255`, 100 eval episodes, deterministic policy evaluation (`model.predict(..., deterministic=True)`). SJF baseline: `genial-serenity-289` (`0v9hol5p`).

| PPO variant | W&B run | Metric | Ties | PPO better | SJF better | Mean margin |
|---|---|---|---|---|---|---|
| Regular reward | `wobbly-microwave-295` (`emwaqlmc`) | `eval/reward` | 79% | 15% | 6% | +0.94% |
| Regular reward | `wobbly-microwave-295` (`emwaqlmc`) | `eval/avg_wait_time` | 79% | 15% | 6% | — |
| Regular reward | `wobbly-microwave-295` (`emwaqlmc`) | `eval/avg_completion_time` | 79% | 15% | 6% | -0.51% |
| Terminate-on-failure reward | `firm-serenity-297` (`crv7ystm`) | `eval/reward` | 75% | 16% | 9% | +0.67% |
| Terminate-on-failure reward | `firm-serenity-297` (`crv7ystm`) | `eval/avg_wait_time` | 75% | 16% | 9% | -0.49% |
| Terminate-on-failure reward | `firm-serenity-297` (`crv7ystm`) | `eval/avg_completion_time` | 75% | 16% | 9% | -0.36% |

Both variants clearly beat the SJF baseline (negative margin on wait/completion time = PPO is faster; positive margin on reward = PPO scores higher) and land within normal run-to-run variance of each other — the terminate-on-failure reward change (episode ends immediately on an invalid allocation with the worst possible remaining-horizon reward, instead of burning a tick and continuing) is not a clear win or regression at this scale.
