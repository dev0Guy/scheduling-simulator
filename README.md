![Tests](https://github.com/dev0Guy/scheduling-simulator/actions/workflows/tests.yml/badge.svg?branch=main)
[![Coverage](https://codecov.io/gh/dev0Guy/scheduling-simulator/branch/main/graph/badge.svg)](https://codecov.io/gh/dev0Guy/scheduling-simulator)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/github/license/dev0Guy/scheduling-simulator)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Base Status
-----------
We've created 3 Environment for scheduling, which implement the `ClusterABC` with `JobCollection` and `MachineCollection` protocols:
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
