# The current model, step by step (V3, two-resource environment)

This is the network used by PPO runs `pfrw14vh` and `vy9asra5`: the M9 network (fit attention with a feasibility mask, concat + project, masked job self-attention) plus three additions (per-resource fit inputs, pair features, a dynamic machine embedding) and a new training setup.

Code files:

- **Feature extractor:** [`new_features.py`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py), the class `StatusSizeWaitTtlCapacityFeaturesExtractor`.
- **Actor and critic:** [`new_sched.py`](../src/scheduling_simulator/expiremnt/policy/new_sched.py).
- **Training:** [`train_runner.py`](../src/scheduling_simulator/expiremnt/runners/train_runner.py) and [`example.py`](../src/scheduling_simulator/expiremnt/example.py).

Line numbers are for the commit that adds this file. "Why it could help" is a hypothesis, not a measured effect, unless the evidence line says otherwise.

## Evidence so far

- **Measured on E4 (usage 63 / 128 / 255), against SJF:** M5 −45.8% → M6 (fit attention) −37.0% → M7 (+ job self-attention) −16.9% → M9 (+ feasibility mask) +2.3%.
- **Measured on the two-resource environment (this branch):** PPO `vy9asra5` +2.37% reward margin over SJF, optimal +7.24% (5000 episodes, same job sets). PPO closes about 33% of the SJF-to-optimal gap.
- **Not ablated one by one:** per-resource inputs, pair features, the dynamic machine embedding and the training changes. Their listed benefits are untested hypotheses.

## Network

### 0. Input scaling
- **What:** size, wait time and remaining time are divided by 110 (the step limit), remaining time and arrival are capped to [0, 1], and arrival becomes "ticks until arrival". Usage and free room are divided by 255 and clamped.
- **Code:** [`new_features.py:351`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) size, `:356` wait, `:361` remaining time, `:367` arrival, `:380` job usage, `:391` machine room.
- **Why it could help:** every input is on a similar 0 to 1 scale, so no input dominates the first layer only because its raw numbers are larger, and training is steadier.

### 1. Per-job input (status embedding + 4 numbers)
- **What:** each job gets a learned 16-number embedding of its status (not arrived, pending, running, completed), plus size, wait, remaining time and arrival, 20 numbers in all. Usage is left out on purpose.
- **Code:** [`new_features.py:142`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) embedding table, `:148` input size, `:413` lookup, `:423` concatenation.
- **Why it could help:** the status is a category, so a learned vector lets the model represent states in its own way. Keeping usage out of the encoder means the encoder describes only a job's timing, and usage joins later where it is compared with the machine's room.

### 2. Job encoder
- **What:** Linear 20 → 256, LayerNorm, ReLU, Linear 256 → 256, LayerNorm, ReLU, with the same weights for all 10 jobs.
- **Code:** [`new_features.py:156`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) definition, `:444` applied to all jobs at once.
- **Why it could help:** shared weights give every job the same description rule, so the model is independent of job order (checked: shuffling the jobs only reorders the outputs, to about 5e-7). It also gives 10 times more training signal to one encoder than separate encoders would. LayerNorm keeps activations stable.

### 3. Per-resource fit inputs (addition 1)
- **What:** the fit attention gets one number per resource for each job (its peak usage over time) and for the machine (its tightest free room over time), 3 numbers each, instead of one number each.
- **Code:** [`new_features.py:99`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) input width, `:375` reduction over time only, `:380` and `:391` the per-resource values.
- **Why it could help:** the environment now gives each resource its own usage (resources 0 and 1 independent, resource 2 always 0). A single "max usage" number cannot tell a job that is heavy on resource 0 from one heavy on resource 1, and which one fits depends on what is free on each resource. Per-resource numbers keep that information.

### 4. Fit attention with feasibility mask
- **What:** three small networks (Linear → 64, ReLU, Linear → 64) turn the job's usage into a query and the machine's room into a key and a value. The score is Q · K / 8, passed through a sigmoid. The weight is set to 0 unless the job fits on every resource. The output is weight × V, 64 numbers per job.
- **Code:** [`new_features.py:181`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) the networks, `:471` query, key, value, `:475` score, `:476` sigmoid, `:482` and `:486` the fit mask, `:488` weighted output.
- **Why it could help:** "does this job fit?" becomes an explicit signal instead of something the network has to infer. The sigmoid (not a softmax) gives each job its own answer even with one machine. The hard mask removes any chance of the model treating a job that cannot run as runnable.
- **Evidence:** fit attention M5 → M6 improved −45.8% → −37.0%; the mask M7 → M9 improved −16.9% → +2.3% (E4).

### 5. Concat and project
- **What:** the 256-number job summary and the 64-number fit vector are joined (320 numbers), then Linear → LayerNorm → ReLU back to 256.
- **Code:** [`new_features.py:193`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) definition, `:490` applied.
- **Why it could help:** the network learns how much of the fit information to mix into each summary. Concatenation kept all the information and worked better than multiplying the summary by the fit vector (M8 was worse than M7, within seed noise).

### 6. Job self-attention
- **What:** 4-head self-attention across the 10 jobs, with finished jobs masked out as keys and a guard for the last step, then residual connection and LayerNorm. There is no positional encoding.
- **Code:** [`new_features.py:206`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) attention layer, `:212` norm, `:501` to `:503` mask and guard, `:505` to `:512` attention, residual and norm.
- **Why it could help:** SJF's rule is relative ("the shortest job that fits"), so each job's score depends on the other jobs. Self-attention lets a summary include how a job compares with the rest. Masking finished jobs stops their stale numbers from diluting the comparisons.
- **Evidence:** M6 → M7 improved −37.0% → −16.9% (E4).

### 7. Attention pooling
- **What:** a small network (Linear 256 → 32, tanh, Linear 32 → 1) scores each job, a softmax over the jobs gives weights, and the global summary is the weighted sum followed by Linear, LayerNorm and ReLU.
- **Code:** [`new_features.py:218`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) scoring network, `:235` output layer, `:518` to `:535` applied.
- **Why it could help:** it compresses a changing number of live jobs into one fixed-size description of the queue, and the learned weights let the model focus on the jobs that matter now.

### 8. Pair features (addition 2)
- **What:** for each machine and job, 9 numbers computed from the free grid: pending flag, fits now, ticks until it fits (found by testing every start offset 0 to 9), finish time, headroom on each of the 3 resources, the job's largest resource share, and a packing score. A Linear layer (9 → 64) projects them.
- **Code:** [`new_features.py:266`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) computed in `_pair_features`, `:559` appended to the output; [`new_sched.py:137`](../src/scheduling_simulator/expiremnt/policy/new_sched.py) projection layer, `:340` applied.
- **Why it could help:** the best schedule often depends on when a job can start and what it leaves free, not only on whether it fits now. The delay, finish time and headroom give the scorer that timing and packing information directly, as exact numbers, instead of leaving the network to learn them from the raw grid.

### 9. Dynamic machine embedding (addition 3)
- **What:** the machine's free grid (3 resources × 10 ticks) divided by 255 passes through a row encoder shared by the three rows (10 → 64 → 64), the rows are joined (192) and passed through Linear → ReLU → Linear to 256. The last layer starts at zero, and the result is added to the static machine embedding.
- **Code:** [`new_features.py:252`](../src/scheduling_simulator/expiremnt/feature_extractor/new_features.py) to `:264` networks and zero init, `:565` computed; [`new_sched.py:273`](../src/scheduling_simulator/expiremnt/policy/new_sched.py) added to the machine embedding.
- **Why it could help:** a fixed machine vector carries no information about what is running. A state-dependent one lets the scorer see the machine's current and upcoming load. Starting the last layer at zero means training begins exactly like M9 and only adds this signal as it proves useful.

### 10. Scorer and skip head
- **What:** for each job, the global summary, the machine embedding, the job summary and the 64 pair numbers (832 in total) go through Linear → 128, ReLU, Linear → 128, ReLU, Linear → 1. A separate skip head (256 → 64 → 1) reads only the global summary. The skip score and the 10 job scores form 11 logits.
- **Code:** [`new_sched.py:146`](../src/scheduling_simulator/expiremnt/policy/new_sched.py) machine embedding table, `:163` score network, `:186` skip head, `:361` job scores, `:369` skip score, `:377` concatenation of logits, `:524` head wired into the policy.
- **Why it could help:** one scorer applied to every job gives a consistent rule and many training examples. A dedicated skip score lets the model decide to wait without competing with a specific job.

### 11. Critic
- **What:** a separate value network (256 → 256 → 256 → 1) on the global summary.
- **Code:** [`new_sched.py:386`](../src/scheduling_simulator/expiremnt/policy/new_sched.py) definition, `:536` wired in.
- **Why it could help:** a value estimate that is accurate (explained variance 0.98 in the runs) gives PPO a low-noise learning signal for the actor.

## Training

### 12. PPO setup
- **What:** PPO with clipped surrogate loss (clip 0.2), GAE advantages (γ = λ = 0.995), value weight 0.5, gradient clipping 0.5, batch 64, 5 epochs per update, 8 parallel environments × 512 steps per update.
- **Code:** [`train_runner.py:123`](../src/scheduling_simulator/expiremnt/runners/train_runner.py) to `:147`; environment count and step settings in [`example.py:41`](../src/scheduling_simulator/expiremnt/example.py).
- **Why it could help:** 8 environments give more varied experience per update, which lowers gradient noise.

### 13. Learning-rate decay and KL early stopping
- **What:** the learning rate (2e-5) decays linearly to 5% of its start, and an epoch stops early when the policy moves more than `target_kl` (0.03) from the old one.
- **Code:** [`train_runner.py:38`](../src/scheduling_simulator/expiremnt/runners/train_runner.py) schedule, `:128` applied, `:132` `target_kl`.
- **Why it could help:** smaller late updates and a cap on policy jumps protect against the late collapse seen in earlier runs.
- **Evidence:** run `cdhnmmw1` (same network, higher constant learning rate, no KL limit) collapsed and lost 4965 of 5000 episodes to SJF, while `pfrw14vh` with these settings did not.

### 14. Entropy schedule
- **What:** the entropy bonus is annealed from 0.05 to 0.01 over training.
- **Code:** [`train_runner.py:162`](../src/scheduling_simulator/expiremnt/runners/train_runner.py), implemented in [`entconf_callback.py:12`](../src/scheduling_simulator/expiremnt/callbacks/entconf_callback.py).
- **Why it could help:** more exploration early and a more decisive policy late. The M9 vs M10 comparison on E4 showed no effect, so this is not expected to matter much.

### 15. Best-checkpoint evaluation
- **What:** a 50-episode deterministic check runs during training, the best checkpoint is saved, and the final 5000-episode test uses it instead of the last model.
- **Code:** [`train_runner.py:155`](../src/scheduling_simulator/expiremnt/runners/train_runner.py) checks, `:181` to `:184` loads the best model.
- **Why it could help:** the last model can be worse than an earlier one. In `vy9asra5` the best check (−80.4) came at 1.6M steps and the last (−91.3) at 2M.
- **Caveat:** choosing the best of a few small checks favours the result a little, so a second seed is needed to confirm the margin.
