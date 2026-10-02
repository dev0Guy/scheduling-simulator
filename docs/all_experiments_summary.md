# All Experiments — Summary

A readable, chronological summary of every experiment run in this project, what was tried, why, and what happened. For full detail (exact configs, run IDs, per-metric numbers) see [docs/experiment_log.md](experiment_log.md); this document is the narrative version.

The project has two phases on two different tasks:

- **Phase 1 (sections A-F below):** single-machine, exclusive-occupation scheduling — each job fully occupies the one machine while running. PPO's job here is pure sequencing: which pending job to run next.
- **Phase 2 (section G below):** the task changed — jobs now claim only a random fraction of a machine's capacity, so multiple jobs can run concurrently on one machine (bin-packing). This is a materially harder, different problem, and results from Phase 1 are not directly comparable to Phase 2.

---

## A. Making the numbers trustworthy (methodology)

Before any experiment result could be trusted, three infrastructure bugs had to be fixed:

1. **Stochastic evaluation bug.** Evaluation was calling `model.predict(obs, deterministic=False)`, so comparing the *same* trained model against itself twice could swing from +1.03% to -6.44%. Switched to `deterministic=True` — this was the single most impactful fix in the project; several early "wins" and "regressions" turned out to be pure noise once corrected.
2. **Job generator correlation bug.** `arrival_time` and `size` were both derived from the same jittered value, making them near-perfectly anti-correlated — "arrived first" and "smallest" were almost always the same job, letting SJF win "for free" as a data artifact rather than a real task property. Decoupled with independent jitter.
3. **Sample size.** Bumped evaluation from 100 to 5000 episodes, and fixed W&B's `history()` calls to request enough samples to avoid its silent ~500-point subsampling cap.

## B. Task-shape experiments (early, mostly reverted)

- **Wider arrival spread** (hypothesis: sparser arrivals create more decision points) — regressed; reverted.
- **More machines/jobs** (`n_machines=1→3`, `n_jobs=10→15`) — PPO lost badly (92% SJF win rate) in a partial run; stopped and reverted before completion, since easier levers hadn't been tried yet.

## C. Feature-extractor experiments — the core arc of Phase 1

This sequence is the heart of the project's early improvement:

1. **Add `arrival_time`** (time-until-arrival for not-yet-PENDING jobs) — first genuine win (+1.03% mean margin). SJF is structurally blind to this; PPO isn't.
2. **Drop `ttl`** (theory: redundant with `size`) — badly wrong, regressed to -14.67%. `ttl` is the only signal for "how soon does the RUNNING job free the machine."
3. **`queue_pressure`, global-only** (active-job-count/n_jobs fed only to the pooled critic embedding) — net-negative; made the policy more reactive without being more accurate.
4. **`queue_pressure`, global + per-job** (same signal also reaches the per-job scoring vector, where the decision is actually made) — fixed it. New best at the time.
5. **Job self-attention (unmasked)** — jobs compare directly against each other instead of only a blurred pooled average. Lowest SJF-win-rate so far, but also lowered PPO's own win rate (more conservative, not more accurate).
6. **Job self-attention with `COMPLETED` masked out of attention keys** — fixed #5's side effect. A finished job's stale data was diluting other jobs' comparisons. **This became the long-standing best config**: 82.00% ties, PPO wins 14.14%, SJF wins 3.86%, mean margin **+1.14%**.
7. **Deeper job-context encoder + deeper score_net** (more capacity/depth) — large regression (-17.51%). More parameters hurt at this small task scale.

## D. Reward-function experiment

- **Terminate on failed allocation** with a maximally negative terminal reward — roughly neutral, not a clear win. Reverted to standard (non-terminating) failure handling.

## E. PPO hyperparameter experiments

- **Higher entropy** — no meaningful effect; ruled out exploration as the bottleneck.
- **More training steps** (2M→4M) — no effect; the reward curve was already flat well before 2M. Confirmed convergence, not undertraining.
- **Higher gamma/gae_lambda** (0.995/0.995 → 0.999/0.99) — mild regression (+1.14%→+0.61%). Reverted.

## F. Optimal-solver ceiling analysis

Built an exact bitmask/subset dynamic program (Pareto-frontier dominance pruning) that computes the provably optimal schedule for this problem class (`1 | r_j | ΣC_j`), exploiting that the simulator already exposes every job's arrival/size up front. This gave a true ceiling to measure against, not just a PPO-vs-SJF comparison:

- PPO matches the literal optimal schedule on **90.68%** of episodes vs. SJF's **80.96%** — a much bigger, more meaningful gap than the ~82% PPO/SJF tie rate suggested (a tie doesn't distinguish "both optimal" from "both made the same mistake").
- PPO closes **~70-73%** of the reward/wait-time/completion-time gap between SJF and true optimal.

This remains the strongest validation of the Phase 1 policy: it isn't just beating a heuristic that might itself be mediocre — it's closing most of the way to a real mathematical ceiling.

### Dual attention network (Phase 1's final architecture push)

Adapted from *"Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning"* (Wang et al., arXiv:2305.05119). Added a real dynamic machine encoder (previously a static per-machine identity embedding with zero load-awareness), machine self-attention, and job↔machine cross-attention.

- **Ungated, full-strength cross-attention** — regressed. Made the policy "more decisive, not more accurate": both PPO's win rate *and* SJF's win rate rose together (707→844, 193→369). Mean margin fell to +0.97%.
- **Gated cross-attention (LayerScale)** — each branch scaled by a learnable per-channel gate initialized to zero, so it starts as a no-op and only grows where it earns its keep (Touvron et al., 2021). Fixed the loss-rate problem (lowest SJF win rate of any dual-attention variant, 3.16%) but only ran at half the training budget of everything else it was compared against, so it landed slightly below the Phase-1 best on win rate/margin (+0.79% vs +1.14%) — not a fair apples-to-apples test yet.

---

## G. Phase 2 — concurrent job packing (the task changes)

The cluster generator changed: jobs previously always claimed full machine capacity (`usage = max_capacity`), so a machine could only ever run one job — that exclusivity was an artifact of job generation, not an allocation rule (`Machine.add_usage` already summed per-cell usage against capacity and could have supported sharing all along). The generator was changed to draw one random utilization value per job, turning this into genuine bin-packing.

**First attempt was a severe regression.** With no changes to the feature set, PPO trained cleanly (loss curves converged normally) but **lost to SJF in 97.42% of episodes** (mean margin -154.14%). Root cause: nothing in the observation told the policy how much of a machine's capacity a job would claim. SJF doesn't need that (its rule — smallest job first — ignores occupancy by construction), but a policy trying to reason about concurrent packing cannot function without it.

From there, four successive interventions each closed part of the gap, all moving every metric in the same direction:

| Step | Change | Mean reward margin | SJF win rate | PPO win rate | Ties |
|---|---|---|---|---|---|
| 0 | (no fix — diagnostic baseline) | -154.14% | 97.42% | 0.76% | 1.82% |
| 1 | + `usage` feature (single scalar per job, how much capacity it claims) | -52.43% | 85.92% | 8.58% | 5.50% |
| 2 | + longer training (1M→2M) + explicit `fit_slack` feature + narrowed utilization curriculum (`[10-50%]`→`[40-60%]`, easier packing regime) | -25.33% | 79.34% | 9.84% | 10.82% |
| 3 | + explicit per-(machine,job) fit term fed directly into the actor's scorer (not just as one more embedded input) | **-15.77%** | 70.86% | 13.20% | 15.94% |

**Where it stands:** PPO has closed roughly 90% of the reward-margin gap from the initial broken attempt, purely through feature/curriculum/training-budget changes — no fundamental architecture redesign yet. It is **not yet beating SJF** on the concurrent-packing task. Each step's improvement has been smaller than the last (diminishing returns), which suggests either more of the same (further curriculum narrowing, more training) will keep closing the gap slowly, or the remaining distance needs a more structural change (e.g., an architecture built around explicit subset/packing selection rather than per-pair scoring).

---

## What the pattern across everything says

Two lessons recur across both phases:

1. **Information placed exactly where the decision is scored beats general capacity.** Every win in this project (`arrival_time`, per-job `queue_pressure`, masked self-attention, `usage`, `fit_slack`, the explicit per-pair fit term) gave the network a specific, decision-relevant fact closer to where the action gets chosen. Every loss from "more capacity" (deeper networks, unmasked/ungated attention, higher discount factors) added general power without adding the *right* information, and reliably backfired at this problem's scale.
2. **A changed task resets what "current best" means.** Phase 1's best configuration (`g231w4jg`, +1.14% margin) is meaningless as a target once the task itself changed to concurrent packing — the features and architecture that won on sequencing had to be re-earned from scratch once the decision problem got harder. Roughly half of all "reasonable-sounding" ideas tried across the whole project turned out to be regressions once actually measured; the discipline of testing everything against a fixed, deterministic 5000-episode evaluation (rather than trusting intuition) is what made the real wins identifiable at all.
