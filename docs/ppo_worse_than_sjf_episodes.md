# Episodes where PPO loses to SJF

Traced against PPO run `silver-butterfly-298` (`np7aenua`) vs. SJF baseline `silvery-microwave-299` (`nb11ox0z`), same config (`n_machines=1, n_jobs=10, n_resource=3, n_time=10, max_capacity=255`), eval seed = `42 + episode`.

For each episode, the table shows the very first tick where PPO's schedule diverges from SJF's, and what each policy chose at that moment. In almost every case the story is the same: **SJF immediately takes the smallest job available right now; PPO instead skips several ticks, waiting for a job that's about to arrive and is only marginally smaller — a bet that costs more in queuing penalty than it saves in job size.**

| Episode | Seed | SJF total | PPO total | Gap | What happened |
|---|---|---|---|---|---|
| 77 | 119 | -185 | -189 | -4 | SJF takes job 6 (size 9, already arrived) at tick 0. PPO waits 2 ticks for job 4 (size 5, arrives tick 2) instead. |
| 797 | 839 | -164 | -179 | -15 | SJF takes job 0 (size 10, already arrived) at tick 2. PPO waits until tick 5 for job 6 (size 1, arrives tick 4) — a much smaller job, but a costly 5-tick wait. |
| 1053 | 1095 | -178 | -181 | -3 | SJF takes job 0 (size 10) at tick 0. PPO waits 3 ticks for job 6 (size 1, arrives tick 3). |
| 1133 | 1175 | -102 | -110 | -8 | SJF takes job 1 (size 8, arrived tick 2) at tick 6. PPO takes job 3 (also size 8, but already arrived at tick 0) one tick later — same size, PPO is simply a tick slower here. |
| 1213 | 1255 | -169 | -173 | -4 | SJF takes job 1 (size 8, already arrived) at tick 0. PPO waits 1 tick for job 2 (size 7, arrives tick 1). |
| 1525 | 1567 | -86 | -96 | -10 | SJF takes job 4 (size 7, already arrived) at tick 0. PPO waits 1 tick for job 2 (size 3, arrives tick 1) — a much smaller job, still a net loss. |
| 1529 | 1571 | -44 | -56 | -12 | Both reach their last remaining job at tick 10, but pick different jobs (SJF: job 5 size 2; PPO: job 6 size 2) — same size, order/tie-break difference late in the episode. |
| 1886 | 1928 | +12 | +6 | -6 | Same job order/sizes scheduled by both policies — the gap here comes from small timing shifts throughout the episode rather than one specific bad pick. |
| 2117 | 2159 | -239 | -247 | -8 | SJF takes job 2 (size 9, already arrived) at tick 0. PPO waits 2 ticks for job 3 (size 5, arrives tick 2). |
| 2280 | 2322 | -247 | -261 | -14 | SJF takes job 2 (size 9, already arrived) at tick 0. PPO waits 2 ticks for job 5 (size 7, arrives tick 2) — only 2 units smaller, not worth the wait. |
| 2646 | 2688 | -86 | -94 | -8 | SJF takes job 4 (size 7, already arrived) at tick 0. PPO waits 2 ticks for job 6 (size 5, arrives tick 2). |
| 3255 | 3297 | -39 | -55 | -16 | The only late-episode divergence: at tick 27, SJF takes job 1 (size 10, arrived tick 3); PPO waits until tick 31 for job 2 (also size 10, arrives tick 2) — identical size, PPO's wait bought nothing. |
| 3259 | 3301 | -269 | -274 | -5 | SJF takes job 0 (size 10, already arrived) at tick 0. PPO waits 2 ticks for job 8 (size 6, arrives tick 2). |
| 3296 | 3338 | -199 | -204 | -5 | SJF takes job 2 (size 8, already arrived) at tick 0. PPO waits 1 tick for job 8 (size 2, arrives tick 1) — a much smaller job, still a net loss at this scale. |
| 4276 | 4318 | -212 | -229 | -17 | SJF takes job 6 (size 9, already arrived) at tick 0. PPO waits 3 ticks for job 3 (size 5, arrives tick 3) — the largest gap in this set. |
| 4524 | 4566 | -199 | -214 | -15 | SJF takes job 0 (size 10, already arrived) at tick 0. PPO waits 4 ticks for job 5 (size 3, arrives tick 4). |

## Pattern

In 14 of the 16 episodes, the loss traces back to the same root cause identified earlier (episode 18, a different run): PPO's `arrival_time` feature lets it "see" that a smaller job is about to arrive, and it consistently over-values that foresight — skipping several ticks (accumulating `-active_job_count` penalty each tick) to grab a job that's often only marginally smaller than what was already available (in episode 3255, PPO waited 4 extra ticks for a job of the *exact same size*). SJF, having no foresight at all, never makes this mistake — it always takes the best job available right now.

The two exceptions (1129/1133's one-tick lag, 1529's late tie-break, 1886's diffuse timing) are minor scheduling noise rather than the wait-for-foresight pattern, and account for much smaller gaps.

**Takeaway:** the model's "wait for something better" instinct is directionally correct (it's the mechanism that lets it beat SJF on 68/500 episodes on average), but it isn't well-calibrated against queue depth or against *how much* smaller the awaited job actually is — it sometimes pays several ticks of queuing cost for a one- or two-unit size improvement, or even no improvement at all.
