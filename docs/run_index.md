# Run index: W&B run id to description

All 88 runs in the W&B project `dev0guy/cluster-scheduling-simulator`, oldest first. Dates are the W&B creation time in UTC (local time is 3 hours later). Descriptions come from the experiment log, the READMEs and the sessions that ran them. Where nothing in the docs or the sessions says what a run was, the row says so and gives only what W&B records.

## Environments

| Label | Job usage |
|---|---|
| E1 | every job uses 255 on every resource (one job at a time) |
| E2 | one value per job, uniform from 25 to 127 (10% to 50% of 255), same on every resource |
| E3 | like E2 but 40% to 60% of capacity |
| E4 | one value per job from 63 / 128 / 255, same on every resource |
| ENV-D | each resource uses one level (64 / 128 / 255); one random resource uses a different level from the other two |
| ENV-2R | resources 0 and 1 each drawn independently from 64 / 128 / 255; the last resource is always 0 |
| ENV-3R | all three resources drawn independently from 64 / 128 / 255 |

Margin = (PPO − SJF) / |SJF| on mean reward over 5000 deterministic episodes unless stated. Model labels M0 to M11 are defined in `docs/experiment_tables.md`.

## Early runs (Aug 28 to Sep 9, E1, before the log's numbered experiments)

These predate the generator and evaluation fixes in the experiment log (section 0). Nothing in the docs identifies them individually.

| Run id | W&B name | Date | Type | Note |
|---|---|---|---|---|
| `3lm3wkgs` | SJF | 08-28 | SJF baseline | not described |
| `whi7m72h` | Random | 08-28 | random-action baseline | not described |
| `33xw8fke` | Pointer-PPO | 08-28 | PPO, pointer-network policy | not described; 754k steps |
| `5mhwaof0` | classic-jazz-243 | 08-30 | PPO | not described; 754k steps |
| `zwc4b4s1` | eager-shape-248 | 08-31 | PPO | not described; 1.0M steps |
| `5bxuc7qo` | breezy-puddle-256 | 09-01 | PPO | not described; 2.0M steps |
| `ojnce1kn` | hearty-flower-261 | 09-02 | PPO | not described; 754k steps |
| `ct1xocum` | smooth-valley-262 | 09-02 | PPO | not described; 754k steps |
| `yqygsy8h` | devoted-fire-263 | 09-03 | PPO | not described; 754k steps |
| `3uqwxwp5` | dry-dust-264 | 09-03 | SJF baseline | not described |
| `h7csu44e` | ancient-bird-265 | 09-03 | SJF baseline | not described |
| `4ee29ngf` | ethereal-star-266 | 09-03 | SJF baseline | not described |
| `pmkkffc6` | wise-monkey-267 | 09-03 | PPO | not described; 754k steps |
| `6gvy1vss` | woven-plasma-268 | 09-03 | PPO | not described; 754k steps |
| `mzuwzwbs` | graceful-smoke-269 | 09-03 | SJF baseline | not described |
| `2nu56wyj` | celestial-darkness-270 | 09-03 | PPO | not described; 1.5M steps |
| `otins5p2` | ethereal-sunset-272 | 09-04 | PPO | not described; 2.0M steps |
| `tuxlxll8` | wild-snowflake-274 | 09-06 | PPO | not described; 2.0M steps |
| `0v5all7l` | celestial-capybara-275 | 09-07 | SJF baseline | not described |
| `kcjm220o` | flowing-puddle-277 | 09-07 | PPO | not described; 2.0M steps |
| `edhor21f` | rose-resonance-280 | 09-07 | PPO | not described; 2.0M steps |
| `fxebp9d5` | honest-donkey-281 | 09-07 | SJF baseline | not described |
| `sffjbsmb` | quiet-armadillo-282 | 09-07 | PPO, 4.0M steps | likely the "more training steps" test (log #12), not confirmed |
| `gq1zl2mz` | smart-spaceship-283 | 09-07 | SJF baseline | not described |
| `09mamkjo` | restful-dew-286 | 09-08 | PPO | not described; 2.0M steps |
| `b8p14bdu` | ethereal-durian-288 | 09-08 | SJF baseline | not described |
| `0v9hol5p` | genial-serenity-289 | 09-08 | SJF baseline | baseline for the terminate-on-failure comparison (README, 100 eval episodes) |
| `t9r6f0kf` | fancy-sun-290 | 09-08 | PPO | not described; 2.0M steps |
| `jnpe8o09` | hardy-grass-291 | 09-08 | PPO | not described; 2.0M steps |
| `ju054red` | eager-grass-292 | 09-08 | PPO | not described; 1.0M steps |
| `nxdeokqn` | silver-haze-293 | 09-08 | PPO | not described; 1.0M steps |
| `emwaqlmc` | wobbly-microwave-295 | 09-08 | PPO, regular reward | +0.94% vs `0v9hol5p` (README) |
| `pk9eabsg` | lively-monkey-296 | 09-09 | PPO | not described; 2.0M steps |
| `crv7ystm` | firm-serenity-297 | 09-09 | PPO, terminate-on-failure reward | +0.67% vs `0v9hol5p` (log #10) |

## E1 experiments (Sep 9 to Sep 15, log #3 to #15)

| Run id | W&B name | Date | What it is | Result |
|---|---|---|---|---|
| `np7aenua` | silver-butterfly-298 | 09-09 | PPO, model M0 baseline (status, size, wait, ttl, arrival; attention pooling) | baseline for the model series |
| `nb11ox0z` | silvery-microwave-299 | 09-09 | SJF baseline for the runs below | −133.15 reference family |
| `dbqcytgc` | version1-improve-global-context (docs: smooth-hill-300) | 09-10 | PPO, M0 + global-only `queue_pressure` (#5) | mixed, +0.74% |
| `t3hmzxfp` | confused-snowball-301 | 09-10 | PPO, M0 + global and per-job `queue_pressure` (#6) | +1.08% |
| `pe9wuqbe` | ethereal-sea-302 | 09-11 | PPO | not described in the docs |
| `2qbzle63` | frosty-sea-303 | 09-11 | PPO, + unmasked job self-attention (#7) | +0.75% |
| `g231w4jg` | stellar-dawn-304 | 09-11 | PPO, M1: + job self-attention with finished jobs masked (#8) | **+1.14%** vs `nb11ox0z`, best on E1 |
| `znx8m6mz` | swept-dew-305 | 09-12 | PPO, deeper job-context encoder and scorer (#9) | −17.51%, reverted |
| `r23681yd` | sparkling-plant-306 | 09-13 | PPO, gamma 0.999 / lambda 0.99, 1.2M steps (#13) | +0.61%, reverted |
| `rpw3vr25` | wise-disco-307 | 09-15 | PPO, dual attention network, 2 machines (#15a) | crashed, stopped early |
| `74v8wj8m` | good-totem-308 | 09-15 | SJF baseline at 2 machines | baseline for #15a |
| `ddsv1ozh` | fancy-galaxy-309 | 09-15 | PPO, dual attention, 1 machine (#15b) | +0.97%, not adopted |
| `cxtxjusj` | amber-dragon-310 | 09-16 | PPO, gated cross-attention, 1M steps (#15c) | +0.79%, not adopted |

## Random-usage experiments (Sep 16 to Sep 29, log #16 to #21)

| Run id | W&B name | Date | Environment | What it is | Result |
|---|---|---|---|---|---|
| `xjnrvczb` | fiery-frost-311 | 09-16 | E2 | PPO, gated dual attention, arrival removed (#16) | +1.01% vs the old SJF baseline (not a fair comparison) |
| `r84hnw8u` | elated-sun-312 | 09-17 | E2 | PPO, same as `xjnrvczb` with gamma = 1.0 | −18.07%, regression |
| `61827rfu` | v3-SJF | 09-17 | E2 | SJF baseline on the new generator (#17) | baseline |
| `h0vwcu5e` | v3-ppo-dual | 09-17 | E2 | PPO without a usage feature (#17) | −154.14% |
| `wey435h6` | v3-ppo-dual-usage | 09-17 | E2 | PPO + usage feature (#18) | −52.43% |
| `zb7tjyfn` | spring-frog-316 | 09-18 | E3 | PPO, longer training, `fit_slack`, narrowed 40% to 60% (#19) | −25.33% |
| `y08ugurp` | peachy-star-317 | 09-18 | E3 | SJF baseline for #19 and #20 | baseline |
| `gg3l5d00` | toasty-tree-318 | 09-18 | E3 | PPO, per-pair fit term in the scorer (#20) | −15.77% |
| `hisjonwe` | avid-bush-319 | 09-29 | E3 | PPO, learned fit gate (#21) | −89.98%, regression |
| `vc54897f` | v4-SJF | 09-29 | not documented | named like an SJF baseline but its config is a PPO dual-attention model | not described |
| `5lxml1tc` | desert-jazz-321 | 09-30 | not documented | PPO, dual attention; probably a rerun of the #20 architecture (the generator comments mention two reruns near −26%) | not confirmed |

## Return to E1 and rebuilt originals (Sep 30 to Oct 1)

| Run id | W&B name | Date | What it is | Result |
|---|---|---|---|---|
| `lbp95v5g` | wandering-firebrand-322 | 09-30 | PPO, dual attention with bin packing, on the restored E1 task | −10.95% vs `mkl0zwp7` |
| `mkl0zwp7` | fresh-tree-322 | 09-30 | SJF baseline on E1 | −133.15, wait 18.39, completion 24.60 |
| `khz1l51q` | gentle-snowflake-324 | 10-01 | PPO, M1 rebuilt (= `g231w4jg`) | −0.15% vs `mkl0zwp7` |
| `iw2n4ky5` | colorful-haze-325 | 10-01 | PPO, M0 rebuild attempt | failed (disk full), no results |
| `6b195fqa` | proud-music-326 | 10-01 | PPO, M0 rebuilt (= `np7aenua`), rerun of `iw2n4ky5` | +0.98% vs `mkl0zwp7` |

## E4 experiments (Oct 2 to Oct 7)

SJF baseline `rjzcixv2`: −50.43, wait 10.09, completion 16.33. Exact optimal (script, no W&B run): −44.41, +11.9%.

| Run id | W&B name | Date | What it is | Result |
|---|---|---|---|---|
| `rjzcixv2` | crisp-haze-327 | 10-02 | SJF baseline on E4 | −50.43 |
| `cf2zhwti` | flowing-sun-328 | 10-02 | PPO, M0 (cannot see usage) | −133.44, −164.6% |
| `z4sayojp` | faithful-cosmos-329 | 10-02 | PPO, M5 (usage as raw inputs) | −73.53, −45.81% |
| `iusicvre` | fluent-surf-330 | 10-03 | PPO, M6 (fit attention, concat) | −69.07, −36.98% |
| `emt46quc` | olive-donkey-331 | 10-03 | PPO, M7 (+ job self-attention) | −58.94, −16.87% |
| `bv1hk6lb` | snowy-disco-332 | 10-03 | PPO, M8 (multiply instead of concat) | −63.20, −25.3% |
| `etudt4qo` | usual-mountain-333 | 10-04 | PPO, M9 (+ feasibility mask) | −49.28, **+2.28%** |
| `ne5qv1d2` | wild-wave-334 | 10-06 | PPO, M10 (M9 with entropy 0.05 to 0.01) | −49.27, +2.31% |
| `j5ojvpto` | rural-shape-335 | 10-07 | PPO, M11 on E1 (not-yet-arrived jobs zeroed and masked) | crashed after episode 3288; partial −3.21% vs `mkl0zwp7` |

## ENV-D runs (Oct 7 to Oct 10, per-resource fit and the `pfrw14vh` family)

Environment: each resource uses one level, one random resource a different level. SJF baseline `ko82z2qr`: −118.82, wait 16.93, completion 23.17.

| Run id | W&B name | Date | What it is | Result |
|---|---|---|---|---|
| `0lpdbtej` | smart-fog-336 | 10-07 | PPO, per-resource fit inputs, 4 envs | failed; only 1268 evaluation episodes, not comparable |
| `ympx0iki` | mild-grass-337 | 10-08 | PPO, same family | failed after 1268 evaluation episodes |
| `a28ckswm` | easy-voice-338 | 10-08 | PPO, per-resource fit, 4 envs × 2048 steps | −11.48% vs `ko82z2qr` |
| `ko82z2qr` | likely-tree-339 | 10-09 | SJF baseline on ENV-D | −118.82 |
| `4slmumq7` | visionary-glitter-340 | 10-09 | PPO, per-resource fit, 8 envs × 512, lr 5e-5 | crashed; −11.46% on 4775 episodes |
| `8cf1jt7g` | lyric-elevator-341 | 10-09 | PPO, same settings | −238.9% (policy collapse) |
| `cdhnmmw1` | upbeat-grass-342 | 10-09 | PPO + pair features and dynamic machine embedding, constant lr 5e-5, no KL limit | −108.9% (policy collapse) |
| `pfrw14vh` | olive-fire-343 | 10-09 | PPO, same network with lr decay, `target_kl` 0.03 and best-checkpoint test | **+0.74%** vs `ko82z2qr` |
| `c3wgtx1j` | neat-plant-344 | 10-09 | Double DQN, same network, ENV-D | stopped by hand at 309k of 2M steps; no results |

## ENV-2R and ENV-3R runs (Oct 9 to Oct 10)

| Run id | W&B name | Date | Environment | What it is | Result |
|---|---|---|---|---|---|
| `xd83r7lf` | e4-sjf (launched as cerulean-pine-345) | 10-09 | ENV-2R | SJF baseline | −101.25, wait 15.17, completion 21.41 |
| (deleted) | fine-wildflower-346, `n7nakopb` | 10-10 | ENV-2R | first PPO attempt, crashed at 397k steps when `models/` vanished | not in W&B any more |
| `vy9asra5` | rare-armadillo-347 | 10-09 | ENV-2R | PPO, `pfrw14vh` settings, seed 1 | −98.84, **+2.37%** |
| `xehw70v3` | deft-capybara-352 | 10-10 | ENV-2R | PPO, same, second run | −98.86, **+2.36%** |
| `f4vkih6o` | wandering-gorge-348 | 10-10 | ENV-3R | SJF baseline | −118.57, wait 16.92, completion 23.14 |
| `ob8g1j1u` | solar-dew-349 | 10-10 | ENV-3R | PPO, seed 1 | −129.10, **−8.88%** |
| `mxzw1d28` | serene-water-350 | 10-10 | ENV-3R | PPO, second run | −117.87, +0.59% |
| `lauk5acn` | devout-bee-350 | 10-10 | ENV-3R | PPO, entropy bonus 0.15 to 0.03 | −117.44, +0.95% |

## Exact optimal (no W&B runs)

| Environment | Mean reward | Where |
|---|---|---|
| E1 | −131.08 | `scripts/solvers/run_optimal.py` |
| E4 | −44.41 | `scripts/solvers/run_optimal_packing.py` |
| ENV-2R | −93.92 | `scripts/solvers/run_optimal_multires.py` (`/tmp/opt_newenv.npz`) |
| ENV-3R | −113.44 | `scripts/solvers/run_optimal_multires.py` (`/tmp/opt_3res.npz`) |

## Other notes

- `models/vy9asra5` (and the weights of older runs) were deleted from `models/` during the sessions; W&B keeps the logged metrics.
- The W&B names of some baselines were changed after the run (`xd83r7lf` is now `e4-sjf`, `dbqcytgc` is `version1-improve-global-context`).
