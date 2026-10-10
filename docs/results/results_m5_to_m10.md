# Results M5 to M10 (E4: job usage 63 / 128 / 255)

5000 deterministic episodes, same job sets as the SJF baseline `rjzcixv2`.
Margin vs SJF = (PPO − SJF) / |SJF| on mean reward. Wait and completion are mean ticks, lower is better.

| Model | Run | Mean reward | Margin vs SJF | Wait | Completion | PPO wins / SJF wins / ties | Failed starts per episode |
|---|---|---|---|---|---|---|---|
| M5 | `z4sayojp` | −73.53 | −45.81% | 12.38 | 18.62 | 266 / 4528 / 206 | 0.02 |
| M6 | `iusicvre` | −69.07 | −36.98% | 11.60 | 17.83 | 497 / 4125 / 378 | 0.34 |
| M7 | `emt46quc` | −58.94 | −16.87% | 10.83 | 17.07 | 855 / 3105 / 1040 | 0.06 |
| M8 | `bv1hk6lb` | −63.20 | −25.3% | 11.36 | 17.60 | 483 / 3841 / 676 | 0.003 |
| M9 | `etudt4qo` | −49.28 | +2.28% | 9.97 | 16.20 | 800 / 662 / 3538 | 0.012 |
| M10 | `ne5qv1d2` | −49.27 | +2.31% | 9.96 | 16.20 | 788 / 730 / 3482 | 0.016 |

References on the same episodes:

| Reference | Run | Mean reward | Margin vs SJF | Wait | Completion |
|---|---|---|---|---|---|
| SJF | `rjzcixv2` | −50.43 | 0% | 10.09 | 16.33 |
| Exact optimal | script | −44.41 | +11.9% | 9.49 | 15.73 |
