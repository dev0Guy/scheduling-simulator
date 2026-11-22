"""Option-DQN models vs SJF and vs the previous model (masked PPO, 1 machine x 10 jobs, the one used in the tournament)
on the same 5000 episodes (seed 42 + episode). Win % excludes ties.   compare_dqn.py name=wandb_run_id ..."""
import os
import sys

import numpy as np
import wandb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sjf = np.load(f"{ROOT}/results/sjf/sjf_m1_n10.npz")["reward"]
ppo = np.load(f"{ROOT}/results/regular/ppo_m1_n10.npy")


def wtl(d):
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    return w, len(d) - w - l, l, 100 * w / max(w + l, 1)


def fmt(x):
    return f"{x[0]}/{x[1]}/{x[2]} ({x[3]:.1f}%)"


api = wandb.Api()
rows = [f"| previous PPO model | {ppo.mean():.2f} | {fmt(wtl(ppo - sjf))} | - |"]
for arg in sys.argv[1:]:
    name, rid = arg.split("=")
    run = api.run(f"dev0guy/cluster-scheduling-simulator/{rid}")
    h = sorted(set((int(r["evaluation/episode"]), r["eval/reward"]) for r in run.history(keys=["evaluation/episode", "eval/reward"], samples=6000, pandas=False)))
    r = np.array([v for _, v in h])
    assert len(r) == 5000, f"{rid}: {len(r)} episodes"
    np.save(f"{ROOT}/results/regular/dqn_{name}_m1_n10.npy", r)
    rows.append(f"| {name} | {r.mean():.2f} | {fmt(wtl(r - sjf))} | {fmt(wtl(r - ppo))} |")
out = f"SJF mean reward {sjf.mean():.2f}\n\n| model | mean reward | vs SJF (W/T/L, win% no ties) | vs previous PPO |\n|---|---|---|---|\n" + "\n".join(rows)
print(out)
open(f"{ROOT}/docs/results/option_dqn_vs_sjf_ppo.md", "w").write("# Option DQN, 1 machine x 10 jobs, 5000 episodes\n\n" + out + "\n")
