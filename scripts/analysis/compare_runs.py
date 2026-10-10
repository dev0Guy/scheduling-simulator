"""Compare a PPO run with an SJF run, episode by episode.

Usage:
    .venv/bin/python scripts/analysis/compare_runs.py <ppo_run_id> <sjf_run_id>
    .venv/bin/python scripts/analysis/compare_runs.py vy9asra5 xd83r7lf

Both runs must have been evaluated on the same job sets (seed 42 + episode),
so episodes are matched on `evaluation/episode`.

For each metric (reward, wait time, completion time) it prints:
  1. how many episodes are ties, how many PPO is better, how many SJF is better;
  2. the same comparison with the ties removed: in how many of the NOT-tie
     episodes each run is better (count and percent of those episodes), the
     mean of each run over them, the percent difference, and which run id is
     better.

"Better" means higher reward, lower wait time, lower completion time.
Percent difference = (better - worse) / |SJF mean|, shown as the improvement
of the PPO run over the SJF run (positive = PPO better, negative = SJF better).
"""
import argparse
import sys
import time

import pandas as pd
import wandb

PROJECT = "dev0guy/cluster-scheduling-simulator"
X = "evaluation/episode"
METRICS = [
    # (W&B key, label, higher_is_better)
    ("eval/reward", "Reward", True),
    ("eval/avg_wait_time", "Wait time", False),
    ("eval/avg_completion_time", "Completion time", False),
]
TOL = 1e-6  # values closer than this are a tie (float32 noise)


def load(api, project, run_id, retries=4):
    run = api.run(f"{project}/{run_id}")
    keys = [k for k, _, _ in METRICS]
    df = run.history(keys=[X] + keys, pandas=True, samples=10_000)[[X] + keys].dropna()
    if len(df) < 5000:  # history() can drop rows; try the full scan
        for _ in range(retries):
            try:
                rows = [
                    r for r in run.scan_history(keys=[X] + keys)
                    if all(r.get(k) is not None for k in [X] + keys)
                ]
                if len(rows) > len(df):
                    df = pd.DataFrame(rows)[[X] + keys]
                break
            except Exception:
                time.sleep(5)
    return run, df.drop_duplicates(subset=X)


def table(rows, header):
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ppo_run_id")
    ap.add_argument("sjf_run_id")
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    api = wandb.Api()
    ppo_run, ppo = load(api, args.project, args.ppo_run_id)
    sjf_run, sjf = load(api, args.project, args.sjf_run_id)
    m = ppo.merge(sjf, on=X, suffixes=("_ppo", "_sjf"))
    n = len(m)
    if n == 0:
        sys.exit("No matching evaluation episodes between the two runs.")

    ppo_id, sjf_id = args.ppo_run_id, args.sjf_run_id
    print(f"PPO run: {ppo_id} ({ppo_run.name}, {ppo_run.state}, {len(ppo)} episodes)")
    print(f"SJF run: {sjf_id} ({sjf_run.name}, {sjf_run.state}, {len(sjf)} episodes)")
    print(f"Matched episodes: {n}\n")

    # ---- 1. ties / PPO better / SJF better
    rows, parts = [], {}
    for key, label, higher in METRICS:
        p, s = m[key + "_ppo"], m[key + "_sjf"]
        diff = (p - s) if higher else (s - p)          # > 0 means PPO better
        ppo_better = diff > TOL
        sjf_better = diff < -TOL
        tie = ~(ppo_better | sjf_better)
        parts[key] = (p, s, tie, higher, ppo_better, sjf_better)
        rows.append([
            label,
            f"{int(tie.sum())} ({tie.mean() * 100:.2f}%)",
            f"{int(ppo_better.sum())} ({ppo_better.mean() * 100:.2f}%)",
            f"{int(sjf_better.sum())} ({sjf_better.mean() * 100:.2f}%)",
        ])
    print("All episodes")
    table(rows, ["Metric", "Ties", f"PPO better ({ppo_id})", f"SJF better ({sjf_id})"])

    # ---- 2. ties removed
    print("\nTies removed (only episodes where the two runs differ)")
    rows = []
    for key, label, higher in METRICS:
        p, s, tie, _, ppo_b, sjf_b = parts[key]
        keep = ~tie
        k = int(keep.sum())
        if k == 0:
            rows.append([label, 0, "-", "-", "-", "-", "-", "-"])
            continue
        nb, sb = int(ppo_b.sum()), int(sjf_b.sum())
        pm, sm = float(p[keep].mean()), float(s[keep].mean())
        pct = ((pm - sm) if higher else (sm - pm)) / abs(sm) * 100   # + = PPO better
        winner = ppo_id if pct > 0 else sjf_id if pct < 0 else "equal"
        rows.append([
            label, k,
            f"{nb} ({nb / k * 100:.2f}%)", f"{sb} ({sb / k * 100:.2f}%)",
            f"{pm:.3f}", f"{sm:.3f}", f"{pct:+.2f}%", f"{winner}",
        ])
    table(rows, [
        "Metric", "Non-tie episodes",
        f"PPO better ({ppo_id})", f"SJF better ({sjf_id})",
        f"Mean {ppo_id}", f"Mean {sjf_id}", "PPO vs SJF (mean)", "Better run",
    ])
    print("\nPPO vs SJF: positive = PPO better (higher reward, or shorter time); negative = SJF better.")


if __name__ == "__main__":
    main()
