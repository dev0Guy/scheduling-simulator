"""Per-episode deltas, PPO (M9) minus SJF, as line graphs.

Two plots (X = evaluation episode):
  1. reward_delta_ppo_vs_sjf.png            Y = eval/reward           (PPO - SJF)  higher is better
  2. completion_time_delta_ppo_vs_sjf.png   Y = eval/avg_completion_time (PPO - SJF)  LOWER is better,
     so below zero means PPO finished jobs faster.

Ties (delta == 0) are dropped per metric unless --keep-ties.

    python scripts/plots/plot_reward_delta.py
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT = "dev0guy/cluster-scheduling-simulator"
PPO_RUN = "etudt4qo"  # M9
SJF_RUN = "rjzcixv2"
X_AXIS = "evaluation/episode"
REWARD = "eval/reward"
COMPLETION = "eval/avg_completion_time"


def fetch(run_id: str) -> pd.DataFrame:
    run = wandb.Api(timeout=60).run(f"{PROJECT}/{run_id}")
    keys = [X_AXIS, REWARD, COMPLETION]
    df = run.history(keys=keys, samples=100_000, pandas=True)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise SystemExit(f"run {run_id} has no columns {missing}; found {list(df.columns)}")
    df = df[keys].dropna(subset=[X_AXIS]).drop_duplicates(subset=X_AXIS)
    return df.set_index(X_AXIS).sort_index()


def plot_delta(delta: pd.Series, title: str, ylabel: str, out_path: str, lower_is_better: bool) -> None:
    ppo_better = (delta < 0).sum() if lower_is_better else (delta > 0).sum()
    sjf_better = (delta > 0).sum() if lower_is_better else (delta < 0).sum()
    print(f"{title}: plotted {len(delta)} | PPO better {ppo_better} | SJF better {sjf_better} | "
          f"mean delta {delta.mean():+.3f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(delta.index, delta.values, color="tab:blue", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(delta.mean(), color="orange", linestyle="--", linewidth=1, label=f"mean = {delta.mean():+.2f}")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def summarize(name: str, delta: pd.Series, sjf_values: pd.Series, lower_is_better: bool, unit: str) -> None:
    """Average improvement of PPO over SJF on the plotted (non-tie) episodes.
    Improvement is always signed so that positive = PPO better."""
    improvement = -delta if lower_is_better else delta
    sjf_on_these = sjf_values.loc[delta.index]
    wins, losses = improvement[improvement > 0], improvement[improvement < 0]
    pct = improvement.mean() / abs(sjf_on_these.mean()) * 100 if sjf_on_these.mean() != 0 else float("nan")
    print(f"\n=== {name}: non-tie episodes only ({len(delta)} episodes) ===")
    print(f"  average improvement per episode : {improvement.mean():+.3f} {unit}  ({pct:+.2f}% vs SJF mean of {sjf_on_these.mean():.3f})")
    print(f"  median improvement              : {improvement.median():+.3f} {unit}")
    print(f"  PPO better in                   : {len(wins)} episodes, avg gain {wins.mean():+.3f} {unit}")
    print(f"  SJF better in                   : {len(losses)} episodes, avg loss {losses.mean():+.3f} {unit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-ties", action="store_true")
    args = parser.parse_args()

    ppo, sjf = fetch(PPO_RUN), fetch(SJF_RUN)
    joined = ppo.join(sjf, lsuffix="_ppo", rsuffix="_sjf", how="inner")
    print(f"episodes matched between runs: {len(joined)}")

    reward_delta = (joined[f"{REWARD}_ppo"] - joined[f"{REWARD}_sjf"]).dropna()
    completion_delta = (joined[f"{COMPLETION}_ppo"] - joined[f"{COMPLETION}_sjf"]).dropna()
    if not args.keep_ties:
        reward_delta = reward_delta[reward_delta != 0]
        completion_delta = completion_delta[completion_delta != 0]

    plot_delta(
        reward_delta,
        "Reward delta per episode (PPO M9 - SJF)   above 0 = PPO better",
        "Reward delta (PPO - SJF)",
        "reward_delta_ppo_vs_sjf.png",
        lower_is_better=False,
    )
    plot_delta(
        completion_delta,
        "Completion time delta per episode (PPO M9 - SJF)   below 0 = PPO better",
        "Avg completion time delta, ticks (PPO - SJF)",
        "completion_time_delta_ppo_vs_sjf.png",
        lower_is_better=True,
    )
    summarize("REWARD", reward_delta, joined[f"{REWARD}_sjf"], lower_is_better=False, unit="reward")
    summarize("COMPLETION TIME", completion_delta, joined[f"{COMPLETION}_sjf"], lower_is_better=True, unit="ticks")
    print("\n(improvement: positive = PPO better; completion time is shown as ticks saved vs SJF)")
    plt.show()


if __name__ == "__main__":
    main()
