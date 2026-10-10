"""Compare M9 vs SJF per evaluation episode, excluding ties.

Produces two figures:
  1. m9_vs_sjf_reward_diff.png            (M9 reward - SJF reward)
  2. m9_vs_sjf_completion_time_diff.png   (SJF completion - M9 completion)

Sign convention: in BOTH graphs, a positive value means M9 is better.
(Completion time is lower-is-better, so its difference is flipped.)

Usage:
    pip install wandb pandas matplotlib
    wandb login            # once, if you haven't already
    python plot_m9_vs_sjf.py
    python plot_m9_vs_sjf.py --tie-on reward   # drop episodes tied on reward from BOTH graphs
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT = "dev0guy/cluster-scheduling-simulator"
M9_RUN = "etudt4qo"
SJF_RUN = "rjzcixv2"

X_AXIS = "evaluation/episode"
REWARD = "eval/reward"
COMPLETION = "eval/avg_completion_time"


def fetch(run_id: str) -> pd.DataFrame:
    """Per-episode history for one run.

    Uses run.history (same call compare_wandb.py uses). samples is set far above
    the ~5000 eval episodes so nothing is downsampled. scan_history is only a
    fallback: newer wandb versions raise "Step column '_step' not found" on it
    for runs like these.
    """
    run = wandb.Api(timeout=60).run(f"{PROJECT}/{run_id}")
    keys = [X_AXIS, REWARD, COMPLETION]
    try:
        df = run.history(keys=keys, samples=100_000, pandas=True)
    except Exception as exc:  # noqa: BLE001
        print(f"run.history failed for {run_id} ({exc}); falling back to scan_history")
        df = pd.DataFrame(list(run.scan_history(keys=keys)))
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise SystemExit(f"run {run_id} has no columns {missing}; found {list(df.columns)}")
    df = df[keys].dropna(subset=[X_AXIS])
    return df.drop_duplicates(subset=X_AXIS).set_index(X_AXIS).sort_index()


def build_frame() -> pd.DataFrame:
    m9, sjf = fetch(M9_RUN), fetch(SJF_RUN)
    df = m9.join(sjf, lsuffix="_m9", rsuffix="_sjf", how="inner").dropna()
    df["reward_diff"] = df[f"{REWARD}_m9"] - df[f"{REWARD}_sjf"]  # >0: M9 better
    df["completion_diff"] = df[f"{COMPLETION}_sjf"] - df[f"{COMPLETION}_m9"]  # >0: M9 better
    return df


def plot_diff(diff: pd.Series, title: str, ylabel: str, out_path: str) -> None:
    wins, losses = (diff > 0).sum(), (diff < 0).sum()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    colors = diff.apply(lambda v: "tab:green" if v > 0 else "tab:red")
    ax1.scatter(diff.index, diff.values, c=colors, s=8, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xlabel("Evaluation episode")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{title}\nnon-tie episodes: {len(diff)}  |  M9 better: {wins}  |  SJF better: {losses}")
    ax1.grid(alpha=0.3)

    ax2.hist(diff.values, bins=40, orientation="horizontal", color="tab:blue", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(diff.mean(), color="orange", linestyle="--", label=f"mean = {diff.mean():+.2f}")
    ax2.set_xlabel("Episodes")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tie-on",
        choices=["metric", "reward"],
        default="metric",
        help="'metric': drop ties separately for each graph (default). "
        "'reward': drop episodes where the rewards tie from both graphs.",
    )
    args = parser.parse_args()

    df = build_frame()
    print(f"episodes matched between runs: {len(df)}")

    if args.tie_on == "reward":
        keep = df["reward_diff"] != 0
        reward_diff = completion_diff = None
        reward_diff = df.loc[keep, "reward_diff"]
        completion_diff = df.loc[keep, "completion_diff"]
    else:
        reward_diff = df.loc[df["reward_diff"] != 0, "reward_diff"]
        completion_diff = df.loc[df["completion_diff"] != 0, "completion_diff"]

    print(f"reward: {len(df) - len(reward_diff)} ties removed, {len(reward_diff)} plotted")
    print(f"completion: {len(df) - len(completion_diff)} ties removed, {len(completion_diff)} plotted")

    plot_diff(
        reward_diff,
        "Reward difference per episode (M9 - SJF)",
        "Reward diff  (positive = M9 better)",
        "m9_vs_sjf_reward_diff.png",
    )
    plot_diff(
        completion_diff,
        "Completion time difference per episode (SJF - M9)",
        "Completion-time reduction, ticks  (positive = M9 better)",
        "m9_vs_sjf_completion_time_diff.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
