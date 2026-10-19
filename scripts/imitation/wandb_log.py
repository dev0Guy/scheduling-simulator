"""Weights & Biases logging for the imitation scripts (project cluster-scheduling-simulator).

Per-episode evaluation uses the same keys as the PPO runs (`evaluation/episode`, `eval/reward`),
so `scripts/analysis/compare_runs.py` can read these runs. Switch off with WANDB_LOG=0.
"""
import os

import numpy as np

PROJECT = "cluster-scheduling-simulator"


def enabled():
    return os.environ.get("WANDB_LOG", "1") != "0"


def start(name, config=None, tags=None):
    """Start a run (None when logging is switched off or W&B is unavailable)."""
    if not enabled():
        return None
    try:
        import wandb
        run = wandb.init(project=PROJECT, name=name, config=config or {}, tags=tags or [],
                         job_type="imitation", reinit=True)
        wandb.define_metric("evaluation/episode")
        wandb.define_metric("eval/*", step_metric="evaluation/episode")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("iteration")
        wandb.define_metric("dagger/*", step_metric="iteration")
        return run
    except Exception as e:      # logging must never break an experiment
        print(f"[wandb] disabled: {e}", flush=True)
        return None


def log(run, data):
    if run is not None:
        run.log(data)


def summarize(policy, sjf, optimal=None):
    """Paired statistics of per-episode rewards (higher is better)."""
    policy, sjf = np.asarray(policy, float), np.asarray(sjf, float)
    d = policy - sjf
    n = len(d)
    out = {
        "n_episodes": n,
        "policy_mean": float(policy.mean()), "sjf_mean": float(sjf.mean()),
        "paired_diff_vs_sjf": float(d.mean()),
        "paired_diff_ci95": float(1.96 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "margin_vs_sjf_pct": float(100 * d.mean() / abs(sjf.mean())),
        "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "ties": int((abs(d) <= 1e-9).sum()),
    }
    if optimal is not None:
        optimal = np.asarray(optimal, float)
        out["optimal_mean"] = float(optimal.mean())
        gap = optimal.mean() - sjf.mean()
        out["gap_closed_pct"] = float(100 * d.mean() / gap) if abs(gap) > 1e-12 else 0.0
        out["equal_to_optimal_pct"] = float(100 * (abs(policy - optimal) < 1e-9).mean())
    return out


def log_episodes(run, policy, sjf=None, optimal=None, prefix=""):
    """One log row per evaluation episode (+ summary). Episode i = seed (start + i)."""
    if run is None:
        return
    policy = np.asarray(policy, float)
    for ep, r in enumerate(policy):
        row = {"evaluation/episode": ep, "eval/reward": float(r)}
        if sjf is not None:
            row["eval/sjf_reward"] = float(sjf[ep]); row["eval/diff_vs_sjf"] = float(r - sjf[ep])
        if optimal is not None:
            row["eval/optimal_reward"] = float(optimal[ep]); row["eval/diff_vs_optimal"] = float(r - optimal[ep])
        run.log(row)
    if sjf is not None:
        for k, v in summarize(policy, sjf, optimal).items():
            run.summary[prefix + k] = v


def upload(run, path, kind="result"):
    """Attach a local file (results .npz or model weights) to the run as an artifact."""
    if run is None or not os.path.exists(path):
        return
    import wandb
    art = wandb.Artifact(os.path.basename(path).replace(".", "_"), type=kind)
    art.add_file(path)
    run.log_artifact(art)


def finish(run):
    if run is not None:
        run.finish()
