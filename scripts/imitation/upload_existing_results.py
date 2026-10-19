"""One-off backfill: send the imitation / DAgger results produced before W&B logging existed.

Reads the saved arrays (/tmp/*.npz) and the training logs (/tmp/*.log) and creates one W&B
run per experiment in the project cluster-scheduling-simulator:

  1 machine (ENV-3R): optimal-1M-3R, bc-pretrain-1M-3R, dagger-1M-3R, dagger-it5-fresh-seeds-1M-3R
  2 machines (ENV-3R): optimal-2M-3R, sjf-2M-3R, bc-pretrain-2M-3R, dagger-2M-3R

    .venv/bin/python scripts/imitation/upload_existing_results.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wandb_log as wl  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
EPOCH_BC = re.compile(r"^epoch (\d+)/(\d+): train loss ([\d.]+)  validation accuracy ([\d.]+)% \(start actions only: ([\d.]+)%\)  critic MSE ([\d.]+)  \((\d+)s\)")
EPOCH_DG = re.compile(r"^  epoch (\d+)/(\d+): loss ([\d.]+)")
ITER = re.compile(r"^iter (\d+): collected (\d+) states, policy disagrees with the optimum on ([\d.]+)% \((\d+)s\)")
TEST = re.compile(r"^test on (\d+) evaluation job sets: policy ([-\d.]+)  SJF ([-\d.]+)  margin ([-+\d.]+)%  paired diff ([-+\d.]+) \(95% CI ([-+\d.]+) to ([-+\d.]+)\)  wins/losses/ties (\d+)/(\d+)/(\d+)  failed starts/episode ([\d.]+)")
DONE = re.compile(r"^iter (\d+) done in (\d+)s")


def lines(path):
    with open(path, errors="ignore") as f:
        return [l.rstrip("\n") for l in f if "objc[" not in l]


def test_dict(m, prefix):
    n, pol, sjf, margin, diff, lo, hi, w, l, t, fail = m.groups()
    return {prefix + "n_episodes": int(n), prefix + "policy_mean": float(pol), prefix + "sjf_mean": float(sjf),
            prefix + "margin_vs_sjf_pct": float(margin), prefix + "paired_diff_vs_sjf": float(diff),
            prefix + "paired_diff_ci95": (float(hi) - float(lo)) / 2, prefix + "wins": int(w),
            prefix + "losses": int(l), prefix + "ties": int(t), prefix + "failed_starts_per_episode": float(fail)}


def log_bc(run, ls):
    last = None
    for l in ls:
        m = EPOCH_BC.match(l)
        if m:
            e, _, loss, acc, sacc, mse, sec = m.groups()
            wl.log(run, {"epoch": int(e), "train/loss": float(loss), "train/val_accuracy": float(acc) / 100,
                         "train/val_start_accuracy": float(sacc) / 100, "train/critic_mse": float(mse),
                         "train/epoch_seconds": float(sec)})
        m = TEST.match(l)
        if m and last is None:
            last = m
            for k, v in test_dict(m, "test300_").items():
                run.summary[k] = v


def log_dagger(run, ls):
    it, epoch_counter = 0, 0
    pending = {}
    for l in ls:
        m = ITER.match(l)
        if m:
            it = int(m.group(1))
            pending = {"iteration": it, "dagger/new_states": int(m.group(2)),
                       "dagger/disagreement_pct": float(m.group(3)), "dagger/collect_seconds": float(m.group(4))}
            continue
        m = EPOCH_DG.match(l)
        if m:
            epoch_counter += 1
            wl.log(run, {"epoch": epoch_counter, "train/loss": float(m.group(3))})
            continue
        m = TEST.match(l)
        if m and pending:
            pending.update(test_dict(m, "dagger/test300_"))
            continue
        m = DONE.match(l)
        if m and pending:
            pending["dagger/seconds"] = float(m.group(2))
            wl.log(run, pending)
            pending = {}
    if pending:                                  # an iteration that was cut short
        wl.log(run, pending)


def episodes(name, path, tags, config, policy=None, sjf=None, optimal=None, files=(), model=None, extra=None, ls=None, kind=None):
    run = wl.start(name, config, tags)
    if run is None:
        print("W&B unavailable"); return
    if ls is not None:
        (log_bc if kind == "bc" else log_dagger)(run, ls)
    if policy is not None:
        wl.log_episodes(run, policy, sjf, optimal)
        if sjf is None:
            run.summary["policy_mean"] = float(np.mean(policy))
    for k, v in (extra or {}).items():
        run.summary[k] = v
    for f in files:
        wl.upload(run, f, "result")
    if model:
        wl.upload(run, model, "model")
    print(name, "->", run.url, flush=True)
    wl.finish(run)


def p(*a):
    return os.path.normpath(os.path.join(ROOT, *a))


def main():
    # ---------------- 1 machine ----------------
    opt3 = np.load("/tmp/opt_3res.npz")
    opt1 = opt3["reward"]
    bc1 = np.load("/tmp/bc_test_5000.npz")
    dg1 = np.load("/tmp/dagger_test_5000.npz")
    fresh = np.load("/tmp/dagger_fresh_1000.npz")

    run = wl.start("optimal-1M-3R", {"n_machines": 1, "episodes": len(opt1), "first_seed": 42}, ["optimal", "exact-solver"])
    for ep in range(len(opt1)):
        wl.log(run, {"evaluation/episode": ep, "eval/reward": float(opt3["reward"][ep]),
                     "eval/avg_wait_time": float(opt3["wait"][ep]), "eval/avg_completion_time": float(opt3["completion"][ep])})
    run.summary["policy_mean"] = float(opt1.mean())
    wl.upload(run, "/tmp/opt_3res.npz", "result"); print("optimal-1M-3R ->", run.url); wl.finish(run)

    episodes("bc-pretrain-1M-3R", None, ["imitation", "behaviour-cloning", "backfill"],
             {"n_machines": 1, "epochs": 12, "dataset": "optimal_dataset.npz (10000 episodes, 687156 samples)"},
             bc1["policy"], bc1["sjf"], opt1, ["/tmp/bc_test_5000.npz"], p("models", "bc_policy.pt"),
             ls=lines("/tmp/bc_pretrain.log"), kind="bc")
    episodes("dagger-1M-3R", None, ["imitation", "dagger", "backfill"],
             {"n_machines": 1, "iterations_planned": 6, "iterations_finished": 5, "episodes_per_iteration": 1500,
              "note": "iteration 6 was stopped (stuck); the 5000-episode test is of iteration 5"},
             dg1["policy"], dg1["sjf"], opt1, ["/tmp/dagger_test_5000.npz", "/tmp/dagger.log"], p("models", "dagger_it5.pt"),
             ls=lines("/tmp/dagger.log"), kind="dagger")
    episodes("dagger-it5-fresh-seeds-1M-3R", None, ["imitation", "dagger", "held-out", "backfill"],
             {"n_machines": 1, "first_seed": 20000, "note": "seeds never used for training, DAgger or checkpoint choice"},
             fresh["policy"], fresh["sjf"], None, ["/tmp/dagger_fresh_1000.npz"])

    # ---------------- 2 machines ----------------
    opt2 = np.load("/tmp/opt_2m.npz")["reward"]
    bc2 = np.load("/tmp/bc2m_test_5000.npz")
    dg2 = np.load("/tmp/dagger2m_test_5000.npz")
    ls2 = lines("/tmp/run2m.log")
    k = next(i for i, l in enumerate(ls2) if l.startswith("[") and "pretrain" in l)
    k2 = next(i for i, l in enumerate(ls2) if l.startswith("[") and "optimal eval" in l)
    k3 = next(i for i, l in enumerate(ls2) if l.startswith("[") and l.endswith("dagger"))
    episodes("optimal-2M-3R", None, ["optimal", "exact-solver", "backfill"], {"n_machines": 2, "first_seed": 42},
             opt2, None, None, ["/tmp/opt_2m.npz"])
    episodes("sjf-2M-3R", None, ["sjf", "baseline", "backfill"], {"n_machines": 2, "first_seed": 42},
             bc2["sjf"], None, None, [])
    episodes("bc-pretrain-2M-3R", None, ["imitation", "behaviour-cloning", "backfill"],
             {"n_machines": 2, "epochs": 12, "dataset": "optimal_dataset_2m.npz (3000 episodes, 123623 samples)"},
             bc2["policy"], bc2["sjf"], opt2, ["/tmp/bc2m_test_5000.npz"], p("models", "bc_policy_2m.pt"),
             ls=ls2[k:k2], kind="bc")
    episodes("dagger-2M-3R", None, ["imitation", "dagger", "backfill"],
             {"n_machines": 2, "iterations": 6, "episodes_per_iteration": 1500},
             dg2["policy"], dg2["sjf"], opt2, ["/tmp/dagger2m_test_5000.npz"], p("models", "dagger2m_it6.pt"),
             ls=ls2[k3:], kind="dagger")


if __name__ == "__main__":
    main()
