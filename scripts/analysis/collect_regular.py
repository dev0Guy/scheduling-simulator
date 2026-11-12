"""Regular masked PPO vs SJF on the machines x jobs grid. Every training run already evaluates its model on
5000 episodes (seed 42 + episode) and logs `eval/reward` per episode to W&B, so no extra evaluation is needed:
this script (1) fetches those per-episode rewards of every finished run, (2) computes the SJF reward on the same
job sets (scripts/analysis/sjf_baseline.py), (3) writes win / tie / loss per cell and (4) uploads the table to W&B.
Raw data is kept in results/regular/ppo_m{M}_n{N}.npy and results/sjf/sjf_m{M}_n{N}.npz (not /tmp: it gets cleaned).

    collect_regular.py            all cells that have a finished run and are not collected yet
    collect_regular.py --watch    repeat every 30 minutes
"""
import os
import subprocess
import sys
import time

import numpy as np
import wandb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROJECT = "dev0guy/cluster-scheduling-simulator"
NS = range(10, 101, 10)
MS = range(1, 9)
# runs of the original 1-4 machines x 10-50 jobs grid (before the scale2 tag existed)
OLD = {(1, 10): "ncpb2fyb", (2, 10): "bsp1klua", (3, 10): "perlec9c", (4, 10): "t3lhn7a9",
       (1, 20): "zg1mge0d", (2, 20): "hc9vv9ax", (3, 20): "wu532otw", (4, 20): "izcv1dfy",
       (1, 30): "0zrvd7me", (2, 30): "fba8tueg", (3, 30): "ed7qild7", (4, 30): "7ay35hqi",
       (1, 40): "3fsgd31v", (2, 40): "ww3t1xql", (3, 40): "wgyki395", (4, 40): "dohhw7vm",
       (1, 50): "hx5f3581", (2, 50): "9hevqmft", (3, 50): "3b3s51h0", (4, 50): "2naso61v"}


def finished_runs(api):
    """(m, n) -> newest finished run with a complete 5000-episode evaluation."""
    out = {}
    for (m, n), rid in OLD.items():
        out[(m, n)] = api.run(f"{PROJECT}/{rid}")
    for r in api.runs(PROJECT, filters={"tags": {"$in": ["scale2"]}}):
        tags = {t for t in r.tags}
        cell = [(int(t[1:]) if t[0] == "m" else None, int(t[1:]) if t[0] == "n" else None) for t in tags if t[0] in "mn" and t[1:].isdigit()]
        m = next((a for a, _ in cell if a), None)
        n = next((b for _, b in cell if b), None)
        if m is None or n is None or r.state != "finished" or r.summary.get("evaluation/episode") != 4999:
            continue
        if (m, n) not in out or r.created_at > out[(m, n)].created_at:
            out[(m, n)] = r
    return out


def tag_runs(runs):
    """Give the selected run of every cell the same tag set (old 1-4 machine runs had none)."""
    for (m, n), r in runs.items():
        want = {"regular", "masked-ppo", "scale-grid", f"m{m}", f"n{n}", "final-model"}
        if not want <= set(r.tags):
            r.tags = sorted(set(r.tags) | want)
            r.update()


def fetch_ppo(run, path):
    rows = [(int(h["evaluation/episode"]), h["eval/reward"]) for h in run.history(keys=["evaluation/episode", "eval/reward"], samples=6000, pandas=False)]
    rows = sorted(set(rows))
    r = np.array([v for _, v in rows])
    assert len(r) == 5000, f"{run.id}: {len(r)} evaluation episodes"
    np.save(path, r)


def make_sjf(m, n, path):
    env = dict(os.environ, N_MACHINES=str(m), N_JOBS=str(n), JOB_USAGE_MODE="random")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analysis", "sjf_baseline.py"), "5000", "2", path],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wtl(d):
    k = len(d)
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    return w, k - w - l, l


def collect():
    api = wandb.Api()
    runs = finished_runs(api)
    tag_runs(runs)
    done = []
    for (m, n), run in sorted(runs.items(), key=lambda x: (x[0][1], x[0][0])):
        p, s = f"{ROOT}/results/regular/ppo_m{m}_n{n}.npy", f"{ROOT}/results/sjf/sjf_m{m}_n{n}.npz"
        if not os.path.exists(p):
            fetch_ppo(run, p)
        if not os.path.exists(s):
            make_sjf(m, n, s)
        d = np.load(p) - np.load(s)["reward"]
        done.append((m, n, run.id, *wtl(d), float(d.mean())))
    return done


def plot(done):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms, ns = list(range(1, 9)), list(range(10, 101, 10))
    W, G = np.full((8, 10), np.nan), np.full((8, 10), np.nan)
    for m, n, rid, w, t, l, g in done:
        W[m - 1, n // 10 - 1], G[m - 1, n // 10 - 1] = w / 50, g
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    for ax, A, ttl, cm, kw in ((axs[0], W, "Win % vs SJF", "RdYlGn", dict(vmin=0, vmax=100)),
                               (axs[1], G, "Mean reward gain vs SJF", "RdYlGn", dict(vmin=-abs(np.nanmax(np.abs(G))), vmax=abs(np.nanmax(np.abs(G)))))):
        im = ax.imshow(A, cmap=cm, aspect="auto", **kw)
        ax.set_xticks(range(10), ns); ax.set_yticks(range(8), ms)
        ax.set_xlabel("jobs"); ax.set_ylabel("machines"); ax.set_title(ttl)
        for i in range(8):
            for j in range(10):
                if not np.isnan(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.0f}" if A is W else f"{A[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Regular masked PPO vs SJF (5000 episodes per cell)")
    png = os.path.join(ROOT, "docs", "results", "regular_vs_sjf_ext.png")
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return png


def publish(done):
    lines = ["| machines | jobs | run | win | tie | loss | mean gain vs SJF |", "|---|---|---|---|---|---|---|"]
    for m, n, rid, w, t, l, g in sorted(done, key=lambda x: (x[0], x[1])):
        lines.append(f"| {m} | {n} | {rid} | {w/50:.1f}% ({w}) | {t/50:.1f}% ({t}) | {l/50:.1f}% ({l}) | {g:+.2f} |")
    md = os.path.join(ROOT, "docs", "results", "regular_vs_sjf_ext.md")
    open(md, "w").write("# Regular masked PPO vs SJF (5000 episodes per cell)\n\n" + "\n".join(lines) + "\n")
    png = plot(done)
    run = wandb.init(project="cluster-scheduling-simulator", id="regular-vs-sjf-ext", resume="allow", name="regular-vs-sjf-ext",
                     job_type="results", tags=["scale2", "regular", "masked-ppo", "scale-grid", "results"])
    tab = wandb.Table(columns=["machines", "jobs", "run", "wins", "ties", "losses", "mean_gain"],
                      data=[[m, n, rid, w, t, l, g] for m, n, rid, w, t, l, g in done])
    run.log({"regular_vs_sjf": tab, "cells": len(done), "chart": wandb.Image(png)})
    run.finish()
    print(f"{len(done)} cells published -> {md}")


if __name__ == "__main__":
    while True:
        try:
            publish(collect())
        except Exception as e:
            print("retry later:", repr(e)[:300], flush=True)
            if "--watch" not in sys.argv:
                raise
            time.sleep(120)
            continue
        if "--watch" not in sys.argv:
            break
        time.sleep(1800)
