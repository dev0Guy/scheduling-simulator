"""DeepRM-paper-style figures (Mao et al. 2016): (1) learning curves vs SJF, (2) average slowdown bars per method, (3) CDF of slowdown.
Slowdown per episode = mean completion time / mean job size (proxy: per-job slowdowns were not stored).
Outputs docs/results/deeprm_paper_{learning,slowdown_bars,slowdown_cdf}.png"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); R = f"{B}/results"; O = f"{B}/docs/results"
MODELS = {"DeepRM-PPO va02clzj (10 slots, 5M)": "tournament_deeprm_n10_5m", "PPO 9e90oa77 (10 slots, 2M)": "tournament_deeprm_n10", "PPO ha9gv7hl (20 slots)": "tournament_deeprm_vft"}
HEUR = {"SJF": "sjf", "SJF+BF": "sjf_bf", "Tetris": "tetris", "FIFO+BF": "fifo_bf"}

def slow(M, N):
    h = f"{R}/heuristics_deeprm/m{M}_n{N}.npz"
    if not os.path.exists(h): return {}
    z = np.load(h); size = z["mean_size"]; d = {k: z[v + "_completion"] / size for k, v in HEUR.items()}
    for lab, dirn in MODELS.items():
        f = f"{R}/{dirn}/m{M}_n{N}.npz"
        if os.path.exists(f):
            t = np.load(f)["t_completion"]; k = min(len(t), len(size)); d[lab] = t[:k] / size[:k]
    return d

# 1) learning curves
import wandb
api = wandb.Api(); fig, ax = plt.subplots(figsize=(9, 5))
for i, (rid, lab) in enumerate([("va02clzj", "va02clzj (5M, scratch)"), ("9e90oa77", "9e90oa77 (2M, fine-tune)")]):
    h = api.run(f"dev0guy/cluster-scheduling-simulator/{rid}").history(keys=["episode/reward"], samples=4000, pandas=True).dropna()
    ax.plot(h["_step"], h["episode/reward"].rolling(100, min_periods=1).mean(), c=f"C{i}", label=lab)
ax.set_xlabel("W&B logged step (training episodes)"); ax.set_ylabel("training episode reward (smoothed)"); ax.set_title("Training curves (no SJF line: job count varies per episode)"); ax.grid(alpha=.25); ax.legend()
plt.tight_layout(); plt.savefig(f"{O}/deeprm_paper_learning.png", dpi=110); plt.close()

# 2) bars
Ns = [10, 20, 40]; fig, axs = plt.subplots(1, 3, figsize=(20, 5))
for ax, M in zip(axs, [1, 2, 3]):
    labs = None
    for j, N in enumerate(Ns):
        d = slow(M, N); labs = labs or list(d)
        for i, k in enumerate(labs):
            if k in d: ax.bar(j + (i - len(labs) / 2 + .5) * .8 / len(labs), d[k].mean(), .8 / len(labs), yerr=1.96 * d[k].std() / len(d[k]) ** .5, color=f"C{i}", label=k if j == 0 else None)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"{n} jobs" for n in Ns]); ax.set_ylabel("average slowdown"); ax.set_title(f"{M} machine{'s' if M > 1 else ''}"); ax.grid(axis="y", alpha=.25)
axs[1].legend(fontsize=8); plt.suptitle("Average job slowdown (lower = better)"); plt.tight_layout(); plt.savefig(f"{O}/deeprm_paper_slowdown_bars.png", dpi=110); plt.close()

# 3) CDF
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
for ax, (M, N) in zip(axs, [(2, 10), (2, 20), (2, 40)]):
    for i, (k, v) in enumerate(slow(M, N).items()): v = np.sort(v); ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=k, c=f"C{i}")
    ax.set_title(f"{M} machines, {N} jobs"); ax.set_xlabel("episode slowdown"); ax.set_ylabel("CDF"); ax.grid(alpha=.25)
axs[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{O}/deeprm_paper_slowdown_cdf.png", dpi=110)
