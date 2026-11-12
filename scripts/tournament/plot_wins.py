"""Bar chart of the win percentage against SJF (same definition as the results table): per (machines, jobs)
setting, the share of the 5000 evaluation episodes in which PPO (trained on that size) / tournament PPO has a
strictly higher original reward than SJF on the same job set (tolerance 1e-3; ties and losses are the rest).
Inputs: SJF /tmp/sjf_m{M}_n{N}.npz, tournament /tmp/tournament/m{M}_n{N}.npz,
PPO per-episode rewards from W&B saved as <ppo_dir>/ppo_m{M}_n{N}.npy.   usage: plot_wins.py <ppo_dir> <out.png>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ppo_dir, out = sys.argv[1], sys.argv[2]
import os
# every cell that has a tournament result; the SJF rewards of the extended grid are stored in the tournament files.
# The regular PPO is shown where a model exists (10 jobs: the same model as the tournament); otherwise the bars are left out.
cells = [(m, n) for n in range(10, 101, 10) for m in range(1, 9) if os.path.exists(f"/tmp/tournament/m{m}_n{n}.npz")]
win = np.full((len(cells), 2), np.nan)
loss = np.full((len(cells), 2), np.nan)
for i, (m, n) in enumerate(cells):
    t = np.load(f"/tmp/tournament/m{m}_n{n}.npz")
    s = t["s_reward"]
    ppo = t["t_reward"] if n == 10 else (np.load(f"{ppo_dir}/ppo_m{m}_n{n}.npy") if os.path.exists(f"{ppo_dir}/ppo_m{m}_n{n}.npy") else None)
    for k, r in enumerate((ppo, t["t_reward"])):
        if r is None:
            continue
        d = r - s
        win[i, k], loss[i, k] = 100 * (d > 1e-3).mean(), 100 * (d < -1e-3).mean()
fig, ax = plt.subplots(figsize=(40, 7))
w, x = 0.2, np.arange(len(cells))
# SJF's own wins = the other side of each comparison: episodes in which SJF has a strictly higher reward
bars = (("SJF beats PPO", "#B8BECB", loss[:, 0]), ("PPO beats SJF", "#1F8A8A", win[:, 0]),
        ("SJF beats Tournament", "#8A93A5", loss[:, 1]), ("Tournament beats SJF", "#D9822B", win[:, 1]))
for k, (name, c, v) in enumerate(bars):
    b = ax.bar(x + (k - 1.5) * w, v, w, label=name, color=c)
    ax.bar_label(b, fmt="%.0f", fontsize=6, padding=2)
ax.axhline(50, color="#8A93A5", ls="--", lw=1)
ax.set_xticks(x, [f"{m}M\n{n}J" for m, n in cells])
for i in range(1, len(cells)):
    if cells[i][1] != cells[i - 1][1]:
        ax.axvline(i - 0.5, color="#D9D6CC", lw=1)
ax.set_ylim(0, 100)
ax.set_xlabel("machines (M) / jobs (J)")
ax.set_ylabel("win % against SJF (5000 episodes)")
ax.set_title("Win percentage per pair: SJF vs PPO and SJF vs tournament PPO (the rest of each pair are ties)")
ax.legend(frameon=False, ncol=4)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out, dpi=150)
for (m, n), a, b in zip(cells, win, loss):
    print(m, n, "win", a.round(1), "loss", b.round(1))
