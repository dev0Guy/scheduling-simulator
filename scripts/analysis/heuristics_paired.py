"""Paired comparison vs SJF (same seeds => per-episode differences). Outputs:
 docs/results/heuristics_paired_diff.png  : mean per-episode completion-time difference vs SJF, % of SJF, with 95% CI (negative = better)
 docs/results/heuristics_paired_wtl.png   : per-episode win/tie/loss vs SJF (completion time), stacked"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); R = f"{B}/results"
Ns = [10, 20, 40, 60, 100]; M_ = {"SJF+BF": "sjf_bf", "Tetris (eps=5)": "tetris5", "Tetris (eps=0.5)": "tetris"}
def comp(M, N):
    z = np.load(f"{R}/heuristics_deeprm/m{M}_n{N}.npz"); d = {k: z[v + "_completion"] for k, v in M_.items() if v != "tetris5"}
    d["Tetris (eps=5)"] = np.load(f"{R}/heuristics_deeprm_tetris5/m{M}_n{N}.npz")["tetris5_completion"]; return z["sjf_completion"], d
fig, axs = plt.subplots(2, 4, figsize=(28, 11)); fig2, axs2 = plt.subplots(2, 4, figsize=(28, 11)); labs = list(M_)
for M, ax, ax2 in zip(range(1, 9), axs.flat, axs2.flat):
    for j, N in enumerate(Ns):
        s, d = comp(M, N); w = .8 / len(labs)
        for i, k in enumerate(labs):
            diff = d[k] - s
            ax.bar(j + (i - 1) * w, 100 * diff.mean() / s.mean(), w, yerr=100 * 1.96 * diff.std() / len(diff) ** .5 / s.mean(), color=f"C{i+1}", capsize=2, label=k if j == 0 else None)
            win, tie, lose = [100 * x.mean() for x in (diff < -1e-9, abs(diff) <= 1e-9, diff > 1e-9)]
            x0 = j + (i - 1) * w
            ax2.bar(x0, win, w, color="#2a9d8f"); ax2.bar(x0, tie, w, bottom=win, color="#cccccc"); ax2.bar(x0, lose, w, bottom=win + tie, color="#e76f51")
    for a in (ax, ax2): a.set_xticks(range(len(Ns))); a.set_xticklabels([f"{n} jobs" for n in Ns]); a.set_title(f"{M} machine{'s' if M > 1 else ''}")
    ax.axhline(0, c="k", lw=1); ax.set_ylabel("completion time vs SJF (% of SJF, negative = better)"); ax.grid(axis="y", alpha=.25)
    ax2.set_ylabel("% of episodes"); ax2.set_ylim(0, 100)
axs[0, 0].legend(fontsize=9); fig.suptitle("Paired difference vs SJF (same seeds), mean +- 95% CI"); fig.tight_layout(); fig.savefig(f"{B}/docs/results/heuristics_paired_diff.png", dpi=110)
fig2.suptitle("Per-episode outcome vs SJF (completion time). bars left->right: SJF+BF, Tetris eps=5, Tetris eps=0.5; green=better, grey=tie, red=worse"); fig2.tight_layout(); fig2.savefig(f"{B}/docs/results/heuristics_paired_wtl.png", dpi=110)
