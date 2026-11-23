"""Bar chart: % improvement in avg completion time vs SJF (positive = better than SJF), per machine count, grouped by jobs.
Reuses the cell-loading of metrics_tables.py. Output: docs/results/deeprm_improvement_bars.png"""
import runpy, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
g = runpy.run_path(os.path.join(B, "scripts/analysis/metrics_tables.py"))
cells, methods = g["cells"], [m for m in g["methods"] if m not in ("SJF", "FIFO+BF", "Tetris (eps=0.5)")]
JOBS = [20, 40, 60, 80, 100]; ms = sorted({k[0] for k in cells})
fig, axs = plt.subplots(2, 4, figsize=(28, 11))
for m, ax in zip(ms, axs.flat):
    w = 0.8 / len(methods)
    for i, x in enumerate(methods):
        v = [100 * (cells[(m, n)]["SJF"][2] - cells[(m, n)][x][2]) / cells[(m, n)]["SJF"][2] if (m, n) in cells and x in cells[(m, n)] else np.nan for n in JOBS]
        ax.bar(np.arange(len(JOBS)) + (i - len(methods) / 2 + .5) * w, v, w, label=x, color=f"C{(['model va02clzj (10 slots, 5M)','model 9e90oa77 (10 slots)','model ha9gv7hl (20 slots, ft)','model q9050mka (20 slots)','SJF','SJF+BF','Tetris (eps=0.5)','FIFO+BF','Tetris (eps=5)'].index(x))}")
    ax.axhline(0, c="k", lw=1); ax.set_xticks(range(len(JOBS))); ax.set_xticklabels(JOBS); ax.set_xlabel("jobs"); ax.set_title(f"{m} machine{'s' if m > 1 else ''}")
    ax.set_ylabel("% better than SJF (completion time)"); ax.grid(axis="y", alpha=.25)
    lo, hi = ax.get_ylim(); ax.set_ylim(max(lo, -8), min(hi, 4))
axs[0, 0].legend(fontsize=8)
fig.suptitle("Completion-time improvement vs SJF (up = better, FIFO and Tetris eps=0.5 omitted, they are 5-30% worse)")
plt.tight_layout(); plt.savefig(f"{B}/docs/results/deeprm_improvement_bars.png", dpi=110)
