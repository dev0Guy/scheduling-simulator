"""Regular vs tournament win % vs SJF (ties excluded) per (machines, jobs), each point annotated with its tie count.
Parsed from docs/results/regular_vs_sjf_ext.md (regular) and scaling_ext_regular_tournament.md (tournament, recovered table)."""
import re, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "results")
reg, tour = {}, {}
for l in open(f"{R}/regular_vs_sjf_ext.md"):
    c = [x.strip() for x in l.split("|")]
    if len(c) > 7 and c[1].isdigit():
        w, t, lo = (int(re.search(r"\((\d+)\)", c[i]).group(1)) for i in (4, 5, 6))
        reg[(int(c[1]), int(c[2]))] = (w, t, lo)
jobs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for l in open(f"{R}/scaling_ext_regular_tournament.md"):
    if l.startswith("| **"):
        m = int(re.match(r"\| \*\*(\d+)", l).group(1))
        for n, cell in zip(jobs, l.split("|")[2:12]):
            x = re.search(r"טורניר: [\d.]+% \((\d+)\) / [\d.]+% \((\d+)\) / [\d.]+% \((\d+)\)", cell)
            if x: tour[(m, n)] = tuple(int(v) for v in x.groups())
import glob, numpy as np
for f in glob.glob(f"{R}/../../results/tournament_grid/m*_n*.npz"):   # raw eval files of the cells missing from the table
    m, n = map(int, re.search(r"m(\d+)_n(\d+)", f).groups())
    z = np.load(f); d = z["t_reward"] - z["s_reward"]
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum()); tour[(m, n)] = (w, len(d) - w - l, l)
reg.update({k: v for k, v in tour.items() if k[1] == 10})
sub = {}   # smaller model on a bigger cluster, shown only 2 machines per slice: results/tournament_subset/m{model}on{M}_n{N}.npz
for f in glob.glob(f"{R}/../../results/tournament_subset/m*on*_n*.npz"):
    mm, m, n = map(int, re.search(r"m(\d+)on(\d+)_n(\d+)", f).groups())
    z = np.load(f); d = z["t_reward"] - z["s_reward"]
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum()); sub[(m, n)] = (w, len(d) - w - l, l, mm)
pct = lambda v: 100 * v[0] / max(v[0] + v[2], 1)
fig, axs = plt.subplots(2, 4, figsize=(26, 10), sharey=True)
for m, ax in zip(range(1, 9), axs.flat):
    for d, st, col, lab in ((reg, "o-", "C0", "regular (model per cell)"), (tour, "s--", "C3", "tournament (model trained on 10 jobs)")):
        ks = sorted(k for k in d if k[0] == m)
        ax.plot([k[1] for k in ks], [pct(d[k]) for k in ks], st, c=col, label=lab)
        for k in ks:
            ax.annotate(str(d[k][1]), (k[1], pct(d[k])), textcoords="offset points", xytext=(0, 7 if d is tour else -13),
                        ha="center", fontsize=7, color=col)
    pts = sorted((n, v) for (mach, n), v in sub.items() if mach == m)
    if pts:   # ONE orange line (one legend entry) for the whole series
        ax.plot([n for n, _ in pts], [pct(v) for _, v in pts], "*-", c="orange", ms=11, mec="k", lw=2,
                label=f"{pts[0][1][3]}-machine model, shown {pts[0][1][3]} machines per slice (tournament)")
        for n, v in pts:
            ax.annotate(str(v[1]), (n, pct(v)), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7, color="orange")
    ax.axhline(50, c="gray", ls=":", lw=.8); ax.set_ylim(-5, 100); ax.set_title(f"{m} machines"); ax.set_xlabel("jobs"); ax.grid(alpha=.25)
    ax.set_xticks(jobs)
axs[0, 0].legend(fontsize=8, loc="lower left"); axs[1, 3].legend(fontsize=7, loc="upper right"); axs[0, 0].set_ylabel("win % vs SJF (ties excluded)"); axs[1, 0].set_ylabel("win % vs SJF (ties excluded)")
fig.suptitle("Regular vs tournament: win rate vs SJF, ties excluded (5000 episodes per cell). Number next to each point = tie count")
fig.tight_layout(); fig.savefig(f"{R}/regular_vs_tournament_win_ties.png", dpi=120)
