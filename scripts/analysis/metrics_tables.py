"""Per cell (machines x jobs): reward, average wait time, average completion time (= wait + size) of every method, same seeds.
Models: results/tournament_deeprm_n10 (9e90oa77), tournament_deeprm_vft (ha9gv7hl), tournament_deeprm (q9050mka);
heuristics: results/heuristics_deeprm.  Wait of a model = its completion - mean job size of the episode (sizes are the same for all methods).
Writes docs/results/deeprm_metrics_tables.md and docs/results/deeprm_completion_time.png"""
import os, re, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODELS = {"model 9e90oa77 (10 slots)": "tournament_deeprm_n10", "model ha9gv7hl (20 slots, ft)": "tournament_deeprm_vft", "model q9050mka (20 slots)": "tournament_deeprm"}
HEUR = {"SJF": "sjf", "SJF+BF": "sjf_bf", "Tetris (eps=0.5)": "tetris", "FIFO+BF": "fifo_bf"}
EXTRA = {"Tetris (eps=5)": "tetris5"}     # results/heuristics_deeprm_tetris5
cells = {}     # (m, n) -> {method: (reward, wait, completion)}
for f in glob.glob(f"{B}/results/heuristics_deeprm/m*_n*.npz"):
    m, n = map(int, re.search(r"m(\d+)_n(\d+)", f).groups()); z = np.load(f); d = cells.setdefault((m, n), {})
    for lab, k in HEUR.items(): d[lab] = (z[k].mean(), z[k + "_wait"].mean(), z[k + "_completion"].mean())
    size = z["mean_size"]
    g5 = f"{B}/results/heuristics_deeprm_tetris5/m{m}_n{n}.npz"
    if os.path.exists(g5):
        z5 = np.load(g5); d["Tetris (eps=5)"] = (z5["tetris5"].mean(), z5["tetris5_wait"].mean(), z5["tetris5_completion"].mean())
    for lab, dirn in MODELS.items():
        g = f"{B}/results/{dirn}/m{m}_n{n}.npz"
        if os.path.exists(g):
            t = np.load(g); k = min(len(t["t_reward"]), len(size))
            d[lab] = (t["t_reward"][:k].mean(), (t["t_completion"][:k] - size[:k]).mean(), t["t_completion"][:k].mean())
methods = list(MODELS) + list(HEUR) + list(EXTRA)
out = ["# DeepRM environment: reward / average wait / average completion time per cell (5000 episodes, same seeds)\n",
       "Each cell: `reward / avg wait / avg completion` (completion = wait + job size). Lower wait/completion is better; best completion in **bold**.\n"]
for m in sorted({k[0] for k in cells}):
    out.append(f"\n## {m} machine{'s' if m > 1 else ''}\n")
    cols = [x for x in methods if any(x in cells[k] for k in cells if k[0] == m)]
    out.append("| jobs | " + " | ".join(cols) + " |"); out.append("|---|" + "---|" * len(cols))
    for n in sorted(k[1] for k in cells if k[0] == m):
        c = cells[(m, n)]; best = min(v[2] for v in c.values())
        out.append(f"| {n} | " + " | ".join(("**" if c[x][2] == best else "") + f"{c[x][0]:.1f} / {c[x][1]:.2f} / {c[x][2]:.2f}" + ("**" if c[x][2] == best else "") if x in c else "-" for x in cols) + " |")
open(f"{B}/docs/results/deeprm_metrics_tables.md", "w").write("\n".join(out) + "\n")
ms = sorted({k[0] for k in cells}); fig, axs = plt.subplots(2, 4, figsize=(26, 10), sharey=False)
for m, ax in zip(ms, axs.flat):
    for i, x in enumerate(methods):
        pts = sorted((n, cells[(mm, n)][x][2]) for (mm, n) in cells if mm == m and x in cells[(mm, n)])
        if pts: ax.plot(*zip(*pts), "o-" if x in HEUR else "s--", c=f"C{i}", label=x)
    ax.set_title(f"{m} machine{'s' if m > 1 else ''}"); ax.set_xlabel("jobs"); ax.set_ylabel("average completion time"); ax.grid(alpha=.25)
axs[0, 0].legend(fontsize=8)
fig.suptitle("Average completion time (wait + size), DeepRM environment, 5000 episodes per cell"); fig.tight_layout()
fig.savefig(f"{B}/docs/results/deeprm_completion_time.png", dpi=110)
