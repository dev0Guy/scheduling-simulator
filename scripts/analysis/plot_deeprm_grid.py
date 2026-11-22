"""DeepRM-model tournament + 2-best-machines: win % vs SJF (ties excluded) per machines x jobs, tie count annotated, plus mean reward gain."""
import re, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
res = {}
for f in glob.glob(f"{B}/results/{os.environ.get("RES","tournament_deeprm")}/m*_n*.npz"):
    m, n = map(int, re.search(r"m(\d+)_n(\d+)", f).groups())
    z = np.load(f); d = z["t_reward"] - z["s_reward"]
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    res[(m, n)] = (w, len(d) - w - l, l, d.mean())
ms = sorted({k[0] for k in res})
fig, axs = plt.subplots(1, 2, figsize=(18, 6.5))
for i, m in enumerate(ms):
    ks = sorted(k for k in res if k[0] == m); c = f"C{i}"
    ns = [k[1] for k in ks]
    ys = [100 * res[k][0] / max(res[k][0] + res[k][2], 1) for k in ks]
    axs[0].plot(ns, ys, "o-", c=c, label=f"{m} machines")
    for k, y in zip(ks, ys): axs[0].annotate(str(res[k][1]), (k[1], y), textcoords="offset points", xytext=(0, 6), fontsize=7, color=c, ha="center")
    axs[1].plot(ns, [res[k][3] for k in ks], "o-", c=c, label=f"{m} machines")
axs[0].axhline(50, c="gray", ls=":"); axs[0].set_ylabel("win % vs SJF (ties excluded)"); axs[0].set_title("Win rate (number = tie count)")
axs[1].axhline(0, c="gray", ls=":"); axs[1].set_ylabel("mean reward gain vs SJF"); axs[1].set_title("Mean reward gain")
for a in axs: a.set_xlabel("jobs"); a.grid(alpha=.25); a.legend(fontsize=8)
fig.suptitle(os.environ.get("TITLE","DeepRM-heavy model (2 machines x 20 jobs)")+", tournament over jobs + 2 best machines shown, 5000 episodes per cell")
fig.tight_layout(); fig.savefig(f"{B}/docs/results/{os.environ.get("OUT","deeprm_tournament_grid")}.png", dpi=120)
for k in sorted(res): print(k, res[k][:3], f"{100*res[k][0]/max(res[k][0]+res[k][2],1):.1f}%", f"{res[k][3]:+.2f}")
