"""Grouped bar charts (like the DeepRM-paper slowdown figure) of avg wait time and avg completion time, heuristics only.
Outputs docs/results/heuristics_{wait,completion}_bars.png"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); R = f"{B}/results"
H = {"SJF": "sjf", "SJF+BF": "sjf_bf", "Tetris (eps=0.5)": "tetris", "Tetris (eps=5)": "tetris5", "FIFO+BF": "fifo_bf"}
Ns = [10, 20, 40, 60, 100]
def get(M, N, key, suf):
    z = np.load(f"{R}/heuristics_deeprm/m{M}_n{N}.npz"); out = {}
    for lab, k in H.items():
        if k == "tetris5": z5 = np.load(f"{R}/heuristics_deeprm_tetris5/m{M}_n{N}.npz"); out[lab] = z5["tetris5" + ("_" + suf if suf else "")]
        else: out[lab] = z[k + ("_" + suf if suf else "")]
    return out
for suf, name in [("wait", "average wait time"), ("completion", "average completion time"), ("", "average reward (higher = better)")]:
    fig, axs = plt.subplots(2, 4, figsize=(28, 11)); w = .8 / len(H)
    for ax, M in zip(axs.flat, range(1, 9)):
        for j, N in enumerate(Ns):
            d = get(M, N, None, suf)
            for i, (lab, v) in enumerate(d.items()):
                ax.bar(j + (i - len(H) / 2 + .5) * w, v.mean(), w, yerr=1.96 * v.std() / len(v) ** .5, color=f"C{i}", label=lab if j == 0 else None)
        ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"{n} jobs" for n in Ns]); ax.set_ylabel(name); ax.set_title(f"{M} machine{'s' if M > 1 else ''}"); ax.grid(axis="y", alpha=.25)
    axs[0, 0].legend(fontsize=9); fig.suptitle(f"Heuristics: {name} " + ("" if not suf else " (lower = better)")); plt.tight_layout(); plt.savefig(f"{B}/docs/results/heuristics_{suf or "reward"}_bars.png", dpi=110); plt.close()
