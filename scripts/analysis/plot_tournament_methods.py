"""All tournament variants vs SJF, 1 machine: win % (ties excluded) as a function of the number of jobs."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = f"{ROOT}/results"
OLD = {20: (1908, 1238), 30: (2609, 1701), 40: (2944, 1828), 50: (3018, 1899), 60: (2985, 1977), 70: (2965, 2018), 80: (3044, 1940)}


def wl(d):
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    return w, l, 100 * w / max(w + l, 1)


def collect(folder, prefix):
    out = {}
    for n in range(10, 101, 10):
        p = f"{R}/{folder}/{prefix}_m1_n{n}.npz"
        if os.path.exists(p):
            z = np.load(p)
            out[n] = wl(z["t_reward"] - z["s_reward"])
    return out


methods = [
    ("Old 10-job model, tournament eval (no extra training)", {n: (w, l, 100 * w / (w + l)) for n, (w, l) in OLD.items()}, "d--", "C0"),
    ("Trained on 1-10 jobs, tournament eval", collect("tournament_var", "var"), "s-", "C1"),
    ("End-to-end from scratch, v1 (collapsed)", collect("tournament_loop", "loop"), "x:", "C3"),
    ("End-to-end from scratch, v2 (fixed)", collect("tournament_loop_v2", "loop2"), "x-", "C4"),
    ("Fine-tuned from 10-job model", collect("tournament_loop_ft", "ft"), "o-", "C5"),
    ("Fine-tuned, best checkpoint", collect("tournament_loop_ft_best", "ftbest"), "o--", "C6"),
    ("Fine-tuned, round 2", collect("tournament_loop_ft2", "ft2"), "^-", "C7"),
    ("Fine-tuned + critic warm-up", collect("tournament_loop_ft3", "ft3"), "o-", "C2"),
]
fig, ax = plt.subplots(figsize=(10.5, 6))
rows = []
for name, d, style, c in methods:
    ns = sorted(d)
    if not ns:
        continue
    ax.plot(ns, [d[n][2] for n in ns], style, color=c, lw=2.2 if "warm-up" in name else 1.6, label=name)
    rows.append(f"| {name} | " + " | ".join(f"{d[n][2]:.1f}" if n in d else "-" for n in range(10, 101, 10)) + " |")
ax.axhline(50, c="gray", lw=.8)
ax.set_ylim(0, 80); ax.set_xlabel("number of jobs"); ax.set_ylabel("win % vs SJF (ties excluded)")
ax.set_title("Tournament variants vs SJF, 1 machine"); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{ROOT}/docs/results/tournament_methods_comparison.png", dpi=130)
hdr = "| method | " + " | ".join(str(n) for n in range(10, 101, 10)) + " |\n|---|" + "---|" * 10 + "\n"
open(f"{ROOT}/docs/results/tournament_methods_comparison.md", "w").write(
    "# Tournament variants vs SJF, 1 machine: win % without ties\n\n" + hdr + "\n".join(rows) + "\n")
print(hdr + "\n".join(rows))
