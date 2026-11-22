"""Comparison chart: end-to-end tournament-trained models (results/tournament_loop) vs SJF, vs the 1-10-job model
(results/tournament_var), vs the old 10-job model (numbers from the earlier table) and the regular per-size model.
Win % excludes ties. Sizes without results yet are left out; run again when more are done."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ns = list(range(20, 101, 10))
OLD = {20: (1908, 1238), 30: (2609, 1701), 40: (2944, 1828), 50: (3018, 1899), 60: (2985, 1977), 70: (2965, 2018), 80: (3044, 1940)}


def wl(d):
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    return w, len(d) - w - l, l, 100 * w / max(w + l, 1)


loop, var, reg, rows, have = {}, {}, {}, [], []
for n in ns:
    p = f"{ROOT}/results/tournament_loop/loop_m1_n{n}.npz"
    if not os.path.exists(p):
        continue
    z = np.load(p); t, s = z["t_reward"], z["s_reward"]
    loop[n] = wl(t - s) + (float((t - s).mean()),)
    v = np.load(f"{ROOT}/results/tournament_var/var_m1_n{n}.npz")["t_reward"]
    var[n] = wl(v - s)
    r = np.load(f"{ROOT}/results/regular/ppo_m1_n{n}.npy")[:len(s)]
    reg[n] = wl(r - s)
    a, b = wl(t - v), wl(t - r)
    f = lambda x: f"{x[0]}/{x[1]}/{x[2]} ({x[3]:.0f}%)"
    rows.append(f"| {n} | {f(loop[n])} | {loop[n][4]:+.1f} | {f(var[n])} | {f(reg[n])} | {f(a)} | {f(b)} |")
    have.append(n)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].plot(have, [loop[n][3] for n in have], "o-", lw=2.5, label="tournament-trained end-to-end (new)")
ax[0].plot(have, [var[n][3] for n in have], "s-", label="trained on 1-10 jobs, tournament eval")
ax[0].plot([n for n in have if n in OLD], [100 * OLD[n][0] / sum(OLD[n]) for n in have if n in OLD], "d--", label="old 10-job model, tournament eval")
ax[0].plot(have, [reg[n][3] for n in have], "^:", label="regular model per size")
ax[0].axhline(50, c="gray", lw=.8); ax[0].set_ylim(0, 100); ax[0].set_xlabel("jobs"); ax[0].set_ylabel("win % vs SJF (ties excluded)")
ax[0].set_title("Win % vs SJF, 1 machine"); ax[0].legend(fontsize=8)
ax[1].bar(have, [loop[n][4] for n in have], width=6, color="C0"); ax[1].axhline(0, c="gray", lw=.8)
ax[1].set_xlabel("jobs"); ax[1].set_ylabel("mean reward gain vs SJF"); ax[1].set_title("End-to-end tournament model: mean gain vs SJF")
fig.tight_layout(); fig.savefig(f"{ROOT}/docs/results/tournament_loop_comparison.png", dpi=130)
hdr = "| jobs | e2e vs SJF | e2e gain | 1-10 model vs SJF | regular vs SJF | e2e vs 1-10 model | e2e vs regular |\n|---|---|---|---|---|---|---|\n"
open(f"{ROOT}/docs/results/tournament_loop_comparison.md", "w").write(
    "# End-to-end tournament-trained models, 1 machine (wins/ties/losses, win% without ties)\n\n" + hdr + "\n".join(rows) + "\n")
print(hdr + "\n".join(rows))
