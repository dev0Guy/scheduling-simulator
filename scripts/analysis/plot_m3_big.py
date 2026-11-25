"""Old 3-machine x 10-job model (perlec9c, original workload) in a tournament, 10..1000 jobs.
10-70 jobs: recovered from docs/results/scaling_ext_regular_tournament.md (5000 episodes); 100 jobs: 300-episode check (/tmp/chk_m3_n100.npz, optional);
200-1000: results/tournament_m3_big (100 episodes); 300/450/550/700: probe runs (30 episodes, /tmp/probe, optional).
Outputs docs/results/m3_old_model_big_tournament.png"""
import os, re, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
pts = []   # (n, wins, losses, comp_model, comp_sjf, episodes)
row = [l for l in open(f"{B}/docs/results/scaling_ext_regular_tournament.md") if l.startswith("| **3** |")][0].split("|")[2:]
for n, cell in zip(range(10, 80, 10), row):
    t = cell.split("טורניר:")[-1]
    w, ti, l = map(int, re.findall(r"\((\d+)\)", t)[:3]); c = re.search(r"זמן ([\d.]+) \(([\d.]+)\)", cell)
    pts.append((n, w, l, float(c.group(1)), float(c.group(2)), 5000, "doc"))
def add(path, n, src):
    if os.path.exists(path):
        z = np.load(path); d = z["t_reward"] - z["s_reward"]
        pts.append((n, int((d > 1e-3).sum()), int((d < -1e-3).sum()), z["t_completion"].mean(), z["s_completion"].mean(), len(d), src))
add("/tmp/chk_m3_n100.npz", 100, "check")
for f in glob.glob(f"{B}/results/tournament_m3_big/m3_n*.npz"): add(f, int(re.search(r"n(\d+)", f).group(1)), "main")
for f in glob.glob("/tmp/probe/m3_n*.npz"): add(f, int(re.search(r"n(\d+)", f).group(1)), "probe")
pts.sort()
fig, axs = plt.subplots(1, 2, figsize=(17, 6))
for src, mk, lab in [("doc", "o", "5000 episodes"), ("check", "s", "300 episodes"), ("main", "D", "100 episodes"), ("probe", "^", "30 episodes")]:
    P = [p for p in pts if p[6] == src]
    if not P: continue
    n = [p[0] for p in P]; win = np.array([100 * p[1] / (p[1] + p[2]) for p in P]); ep = np.array([p[5] for p in P])
    ci = 196 * np.sqrt(win / 100 * (1 - win / 100) / ep)
    axs[0].errorbar(n, win, yerr=ci, fmt=mk + "-" if src == "doc" else mk, capsize=3, label=lab)
    axs[1].plot(n, [100 * (p[3] - p[4]) / p[4] for p in P], mk + ("-" if src == "doc" else ""), label=lab)
axs[0].axhline(50, c="gray", ls=":"); axs[0].set_xscale("log"); axs[0].set_xlabel("jobs"); axs[0].set_ylabel("win % vs SJF (ties excluded)"); axs[0].set_title("Win rate"); axs[0].legend(); axs[0].grid(alpha=.25)
axs[1].axhline(0, c="gray", ls=":"); axs[1].set_xscale("log"); axs[1].set_xlabel("jobs"); axs[1].set_ylabel("completion time vs SJF (%, + = worse)"); axs[1].set_title("Completion-time gap"); axs[1].legend(); axs[1].grid(alpha=.25)
fig.suptitle("Old 3-machine x 10-job model, tournament over larger job pools (original workload)"); plt.tight_layout(); plt.savefig(f"{B}/docs/results/m3_old_model_big_tournament.png", dpi=110)
for p in pts: print(p[0], p[6], f"{100*p[1]/(p[1]+p[2]):.1f}%", f"{100*(p[3]-p[4])/p[4]:+.2f}%")
