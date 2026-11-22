"""SJF on the DeepRM-workload environment (JOB_GEN_MODE=deeprm), seeds 42+ep, same wrapper as training; compares with the
per-episode eval rewards of a W&B run.   usage: sjf_deeprm.py <wandb run id> <out.npz>   env: N_MACHINES N_JOBS N_TIME + job-gen env vars"""
import os, sys
from multiprocessing import Pool
import numpy as np
M, N, T = int(os.environ["N_MACHINES"]), int(os.environ["N_JOBS"]), int(os.environ.get("N_TIME", "10"))
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": T, "max_capacity": 255}


def run(ep):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    env = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)
    o, _ = env.reset(seed=42 + ep); sj = ShortestJobScheduler(); tot, done = 0.0, False
    while not done:
        sk, j, m = sj.select(o); o, r, done, _, _ = env.step(0 if sk else 1 + m * N + j); tot += float(r)
    return tot, float((o["wait_time"] + o["size"]).mean())


if __name__ == "__main__":
    import wandb
    h = wandb.Api().run(f"dev0guy/cluster-scheduling-simulator/{sys.argv[1]}").history(keys=["evaluation/episode", "eval/reward", "eval/avg_completion_time"], samples=6000, pandas=True).dropna().sort_values("evaluation/episode")
    with Pool(6) as p:
        s = np.array(p.map(run, range(len(h)), chunksize=20))
    t = h["eval/reward"].to_numpy(); d = t - s[:, 0]
    np.savez(sys.argv[2], t_reward=t, s_reward=s[:, 0], t_completion=h["eval/avg_completion_time"].to_numpy(), s_completion=s[:, 1])
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    print(f"episodes {len(d)}  model {t.mean():.2f}  SJF {s[:, 0].mean():.2f}  gain {d.mean():+.2f} +- {1.96 * d.std(ddof=1) / len(d) ** .5:.2f}")
    print(f"completion model {h['eval/avg_completion_time'].mean():.2f}  SJF {s[:, 1].mean():.2f}")
    print(f"W/T/L {w}/{len(d) - w - l}/{l}  win% without ties {100 * w / (w + l):.1f}")
