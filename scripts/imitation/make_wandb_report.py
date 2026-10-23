"""Build the W&B report that summarises all experiments (PPO, masking, imitation, DAgger).

1. Creates one tiny "summary" run per policy in the project `cluster-scheduling-report`
   (config: env, family; summary: mean reward, paired difference vs SJF, gap closed, W/T/L ...),
   computed from the per-episode results of the original runs / saved .npz files.
2. Creates the report (text, tables, charts) in the same project. Training curves are taken
   from the original runs in `cluster-scheduling-simulator`.

    .venv/bin/python scripts/imitation/make_wandb_report.py
"""
import os
import sys

import numpy as np
import wandb
import wandb_workspaces.reports.v2 as wr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import compare_runs as cr  # noqa: E402

ENTITY = "dev0guy"
SRC = "cluster-scheduling-simulator"      # original runs
REP = "cluster-scheduling-report"          # summary runs + the report
api = wandb.Api()


def episodes(run_id):
    _, d = cr.load(api, cr.PROJECT, run_id)
    return d.set_index(cr.X).sort_index()["eval/reward"]


def stats(pol, sjf, opt):
    d = pol - sjf
    n = len(d)
    gap = opt.mean() - sjf.mean()
    return {
        "mean_reward": float(pol.mean()), "paired_diff_vs_sjf": float(d.mean()),
        "ci95": float(1.96 * d.std(ddof=1) / n ** .5),
        "gap_closed_pct": float(100 * d.mean() / gap) if abs(gap) > 1e-12 else 0.0,
        "better_than_sjf": int((d > 1e-6).sum()), "tie": int((abs(d) <= 1e-6).sum()),
        "worse_than_sjf": int((d < -1e-6).sum()), "total": int(n),
        "equal_to_optimal_pct": float(100 * (abs(pol - opt) < 1e-6).mean()),
    }


def collect():
    rows = []        # (env, family, name, source, stats)

    def env_block(env, sjf_id, opt, runs, extra=()):
        s = episodes(sjf_id)
        o = opt[s.index.values.astype(int)]
        rows.append((env, "Optimal", "Optimal", "exact solver", stats(o, s.values, o)))
        for fam, name, rid in runs:
            p = episodes(rid)
            idx = s.index.intersection(p.index)
            rows.append((env, fam, name, rid, stats(p.loc[idx].values, s.loc[idx].values, o[idx.values.astype(int)])))
        for fam, name, src, arr, sj in extra:
            rows.append((env, fam, name, src, stats(arr, sj, o)))
        rows.append((env, "SJF", "SJF", sjf_id, stats(s.values, s.values, o)))

    load = lambda f: (lambda z: z["reward"][np.argsort(z["episode"])])(np.load(f))
    bc, dg = np.load("/tmp/bc_test_5000.npz"), np.load("/tmp/dagger_test_5000.npz")
    env_block("A", "f4vkih6o", load("/tmp/opt_3res.npz"), [
        ("PPO scratch", "PPO baseline", "ob8g1j1u"), ("PPO scratch", "PPO baseline (2nd)", "mxzw1d28"),
        ("PPO scratch", "PPO entropy 0.15-0.03", "lauk5acn"), ("PPO scratch", "PPO net 512", "icvnlb3f"),
        ("PPO scratch", "PPO net 512 (2nd)", "55g3kkbr"), ("PPO scratch", "PPO no attention", "v3wyp6zb"),
        ("PPO scratch", "PPO no attention (2nd)", "nj5red65"), ("PPO scratch", "PPO SJF-relative reward", "km2jhgpg"),
        ("PPO scratch", "PPO SJF-relative reward (2nd)", "qur1u23x"),
        ("PPO scratch", "PPO explore: entropy hold", "i5wysdxy"), ("PPO scratch", "PPO explore: big batch", "20vuspg5"),
        ("PPO + mask", "PPO + mask (ent 0.05-0.01)", "bav4y767"), ("PPO + mask", "PPO + mask (ent 0.15-0.03)", "0nuxh44a"),
        ("PPO after imitation", "PPO after BC (A)", "cwzzndfn"), ("PPO after imitation", "PPO after BC (B)", "6gm95uhc")],
        [("Imitation", "Behaviour cloning", "bc_test_5000.npz", bc["policy"], bc["sjf"]),
         ("DAgger", "DAgger (iter 5)", "dagger_test_5000.npz", dg["policy"], dg["sjf"])])
    env_block("B", "hux3eqdd", load("/tmp/opt_rand.npz"), [
        ("PPO + mask", "PPO + mask (ent 0.05-0.01)", "pj5by5dg"), ("PPO + mask", "PPO + mask (ent 0.15-0.03)", "x77j3j04")])
    o2, b2, d2 = np.load("/tmp/opt_2m.npz")["reward"], np.load("/tmp/bc2m_test_5000.npz"), np.load("/tmp/dagger2m_test_5000.npz")
    s2 = b2["sjf"]
    rows.append(("C", "Optimal", "Optimal", "exact A* solver", stats(o2, s2, o2)))
    rows.append(("C", "Imitation", "Behaviour cloning", "bc2m_test_5000.npz", stats(b2["policy"], s2, o2)))
    rows.append(("C", "DAgger", "DAgger (iter 6)", "dagger2m_test_5000.npz", stats(d2["policy"], s2, o2)))
    rows.append(("C", "SJF", "SJF", "6rfszvov", stats(s2, s2, o2)))
    return rows


def collect_final(rows):
    """Rows for env "A-final": the models at the END of training (2M steps), evaluated offline with
    scripts/imitation/eval_with_rule.py (/tmp/fin_<tag>_<id>.npz), on the same 5000 job sets."""
    s = episodes("f4vkih6o")
    o = np.load("/tmp/opt_3res.npz")
    opt = o["reward"][np.argsort(o["episode"])]
    rows.append(("A-final", "Optimal", "Optimal", "exact solver", stats(opt, s.values, opt)))
    groups = [
        ("PPO scratch", "Unmasked", "final", [("b18zno0o", 1), ("bj8j70tf", 2), ("rryyfl76", 3), ("w7c4m4b1", 4)]),
        ("PPO + mask", "Masked", "final", [("meahqy5r", 1), ("2ges9t6z", 2), ("c1msmg2b", 3), ("2wmpntfb", 4)]),
        ("PPO + mask", "Masked, big batch", "final", [("n48o070u", 1), ("gbduigll", 2)]),
        ("PPO + mask", "Masked, reward/100", "final", [("dc1okvl3", 1), ("syvz0k55", 2)]),
        ("PPO + mask", "Masked, clean reward/100", "final", [("xw5c0hau", 1), ("ky7dxq5g", 2)]),
        ("PPO + mask + rule", "Masked, reward/100 + idle rule at inference", "finalrule", [("dc1okvl3", 1), ("syvz0k55", 2)]),
        ("PPO + mask + rule", "Masked, clean reward/100 + idle rule at inference", "finalrule", [("xw5c0hau", 1), ("ky7dxq5g", 2)]),
        ("PPO + mask + rule", "Masked + idle rule (trained), original reward", "final", [("dja2teug", 1), ("c58esc00", 2)]),
        ("PPO + mask + rule", "Masked + idle rule (trained), clean reward/100", "final", [("8tgjlqxs", 1), ("w9hbwau2", 2)]),
    ]
    for fam, name, tag, runs in groups:
        for rid, seed in runs:
            f = f"/tmp/fin_{tag}_{rid}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            ep = z["episode"]
            rows.append(("A-final", fam, f"{name}, seed {seed}", rid, stats(z["reward"], s.loc[ep].values, opt[ep])))
    rows.append(("A-final", "SJF", "SJF", "f4vkih6o", stats(s.values, s.values, opt)))
    return rows


def log_summary_runs(rows):
    for env, fam, name, src, st in rows:
        run = wandb.init(project=REP, entity=ENTITY, name=f"{env} | {name}", reinit=True, tags=["report-summary"],
                         config={"env": env, "family": fam, "policy": name, "source": src})
        for k, v in st.items():
            run.summary[k] = v
        run.finish()


def md_table(rows, env):
    out = ["| Policy | Mean reward | Paired diff vs SJF (95% CI) | Better / Tie / Worse | Gap closed | Source |", "|---|---|---|---|---|---|"]
    for e, fam, name, src, st in rows:
        if e != env:
            continue
        diff = "n/a" if fam == "SJF" else f"{st['paired_diff_vs_sjf']:+.2f} ± {st['ci95']:.2f}"
        cnt = "n/a" if fam == "SJF" else f"{st['better_than_sjf']} / {st['tie']} / {st['worse_than_sjf']}"
        gap = "n/a" if fam == "SJF" else f"{st['gap_closed_pct']:.1f}%" if abs(st['gap_closed_pct']) < 1000 else "n/m"
        out.append(f"| {name} | {st['mean_reward']:.2f} | {diff} | {cnt} | {gap} | `{src}` |")
    return "\n".join(out)


def results_grid(env, title):
    rs = wr.Runset(entity=ENTITY, project=REP, name=f"Environment {env}", filters=f"Config('env') == '{env}'")
    return wr.PanelGrid(runsets=[rs], panels=[
        wr.BarPlot(title=f"{title}: paired reward difference vs SJF (higher is better)", metrics=[wr.SummaryMetric("paired_diff_vs_sjf")], orientation="h", max_runs_to_show=40, layout=wr.Layout(x=0, y=0, w=12, h=9)),
        wr.BarPlot(title=f"{title}: share of the SJF-to-optimal gap closed (%)", metrics=[wr.SummaryMetric("gap_closed_pct")], orientation="h", max_runs_to_show=40, layout=wr.Layout(x=12, y=0, w=12, h=9)),
        wr.BarPlot(title=f"{title}: episodes better than SJF", metrics=[wr.SummaryMetric("better_than_sjf")], orientation="h", max_runs_to_show=40, layout=wr.Layout(x=0, y=9, w=12, h=9)),
        wr.BarPlot(title=f"{title}: episodes worse than SJF", metrics=[wr.SummaryMetric("worse_than_sjf")], orientation="h", max_runs_to_show=40, layout=wr.Layout(x=12, y=9, w=12, h=9)),
    ])


def runset(name, ids, project=SRC):
    return wr.Runset(entity=ENTITY, project=project, name=name, filters="Metric('ID') in [" + ", ".join(f"'{i}'" for i in ids) + "]")


def build(rows, update_url=None):
    ppo_ids = ["ob8g1j1u", "mxzw1d28", "lauk5acn", "i5wysdxy", "20vuspg5", "bav4y767", "0nuxh44a"]
    il_ids = ["qua2lwh5", "qxos89fx", "v014in2y", "o1h40tlh"]
    curves = wr.PanelGrid(runsets=[runset("PPO from scratch, with and without the mask", ppo_ids)], panels=[
        wr.LinePlot(title="Training episode reward (rollout/ep_rew_mean)", x="global_step", y=["rollout/ep_rew_mean"], smoothing_factor=0.6, layout=wr.Layout(x=0, y=0, w=12, h=8)),
        wr.LinePlot(title="Policy entropy (train/entropy_loss)", x="global_step", y=["train/entropy_loss"], smoothing_factor=0.6, layout=wr.Layout(x=12, y=0, w=12, h=8)),
        wr.LinePlot(title="50-episode evaluation (eval/mean_reward)", x="global_step", y=["eval/mean_reward"], layout=wr.Layout(x=0, y=8, w=12, h=8)),
        wr.LinePlot(title="Critic quality (train/explained_variance)", x="global_step", y=["train/explained_variance"], smoothing_factor=0.6, layout=wr.Layout(x=12, y=8, w=12, h=8)),
        wr.LinePlot(title="Update size (train/approx_kl)", x="global_step", y=["train/approx_kl"], smoothing_factor=0.6, layout=wr.Layout(x=0, y=16, w=12, h=8)),
    ])
    il = wr.PanelGrid(runsets=[runset("Behaviour cloning and DAgger", il_ids)], panels=[
        wr.LinePlot(title="Behaviour cloning / DAgger loss", x="epoch", y=["train/loss"], layout=wr.Layout(x=0, y=0, w=12, h=8)),
        wr.LinePlot(title="Validation accuracy of the cloned actions", x="epoch", y=["train/val_accuracy", "train/val_start_accuracy"], layout=wr.Layout(x=12, y=0, w=12, h=8)),
        wr.LinePlot(title="DAgger: policy disagrees with the optimum (% of visited states)", x="iteration", y=["dagger/disagreement_pct"], layout=wr.Layout(x=0, y=8, w=12, h=8)),
        wr.LinePlot(title="DAgger: paired reward difference vs SJF (300-episode test)", x="iteration", y=["dagger/test300_paired_diff_vs_sjf"], layout=wr.Layout(x=12, y=8, w=12, h=8)),
        wr.LinePlot(title="Final test: per-episode reward minus SJF (smoothed)", x="evaluation/episode", y=["eval/diff_vs_sjf"], smoothing_factor=0.9, layout=wr.Layout(x=0, y=16, w=12, h=8)),
        wr.LinePlot(title="Final test: per-episode reward minus optimal (smoothed)", x="evaluation/episode", y=["eval/diff_vs_optimal"], smoothing_factor=0.9, layout=wr.Layout(x=12, y=16, w=12, h=8)),
    ])
    M = wr.MarkdownBlock
    blocks = [
        wr.H1("Cluster scheduling: PPO, invalid-action masking, imitation from the optimum and DAgger"),
        wr.TableOfContents(),
        wr.H2("Summary"),
        M("""* **Task.** One machine (or two), 10 jobs, 3 resources, 10 ticks. Each step the policy either starts a pending job or skips one tick. The reward is +1 per start, minus the number of active jobs per skipped tick, +100 when all jobs finish and -100 at the step limit, so it measures the sum of completion times.
* **Baselines.** SJF (shortest job first) and an **exact optimal solver**. In environment A the optimum is only 4.3% above SJF, in the random-usage environment B it is 8.6% above, in the 2-machine environment C it is 3.9 points above (about 24% of SJF's reward).
* **How the models are chosen matters.** PPO saves a checkpoint every 400k steps and tests it on only 50 episodes (standard error about 8, while the differences between models are 1-2). Picking the "best" of five such noisy numbers mostly picks luck and often an early checkpoint. The first tests in this report used that pick ("best-checkpoint test"); the section "Final models" re-tests the model at the END of training (2M steps) on the same 5000 job sets, and it changes several conclusions.
* **PPO from scratch, final models:** every variant without the extra idle rule lands about +1.7 to +2.0 above SJF (environment A), except one unmasked run that collapsed (seed 3: -13.4). Masked runs: +1.89, +2.02, +2.02, +2.00 (4 of 4 stable). Unmasked: +2.10, +1.80, -13.44, +1.49.
* **Invalid-action masking** removes the risk of collapse in these seeds (0 of 4 masked, 1 of 4 unmasked) and gives 0 failed starts, but it does not raise the ceiling: when an unmasked run does not collapse it is as good as a masked one (+1.80 vs +1.98 on average).
* **Reward scaling/cleaning, bigger batch, the idle-skip rule:** scaled and cleaned rewards are as good as the plain reward (+1.91, +1.73), the bigger batch is worse (+1.01) and the idle-skip rule hurts the final models (+0.65 to +0.90).
* **Imitation from the exact solver** closes 53.5% of the gap in A; **DAgger** closes **66.3%** (A) and **69.8%** (two machines); the best PPO runs close about 40%.
* **PPO after imitation** made the policy slightly worse (36.5% and 40.0%)."""),
        wr.H2("Setup and metrics"),
        M("""| Environment | Job usage per resource | Machines |
|---|---|---|
| A | one of {64, 128, 255}, drawn independently per resource | 1 |
| B | uniform random integer in [1, 255] | 1 |
| C | one of {64, 128, 255} | 2 |

All results use **5000 job sets (seeds 42..5041)**, the same for every policy. *Paired diff* = mean over episodes of (policy reward - SJF reward) with a 95% interval; *gap closed* = (policy - SJF) / (optimal - SJF); *better / tie / worse* counts episodes against SJF. Episodes where the policy ties SJF are 40-75% of the total."""),
        wr.H2("How each method works"),
        M("""**PPO.** An actor-critic: the actor outputs a probability for each of the 11 actions (0 = skip, k = start job k-1), the critic estimates the value of the state. Every update collects 8 environments x 512 steps, computes advantages with GAE (gamma = lambda = 0.995), and maximises a clipped objective `min(r*A, clip(r, 0.8, 1.2)*A)` where `r = pi_new/pi_old`, for 5 epochs over minibatches of 64, with a KL stop at 0.03 and a linearly decaying learning rate (2e-5). The total loss is `policy + 0.5*value - c_ent*entropy`.

**Exploration vs exploitation.** The policy is stochastic during training, so every action with non-zero probability gets tried. The entropy bonus `c_ent*H(pi)` rewards a spread-out distribution and is annealed from high to low, moving from exploration to exploitation. Too little entropy collapses the policy early; too much keeps it random (see the "explore" runs).

**Invalid-action masking.** A start is valid only if the job is PENDING and its usage fits the machine's free capacity on every resource and tick (exactly the environment's test); skip is always valid. The logits of invalid actions are set to -1e8 before the categorical distribution is built, so they get probability 0, no gradient and no entropy. The same mask is used when sampling, in the PPO update and at evaluation. Measured on SJF trajectories there are only 1.58 valid actions on average out of 11, and only "skip" is valid in 85.5% of states; an unmasked uniform policy puts 85.7% of its probability on actions that fail (and a failed start is turned into a skip, so these are disguised skips). With the mask, exploration is spent on real decisions only.

**masked vs masked + entropy.** Both use the mask and the same network. `masked` uses the baseline entropy schedule 0.05 -> 0.01; `masked_entropy` uses 0.15 -> 0.03. The entropy is computed over the valid actions only, and is 0 in the forced states, so the bonus acts only on the ~15% of states with a real choice. The stronger bonus did not help (A: +1.75 vs +2.10, B: +0.73 vs +1.60).

**Exact optimal solver.** Event-driven memoised search for one machine; A* with an admissible lower bound for several machines. Validated against exhaustive search and against the real environment (0 mismatches, 0 failed starts).

**Behaviour cloning.** The solver's schedules are replayed in the environment; the network is trained with cross-entropy on (observation, optimal action) pairs plus an MSE loss for the critic.

**DAgger.** Roll out the current policy, and in every state it visits re-run the exact solver *from that state* to get the optimal action; add the pairs to the data and retrain. This fixes the distribution shift of behaviour cloning, where the policy only sees states the optimum visits."""),
        wr.H2("The network"),
        M("""Shared by actor and critic, 899,028 parameters: scaled inputs and a status embedding -> a per-job encoder (shared weights) -> **fit attention** (query from the job's usage, key/value from the machine's free capacity, sigmoid, hard feasibility mask) -> 4-head self-attention between jobs -> attention pooling into a queue summary (256) -> dynamic machine embedding from the free-capacity grid and 9 exact pair features -> a 3162-number feature vector. The **actor** scores each job with a shared MLP (832 -> 128 -> 128 -> 1) and adds a separate skip score, giving 11 logits (then the mask). The **critic** is an MLP on the queue summary."""),
        wr.H2("Results, environment A (best-checkpoint test): usage {64, 128, 255}, one machine"),
        M("These results use the checkpoint with the highest 50-episode check (see the note in the summary). They are kept for reference; the next section uses the final models."),
        results_grid("A", "Environment A, best checkpoint"),
        M(md_table(rows, "A")),
        wr.H2("Final models, environment A: the model at the end of training (2M steps)"),
        M("Each model is evaluated offline on the same 5000 job sets (seeds 42..5041), deterministic policy, original reward. The offline evaluation of a best-checkpoint model reproduces the W&B test exactly (-117.19). Masked models are evaluated with the mask; models marked 'rule' use the idle-skip rule (skip is invalid when no job runs and a pending job fits)."),
        results_grid("A-final", "Environment A, final models"),
        M(md_table(rows, "A-final")),
        wr.H2("Results, environment B: random usage [1, 255], one machine"),
        results_grid("B", "Environment B"),
        M(md_table(rows, "B")),
        wr.H2("Results, environment C: two machines"),
        results_grid("C", "Environment C"),
        M(md_table(rows, "C") + "\n\nRewards are small here (about -15) because of the +100 completion bonus, so absolute differences and the share of the gap closed are more meaningful than percentages of SJF's reward."),
        wr.H2("Training curves of the PPO runs"),
        M("Baseline (`ob8g1j1u`, `mxzw1d28`), higher entropy (`lauk5acn`), the two exploration attempts (`i5wysdxy` entropy hold, `20vuspg5` big batch) and the two masked runs (`bav4y767`, `0nuxh44a`). The masked runs have a very low entropy value because in 85% of states only one action is valid, not because the policy collapsed."),
        curves,
        wr.H2("Imitation and DAgger"),
        il,
        wr.H2("Conclusions and caveats"),
        M("""1. **Use the final model, not the "best" checkpoint.** The 50-episode checks used to pick it are too noisy and often pick an early checkpoint. Re-testing the final models changed the picture: scaled and cleaned rewards went from -2.58 / -0.87 to +1.91 / +1.73 against SJF, the "deadlock" of the deterministic policy (19-39 timed-out episodes per run) almost disappeared (0-2), and the idle-skip rule turned out to hurt rather than help.
2. **Masking invalid actions** gave 4 of 4 stable runs against 3 of 4 without it (one unmasked run collapsed to -13.4) and 0 failed starts, but it did not raise the ceiling: non-collapsed unmasked runs average +1.80 against +1.98 masked. This is a hint about stability from 4 paired seeds, not a proof. The masked policy cannot be used without the mask (its logits for invalid actions were never trained).
3. **A bigger batch is worse** (+1.01, 2 seeds), probably because it makes 4x fewer updates. **Scaling or cleaning the reward** is neutral (+1.91, +1.73 vs +1.98). The **idle-skip rule** costs about 1 point because the optimum sometimes waits for a short job that is about to arrive.
4. **Imitating the exact optimum and correcting with DAgger** is clearly better (66-70% of the gap) than any PPO from scratch (about 40%) but needs an exact solver, which only exists for small instances.
5. With random usage the optimum is further from SJF and the best-checkpoint masked PPO closed only 19.7% of the gap (final models were not re-tested there).

**Caveats.** 2-4 runs per setting and a few points of seed-to-seed variance; the intervals are over job sets, not over training seeds, so they understate the real uncertainty. The final models are not selected on the 5000 job sets, so they carry no selection bias; the best-checkpoint results do, via the 50-episode checks. Environments B and C still show best-checkpoint results for PPO. DAgger iteration 5 was chosen by a 300-episode test, but a separate test on 1000 unseen seeds gave +3.45 ± 0.53. The observation contains information about jobs that have not arrived yet, for every policy and for the solver.

**Code.** Branches `v3-Dagger` (imitation, DAgger, multi-machine solver, W&B logging) and `v3-masked` (masking, seeds, reward wrapper, idle-skip rule, exploration runs). The training script now evaluates the final model by default (`use_best_model=False`)."""),
    ]
    if update_url:
        rep = wr.Report.from_url(update_url)
        rep.blocks = blocks
        rep.title = "Cluster scheduling: PPO, masking, imitation and DAgger"
        rep.description = "All experiments: PPO from scratch, invalid-action masking, imitation from the exact optimum and DAgger, against SJF and the optimum (updated with the final models)."
    else:
        rep = wr.Report(project=REP, entity=ENTITY, title="Cluster scheduling: PPO, masking, imitation and DAgger",
                        description="All experiments: PPO from scratch, invalid-action masking, imitation from the exact optimum and DAgger, against SJF and the optimum.",
                        blocks=blocks, width="fluid")
    rep.save()
    return rep.url


if __name__ == "__main__":
    # first run:   make_wandb_report.py            -> creates the summary runs and a new report
    # later runs:  make_wandb_report.py --update <report url>  -> adds the final-model rows and edits that report in place
    update = sys.argv[sys.argv.index("--update") + 1] if "--update" in sys.argv else None
    rows = collect_final(collect())
    for r in rows:
        print(r[0], r[2], {k: round(v, 2) for k, v in r[4].items() if k in ("mean_reward", "paired_diff_vs_sjf", "gap_closed_pct")})
    log_summary_runs([r for r in rows if r[0] == "A-final"] if update else rows)
    print("REPORT:", build(rows, update))
