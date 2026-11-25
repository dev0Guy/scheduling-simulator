import numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns=[n for n in list(range(10,101,10))+list(range(150,501,50)) if os.path.exists(f"results/model_only_orig/m2_n{n}.npz")]
M={n:np.load(f"results/model_only_orig/m2_n{n}.npz") for n in ns};S={n:np.load(f"results/sjf_orig_m2/m2_n{n}.npz") for n in ns}
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
for a,(k,title,sign) in zip(ax,[("wait","Wait time: SJF - model (>0 = model better)",1),("completion","Completion time: SJF - model (>0 = model better)",1),("reward","Reward: model - SJF (>0 = model better)",-1)]):
    d=[sign*(S[n]["s_"+k]-M[n]["t_"+k]) for n in ns]   # per-episode, paired by seed
    m=np.array([x.mean() for x in d]);ci=np.array([1.96*x.std()/len(x)**.5 for x in d])
    a.errorbar(ns,m,yerr=ci,fmt='o-',capsize=3,color='tab:green');a.axhline(0,color='k')
    a.fill_between(ns,m-ci,m+ci,alpha=.2,color='tab:green');a.set_title(title,fontsize=10);a.set_xlabel("jobs");a.grid(alpha=.3)
fig.suptitle("q1w0791o vs SJF - paired per-episode difference, mean ± 95% CI (original workload, 2 machines)");plt.tight_layout();plt.savefig("docs/results/q1w0791o_diff_vs_sjf.png",dpi=130)
for n in ns:
    r=M[n]["t_reward"]-S[n]["s_reward"];c=S[n]["s_completion"]-M[n]["t_completion"]
    print(n,"reward %+.1f ±%.1f | completion %+.2f ±%.2f"%(r.mean(),1.96*r.std()/len(r)**.5,c.mean(),1.96*c.std()/len(c)**.5))
