import numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns=[n for n in list(range(10,101,10))+list(range(150,501,50)) if os.path.exists(f"results/tournament_deeprm_n50_10m/m2_n{n}.npz")]
w=[];g=[];e=[]
for n in ns:
    d=np.load(f"results/tournament_deeprm_n50_10m/m2_n{n}.npz");a,b=d['t_reward'],d['s_reward']
    W=(a>b).sum();L=(a<b).sum();w.append(100*W/max(W+L,1))
    r=(d['s_completion']-d['t_completion'])/d['s_completion'];g.append(100*r.mean());e.append(196*r.std()/len(r)**.5)
x=range(len(ns));fig,ax=plt.subplots(1,2,figsize=(14,4.5))
ax[0].bar(x,w,color=['tab:green' if v>50 else 'tab:red' for v in w]);ax[0].axhline(50,color='k',ls='--');ax[0].set_title("Win % vs SJF (ties excluded)")
ax[1].bar(x,g,yerr=e,color=['tab:green' if v>0 else 'tab:red' for v in g]);ax[1].axhline(0,color='k');ax[1].set_title("Completion time improvement vs SJF (%, 95% CI)")
for a in ax:a.set_xticks(list(x));a.set_xticklabels(ns,rotation=45);a.set_xlabel("jobs")
fig.suptitle("q1w0791o (50 slots, 2 machines, 10M steps) - DeepRM workload");plt.tight_layout();plt.savefig("docs/results/q1w0791o_vs_sjf.png",dpi=130)
