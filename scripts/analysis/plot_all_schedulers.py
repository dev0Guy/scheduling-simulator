import numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns=[n for n in list(range(10,101,10))+list(range(150,501,50))]
M={n:np.load(f"results/model_only_orig/m2_n{n}.npz") for n in ns};S={n:np.load(f"results/sjf_orig_m2/m2_n{n}.npz") for n in ns};H={n:np.load(f"results/heur_multi_orig_m2/m2_n{n}.npz") for n in ns}
def get(name,key):   # key in wait/completion/reward
    if name=="Model q1w0791o": return np.array([M[n]["t_"+key].mean() for n in ns])
    if name=="SJF": return np.array([S[n]["s_"+key].mean() for n in ns])
    return np.array([H[n][name+"_"+key].mean() for n in ns])
names=["Model q1w0791o","SJF","sjf_bf","fifo_bf","tetris_eps0.5","tetris_eps2","tetris_eps5","tetris_eps20"]
lab={"sjf_bf":"SJF+BestFit","fifo_bf":"FIFO+BestFit","tetris_eps0.5":"Tetris eps=0.5","tetris_eps2":"Tetris eps=2","tetris_eps5":"Tetris eps=5","tetris_eps20":"Tetris eps=20"}
sty={"Model q1w0791o":dict(color='k',lw=2.5,ls='--',marker='s'),"SJF":dict(color='tab:red',lw=2,marker='o')}
fig,ax=plt.subplots(1,3,figsize=(18,5))
for a,(k,t) in zip(ax,[("wait","Average wait time"),("completion","Average completion time"),("reward","Average reward")]):
    for nm in names: a.plot(ns,get(nm,k),label=lab.get(nm,nm),**sty.get(nm,dict(marker='.')))
    a.set_title(t);a.set_xlabel("jobs");a.grid(alpha=.3)
ax[0].legend(fontsize=8);fig.suptitle("All schedulers, original workload, 2 machines");plt.tight_layout();plt.savefig("docs/results/all_schedulers_orig_m2.png",dpi=130)
base=get("SJF","completion");fig,a=plt.subplots(figsize=(10,5.5))
for nm in names:
    if nm=="SJF": continue
    a.plot(ns,100*(get(nm,"completion")-base)/base,label=lab.get(nm,nm),**sty.get(nm,dict(marker='.')))
a.axhline(0,color='k',lw=.8);a.set_yscale('symlog',linthresh=1);a.set_xlabel("jobs");a.set_ylabel("completion time vs SJF (% worse; <0 = better)");a.grid(alpha=.3);a.legend(fontsize=8)
a.set_title("Completion time relative to SJF (symlog axis)");plt.tight_layout();plt.savefig("docs/results/all_schedulers_orig_m2_vs_sjf.png",dpi=130)
print("n",*[lab.get(x,x) for x in names if x!="SJF"])
for i,n in enumerate(ns):print(n,*[round(float(100*(get(x,"completion")[i]-base[i])/base[i]),2) for x in names if x!="SJF"])
