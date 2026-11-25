import numpy as np, matplotlib, glob, re; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A="results/heur_orig_m2/m2_n%d.npz"; B="results/heur_orig_m2_tetris5/m2_n%d.npz"
ns=sorted(int(re.findall(r'_n(\d+)',f)[0]) for f in glob.glob("results/heur_orig_m2_tetris5/*.npz"))
S={"SJF":("sjf",A),"SJF+BestFit":("sjf_bf",A),"FIFO+BestFit":("fifo_bf",A),"Tetris eps=0.5":("tetris",A),"Tetris eps=5":("tetris5",B)}
comp={k:[] for k in S};per={k:[] for k in S}
for n in ns:
    for k,(key,P) in S.items():
        d=np.load(P%n); comp[k].append(d[key+"_completion"].mean())
base=np.array(comp["SJF"])
fig,ax=plt.subplots(1,2,figsize=(14,5))
for k in S:
    ax[0].plot(ns,comp[k],marker='o',label=k)
    if k!="SJF": ax[1].plot(ns,100*(np.array(comp[k])-base)/base,marker='o',label=k)
ax[0].set_title("Avg completion time (2 machines, original workload)");ax[0].legend();ax[0].set_xlabel("jobs")
ax[1].set_title("Completion time vs SJF (% worse, <0 = better)");ax[1].axhline(0,color='k');ax[1].set_xlabel("jobs");ax[1].legend()
ax[1].set_yscale('symlog',linthresh=1)
plt.tight_layout();plt.savefig("docs/results/heuristics_orig_m2.png",dpi=130)
for i,n in enumerate(ns): print(n,{k:round(100*(comp[k][i]-base[i])/base[i],2) for k in S if k!="SJF"})
