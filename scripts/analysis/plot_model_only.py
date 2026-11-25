import numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns=[n for n in list(range(10,101,10))+list(range(150,501,50)) if os.path.exists(f"results/model_only_orig/m2_n{n}.npz")]
D={n:np.load(f"results/model_only_orig/m2_n{n}.npz") for n in ns}
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
for a,(key,title) in zip(ax,[("t_wait","Average wait time"),("t_completion","Average completion time"),("t_reward","Average reward")]):
    m=np.array([D[n][key].mean() for n in ns]);s=np.array([D[n][key].std() for n in ns])
    a.plot(ns,m,marker='o',color='tab:blue');a.fill_between(ns,m-s,m+s,alpha=.25,color='tab:blue',label="mean ± 1 std (over episodes)")
    a.set_title(title);a.set_xlabel("jobs");a.legend();a.grid(alpha=.3)
fig.suptitle("q1w0791o (50 slots, 2 machines) - model only, original workload");plt.tight_layout();plt.savefig("docs/results/q1w0791o_model_only.png",dpi=130)
print("n mean_wait std_wait mean_compl std_compl mean_reward std_reward")
for n in ns:
    d=D[n];print(n,*[round(float(f(d[k])),1) for k in("t_wait","t_completion","t_reward") for f in (np.mean,np.std)],len(d["t_wait"]))
