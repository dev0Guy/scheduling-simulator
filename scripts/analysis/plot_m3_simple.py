import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="results/"; ns=list(range(10,101,10))+[500,1000]
f=lambda n:R+("tournament_m3_mid/m3_n%d.npz"%n if n<=100 else "tournament_m3_big/m3_n%d.npz"%n)
w=[];g=[]
for n in ns:
    d=np.load(f(n)); a,b=d['t_reward'],d['s_reward']
    W=(a>b).sum();L=(a<b).sum(); w.append(100*W/max(W+L,1))
    g.append(100*(d['s_completion'].mean()-d['t_completion'].mean())/d['s_completion'].mean())
x=range(len(ns)); lab=[str(n) for n in ns]
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
ax[0].bar(x,w,color=['tab:green' if v>50 else 'tab:red' for v in w]);ax[0].axhline(50,color='k',ls='--')
ax[0].set_title("Win % vs SJF (ties excluded)");ax[0].set_ylabel("%")
ax[1].bar(x,g,color=['tab:green' if v>0 else 'tab:red' for v in g]);ax[1].axhline(0,color='k')
ax[1].set_title("Completion time improvement vs SJF (%, >0 = better)")
for a in ax: a.set_xticks(list(x));a.set_xticklabels(lab);a.set_xlabel("jobs")
plt.tight_layout();plt.savefig("docs/results/m3_simple_vs_sjf.png",dpi=130)
for n,a,b in zip(ns,w,g):print(n,round(a,1),round(b,2))
