"""Diagnostic: how often does the tournament pick the same job as SJF, and how big is the pending queue, vs N."""
import os, sys, numpy as np
N=int(sys.argv[1]); EP=int(sys.argv[2])
os.environ.update(N_MACHINES="2",N_JOBS=str(N),MODEL_MACHINES="2",MODEL="models/q1w0791o/final_model.zip",PYTHONPATH="src",
    JOB_USAGE_MODE="random",JOB_GEN_MODE="deeprm",DEEPRM_LOAD="1.5",DEEPRM_DEMAND_SCALE="2",N_TIME="15")
sys.path.insert(0,"scripts/tournament"); sys.path.insert(0,"src")
import tournament_eval as T
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
from scheduling_simulator.scheduler.abc_scheduler import Scheduler
T._load(); sj=ShortestJobScheduler(); agree=tot=0; pend=[]; sizeratio=[]
for ep in range(EP):
    env=FailureSkipTimeWrapper(SchedulingEnviorment(T.CFG,render_mode="none"),max_time=11*N)
    obs,_=env.reset(seed=42+ep); done=False
    while not done:
        if Scheduler.options(obs):
            a=T.choose(obs); skip,j,m=sj.select(obs); b=0 if skip else 1+m*N+j
            tot+=1; agree+=(a==b); pend.append(int((obs["status"]==0).sum()))
            if a and b: sizeratio.append(float(obs["size"][(a-1)%N])/max(float(obs["size"][(b-1)%N]),1))
        else: a=0
        obs,r,te,tr,_=env.step(a); done=te or tr
print(f"N={N} decisions={tot} same-action-as-SJF={100*agree/tot:.1f}% mean pending(status0)={np.mean(pend):.1f} max={max(pend)} mean size(model pick)/size(SJF pick)={np.mean(sizeratio):.3f}")
