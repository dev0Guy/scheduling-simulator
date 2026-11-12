"""Pre-train the set autoencoder on many collected machine states, logging the loss every epoch.

    .venv/bin/python scripts/machine_ae/train.py "/tmp/ae_big_k4_n10.npz,/tmp/ae_big_k4_n30.npz,/tmp/ae_big_k4_n50.npz" models/machine_ae_k4.pt [prior_weight=1.0] [epochs=100]

Hold-out = the last 5% of every file (whole episodes, never seen in training). Per epoch it prints, for train and
hold-out: Chamfer reconstruction loss, prior loss (view vs elementwise max of the free grids) and the total
(recon + prior_weight * prior); the history is also written to <out>.csv. At the end it reports the decision that
matters for scheduling: does "job fits the view" agree with "job fits SOME real machine" (on the real pending jobs
of the hold-out states of the file that has N=30 jobs), and checks the encoder is invariant to the machine order.
env: LR (default 1e-3), BATCH (default 512), PAIRS=1 (K=2 from pairs), AT_LEAST_MAX=1 (view >= elementwise max)
"""
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SetMachineAE, chamfer  # noqa: E402

files, out = sys.argv[1].split(","), sys.argv[2]
prior_w = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100
lr, bs = float(os.environ.get("LR", "1e-3")), int(os.environ.get("BATCH", "512"))
th.manual_seed(0)

PAIRS = os.environ.get("PAIRS") == "1"      # train a K=2 autoencoder on random machine pairs drawn from the K-machine states


def make_pairs(f, draws, seed):
    """Random 2-machine sets from (S, K, R, T) states; 15% of the pairs get a zero-padded machine (free = 0 everywhere)."""
    g = th.Generator().manual_seed(seed)
    out = []
    for _ in range(draws):
        order = th.rand(f.shape[0], f.shape[1], generator=g).argsort(1)[:, :2]
        pair = th.gather(f, 1, order[:, :, None, None].expand(-1, -1, *f.shape[2:])).clone()
        pad = th.rand(f.shape[0], generator=g) < 0.15
        pair[pad, 1] = 0.0
        out.append(pair)
    return th.cat(out)


tr_parts, val_parts, fit_val = [], [], None
for f in files:
    d = np.load(f)
    free = th.tensor(d["free"])
    cut = int(len(free) * 0.95)
    if PAIRS:
        tr_parts.append(make_pairs(free[:cut], 2, 1)); val_parts.append(make_pairs(free[cut:], 1, 2))
        continue
    tr_parts.append(free[:cut]); val_parts.append(free[cut:])
    if d["jobs"].shape[1] == 30:
        fit_val = (th.tensor(d["jobs"][cut:]).float(), th.tensor(d["status"][cut:]), free[cut:])
tr, val = th.cat(tr_parts), th.cat(val_parts)
print(f"train states {len(tr)}, hold-out states {len(val)}, machines K={tr.shape[1]}, prior_weight={prior_w}, lr={lr}, batch={bs}", flush=True)

AT_LEAST_MAX = os.environ.get("AT_LEAST_MAX") == "1"
model = SetMachineAE(tr.shape[1], at_least_max=AT_LEAST_MAX)
opt = th.optim.Adam(model.parameters(), lr)
sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)


def parts(f):
    view, rec = model(f)
    return chamfer(rec, f), ((view - f.max(1).values) ** 2).mean()


def evaluate(f):
    model.eval()
    with th.no_grad():
        r = p = 0.0
        for i in range(0, len(f), 4096):
            rr, pp = parts(f[i:i + 4096])
            r += float(rr) * len(f[i:i + 4096]); p += float(pp) * len(f[i:i + 4096])
    return r / len(f), p / len(f)


rows = ["epoch,train_recon,train_prior,train_total,val_recon,val_prior,val_total"]
print("epoch | train: recon  prior  total | hold-out: recon  prior  total", flush=True)
for ep in range(epochs):
    model.train()
    idx = th.randperm(len(tr))
    s_r = s_p = 0.0
    for i in range(0, len(idx), bs):
        f = tr[idx[i:i + bs]]
        f = f[:, th.randperm(f.shape[1])]                      # the machine order is arbitrary: shuffle every batch
        rec, pri = parts(f)
        loss = rec + prior_w * pri
        opt.zero_grad(); loss.backward(); opt.step()
        s_r += float(rec) * len(f); s_p += float(pri) * len(f)
    sched.step()
    tr_r, tr_p = s_r / len(idx), s_p / len(idx)
    v_r, v_p = evaluate(val)
    rows.append(f"{ep + 1},{tr_r:.6f},{tr_p:.6f},{tr_r + prior_w * tr_p:.6f},{v_r:.6f},{v_p:.6f},{v_r + prior_w * v_p:.6f}")
    print(f"{ep + 1:5d} | {tr_r:.5f} {tr_p:.5f} {tr_r + prior_w * tr_p:.5f} | {v_r:.5f} {v_p:.5f} {v_r + prior_w * v_p:.5f}", flush=True)
open(out.replace(".pt", ".csv"), "w").write("\n".join(rows) + "\n")

model.eval()
with th.no_grad():
    f = val[:20000]
    shuf = model.encode(f[:, th.randperm(f.shape[1])])
    print("order invariance, max |view - view(shuffled)|:", float((model.encode(f) - shuf).abs().max()))
    if fit_val is not None:
        j, st, ff = fit_val
        view = model.encode(ff)
        need = j / 255.0                                           # (S, N, R, T)
        pend = st == 1
        fit_real = (need[:, None] <= ff[:, :, None] + 1e-6).flatten(3).all(-1).any(1)
        for name, v in (("view", view), ("elementwise max (no learning)", ff.max(1).values)):
            fv = (need <= v[:, None] + 1e-6).flatten(2).all(-1)
            tp = (fv & fit_real & pend).sum().item(); fp = (fv & ~fit_real & pend).sum().item(); fn = (~fv & fit_real & pend).sum().item()
            print(f"fit agreement [{name}]: precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}")
th.save({"state": model.state_dict(), "k": tr.shape[1], "at_least_max": AT_LEAST_MAX}, out)
