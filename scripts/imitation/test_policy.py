"""Test saved policy weights against SJF on the evaluation job sets (seed 42 + episode).

    .venv/bin/python scripts/imitation/test_policy.py models/bc_policy.pt 5000 /tmp/bc_test_5000.npz
"""
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain_policy import build_policy, test_policy  # noqa: E402

if __name__ == "__main__":
    weights, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    model, env, device = build_policy()
    model.policy.load_state_dict(th.load(weights, map_location=device))
    model.policy.set_training_mode(False)
    pol, sjf = test_policy(model.policy, n_episodes=n)
    np.savez(out, policy=pol, sjf=sjf)
    print("saved", out)
