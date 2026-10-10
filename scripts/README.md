# scripts

| Folder | Contents | Run from the repo root |
|---|---|---|
| `solvers/` | Exact optimal solvers and their replay runners. `optimal_solver.py` + `run_optimal.py` for E1 (one job at a time); `optimal_solver_packing.py` + `run_optimal_packing.py` for jobs that can share the machine (one usage number per job); `optimal_solver_multires.py` + `run_optimal_multires.py` for jobs with a separate usage per resource | `.venv/bin/python scripts/solvers/run_optimal.py` |
| `plots/` | Per-episode PPO vs SJF delta plots from W&B history | `.venv/bin/python scripts/plots/plot_reward_delta.py` |
| `docs_tools/` | `make_diagrams.py` writes `docs/diagrams/*.svg`; `make_deck.js` writes `docs/presentations/experiments_deck.pptx` (needs `pptxgenjs`) | `.venv/bin/python scripts/docs_tools/make_diagrams.py` |
| `analysis/` | `compare_runs.py <ppo_run_id> <sjf_run_id>`: per-episode ties / PPO better / SJF better, then the comparison with ties removed | `.venv/bin/python scripts/analysis/compare_runs.py vy9asra5 xd83r7lf` |
