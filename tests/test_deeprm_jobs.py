import numpy as np

from scheduling_simulator.core.creator import generate_cluster_python

CFG = {"n_machines": 2, "n_jobs": 20, "n_resource": 3, "n_time": 15, "max_capacity": 255}


def _obs(seed, monkeypatch):
    monkeypatch.setenv("JOB_GEN_MODE", "deeprm")
    return generate_cluster_python(CFG, np.random.default_rng(seed)).get_observation().to_dict()


def test_durations_follow_the_paper(monkeypatch):
    sizes = np.concatenate([_obs(s, monkeypatch)["size"] for s in range(300)])
    assert set(np.unique(sizes)) <= set(range(1, 4)) | set(range(10, 16))
    assert abs((sizes <= 3).mean() - 0.8) < 0.03


def test_dominant_and_other_demands(monkeypatch):
    for s in range(100):
        o = _obs(s, monkeypatch)
        for j in range(CFG["n_jobs"]):
            row = o["jobs_usage"][j, :, 0]
            dom = np.sort(row)[::-1]
            assert 63 <= dom[0] <= 127                 # 0.25 .. 0.5 of 255
            assert (dom[1:] >= 1).all() and (dom[1:] <= 25).all()   # 0.05 .. 0.1 of 255 (at least 1)
            # constant over the duration, zero afterwards
            size = int(o["size"][j])
            assert (o["jobs_usage"][j, :, :size] == row[:, None]).all() and (o["jobs_usage"][j, :, size:] == 0).all()


def test_first_job_arrives_at_zero_and_arrivals_are_sorted(monkeypatch):
    for s in range(50):
        a = _obs(s, monkeypatch)["arrival"]
        assert a[0] == 0 and (np.diff(a) >= 0).all()


def test_default_mode_is_unchanged(monkeypatch):
    monkeypatch.delenv("JOB_GEN_MODE", raising=False)
    cfg = {"n_machines": 1, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}
    o = generate_cluster_python(cfg, np.random.default_rng(42)).get_observation().to_dict()
    assert o["size"].max() <= 10 and o["arrival"].max() <= 8
