import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "tournament"))
from machine_select import select_machines, PENDING   # noqa: E402

R, T, CAP = 3, 10, 255


def _state(used):
    """used[m] = constant usage of machine m on every resource and tick."""
    usage = np.stack([np.full((R, T), u, np.int32) for u in used])
    return usage, np.full_like(usage, CAP)


def _job(u, size=3):
    j = np.zeros((R, T), np.int32)
    j[:, :size] = u
    return j


def test_picks_the_two_tightest_machines_that_fit():
    usage, cap = _state([0, 200, 100, 250, 150, 50])
    jobs = np.stack([_job(50)])
    # free: 255, 55, 155, 5, 105, 205 -> fits (>=50): m0, m1, m2, m4, m5 ; tightest two: m1 (55), m4 (105)
    sel = select_machines(usage, cap, jobs, np.array([PENDING]), 2, np.random.default_rng(0))
    assert sel == [1, 4]


def test_machines_that_do_not_fit_are_skipped_even_if_tight():
    usage, cap = _state([240, 0, 100])          # m0 has 15 free, job needs 50
    sel = select_machines(usage, cap, np.stack([_job(50)]), np.array([PENDING]), 2, np.random.default_rng(0))
    assert sel == [1, 2]


def test_fit_is_checked_on_every_resource_and_tick():
    usage, cap = _state([0, 0, 0])
    usage[1, 2, 7] = 250                         # m1: one cell nearly full -> job (needs 50 for 3 ticks) fits, 10-tick one not
    job = _job(50, size=10)
    sel = select_machines(usage, cap, np.stack([job]), np.array([PENDING]), 1, np.random.default_rng(0))
    assert sel[0] != 1


def test_union_over_the_pending_jobs_of_the_slice_and_non_pending_ignored():
    usage, cap = _state([0, 200, 230])
    jobs = np.stack([_job(100), _job(20), _job(5)])
    # job0 (running, ignored) ; job1 fits m0, m1 (55 free) ; job2 pending fits m0, m1, m2 (25 free) -> tightest: m2, m1
    sel = select_machines(usage, cap, jobs, np.array([2, PENDING, PENDING]), 2, np.random.default_rng(0))
    assert sel == [1, 2]


def test_fills_deterministically_when_fewer_than_k_fit():
    usage, cap = _state([250, 240, 0, 250, 245])          # only machine 2 fits; fill = most free non-fitting (machine 1: 15 free)
    for seed in range(10):
        sel = select_machines(usage, cap, np.stack([_job(100)]), np.array([PENDING]), 2, np.random.default_rng(seed))
        assert sel == [1, 2]


def test_nothing_fits_is_deterministic_most_free_machines():
    usage, cap = _state([250, 240, 230, 250, 245, 250, 250, 250])
    for seed in range(10):
        sel = select_machines(usage, cap, np.stack([_job(100)]), np.array([PENDING]), 2, np.random.default_rng(seed))
        assert sel == [1, 2]


def test_no_pending_job_is_deterministic_and_never_random():
    usage, cap = _state([0] * 8)
    outs = {tuple(select_machines(usage, cap, np.stack([_job(10)]), np.array([2]), 2, np.random.default_rng(s))) for s in range(20)}
    assert outs == {(0, 1)}


def test_not_more_machines_than_k_returns_all():
    usage, cap = _state([0, 0])
    assert select_machines(usage, cap, np.stack([_job(10)]), np.array([PENDING]), 2, np.random.default_rng(0)) == [0, 1]
