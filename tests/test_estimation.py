"""Estimators and seeding."""
import math

import numpy as np

from scrl import estimation as est
from scrl.configs import ExpConfig
from scrl.models import GBM, simulate_history
from scrl.protocols import _seed


def test_fit_gbm_recovers_parameters():
    m = GBM(r=0.02, mu=0.08, sigma=0.2)
    r, _, _ = simulate_history(m, 12 * 2000, 1 / 12, seed=3)
    f = est.fit_gbm(r, 1 / 12, 0.02)
    assert abs(f.sigma - 0.2) < 0.005
    assert abs(f.mu - 0.08) < 0.02


def test_skeptical_prior_shrinks_predictability():
    """On i.i.d. returns the Kalman MLE tends to find spurious predictability; the MAP estimate
    with the skeptical prior must imply a smaller stationary s.d. of the Sharpe ratio."""
    m = GBM(r=0.02, mu=0.08, sigma=0.2)
    dt = 1 / 12
    sd = lambda k: k.sigma_X / math.sqrt(2 * k.lam)
    worse = 0
    for s in range(3):
        r, _, _ = simulate_history(m, 360, dt, seed=10 + s)
        mle = est.fit_ko_partial(r, dt, 0.02, n_starts=2, seed=0)
        mapf = est.fit_ko_partial(r, dt, 0.02, n_starts=2, seed=0, prior=est.SKEPTICAL_PRIOR)
        worse += sd(mapf) > sd(mle) + 1e-6
    assert worse == 0


def test_seeds_are_stable_and_label_based():
    cfg = ExpConfig(seed=0)
    a = _seed(cfg, "oracle", "ko_full", "test")
    assert a == _seed(cfg, "oracle", "ko_full", "test")
    assert a != _seed(cfg, "oracle", "ko_partial", "test")
    assert a != _seed(ExpConfig(seed=1), "oracle", "ko_full", "test")
    assert 0 <= a < 2 ** 31 - 1
