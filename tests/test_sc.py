"""Correctness tests for the stochastic-control side (formulas, discretization, filters, DP)."""
import math

import numpy as np
import pytest
import sympy as sp
import torch
from scipy.integrate import solve_ivp

from scrl import sc
from scrl.configs import jump_matched, model_zoo
from scrl.env import EnvConfig, rollout, wealth_step
from scrl.models import GBM, ChackoViceira, FeatureSpec, Heston, KimOmberg, SimSampler
from scrl.policies import KOStatePolicy
from scrl.rl.common import normalized_reward
from scrl.utils import ce_stats


# ---------------------------------------------------------------------------
# HJB derivations (symbolic) vs. the code
# ---------------------------------------------------------------------------
def test_kim_omberg_hjb_matches_code():
    w, x, r, sig, lam, Xb, sx, rho, g = sp.symbols("w x r sigma lambda Xbar sigma_X rho gamma", real=True)
    A, B, C, Ad, Bd, Cd, pi = sp.symbols("A B C Ad Bd Cd pi")
    V = w ** (1 - g) / (1 - g) * sp.exp(A + B * x + C * x ** 2 / 2)
    Vt = -V * (Ad + Bd * x + Cd * x ** 2 / 2)                      # tau = T - t
    H = (Vt + w * (r + pi * sig * x) * sp.diff(V, w) + sp.Rational(1, 2) * pi ** 2 * sig ** 2 * w ** 2 * sp.diff(V, w, 2)
         + pi * sig * w * rho * sx * sp.diff(V, w, x) + lam * (Xb - x) * sp.diff(V, x)
         + sp.Rational(1, 2) * sx ** 2 * sp.diff(V, x, 2))
    pis = sp.solve(sp.diff(H, pi), pi)[0]
    res = sp.expand(sp.simplify(H.subs(pi, pis) / V))
    co = sp.Poly(res, x).all_coeffs()[::-1]
    sol = {Ad: sp.solve(co[0], Ad)[0], Bd: sp.solve(co[1], Bd)[0], Cd: sp.solve(co[2], Cd)[0]}
    rng = np.random.default_rng(0)
    for _ in range(5):
        vals = dict(r=0.02, sigma=0.2, lam=rng.uniform(0.1, 1), Xbar=rng.uniform(0, .5), sx=rng.uniform(.05, .3),
                    rho=rng.uniform(-.95, .95), g=rng.uniform(1.5, 8), A=rng.normal(), B=rng.normal(), C=rng.normal())
        subs = {r: vals["r"], sig: vals["sigma"], lam: vals["lam"], Xb: vals["Xbar"], sx: vals["sx"], rho: vals["rho"],
                g: vals["g"], A: vals["A"], B: vals["B"], C: vals["C"]}
        want = [float(sol[k].subs(subs)) for k in (Ad, Bd, Cd)]
        got = sc.ko_riccati_rhs(vals["A"], vals["B"], vals["C"], vals["sx"] ** 2, vals["rho"] * vals["sx"],
                                vals["lam"], vals["Xbar"], vals["r"], vals["g"])
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
        x0 = 0.37
        pi_sym = float(pis.subs(subs).subs(x, x0))
        ko = KOSol = sc.KOSolution(np.array([0.0, 1.0]), np.zeros(2), np.full(2, vals["B"]), np.full(2, vals["C"]),
                                   np.full(2, vals["rho"] * vals["sx"]), vals["g"], vals["sigma"])
        assert abs(ko.policy(0.5, x0) - pi_sym) < 1e-10


@pytest.mark.parametrize("model", ["heston", "cv"])
def test_affine_sv_hjb_matches_code(model):
    w, s, r, prem, kap, th, sv, rho, g = sp.symbols("w s r prem kappa theta sv rho gamma", positive=True)
    A, B, Ad, Bd, pi = sp.symbols("A B Ad Bd pi")
    V = w ** (1 - g) / (1 - g) * sp.exp(A + B * s)
    Vt = -V * (Ad + Bd * s)
    if model == "heston":   # dS/S = (r + prem v)dt + sqrt(v) dW, dv = kap(th - v)dt + sv sqrt(v) dW_v
        drift, var, cov = prem * s, s, rho * sv * s
    else:                   # dS/S = (r + prem)dt + sqrt(1/y) dW, dy = kap(th - y)dt + sv sqrt(y) dW_y
        drift, var, cov = prem, 1 / s, rho * sv
    H = (Vt + w * (r + pi * drift) * sp.diff(V, w) + sp.Rational(1, 2) * pi ** 2 * var * w ** 2 * sp.diff(V, w, 2)
         + pi * w * cov * sp.diff(V, w, s) + kap * (th - s) * sp.diff(V, s) + sp.Rational(1, 2) * sv ** 2 * s * sp.diff(V, s, 2))
    pis = sp.simplify(sp.solve(sp.diff(H, pi), pi)[0])
    a, b, c = sc.affine_riccati_coefficients(prem, kap, sv, rho, g)
    res = sp.simplify((H.subs(pi, pis) / V).subs({Bd: a * B ** 2 + b * B + c, Ad: (1 - g) * r + kap * th * B}))
    assert res == 0
    vals = {prem: 1.5 if model == "heston" else 0.06, kap: 2.0, sv: 0.4 if model == "heston" else 3.0,
            rho: -0.7 if model == "heston" else 0.5, g: 5.0, s: 0.05 if model == "heston" else 25.0}
    Bv = 0.123
    want = float(pis.subs(vals).subs(B, Bv))
    if model == "heston":
        got = (vals[prem] + vals[rho] * vals[sv] * Bv) / vals[g]
    else:
        got = vals[s] * (vals[prem] + vals[rho] * vals[sv] * Bv) / vals[g]
    assert abs(got - want) < 1e-12


def test_kraft_closed_form_matches_ode():
    m = Heston()
    for g in (2.0, 5.0, 10.0):
        a, b, c = sc.kraft_coefficients(m, g)
        taus = np.linspace(0, 10, 21)
        ode = solve_ivp(lambda t, y: a * y * y + b * y + c, (0, 10), [0.0], t_eval=taus, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(sc.kraft_B(taus, m, g), ode.y[0], rtol=1e-7, atol=1e-10)


# ---------------------------------------------------------------------------
# Simulation and filtering
# ---------------------------------------------------------------------------
def test_ko_exact_moments_vs_fine_euler():
    m = KimOmberg()
    h = 1 / 12
    q = m.transition_moments(h)
    rng = np.random.default_rng(1)
    N, M, X0 = 200_000, 100, 0.5
    dt = h / M
    X, L = np.full(N, X0), np.zeros(N)
    for _ in range(M):
        z1 = rng.standard_normal(N)
        z2 = m.rho * z1 + math.sqrt(1 - m.rho ** 2) * rng.standard_normal(N)
        L += (m.r + m.sigma * X - m.sigma ** 2 / 2) * dt + m.sigma * math.sqrt(dt) * z1
        X += m.lam * (m.Xbar - X) * dt + m.sigma_X * math.sqrt(dt) * z2
    se_L = math.sqrt(q["VL"] / N)
    assert abs(L.mean() - (q["c0"] + q["c1"] * X0)) < 4 * se_L
    assert abs(X.mean() - (q["a"] + q["b"] * X0)) < 4 * math.sqrt(q["VX"] / N)
    assert abs(L.var() / q["VL"] - 1) < 0.02
    assert abs(X.var() / q["VX"] - 1) < 0.02
    assert abs(np.cov(L, X)[0, 1] / q["CLX"] - 1) < 0.03


def test_kalman_filter_is_calibrated():
    m = KimOmberg()
    dt = 1 / 12
    r, lat, _ = m.simulate(20000, 120, dt, np.random.default_rng(2))
    kf = sc.KOKalman(m, dt)
    mk = kf.filter(np.log1p(r))
    P = kf.P_sequence(120)
    for k in (10, 60, 120):
        err = lat[:, k] - mk[:, k]
        assert abs(err.mean()) < 4 * math.sqrt(P[k] / len(err))
        assert abs(err.var() / P[k] - 1) < 0.05


def test_moment_matching_of_zoo():
    zoo = model_zoo()
    dt = 1 / 12
    for k in ("gbm", "jump", "heston", "cv", "ko"):
        r, _, _ = zoo[k].simulate(20000, 24, dt, np.random.default_rng(3))
        lr = np.log1p(r)
        assert abs(r.mean() / dt - 0.08) < 0.01, k
        assert abs(lr.var() / dt - 0.04) < 0.006, k


# ---------------------------------------------------------------------------
# Discrete-time optimality
# ---------------------------------------------------------------------------
def test_dp_matches_continuous_time_ko():
    m = KimOmberg()
    g, dt, steps = 5.0, 1 / 12, 60
    dp = sc.dp_ko_full(m, g, dt, steps, -1.0, 2.0, n_grid=161, n_quad=12)
    sol = sc.ko_full_solution(m, g, steps * dt)
    xs = np.array([0.0, 0.3, 0.6])
    for t in (0, 30, 59):
        np.testing.assert_allclose(dp.policy(t, xs), sol.policy((steps - t) * dt, xs), atol=0.01)


def test_iid_dp_matches_closed_forms():
    g, dt = 5.0, 1 / 12
    assert abs(sc.dp_iid_fraction(GBM(), g, dt, -1, 2) - 0.3) < 2e-3
    jm = jump_matched()
    assert abs(sc.dp_iid_fraction(jm, g, dt, -1, 2) - sc.jump_optimal_fraction(jm, g)) < 2e-3


def test_ko_analytic_value_matches_monte_carlo():
    """CE of the analytical KO policy by simulation (weekly rebalancing) vs the value function."""
    m = KimOmberg()
    g, dt, years = 5.0, 1 / 52, 5.0
    steps = int(years / dt)
    sol = sc.ko_full_solution(m, g, years)
    s = SimSampler(m, dt, 0, steps, FeatureSpec(info="full"), seed=4, dtype=torch.float64)
    b = s.sample(40000)
    pol = KOStatePolicy("ko", "analytic", "latent", g, m.sigma, sol=sol)
    w = rollout(b, pol, EnvConfig(gamma=g, lo=-5, hi=5))["wT"].numpy()
    mc_rate = ce_stats(w, g, years)["ce_rate"]
    z, wq = sc._gh(64)
    x0 = m.Xbar + math.sqrt(m.stat_var) * z
    logf = sol.log_f(years, x0)
    th_rate = math.log(np.sum(wq * np.exp(logf))) / (1 - g) / years
    assert abs(mc_rate - th_rate) * 1e4 < 6.0     # bp/yr (MC error + discrete rebalancing)


# ---------------------------------------------------------------------------
# Environment and RL helpers
# ---------------------------------------------------------------------------
def test_wealth_step_with_costs():
    w, d = torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64)
    pi, R = torch.tensor([0.3], dtype=torch.float64), torch.tensor([0.1], dtype=torch.float64)
    w1, d1 = wealth_step(w, d, pi, R, 0.01, 0.002)
    gross = 1 + 0.01 + 0.3 * (0.1 - 0.01)
    assert abs(float(w1) - gross * (1 - 0.002 * 0.2)) < 1e-12
    assert abs(float(d1) - 0.3 * 1.1 / gross) < 1e-12


def test_normalized_bellman_telescopes():
    g = 5.0
    G = torch.tensor([1.02, 0.97, 1.05, 0.9, 1.01], dtype=torch.float64)
    rt, beta = normalized_reward(G, g)
    acc, disc = 0.0, 1.0
    for k in range(len(G)):
        acc += disc * float(rt[k])
        disc *= float(beta[k])
    W = float(torch.prod(G))
    assert abs(acc - (W ** (1 - g) - 1) / (1 - g)) < 1e-12


def test_band_halfwidth_formula():
    assert abs(sc.js_band_halfwidth(0.3, 5.0, 0.005) - (1.5 / 5 * 0.09 * 0.49 * 0.005) ** (1 / 3)) < 1e-12
