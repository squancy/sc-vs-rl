"""Parameter estimation from one observed history (Protocol 2: the SC side must estimate the
model before it can use the analytical solution).

  * GBM:             Gaussian MLE of log returns.
  * Kim-Omberg full: exact conditional Gaussian MLE of (L_k, X_{k+1}) | X_k, i.e. a predictive
                     regression with an observed predictor (subject to Stambaugh 1999 bias).
  * Kim-Omberg partial: exact Kalman-filter likelihood of returns only (latent Sharpe ratio).
  * Heston / Chacko-Viceira: method of moments on realized variance (RV) with an
                     errors-in-variables-robust AR(1) estimate (ratio of lag-2 to lag-1
                     autocovariances); deliberately simple and transparent - replace with QMLE or
                     a particle filter for a more efficient SC estimator.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .models import GBM, ChackoViceira, Heston, KimOmberg
from .sc import KOKalman


def fit_gbm(returns: np.ndarray, dt: float, r: float) -> GBM:
    lr = np.log1p(np.asarray(returns, float))
    s2 = lr.var(ddof=1) / dt
    mu = lr.mean() / dt + 0.5 * s2
    return GBM(r=r, mu=mu, sigma=math.sqrt(s2))


# ----------------------------------------------------------------------------
# Kim-Omberg
# ----------------------------------------------------------------------------
def _ko_from_theta(th, r) -> KimOmberg:
    sig, lam, xb, sx, rho = math.exp(th[0]), math.exp(th[1]), th[2], math.exp(th[3]), math.tanh(th[4])
    return KimOmberg(r=r, sigma=sig, lam=lam, Xbar=xb, sigma_X=sx, rho=rho)


def _ko_theta(m: KimOmberg):
    return np.array([math.log(m.sigma), math.log(m.lam), m.Xbar, math.log(m.sigma_X),
                     math.atanh(max(min(m.rho, 0.999), -0.999))])


_KO_BOUNDS = [(math.log(0.02), math.log(1.5)), (math.log(0.005), math.log(10.0)), (-2.0, 3.0),
              (math.log(1e-3), math.log(2.0)), (-3.0, 3.0)]


def fit_ko_full(returns: np.ndarray, X: np.ndarray, dt: float, r: float) -> KimOmberg:
    """Exact MLE given the observed predictor path X_0..X_n."""
    L = np.log1p(np.asarray(returns, float))
    X = np.asarray(X, float)
    x0, x1 = X[:-1], X[1:]
    # starting values from OLS
    b_ols = np.polyfit(x0, x1, 1)[0]
    b_ols = min(max(b_ols, 1e-3), 0.9999)
    lam0 = -math.log(b_ols) / dt
    sig0 = L.std() / math.sqrt(dt)
    ex = x1 - np.polyval(np.polyfit(x0, x1, 1), x0)
    sx0 = ex.std() * math.sqrt(2 * lam0 / (1 - b_ols ** 2))
    eL = L - np.polyval(np.polyfit(x0, L, 1), x0)
    rho0 = float(np.corrcoef(eL, ex)[0, 1])
    init = KimOmberg(r=r, sigma=sig0, lam=lam0, Xbar=float(X.mean()), sigma_X=max(sx0, 1e-3), rho=rho0)

    def nll(th):
        m = _ko_from_theta(th, r)
        q = m.transition_moments(dt)
        det = q["VL"] * q["VX"] - q["CLX"] ** 2
        if det <= 0:
            return 1e12
        e1 = L - q["c0"] - q["c1"] * x0
        e2 = x1 - q["a"] - q["b"] * x0
        quad = (q["VX"] * e1 ** 2 - 2 * q["CLX"] * e1 * e2 + q["VL"] * e2 ** 2) / det
        return 0.5 * np.sum(quad + math.log(det) + 2 * math.log(2 * math.pi))
    res = minimize(nll, _ko_theta(init), method="L-BFGS-B", bounds=_KO_BOUNDS)
    return _ko_from_theta(res.x, r)


SKEPTICAL_PRIOR = {"sd_X": 0.15, "lam": 0.3, "log_lam_sd": 1.0}


def ko_log_prior(m: KimOmberg, prior: dict) -> float:
    """'Skeptical' log prior on the predictability parameters, in the spirit of Wachter &
    Warusawitharana (2009): half-normal on the stationary s.d. of the Sharpe ratio,
    sd_X = sigma_X / sqrt(2 lam) (shrinks towards no predictability), and log-normal on the
    mean-reversion speed (penalizes the spurious very fast mean reversion that the Kalman MLE
    tends to find when rho -> -1, which turns noise in short-horizon autocorrelations into
    large, rapidly changing positions)."""
    sdx = m.sigma_X / math.sqrt(2.0 * m.lam)
    return -0.5 * (sdx / prior["sd_X"]) ** 2 - 0.5 * ((math.log(m.lam) - math.log(prior["lam"])) / prior["log_lam_sd"]) ** 2


def fit_ko_partial(returns: np.ndarray, dt: float, r: float, n_starts: int = 4, seed: int = 0,
                   prior: Optional[dict] = None) -> KimOmberg:
    """Kalman-filter MLE from returns only (prior for X_0: stationary law of the candidate model).
    With ``prior`` (e.g. SKEPTICAL_PRIOR) the posterior mode (MAP) is returned instead."""
    L = np.log1p(np.asarray(returns, float))
    rng = np.random.default_rng(seed)
    sig0 = L.std() / math.sqrt(dt)
    sr0 = (L.mean() / dt + 0.5 * sig0 ** 2 - r) / sig0

    def nll(th):
        m = _ko_from_theta(th, r)
        try:
            val = -KOKalman(m, dt).log_likelihood(L)
            if prior is not None:
                val -= ko_log_prior(m, prior)
            return val
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1e12
    best = None
    for k in range(n_starts):
        init = KimOmberg(r=r, sigma=sig0, lam=float(rng.uniform(0.1, 1.0)), Xbar=sr0,
                         sigma_X=float(rng.uniform(0.05, 0.3)), rho=float(rng.uniform(-0.9, 0.0)))
        res = minimize(nll, _ko_theta(init), method="L-BFGS-B", bounds=_KO_BOUNDS,
                       options={"maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    return _ko_from_theta(best.x, r)


# ----------------------------------------------------------------------------
# Stochastic volatility from realized variance
# ----------------------------------------------------------------------------
def _ar1_eiv(x: np.ndarray):
    """AR(1) persistence robust to i.i.d. measurement error: b = acov(2) / acov(1).
    Returns (b, mean, noise-free variance)."""
    xc = x - x.mean()
    g1 = np.mean(xc[1:] * xc[:-1])
    g2 = np.mean(xc[2:] * xc[:-2])
    b = g2 / g1 if g1 > 0 else 0.5
    b = min(max(b, 0.05), 0.999)
    var = max(g1 / b, 1e-12)
    return b, float(x.mean()), var


def fit_heston_rv(returns: np.ndarray, rv: np.ndarray, dt: float, r: float, n_sub: int = 8) -> Heston:
    e = np.asarray(returns, float) - (math.exp(r * dt) - 1)
    v = np.asarray(rv, float) / dt
    b, vbar, varv = _ar1_eiv(v)
    kappa = -math.log(b) / dt
    sigma_v = math.sqrt(max(2 * kappa * varv / vbar, 1e-6))
    xi = e.sum() / (v.sum() * dt)
    innov = v[1:] - vbar - b * (v[:-1] - vbar)
    zs = e[:-1] / np.sqrt(np.maximum(v[:-1], 1e-8) * dt)
    rho = float(np.clip(np.corrcoef(zs, innov)[0, 1], -0.99, 0.99))
    return Heston(r=r, xi=xi, kappa=kappa, vbar=vbar, sigma_v=sigma_v, rho=rho, n_sub=n_sub)


def fit_cv_rv(returns: np.ndarray, rv: np.ndarray, dt: float, r: float, n_sub: int = 21) -> ChackoViceira:
    e = np.asarray(returns, float) - (math.exp(r * dt) - 1)
    y = dt / np.maximum(np.asarray(rv, float), 1e-12)
    b, theta, vary = _ar1_eiv(y)
    kappa = -math.log(b) / dt
    sigma_y = math.sqrt(max(2 * kappa * vary / theta, 1e-6))
    mu_e = e.mean() / dt
    innov = y[1:] - theta - b * (y[:-1] - theta)
    zs = e[:-1] * np.sqrt(y[:-1] / dt)
    rho = float(np.clip(np.corrcoef(zs, innov)[0, 1], -0.99, 0.99))
    return ChackoViceira(r=r, mu_e=mu_e, kappa=kappa, theta=theta, sigma_y=sigma_y, rho=rho, n_sub=n_sub)
