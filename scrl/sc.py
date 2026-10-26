"""Stochastic-control (SC) solutions.

Analytical continuous-time optimal policies (applied at discrete decision dates):
  * Merton (1969, 1971):        pi* = (mu - r) / (gamma sigma^2)
  * Kraft (2005) / Liu (2007):   Heston with premium xi*v:  pi*(tau) = xi/gamma + rho sigma_v B(tau)/gamma
  * Chacko & Viceira (2005):     CIR precision y = 1/v, constant premium mu_e:
                                 pi*(tau, y) = y (mu_e + rho sigma_y B(tau)) / gamma
  * Kim & Omberg (1996):         OU Sharpe ratio, full information: pi*(tau, x)
  * Partial information KO:      Kalman filter + KO solution in the filtered state
                                 (Lakner 1998; Brendle 2006; Bjork, Davis & Landen 2010)
  * Merton jump-diffusion:       constant pi* solving the HJB first-order condition
                                 (Aase 1984; Liu, Longstaff & Pan 2003)
  * Proportional costs:          no-trade band half-width from Janecek & Shreve (2004)

Exact *discrete-time* benchmarks for the simulated environments (dynamic programming):
  * i.i.d. models (GBM, jumps):  myopia is optimal for CRRA (Samuelson 1969), so the
                                 discrete optimum is the one-period optimum.
  * Kim-Omberg, full and partial information: backward induction on a 1-D state grid
    with Gauss-Hermite quadrature over the exact Gaussian transition.

All value functions use CRRA homogeneity: V_t(w, s) = w^(1-gamma)/(1-gamma) * f_t(s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .models import GBM, ChackoViceira, Heston, KimOmberg, MarketModel, MertonJump


# ----------------------------------------------------------------------------
# Merton
# ----------------------------------------------------------------------------
def merton_fraction(mu_excess: float, var: float, gamma: float) -> float:
    return mu_excess / (gamma * var)


# ----------------------------------------------------------------------------
# Kraft (2005) Heston solution
# ----------------------------------------------------------------------------
def affine_riccati_coefficients(prem: float, kappa: float, sig: float, rho: float, gamma: float
                                ) -> Tuple[float, float, float]:
    """B' = a B^2 + b B + c, B(0) = 0, shared by both affine SV models (Liu 2007):
        a = sig^2 (1 + (1-g) rho^2 / g) / 2,  b = -kappa + (1-g) prem rho sig / g,  c = (1-g) prem^2 / (2 g).
    Heston/Kraft: prem = xi, sig = sigma_v.  Chacko-Viceira: prem = mu_e, sig = sigma_y."""
    g = gamma
    a = 0.5 * sig ** 2 * (1.0 + (1.0 - g) * rho ** 2 / g)
    b = -kappa + (1.0 - g) * prem * rho * sig / g
    c = (1.0 - g) * prem ** 2 / (2.0 * g)
    return a, b, c


def affine_riccati_B(tau: np.ndarray, prem: float, kappa: float, sig: float, rho: float,
                     gamma: float) -> np.ndarray:
    """Explicit solution (Kraft 2005). For gamma > 1: a > 0 and c < 0, hence D = b^2 - 4ac > 0
    and the solution exists for all tau. Other cases use a numerical ODE solve."""
    tau = np.asarray(tau, float)
    a, b, c = affine_riccati_coefficients(prem, kappa, sig, rho, gamma)
    D = b * b - 4 * a * c
    if D > 0:
        sD = math.sqrt(D)
        em = np.exp(-sD * tau)
        den = (sD - b) * (1 - em) + 2 * sD * em
        if np.all(den > 0):
            return 2 * c * (1 - em) / den
    tmax = float(np.max(tau))
    sol = solve_ivp(lambda t, y: a * y * y + b * y + c, (0.0, tmax), [0.0], dense_output=True,
                    rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError("Riccati equation blows up on the horizon (no finite solution)")
    return sol.sol(tau)[0]


def kraft_coefficients(m: Heston, gamma: float) -> Tuple[float, float, float]:
    return affine_riccati_coefficients(m.xi, m.kappa, m.sigma_v, m.rho, gamma)


def kraft_B(tau: np.ndarray, m: Heston, gamma: float) -> np.ndarray:
    return affine_riccati_B(tau, m.xi, m.kappa, m.sigma_v, m.rho, gamma)


def kraft_policy(tau: np.ndarray, m: Heston, gamma: float) -> np.ndarray:
    return m.xi / gamma + m.rho * m.sigma_v * kraft_B(tau, m, gamma) / gamma


def cv_B(tau: np.ndarray, m: ChackoViceira, gamma: float) -> np.ndarray:
    return affine_riccati_B(tau, m.mu_e, m.kappa, m.sigma_y, m.rho, gamma)


def cv_policy(tau, y, m: ChackoViceira, gamma: float, hedge: bool = True):
    """pi*(tau, y) = y (mu_e + rho sigma_y B(tau)) / gamma."""
    hb = m.rho * m.sigma_y * cv_B(tau, m, gamma) if hedge else 0.0
    return y * (m.mu_e + hb) / gamma


def kraft_log_f(tau: float, v: float, m: Heston, gamma: float) -> float:
    """log f(tau, v) = A(tau) + B(tau) v with A' = (1-gamma) r + kappa vbar B (value function)."""
    taus = np.linspace(0.0, tau, 2001)
    Bv = kraft_B(taus, m, gamma)
    A = (1 - gamma) * m.r * tau + m.kappa * m.vbar * np.trapezoid(Bv, taus)
    return float(A + Bv[-1] * v)


# ----------------------------------------------------------------------------
# Kim & Omberg (1996) solution, with time-varying state loadings
# ----------------------------------------------------------------------------
@dataclass
class KOSolution:
    """Value V = w^(1-g)/(1-g) exp(A + B x + C x^2 / 2) and policy
        pi*(tau, x) = (x + b(tau) (B(tau) + C(tau) x)) / (gamma sigma),
    where a(tau) is the instantaneous variance of the state and b(tau) its covariance with
    the standardized return shock. Full information: a = sigma_X^2, b = rho sigma_X.
    Partial information (state = filtered Sharpe ratio m): a = s_m^2, b = s_m with
    s_m(t) = P(t) + rho sigma_X the Kalman gain."""
    taus: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    b_load: np.ndarray
    gamma: float
    sigma: float

    def coef(self, tau):
        tau = np.asarray(tau, float)
        return (np.interp(tau, self.taus, self.B), np.interp(tau, self.taus, self.C),
                np.interp(tau, self.taus, self.b_load), np.interp(tau, self.taus, self.A))

    def policy(self, tau, x):
        B, C, bl, _ = self.coef(tau)
        return (x + bl * (B + C * x)) / (self.gamma * self.sigma)

    def log_f(self, tau, x):
        B, C, _, A = self.coef(tau)
        return A + B * x + 0.5 * C * x * x


def ko_riccati_rhs(A: float, B: float, C: float, a: float, b: float, lam: float, Xbar: float,
                   r: float, g: float):
    """d(A, B, C)/dtau for the Kim-Omberg value function exp(A + B x + C x^2/2)."""
    k2 = a - b * b * (g - 1) / g
    dC = k2 * C * C + 2 * (b * (1 - g) / g - lam) * C + (1 - g) / g
    dB = k2 * B * C + (b * (1 - g) / g - lam) * B + lam * Xbar * C
    dA = 0.5 * k2 * B * B + lam * Xbar * B + 0.5 * a * C + (1 - g) * r
    return [dA, dB, dC]


def solve_kim_omberg(T: float, r: float, sigma: float, lam: float, Xbar: float, gamma: float,
                     a_fn: Callable[[float], float], b_fn: Callable[[float], float],
                     n_grid: int = 2001) -> KOSolution:
    """Integrate the Riccati system in tau = T - t from A = B = C = 0 (derived from the HJB
    equation; verified symbolically in tests/test_sc.py):
        C' = k2 C^2 + 2 (b (1-g)/g - lam) C + (1-g)/g
        B' = k2 B C + (b (1-g)/g - lam) B + lam Xbar C
        A' = k2 B^2 / 2 + lam Xbar B + a C / 2 + (1-g) r,      k2 = a - b^2 (g-1)/g.
    """
    def rhs(tau, y):
        return ko_riccati_rhs(y[0], y[1], y[2], a_fn(tau), b_fn(tau), lam, Xbar, r, gamma)

    taus = np.linspace(0.0, T, n_grid)
    sol = solve_ivp(rhs, (0.0, T), [0.0, 0.0, 0.0], t_eval=taus, rtol=1e-10, atol=1e-12)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Kim-Omberg Riccati system blows up on the horizon")
    bl = np.array([b_fn(t) for t in taus])
    return KOSolution(taus, sol.y[0], sol.y[1], sol.y[2], bl, gamma, sigma)


def ko_full_solution(m: KimOmberg, gamma: float, T: float) -> KOSolution:
    return solve_kim_omberg(T, m.r, m.sigma, m.lam, m.Xbar, gamma,
                            lambda tau: m.sigma_X ** 2, lambda tau: m.rho * m.sigma_X)


def ko_kalman_bucy_P(m: KimOmberg, P0: float, T: float, n_grid: int = 2001) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time filter variance: dP/dt = -2 lam P + sigma_X^2 - (P + rho sigma_X)^2."""
    ts = np.linspace(0.0, T, n_grid)
    sol = solve_ivp(lambda t, P: -2 * m.lam * P + m.sigma_X ** 2 - (P + m.rho * m.sigma_X) ** 2,
                    (0.0, T), [P0], t_eval=ts, rtol=1e-10, atol=1e-12)
    return ts, sol.y[0]


def ko_steady_state_P(m: KimOmberg) -> float:
    c = m.lam + m.rho * m.sigma_X
    return -c + math.sqrt(c * c + m.sigma_X ** 2 * (1 - m.rho ** 2))


def ko_partial_solution(m: KimOmberg, gamma: float, T: float, P0: float) -> KOSolution:
    """Partial information: the filtered Sharpe ratio m_t solves
        dm = lam (Xbar - m) dt + s_m(t) dW_hat,   s_m = P(t) + rho sigma_X,
    driven by the same innovation W_hat as returns, i.e. a KO model with correlation one."""
    ts, P = ko_kalman_bucy_P(m, P0, T)
    s_m = P + m.rho * m.sigma_X

    def sm_tau(tau):  # calendar time t = T - tau
        return float(np.interp(T - tau, ts, s_m))
    return solve_kim_omberg(T, m.r, m.sigma, m.lam, m.Xbar, gamma,
                            lambda tau: sm_tau(tau) ** 2, sm_tau)


# ----------------------------------------------------------------------------
# Discrete-time exact Kalman filter for the Kim-Omberg environment
# ----------------------------------------------------------------------------
class KOKalman:
    """Exact filter for the simulated KO environment, observing period log returns.
    State X_k, observation L_k = c0 + c1 X_k + e_L, X_{k+1} = a + b X_k + e_X with correlated
    (e_L, e_X). After observing L_0..L_{k-1}: X_k | F_k ~ N(m_k, P_k), P_k deterministic."""

    def __init__(self, m: KimOmberg, dt: float, m0: Optional[float] = None, P0: Optional[float] = None):
        self.model, self.dt = m, dt
        self.tm = m.transition_moments(dt)
        self.m0 = m.Xbar if m0 is None else m0
        self.P0 = m.stat_var if P0 is None else P0

    def P_sequence(self, n: int) -> np.ndarray:
        q = self.tm
        P = np.empty(n + 1)
        P[0] = self.P0
        for k in range(n):
            S = q["c1"] ** 2 * P[k] + q["VL"]
            Cv = q["c1"] * q["b"] * P[k] + q["CLX"]
            P[k + 1] = q["b"] ** 2 * P[k] + q["VX"] - Cv * Cv / S
        return P

    def gains(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kalman gain K_k, innovation variance S_k for k = 0..n-1, and P_0..P_n."""
        q = self.tm
        P = self.P_sequence(n)
        S = q["c1"] ** 2 * P[:-1] + q["VL"]
        K = (q["c1"] * q["b"] * P[:-1] + q["CLX"]) / S
        return K, S, P

    def filter(self, log_returns):
        """log_returns: torch or numpy [N, K]  ->  m [N, K+1] (same type)."""
        import torch
        q = self.tm
        K, _, _ = self.gains(log_returns.shape[1])
        is_t = isinstance(log_returns, torch.Tensor)
        N, n = log_returns.shape
        if is_t:
            out = torch.empty(N, n + 1, dtype=log_returns.dtype, device=log_returns.device)
        else:
            out = np.empty((N, n + 1))
        mk = (torch.full((N,), self.m0, dtype=log_returns.dtype, device=log_returns.device)
              if is_t else np.full(N, self.m0))
        out[:, 0] = mk
        for k in range(n):
            innov = log_returns[:, k] - q["c0"] - q["c1"] * mk
            mk = q["a"] + q["b"] * mk + K[k] * innov
            out[:, k + 1] = mk
        return out

    def log_likelihood(self, log_returns_1d: np.ndarray) -> float:
        q = self.tm
        n = len(log_returns_1d)
        K, S, _ = self.gains(n)
        mk = self.m0
        ll = 0.0
        for k in range(n):
            innov = log_returns_1d[k] - q["c0"] - q["c1"] * mk
            ll += -0.5 * (math.log(2 * math.pi * S[k]) + innov * innov / S[k])
            mk = q["a"] + q["b"] * mk + K[k] * innov
        return ll


# ----------------------------------------------------------------------------
# Merton jump-diffusion
# ----------------------------------------------------------------------------
def _gh(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights for the standard normal: E[f(Z)] ~ sum w f(z)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return x * math.sqrt(2.0), w / math.sqrt(math.pi)


def jump_foc(pi: float, m: MertonJump, gamma: float, n_quad: int = 64) -> float:
    z, w = _gh(n_quad)
    J1 = np.exp(m.muJ + m.sigJ * z) - 1.0
    integ = np.sum(w * J1 * ((1.0 + pi * J1) ** (-gamma) - 1.0))
    return m.mu - m.r - gamma * pi * m.sigma ** 2 + m.lam * integ


def jump_optimal_fraction(m: MertonJump, gamma: float) -> float:
    """Root of  mu - r - gamma pi sigma^2 + lam E[(J-1)((1 + pi (J-1))^(-gamma) - 1)] = 0.
    With lognormal jumps J - 1 has support (-1, inf), so admissible fractions lie in [0, 1]."""
    lo, hi = 0.0, 1.0 - 1e-9
    flo, fhi = jump_foc(lo, m, gamma), jump_foc(hi, m, gamma)
    if flo <= 0:
        return 0.0
    if fhi >= 0:
        return hi
    return brentq(lambda p: jump_foc(p, m, gamma), lo, hi, xtol=1e-12)


# ----------------------------------------------------------------------------
# Proportional transaction costs
# ----------------------------------------------------------------------------
def js_band_halfwidth(pi_star, gamma: float, cost: float):
    """No-trade band pi* +/- (3/(2 gamma) pi*^2 (1-pi*)^2 eps)^(1/3) (Janecek & Shreve 2004;
    Gerhold, Guasoni, Muhle-Karbe & Schachermayer 2014). First-order in eps^(1/3)."""
    return (1.5 / gamma * pi_star ** 2 * (1 - pi_star) ** 2 * cost) ** (1.0 / 3.0)


# ----------------------------------------------------------------------------
# Dynamic programming (exact discrete-time benchmarks)
# ----------------------------------------------------------------------------
def _golden_max(obj: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, shape, iters: int = 40):
    """Vectorized golden-section search maximizing obj elementwise on [lo, hi]."""
    gr = (math.sqrt(5) - 1) / 2
    a = np.full(shape, lo, float)
    b = np.full(shape, hi, float)
    for _ in range(iters):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        left = obj(c) > obj(d)          # maximum lies in [a, d]
        b = np.where(left, d, b)
        a = np.where(left, a, c)
    x = 0.5 * (a + b)
    return x, obj(x)


def _crra_obj(Rp: np.ndarray, fnext: np.ndarray, weights: np.ndarray, gamma: float) -> np.ndarray:
    """E[Rp^(1-g) f'] / (1-g) along the last axis (to be maximized). Bankrupt nodes -> -inf."""
    if abs(gamma - 1.0) < 1e-12:
        raise NotImplementedError("DP implemented for gamma != 1")
    base = np.maximum(Rp, 1e-12)
    val = np.sum(weights * base ** (1 - gamma) * fnext, axis=-1) / (1 - gamma)
    bad = np.any(Rp <= 0, axis=-1)
    return np.where(bad, -np.inf, val)


def one_period_nodes(model: MarketModel, dt: float, n_quad: int = 64, max_jumps: int = 8
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature for the one-period simple return of an i.i.d. model."""
    z, w = _gh(n_quad)
    if isinstance(model, GBM):
        R = np.expm1((model.mu - 0.5 * model.sigma ** 2) * dt + model.sigma * math.sqrt(dt) * z)
        return R, w
    if isinstance(model, MertonJump):
        Rs, ws = [], []
        lt = model.lam * dt
        for n in range(max_jumps + 1):
            pn = math.exp(-lt) * lt ** n / math.factorial(n)
            mean = (model.mu - model.lam * model.k - 0.5 * model.sigma ** 2) * dt + n * model.muJ
            sd = math.sqrt(model.sigma ** 2 * dt + n * model.sigJ ** 2)
            Rs.append(np.expm1(mean + sd * z))
            ws.append(pn * w)
        W = np.concatenate(ws)
        return np.concatenate(Rs), W / W.sum()
    raise TypeError(f"no i.i.d. quadrature for {type(model).__name__}")


def dp_iid_fraction(model: MarketModel, gamma: float, dt: float, lo: float, hi: float) -> float:
    """Exact discrete-time optimum for an i.i.d. model (myopic, Samuelson 1969)."""
    R, w = one_period_nodes(model, dt)
    rf = model.rf_step(dt)

    def obj(p):
        p = np.atleast_1d(p)
        Rp = 1 + rf + p[:, None] * (R[None, :] - rf)
        return _crra_obj(Rp, np.ones_like(Rp), w[None, :], gamma)
    x, _ = _golden_max(obj, lo, hi, (1,))
    return float(x[0])


@dataclass
class DPTable:
    """pi[t, i] and log f[t, i] on a state grid; t = 0..T-1 are episode decision times."""
    grid: np.ndarray
    pi: np.ndarray
    log_f: np.ndarray

    def policy(self, t: int, s):
        return np.interp(s, self.grid, self.pi[t])


def dp_ko_full(m: KimOmberg, gamma: float, dt: float, steps: int, lo: float, hi: float,
               n_grid: int = 241, n_quad: int = 16, width: float = 6.0) -> DPTable:
    """Backward induction for the KO environment with X observed (exact 2-D Gaussian transition)."""
    q = m.transition_moments(dt)
    rf = m.rf_step(dt)
    sd = math.sqrt(m.stat_var)
    grid = np.linspace(m.Xbar - width * sd, m.Xbar + width * sd, n_grid)
    z, w = _gh(n_quad)
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    W = np.outer(w, w).ravel()
    chol = np.linalg.cholesky(np.array([[q["VL"], q["CLX"]], [q["CLX"], q["VX"]]]))
    E = chol @ np.vstack([Z1.ravel(), Z2.ravel()])          # [2, Q]
    L = q["c0"] + q["c1"] * grid[:, None] + E[0][None, :]    # [S, Q]
    Xn = q["a"] + q["b"] * grid[:, None] + E[1][None, :]
    R = np.expm1(L)
    log_f_next = np.zeros(n_grid)
    pis = np.empty((steps, n_grid))
    logfs = np.empty((steps + 1, n_grid))
    logfs[steps] = 0.0
    for t in range(steps - 1, -1, -1):
        lfn = np.interp(Xn, grid, log_f_next)
        shift = lfn.max()
        fn = np.exp(lfn - shift)

        def obj(p):
            Rp = 1 + rf + p[:, None] * (R - rf)
            return _crra_obj(Rp, fn, W[None, :], gamma)
        p, val = _golden_max(obj, lo, hi, (n_grid,))
        pis[t] = p
        log_f_next = np.log(val * (1 - gamma)) + shift   # f_t = E[Rp^(1-g) f_{t+1}]
        logfs[t] = log_f_next
    return DPTable(grid, pis, logfs)


def dp_ko_partial(m: KimOmberg, gamma: float, dt: float, burn: int, steps: int, lo: float, hi: float,
                  n_grid: int = 241, n_quad: int = 32, width: float = 6.0) -> DPTable:
    """Backward induction in the filtered state m_k (known parameters, returns observed).
    Predictive law: L | m ~ N(c0 + c1 m, S_k);  m' = a + b m + K_k (L - c0 - c1 m)."""
    kf = KOKalman(m, dt)
    K, S, P = kf.gains(burn + steps)
    q = kf.tm
    rf = m.rf_step(dt)
    sd = math.sqrt(m.stat_var)
    grid = np.linspace(m.Xbar - width * sd, m.Xbar + width * sd, n_grid)
    z, w = _gh(n_quad)
    log_f_next = np.zeros(n_grid)
    pis = np.empty((steps, n_grid))
    logfs = np.empty((steps + 1, n_grid))
    logfs[steps] = 0.0
    for t in range(steps - 1, -1, -1):
        k = burn + t
        innov = math.sqrt(S[k]) * z                            # [Q]
        L = q["c0"] + q["c1"] * grid[:, None] + innov[None, :]  # [S, Q]
        Mn = q["a"] + q["b"] * grid[:, None] + K[k] * innov[None, :]
        R = np.expm1(L)
        lfn = np.interp(Mn, grid, log_f_next)
        shift = lfn.max()
        fn = np.exp(lfn - shift)

        def obj(p):
            Rp = 1 + rf + p[:, None] * (R - rf)
            return _crra_obj(Rp, fn, w[None, :], gamma)
        p, val = _golden_max(obj, lo, hi, (n_grid,))
        pis[t] = p
        log_f_next = np.log(val * (1 - gamma)) + shift
        logfs[t] = log_f_next
    return DPTable(grid, pis, logfs)
