"""
A path consists of ``burn + T`` periods. ``returns[:, k]`` is the simple return
over period k (from time k to k+1) and ``latent[:, k]`` is the state at time k.
The investor trades only over the last T periods (the "episode"). The first
``burn`` periods are a pre-sample that every method may use to warm up filters
and rolling estimators, so that all methods share the same information set.
At decision time t of the episode (t = 0..T-1) the observed returns are
``returns[:, :burn + t]`` and (full information only) the state is
``latent[:, burn + t]``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Path container
# ----------------------------------------------------------------------------
@dataclass
class PathBatch:
    """
    A batch (of size N) paths (simple returns of size ``burn + T``).

    Attributes:
        returns (torch.Tensor): [N, burn+T] simple returns of the risky asset.
        latent (torch.Tensor | None): [N, burn+T+1] state at the start of each period.
        rf (float): Per-period simple risk-free return.
        dt (float): Period length in years.
        burn (int): Number of pre-sample periods.
        horizon (int): Number of trading periods T.
        feats (torch.Tensor | None): [N, T, F] exogenous features at decision times.
        model_name (str): Name of the SC or RL model.
        rv (torch.Tensor | None): [N, burn+T] realized variance of each period
            (used by stochastic volatility models).
    """
    returns: torch.Tensor                 
    latent: torch.Tensor | None
    rf: float                             
    dt: float                             
    burn: int                             
    horizon: int                          
    feats: torch.Tensor | None = None
    model_name: str = ""
    rv: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return self.returns.shape[0]

    @property
    def years(self) -> float:
        return self.horizon * self.dt

    def episode_returns(self) -> torch.Tensor:
        """
        Returns for each path.

        Returns:
            torch.Tensor: [N, T] returns.
        """
        return self.returns[:, self.burn:]

    def latent_at(self, t: int) -> torch.Tensor:
        """
        Returns the latent state at index t for each path.

        Args:
            t (int): Index.
        
        Returns:
            torch.Tensor: [N, 1] latent states.
    
        Raises:
            ValueError: if no latent state is defined.
        """
        if self.latent is None:
            raise ValueError(f"model '{self.model_name}' has no latent state (full information unavailable)")
        return self.latent[:, self.burn + t]

    def to(self, device=None, dtype=None) -> PathBatch:
        """
        Creates a new ``PathBatch`` based on the current object.

        Args:
            device: PyTorch device.
            dtype: PyTorch type.

        Returns:
            PathBatch: New ``PathBatch`` object with the same returns, latent states, features
                and realized variances as this object.
        """
        def cv(x):
            return None if x is None else x.to(device=device, dtype=dtype)
        return replace(self, returns=cv(self.returns), latent=cv(self.latent), feats=cv(self.feats),
                       rv=cv(self.rv))

    def subset(self, idx) -> PathBatch:
        """
        Creates a new ``PathBatch`` based on the current object, containing only the
        paths specified by ``idx``.

        Args:
            device: PyTorch device.
            dtype: PyTorch type.

        Returns:
            PathBatch: New ``PathBatch`` object the same returns, latent states, features
                and realized variances as this object (except for the excluded indicies). 
        """
        def sl(x):
            return None if x is None else x[idx]
        return replace(self, returns=sl(self.returns), latent=sl(self.latent), feats=sl(self.feats),
                       rv=sl(self.rv))


# ----------------------------------------------------------------------------
# Market models
# ----------------------------------------------------------------------------
class MarketModel:
    """
    Base class for a market model.

    Attributes:
        name (str): Name of the market model.
        latent_name (str | None): Name of the latent market model.
    """
    name: str = "base"
    latent_name: str | None = None

    def __init__(self, r: float) -> None:
        self.r = float(r)

    def rf_step(self, dt: float) -> float:
        """
        Total return over the horizon (e ^ (r * dt) - 1). 

        Args:
            dt (float): Period length.
        
        Returns:
            float: e ^ (r * dt) - 1.
        """
        return math.exp(self.r * dt) - 1.0

    def simulate(self, n: int, steps: int, dt: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray | None , np.ndarray | None]:
        """
        Base function for simulating episodes.

        Args:
            n (int): Number of trajectories to simulate.
            steps (int): Number of steps in each trajectory.
            dt (float): Period length.
            rng (np.random.Generator): Random number generator.

        Returns: tuple[np.ndarray, np.ndarray | None , np.ndarray | None]: simple
        returns [n, steps], latent [n, steps+1] or None, realized variance [n, steps] or None.
        """
        raise NotImplementedError

    def params(self) -> dict[str, float]:
        """
        Retrieves the set of float/integer-valued attributes on the instance.

        Returns: dict[str, float]: Attributes.
        """
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float))}

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v:.4g}" for k, v in self.params().items())
        return f"{self.__class__.__name__}({p})"


class GBM(MarketModel):
    """
    dS/S = mu dt + sigma dW (Merton 1969, 1971).
    """
    name = "gbm"

    def __init__(self, r: float = 0.02, mu: float = 0.08, sigma: float = 0.20) -> None:
        super().__init__(r)
        self.mu, self.sigma = float(mu), float(sigma)

    def simulate(self, n, steps, dt, rng):
        z = rng.standard_normal((n, steps))
        logr = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * z
        return np.expm1(logr), None, None


class Heston(MarketModel):
    """
    Heston (1993) stochastic volatility with a risk premium proportional to variance
    (the specification solved in closed form by Kraft 2005 and Liu 2007):

        dS/S = (r + xi v) dt + sqrt(v) dW_S
        dv   = kappa (vbar - v) dt + sigma_v sqrt(v) dW_v,   d<W_S, W_v> = rho dt.

    Simulated with the full-truncation Euler scheme of Lord, Koekkoek & van Dijk (2010)
    on ``n_sub`` sub-steps per decision period. The latent state is v at decision dates.
    v_0 is drawn from the stationary Gamma(2 kappa vbar / sigma_v^2, sigma_v^2 / (2 kappa)) law.
    """
    name = "heston"
    latent_name = "v"

    def __init__(self, r=0.02, xi=1.5, kappa=2.0, vbar=0.04, sigma_v=0.4, rho=-0.7, n_sub=8):
        super().__init__(r)
        self.xi, self.kappa, self.vbar = float(xi), float(kappa), float(vbar)
        self.sigma_v, self.rho, self.n_sub = float(sigma_v), float(rho), int(n_sub)

    def stationary_v(self, n, rng):
        shape = 2 * self.kappa * self.vbar / self.sigma_v ** 2
        scale = self.sigma_v ** 2 / (2 * self.kappa)
        return rng.gamma(shape, scale, size=n)

    def simulate(self, n, steps, dt, rng):
        h = dt / self.n_sub
        v = self.stationary_v(n, rng)
        rets = np.empty((n, steps))
        lat = np.empty((n, steps + 1))
        rv = np.empty((n, steps))
        sq = math.sqrt(1 - self.rho ** 2)
        for k in range(steps):
            lat[:, k] = np.maximum(v, 0.0)
            logs = np.zeros(n)
            rvk = np.zeros(n)
            for _ in range(self.n_sub):
                vp = np.maximum(v, 0.0)
                z1 = rng.standard_normal(n)
                z2 = self.rho * z1 + sq * rng.standard_normal(n)
                sv = np.sqrt(vp * h)
                inc = (self.r + self.xi * vp - 0.5 * vp) * h + sv * z1
                logs += inc
                rvk += inc * inc
                v = v + self.kappa * (self.vbar - vp) * h + self.sigma_v * sv * z2
            rets[:, k] = np.expm1(logs)
            rv[:, k] = rvk
        lat[:, steps] = np.maximum(v, 0.0)
        return rets, lat, rv


class ChackoViceira(MarketModel):
    """
    Stochastic volatility with a constant risk premium and CIR precision (Chacko & Viceira 2005):

        dS/S = (r + mu_e) dt + sqrt(1/y) dW_S
        dy   = kappa (theta - y) dt + sigma_y sqrt(y) dW_y,   d<W_S, W_y> = rho dt,

    y = 1/variance. With CRRA utility of terminal wealth the HJB equation is affine in y and the
    optimal policy is pi* = y (mu_e + rho sigma_y B(tau)) / gamma (volatility timing plus a small
    hedging term; derivation verified in tests/test_sc.py). Full-truncation Euler on ``n_sub``
    sub-steps; y is floored at ``y_floor`` when computing the variance so that it stays finite.
    The latent state is y at decision dates; ``rv`` is the realized variance of each period.
    """
    name = "chacko_viceira"
    latent_name = "y"

    def __init__(self, r=0.02, mu_e=0.06, kappa=1.0, theta=30.0, sigma_y=3.1623, rho=0.5,
                 n_sub=21, y_floor=1.0):
        super().__init__(r)
        self.mu_e, self.kappa, self.theta = float(mu_e), float(kappa), float(theta)
        self.sigma_y, self.rho, self.n_sub, self.y_floor = float(sigma_y), float(rho), int(n_sub), float(y_floor)

    def stationary_y(self, n, rng):
        shape = 2 * self.kappa * self.theta / self.sigma_y ** 2
        scale = self.sigma_y ** 2 / (2 * self.kappa)
        return rng.gamma(shape, scale, size=n)

    @property
    def mean_variance(self) -> float:
        """E[1/y] under the stationary Gamma law (finite when shape > 1)."""
        shape = 2 * self.kappa * self.theta / self.sigma_y ** 2
        scale = self.sigma_y ** 2 / (2 * self.kappa)
        return 1.0 / ((shape - 1) * scale)

    def simulate(self, n, steps, dt, rng):
        h = dt / self.n_sub
        y = self.stationary_y(n, rng)
        rets = np.empty((n, steps))
        lat = np.empty((n, steps + 1))
        rv = np.empty((n, steps))
        sq = math.sqrt(1 - self.rho ** 2)
        for k in range(steps):
            lat[:, k] = np.maximum(y, self.y_floor)
            logs = np.zeros(n)
            rvk = np.zeros(n)
            for _ in range(self.n_sub):
                yp = np.maximum(y, 0.0)
                var = 1.0 / np.maximum(y, self.y_floor)
                z1 = rng.standard_normal(n)
                z2 = self.rho * z1 + sq * rng.standard_normal(n)
                inc = (self.r + self.mu_e - 0.5 * var) * h + np.sqrt(var * h) * z1
                logs += inc
                rvk += inc * inc
                y = y + self.kappa * (self.theta - yp) * h + self.sigma_y * np.sqrt(yp * h) * z2
            rets[:, k] = np.expm1(logs)
            rv[:, k] = rvk
        lat[:, steps] = np.maximum(y, self.y_floor)
        return rets, lat, rv


class KimOmberg(MarketModel):
    """
    Mean-reverting market price of risk (Kim & Omberg 1996; Wachter 2002):

        dS/S = (r + sigma X) dt + sigma dW_S
        dX   = lam (Xbar - X) dt + sigma_X dW_X,   d<W_S, W_X> = rho dt.

    X is the (instantaneous) Sharpe ratio. The pair (log S, X) is a linear Gaussian
    SDE, so it is simulated *exactly* at the decision frequency: given X_k, the
    period log return L_k and X_{k+1} are bivariate normal (see transition_moments).
    X_0 is drawn from the stationary N(Xbar, sigma_X^2 / (2 lam)) law.
    """
    name = "kim_omberg"
    latent_name = "X"

    def __init__(self, r=0.02, sigma=0.2, lam=0.2, Xbar=0.3, sigma_X=0.1265, rho=-0.9):
        super().__init__(r)
        self.sigma, self.lam, self.Xbar = float(sigma), float(lam), float(Xbar)
        self.sigma_X, self.rho = float(sigma_X), float(rho)

    @property
    def stat_var(self) -> float:
        return self.sigma_X ** 2 / (2 * self.lam)

    def transition_moments(self, h: float) -> dict[str, float]:
        """Exact moments over a period of length h:
            L = c0 + c1 X + e_L,   X' = a + b X + e_X,   (e_L, e_X) ~ N(0, [[VL, CLX], [CLX, VX]]).
        Derived from Y = X - Xbar, I = int_0^h Y ds and Ito's isometry.
        """
        r, s, lam, xb, sx, rho = self.r, self.sigma, self.lam, self.Xbar, self.sigma_X, self.rho
        e1, e2 = math.exp(-lam * h), math.exp(-2 * lam * h)
        q1, q2 = (1 - e1) / lam, (1 - e2) / (2 * lam)
        var_y = sx ** 2 * q2
        var_i = sx ** 2 / lam ** 2 * (h - 2 * q1 + q2)
        cov_iy = sx ** 2 / lam * (q1 - q2)
        cov_wy = rho * sx * q1
        cov_wi = rho * sx / lam * (h - q1)
        return dict(
            c0=(r - 0.5 * s ** 2) * h + s * xb * (h - q1),
            c1=s * q1,
            a=xb * (1 - e1),
            b=e1,
            VL=s ** 2 * (var_i + h + 2 * cov_wi),
            VX=var_y,
            CLX=s * (cov_iy + cov_wy),
        )

    def simulate(self, n, steps, dt, rng):
        m = self.transition_moments(dt)
        cov = np.array([[m["VL"], m["CLX"]], [m["CLX"], m["VX"]]])
        chol = np.linalg.cholesky(cov)
        x = self.Xbar + math.sqrt(self.stat_var) * rng.standard_normal(n)
        rets = np.empty((n, steps))
        lat = np.empty((n, steps + 1))
        for k in range(steps):
            lat[:, k] = x
            z = rng.standard_normal((2, n))
            e = chol @ z
            L = m["c0"] + m["c1"] * x + e[0]
            x = m["a"] + m["b"] * x + e[1]
            rets[:, k] = np.expm1(L)
        lat[:, steps] = x
        return rets, lat, None


class MertonJump(MarketModel):
    """
    Merton (1976) jump-diffusion with lognormal jumps:

        dS/S = (mu - lam k) dt + sigma dW + (J - 1) dN,  log J ~ N(muJ, sigJ^2),  k = E[J] - 1,

    so that E[dS/S] = mu dt. Simulated exactly at the decision frequency.
    """
    name = "jump"

    def __init__(self, r=0.02, mu=0.08, sigma=0.144, lam=0.5, muJ=-0.2, sigJ=0.1):
        super().__init__(r)
        self.mu, self.sigma, self.lam = float(mu), float(sigma), float(lam)
        self.muJ, self.sigJ = float(muJ), float(sigJ)

    @property
    def k(self) -> float:
        return math.exp(self.muJ + 0.5 * self.sigJ ** 2) - 1.0

    @property
    def jump_second_moment(self) -> float:
        """E[(J-1)^2]."""
        ej = math.exp(self.muJ + 0.5 * self.sigJ ** 2)
        ej2 = math.exp(2 * self.muJ + 2 * self.sigJ ** 2)
        return ej2 - 2 * ej + 1

    @property
    def total_var(self) -> float:
        return self.sigma ** 2 + self.lam * self.jump_second_moment

    def simulate(self, n, steps, dt, rng):
        z = rng.standard_normal((n, steps))
        nj = rng.poisson(self.lam * dt, size=(n, steps))
        zj = rng.standard_normal((n, steps))
        logr = ((self.mu - self.lam * self.k - 0.5 * self.sigma ** 2) * dt
                + self.sigma * math.sqrt(dt) * z + nj * self.muJ + np.sqrt(nj) * self.sigJ * zj)
        return np.expm1(logr), None, None


class Mixture(MarketModel):
    """
    Mixture model. Each path is drawn from one of several component models (domain randomization).

    Attributes:
        models (list[MarketModel]): List of market models.
        weights (np.array): Relative weights of models.
    """
    name = "mixture"

    def __init__(self, models: Sequence[MarketModel], weights: Sequence[float] | None = None):
        """
        Creates a mixture model.

        Args:
            models (Sequence[MarketModel]): Sequence of models.
            weights (Sequence[float] | None = None): Relative weights of models. Models with a higher
                weight will have more probability of being chosen during simulation.
        """
        rs = {m.r for m in models}
        if len(rs) != 1:
            raise ValueError("mixture components must share the risk-free rate")
        super().__init__(models[0].r)
        self.models = list(models)
        w = np.ones(len(models)) if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()

    def params(self):
        return {"n_components": len(self.models)}

    def simulate(self, n, steps, dt, rng):
        counts = rng.multinomial(n, self.weights)
        parts = [m.simulate(int(c), steps, dt, rng)[0] for m, c in zip(self.models, counts) if c > 0]
        rets = np.concatenate(parts, axis=0)
        return rets[rng.permutation(n)], None, None

    def __repr__(self):
        return "Mixture(" + ", ".join(m.name for m in self.models) + ")"


# ----------------------------------------------------------------------------
# Features (the information set handed to learned policies)
# ----------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    """Exogenous features observed by a learned policy at each decision time.

    info:
      'time'    -> [tau] only (sufficient for i.i.d. models)
      'full'    -> [tau, latent state]  (requires a model with a latent state)
      'returns' -> [tau, EWMA bank of past excess log returns and squared returns]
      'both'    -> union of 'full' and 'returns'
    EWMAs are bias-corrected averages of all returns observed so far (including the
    pre-sample), annualized (divided by dt). Half-lives are in periods.
    A steady-state Kalman filter of a linear Gaussian drift is an EWMA of returns, so
    this bank can represent (approximately) the partial-information optimal policies.

    Attributes:
        info (str): Information set.
        half_lives (tuple[float, ...]): List of half-lives for EWMA.
        sq_half_lives (tuple[float, ...]): Half-lives for squared returns.
        use_rv (bool): True, if EWMA should be built from realized varainces.
        rv_half_lives (tuple[float, ...]): Half-lives for RV.
    """
    info: str = "returns"
    half_lives: tuple[float, ...] = (3.0, 12.0, 36.0, 120.0)
    sq_half_lives: tuple[float, ...] = (3.0, 12.0, 36.0)
    use_rv: bool = False                     # add EWMAs of realized variance (SV models only)
    rv_half_lives: tuple[float, ...] = (1.0, 3.0, 12.0)

    def dim(self) -> int:
        d = 1
        if self.info in ("full", "both"):
            d += 1
        if self.info in ("returns", "both"):
            d += len(self.half_lives) + len(self.sq_half_lives)
            if self.use_rv:
                d += len(self.rv_half_lives)
        return d


def _ewma_states(x: torch.Tensor, half_lives: Sequence[float], burn: int, T: int) -> torch.Tensor:
    """Bias-corrected EWMA state *before* each episode return. x: [N, burn+T] -> [N, T, H]."""
    N = x.shape[0]
    alphas = torch.tensor([1 - 0.5 ** (1.0 / h) for h in half_lives], dtype=x.dtype, device=x.device)
    s = torch.zeros(N, len(half_lives), dtype=x.dtype, device=x.device)
    decay = torch.ones(len(half_lives), dtype=x.dtype, device=x.device)
    out = torch.empty(N, T, len(half_lives), dtype=x.dtype, device=x.device)
    for k in range(burn + T):
        if k >= burn:
            corr = torch.clamp(1 - decay, min=1e-12)
            out[:, k - burn] = s / corr
        s = (1 - alphas) * s + alphas * x[:, k:k + 1]
        decay = decay * (1 - alphas)
    return out


def build_features(returns: torch.Tensor, latent: torch.Tensor | None, rf: float, dt: float,
                   burn: int, T: int, spec: FeatureSpec, rv: torch.Tensor | None = None) -> torch.Tensor:
    """
    Builds the feature vector.

    Args:
        returns (torch.Tensor): Vector of returns so far.
        latent (torch.Tensor | None): Optional latent states.
        rf (float): Constant risk-free return.
        dt (float): Period length.
        burn (int): Initial burn period.
        T (int): Episode length.
        spec (FeatureSpec): Feature information.
        rv (torch.Tensor | None = None): Realized variances.

    Returns:
        torch.Tensor: Feature vector.
    """
    N = returns.shape[0]
    tau = (T - torch.arange(T, dtype=returns.dtype, device=returns.device)) / T
    parts = [tau.view(1, T, 1).expand(N, T, 1)]
    if spec.info in ("full", "both"):
        if latent is None:
            raise ValueError("info='full' requires a latent state")
        parts.append(latent[:, burn:burn + T].unsqueeze(-1))
    if spec.info in ("returns", "both"):
        ex = torch.log1p(returns) - math.log1p(rf)
        if spec.half_lives:
            parts.append(_ewma_states(ex, spec.half_lives, burn, T) / dt)
        if spec.sq_half_lives:
            parts.append(_ewma_states(ex ** 2, spec.sq_half_lives, burn, T) / dt)
        if spec.use_rv:
            if rv is None:
                raise ValueError("use_rv=True requires a model that produces realized variance")
            parts.append(_ewma_states(rv, spec.rv_half_lives, burn, T) / dt)
    return torch.cat(parts, dim=-1)


# ----------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------
class Sampler:
    """
    Sampler base class.
    Produces PathBatch objects (with features) for training or evaluation.

    Class attributes:
        dt (float): Period length.
        burn (int): Initial burn period.
        horizon (int): Horizon length.
        spec (FeatureSpec): Feature vector information.
        rf (float): Constant risk-free return.
    """
    dt: float
    burn: int
    horizon: int
    spec: FeatureSpec
    rf: float

    def sample(self, n: int) -> PathBatch:
        raise NotImplementedError

    def _make(self, rets: np.ndarray, lat: np.ndarray | None, name: str,
              rv: np.ndarray | None = None) -> PathBatch:
        """
        Creates a ``PathBatch`` from the information given.

        Args:
            rets (np.ndarray): Returns.
            lat (np.ndarray | None): Latent states.
            name (str): Human-readable name for the batch.
            rv (np.ndarray | None = None): Realized variances.
        """
        R = torch.as_tensor(rets, dtype=torch.float64)
        Lt = None if lat is None else torch.as_tensor(lat, dtype=torch.float64)
        RV = None if rv is None else torch.as_tensor(rv, dtype=torch.float64)
        feats = build_features(R, Lt, self.rf, self.dt, self.burn, self.horizon, self.spec, RV)
        b = PathBatch(R, Lt, self.rf, self.dt, self.burn, self.horizon, feats, name, RV)
        return b.to(device=self.device, dtype=self.dtype)


class SimSampler(Sampler):
    """
    Fresh paths simulated from a market model (Protocol 1 and 3).

    Attributes:
        model (MarketModel): Market model.
        dt (float): Period length.
        burn (int): Initial burn period.
        horizon (int): Horizon length.
        spec (FeatureSpec): Feature vector information.
        rf (float): Constant risk-free return.
        rng (np.random.Generator): Random number generator.
        device (str): PyTorch device.
        dtype (torch.float32): PyTorch datatype.
        pool (PathBatch | None): Fixed pool of simulated paths.
    """

    def __init__(self, model: MarketModel, dt: float, burn: int, horizon: int,
                 spec: FeatureSpec, seed: int = 0, device="cpu", dtype=torch.float32,
                 pool_size: int | None = None, chunk: int = 20000):
        """
        Creates a simulation sampler.

        Args:
            model (MarketModel): Market model.
            dt (float): Period length.
            burn (int): Initial burn period.
            horizon (int): Horizon length.
            spec (FeatureSpec): Feature vector information.
            seed (int): Seed for the RNG.
            device (str): PyTorch device.
            dtype (torch.float32): PyTorch dtype.
            pool_size (int | None = None): Pool size.
            chunk (int): Size of each part in the pool

        If ``pool_size`` is given, a fixed pool of simulated paths is generated once and
        batches are drawn from it (the pool size is then the simulation budget).
        """
        self.model, self.dt, self.burn, self.horizon, self.spec = model, dt, burn, horizon, spec
        self.rng = np.random.default_rng(seed)
        self.rf = model.rf_step(dt)
        self.device, self.dtype = device, dtype
        self.pool: PathBatch | None = None
        if pool_size:
            parts = [self._fresh(min(chunk, pool_size - i)) for i in range(0, pool_size, chunk)]
            self.pool = _concat(parts)

    def _fresh(self, n: int) -> PathBatch:
        """
        Simulates fresh episodes.

        Args:
            n (int): Number of episodes to simulate.

        Returns:
            PathBatch: Batch of episodes.
        """
        rets, lat, rv = self.model.simulate(n, self.burn + self.horizon, self.dt, self.rng)
        return self._make(rets, lat, self.model.name, rv)

    def sample(self, n: int) -> PathBatch:
        """
        Samples episodes from the pool.

        Args:
            n (int): Number of trajectories to sample.

        Returns:
            PathBatch: Sampled episodes.
        """
        if self.pool is None:
            return self._fresh(n)
        idx = torch.as_tensor(self.rng.choice(self.pool.n, size=min(n, self.pool.n), replace=False),
                              device=self.pool.returns.device)
        return self.pool.subset(idx)


class HistorySampler(Sampler):
    """
    Random (overlapping) windows of length burn+T from one observed history (Protocol 2).
    
    Attributes:
        rf (float): Risk-free return.
        dt (float): Period length.
        burn (int): Initial burn period.
        horizon (int): Horizon length.
        spec (FeatureSpec): Feature vector information.
        device (str): PyTorch device.
        dtype (torch.float32): PyTorch dtype.
        name (str): Human-readable name for the sampler.
        hist_rv (np.ndarray | None): History of realized variances.
        hist_l (np.ndarray | None): History of latent states.
        hist_r (np.ndarray | None): History of returns.
        L (int): burn + horizon.
    """

    def __init__(self, returns: np.ndarray, latent: np.ndarray | None, rf: float, dt: float,
                 burn: int, horizon: int, spec: FeatureSpec, seed: int = 0, device="cpu",
                 dtype=torch.float32, name: str = "history", rv: np.ndarray | None = None):
        self.hist_r = np.asarray(returns, float)
        self.hist_l = None if latent is None else np.asarray(latent, float)
        self.hist_rv = None if rv is None else np.asarray(rv, float)
        self.rf, self.dt, self.burn, self.horizon, self.spec = rf, dt, burn, horizon, spec
        self.rng = np.random.default_rng(seed)
        self.device, self.dtype, self.name = device, dtype, name
        self.L = burn + horizon
        if len(self.hist_r) < self.L:
            raise ValueError(f"history of length {len(self.hist_r)} shorter than window {self.L}")

    @property
    def n_windows(self) -> int:
        return len(self.hist_r) - self.L + 1

    def all_windows(self) -> PathBatch:
        """
        Every window of the history once (deterministic, used for held-out validation).
        
        Returns:
            PathBatch: Windows of history.
        """
        starts = np.arange(self.n_windows)
        idx = starts[:, None] + np.arange(self.L)[None, :]
        rv = None if self.hist_rv is None else self.hist_rv[idx]
        lat = None if self.hist_l is None else self.hist_l[starts[:, None] + np.arange(self.L + 1)[None, :]]
        return self._make(self.hist_r[idx], lat, self.name, rv)

    def sample(self, n: int) -> PathBatch:
        """
        Samples possibly overlapping windows from history.

        Args:
            n (int): Number of windows to sample.

        Returns:
            PathBatch: Batch of windows.
        """
        starts = self.rng.integers(0, self.n_windows, size=n)
        idx = starts[:, None] + np.arange(self.L)[None, :]
        rets = self.hist_r[idx]
        rv = None if self.hist_rv is None else self.hist_rv[idx]
        lat = None
        if self.hist_l is not None:
            lat = self.hist_l[starts[:, None] + np.arange(self.L + 1)[None, :]]
        return self._make(rets, lat, self.name, rv)


class BlockBootstrapSampler(HistorySampler):
    """
    Paths built by concatenating random blocks of the history (moving-block bootstrap).

    Attributes:
        block (int): Length of a block. 
    """

    def __init__(self, *args, block: int = 24, **kw):
        super().__init__(*args, **kw)
        self.block = int(block)

    def sample(self, n: int) -> PathBatch:
        """
        Samples paths by concatenating random blocks of the history.

        Args:
            n (int): Number of paths to sample.

        Returns:
            PathBatch: Batch of paths.
        """
        H = len(self.hist_r)
        nb = math.ceil(self.L / self.block)
        starts = self.rng.integers(0, H - self.block + 1, size=(n, nb))
        idx = (starts[:, :, None] + np.arange(self.block)[None, None, :]).reshape(n, -1)[:, :self.L]
        rets = self.hist_r[idx]
        rv = None if self.hist_rv is None else self.hist_rv[idx]
        lat = None
        if self.hist_l is not None:
            idx_l = np.concatenate([idx, idx[:, -1:] + 1], axis=1).clip(max=len(self.hist_l) - 1)
            lat = self.hist_l[idx_l]
        return self._make(rets, lat, self.name + "_bootstrap", rv)


def _concat(parts: list[PathBatch]) -> PathBatch:
    """
    Concatenate a list of paths.

    Args:
        parts (list[PathBatch]): List of paths.
    
    Returns:
        PathBatch: Concatenated paths.
    """
    def cat(attr):
        xs = [getattr(p, attr) for p in parts]
        return None if xs[0] is None else torch.cat(xs, 0)
    return replace(parts[0], returns=cat("returns"), latent=cat("latent"), feats=cat("feats"), rv=cat("rv"))


def simulate_history(model: MarketModel, n_periods: int, dt: float, seed: int
                     ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    One long observed history (used as the training data set in Protocol 2).

    Args:
        model (MarketModel): Market model.
        n_periods (int): Number of steps in each path.
        dt (int): Period length in years.
        seed (int): Seed for RNG.
    
    Returns:
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]: Returns,
            latent states, realized variances.
    """
    rng = np.random.default_rng(seed)
    r, l, rv = model.simulate(1, n_periods, dt, rng)
    return r[0], (None if l is None else l[0]), (None if rv is None else rv[0])
