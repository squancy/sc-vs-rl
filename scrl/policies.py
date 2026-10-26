from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from . import sc
from .env import EnvConfig, Obs, Policy, best_constant_fraction
from .models import GBM, ChackoViceira, Heston, KimOmberg, MarketModel, MertonJump, Mixture, PathBatch


# ----------------------------------------------------------------------------
# Generic wrappers
# ----------------------------------------------------------------------------
class ConstantPolicy(Policy):
    """
    Constant policy.

    Attributes:
        name (str): Human readable name.
        value (float): Value of the policy.
    """
    def __init__(self, name: str, value: float):
        self.name, self.value = name, float(value)

    def act(self, t: int, obs: Obs) -> torch.Tensor:
        """
        Acts using the constant value.

        Args:
            t (int): Current time (not used).
            obs (Obs): Observations containing current wealth.

        Returns:
            torch.Tensor: Current wealth, policy value.
        """
        return torch.full_like(obs.w, self.value)


class TimeTablePolicy(Policy):
    """
    Deterministic function of time only (e.g. the Kraft Heston policy).

    Attributes:
        name (str): Human readable name.
        table (str): Table of precomputed policy values indexed by time.
    """

    def __init__(self, name: str, table: np.ndarray):
        self.name, self.table = name, np.asarray(table, float)

    def act(self, t: int, obs: Obs) -> torch.Tensor:
        """
        Acts using the table.

        Args:
            t (int): Current time.
            obs (Obs): Observations containing current wealth.

        Returns:
            torch.Tensor: Current wealth, policy value.
        """
        return torch.full_like(obs.w, float(self.table[t]))


class PrecomputedPolicy(Policy):
    """
    Base for policies whose actions depend only on exogenous information:
    ``compute(batch) -> [N, T]`` is evaluated once at reset.
    """

    def compute(self, batch: PathBatch) -> torch.Tensor:
        """
        Precompute policy values using a batch.

        Args:
            batch (PathBatch): Batch of paths.

        Returns:
            torch.Tensor: Precomputed values.
        """
        raise NotImplementedError

    def reset(self, batch: PathBatch):
        """
        Precomputes policy values. 

        Args:
            batch (PathBatch): Batch of paths.
        """
        self._table = self.compute(batch)

    def act(self, t: int, obs: Obs) -> torch.Tensor:
        """
        Acts using the precomputed values.

        Args:
            t (int): Current time.
            obs (Obs): Observation.
        
        Returns:
            torch.Tensor: Precomputed value at time t.
        """
        return self._table[:, t]


def _log_rets(batch: PathBatch) -> torch.Tensor:
    return torch.log1p(batch.returns.double())


def _taus(batch: PathBatch) -> np.ndarray:
    return (batch.horizon - np.arange(batch.horizon)) * batch.dt


# ----------------------------------------------------------------------------
# Kim-Omberg policies
# ----------------------------------------------------------------------------
class KOStatePolicy(PrecomputedPolicy):
    """pi_t = function of a scalar state s_t (true X or the Kalman estimate m_t).

    mode:
      'analytic' -> continuous-time KO solution ``sol`` evaluated at s_t
      'myopic'   -> s_t / (gamma sigma)
      'dp'       -> exact discrete-time DP table on s_t
    state:
      'latent'   -> true X (full information)
      'filter'   -> Kalman-filtered X from returns using model ``filt_model``

    Attributes:
        name (str): Human-readable name.
        mode (str): One of "analytic", "myopic", "dp".
        state (str): One of "latent", "filter".
        gamma (float): Risk-aversion rate.
        sigma (float): Variance term of the price.
        sol (sc.KOSolution | None = None): Analytical solution.
        table (sc.DPTable | None = None): DP table for the discrete time solution.
        filt_model (KimOmberg | None = None): Model used for filtering.
        feasible (bool = True): True, if the model is feasible.
    """

    def __init__(self, name: str, mode: str, state: str, gamma: float, sigma: float,
                 sol: sc.KOSolution | None = None, table: sc.DPTable | None = None,
                 filt_model: KimOmberg | None = None, feasible: bool = True):
        self.name, self.mode, self.state = name, mode, state
        self.gamma, self.sigma, self.sol, self.table = gamma, sigma, sol, table
        self.filt_model, self.feasible = filt_model, feasible

    def _states(self, batch: PathBatch) -> torch.Tensor:
        """
        Retrieves the latent states (true Sharpe ratios)
        or estimates them from past returns.

        Args:
            batch (PathBatch): Episodes containting previous
                returns. 
        
        Returns:
            torch.Tensor: Latent or filtered states.
        """
        B, T = batch.burn, batch.horizon
        if self.state == "latent":
            return batch.latent[:, B:B + T].double()
        kf = sc.KOKalman(self.filt_model, batch.dt)
        m = kf.filter(_log_rets(batch))            # m_k uses returns 0..k-1 only
        return m[:, B:B + T]

    def compute(self, batch: PathBatch) -> torch.Tensor:
        """
        Computes the optimal policy depending on the mode. 

        Args:
            batch (PathBatch): Batch of paths.

        Returns:
            torch.Tensor: Optimal policy.
        """
        s = self._states(batch).cpu().numpy()
        taus = _taus(batch)
        out = np.empty_like(s)
        for t in range(batch.horizon):
            if self.mode == "analytic":
                out[:, t] = self.sol.policy(taus[t], s[:, t])
            elif self.mode == "myopic":
                out[:, t] = s[:, t] / (self.gamma * self.sigma)
            elif self.mode == "dp":
                out[:, t] = self.table.policy(t, s[:, t])
            else:
                raise ValueError(self.mode)
        return torch.as_tensor(out, dtype=batch.returns.dtype, device=batch.returns.device)


# ----------------------------------------------------------------------------
# Chacko-Viceira policies
# ----------------------------------------------------------------------------
def ewma_before(x: torch.Tensor, half_life: float) -> torch.Tensor:
    """
    Bias-corrected EWMA of x[:, :k] for every k (state *before* observation k). [N, K] -> [N, K].
    
    Args:
        x (torch.Tensor): Vector containing the data.
        half_life (float): Half life for the exponential weighted moving average.

    Returns:
        torch.Tensor: Bias-corrected EWMA of x[:, :k].
    """
    N, K = x.shape
    alpha = 1 - 0.5 ** (1.0 / half_life)
    s = torch.zeros(N, dtype=x.dtype, device=x.device)
    dec = 1.0
    out = torch.empty_like(x)
    for k in range(K):
        out[:, k] = s / max(1 - dec, 1e-12) if k > 0 else torch.full_like(s, float("nan"))
        s = (1 - alpha) * s + alpha * x[:, k]
        dec *= (1 - alpha)
    return out


class CVPolicy(PrecomputedPolicy):
    """
    pi = y_t (mu_e + rho sigma_y B(tau) * hedge) / gamma with y_t either the true precision
    (state='latent') or a plug-in 1 / (EWMA variance) estimated from realized variance
    (state='rv') or from squared returns (state='sqret') with half-life ``half_life`` periods.

    Attributes:
        name (str): Human-readable name.
        mode (str): One of "analytic", "myopic", "dp".
        state (str): One of "latent", "filter".
        gamma (float): Risk-aversion rate.
        hedge (bool = True): True, if the hedging term should be included in the policy.
        half_life (float): Half life used for EWMA.
        feasible (bool = True): True, if the model is feasible.
    """

    def __init__(self, name: str, model: ChackoViceira, gamma: float, state: str = "latent",
                 hedge: bool = True, half_life: float = 3.0, feasible: bool = True):
        self.name, self.model, self.gamma, self.state = name, model, gamma, state
        self.hedge, self.half_life, self.feasible = hedge, half_life, feasible

    def _precision(self, batch: PathBatch) -> torch.Tensor:
        """
        Calculates the precision term 1 / y_t using latent states,
        realized variances or squared log returns.

        Args:
            batch (PathBatch): Batch of paths used for estimating y_t.

        Returns:
            torch.Tensor: 1 / y_t, after EWMA.
        """
        B, T = batch.burn, batch.horizon
        if self.state == "latent":
            return batch.latent[:, B:B + T].double()
        if self.state == "rv" and batch.rv is not None:
            x = batch.rv.double()
        else:
            x = (torch.log1p(batch.returns.double()) - math.log1p(batch.rf)) ** 2
        var = ewma_before(x, self.half_life)[:, B:B + T] / batch.dt
        return 1.0 / torch.clamp(var, min=1e-4)

    def compute(self, batch: PathBatch) -> torch.Tensor:
        """
        Computes the optimal policy. 

        Args:
            batch (PathBatch): Batch of paths.

        Returns:
            torch.Tensor: Optimal policy.
        """
        y = self._precision(batch).cpu().numpy()
        taus = _taus(batch)
        out = np.stack([sc.cv_policy(taus[t], y[:, t], self.model, self.gamma, self.hedge)
                        for t in range(batch.horizon)], axis=1)
        return torch.as_tensor(out, dtype=batch.returns.dtype, device=batch.returns.device)


# ----------------------------------------------------------------------------
# Plug-in estimator policies (model-free SC with estimated parameters)
# ----------------------------------------------------------------------------
class RollingMertonPolicy(PrecomputedPolicy):
    """pi_t = mu_hat / (gamma sigma_hat^2) from the trailing ``window`` excess returns.

    Optional Bayesian shrinkage of the rolling mean towards ``prior_mu`` with prior weight
    ``prior_n`` periods: mu = (n mu_roll + prior_n prior_mu) / (n + prior_n).
    ``vol_halflife`` replaces the rolling variance with an EWMA variance (volatility timing).

    Attributes:
        name (str): Human-readable name.
        gamma (float): Risk-aversion rate.
        window (int): Window length.
        prior_mu (int | None = None): Prior mu for Bayesian shrinkage. 
        prior_n (float = 0.0): Prior weight for Bayesian shrinkage.
        vol_halflife (float | None = None): Half-life for EWMA.
        fixed_mu (float | None = None): Fixed mu instead of estimating it from previous
            returns.
    """

    def __init__(self, name: str, gamma: float, window: int, prior_mu: float | None = None,
                 prior_n: float = 0.0, vol_halflife: float | None = None, fixed_mu: float | None = None):
        self.name, self.gamma, self.window = name, gamma, int(window)
        self.prior_mu, self.prior_n = prior_mu, float(prior_n)
        self.vol_halflife, self.fixed_mu = vol_halflife, fixed_mu

    def compute(self, batch: PathBatch) -> torch.Tensor:
        """
        Computes the optimal policy.
        Mu and sigma can be computed from previous returns 
        (using EWMA, Bayesian shrinkage or naive plug-in estimates).
        Mu can be also fixed.

        Args:
            batch (PathBatch): Batch of paths.

        Returns:
            torch.Tensor: Optimal policy.
        """
        B, T, dt = batch.burn, batch.horizon, batch.dt
        e = (batch.returns.double() - batch.rf)
        N = e.shape[0]
        cs = torch.cat([torch.zeros(N, 1, dtype=e.dtype, device=e.device), torch.cumsum(e, 1)], 1)
        cs2 = torch.cat([torch.zeros(N, 1, dtype=e.dtype, device=e.device), torch.cumsum(e * e, 1)], 1)
        out = torch.empty(N, T, dtype=e.dtype, device=e.device)
        if self.vol_halflife is not None:
            alpha = 1 - 0.5 ** (1.0 / self.vol_halflife)
            s, dec = torch.zeros(N, dtype=e.dtype, device=e.device), 1.0
            ew = torch.empty(N, B + T, dtype=e.dtype, device=e.device)
            for k in range(B + T):
                ew[:, k] = s / max(1 - dec, 1e-12)   # EWMA of e^2 before return k
                s = (1 - alpha) * s + alpha * e[:, k] ** 2
                dec *= (1 - alpha)
        for t in range(T):
            n_obs = B + t
            L = min(self.window, n_obs)
            s1 = cs[:, n_obs] - cs[:, n_obs - L]
            s2 = cs2[:, n_obs] - cs2[:, n_obs - L]
            mu = s1 / L
            var = torch.clamp(s2 / L - mu * mu, min=1e-10) * L / max(L - 1, 1)
            if self.prior_mu is not None and self.prior_n > 0:
                mu = (L * mu + self.prior_n * self.prior_mu * dt) / (L + self.prior_n)
            if self.fixed_mu is not None:
                mu = torch.full_like(mu, self.fixed_mu * dt)
            if self.vol_halflife is not None:
                var = torch.clamp(ew[:, n_obs], min=1e-10)
            out[:, t] = (mu / dt) / (self.gamma * var / dt)
        return out.to(batch.returns.dtype)


class BandPolicy(Policy):
    """
    No-trade band around a base target (Janecek & Shreve 2004 half-width, computed from the
    current target): trade to the nearest band edge only when the drifted weight leaves it.

    Attributes:
        base (Policy): Base policy.
        gamma (float): Risk-aversion rate.
        cost (float): Trading costs.
        name (str | None = None): Human-readable name.
    """

    def __init__(self, base: Policy, gamma: float, cost: float, name: str | None = None):
        self.base, self.gamma, self.cost = base, gamma, cost
        self.name = name or f"{base.name}+band"

    def reset(self, batch: PathBatch):
        """
        Resets the base policy.

        Args:
            batch (PathBatch): Batch of paths.
        """
        self.base.reset(batch)

    def act(self, t: int, obs: Obs) -> torch.Tensor:
        """
        Acts according to the no-trade band policy.

        Args:
            t (int): Current timestep.
            obs (Obs): Observation.

        Returns:
            torch.Tensor: Drifted weights if it's inside the bands, and the upper/lower
            band otherwise. 
        """
        target = self.base.act(t, obs)
        if t == 0:
            return target
        hw = torch.as_tensor(sc.js_band_halfwidth(target.double().cpu().numpy(), self.gamma, self.cost),
                             dtype=target.dtype, device=target.device)
        return torch.minimum(torch.maximum(obs.drift, target - hw), target + hw)


# ----------------------------------------------------------------------------
# Benchmark factories
# ----------------------------------------------------------------------------
def optimal_policy_set(model: MarketModel, info: str, gamma: float, dt: float, burn: int, steps: int,
                       env: EnvConfig, dp: bool = True, ref_batch: PathBatch | None = None
                       ) -> tuple[dict[str, Policy], str]:
    """SC benchmarks for a *known* model and information set.

    Returns (policies, reference_name). The reference is the exact discrete-time optimum
    when available ('DP:...'), otherwise the continuous-time analytical policy.
    ``info`` in {'full', 'returns', 'time'}; for models without a latent state the information
    set is irrelevant. ``ref_batch`` (if given) is used to fit the best constant fraction.

    Args:
        model (MarketModel): Market model (GBM, Heston, MertonJump, ...).
        info (str): Information set. One of "full", "returns", "time".
        gamma (float): Risk-aversion rate.
        dt (float): Period length.
        burn (int): Initial burn period.
        steps (int): Number of steps.
        env (EnvConfig): Configuration object.
        dp (bool = True): True, if the DP policy should be included.
        ref_batch (PathBatch | None = None): Reference dataset for the constant policy
            and CV.

    Returns:
        tuple[dict[str, Policy], str]: SC policies.
    """
    T_years = steps * dt
    pols: dict[str, Policy] = {}
    ref = None
    lo, hi = env.lo, env.hi
    if isinstance(model, GBM):
        pols["SC:Merton"] = ConstantPolicy("SC:Merton", sc.merton_fraction(model.mu - model.r, model.sigma ** 2, gamma))
        if dp:
            pols["DP:discrete"] = ConstantPolicy("DP:discrete", sc.dp_iid_fraction(model, gamma, dt, lo, hi))
            ref = "DP:discrete"
        ref = ref or "SC:Merton"
    elif isinstance(model, MertonJump):
        pols["SC:JumpFOC"] = ConstantPolicy("SC:JumpFOC", sc.jump_optimal_fraction(model, gamma))
        pols["SC:Merton(total var)"] = ConstantPolicy(
            "SC:Merton(total var)", sc.merton_fraction(model.mu - model.r, model.total_var, gamma))
        if dp:
            pols["DP:discrete"] = ConstantPolicy("DP:discrete", sc.dp_iid_fraction(model, gamma, dt, lo, hi))
            ref = "DP:discrete"
        ref = ref or "SC:JumpFOC"
    elif isinstance(model, Heston):
        taus = (steps - np.arange(steps)) * dt
        pols["SC:Kraft"] = TimeTablePolicy("SC:Kraft", sc.kraft_policy(taus, model, gamma))
        pols["SC:Myopic"] = ConstantPolicy("SC:Myopic", model.xi / gamma)
        ref = "SC:Kraft"
    elif isinstance(model, ChackoViceira):
        if info in ("full", "both"):
            pols["SC:ChackoViceira"] = CVPolicy("SC:ChackoViceira", model, gamma, "latent", True)
            pols["SC:CV-myopic"] = CVPolicy("SC:CV-myopic", model, gamma, "latent", False)
            ref = "SC:ChackoViceira"
        else:
            # No closed-form partial-information solution: SC uses a plug-in precision estimate.
            # The EWMA half-life is tuned on the reference batch (generous to SC).
            state = "rv" if (ref_batch is None or ref_batch.rv is not None) else "sqret"
            best_hl, best_ce = 3.0, -math.inf
            if ref_batch is not None:
                from .env import rollout
                from .utils import ce_stats
                for hl in (1.0, 2.0, 3.0, 6.0, 12.0):
                    o = rollout(ref_batch, CVPolicy("tmp", model, gamma, state, True, hl), env)
                    c = ce_stats(o["wT"], gamma, ref_batch.years)["ce_rate"]
                    if c > best_ce:
                        best_hl, best_ce = hl, c
            nm = f"SC:CV-plugin({state}, hl={best_hl:g})"
            pols[nm] = CVPolicy(nm, model, gamma, state, True, best_hl)
            pols["SC:CV-unconditional"] = ConstantPolicy(
                "SC:CV-unconditional", sc.merton_fraction(model.mu_e, model.mean_variance, gamma))
            pols["Oracle:full-info (infeasible)"] = CVPolicy(
                "Oracle:full-info (infeasible)", model, gamma, "latent", True, feasible=False)
            ref = "Oracle:full-info (infeasible)"
    elif isinstance(model, KimOmberg):
        if info in ("full", "both"):
            sol = sc.ko_full_solution(model, gamma, T_years)
            pols["SC:KimOmberg"] = KOStatePolicy("SC:KimOmberg", "analytic", "latent", gamma, model.sigma, sol=sol)
            pols["SC:Myopic"] = KOStatePolicy("SC:Myopic", "myopic", "latent", gamma, model.sigma)
            if dp:
                tab = sc.dp_ko_full(model, gamma, dt, steps, lo, hi)
                pols["DP:full-info"] = KOStatePolicy("DP:full-info", "dp", "latent", gamma, model.sigma, table=tab)
                ref = "DP:full-info"
            ref = ref or "SC:KimOmberg"
        else:
            kf = sc.KOKalman(model, dt)
            P_burn = kf.P_sequence(burn)[-1]
            solp = sc.ko_partial_solution(model, gamma, T_years, P_burn)
            solf = sc.ko_full_solution(model, gamma, T_years)
            pols["SC:KO-partial"] = KOStatePolicy("SC:KO-partial", "analytic", "filter", gamma, model.sigma,
                                                  sol=solp, filt_model=model)
            pols["SC:KO-cert-equiv"] = KOStatePolicy("SC:KO-cert-equiv", "analytic", "filter", gamma, model.sigma,
                                                     sol=solf, filt_model=model)
            pols["SC:Myopic-filtered"] = KOStatePolicy("SC:Myopic-filtered", "myopic", "filter", gamma,
                                                       model.sigma, filt_model=model)
            if dp:
                tab = sc.dp_ko_partial(model, gamma, dt, burn, steps, lo, hi)
                pols["DP:partial-info"] = KOStatePolicy("DP:partial-info", "dp", "filter", gamma, model.sigma,
                                                        table=tab, filt_model=model)
                ref = "DP:partial-info"
            ref = ref or "SC:KO-partial"
            # Infeasible upper bound: uses the true (unobserved) Sharpe ratio.
            pols["Oracle:full-info (infeasible)"] = KOStatePolicy(
                "Oracle:full-info (infeasible)", "analytic", "latent", gamma, model.sigma, sol=solf, feasible=False)
    elif isinstance(model, Mixture):
        ref = None
    else:
        raise TypeError(type(model))
    if ref_batch is not None:
        pols["Best-constant"] = ConstantPolicy("Best-constant", best_constant_fraction(ref_batch, env))
    return pols, ref


def believer_policy(model: MarketModel, gamma: float, dt: float, burn: int, steps: int,
                    name: str | None = None) -> Policy:
    """
    The returns-only SC policy of an investor who believes ``model`` (Protocol 3).
    
    Args:
        model (MarketModel): Market model.
        gamma (float): Risk-aversion rate.
        dt (float): Period length.
        burn (int): Initial burn period.
        steps (int): Number of steps.
        name (str | None = None): Human-readable name.

    Returns:
        Policy: Policy depending on the market model.
    """
    T_years = steps * dt
    if isinstance(model, GBM):
        p = ConstantPolicy("", sc.merton_fraction(model.mu - model.r, model.sigma ** 2, gamma))
    elif isinstance(model, MertonJump):
        p = ConstantPolicy("", sc.jump_optimal_fraction(model, gamma))
    elif isinstance(model, Heston):
        taus = (steps - np.arange(steps)) * dt
        p = TimeTablePolicy("", sc.kraft_policy(taus, model, gamma))
    elif isinstance(model, KimOmberg):
        P_burn = sc.KOKalman(model, dt).P_sequence(burn)[-1]
        solp = sc.ko_partial_solution(model, gamma, T_years, P_burn)
        p = KOStatePolicy("", "analytic", "filter", gamma, model.sigma, sol=solp, filt_model=model)
    elif isinstance(model, ChackoViceira):
        p = CVPolicy("", model, gamma, state="rv", hedge=True, half_life=3.0)
    else:
        raise TypeError(type(model))
    p.name = name or f"SC[{model.name}]"
    return p


def fitted_model_policy(model: MarketModel, info: str, gamma: float, dt: float, burn: int, steps: int,
                        name: str) -> Policy:
    """
    Analytical policy of a model with *estimated* parameters (Protocol 2).
    
    Args:
        model (MarketModel): Market model (GBM, Heston, MertonJump, ...).
        info (str): Information set. One of "full", "returns", "time".
        gamma (float): Risk-aversion rate.
        dt (float): Period length.
        burn (int): Initial burn period.
        steps (int): Number of steps.
        name (str): Human-readable name.

    Returns:
        Policy: Policy based on the market model.
    """
    if isinstance(model, KimOmberg) and info in ("full", "both"):
        sol = sc.ko_full_solution(model, gamma, steps * dt)
        return KOStatePolicy(name, "analytic", "latent", gamma, model.sigma, sol=sol)
    if isinstance(model, ChackoViceira) and info in ("full", "both"):
        return CVPolicy(name, model, gamma, "latent", True)
    return believer_policy(model, gamma, dt, burn, steps, name=name)
