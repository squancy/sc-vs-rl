from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .models import PathBatch
from .utils import ce_stats, paired_ce_gap


@dataclass
class EnvConfig:
    gamma: float = 5.0
    lo: float = -1.0       # bounds on the risky fraction, applied to every policy
    hi: float = 2.0
    cost: float = 0.0      # proportional transaction cost on turnover
    w0: float = 1.0


@dataclass
class Obs:
    t: int
    w: torch.Tensor        # [N] current wealth
    drift: torch.Tensor    # [N] drifted weight before trading (0 at t = 0)
    batch: PathBatch


class Policy:
    """
    Interface shared by SC benchmarks and learned policies.

    ``reset`` is called once per batch before the first decision
    Policies whose actions depend only on exogenous information
    (returns, latent state, time) may precompute all actions there.
    
    ``act`` returns the risky fraction for every path at decision time t.
    """
    name: str = "policy"
    feasible: bool = True   # False for policies that use information the investor does not have

    def reset(self, batch: PathBatch) -> None:
        pass

    def act(self, t: int, obs: Obs) -> torch.Tensor:
        raise NotImplementedError


def wealth_step(w: float, drift: float, pi: float, R: float, rf: float, cost: float) -> tuple[float, float]:
    """
    Performs a single timestep according to the wealth dynamics.

    On a transition from timestep t to t + 1, the following happens:
    
    Wealth progresses as  
    W_{t+1} = W_t (1 - c |pi_t - d_t|) (1 + r_f + pi_t (R_{t+1} - r_f)).

    where W_t is the wealth at time t, c is the proportional transaction
    cost per unit of wealth traded, pi_t is the fraction of wealth in
    the risy asset, d_t is the drifted weight, r_f is the return of the
    risk-free asset and R_{t + 1} is the return of the risy asset over (t, t + 1].
    
    Drifted weight progresses as
    d_{t+1} = pi_t (1 + R_{t+1}) / (1 + r_f + pi_t (R_{t+1} - r_f)).

    This is the risky fraction the portfolio has at time t before rebalancing,
    after last period's price move.

    Args:
        w (float): Wealth at current time t.
        drift (float): Drifted weights at time t.
        pi (float): Fraction of wealth in the risky asset at time t.
        R (float): Return of the risky asset in (t, t + 1].
        rf (float): Return of the risk-free asset.
    
    Returns:
        tuple[float, float]: Wealth and drift at time t + 1.
    """
    gross = 1.0 + rf + pi * (R - rf)
    w_new = w * gross
    if cost > 0:
        w_new = w_new * (1.0 - cost * torch.abs(pi - drift))
    drift_new = pi * (1.0 + R) / gross
    return w_new, drift_new


@torch.no_grad()
def rollout(batch: PathBatch, policy: Policy, cfg: EnvConfig, record: bool = False) -> dict[str, torch.Tensor]:
    """
    Does a single episode rollout.

    Args:
        batch (PathBatch): Batch of returns.
        policy (Policy): Policy that acts in the environment.
        cfg (EnvConfig): Config object.
        record (bool = False): True, if the allocations should be recorded.
    
    Returns:
        dict[str, torch.Tensor]: Final wealth and allocations (optionally).
    """
    N, T = batch.n, batch.horizon
    dev, dt_ = batch.returns.device, batch.returns.dtype
    w = torch.full((N,), cfg.w0, dtype=dt_, device=dev)
    drift = torch.zeros(N, dtype=dt_, device=dev)
    policy.reset(batch)
    R_ep = batch.episode_returns()
    pis = torch.empty(N, T, dtype=dt_, device=dev) if record else None
    for t in range(T):
        pi = policy.act(t, Obs(t, w, drift, batch)).to(dt_)
        pi = torch.clamp(pi, cfg.lo, cfg.hi)
        if record:
            pis[:, t] = pi
        w, drift = wealth_step(w, drift, pi, R_ep[:, t], batch.rf, cfg.cost)
    out = {"wT": w}
    if record:
        out["pi"] = pis
    return out


def evaluate_policies(batch: PathBatch, policies: dict[str, Policy], cfg: EnvConfig,
                      ref: str | None, policy_ref: str | None) -> tuple[dict, dict]:
    """
    Roll out every policy on the same paths. Reports CE statistics, the paired CE gap
    (bp/year) to the reference policy ``ref`` and the RMSE of the action relative to
    ``policy_ref`` (defaults to ``ref``) over all visited decision states.

    Args:
        batch (PathBatch): Batch to roll out.
        policies (dict[str, Policy]): Dictionary of policies to evaluate.
        cfg (EnvConfig): Config object.
        ref (str | None): Name of the reference policy.
        policy_ref (str | None): Name of the action reference policy.
    
    Returns:
        tuple[dict, dict]: For each policy, the result of the rollout
        and the calculates CE statistics.
    """
    res, outs = {}, {}
    for name, pol in policies.items():
        outs[name] = rollout(batch, pol, cfg, record=True)
    years = batch.years
    policy_ref = policy_ref or ref
    for name, o in outs.items():
        r = ce_stats(o["wT"], cfg.gamma, years, cfg.w0)
        r["feasible"] = getattr(policies[name], "feasible", True)
        if ref is not None and ref in outs:
            r.update(paired_ce_gap(outs[ref]["wT"], o["wT"], cfg.gamma, years))
        if policy_ref is not None and policy_ref in outs:
            d = (o["pi"] - outs[policy_ref]["pi"]).double()
            r["pi_rmse"] = float(torch.sqrt(torch.mean(d ** 2)))
        p = o["pi"].double()
        r["pi_mean"] = float(p.mean())
        r["pi_sd_cross"] = float(p.std(dim=0).mean())
        r["turnover"] = float((p[:, 1:] - p[:, :-1]).abs().mean())
        res[name] = r
    return res, outs


# ----------------------------------------------------------------------------
# Best constant fraction (a useful "static" baseline)
# ----------------------------------------------------------------------------
@torch.no_grad()
def best_constant_fraction(batch: PathBatch, cfg: EnvConfig, iters: int = 50) -> float:
    """
    Maximize the empirical CE by using a constant policy.
    For a constant policy pi (rebalanced to pi after every timestep)
    and assuming no transaction costs, the terminal wealth on path i
    is W_T^i(pi) = W_0 * prod(1 + r_f * pi * (R_{t + 1}^i - r_f))

    Then, we maximize the empirical log CE:

    log_CE(pi) = log(1/N * sum(W_T^i(pi)^{1 - gamma})) / (1 - gamma)

    over [lo, hi]. It is a concave function with a single maximum, so
    a 1-D golden search is sufficient.

    Args:
        batch (PathBatch): Batch of paths.
        cfg (EnvConfig): Config object.
        iters (int = 50): Number of iterations in the numerical estimation.

    Returns:
        float: Constant allocation that maximized the empirical CE.
    """
    R = batch.episode_returns().double()
    rf = batch.rf
    g = cfg.gamma

    def ce(p: float) -> float:
        """
        Calculates the empirical CE based on the vector of returns in a given batch.

        Args:
            p (float): Fraction of wealth in the risky asset.

        Returns:
            float: Calculated CE value.
        """
        logw = torch.log(torch.clamp(1 + rf + p * (R - rf), min=1e-300)).sum(1)
        if abs(g - 1) < 1e-12:
            return float(logw.mean())
        z = (1 - g) * logw
        lm = float(torch.logsumexp(z, 0) - math.log(z.numel()))
        return lm / (1 - g)
    gr = (math.sqrt(5) - 1) / 2
    a, b = cfg.lo, cfg.hi
    for _ in range(iters):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if ce(c) > ce(d):
            b = d
        else:
            a = c
    return 0.5 * (a + b)
