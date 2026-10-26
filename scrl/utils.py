from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Reproducibility / devices
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Seeds the standard Python random library, NumPy and PyTorch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(pref: str = "auto") -> torch.device:
    """
    Creates a PyTorch device based on preference.
    If ``pref`` is ``"auto"``, it will select a CUDA
    device if available and CPU otherwise.

    Args:
        pref (str = "auto"): Device preference.

    Returns:
        torch.device: Device with the given preference.
    """
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class Timer:
    """
    Measures time between now and a past timestamp.

    Attributes:
        t0 (float): Past timestamp.
    """
    def __init__(self) -> None:
        self.t0 = time.time()

    def __call__(self) -> float:
        """
        Returns the elapsed time between the past timestamp and now.

        Returns:
            float: Elapsed time.
        """
        return time.time() - self.t0


# ----------------------------------------------------------------------------
# CRRA utility
# ----------------------------------------------------------------------------
def crra_utility(w: torch.Tensor | np.array, gamma: float):
    """
    CRRA utility U(w) = w ^ (1 - gamma) / (1 - gamma), log(w) for gamma == 1.
    
    Args:
        w (torch.Tensor | np.array): Input to CRRA.
        gamma (float): Risk-aversion rate.

    Returns:
        float: CRRA of ``w``.
    """
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(w) if isinstance(w, torch.Tensor) else np.log(w)
    return w ** (1.0 - gamma) / (1.0 - gamma)


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def log_mean_y(log_w: np.ndarray, gamma: float) -> float:
    """
    Numerically stable log E[W^(1-gamma)] computed from log weights.

    Args:
        log_w (np.ndarray): Log weights.
        gamma (float): Risk-aversion rate.

    Returns:
        float: Log mean.
    """
    z = (1.0 - gamma) * log_w
    zmax = np.max(z)
    return float(zmax + np.log(np.mean(np.exp(z - zmax))))


def ce_stats(w_T: torch.Tensor | np.array, gamma: float, horizon_years: float, w0: float = 1.0) -> dict[str, float]:
    """
    Certainty equivalent of terminal wealth and its annualized log rate.

    CE = U^{-1}(E[U(W_T)]). If any path is bankrupt (W_T <= 0) and gamma >= 1,
    expected utility is -inf and CE = 0.

    Args:
        w_T (torch.Tensor | np.array): Terminal wealths.
        gamma (float): Risk-aversion rate.
        horizon_years (float): Number of years in the rollout.
        w_0 (float = 1.0): Initial wealth.

    Returns:
        dict[str, float]: Various statistics based on ``w``.
    """
    w = _to_np(w_T)
    n = w.size
    n_bankrupt = int(np.sum(w <= 0))
    out: dict[str, float] = {
        "n_paths": n,
        "n_bankrupt": n_bankrupt,
        "mean_wT": float(np.mean(w)),
        "median_wT": float(np.median(w)),
        "std_wT": float(np.std(w)),
        "p05_wT": float(np.quantile(w, 0.05)),
        "p95_wT": float(np.quantile(w, 0.95)),
    }
    if n_bankrupt > 0 and gamma >= 1.0:
        out.update(ce=0.0, log_ce=-math.inf, ce_rate=-math.inf)
        return out
    log_w = np.log(np.maximum(w, 1e-300))
    if abs(gamma - 1.0) < 1e-12:
        log_ce = float(np.mean(log_w))
    else:
        log_ce = log_mean_y(log_w, gamma) / (1.0 - gamma)
    out["log_ce"] = log_ce
    out["ce"] = float(math.exp(log_ce))
    out["ce_rate"] = (log_ce - math.log(w0)) / horizon_years
    return out


def paired_ce_gap(w_ref: torch.Tensor | np.array, w_pol: torch.Tensor | np.array, gamma: float, horizon_years: float) -> dict[str, float]:
    """
    Annualized CE gap (reference minus policy) in basis points per year, with a
    delta-method standard error that exploits common random numbers (paired paths).

    gap = (log CE_ref - log CE_pol) / T * 1e4. Positive means the policy is worse.

    Args:
        w_ref (torch.Tensor | np.array): Terminal wealths using the reference policy. 
        w_pol (torch.Tensor | np.array): Terminal wealths using the current policy.
        gamma (float): Risk-aversion rate.
        horizon_years (float): Number of years in the rollout (T).
    
    Returns:
        dict[str, float]: CE gap, standard error in basis points per year and
        relative loss.
    """
    a = _to_np(w_ref)
    b = _to_np(w_pol)
    if np.any(a <= 0) or np.any(b <= 0):
        if np.any(b <= 0) and gamma >= 1.0:
            return {"gap_bp": math.inf, "se_bp": math.nan, "rel_loss": 1.0}
        if np.any(a <= 0) and gamma >= 1.0:
            return {"gap_bp": -math.inf, "se_bp": math.nan, "rel_loss": -math.inf}
    la, lb = np.log(a), np.log(b)
    n = a.size
    if abs(gamma - 1.0) < 1e-12:
        d = la - lb
        diff = float(np.mean(d))
        se = float(np.std(d, ddof=1) / math.sqrt(n))
    else:
        lma = log_mean_y(la, gamma)
        lmb = log_mean_y(lb, gamma)
        ya = np.exp((1.0 - gamma) * la - lma)  # Y_a / mean(Y_a)
        yb = np.exp((1.0 - gamma) * lb - lmb)
        diff = (lma - lmb) / (1.0 - gamma)  # log CE_a - log CE_b
        se = float(np.std(ya - yb, ddof=1) / math.sqrt(n) / abs(1.0 - gamma))
    return {
        "gap_bp": diff / horizon_years * 1e4,
        "se_bp": se / horizon_years * 1e4,
        "rel_loss": 1.0 - math.exp(-diff),
    }


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    return obj


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def make_run_dir(root: str, name: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, f"{name}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def fmt_bp(x: float | None, se: float | None = None) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "bankrupt" if x > 0 else "-inf"
    if se is None or (isinstance(se, float) and math.isnan(se)):
        return f"{x:.1f}"
    return f"{x:.1f} ± {se:.1f}"
