"""Static figures (matplotlib) for the three protocols.

Colors follow the policy family, never its rank (reference categorical palette, light mode):
SC analytical = blue, RL:DPO = orange, RL:PPO = aqua, RL:TD3 = yellow, DP = violet,
myopic/heuristic SC = magenta, estimated SC = green, oracle/infeasible = red, static = gray.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
           "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
           "gray": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2.0,
})


def policy_color(name: str) -> str:
    n = name.lower()
    if n.startswith("rl:dpo") or "dpo" in n:
        return PALETTE["orange"]
    if "ppo" in n:
        return PALETTE["aqua"]
    if "td3" in n:
        return PALETTE["yellow"]
    if n.startswith("dp:"):
        return PALETTE["violet"]
    if "oracle" in n or "infeasible" in n:
        return PALETTE["red"]
    if "constant" in n or "unconditional" in n or "1/n" in n:
        return PALETTE["gray"]
    if "myopic" in n or "rolling" in n or "plugin" in n or "cert-equiv" in n or "voltiming" in n:
        return PALETTE["magenta"]
    if "fitted" in n or "mle" in n or "calibrated" in n:
        return PALETTE["green"]
    return PALETTE["blue"]


def _finish(fig, title: str) -> None:
    """Suptitle plus tight layout that reserves a fixed ~0.45 in band for the title, so it never
    collides with the panel titles regardless of the figure height."""
    fig.suptitle(title, color=INK, y=1 - 0.12 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def plot_gap_bars(results: Dict[str, Dict[str, dict]], path: str, title: str, cap: float = 150.0) -> None:
    """One panel per case: CE gap to the reference (bp/yr) with +/- 2 SE whiskers."""
    cases = list(results)
    ncol = min(3, len(cases))
    nrow = int(math.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 0.45 * max(len(r) for r in results.values()) * nrow + 1.2 * nrow),
                             squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        res = results[case]
        names = [k for k, v in res.items() if "gap_bp" in v]
        gaps = [res[k]["gap_bp"] for k in names]
        ses = [res[k].get("se_bp", 0.0) for k in names]
        y = np.arange(len(names))
        for i, (n, g, s) in enumerate(zip(names, gaps, ses)):
            if not _finite(g):
                ax.text(0, i, "  bankrupt / -inf", va="center", color=INK2, fontsize=8)
                continue
            gv = min(g, cap)
            ax.barh(i, gv, color=policy_color(n), height=0.6)
            whisk = 2 * s if (_finite(s) and s > 0) else 0.0
            if whisk > 0:
                ax.errorbar(gv, i, xerr=whisk, color=INK2, capsize=2, lw=1)
            ax.text(max(gv, 0) + whisk, i, "  " + f"{g:.1f}" + (" (capped)" if g > cap else ""), va="center",
                    color=INK, fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(case)
        ax.set_xlabel("CE gap to reference (bp / year)")
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    _finish(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(curves: Dict[str, Dict[str, List[dict]]], path: str, title: str) -> None:
    cases = [c for c in curves if curves[c]]
    if not cases:
        return
    ncol = min(3, len(cases))
    nrow = int(math.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        for name, hist in curves[case].items():
            xs = [h["env_steps"] for h in hist if h.get("env_steps", 0) > 0 and _finite(h.get("gap_bp"))]
            ys = [h["gap_bp"] for h in hist if h.get("env_steps", 0) > 0 and _finite(h.get("gap_bp"))]
            if xs:
                ax.plot(xs, ys, color=policy_color(name), label=name, marker="o", ms=3)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=10)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(case)
        ax.set_xlabel("environment steps (transitions)")
        ax.set_ylabel("validation CE gap (bp / year)")
        ax.legend(fontsize=7.5)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    _finish(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_diagnostics(case: str, latent: Optional[np.ndarray], pis: Dict[str, np.ndarray],
                            path: str, state_name: str = "state", n_points: int = 1500) -> None:
    """Left: mean action over time (+/- 1 sd across paths). Right: action vs the true state at
    three decision times (if the model has a state)."""
    T = next(iter(pis.values())).shape[1]
    has_state = latent is not None
    fig, axes = plt.subplots(1, 4 if has_state else 1, figsize=(15 if has_state else 5, 3.3), squeeze=False)
    ax = axes[0, 0]
    t = np.arange(T)
    for name, p in pis.items():
        m, s = p.mean(0), p.std(0)
        c = policy_color(name)
        ax.plot(t, m, color=c, label=name, lw=1.8)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.12, lw=0)
    ax.set_xlabel("decision time t (periods)")
    ax.set_ylabel("risky fraction")
    ax.set_title(f"{case}: mean action ± 1 sd")
    ax.legend(fontsize=7)
    if has_state:
        rng = np.random.default_rng(0)
        idx = rng.choice(latent.shape[0], size=min(n_points, latent.shape[0]), replace=False)
        for ax, tt in zip(axes[0, 1:], (0, T // 2, T - 1)):
            for name, p in pis.items():
                ax.scatter(latent[idx, tt], p[idx, tt], s=4, color=policy_color(name), alpha=0.5,
                           label=name, linewidths=0)
            ax.set_xlabel(f"true {state_name} at t={tt}")
            ax.set_ylabel("risky fraction")
            ax.set_title(f"t = {tt}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_estimation_strip(per_case: Dict[str, Dict[str, List[float]]], path: str, title: str,
                          cap: float = 300.0) -> None:
    """Gap to the true-parameter optimum across training histories (one dot per history)."""
    cases = list(per_case)
    ncol = min(3, len(cases))
    nrow = int(math.ceil(len(cases) / ncol))
    nmax = max(len(v) for v in per_case.values())
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 0.42 * nmax * nrow + 1.3 * nrow), squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        d = per_case[case]
        names = list(d)
        for i, n in enumerate(names):
            vals = np.array([v for v in d[n] if _finite(v)])
            nb = sum(1 for v in d[n] if not _finite(v))
            if vals.size:
                ax.scatter(np.minimum(vals, cap), np.full(vals.size, i), color=policy_color(n), s=18, zorder=3)
                ax.plot([np.median(np.minimum(vals, cap))] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.5)
            if nb:
                ax.text(cap, i, f" {nb}× bankrupt", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"CE gap (bp / yr, capped at {cap:g})")
        ax.set_title(case)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    _finish(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(matrix: np.ndarray, rows: List[str], cols: List[str], path: str, title: str,
                 cap: float = 100.0) -> None:
    """Sequential single-hue heatmap of CE gaps (bp/yr); NaN/inf cells are labelled."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cfe0f6", "#2a78d6", "#0d3a73"])
    M = np.array(matrix, float)
    shown = np.clip(np.where(np.isfinite(M), M, cap), 0, cap)
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 3, 0.42 * len(rows) + 1.6))
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=cap, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            txt = "—" if np.isnan(v) else ("bankr." if np.isinf(v) else f"{v:.0f}")
            dark = np.isinf(v) or (np.isfinite(v) and v > 0.6 * cap)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="#fcfcfb" if dark else INK)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"CE gap to correctly specified SC (bp / yr, capped at {cap:g})")
    ax.set_xlabel("evaluation (true) model")
    ax.set_title(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
