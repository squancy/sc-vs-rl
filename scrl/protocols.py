"""The three comparison protocols of the proposal (plus a real-data hook).

Protocol 1 (oracle):          true parameters -> SC optimum; RL trains on simulated paths.
Protocol 2 (estimation):      both sides see one finite history; SC estimates parameters,
                              RL learns from the same data. Repeated over histories.
Protocol 3 (misspecification): SC/RL trained or calibrated under model A, evaluated under B;
                              plus proportional transaction costs at evaluation time.
Every comparison uses common random numbers (identical test paths for all policies) and the
certainty equivalent (CE) of terminal wealth under CRRA utility.
"""
from __future__ import annotations

import math
import os
import zlib
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import estimation as est
from . import plotting
from .configs import ESTIMATION_CASES, ORACLE_CASES, ExpConfig, model_zoo
from .env import EnvConfig, evaluate_policies, rollout
from .models import (BlockBootstrapSampler, ChackoViceira, FeatureSpec, GBM, Heston, HistorySampler, KimOmberg,
                     Mixture, PathBatch, SimSampler, simulate_history)
from .policies import (BandPolicy, ConstantPolicy, CVPolicy, RollingMertonPolicy, believer_policy,
                       fitted_model_policy, optimal_policy_set)
from .rl.common import Validator
from .rl.dpo import train_dpo
from .rl.ppo import train_ppo
from .rl.td3 import train_td3
from .utils import Timer, fmt_bp, make_run_dir, save_json, to_jsonable


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _env(cfg: ExpConfig, cost: float = 0.0) -> EnvConfig:
    return EnvConfig(gamma=cfg.gamma, lo=cfg.lo, hi=cfg.hi, cost=cost)


def train_learner(key: str, sampler, env: EnvConfig, cfg: ExpConfig, validator: Validator,
                  use_drift: bool = False, name: Optional[str] = None, log: Callable = print, conf=None,
                  seed: Optional[int] = None):
    """Train one learner. ``seed`` fixes network initialization and exploration noise (torch's
    global RNG), so a learner's result does not depend on what was trained before it."""
    if seed is not None:
        torch.manual_seed(seed)
    fn, default = {"dpo": (train_dpo, cfg.dpo), "ppo": (train_ppo, cfg.ppo), "td3": (train_td3, cfg.td3)}[key]
    conf = conf or default
    name = name or f"RL:{key.upper()}"
    timer = Timer()
    pol = fn(sampler, env, conf, validator, use_drift=use_drift, name=name, device=cfg.device, log=log)
    return pol, timer()


def results_table(res: Dict[str, dict], ref: Optional[str]) -> str:
    head = ("| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover "
            "| P5 W_T | bankrupt |\n|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for k, r in res.items():
        ce = r.get("ce_rate", float("nan"))
        ce_s = f"{ce * 100:.3f}%" if isinstance(ce, float) and math.isfinite(ce) else "-inf"
        rows.append(f"| {k}{' (ref)' if k == ref else ''} | {'yes' if r.get('feasible', True) else 'no'} | {ce_s} "
                    f"| {fmt_bp(r.get('gap_bp'), r.get('se_bp'))} | {r.get('pi_rmse', float('nan')):.4f} "
                    f"| {r.get('pi_mean', float('nan')):.3f} | {r.get('turnover', float('nan')):.4f} "
                    f"| {r.get('p05_wT', float('nan')):.3f} | {r.get('n_bankrupt', 0)} |")
    return head + "\n".join(rows) + "\n"


def _seed(cfg: ExpConfig, *parts) -> int:
    """Stable seed derived from the run seed and a label (protocol, case, role, replication), so the
    random numbers of a case do not depend on which other cases are run or on their order."""
    h = zlib.crc32("/".join(str(p) for p in parts).encode())
    return int((h + 1_000_003 * cfg.seed) % (2 ** 31 - 1))


def _sampler(model, cfg: ExpConfig, spec: FeatureSpec, seed: int, dtype=torch.float32, pool=None):
    return SimSampler(model, cfg.dt, cfg.burn, cfg.steps, spec, seed=seed, device=cfg.device,
                      dtype=dtype, pool_size=pool)


def _diag_pis(outs: Dict[str, dict], names: Sequence[str]) -> Dict[str, np.ndarray]:
    return {n: outs[n]["pi"].double().cpu().numpy() for n in names if n in outs}


# ----------------------------------------------------------------------------
# Protocol 1: oracle test
# ----------------------------------------------------------------------------
def run_oracle(cfg: ExpConfig, cases: Optional[Sequence[str]] = None, out_root: str = "results",
               log: Callable = print) -> dict:
    cases = list(cases or ORACLE_CASES)
    out = make_run_dir(out_root, "oracle")
    save_json(cfg, os.path.join(out, "config.json"))
    models = model_zoo()
    summary, curves_all, res_all = {}, {}, {}
    md = ["# Protocol 1 — oracle test\n",
          f"gamma = {cfg.gamma}, horizon = {cfg.years} years, dt = {cfg.dt:.4f}, pre-sample = {cfg.burn} periods, "
          f"bounds = [{cfg.lo}, {cfg.hi}], test paths = {cfg.n_eval} (common random numbers).\n",
          "Gap = (log CE_ref − log CE_policy)/T in basis points per year (positive = worse than the reference), "
          "± delta-method standard error on paired paths.\n"]
    for ci, case in enumerate(cases):
        mkey, info, use_rv = ORACLE_CASES[case]
        m = models[mkey]
        spec = FeatureSpec(info=info, use_rv=use_rv)
        log(f"\n=== [oracle] {case}: {m!r}, info={info}, rv={use_rv}")
        timer = Timer()
        train_s = _sampler(m, cfg, spec, _seed(cfg, "oracle", case, "train"), torch.float32, cfg.pool_size)
        val = _sampler(m, cfg, spec, _seed(cfg, "oracle", case, "val"), torch.float64).sample(cfg.n_val)
        test = _sampler(m, cfg, spec, _seed(cfg, "oracle", case, "test"), torch.float64).sample(cfg.n_eval)
        refb = _sampler(m, cfg, spec, _seed(cfg, "oracle", case, "ref"), torch.float64).sample(cfg.n_ref)
        env = _env(cfg)
        pols, ref = optimal_policy_set(m, info, cfg.gamma, cfg.dt, cfg.burn, cfg.steps, env, ref_batch=refb)
        log(f"  SC benchmarks ready ({timer():.1f}s); reference = {ref}")
        ref_val = rollout(val, pols[ref], env)["wT"]
        curves, times = {}, {}
        for key in cfg.learners:
            v = Validator(val, env, ref_val)
            pol, secs = train_learner(key, train_s, env, cfg, v, log=log, seed=_seed(cfg, "oracle", case, "init", key))
            pols[pol.name], curves[pol.name], times[pol.name] = pol, v.history, secs
            log(f"  trained {pol.name} in {secs:.0f}s")
        res, outs = evaluate_policies(test, pols, env, ref=ref)
        for k, s in times.items():
            res[k]["train_seconds"] = s
        summary[case] = {"model": repr(m), "info": info, "use_rv": use_rv, "reference": ref,
                         "results": res, "curves": curves}
        res_all[case], curves_all[case] = res, curves
        md += [f"\n## {case}\n", f"Model: `{m!r}`, information: `{info}`, reference: `{ref}`.\n\n",
               results_table(res, ref)]
        for k, r in res.items():
            log(f"  {k:34s} CE {r['ce_rate'] * 100:7.3f}%  gap {fmt_bp(r.get('gap_bp'), r.get('se_bp')):>15s} bp/yr"
                f"  π-RMSE {r.get('pi_rmse', float('nan')):.4f}")
        show = [ref] + [k for k in pols if k.startswith("SC:") and k != ref][:2] + list(curves)
        lat = None
        if test.latent is not None and info in ("full", "both", "returns"):
            lat = test.latent[:, cfg.burn:cfg.burn + cfg.steps].double().cpu().numpy()
        plotting.plot_policy_diagnostics(case, lat, _diag_pis(outs, show),
                                         os.path.join(out, f"policy_{case}.png"),
                                         state_name=getattr(m, "latent_name", None) or "state")
        save_json(summary, os.path.join(out, "summary.json"))
    plotting.plot_gap_bars(res_all, os.path.join(out, "gaps.png"), "Protocol 1: CE gap to the SC/DP optimum")
    plotting.plot_learning_curves(curves_all, os.path.join(out, "learning_curves.png"),
                                  "Protocol 1: validation CE gap during training")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("".join(md))
    log(f"\n[oracle] results written to {out}")
    return {"dir": out, "summary": summary}


# ----------------------------------------------------------------------------
# Protocol 2: estimation test
# ----------------------------------------------------------------------------
def _fitted_sc_policies(case: str, m, hist, cfg: ExpConfig, info: str):
    """SC policies built from the observed history only. Returns (policies, fitted model for
    the 'RL on fitted simulator' variant)."""
    r_hist, l_hist, rv_hist = hist
    g, dt, B, T = cfg.gamma, cfg.dt, cfg.burn, cfg.steps
    pols = {}
    gbm_fit = est.fit_gbm(r_hist, dt, m.r)
    mu_e = gbm_fit.mu - gbm_fit.r
    pols["SC:Merton-MLE"] = ConstantPolicy("SC:Merton-MLE", mu_e / (g * gbm_fit.sigma ** 2))
    for L in cfg.rolling_windows:
        n = f"SC:Rolling-Merton({L})"
        pols[n] = RollingMertonPolicy(n, g, L)
    L0 = cfg.rolling_windows[0]
    n = f"SC:Rolling-shrink({L0})"
    pols[n] = RollingMertonPolicy(n, g, L0, prior_mu=mu_e, prior_n=len(r_hist))
    fitted = gbm_fit
    if case == "ko_full":
        fitted = est.fit_ko_full(r_hist, l_hist, dt, m.r)
        pols["SC:KO-fitted(MLE)"] = fitted_model_policy(fitted, "full", g, dt, B, T, "SC:KO-fitted(MLE)")
    elif case == "ko_partial":
        fitted = est.fit_ko_partial(r_hist, dt, m.r, seed=cfg.seed)
        pols["SC:KO-partial-fitted(Kalman MLE)"] = fitted_model_policy(
            fitted, "returns", g, dt, B, T, "SC:KO-partial-fitted(Kalman MLE)")
        fmap = est.fit_ko_partial(r_hist, dt, m.r, seed=cfg.seed, prior=est.SKEPTICAL_PRIOR)
        pols["SC:KO-partial-fitted(Kalman MAP)"] = fitted_model_policy(
            fmap, "returns", g, dt, B, T, "SC:KO-partial-fitted(Kalman MAP)")
    elif case == "heston":
        fitted = est.fit_heston_rv(r_hist, rv_hist, dt, m.r, n_sub=m.n_sub)
        pols["SC:Kraft-fitted(RV moments)"] = fitted_model_policy(
            fitted, "returns", g, dt, B, T, "SC:Kraft-fitted(RV moments)")
        n = "SC:VolTiming(EWMA, const premium)"
        pols[n] = RollingMertonPolicy(n, g, L0, vol_halflife=3.0, fixed_mu=mu_e)
    elif case == "cv_returns":
        fitted = est.fit_cv_rv(r_hist, rv_hist, dt, m.r, n_sub=m.n_sub)
        pols["SC:CV-fitted(RV moments)"] = CVPolicy("SC:CV-fitted(RV moments)", fitted, g, "rv", True, 3.0)
    return pols, fitted


def run_estimation(cfg: ExpConfig, cases: Optional[Sequence[str]] = None, out_root: str = "results",
                   log: Callable = print) -> dict:
    cases = list(cases or ESTIMATION_CASES)
    out = make_run_dir(out_root, "estimation")
    save_json(cfg, os.path.join(out, "config.json"))
    models = model_zoo()
    summary, strip = {}, {}
    md = ["# Protocol 2 — estimation test\n",
          f"Each replication draws one observed history of {cfg.history_periods} periods "
          f"({cfg.history_periods * cfg.dt:.0f} years) from the true model. SC methods estimate parameters "
          "from it; RL methods train on windows of the same history (and, optionally, on a simulator of the "
          "fitted model). No method sees the true parameters. Gaps are to the true-parameter optimum for the "
          "same information set, evaluated on common test paths.\n"]
    for ci, case in enumerate(cases):
        mkey, info, use_rv = ESTIMATION_CASES[case]
        m = models[mkey]
        spec = FeatureSpec(info=info, use_rv=use_rv)
        log(f"\n=== [estimation] {case}: {m!r}, info={info}")
        env = _env(cfg)
        test = _sampler(m, cfg, spec, _seed(cfg, "estimation", case, "test"), torch.float64).sample(cfg.n_eval)
        refb = _sampler(m, cfg, spec, _seed(cfg, "estimation", case, "ref"), torch.float64).sample(cfg.n_ref)
        oracle, ref = optimal_policy_set(m, info, cfg.gamma, cfg.dt, cfg.burn, cfg.steps, env, ref_batch=refb)
        oracle = {k: v for k, v in oracle.items() if k == ref or k.startswith("Oracle")}
        reps = []
        per_policy: Dict[str, List[float]] = {}
        for rep in range(cfg.n_reps):
            hist = simulate_history(m, cfg.history_periods, cfg.dt, seed=_seed(cfg, "estimation", case, "history", rep))
            pols = dict(oracle)
            timer = Timer()
            fitted_pols, fitted = _fitted_sc_policies(case, m, hist, cfg, info)
            pols.update(fitted_pols)
            log(f"  rep {rep}: fitted {fitted!r} ({timer():.1f}s)")
            # RL on the observed history: time-series split. Windows of the first part are used for
            # training, windows of the held-out last part for early stopping / model selection.
            H = len(hist[0])
            n_val_seg = max(int(cfg.val_frac * H), cfg.burn + cfg.steps)
            n_tr_seg = H - n_val_seg
            cut = lambda x, a, b: None if x is None else x[a:b]
            variants = [("", spec)] + ([(", τ-only", FeatureSpec(info="time"))] if cfg.est_tau_only else [])
            for key in cfg.est_learners:
                for tag, sp in variants:
                    if n_tr_seg < cfg.burn + cfg.steps:
                        log("    history too short for a time-series split; skipping RL on history")
                        break
                    tr_args = (hist[0][:n_tr_seg], cut(hist[1], 0, n_tr_seg + 1), m.rf_step(cfg.dt),
                               cfg.dt, cfg.burn, cfg.steps, sp)
                    tr_kw = dict(seed=_seed(cfg, "estimation", case, "rl-train", rep), device=cfg.device, rv=cut(hist[2], 0, n_tr_seg))
                    if cfg.est_rl_data == "bootstrap":
                        hs = BlockBootstrapSampler(*tr_args, block=cfg.est_block, **tr_kw)
                    else:
                        hs = HistorySampler(*tr_args, **tr_kw)
                    hv = HistorySampler(hist[0][n_tr_seg:], cut(hist[1], n_tr_seg, H + 1), m.rf_step(cfg.dt),
                                        cfg.dt, cfg.burn, cfg.steps, sp, device=cfg.device, dtype=torch.float64,
                                        rv=cut(hist[2], n_tr_seg, H))
                    v = Validator(hv.all_windows(), env, None)
                    nm = f"RL:{key.upper()}[history{tag}]"
                    pol, secs = train_learner(key, hs, env, cfg, v, name=nm, log=lambda *a: None,
                                              conf=cfg.est_dpo if key == "dpo" else None,
                                              seed=_seed(cfg, "estimation", case, "init", nm, rep))
                    pols[pol.name] = pol
                    log(f"    trained {pol.name} on {cfg.est_rl_data} paths from {n_tr_seg} periods; "
                        f"{hv.n_windows} held-out validation windows ({secs:.0f}s)")
            if cfg.rl_on_fitted_sim and not isinstance(fitted, Mixture):
                fs = _sampler(fitted, cfg, spec, _seed(cfg, "estimation", case, "fitted-train", rep), torch.float32, cfg.pool_size)
                v = Validator(_sampler(fitted, cfg, spec, _seed(cfg, "estimation", case, "fitted-val", rep), torch.float64).sample(
                    min(cfg.n_val, 4096)), env, None)
                pol, secs = train_learner("dpo", fs, env, cfg, v, name="RL:DPO[fitted-sim]", log=lambda *a: None,
                                         seed=_seed(cfg, "estimation", case, "init", "fitted-sim", rep))
                pols[pol.name] = pol
                log(f"    trained {pol.name} in {secs:.0f}s")
            res, _ = evaluate_policies(test, pols, env, ref=ref)
            reps.append({"fitted_model": repr(fitted), "results": res})
            for k, r in res.items():
                if k == ref:
                    continue
                per_policy.setdefault(k, []).append(r.get("gap_bp", float("nan")))
                log(f"    {k:38s} gap {fmt_bp(r.get('gap_bp'), r.get('se_bp')):>15s} bp/yr")
        agg = {}
        for k, vals in per_policy.items():
            a = np.array([x for x in vals if isinstance(x, float) and math.isfinite(x)])
            agg[k] = {"mean_gap_bp": float(a.mean()) if a.size else float("nan"),
                      "sd_gap_bp": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
                      "median_gap_bp": float(np.median(a)) if a.size else float("nan"),
                      "n_bankrupt_reps": int(len(vals) - a.size), "gaps": vals}
        summary[case] = {"model": repr(m), "info": info, "reference": ref, "aggregate": agg, "reps": reps}
        strip[case] = per_policy
        md += [f"\n## {case}\n", f"True model `{m!r}`, information `{info}`, reference `{ref}`, "
               f"{cfg.n_reps} histories.\n\n| policy | mean gap (bp/yr) | sd across histories | median | bankrupt reps |\n"
               "|---|---|---|---|---|\n"]
        for k, a in sorted(agg.items(), key=lambda kv: kv[1]["median_gap_bp"]):
            md.append(f"| {k} | {a['mean_gap_bp']:.1f} | {a['sd_gap_bp']:.1f} | {a['median_gap_bp']:.1f} "
                      f"| {a['n_bankrupt_reps']} |\n")
        save_json(summary, os.path.join(out, "summary.json"))
    plotting.plot_estimation_strip(strip, os.path.join(out, "estimation_gaps.png"),
                                   "Protocol 2: gap to the true-parameter optimum across observed histories")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("".join(md))
    log(f"\n[estimation] results written to {out}")
    return {"dir": out, "summary": summary}


# ----------------------------------------------------------------------------
# Protocol 3: misspecification test
# ----------------------------------------------------------------------------
def run_misspec(cfg: ExpConfig, out_root: str = "results", log: Callable = print) -> dict:
    out = make_run_dir(out_root, "misspec")
    save_json(cfg, os.path.join(out, "config.json"))
    models = model_zoo()
    spec = FeatureSpec(info="returns", use_rv=False)     # common information set across all models
    g, dt, B, T = cfg.gamma, cfg.dt, cfg.burn, cfg.steps
    env = _env(cfg)
    log("\n=== [misspec] information set: returns only (no realized variance)")
    tests = {em: _sampler(models[em], cfg, spec, _seed(cfg, "misspec", em, "test"), torch.float64).sample(cfg.n_eval)
             for i, em in enumerate(cfg.eval_models)}
    believers = {}
    for tm in list(cfg.train_models) + [em for em in cfg.eval_models if em not in cfg.train_models]:
        if tm == "mix":
            continue
        believers[tm] = believer_policy(models[tm], g, dt, B, T, name=f"SC[{tm}]")
    learned = {}
    for i, tm in enumerate(cfg.train_models):
        ts = _sampler(models[tm], cfg, spec, _seed(cfg, "misspec", tm, "train"), torch.float32, cfg.pool_size)
        val = _sampler(models[tm], cfg, spec, _seed(cfg, "misspec", tm, "val"), torch.float64).sample(cfg.n_val)
        ref_val = rollout(val, believers[tm], env)["wT"] if tm in believers else None
        for key in cfg.mis_learners:
            v = Validator(val, env, ref_val)
            pol, secs = train_learner(key, ts, env, cfg, v, name=f"RL:{key.upper()}[{tm}]", log=lambda *a: None,
                                      seed=_seed(cfg, "misspec", tm, "init", key))
            learned[pol.name] = pol
            log(f"  trained {pol.name} in {secs:.0f}s")
    row_names = [f"SC[{tm}]" for tm in cfg.train_models if tm != "mix"] + list(learned)
    M = np.full((len(row_names), len(cfg.eval_models)), np.nan)
    eval_res = {}
    for j, em in enumerate(cfg.eval_models):
        pols = {f"SC[{tm}]": believers[tm] for tm in cfg.train_models if tm != "mix"}
        pols[f"SC[{em}]"] = believers[em]
        pols.update(learned)
        ref = f"SC[{em}]"
        res, _ = evaluate_policies(tests[em], pols, env, ref=ref)
        eval_res[em] = {"reference": ref, "model": repr(models[em]), "results": res}
        for i, rn in enumerate(row_names):
            if rn in res:
                M[i, j] = res[rn].get("gap_bp", np.nan)
        log(f"  eval on {em}: " + ", ".join(f"{k}={fmt_bp(r.get('gap_bp'))}" for k, r in res.items()))
    plotting.plot_heatmap(M, row_names, list(cfg.eval_models), os.path.join(out, "misspec_heatmap.png"),
                          "Protocol 3: train/calibrate on row model, evaluate on column model")
    # ---- transaction costs at evaluation time
    cost_res = {}
    for c in cfg.costs:
        envc = _env(cfg, c)
        for i, em in enumerate(cfg.cost_models):
            base = believers[em]
            band = BandPolicy(believer_policy(models[em], g, dt, B, T, name=f"SC[{em}]"), g, c,
                              name=f"SC[{em}]+band")
            pols = {f"SC[{em}] (ignores costs)": base, band.name: band}
            for key in cfg.mis_learners:
                fr = f"RL:{key.upper()}[{em}]"
                if fr in learned:
                    pols[fr + " (trained w/o costs)"] = learned[fr]
            ts = _sampler(models[em], cfg, spec, _seed(cfg, "misspec", em, "cost-train"), torch.float32, cfg.pool_size)
            val = _sampler(models[em], cfg, spec, _seed(cfg, "misspec", em, "cost-val"), torch.float64).sample(cfg.n_val)
            v = Validator(val, envc, rollout(val, band, envc)["wT"])
            pol, secs = train_learner("dpo", ts, envc, cfg, v, use_drift=True,
                                      name=f"RL:DPO[{em}, cost={c:g}]", log=lambda *a: None,
                                      seed=_seed(cfg, "misspec", em, "init", "cost", c))
            pols[pol.name] = pol
            res, _ = evaluate_policies(tests[em] if em in tests else
                                       _sampler(models[em], cfg, spec, _seed(cfg, "misspec", em, "test"), torch.float64).sample(cfg.n_eval),
                                       pols, envc, ref=band.name)
            cost_res[f"{em}@{c:g}"] = {"reference": band.name, "results": res}
            log(f"  costs {c:g} on {em}: " + ", ".join(f"{k}={fmt_bp(r.get('gap_bp'), r.get('se_bp'))}"
                                                     for k, r in res.items()))
    summary = {"matrix": {"rows": row_names, "cols": list(cfg.eval_models), "gap_bp": M},
               "eval": eval_res, "costs": cost_res}
    save_json(summary, os.path.join(out, "summary.json"))
    md = ["# Protocol 3 — misspecification test\n",
          "Information set: returns only. Rows are policies trained/calibrated on one model (SC[x] is the "
          "optimal returns-only SC policy of an investor who believes model x with its true parameters; "
          "RL[x] is trained on simulated paths of x; RL[mix] on an equal mixture of the five base models). "
          "Columns are the true evaluation model. Entries: CE gap (bp/yr) to the correctly specified SC policy "
          "of the column model.\n\n",
          "| policy | " + " | ".join(cfg.eval_models) + " |\n|---|" + "---|" * len(cfg.eval_models) + "\n"]
    for i, rn in enumerate(row_names):
        md.append(f"| {rn} | " + " | ".join("-" if np.isnan(x) else ("bankrupt" if np.isinf(x) else f"{x:.1f}")
                                           for x in M[i]) + " |\n")
    md.append("\n## Proportional transaction costs at evaluation time\n")
    for k, v in cost_res.items():
        md += [f"\n### {k}\n\n", results_table(v["results"], v["reference"])]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("".join(md))
    log(f"\n[misspec] results written to {out}")
    return {"dir": out, "summary": summary}


# ----------------------------------------------------------------------------
# Real data hook (Protocol 2 on an observed price series)
# ----------------------------------------------------------------------------
def load_monthly_csv(path: str, price_col: str = "price", rf_col: Optional[str] = None,
                     rf_annual: float = 0.02):
    """CSV with one row per period (chronological), a total-return price column and optionally a
    per-period simple risk-free return column. Returns (returns, rf_per_period)."""
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    p = np.array([float(r[price_col]) for r in rows])
    rets = p[1:] / p[:-1] - 1.0
    rf = float(np.mean([float(r[rf_col]) for r in rows[1:]])) if rf_col else None
    return rets, rf


def run_real(cfg: ExpConfig, csv_path: str, price_col: str = "price", rf_col: Optional[str] = None,
             train_frac: float = 0.6, out_root: str = "results", log: Callable = print) -> dict:
    """Train/calibrate on the first part of an observed series; evaluate the realized CE over all
    (overlapping) evaluation windows of length burn+T in the remaining part. A constant risk-free
    rate equal to the sample mean is assumed. Windows overlap, so SEs are not reported."""
    out = make_run_dir(out_root, "real")
    rets, rf_step = load_monthly_csv(csv_path, price_col, rf_col)
    if rf_step is None:
        rf_step = math.exp(0.02 * cfg.dt) - 1
    r_cont = math.log1p(rf_step) / cfg.dt
    n_tr = int(len(rets) * train_frac)
    tr, te = rets[:n_tr], rets[n_tr:]
    spec = FeatureSpec(info="returns")
    env = _env(cfg)
    L = cfg.burn + cfg.steps
    if len(te) < L or len(tr) < L:
        raise ValueError("series too short for the chosen burn-in and horizon")
    starts = np.arange(0, len(te) - L + 1)
    R = torch.as_tensor(te[starts[:, None] + np.arange(L)[None, :]], dtype=torch.float64)
    from .models import build_features
    test = PathBatch(R, None, rf_step, cfg.dt, cfg.burn, cfg.steps,
                     build_features(R, None, rf_step, cfg.dt, cfg.burn, cfg.steps, spec), "real")
    gbm_fit = est.fit_gbm(tr, cfg.dt, r_cont)
    mu_e = gbm_fit.mu - gbm_fit.r
    pols = {"SC:Merton-MLE": ConstantPolicy("SC:Merton-MLE", mu_e / (cfg.gamma * gbm_fit.sigma ** 2))}
    for W in cfg.rolling_windows:
        pols[f"SC:Rolling-Merton({W})"] = RollingMertonPolicy(f"SC:Rolling-Merton({W})", cfg.gamma, W)
    kof = est.fit_ko_partial(tr, cfg.dt, r_cont, seed=cfg.seed)
    pols["SC:KO-partial-fitted(Kalman MLE)"] = fitted_model_policy(
        kof, "returns", cfg.gamma, cfg.dt, cfg.burn, cfg.steps, "SC:KO-partial-fitted(Kalman MLE)")
    kom = est.fit_ko_partial(tr, cfg.dt, r_cont, seed=cfg.seed, prior=est.SKEPTICAL_PRIOR)
    pols["SC:KO-partial-fitted(Kalman MAP)"] = fitted_model_policy(
        kom, "returns", cfg.gamma, cfg.dt, cfg.burn, cfg.steps, "SC:KO-partial-fitted(Kalman MAP)")
    pols["Cash"] = ConstantPolicy("Cash", 0.0)
    # same recipe as Protocol 2: block-bootstrap training paths from the early part of the training
    # sample, early stopping on held-out windows from its last part
    n_val_seg = max(int(cfg.val_frac * len(tr)), L)
    tr_fit, tr_val = tr[:len(tr) - n_val_seg], tr[len(tr) - n_val_seg:]
    if len(tr_fit) >= L:
        Smp = BlockBootstrapSampler if cfg.est_rl_data == "bootstrap" else HistorySampler
        kw = {"block": cfg.est_block} if cfg.est_rl_data == "bootstrap" else {}
        hs = Smp(tr_fit, None, rf_step, cfg.dt, cfg.burn, cfg.steps, spec, seed=cfg.seed, device=cfg.device, **kw)
        hv = HistorySampler(tr_val, None, rf_step, cfg.dt, cfg.burn, cfg.steps, spec, device=cfg.device,
                            dtype=torch.float64)
        for key in cfg.est_learners:
            v = Validator(hv.all_windows(), env, None)
            pol, _ = train_learner(key, hs, env, cfg, v, name=f"RL:{key.upper()}[history]", log=log,
                                   seed=_seed(cfg, "real", "init", key),
                                   conf=cfg.est_dpo if key == "dpo" else None)
            pols[pol.name] = pol
    else:
        log("[real] training sample too short for a held-out validation split; RL skipped")
    res, _ = evaluate_policies(test, pols, env, ref="SC:Merton-MLE")
    save_json({"n_windows": int(len(starts)), "fitted_ko": repr(kof), "fitted_ko_map": repr(kom), "results": res},
              os.path.join(out, "summary.json"))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"# Real data ({csv_path})\n\n{len(starts)} overlapping evaluation windows; "
                "standard errors are not meaningful because windows overlap.\n\n" + results_table(res, "SC:Merton-MLE"))
    log(f"[real] results written to {out}")
    return {"dir": out, "results": res}
