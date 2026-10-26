from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from scrl.configs import ESTIMATION_CASES, ORACLE_CASES, preset
from scrl.protocols import run_estimation, run_misspec, run_oracle, run_real
from scrl.utils import get_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Stochastic control vs reinforcement learning for portfolio optimization")
    p.add_argument("protocol", choices=["oracle", "estimation", "misspec", "real", "all"])
    p.add_argument("--preset", default="quick", choices=["smoke", "quick", "full"])
    p.add_argument("--cases", default=None, help="comma-separated case names "
                   f"(oracle: {','.join(ORACLE_CASES)}; estimation: {','.join(ESTIMATION_CASES)})")
    p.add_argument("--learners", default=None, help="comma-separated subset of dpo,ppo,td3")
    p.add_argument("--reps", type=int, default=None, help="estimation protocol: number of histories")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--years", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--csv", default=None, help="real-data protocol: path to a price CSV")
    p.add_argument("--price-col", default="price")
    p.add_argument("--rf-col", default=None)
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = preset(a.preset)
    over = {"seed": a.seed, "device": str(get_device(a.device))}
    if a.gamma is not None:
        over["gamma"] = a.gamma
    if a.years is not None:
        over["years"] = a.years
    if a.reps is not None:
        over["n_reps"] = a.reps
    if a.learners:
        ls = tuple(x.strip() for x in a.learners.split(","))
        over.update(learners=ls, est_learners=ls, mis_learners=ls)
    cfg = replace(cfg, **over)
    set_seed(cfg.seed)
    cases = a.cases.split(",") if a.cases else None

    if a.protocol in ("oracle", "all"):
        run_oracle(cfg, cases if a.protocol == "oracle" else None, a.out)
    if a.protocol in ("estimation", "all"):
        run_estimation(cfg, cases if a.protocol == "estimation" else None, a.out)
    if a.protocol in ("misspec", "all"):
        run_misspec(cfg, a.out)
    if a.protocol == "real":
        if not a.csv:
            p.error("--csv is required for the real-data protocol")
        run_real(cfg, a.csv, a.price_col, a.rf_col, out_root=a.out)


if __name__ == "__main__":
    main()
