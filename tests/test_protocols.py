"""End-to-end smoke tests: every protocol runs and produces its outputs (tiny budgets)."""
import os
from dataclasses import replace

from scrl.configs import preset
from scrl.protocols import run_estimation, run_misspec, run_oracle


def _quiet(*a, **k):
    pass


def test_oracle_smoke(tmp_path):
    cfg = preset("smoke")
    r = run_oracle(cfg, ["gbm", "ko_partial"], str(tmp_path), log=_quiet)
    for f in ("summary.json", "summary.md", "gaps.png", "learning_curves.png"):
        assert os.path.exists(os.path.join(r["dir"], f))
    res = r["summary"]["ko_partial"]["results"]
    assert {"RL:DPO", "RL:PPO", "RL:TD3", "DP:partial-info", "SC:KO-partial"} <= set(res)
    assert abs(res["SC:KO-partial"]["gap_bp"]) < 3.0       # analytic vs exact discrete optimum


def test_estimation_smoke(tmp_path):
    cfg = replace(preset("smoke"), rl_on_fitted_sim=False)
    r = run_estimation(cfg, ["gbm"], str(tmp_path), log=_quiet)
    assert "SC:Merton-MLE" in r["summary"]["gbm"]["aggregate"]


def test_misspec_smoke(tmp_path):
    cfg = replace(preset("smoke"), train_models=("gbm", "ko"), eval_models=("gbm", "ko"))
    r = run_misspec(cfg, str(tmp_path), log=_quiet)
    assert os.path.exists(os.path.join(r["dir"], "misspec_heatmap.png"))
