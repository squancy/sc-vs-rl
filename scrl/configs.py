from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, tuple

from .models import GBM, ChackoViceira, Heston, KimOmberg, MarketModel, MertonJump, Mixture
from .rl.dpo import DPOConfig
from .rl.ppo import PPOConfig
from .rl.td3 import TD3Config


def jump_matched(r: float = 0.02, mu: float = 0.08, total_var: float = 0.04,
                 lam: float = 0.5, muJ: float = -0.2, sigJ: float = 0.1) -> MertonJump:
    """
    Merton jump-diffusion whose total return variance equals ``total_var``.
    """
    m = MertonJump(r=r, mu=mu, sigma=0.1, lam=lam, muJ=muJ, sigJ=sigJ)
    m.sigma = math.sqrt(total_var - lam * m.jump_second_moment)
    return m


def model_zoo(r: float = 0.02) -> dict[str, MarketModel]:
    """
    Initializes base models.

    **Note**: All base models share r = 2%, an expected excess return of 6%/yr and a return volatility
    of about 20%/yr, so that a myopic Merton investor with gamma = 5 holds about 30% in every
    model and differences between models come from the dynamics only.

    Args:
        r (float = 0.02): Return of the risk-free asset.
    
    Returns:
        dict[str, MarketModel]: Dictionary of base models.
    """
    base = {
        "gbm": GBM(r=r, mu=0.08, sigma=0.20),
        "jump": jump_matched(r),
        "heston": Heston(r=r, xi=1.5, kappa=2.0, vbar=0.04, sigma_v=0.4, rho=-0.7, n_sub=8),
        "cv": ChackoViceira(r=r, mu_e=0.06, kappa=1.0, theta=30.0, sigma_y=math.sqrt(10.0), rho=0.5, n_sub=21),
        "ko": KimOmberg(r=r, sigma=0.20, lam=0.2, Xbar=0.3, sigma_X=0.2 * math.sqrt(0.4), rho=-0.9),
    }
    # regime-shift variants used in the misspecification protocol
    base["gbm_highvol"] = GBM(r=r, mu=0.08, sigma=0.30)
    base["gbm_lowprem"] = GBM(r=r, mu=0.05, sigma=0.20)
    base["ko_fast"] = KimOmberg(r=r, sigma=0.20, lam=1.0, Xbar=0.3, sigma_X=0.2 * math.sqrt(2.0), rho=-0.9)
    base["mix"] = Mixture([base[k] for k in ("gbm", "jump", "heston", "cv", "ko")])
    return base


# Oracle-test cases: (model key, information set, use realized variance in RL features)
ORACLE_CASES: dict[str, tuple[str, str, bool]] = {
    "gbm": ("gbm", "returns", False),
    "jump": ("jump", "returns", False),
    "heston": ("heston", "full", False),
    "cv_full": ("cv", "full", False),
    "cv_returns": ("cv", "returns", True),
    "ko_full": ("ko", "full", False),
    "ko_partial": ("ko", "returns", False),
}

# Estimation-test cases
ESTIMATION_CASES: dict[str, tuple[str, str, bool]] = {
    "gbm": ("gbm", "returns", False),
    "ko_full": ("ko", "full", False),
    "ko_partial": ("ko", "returns", False),
    "heston": ("heston", "returns", True),
    "cv_returns": ("cv", "returns", True),
}


@dataclass
class ExpConfig:
    gamma: float = 5.0
    dt: float = 1.0 / 12.0
    years: float = 10.0
    burn: int = 60
    lo: float = -1.0
    hi: float = 2.0
    n_eval: int = 20000
    n_val: int = 8192
    n_ref: int = 8192                 # batch used to fit "Best-constant" and to tune plug-ins
    pool_size: int | None = 50000  # simulated training paths (None = fresh paths each batch)
    seed: int = 0
    device: str = "cpu"
    learners: tuple[str, ...] = ("dpo", "ppo", "td3")
    dpo: DPOConfig = field(default_factory=DPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    # estimation protocol
    history_periods: int = 600        # length of the observed history (50 years monthly)
    n_reps: int = 3
    est_learners: tuple[str, ...] = ("dpo",)
    # RL trained on one history: time-series split (last val_frac of the history is held out for
    # early stopping / model selection) and a small, regularized network to limit memorization.
    val_frac: float = 0.4
    est_dpo: DPOConfig = field(default_factory=lambda: DPOConfig(
        iters=300, batch=256, hidden=(16, 16), weight_decay=1e-4, eval_every=10))
    est_tau_only: bool = True         # also train a tau-only policy (same information as Merton-MLE)
    # Training data for RL on the history. 'bootstrap': moving-block bootstrap paths built from the
    # training segment (default). 'windows': the overlapping windows themselves - these contain only
    # a few independent long-horizon outcomes, so the in-sample CRRA objective badly underestimates
    # terminal-wealth risk and the learned policy is far too aggressive.
    est_rl_data: str = "bootstrap"
    est_block: int = 24
    rolling_windows: tuple[int, ...] = (60, 120)
    rl_on_fitted_sim: bool = True
    # misspecification protocol
    train_models: tuple[str, ...] = ("gbm", "jump", "heston", "cv", "ko", "mix")
    eval_models: tuple[str, ...] = ("gbm", "jump", "heston", "cv", "ko", "gbm_highvol", "ko_fast")
    mis_learners: tuple[str, ...] = ("dpo",)
    costs: tuple[float, ...] = (0.005,)
    cost_models: tuple[str, ...] = ("gbm", "ko")

    @property
    def steps(self) -> int:
        return round(self.years / self.dt)


def preset(name: str) -> ExpConfig:
    """
    Presets configs based on the run type (test, quick run, full run).

    Args:
        name (str): One of "smoke", "quick", "full".

    Returns:
        ExpConfig: Config based on the run type.
    """
    if name == "smoke":   # seconds; used by the tests
        return ExpConfig(years=2.0, burn=24, n_eval=2000, n_val=1024, n_ref=1024, pool_size=4000,
                         dpo=DPOConfig(iters=20, batch=256, eval_every=10),
                         ppo=PPOConfig(iters=3, n_envs=128, eval_every=1, minibatch=1024),
                         td3=TD3Config(iters=3, n_envs=64, eval_every=1, update_every=64, warmup_iters=1),
                         history_periods=240, n_reps=1, rolling_windows=(24,), val_frac=0.5,
                         est_dpo=DPOConfig(iters=20, batch=128, hidden=(16, 16), eval_every=10),
                         train_models=("gbm", "ko", "mix"), eval_models=("gbm", "ko"), costs=(0.005,),
                         cost_models=("gbm",))
    if name == "quick":   # tens of minutes on a laptop CPU
        return ExpConfig(dpo=DPOConfig(iters=400, batch=512, eval_every=50),
                         ppo=PPOConfig(iters=120, n_envs=256, eval_every=10, minibatch=4096),
                         td3=TD3Config(iters=30, n_envs=128, eval_every=5))
    if name == "full":    # hours; a GPU helps for the learners
        return ExpConfig(n_eval=100000, n_val=16384, n_ref=20000, pool_size=500000,
                         dpo=DPOConfig(iters=4000, batch=2048, eval_every=100),
                         ppo=PPOConfig(iters=1500, n_envs=1024, eval_every=25, minibatch=16384),
                         td3=TD3Config(iters=400, n_envs=256, eval_every=20),
                         n_reps=20, est_learners=("dpo", "ppo", "td3"),
                         mis_learners=("dpo", "ppo", "td3"))
    raise ValueError(f"unknown preset '{name}'")
