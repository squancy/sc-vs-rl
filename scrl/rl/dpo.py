"""
Direct policy optimization with pathwise gradients (DPO) (Han & E 2016; Buehler et al. 2019, "deep hedging").

Because the investor's actions do not affect prices (we assume this), terminal wealth is a differentiable
function of the actions given a realized return path:
    W_T = prod_t (1 - c|pi_t - d_t|) (1 + r_f + pi_t (R_{t+1} - r_f)).

So, for a sampled return path omega, the gradient of the loss is:
    grad_theta J(theta) = grad_theta E[U(W_T(theta; omega))] = E[grad_theta U(W_T(theta; omega))]
where theta is the parameter set of the policy pi.

The loss is log mean(W_T ^ (1 - gamma)) / (gamma - 1) / years, i.e., -log CE per year.
For gamma > 1 (we usually use gamma = 5), this is the same as maximizing E[U(W_T)].

This loss has a few advantages compared to E[U(W_T)]:
    - if we take a batch X, then grad log mean(X) = grad mean(X) / mean(X), so each
    batch is scaled by a batch-dependent factor, making the loss scale-free. It
    can be beneficial when the gradient would otherwise be very small (learning would be very slow).
    - grad E[U(W_T)] = E[W_T ^ {-gamma} grad W_T], so each path is weighted by W_T ^ {-gamma}.
    It means that bad paths (where W_T is low) will dominate the loss, which is reasonable.

The final policy is deterministic and depends on the features and drift weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..env import EnvConfig
from ..models import Sampler
from .common import Actor, RLPolicy, Validator


@dataclass
class DPOConfig:
    """
    Attributes:
        iters (int): Number of iterations during training.
        batch (int): Batch size.
        lr (float): Initial learning rate.
        lr (float): Final learning rate (decaying).
        hidden (tuple[int, ...]): Hidden layer dimensions in the neural network.
        grad_clip (float): Clip gradient if it's too large.
        weight_decay (float): Weight decay regularization amount.
        eval_every (int): Evaluation at every n steps.
        w_floor (float): Minimum wealth value.
        pi_init (float): Initial value for the policy.
    """
    iters: int = 1500
    batch: int = 1024
    lr: float = 1e-3
    lr_final: float = 1e-4
    hidden: tuple[int, ...] = (64, 64)
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    eval_every: int = 50
    w_floor: float = 1e-4
    pi_init: float = 0.0


def dpo_loss(wT: torch.Tensor, gamma: float, years: float, floor: float) -> torch.Tensor:
    """
    Numerically safe implementation of -log CE.

    Args:
        wT (torch.Tensor): Vector of final wealths.
        gamma (float): Risk-aversion rate.
        years (float): Number of years in the rollout.
        floor (float): Minimum wealth allowed (for numerical stability).

    Returns:
        torch.Tensor: -log CE.
    """
    w = torch.clamp(wT, min=floor)
    if abs(gamma - 1) < 1e-12:
        return -torch.log(w).mean() / years
    z = (1 - gamma) * torch.log(w)
    lm = torch.logsumexp(z, 0) - math.log(z.numel())
    return lm / (gamma - 1) / years


def train_dpo(sampler: Sampler, env: EnvConfig, cfg: DPOConfig, validator: Validator,
              use_drift: bool = False, name: str = "RL:DPO", device:str="cpu", log=print) -> RLPolicy:
    """
    DPO training.

    Args:
        sampler (Sampler): Sampler object used for sampling trajectories.
        env (EnvConfig): Environment configurations.
        cfg (DPOConfig): DPO-specific configs.
        validator (Validator): Validator object for selecting the best-performing policy on a held-out validation
        set to avoid overfitting.
        use_drift (bool = False): True, if the actor output should depend on the drifted weights.
        name (str = "RL:DPO"): Name of the algorithm as seen in plots.
        device (str = "cpu"): Device on which the training should run.
        log (=print): Logging function.
    
    Returns:
        RLPolicy: Trained DPO policy.
    """
    probe = sampler.sample(min(4096, max(cfg.batch, 1024)))
    actor = Actor(probe.feats.shape[-1], env.lo, env.hi, cfg.hidden, use_drift, cfg.pi_init).to(device)
    actor.norm.fit(probe.feats.to(device))
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gamma_lr = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.iters, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma_lr)
    steps_seen = 0
    for it in range(cfg.iters + 1):
        if it % cfg.eval_every == 0 or it == cfg.iters:
            validator(actor, steps_seen, name)
            h = validator.history[-1] if validator.history else {}
            log(f"  [{name}] it {it:5d}  env_steps {steps_seen:9d}  val CE {h.get('ce_rate', float('nan'))*100:.3f}%"
                f"  gap {h.get('gap_bp', float('nan')):7.1f} bp/yr")
        if it == cfg.iters:
            break
        b = sampler.sample(cfg.batch).to(device=device, dtype=torch.float32)
        R = b.episode_returns()
        w = torch.ones(b.n, device=device)
        d = torch.zeros(b.n, device=device)
        for t in range(b.horizon):
            pi = actor(b.feats[:, t], d)
            gross = 1.0 + b.rf + pi * (R[:, t] - b.rf)
            w = w * gross
            if env.cost > 0:
                w = w * (1.0 - env.cost * torch.abs(pi - d))
            d = pi * (1.0 + R[:, t]) / gross
        loss = dpo_loss(w, env.gamma, b.years, cfg.w_floor)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        steps_seen += b.n * b.horizon
    validator.restore_best(actor)
    actor.eval()
    return RLPolicy(name, actor, getattr(sampler, "spec", None))
