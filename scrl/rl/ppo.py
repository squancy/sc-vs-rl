"""
Proximal policy optimization (Schulman et al. 2017) with GAE (Schulman et al. 2016).

The episode objective is E[U(W_T)], as for the SC benchmarks:
    - ``wealth_normalized=False``: telescoping reward r_t = U(W_{t+1}) - U(W_t), no discounting,
    critic V(features, log W). The return of an episode is U(W_T) - U(W_0).
    - ``wealth_normalized=True`` (default): the exact wealth-free Bellman equation implied by CRRA
    homogeneity (see common.normalized_reward): reward r~_t, random multiplicative discount
    beta_t = G_t^(1-gamma), critic V(features). GAE becomes
        delta_t = r~_t + beta_t V(s_{t+1}) - V(s_t),   A_t = delta_t + lambda beta_t A_{t+1}.
    This weights decision states by W_t^(1-gamma)-free advantages. Since the optimal action is
    wealth-independent the optimal policy is unchanged, but the targets stay bounded.
A per-step reward such as U(W_{t+1}/W_t) without the beta factor would define a myopic
objective whose optimizer ignores hedging.

The Gaussian policy has a squashed mean in [lo, hi] and a state-independent log std;
applied actions are clipped to [lo, hi].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..env import EnvConfig
from ..models import Sampler
from .common import Actor, Critic, RLPolicy, Validator, crra_u_safe, normalized_reward


@dataclass
class PPOConfig:
    """
    Hyperparameters for PPO.

    Attributes:
        iters (int): Total number of iterations in PPO.
        n_envs (int): Number of sampled episodes (rollouts).
        epochs (int): Number of epochs during training.
        minibatch (int): Minibatch size.
        lr (float): Learning rate.
        clip (float): Clip gradient between 1 - clip and 1 + clip.
        gae_lambda (float): Coefficient in GAE.
        vf_coef (float): Critic-value MSE coefficient in the loss function.
        ent_coef (float): Entropy coefficient in the loss function
            based on the volatility of the action.
        init_std (float): Initial volatility of the action.
        min_std (float): Minimum action volatility allowed.
        max_grad_norm (float): Maximum gradient norm allowed.
        reward_scale (float): Scales rewards so that too small/large
        gradients will not cause learning problems.
        hidden (tuple[int, ...]): Hidden layer structure of the policy.
        eval_every (int): Evaluate and validate at every n iterations.
        pi_init (float): Initial policy value.
        wealth_normalized (bool): True, if ``common.normalized_reward`` should be used
            for rewards.
    """
    iters: int = 300
    n_envs: int = 512
    epochs: int = 8
    minibatch: int = 4096
    lr: float = 3e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    init_std: float = 0.2
    min_std: float = 0.01
    max_grad_norm: float = 0.5
    reward_scale: float = 10.0
    hidden: tuple[int, ...] = (64, 64)
    eval_every: int = 10
    pi_init: float = 0.0
    wealth_normalized: bool = True


def _logp(a, mean, std):
    return -0.5 * ((a - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)


def train_ppo(sampler: Sampler, env: EnvConfig, cfg: PPOConfig, validator: Validator,
              use_drift: bool = False, name: str = "RL:PPO", device: str="cpu", log=print) -> RLPolicy:
    """
    Trains PPO.

    Args:
        sampler (Sampler): Sampler to generate initial data and rollouts.
        env (EnvConfig): Configuration object for PPO containing hyperparameters.
        validator (Validator): Validates the policy on a held-out validation set at every
            few iterations and selects the best-performing policy on this set.
        use_drift (bool = False): True, if drifted weights should be part of the actor input.
        name (str = "RL:PPO"): Human-readable name for the algorithm.
        device (str = "cpu"): Device for PyTorch.
        log (print): Logging function.
    
    Returns:
        RLPolicy: Trained PPO policy.
    """
    probe = sampler.sample(2048)
    F = probe.feats.shape[-1]
    actor = Actor(F, env.lo, env.hi, cfg.hidden, use_drift, cfg.pi_init).to(device)
    actor.norm.fit(probe.feats.to(device))
    critic = Critic(F, cfg.hidden, use_drift, action=False, norm=actor.norm,
                    use_logw=not cfg.wealth_normalized).to(device)
    log_std = nn.Parameter(torch.full((), math.log(cfg.init_std), device=device))
    params = list(actor.parameters()) + list(critic.net.parameters()) + [log_std]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    steps_seen = 0
    g = env.gamma
    for it in range(cfg.iters + 1):
        if it % cfg.eval_every == 0 or it == cfg.iters:
            validator(actor, steps_seen, name, extra={"std": float(log_std.detach().exp())})
            h = validator.history[-1] if validator.history else {}
            log(f"  [{name}] it {it:4d}  env_steps {steps_seen:9d}  val CE {h.get('ce_rate', float('nan'))*100:.3f}%"
                f"  gap {h.get('gap_bp', float('nan')):7.1f} bp/yr  std {float(log_std.detach().exp()):.3f}")
        if it == cfg.iters:
            break
        # ---------------- rollout ----------------
        b = sampler.sample(cfg.n_envs).to(device=device, dtype=torch.float32)
        N, T = b.n, b.horizon
        R = b.episode_returns()
        S_f = b.feats
        S_lw = torch.empty(N, T, device=device)
        S_d = torch.empty(N, T, device=device)
        A = torch.empty(N, T, device=device)
        LP = torch.empty(N, T, device=device)
        V = torch.zeros(N, T + 1, device=device)
        RW = torch.empty(N, T, device=device)
        BETA = torch.ones(N, T, device=device)
        with torch.no_grad():
            w = torch.ones(N, device=device)
            d = torch.zeros(N, device=device)
            std = torch.clamp(log_std.exp(), min=cfg.min_std)
            for t in range(T):
                f = S_f[:, t]
                lw = torch.log(torch.clamp(w, min=1e-8))
                mean = actor(f, d)
                a = mean + std * torch.randn(N, device=device)
                LP[:, t] = _logp(a, mean, std)
                V[:, t] = critic(f, lw, d)
                S_lw[:, t], S_d[:, t], A[:, t] = lw, d, a
                pi = torch.clamp(a, env.lo, env.hi)
                gross = 1.0 + b.rf + pi * (R[:, t] - b.rf)
                growth = gross * (1.0 - env.cost * torch.abs(pi - d)) if env.cost > 0 else gross
                w_new = w * growth
                if cfg.wealth_normalized:
                    rt, bt = normalized_reward(growth, g)
                    RW[:, t], BETA[:, t] = cfg.reward_scale * rt, bt
                else:
                    RW[:, t] = cfg.reward_scale * (crra_u_safe(w_new, g) - crra_u_safe(w, g))
                d = pi * (1.0 + R[:, t]) / gross
                w = w_new
            adv = torch.zeros(N, T, device=device)
            last = torch.zeros(N, device=device)
            for t in range(T - 1, -1, -1):
                delta = RW[:, t] + BETA[:, t] * V[:, t + 1] - V[:, t]
                last = delta + cfg.gae_lambda * BETA[:, t] * last
                adv[:, t] = last
            ret = adv + V[:, :T]
        steps_seen += N * T
        # ---------------- update ----------------
        flat = lambda x: x.reshape(N * T, *x.shape[2:])
        f_, lw_, d_, a_, lp_, adv_, ret_ = map(flat, (S_f, S_lw, S_d, A, LP, adv, ret))
        M = N * T
        for _ in range(cfg.epochs):
            perm = torch.randperm(M, device=device)
            for s in range(0, M, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                std = torch.clamp(log_std.exp(), min=cfg.min_std)
                mean = actor(f_[idx], d_[idx])
                ratio = torch.exp(_logp(a_[idx], mean, std) - lp_[idx])
                ad = adv_[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pg = -torch.min(ratio * ad, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ad).mean()
                vl = ((critic(f_[idx], lw_[idx], d_[idx]) - ret_[idx]) ** 2).mean()
                ent = torch.log(std) + 0.5 * math.log(2 * math.pi * math.e)
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
    validator.restore_best(actor)
    actor.eval()
    return RLPolicy(name, actor, getattr(sampler, "spec", None))
