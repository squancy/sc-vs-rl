"""
TD3 (Fujimoto, van Hoof & Meger 2018): DDPG with clipped double-Q learning, target policy
smoothing and delayed actor updates. Rewards as in PPO: either the raw
telescoping reward U(W_{t+1}) - U(W_t) with critic Q(features, log W, a), or (default) the exact
wealth-normalized Bellman equation q(s,a) = E[r~ + beta q(s', a')] with beta = G^(1-gamma)
(see common.normalized_reward). Off-policy: all past transitions are kept in a replay buffer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from ..env import EnvConfig
from ..models import Sampler
from .common import Actor, Critic, RLPolicy, Validator, crra_u_safe, normalized_reward


@dataclass
class TD3Config:
    """
    Hyperparameters for TD3.

    Attributes:
        iters (int): Total number of iterations.
        n_envs (int): Number of sampled episodes (rollouts).
        buffer (int): Size of the replay buffer.
        batch (int): Batch size during update.
        update_every (int): Update TD3 every n steps.
        lr_actor (float): Actor learning rate.
        lr_critic (float): Critic learning rate.
        tau (float): Coefficient used for Polyak averaging the actor.
        policy_noise (float): Random noise added to the actor value during update.
        noise_clip (float): Clip policy noise at this level.
        expl_noise (float): Random noise added to the actor during exploration.
        expl_noise_final (float): Final noise level during exploration (decreases).
        warmup_iters (int): During the first few warmup iterations, exploration
            noise is higher and the actor is not updated.
        policy_delay (int): Only do a policy gradient backward step at every n updates.
        reward_scale (float): Scales rewards so that too small/large
            gradients will not cause learning problems.
        hidden (tuple[int, ...]): Hidden layer structure of the policy.
        eval_every (int): Evaluate and validate at every n iterations.
        pi_init (float): Initial policy value.
        wealth_normalized (bool): True, if ``common.normalized_reward`` should be used
            for rewards.
        grad_clip (float): Clip gradient norm during update.
    """
    iters: int = 60                 # episode batches
    n_envs: int = 128               # paths per episode batch
    buffer: int = 1_000_000
    batch: int = 512
    update_every: int = 32          # one gradient step per this many new transitions
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    tau: float = 0.005
    policy_noise: float = 0.05
    noise_clip: float = 0.1
    expl_noise: float = 0.1
    expl_noise_final: float = 0.02
    warmup_iters: int = 1
    policy_delay: int = 2
    reward_scale: float = 10.0
    hidden: tuple[int, ...] = (128, 128)
    eval_every: int = 5
    pi_init: float = 0.0
    wealth_normalized: bool = True
    grad_clip: float = 1.0


class _Buffer:
    """
    Replay buffer.

    Attributes:
        cap (int): Capacity.
        i (int): Number of entries stored in the buffer.
        full (bool): True, if the buffer is full.
        f (torch.Tensor): Features at time t.
        f2 (torch.Tensor): Features at time t + 1.
        lw (torch.Tensor): Log wealth at time t.
        lw2 (torch.Tensor): Log wealth at time t + 1.
        d (torch.Tensor): Drifted weights at time t.
        d2 (torch.Tensor): Drifted weights at time t + 1.
        a (torch.Tensor): Action at time t.
        r (torch.Tensor): Reward at time t.
        d (torch.Tensor): Boolean indicating if the episode has ended.
        beta (torch.Tensor): Beta value (G ^ (1 - gamma)) at time t.
    """
    def __init__(self, cap: int, F: int, device):
        self.cap, self.i, self.full = cap, 0, False
        z = lambda *s: torch.zeros(cap, *s, device=device)
        self.f, self.f2 = z(F), z(F)
        self.lw, self.lw2, self.d, self.d2 = z(), z(), z(), z()
        self.a, self.r, self.done, self.beta = z(), z(), z(), z()

    def add(self, **kw):
        """
        Adds an entry to the replay buffer.

        Args:
            **kw: Any keyword argument whose value will be added to the
                buffer object with the same name at the current index.
        """
        n = kw["a"].shape[0]
        idx = (torch.arange(n, device=self.a.device) + self.i) % self.cap
        for k, v in kw.items():
            getattr(self, k)[idx] = v
        self.i = (self.i + n) % self.cap
        self.full = self.full or (self.i < n)

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n: int) -> dict[str, torch.Tensor]:
        """
        Samples n entries randomly from the replay buffer.

        Args:
            n (int): Number of elements to sample.

        Returns:
            dict[str, torch.Tensor]: Sampled entries.
        """
        idx = torch.randint(0, len(self), (n,), device=self.a.device)
        return {k: getattr(self, k)[idx] for k in ("f", "lw", "d", "a", "r", "f2", "lw2", "d2", "done", "beta")}


def train_td3(sampler: Sampler, env: EnvConfig, cfg: TD3Config, validator: Validator,
              use_drift: bool = False, name: str = "RL:TD3", device: str = "cpu", log=print) -> RLPolicy:
    """
    Trains TD3.

    Args:
        sampler (Sampler): Sampler object to generate initial data and rollouts.
        env (EnvConfig): Configuration object.
        cfg (TD3Config): TD3-specific config.
        validator (Validator): Validates the policy on a held out validation set
            every few iterations and selects best-performing policy.
        use_drift (bool = False): True, if the input to the actor should contain
            drifted weights.
        name (str = "RL:TD3"): Human-readable name of the algorithm.
        device (str = "cpu"): Device for PyTorch.
        log (print): Logging function.
    """

    probe = sampler.sample(2048)
    F = probe.feats.shape[-1]
    actor = Actor(F, env.lo, env.hi, cfg.hidden, use_drift, cfg.pi_init).to(device)
    actor.norm.fit(probe.feats.to(device))
    ul = not cfg.wealth_normalized
    q1 = Critic(F, cfg.hidden, use_drift, action=True, norm=actor.norm, use_logw=ul).to(device)
    q2 = Critic(F, cfg.hidden, use_drift, action=True, norm=actor.norm, use_logw=ul).to(device)
    actor_t, q1_t, q2_t = copy.deepcopy(actor), copy.deepcopy(q1), copy.deepcopy(q2)
    opt_a = torch.optim.Adam(actor.net.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(list(q1.net.parameters()) + list(q2.net.parameters()), lr=cfg.lr_critic)
    buf = _Buffer(cfg.buffer, F, device)
    g = env.gamma
    steps_seen, n_updates = 0, 0
    for it in range(cfg.iters + 1):
        if it % cfg.eval_every == 0 or it == cfg.iters:
            validator(actor, steps_seen, name)
            h = validator.history[-1] if validator.history else {}
            log(f"  [{name}] it {it:4d}  env_steps {steps_seen:9d}  val CE {h.get('ce_rate', float('nan'))*100:.3f}%"
                f"  gap {h.get('gap_bp', float('nan')):7.1f} bp/yr  buffer {len(buf)}")
        if it == cfg.iters:
            break
        frac = it / max(cfg.iters - 1, 1)
        sig = cfg.expl_noise + (cfg.expl_noise_final - cfg.expl_noise) * frac
        if it < cfg.warmup_iters:
            sig = 3 * cfg.expl_noise
        b = sampler.sample(cfg.n_envs).to(device=device, dtype=torch.float32)
        N, T = b.n, b.horizon
        R = b.episode_returns()
        with torch.no_grad():
            w = torch.ones(N, device=device)
            d = torch.zeros(N, device=device)
            for t in range(T):
                f = b.feats[:, t]
                lw = torch.log(torch.clamp(w, min=1e-8))
                a = torch.clamp(actor(f, d) + sig * torch.randn(N, device=device), env.lo, env.hi)
                gross = 1.0 + b.rf + a * (R[:, t] - b.rf)
                growth = gross * (1.0 - env.cost * torch.abs(a - d)) if env.cost > 0 else gross
                w_new = w * growth
                if cfg.wealth_normalized:
                    r, beta = normalized_reward(growth, g)
                else:
                    r, beta = crra_u_safe(w_new, g) - crra_u_safe(w, g), torch.ones_like(w)
                d_new = a * (1.0 + R[:, t]) / gross
                last = t == T - 1
                f2 = b.feats[:, t + 1] if not last else f
                buf.add(f=f, lw=lw, d=d, a=a, r=cfg.reward_scale * r, f2=f2,
                        lw2=torch.log(torch.clamp(w_new, min=1e-8)), d2=d_new,
                        done=torch.full((N,), float(last), device=device), beta=beta)
                w, d = w_new, d_new
        steps_seen += N * T
        if it < cfg.warmup_iters:
            continue
        for _ in range(max(1, N * T // cfg.update_every)):
            s = buf.sample(cfg.batch)
            with torch.no_grad():
                noise = torch.clamp(cfg.policy_noise * torch.randn_like(s["a"]), -cfg.noise_clip, cfg.noise_clip)
                a2 = torch.clamp(actor_t(s["f2"], s["d2"]) + noise, env.lo, env.hi)
                qt = torch.min(q1_t(s["f2"], s["lw2"], s["d2"], a2), q2_t(s["f2"], s["lw2"], s["d2"], a2))
                y = s["r"] + (1.0 - s["done"]) * s["beta"] * qt
            lc = ((q1(s["f"], s["lw"], s["d"], s["a"]) - y) ** 2).mean() + \
                 ((q2(s["f"], s["lw"], s["d"], s["a"]) - y) ** 2).mean()
            opt_c.zero_grad()
            lc.backward()
            torch.nn.utils.clip_grad_norm_(list(q1.net.parameters()) + list(q2.net.parameters()), cfg.grad_clip)
            opt_c.step()
            n_updates += 1
            if n_updates % cfg.policy_delay == 0:
                la = -q1(s["f"], s["lw"], s["d"], actor(s["f"], s["d"])).mean()
                opt_a.zero_grad()
                la.backward()
                torch.nn.utils.clip_grad_norm_(actor.net.parameters(), cfg.grad_clip)
                opt_a.step()
                with torch.no_grad():
                    for net, tgt in ((actor, actor_t), (q1, q1_t), (q2, q2_t)):
                        for p, pt in zip(net.net.parameters(), tgt.net.parameters()):
                            pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
    validator.restore_best(actor)
    actor.eval()
    return RLPolicy(name, actor, getattr(sampler, "spec", None))
