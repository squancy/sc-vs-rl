from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..env import EnvConfig, Obs, Policy, rollout
from ..models import FeatureSpec, PathBatch, Sampler, build_features
from ..utils import paired_ce_gap, ce_stats


class Normalizer(nn.Module):
    """
    Fixed affine feature normalization fitted on a reference batch.
    
    Attributes:
        mean (torch.Tensor): Mean of the batch.
        std (torch.Tensor): Std of the batch.
    """

    def __init__(self, dim: int):
        """
        Initializes a Normalizer.

        Registers buffers for the mean and volatility of the batch.

        Args:
            dim (int): Dimension of the mean and std.
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x: torch.Tensor, skip: Sequence[int] = (0,)) -> Normalizer:
        """
        Calculate the mean and volatility of ``x``, optionally skipping indicies.

        Args:
            x (torch.Tensor): Data to normalize.
            skip (Sequence[int] = (0,)): Sequence of indicies for which normalization
            is not applied (in this case, mean is set to 0 and volatility is set to 1).

        Returns:
            Normalizer: Class instance with mean and std attributes.
        """
        x = x.reshape(-1, x.shape[-1]).float()
        mu, sd = x.mean(0), x.std(0).clamp(min=1e-6)
        for j in skip:                     
            mu[j], sd[j] = 0.0, 1.0
        self.mean.copy_(mu)
        self.std.copy_(sd)
        return self

    def forward(self, x):
        return (x - self.mean) / self.std


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, act=nn.Tanh) -> nn.Sequential:
    """
    Creates an MLP with some hidden layers.

    Args:
        in_dim (int): Input dimension to the neural network.
        hidden (Sequence[int]): Sequence of hidden layer sizes.
        out_dim (int): Output dimension.
        act (nn.Tanh): Hidden layer activation function.

    Returns:
        nn.Sequential: Neural network with the specified layers.
    """
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Deterministic policy pi(features[, drifted weight]) in [lo, hi] via a scaled tanh.
    Wealth is deliberately *not* an input: under CRRA the optimal fraction is wealth-free.

    Attributes:
        lo (float): Lower bound for pi.
        hi (float): Upper bound for pi.
        use_drift (bool): True, if the input to the policy should contain the drifted weights.
        norm (Normalizer): Input normalizer.
    """

    def __init__(self, feat_dim: int, lo: float, hi: float, hidden=(64, 64), use_drift: bool = False,
                 pi_init: float = 0.0):
        super().__init__()
        self.lo, self.hi, self.use_drift = lo, hi, use_drift
        self.norm = Normalizer(feat_dim)
        self.net = mlp(feat_dim + int(use_drift), hidden, 1)
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            u = max(min((pi_init - mid) / half, 0.999), -0.999)
            self.net[-1].bias.fill_(math.atanh(u))

    def forward(self, feats: torch.Tensor, drift: torch.Tensor | None) -> torch.Tensor:
        """
        Does a single forward pass through the neural network.

        Args:
            feats (torch.Tensor): Input.
            drift (torch.Tensor | None): Optional drift weights.

        Returns:
            torch.Tensor: Output of the network after one pass.
        """
        x = self.norm(feats)
        if self.use_drift:
            x = torch.cat([x, drift.unsqueeze(-1)], -1)
        mid, half = 0.5 * (self.hi + self.lo), 0.5 * (self.hi - self.lo)
        return mid + half * torch.tanh(self.net(x).squeeze(-1))


class Critic(nn.Module):
    """
    V(s) or Q(s, a) with s = (features[, log wealth][, drift]).

    With the raw telescoping reward U(W_{t+1}) - U(W_t) the value scales with W_t^(1-gamma),
    so log wealth must be an input (use_logw=True). With the wealth-normalized Bellman equation
    (see ``normalized_reward``) the value is wealth-free and log wealth is not needed.

    Attributes:
        use_drift (bool = False): True, if the input to the network should contain the drifted weights.
        action (bool = False): True, if the input to the network should contain the action.
        use_logw (bool = True): True, if the input to the network should contain the log weights.
        norm (Normalizer | None = None): Normalizer to use before the network.
        net (nn.Sequential): Neural network for the critic.
    """

    def __init__(self, feat_dim: int, hidden=(64, 64), use_drift: bool = False, action: bool = False,
                 norm: Normalizer | None = None, use_logw: bool = True):
        super().__init__()
        self.use_drift, self.action, self.use_logw = use_drift, action, use_logw
        self.norm = norm if norm is not None else Normalizer(feat_dim)
        self.net = mlp(feat_dim + int(use_logw) + int(use_drift) + int(action), hidden, 1)

    def forward(self, feats: torch.Tensor, logw: torch.Tensor | None = None, drift: torch.Tensor | None = None, a: torch.Tensor | None = None):
        """
        Does a single forward pass through the network.

        Args:
            feats (torch.Tensor): Input features.
            logw (torch.Tensor | None = None): Optional log weights.
            drift (torch.Tensor | None = None): Optional drift weights.
            a (torch.Tensor | None = None): Optional action.
        """
        parts = [self.norm(feats)]
        if self.use_logw:
            parts.append(logw.unsqueeze(-1))
        if self.use_drift:
            parts.append(drift.unsqueeze(-1))
        if self.action:
            parts.append(a.unsqueeze(-1))
        return self.net(torch.cat(parts, -1)).squeeze(-1)


class RLPolicy(Policy):
    """Wraps a trained (deterministic) actor in the common Policy interface.

    If ``spec`` is given, the policy rebuilds its own features from the batch at reset, so it can
    be evaluated on batches built for a different information set (e.g. a tau-only policy on a
    batch whose features include returns). Otherwise it uses ``batch.feats``.
    
    Attributes:
        name (str): Human-readable policy name.
        actor (Actor): Actor underlying the policy.
        spec (FeatureSpec | None = None): Feature specification.
    """

    def __init__(self, name: str, actor: Actor, spec: FeatureSpec | None = None):
        self.name, self.actor, self.spec = name, actor, spec
        self._feats = None

    def reset(self, batch: PathBatch) -> None:
        """
        Resets policy features.

        Args:
            batch (PathBatch): Batch of features to reset to.
        """
        if self.spec is None:
            self._feats = batch.feats
            return
        d = lambda x: None if x is None else x.double()
        self._feats = build_features(d(batch.returns), d(batch.latent), batch.rf, batch.dt, batch.burn,
                                     batch.horizon, self.spec, d(batch.rv))

    @torch.no_grad()
    def act(self, t: int, obs: Obs) -> torch.Tensor:
        """
        Policy output using the underlying actor.

        Args:
            t (int): First ``t`` feature dimensions to use.
            obs (Obs): Input to the actor.
        
        Returns:
            torch.Tensor: Output of the actor.
        """
        p = next(self.actor.parameters())
        feats = self._feats if self._feats is not None else obs.batch.feats
        f = feats[:, t].to(device=p.device, dtype=p.dtype)
        d = obs.drift.to(device=p.device, dtype=p.dtype)
        return self.actor(f, d).to(device=obs.w.device, dtype=obs.w.dtype)


def normalized_reward(growth: torch.Tensor, gamma: float, floor: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wealth-normalized Bellman quantities for CRRA utility.

    TD3 and PPO need per-step rewards that sum to E[U(W_T)] (the objective).
    If G_t = W_{t + 1} / W_t (the gain in (t, t + 1]), then the optimal value
    function V_t(W, s) = W ^ {1 - gamma} v_t(s), where v_t(s) is the maximum
    expected utility at time T when starting at time t in state s with one unit
    of wealth. This requires v_T = 1 / (1 - gamma). So,
    V_t(W, s) = max_{a} E[V_{t + 1}(W_t * G_t, s')]. Simplifying with W_{t} ^ {1 - gamma}
    gives v_t(s) = max_{a} E[G_t ^ {1 - gamma}v_{t + 1}(s')].

    By letting q_t = v_t - 1 / (1 - gamma) (so q_T = 0), we have q_t = max_{a} E[r' + beta * q_{t + 1}]
    where r' = G_t ^ {1 - gamma} / (1 - gamma), beta = G_t ^ {1 - gamma}.

    Note that the sum of q_i's from t to T is (U(W_T) - U(W_t)) / (W_t ^ {1 - gamma}).
    If we simply used the telescoping reward U(W_{t + 1}) - U(W_t), it would be scaled
    by W ^ {1 - gamma}, which makes the range orders of magnitude larger and can cause
    TD3 to diverge.

    Args:
        growth (torch.Tensor): G_t.
        gamma (float): Risk-aversion rate.
        floor (float = 1e-3): G_t is clamped at ``floor``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: r' and beta. 
    """
    G = torch.clamp(growth, min=floor)
    if abs(gamma - 1) < 1e-12:
        return torch.log(G), torch.ones_like(G)
    beta = G ** (1 - gamma)
    return (beta - 1) / (1 - gamma), beta


def crra_u_safe(w: torch.Tensor, gamma: float, floor: float = 1e-4) -> torch.Tensor:
    """
    Numerically safe CRRA utility.

    Args:
        w (torch.Tensor): Input to CRRA.
        gamma (float): Risk-aversion rate.
        floor (float = 1e-4): w is clamped at ``floor``.
    
    Returns:
        torch.Tensor: Calculated CRRA utility.
    """
    w = torch.clamp(w, min=floor)
    if abs(gamma - 1) < 1e-12:
        return torch.log(w)
    return w ** (1 - gamma) / (1 - gamma)


class Validator:
    """
    Periodic validation on a fixed batch: CE gap (bp/yr) to a reference policy's terminal
    wealth on the same paths. Keeps the best actor (model selection on validation paths).
    
    Attributes:
        batch (PathBatch): Batch used for evaluation. 
        env (EnvConfig): Config object.
        ref_wT (torch.Tensor | None = None): Reference policy's terminal wealth.
        history (list[dict[str, float]]): History of CE rates.
        best_score (float): Best CE rate.
        best_state (state_dict): State with the best CE rate.
    """

    def __init__(self, batch: PathBatch | None, env: EnvConfig, ref_wT: torch.Tensor | None = None):
        self.batch, self.env, self.ref_wT = batch, env, ref_wT
        self.history: list[dict[str, float]] = []
        self.best_score, self.best_state = -math.inf, None

    def __call__(self, actor: Actor, env_steps: int, name: str = "rl", extra: dict | None = None) -> float:
        """
        Evaluation on a single rollout.

        Args:
            actor (Actor): Policy to evaluate.
            env_steps (int): Number of steps in the rollout.
            name (str = "rl"): Name of the policy.
            extra (dict | None = None): If the reference policy is not given,
            ``extra`` will be added instead of the CE gap.
        
        Returns:
            float: The CE rate of the evaluated policy.
        """
        if self.batch is None:
            return math.nan
        actor.eval()
        out = rollout(self.batch, RLPolicy(name, actor), self.env)
        actor.train()
        st = ce_stats(out["wT"], self.env.gamma, self.batch.years)
        rec = {"env_steps": env_steps, "ce_rate": st["ce_rate"]}
        if self.ref_wT is not None:
            rec.update(paired_ce_gap(self.ref_wT, out["wT"], self.env.gamma, self.batch.years))
        if extra:
            rec.update(extra)
        self.history.append(rec)
        score = st["ce_rate"]
        if score > self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(actor.state_dict())
        return score

    def restore_best(self, actor: Actor) -> None:
        """
        Restores the best performing policy on evaluation.

        Args:
            actor (Actor): The actor to reset.
        """
        if self.best_state is not None:
            actor.load_state_dict(self.best_state)


def fit_feature_normalizer(sampler: Sampler, n: int = 2048) -> Normalizer:
    """
    Given a ``sampler``, samples a ``PathBatch`` of length ``n`` and and
    returns a Normalizer based on this sample.

    Args:
        sampler (Sampler): The sampler.
        n (int = 2048): Sample size.

    Returns:
        Normalizer: Fitted normalizer on the sample.
    """
    b = sampler.sample(n)
    return Normalizer(b.feats.shape[-1]).fit(b.feats)
