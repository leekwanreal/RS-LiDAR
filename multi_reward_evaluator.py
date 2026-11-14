"""Multi-reward Pareto potential (kept for future use; not part of the reported experiments).

Each reward's smoothed Monte Carlo estimate is normalized by a local Lipschitz scale
Lambda_j = sqrt(Var(R_j) + kappa * sigma^2 * E[|grad R_j|^2]) and the normalized rewards are
combined with a smooth-Tchebycheff potential (1 - rho) * sum_j w_j r_j + rho * softmin_tau(r).
Enabled in Phase 1 with `--use_pareto` when more than one metric is computed.
"""
from typing import Any, Dict, Optional, Tuple

import torch

from lookahead_sampling import compute_entropic_antithetic_rewards


@torch.no_grad()
def compute_multi_reward_pareto_potential(
    multi_rewards_mc: Dict[str, torch.Tensor],
    sigma: float = 1.0,
    kappa_lip: float = 1.0,
    pareto_rho: float = 0.5,
    tau_p: float = 2.0,
    beta_risk: float = 0.5,
    reward_weights: Optional[Dict[str, float]] = None,
    noise_mode: str = "independent_antithetic"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Combine heterogeneous rewards into one Pareto potential per particle.

    Args:
        multi_rewards_mc: reward name -> Monte Carlo reward matrix of shape [M, N].
        sigma: smoothing noise level.
        kappa_lip: weight of the gradient-energy term in the Lipschitz scale.
        pareto_rho: mixing weight (0.0 = weighted mean only, 1.0 = soft-min only).
        tau_p: soft-min temperature (must be > 0).
        beta_risk: entropic risk coefficient passed to `compute_entropic_antithetic_rewards`.
        reward_weights: optional relative weight per reward name.
        noise_mode: 'independent_antithetic', 'independent_iid' or 'coupled_crn'.

    Returns:
        (pareto_potential [N], diagnostics dict).
    """
    if not multi_rewards_mc:
        raise ValueError("multi_rewards_mc cannot be empty. Must contain at least one reward.")
    if tau_p <= 0.0:
        raise ValueError(f"tau_p must be positive, got {tau_p}.")

    normalized_list = []
    diagnostics = {}
    metric_names = list(multi_rewards_mc.keys())

    for name in metric_names:
        r_mc = multi_rewards_mc[name]
        # Smoothed (optionally risk-corrected) reward and gradient energy from the M samples.
        stats = compute_entropic_antithetic_rewards(r_mc, sigma=sigma, beta_risk=beta_risk, noise_mode=noise_mode)
        r_j = stats["r_corrected"]         # [N]
        grad_e_j = stats["grad_energy_sq"]  # [N]

        # Local Lipschitz normalization. The variance uses the raw MC mean so that Var(R_j) has
        # units [R]^2 (the risk-corrected reward would mix in a squared penalty term).
        r_raw_mean = r_mc.to(torch.float32).mean(dim=0)
        mu_j = r_j.mean(dim=-1, keepdim=True)
        var_j = r_raw_mean.var(dim=-1, unbiased=(r_mc.shape[-1] > 1), keepdim=True)

        # Gradient-variance scale, also in units [R]^2.
        lip_var_j = grad_e_j.mean(dim=-1, keepdim=True) * (sigma ** 2)

        # Lambda_j: per-reward scale that prevents one reward from starving the others.
        lambda_j = torch.sqrt(var_j + kappa_lip * lip_var_j + 1e-6)

        r_norm = (r_j - mu_j) / lambda_j
        normalized_list.append(r_norm)

        diagnostics[f"{name}_lambda"] = lambda_j.item() if lambda_j.numel() == 1 else lambda_j
        diagnostics[f"{name}_mu"] = mu_j.item() if mu_j.numel() == 1 else mu_j
        diagnostics[f"{name}_r_norm"] = r_norm

    # Per-reward weights.
    device = normalized_list[0].device
    J = len(metric_names)
    if reward_weights is not None:
        raw_w = torch.tensor([float(reward_weights.get(k, 1.0)) for k in metric_names], device=device, dtype=torch.float32)
        w_norm = raw_w / (raw_w.sum() + 1e-8)
    else:
        w_norm = torch.ones(J, device=device, dtype=torch.float32) / float(J)

    # Smooth-Tchebycheff potential.
    stacked = torch.stack(normalized_list, dim=0)  # [J, N]
    diagnostics["stacked_normalized"] = stacked

    linear_term = torch.sum(w_norm.unsqueeze(-1) * stacked, dim=0)

    # Weighted soft-min: -1/tau * log(sum_j w_j * exp(-tau * R_j)).
    log_w = torch.log(w_norm.clamp(min=1e-8)).unsqueeze(-1)  # [J, 1]
    softmin_term = -(1.0 / tau_p) * torch.logsumexp(-tau_p * stacked + log_w, dim=0)

    pareto_potential = (1.0 - pareto_rho) * linear_term + pareto_rho * softmin_term

    diagnostics["linear_term"] = linear_term
    diagnostics["softmin_term"] = softmin_term
    diagnostics["weights"] = w_norm

    return pareto_potential, diagnostics


class LocalLipschitzParetoEvaluator:
    """Object-oriented wrapper around `compute_multi_reward_pareto_potential`."""
    def __init__(
        self,
        sigma: float = 1.0,
        kappa_lip: float = 1.0,
        pareto_rho: float = 0.5,
        tau_p: float = 2.0,
        beta_risk: float = 0.5,
        reward_weights: Optional[Dict[str, float]] = None,
        noise_mode: str = "independent_antithetic"
    ):
        self.sigma = sigma
        self.kappa_lip = kappa_lip
        self.pareto_rho = pareto_rho
        self.tau_p = tau_p
        self.beta_risk = beta_risk
        self.reward_weights = reward_weights
        self.noise_mode = noise_mode

    @torch.no_grad()
    def evaluate(self, multi_rewards_mc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return compute_multi_reward_pareto_potential(
            multi_rewards_mc=multi_rewards_mc,
            sigma=self.sigma,
            kappa_lip=self.kappa_lip,
            pareto_rho=self.pareto_rho,
            tau_p=self.tau_p,
            beta_risk=self.beta_risk,
            reward_weights=self.reward_weights,
            noise_mode=self.noise_mode
        )
