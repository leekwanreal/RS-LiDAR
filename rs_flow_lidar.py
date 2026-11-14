"""Lookahead velocity guidance for rectified-flow models (FLUX), used by `FluxPipeline_LiDAR`.

With flow time t (1 = noise, 0 = data), z_t = (1 - t) z_0 + t eps. Given K lookahead samples z_{0,k}
with (optionally smoothed) rewards r_k, `compute_velocity_guidance` returns the velocity correction

    w   = softmax_k(-||(1 - t) z_{0,k} - z_t||^2 / (2 t^2)),
    w^r = softmax_k(-||(1 - t) z_{0,k} - z_t||^2 / (2 t^2) + lambda_r * r_k),
    delta_v_t = -(s_flow / t) * sum_k (w^r_k - w_k) z_{0,k},

which is added to the predicted velocity inside the guidance window [t_end, t_start].
"""
from typing import Optional

import torch


class RSFlowLiDARGuidance:
    def __init__(
        self,
        s_flow: float = 40.0,
        lambda_r: float = 5000.0,
        t_start: float = 0.98,
        t_end: float = 0.20
    ):
        r"""
        Args:
            s_flow: velocity guidance scale (40.0 is the LiDAR FLUX setting).
            lambda_r: reward inverse temperature in the tilted softmax.
            t_start, t_end: default guidance window in flow time (FLUX uses [0.95, 1.0]).
        """
        self.s_flow = s_flow
        self.lambda_r = lambda_r
        self.t_start = t_start
        self.t_end = t_end

    @torch.no_grad()
    def compute_velocity_guidance(
        self,
        z_t: torch.Tensor,
        lookahead_z0: torch.Tensor,
        smoothed_rewards: torch.Tensor,
        sigma_t: float,
        bypass_fast_path: bool = False,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        return_diagnostics: bool = False
    ):
        r"""Velocity correction delta_v_t (see the module docstring).

        Args:
            z_t: current latents [B, C, H, W].
            lookahead_z0: lookahead clean latents [K, C, H, W].
            smoothed_rewards: rewards of the lookahead samples [K].
            sigma_t: flow time t in [0, 1] (1 = noise, 0 = data).
            bypass_fast_path: compute the guidance even outside the guidance window.
            t_start, t_end: guidance window for this call (defaults: `self.t_start`, `self.t_end`).
            return_diagnostics: also return {"w_diff": w^r - w as a [B, K] CPU tensor}.

        Returns:
            delta_v [B, C, H, W] in the dtype of z_t, or (delta_v, diagnostics).

        Distances, softmaxes and the shift of z_0 are computed in float32: with D = 16*128*128 the squared
        distances are ~2.4e5, where bfloat16 has a quantization step of 1024 and would round the K distances
        to the same value.
        """
        win_start = t_start if t_start is not None else self.t_start
        win_end = t_end if t_end is not None else self.t_end

        # Outside the guidance window the correction is zero. A window that starts at pure noise
        # (t_start ~ 1, as for FLUX) includes both ends.
        if not bypass_fast_path:
            if win_start >= 0.999:
                outside = sigma_t < win_end or sigma_t > win_start
            else:
                outside = sigma_t <= win_end or sigma_t >= win_start
            if outside:
                zeros = torch.zeros_like(z_t)
                return (zeros, {"w_diff": None}) if return_diagnostics else zeros

        device = z_t.device
        out_dtype = z_t.dtype
        dtype = torch.float32
        B, C, H, W = z_t.shape
        K = lookahead_z0.shape[0]

        z_t = z_t.to(dtype=dtype)
        lookahead_z0 = lookahead_z0.to(device=device, dtype=dtype)
        smoothed_rewards = smoothed_rewards.to(device=device, dtype=dtype)

        z_t_flat = z_t.view(B, -1)            # [B, D]
        z0_flat = lookahead_z0.view(K, -1)    # [K, D]

        alpha_t = 1.0 - sigma_t
        t_val = max(sigma_t, 0.05)
        var_k = t_val ** 2

        # Squared L2 distance || (1 - t) z_0 - z_t ||^2: [B, K]
        diff = (alpha_t * z0_flat.unsqueeze(0)) - z_t_flat.unsqueeze(1)  # [B, K, D]
        sq_dist = torch.sum(diff ** 2, dim=-1)

        log_base = -sq_dist / (2.0 * var_k + 1e-8)
        log_tilted = log_base + self.lambda_r * smoothed_rewards.unsqueeze(0)

        w_base = torch.softmax(log_base, dim=-1)
        w_tilted = torch.softmax(log_tilted, dim=-1)
        w_diff = (w_tilted - w_base).unsqueeze(-1)  # [B, K, 1]

        # Shift of the clean-sample estimate: sum_k (w^r_k - w_k) z_{0,k}
        z0_shift_flat = torch.sum(w_diff * z0_flat.unsqueeze(0), dim=1)  # [B, D]

        # delta_v_t = -(s_flow / t) * shift
        velocity_shift_flat = - (self.s_flow / t_val) * z0_shift_flat

        delta_v = velocity_shift_flat.view(B, C, H, W).to(dtype=out_dtype)
        if return_diagnostics:
            return delta_v, {"w_diff": w_diff.squeeze(-1).detach().cpu()}
        return delta_v
