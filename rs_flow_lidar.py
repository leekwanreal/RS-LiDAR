import torch
from typing import Optional

class RSFlowLiDARGuidance:
    # Empirical VAE Whitening \Sigma^{-1/2} cho 16-channel FLUX (Định lý 7 & Table 2)
    FLUX_EMPIRICAL_INV_STDS = torch.tensor([
        0.546, 0.690, 0.800, 0.909, 1.053, 1.136, 1.250, 1.389,
        1.538, 1.724, 1.923, 2.128, 2.381, 2.632, 2.857, 3.125
    ], dtype=torch.float32)

    def __init__(
        self, 
        s_flow: float = 12.5, 
        lambda_r: float = 5000.0, 
        whitening_weights: Optional[torch.Tensor] = None
    ):
        r"""
        Module điều khiển vận tốc Lookahead cho Rectified Flows (Định lý 7).
        
        Args:
            s_flow: Hệ số khuếch đại vận tốc dẫn đường (default: 12.5).
            lambda_r: Độ nhạy nhiệt độ Softmax exponentiation cho reward (default: 5000.0).
            whitening_weights: Trọng số làm trắng latent space \Sigma^{-1/2} (default: 16-channel identity / FLUX empirical).
        """
        self.s_flow = s_flow
        self.lambda_r = lambda_r
        # Default VAE Whitening cho 16-channel nếu không cung cấp (bảo toàn đơn vị phương sai cho latents chuẩn hóa)
        if whitening_weights is None:
            self.whitening_weights = torch.ones((1, 16, 1, 1))
        else:
            self.whitening_weights = whitening_weights.view(1, -1, 1, 1)

    @torch.no_grad()
    def compute_velocity_guidance(
        self, 
        z_t: torch.Tensor, 
        lookahead_z0: torch.Tensor, 
        smoothed_rewards: torch.Tensor, 
        sigma_t: float,
        use_posterior_damper: bool = True,
        bypass_fast_path: bool = False
    ) -> torch.Tensor:
        r"""
        Tính toán vector lực đẩy vận tốc an toàn với Parabolic Damper (Định lý 7).
        
        Args:
            z_t: Latent hiện tại [B, C, H, W].
            lookahead_z0: Các điểm ảnh sạch dự phóng [K, C, H, W].
            smoothed_rewards: Điểm thưởng đã chuẩn hóa [K].
            sigma_t: Mức độ nhiễu hiện tại in [0, 1] (1=noise, 0=clean).
            use_posterior_damper: Kích hoạt Parabolic Damper \omega(t) = t(1-t) triệt tiêu kỳ dị.
            bypass_fast_path: Cho phép tắt Fast-path để kiểm tra Parabolic Damper tại biên cực tiểu.
            
        Returns:
            Gia số vận tốc delta_v [B, C, H, W].
        """
        # 1. Fast-Path Early Return (Cửa sổ hoạt động 0.20 - 0.98)
        if not bypass_fast_path and (sigma_t <= 0.20 or sigma_t >= 0.98):
            return torch.zeros_like(z_t)
            
        device = z_t.device
        B, C, H, W = z_t.shape
        K = lookahead_z0.shape[0]
        
        # 2. 16-Channel / C-Channel Whitening (Tương thích kích thước động)
        if self.whitening_weights.shape[1] == C:
            w_weights = self.whitening_weights.to(device, dtype=z_t.dtype)
        else:
            w_weights = torch.ones((1, C, 1, 1), device=device, dtype=z_t.dtype)
            
        z_t_w = z_t * w_weights
        z0_w = lookahead_z0 * w_weights
        
        # 3. Spatial Flattening để chặn VRAM Leak (tránh tensor 5D [B, K, C, H, W])
        z_t_flat = z_t_w.view(B, -1)  # [B, D]
        z0_flat = z0_w.view(K, -1)    # [K, D]
        
        alpha_t = 1.0 - sigma_t
        var_k = max(sigma_t, 0.15) ** 2
        
        # Khoảng cách L2: || (1-t)*z_0 - z_t ||^2 trên không gian 2D
        diff = (alpha_t * z0_flat.unsqueeze(0)) - z_t_flat.unsqueeze(1)  # [B, K, D]
        sq_dist = torch.sum(diff ** 2, dim=-1)                           # [B, K]
        
        # 4. Softmax Tilting
        log_base = -sq_dist / (2.0 * var_k + 1e-8)
        log_tilted = log_base + self.lambda_r * smoothed_rewards.unsqueeze(0)
        
        w_base = torch.softmax(log_base, dim=-1)
        w_tilted = torch.softmax(log_tilted, dim=-1)
        w_diff = (w_tilted - w_base).unsqueeze(-1)  # [B, K, 1]
        
        # Trọng tâm vector dịch chuyển \sum (w^r - w) ((1-t)z0 - z_t)
        mean_shift_flat = torch.sum(w_diff * diff, dim=1)  # [B, D]
        
        # 5. Singularity-Free Parabolic Damped Velocity Shift (Định lý 7 & Theorem 7 formula 38)
        omega_t = sigma_t * (1.0 - sigma_t)
        if use_posterior_damper:
            velocity_shift_flat = -self.s_flow * omega_t * mean_shift_flat
        else:
            efr_score_grad = mean_shift_flat / (var_k + 1e-8)
            velocity_shift_flat = -self.s_flow * (sigma_t / (alpha_t + 1e-8)) * efr_score_grad
        
        return velocity_shift_flat.view(B, C, H, W)
