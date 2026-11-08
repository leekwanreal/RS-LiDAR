import torch
from typing import Optional

class RSFlowLiDARGuidance:
    # Empirical VAE Whitening \Sigma^{-1/2} cho 16-channel FLUX
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
        Module điều khiển vận tốc Lookahead cho Rectified Flows (Flow Matching).
        
        Args:
            s_flow: Hệ số khuếch đại vận tốc dẫn đường (default: 12.5).
            lambda_r: Độ nhạy nhiệt độ Softmax exponentiation cho reward (default: 5000.0).
            whitening_weights: Trọng số làm trắng latent space \Sigma^{-1/2} cho khoảng cách Mahalanobis.
        """
        self.s_flow = s_flow
        self.lambda_r = lambda_r
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
        bypass_fast_path: bool = False
    ) -> torch.Tensor:
        r"""
        Tính toán vector gia số vận tốc lái dẫn đường chuẩn xác cho Flow Matching:
            \Delta v_t(z_t) = - (s_flow / t) * \sum_{k=1}^K (w_k^r - w_k) z_{0,k}
        
        Args:
            z_t: Latent hiện tại [B, C, H, W].
            lookahead_z0: Các điểm ảnh sạch dự phóng [K, C, H, W].
            smoothed_rewards: Điểm thưởng đã làm trơn bằng RS [K].
            sigma_t: Mức độ nhiễu hiện tại t in [0, 1] (1=noise, 0=clean).
            bypass_fast_path: Cho phép tắt Fast-path để kiểm tra hành vi số học tại biên.
            
        Returns:
            Gia số vận tốc delta_v [B, C, H, W].
        """
        # 1. Fast-Path Early Return (Cửa sổ hoạt động an toàn [0.20, 0.98])
        if not bypass_fast_path and (sigma_t <= 0.20 or sigma_t >= 0.98):
            return torch.zeros_like(z_t)
            
        device = z_t.device
        B, C, H, W = z_t.shape
        K = lookahead_z0.shape[0]
        
        # 2. VAE Whitening cho khoảng cách Mahalanobis (Tương thích kích thước động)
        if self.whitening_weights.shape[1] == C:
            w_weights = self.whitening_weights.to(device, dtype=z_t.dtype)
        else:
            w_weights = torch.ones((1, C, 1, 1), device=device, dtype=z_t.dtype)
            
        z_t_w = z_t * w_weights
        z0_w = lookahead_z0 * w_weights
        
        # 3. Spatial Flattening để chặn VRAM Leak
        z_t_w_flat = z_t_w.view(B, -1)   # [B, D]
        z0_w_flat = z0_w.view(K, -1)     # [K, D]
        z0_raw_flat = lookahead_z0.view(K, -1)  # [K, D] (tọa độ tự nhiên)
        
        alpha_t = 1.0 - sigma_t
        t_val = max(sigma_t, 0.05)
        var_k = t_val ** 2
        
        # Khoảng cách L2 / Mahalanobis: || (1-t)*z_0 - z_t ||^2_Sigma
        diff_w = (alpha_t * z0_w_flat.unsqueeze(0)) - z_t_w_flat.unsqueeze(1)  # [B, K, D]
        sq_dist = torch.sum(diff_w ** 2, dim=-1)                               # [B, K]
        
        # 4. Softmax Tilting
        log_base = -sq_dist / (2.0 * var_k + 1e-8)
        log_tilted = log_base + self.lambda_r * smoothed_rewards.unsqueeze(0)
        
        w_base = torch.softmax(log_base, dim=-1)
        w_tilted = torch.softmax(log_tilted, dim=-1)
        w_diff = (w_tilted - w_base).unsqueeze(-1)  # [B, K, 1]
        
        # 5. Độ dịch chuyển ảnh sạch lookahead trong tọa độ tự nhiên:
        # \Delta z_0 = \sum_{k=1}^K (w_k^r - w_k) z_{0,k}
        z0_shift_flat = torch.sum(w_diff * z0_raw_flat.unsqueeze(0), dim=1)  # [B, D]
        
        # 6. Gia số vận tốc lái chuẩn của Flow Matching:
        # \Delta v_t = - (s_flow / t) * \Delta z_0
        velocity_shift_flat = - (self.s_flow / t_val) * z0_shift_flat
        
        return velocity_shift_flat.view(B, C, H, W)
