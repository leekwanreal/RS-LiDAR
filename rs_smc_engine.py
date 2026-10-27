import torch
from typing import Callable, Tuple

class HarmonizedRSLiDARSMCEngine:
    def __init__(
        self,
        sigma_high: float = 1.0,
        sigma_smc: float = 0.25,
        beta_risk: float = 0.5,
        eta_drift: float = 0.4
    ):
        """
        Động cơ SMC kết hợp Doob-Tweedie Innovation Residual và 64-bit Seed Replay.
        
        Args:
            sigma_high: Giới hạn độ lệch chuẩn nhiễu tối đa.
            sigma_smc: Độ lệch chuẩn nhiễu chuẩn hóa cho SMC Tweedie perturbation.
            beta_risk: Hệ số phạt rủi ro Entropic Risk (Định lý 5.2).
            eta_drift: Tỷ lệ thế năng lookahead đã hấp thụ vào drift (Định lý Girsanov).
        """
        self.sigma_high = sigma_high
        self.sigma_smc = sigma_smc
        self.beta_risk = beta_risk
        self.eta_drift = eta_drift

    @torch.no_grad()
    def compute_step(
        self,
        x_t: torch.Tensor,
        tweedie_x0: torch.Tensor,
        v_lidar: torch.Tensor,
        seed_int: int,
        rho_t: float,
        reward_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thực thi 1 bước đánh giá thặng dư thông tin mới (Innovation).
        
        Args:
            x_t: Latent tensor hiện tại [N, C, H, W].
            tweedie_x0: Ước lượng điểm sạch Tweedie [N, C, H, W].
            v_lidar: Thế năng tương lai đã hấp thụ vào Drift [N].
            seed_int: Seed 64-bit nguyên thủy từ bước lookahead.
            rho_t: Tỷ lệ tín hiệu trên nhiễu (alpha_bar_t / alpha_bar_0).
            reward_fn: Hàm gọi Reward Model.
            
        Returns:
            Tuple (phi_residual, v_smc)
        """
        device, dtype = x_t.device, x_t.dtype
        N = x_t.shape[0]
        
        # 1. 64-bit Seed Replay: Tái tạo nhiễu đối xứng (0.00 GiB VRAM overhead)
        generator = torch.Generator(device=device).manual_seed(seed_int)
        u_base = torch.randn(x_t.shape, generator=generator, device=device, dtype=dtype)
        
        # 2. Sinh mẫu Tweedie Đối ngẫu
        sigma_t = self.sigma_smc * rho_t
        tw_pos = torch.clamp(tweedie_x0 + sigma_t * u_base, -1.0, 1.0)
        tw_neg = torch.clamp(tweedie_x0 - sigma_t * u_base, -1.0, 1.0)
        del u_base # Giải phóng ngay lập tức
        
        # 3. Đánh giá Reward 
        # Gọi batch 2N để tối ưu tốc độ, sau đó tách ra N và N
        batch_eval = torch.cat([tw_pos, tw_neg], dim=0)
        r_eval = reward_fn(batch_eval) # Trả về tensor [2N]
        r_pos, r_neg = r_eval[:N], r_eval[N:]
        
        # 4. Phạt Entropic Risk cho SMC (Định lý 5.2 bản rút gọn cho cặp đối ngẫu)
        v_smc = 0.5 * (r_pos + r_neg) - (self.beta_risk / 4.0) * ((0.5 * (r_pos - r_neg)) ** 2)
        
        # 5. Doob-Tweedie Innovation Residual (Triệt tiêu Double-Counting)
        # Chỉ đánh giá lượng giá trị MỚI chưa được cộng vào drift
        phi_residual = v_smc - (self.eta_drift * rho_t * v_lidar)
        
        return phi_residual, v_smc

    @torch.no_grad()
    def resample_if_needed(
        self, 
        log_weights: torch.Tensor, 
        x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Tính ESS và Resample nếu hạt bị sụp đổ."""
        N = x_t.shape[0]
        # Chuẩn hóa log-sum-exp an toàn
        max_log_w = log_weights.max()
        w = torch.exp(log_weights - max_log_w)
        w_norm = w / w.sum()
        
        ess = 1.0 / torch.sum(w_norm ** 2)
        
        if ess < 0.5 * N:
            # Resampling bằng Multinomial
            indices = torch.multinomial(w_norm, N, replacement=True)
            x_t_new = x_t[indices].clone()
            # Reset trọng số về uniform (log = 0)
            log_weights_new = torch.zeros_like(log_weights)
            return x_t_new, log_weights_new, True
            
        return x_t, log_weights, False
