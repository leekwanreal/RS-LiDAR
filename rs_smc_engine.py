import torch
from typing import Callable, Tuple

class HarmonizedRSLiDARSMCEngine:
    def __init__(
        self,
        sigma_high: float = 1.0,
        sigma_smc: float = 0.25,
        beta_risk: float = 0.5,
        eta_drift: float = 0.4,
        alpha_couple: float = 0.7
    ):
        """
        Động cơ SMC kết hợp Doob-Tweedie Innovation Residual, 64-bit Seed Replay và Kernel-Coupled Noise.
        
        Args:
            sigma_high: Giới hạn độ lệch chuẩn nhiễu tối đa.
            sigma_smc: Độ lệch chuẩn nhiễu chuẩn hóa cho SMC Tweedie perturbation.
            beta_risk: Hệ số phạt rủi ro Entropic Risk (Định lý 5.2).
            eta_drift: Tỷ lệ thế năng lookahead đã hấp thụ vào drift (Định lý Girsanov).
            alpha_couple: Hệ số ghép cặp hạt lookahead và SMC (Theorem 5.1).
        """
        self.sigma_high = sigma_high
        self.sigma_smc = sigma_smc
        self.beta_risk = beta_risk
        self.eta_drift = eta_drift
        self.alpha_couple = alpha_couple

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
            x_t: Latent tensor hiện tại [N, C_lat, H_lat, W_lat].
            tweedie_x0: Ước lượng điểm sạch Tweedie [N, C_img, H_img, W_img].
            v_lidar: Thế năng tương lai đã hấp thụ vào Drift [N].
            seed_int: Seed 64-bit nguyên thủy từ bước lookahead.
            rho_t: Tỷ lệ tín hiệu trên nhiễu (alpha_bar_t / alpha_bar_0).
            reward_fn: Hàm gọi Reward Model.
            
        Returns:
            Tuple (phi_residual, v_smc)
        """
        device, dtype = tweedie_x0.device, tweedie_x0.dtype
        N = tweedie_x0.shape[0]
        
        # 1. 64-bit Seed Replay & Kernel Coupling: Tái tạo nhiễu đối xứng (0.00 GiB VRAM overhead)
        # Khớp tuyệt đối kích thước không gian với tweedie_x0 (thường là 3 channel pixel domain)
        if seed_int is not None and self.alpha_couple > 0.0:
            if isinstance(seed_int, (list, tuple, torch.Tensor)) and len(seed_int) == N:
                u_lookahead = torch.empty(tweedie_x0.shape, device=device, dtype=dtype)
                zeta = torch.empty(tweedie_x0.shape, device=device, dtype=dtype)
                for i in range(N):
                    gen_seed = int(seed_int[i].item() if isinstance(seed_int[i], torch.Tensor) else seed_int[i]) & 0x7FFFFFFF
                    gen = torch.Generator(device=device).manual_seed(gen_seed)
                    u_lookahead[i] = torch.randn(tweedie_x0.shape[1:], generator=gen, device=device, dtype=dtype)
                    zeta[i] = torch.randn(tweedie_x0.shape[1:], generator=gen, device=device, dtype=dtype)
            else:
                scalar_val = seed_int[0] if isinstance(seed_int, (list, tuple, torch.Tensor)) else seed_int
                scalar_seed = int(scalar_val.item() if isinstance(scalar_val, torch.Tensor) else scalar_val) & 0x7FFFFFFF
                gen = torch.Generator(device=device).manual_seed(scalar_seed)
                u_lookahead = torch.randn(tweedie_x0.shape, generator=gen, device=device, dtype=dtype)
                zeta = torch.randn(tweedie_x0.shape, generator=gen, device=device, dtype=dtype)

            if self.alpha_couple >= 1.0:
                u_base = u_lookahead
            else:
                alpha = self.alpha_couple
                u_base = ((1.0 - alpha ** 2) ** 0.5) * zeta + alpha * u_lookahead
            del zeta, u_lookahead  # Giải phóng ngay lập tức
        else:
            u_base = torch.randn(tweedie_x0.shape, device=device, dtype=dtype)
        
        # 2. Sinh mẫu Tweedie Đối ngẫu
        sigma_t = self.sigma_smc * rho_t
        tw_pos = torch.clamp(tweedie_x0 + sigma_t * u_base, -1.0, 1.0)
        tw_neg = torch.clamp(tweedie_x0 - sigma_t * u_base, -1.0, 1.0)
        del u_base  # Giải phóng ngay lập tức
        
        # 3. Đánh giá Reward 
        # Gọi batch 2N để tối ưu tốc độ, sau đó tách ra N và N
        batch_eval = torch.cat([tw_pos, tw_neg], dim=0)
        r_eval = reward_fn(batch_eval) # Trả về tensor [2N]
        r_pos, r_neg = r_eval[:N], r_eval[N:]
        
        # 4. Phạt Entropic Risk cho SMC (Định lý 5.2 bản rút gọn cho cặp đối ngẫu)
        v_smc = 0.5 * (r_pos + r_neg) - (self.beta_risk / 4.0) * ((0.5 * (r_pos - r_neg)) ** 2)
        
        # 5. Doob-Tweedie Innovation Residual (Triệt tiêu Double-Counting)
        # Chỉ đánh giá lượng giá trị MỚI chưa được cộng vào drift
        v_lidar_aligned = v_lidar.to(device=device, dtype=v_smc.dtype) if isinstance(v_lidar, torch.Tensor) else torch.tensor(v_lidar, device=device, dtype=v_smc.dtype)
        phi_residual = v_smc - (self.eta_drift * rho_t * v_lidar_aligned)
        
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
