import torch
from typing import Dict, Tuple, Any, Optional
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
    """
    Điều hòa các hàm thưởng không đồng nhất qua chuẩn hóa Local-Lipschitz 
    và tổng hợp bằng thế năng Smooth Tchebycheff Soft-Min (Định lý 6).
    
    Args:
        multi_rewards_mc: Dict mapping tên reward -> Tensor shape [M, N].
        sigma: Bán kính nhiễu smoothing.
        kappa_lip: Hệ số đóng góp của năng lượng gradient vào Lipschitz scale.
        pareto_rho: Hệ số trộn Pareto (0.0 = chỉ dùng mean, 1.0 = chỉ dùng soft-min).
        tau_p: Nhiệt độ mượt của soft-min (bắt buộc > 0).
        beta_risk: Hệ số phạt rủi ro Entropic.
        reward_weights: Trọng số tương đối tùy chọn cho từng reward {name: weight}.
        noise_mode: Chế độ nhiễu ('independent_antithetic', 'independent_iid', 'coupled_crn').
        
    Returns:
        Tuple chứa:
        - pareto_potential: Tensor shape [N] thế năng dẫn đường tổng hợp.
        - diagnostics: Dict chứa các thông tin chuẩn hóa để debug.
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
        # Bước 1: Tính Entropic Risk từ M mẫu (Định lý 5.2) với noise_mode phù hợp
        stats = compute_entropic_antithetic_rewards(r_mc, sigma=sigma, beta_risk=beta_risk, noise_mode=noise_mode)
        r_j = stats["r_corrected"]         # [N] (phần thưởng đã phạt rủi ro Entropic)
        grad_e_j = stats["grad_energy_sq"] # [N]
        
        # Bước 2: Chuẩn hóa Local Lipschitz theo Thứ nguyên Đồng bậc (Định lý 6)
        # Sử dụng kỳ vọng phần thưởng gốc r_raw_mean = r_mc.mean(dim=0) để tính phương sai
        # nhằm bảo toàn thứ nguyên [R]^2 cho Var(R_j), tránh méo mó bậc 4 do thành phần phạt bình phương
        r_raw_mean = r_mc.to(torch.float32).mean(dim=0)
        mu_j = r_j.mean(dim=-1, keepdim=True)
        var_j = r_raw_mean.var(dim=-1, unbiased=(r_mc.shape[-1] > 1), keepdim=True) 
        
        # Gradient variance scale do M mẫu sinh ra có thứ nguyên [R]^2:
        lip_var_j = grad_e_j.mean(dim=-1, keepdim=True) * (sigma ** 2)
        
        # Lambda_j: Scale động ngăn Gradient Starvation (Định lý 6) có thứ nguyên chuẩn [R]
        lambda_j = torch.sqrt(var_j + kappa_lip * lip_var_j + 1e-6)
        
        r_norm = (r_j - mu_j) / lambda_j
        normalized_list.append(r_norm)
        
        # Lưu vết diagnostic
        diagnostics[f"{name}_lambda"] = lambda_j.item() if lambda_j.numel() == 1 else lambda_j
        diagnostics[f"{name}_mu"] = mu_j.item() if mu_j.numel() == 1 else mu_j
        diagnostics[f"{name}_r_norm"] = r_norm

    # Bước 3: Trọng số phân bổ người dùng
    device = normalized_list[0].device
    J = len(metric_names)
    if reward_weights is not None:
        raw_w = torch.tensor([float(reward_weights.get(k, 1.0)) for k in metric_names], device=device, dtype=torch.float32)
        w_norm = raw_w / (raw_w.sum() + 1e-8)
    else:
        w_norm = torch.ones(J, device=device, dtype=torch.float32) / float(J)

    # Bước 4: Smooth Tchebycheff Soft-Min
    stacked = torch.stack(normalized_list, dim=0) # [J, N]
    diagnostics["stacked_normalized"] = stacked
    
    # Linear component (Kỳ vọng trung bình có trọng số)
    linear_term = torch.sum(w_norm.unsqueeze(-1) * stacked, dim=0)
    
    # Soft-Min component (Kéo đáy Pareto có trọng số)
    # logsumexptrick với log(w_j): -1/tau * log(sum(w_j * exp(-tau * R_j)))
    log_w = torch.log(w_norm.clamp(min=1e-8)).unsqueeze(-1)  # [J, 1]
    softmin_term = -(1.0 / tau_p) * torch.logsumexp(-tau_p * stacked + log_w, dim=0)
    
    # Gộp thế năng Smooth Tchebycheff
    pareto_potential = (1.0 - pareto_rho) * linear_term + pareto_rho * softmin_term
    
    diagnostics["linear_term"] = linear_term
    diagnostics["softmin_term"] = softmin_term
    diagnostics["weights"] = w_norm
    
    return pareto_potential, diagnostics


class LocalLipschitzParetoEvaluator:
    """
    Object-oriented wrapper for Local-Lipschitz Multi-Reward Pareto Evaluator (Theorem 6).
    """
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
