import torch
from typing import Dict, Tuple, Any
from lookahead_sampling import compute_entropic_antithetic_rewards


@torch.no_grad()
def compute_multi_reward_pareto_potential(
    multi_rewards_mc: Dict[str, torch.Tensor],
    sigma: float = 1.0,
    kappa_lip: float = 1.0,
    pareto_rho: float = 0.5,
    tau_p: float = 2.0,
    beta_risk: float = 0.5
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
    
    for name, r_mc in multi_rewards_mc.items():
        # Bước 1: Tính Entropic Risk từ M mẫu (Định lý 5.2)
        stats = compute_entropic_antithetic_rewards(r_mc, sigma=sigma, beta_risk=beta_risk)
        r_j = stats["r_corrected"]         # [N]
        grad_e_j = stats["grad_energy_sq"] # [N]
        
        # Bước 2: Chuẩn hóa Local Lipschitz (Định lý 6)
        mu_j = r_j.mean(dim=-1, keepdim=True)
        # unbiased=False fallback khi N=1, nhưng thường N >= 2
        var_j = r_j.var(dim=-1, unbiased=(r_j.shape[-1] > 1), keepdim=True) 
        
        # Gradient variance scale do M mẫu sinh ra
        lip_var_j = grad_e_j.mean(dim=-1, keepdim=True) * (sigma ** 2)
        
        # Lambda_j: Scale động ngăn Gradient Starvation (Định lý 6)
        lambda_j = torch.sqrt(var_j + kappa_lip * lip_var_j + 1e-6)
        
        r_norm = (r_j - mu_j) / lambda_j
        normalized_list.append(r_norm)
        
        # Lưu vết diagnostic
        diagnostics[f"{name}_lambda"] = lambda_j.item() if lambda_j.numel() == 1 else lambda_j
        diagnostics[f"{name}_mu"] = mu_j.item() if mu_j.numel() == 1 else mu_j
        diagnostics[f"{name}_r_norm"] = r_norm

    # Bước 3: Smooth Tchebycheff Soft-Min
    stacked = torch.stack(normalized_list, dim=0) # [J, N]
    diagnostics["stacked_normalized"] = stacked
    
    # Linear component (Kỳ vọng trung bình)
    linear_term = stacked.mean(dim=0)
    
    # Soft-Min component (Kéo đáy Pareto)
    # Sử dụng logsumexptrick: -1/tau * log(sum(exp(-tau * R)))
    # Khai triển dấu âm bên trong exp để logsumexp hoạt động an toàn tuyệt đối
    softmin_term = -(1.0 / tau_p) * torch.logsumexp(-tau_p * stacked, dim=0)
    
    # Gộp thế năng Smooth Tchebycheff
    pareto_potential = (1.0 - pareto_rho) * linear_term + pareto_rho * softmin_term
    
    diagnostics["linear_term"] = linear_term
    diagnostics["softmin_term"] = softmin_term
    
    return pareto_potential, diagnostics
