import unittest
import torch
import sys
import os

# Đảm bảo import được các module ở thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from lookahead_sampling import generate_smoothed_pixel_batch, compute_entropic_antithetic_rewards
    from multi_reward_evaluator import compute_multi_reward_pareto_potential
except ImportError:
    print("CẢNH BÁO: Không tìm thấy các module chính. Hãy chắc chắn bạn đã chạy từ root của dự án.")


class TestWeek1MathAndRegression(unittest.TestCase):
    """
    Test suite tích hợp Tuần 1: Dual-Order Antithetic, Entropic Risk Scorer và
    Local-Lipschitz Multi-Reward Pareto Evaluator. Đảm bảo bảo vệ không hồi quy (Zero-Regression).
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.N, self.C, self.H, self.W = 2, 4, 16, 16  # Kích thước nhỏ gọn, chạy siêu nhanh trên CPU
        self.M = 4
        self.sigma = 0.5
        self.images_0 = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)

    def test_1_antithetic_symmetry(self):
        """Kiểm chứng tính đối xứng tuyệt đối của Antithetic Noise (Định lý 5.1)."""
        torch.manual_seed(42)
        perturbed = generate_smoothed_pixel_batch(
            self.images_0, self.sigma, self.M, noise_mode="independent_antithetic"
        )
        # Vì images_0 = 0, perturbed chính là sigma * u
        u = perturbed / self.sigma
        sum_u = u.sum(dim=0)
        self.assertTrue(
            torch.allclose(sum_u, torch.zeros_like(sum_u), atol=1e-6),
            "Tổng của batch Antithetic phải bằng 0."
        )

    def test_2_donsker_varadhan_penalty(self):
        """Kiểm chứng Winner's Curse bị triệt tiêu bởi Entropic Risk (Định lý 5.2)."""
        # Giả lập 2 hạt có cùng trung bình, nhưng hạt 2 có phương sai cao hơn rất nhiều
        r_mc = torch.tensor([
            [10.0,  10.0],
            [10.0,  50.0],
            [10.0,  10.0],
            [10.0, -30.0]
        ], dtype=torch.float32)  # [M=4, N=2]

        # Hạt 1: r_mean = 10, phương sai = 0
        # Hạt 2: r_mean = 10, phương sai > 0
        stats = compute_entropic_antithetic_rewards(r_mc, sigma=1.0, beta_risk=0.5)
        r_corrected = stats["r_corrected"]

        self.assertAlmostEqual(
            r_corrected[0].item(), 10.0, places=4,
            msg="Hạt không có phương sai không bị phạt."
        )
        self.assertTrue(
            r_corrected[1].item() < 10.0,
            msg="Hạt có phương sai cao phải bị trừ điểm phạt Entropic."
        )

    def test_3_local_lipschitz_pareto(self):
        """Kiểm chứng Gradient Starvation được giải quyết qua chuẩn hóa Lipschitz (Định lý 6)."""
        torch.manual_seed(42)
        # Giả lập 2 reward có scale chênh lệch 100 lần
        r1_mc = torch.randn(self.M, self.N)  # mean ~0, std ~1
        r2_mc = r1_mc * 100.0                # mean ~0, std ~100

        rewards_dict = {"reward_small": r1_mc, "reward_large": r2_mc}
        _, diagnostics = compute_multi_reward_pareto_potential(
            rewards_dict, sigma=1.0, kappa_lip=1.0, pareto_rho=0.5
        )

        lambda_small = diagnostics["reward_small_lambda"]
        lambda_large = diagnostics["reward_large_lambda"]

        # Scale lambda của reward_large phải gấp khoảng 100 lần reward_small
        if isinstance(lambda_large, torch.Tensor):
            ratio = (lambda_large / lambda_small).mean().item()
        else:
            ratio = float(lambda_large) / float(lambda_small)

        self.assertTrue(
            90.0 < ratio < 110.0,
            f"Lambda chưa cân bằng đúng tỷ lệ độ dốc. Ratio hiện tại: {ratio}"
        )

    def test_4_zero_regression_crn(self):
        """Bảo vệ Backward Compatibility: Đảm bảo code cũ không lệch 1 bit."""
        # 1. Sinh bằng hàm mới với chế độ cũ
        torch.manual_seed(1024)
        new_crn = generate_smoothed_pixel_batch(
            self.images_0, self.sigma, self.M, noise_mode="coupled_crn"
        )

        # 2. Sinh bằng logic code cũ thuần túy
        torch.manual_seed(1024)
        old_u = torch.randn((self.M, 1, self.C, self.H, self.W)).expand(self.M, self.N, self.C, self.H, self.W)
        old_crn = torch.clamp(self.images_0.unsqueeze(0) + self.sigma * old_u, -1.0, 1.0)

        self.assertTrue(
            torch.equal(new_crn, old_crn),
            "LỖI HỒI QUY: Chế độ CRN mới tạo tensor khác bit so với mã nguồn gốc!"
        )


if __name__ == '__main__':
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ KHÔNG HỒI QUY - WEEK 1 (Toán học & Kiến trúc)")
    print("=" * 60)
    unittest.main(verbosity=2)
