import os
import sys
import unittest
import torch

# Ensure Diffusion-LiDAR-Sampling is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from rs_flow_lidar import RSFlowLiDARGuidance


class TestRSFlowLiDARVelocityGuidance(unittest.TestCase):
    """
    Bộ kiểm thử chuẩn mực cho RS-FlowLiDAR Velocity Guidance trong Flow Matching (FLUX.1):
    Xác minh: định dạng 16 kênh FLUX, cơ chế Fast-path [0.20, 0.98], hướng lực kéo tới ảnh thưởng cao,
    và hệ số co giãn tỉ lệ nghịch với thời gian 1/t (hoàn toàn theo lý thuyết vật lý dòng chảy).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.B, self.C, self.H, self.W = 1, 16, 64, 64  # FLUX.1 latent format
        self.K = 4                                     # Lookahead candidate particles

        self.z_t = torch.randn((self.B, self.C, self.H, self.W), device=self.device)
        self.lookahead_z0 = torch.randn((self.K, self.C, self.H, self.W), device=self.device)
        self.smoothed_rewards = torch.tensor([1.2, 0.8, 1.5, 0.4], device=self.device)

        self.guidance = RSFlowLiDARGuidance(s_flow=12.5, lambda_r=5000.0)

    def test_01_flux_output_shape(self):
        """Kiểm tra định dạng tensor đầu ra trên không gian FLUX 16 kênh."""
        delta_v = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.5
        )
        self.assertEqual(delta_v.shape, (self.B, self.C, self.H, self.W))
        self.assertTrue(torch.isfinite(delta_v).all(), "delta_v phải hữu hạn, không có NaN/Inf.")
        self.assertGreater(torch.count_nonzero(delta_v).item(), 0, "Tại sigma_t=0.5, delta_v phải khác 0.")

    def test_02_fast_path_early_return(self):
        """Kiểm tra Fast-Path: Ngoài cửa sổ [0.20, 0.98] lập tức trả về tensor 0 tuyệt đối."""
        # 1. Tại sigma_t = 0.05 (gần ảnh sạch)
        delta_v_low = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.05
        )
        self.assertEqual(torch.count_nonzero(delta_v_low).item(), 0, 
                         "Fast-Path phải trả về 0 tại sigma_t = 0.05.")

        # 2. Tại sigma_t = 0.99 (nhiễu thuần túy)
        delta_v_high = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.99
        )
        self.assertEqual(torch.count_nonzero(delta_v_high).item(), 0, 
                         "Fast-Path phải trả về 0 tại sigma_t = 0.99.")

    def test_03_direct_clean_shift_direction(self):
        """
        Kiểm tra hướng lực đẩy: Vector delta_v = - (s_flow / t) * sum( (w_r - w) * z0 ).
        Trong bước tích phân Euler ngược: z_{t - dt} = z_t - dt * (v_pred + delta_v),
        thành phần - dt * delta_v = + (dt * s_flow / t) * sum((w_r - w) * z0) mang dấu DƯƠNG (+),
        kéo hạt latent về phía các hạt có phần thưởng cao nhất.
        """
        # Cho hạt k=2 có phần thưởng vượt trội
        rewards = torch.tensor([0.0, 0.0, 10.0, 0.0], device=self.device)
        delta_v = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, rewards, sigma_t=0.5
        )
        # Vector gia số vận tốc delta_v phải mang hướng âm của hạt 2
        # Tức là tích vô hướng giữa -delta_v và lookahead_z0[2] phải dương
        pull_force = -delta_v[0]  # Hướng kéo của bước tích phân lùi
        target_particle = self.lookahead_z0[2]
        dot_product = torch.sum(pull_force * target_particle).item()
        self.assertGreater(dot_product, 0, "Bước tích phân lùi -delta_v phải kéo thuận hướng hạt có reward cao nhất!")

    def test_04_inverse_time_scaling(self):
        """Kiểm tra hệ số co giãn tỉ lệ nghịch với thời gian: tỉ lệ giữa delta_v tại t=0.5 và t=0.8 xấp xỉ 0.8/0.5 = 1.6 lần."""
        v_05 = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.5
        )
        v_08 = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.8
        )
        norm_05 = torch.norm(v_05).item()
        norm_08 = torch.norm(v_08).item()
        # Vì t=0.5 nhỏ hơn t=0.8, vận tốc lái tại t=0.5 phải lớn hơn tại t=0.8 tỉ lệ nghịch với t
        self.assertGreater(norm_05, norm_08, "Vận tốc lái tại t=0.5 phải lớn hơn tại t=0.8 do chia cho thời gian t.")
        scaling_ratio = norm_05 / norm_08
        self.assertAlmostEqual(scaling_ratio, 0.8 / 0.5, delta=0.1, 
                               msg="Tỉ lệ vận tốc lái phải xấp xỉ chính xác 1.6 (tỉ lệ nghịch với t).")

    def test_05_dynamic_channel_compatibility_4_channel(self):
        """Kiểm tra tính tương thích động với không gian 4 kênh (SD3 / SDXL) mà không cần cấu hình lại."""
        z_t_4c = torch.randn((2, 4, 32, 32), device=self.device)
        lookahead_4c = torch.randn((self.K, 4, 32, 32), device=self.device)

        delta_v_4c = self.guidance.compute_velocity_guidance(
            z_t_4c, lookahead_4c, self.smoothed_rewards, sigma_t=0.5
        )
        self.assertEqual(delta_v_4c.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(delta_v_4c).all())


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING RS-FLOWLIDAR VELOCITY GUIDANCE TEST SUITE (Clean Flow Matching)")
    print("=" * 60)
    unittest.main(verbosity=2)
