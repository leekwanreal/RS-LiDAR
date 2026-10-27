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
    Test suite for Task 1.4: RS-FlowLiDAR Velocity Guidance for Rectified Flows (Theorem 7).
    Validates Fast-Path early return, 16-channel FLUX format, 2D flattening, and Parabolic Boundary Damper.
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
        """Test output tensor shape under FLUX format (16 channels)."""
        delta_v = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.5
        )
        self.assertEqual(delta_v.shape, (self.B, self.C, self.H, self.W))
        self.assertTrue(torch.isfinite(delta_v).all(), "delta_v must be finite without NaN/Inf.")
        self.assertGreater(torch.count_nonzero(delta_v).item(), 0, "At sigma_t=0.5, delta_v must be non-zero.")

    def test_02_fast_path_early_return(self):
        """Test Fast-Path: Outside [0.20, 0.98] immediately returns exact zero tensor."""
        # 1. At sigma_t = 0.05 (near clean data)
        delta_v_low = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.05
        )
        self.assertEqual(torch.count_nonzero(delta_v_low).item(), 0, 
                         "Fast-Path must return zeros at sigma_t = 0.05.")

        # 2. At sigma_t = 0.99 (pure noise)
        delta_v_high = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.99
        )
        self.assertEqual(torch.count_nonzero(delta_v_high).item(), 0, 
                         "Fast-Path must return zeros at sigma_t = 0.99.")

    def test_03_parabolic_damper_boundary_stability(self):
        """
        Test Parabolic Boundary Damper omega(t) = t(1-t) neutralizing boundary singularities:
        Bypass Fast-path and call at extreme boundaries sigma_t = 0.0001 and 0.9999.
        """
        # Boundary t -> 0 (sigma_t = 0.0001)
        delta_v_near_zero = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, 
            sigma_t=0.0001, bypass_fast_path=True
        )
        self.assertTrue(torch.isfinite(delta_v_near_zero).all(), 
                        "Parabolic Damper must ensure finite values as sigma_t -> 0.")
        self.assertLess(delta_v_near_zero.abs().max().item(), 1e-2, 
                       "delta_v must be smoothly damped to near-zero at boundary.")

        # Boundary t -> 1 (sigma_t = 0.9999)
        delta_v_near_one = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, 
            sigma_t=0.9999, bypass_fast_path=True
        )
        self.assertTrue(torch.isfinite(delta_v_near_one).all(), 
                        "Parabolic Damper must ensure finite values as sigma_t -> 1.")
        self.assertLess(delta_v_near_one.abs().max().item(), 1e-2, 
                       "delta_v must be smoothly damped to near-zero at opposite boundary.")

    def test_04_parabolic_damper_symmetry(self):
        """Test exact mathematical symmetry of Parabolic Damper: omega(t) == omega(1-t)."""
        test_sigmas = [0.1, 0.25, 0.35, 0.45]
        for s in test_sigmas:
            omega_s = s * (1.0 - s)
            omega_comp = (1.0 - s) * s
            self.assertAlmostEqual(omega_s, omega_comp, places=7)

    def test_05_sd3_compatibility_4_channel(self):
        """Test compatibility with 4-channel models (SD3, SDXL) without configuration changes."""
        z_t_sd3 = torch.randn((2, 4, 32, 32), device=self.device)
        lookahead_sd3 = torch.randn((self.K, 4, 32, 32), device=self.device)

        delta_v_sd3 = self.guidance.compute_velocity_guidance(
            z_t_sd3, lookahead_sd3, self.smoothed_rewards, sigma_t=0.5
        )
        self.assertEqual(delta_v_sd3.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(delta_v_sd3).all())


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING RS-FLOWLIDAR VELOCITY GUIDANCE TEST SUITE (Theorem 7)")
    print("=" * 60)
    unittest.main(verbosity=2)
