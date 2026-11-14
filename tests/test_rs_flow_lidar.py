import unittest
import torch

from rs_flow_lidar import RSFlowLiDARGuidance


class TestRSFlowLiDARVelocityGuidance(unittest.TestCase):
    """
    Tests of the lookahead velocity guidance for flow matching (FLUX.1):
    16-channel FLUX shapes, the [0.20, 0.98] guidance window, the pull towards high-reward samples,
    and the 1/t scaling of the correction.
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
        """Output shape on the 16-channel FLUX latent space."""
        delta_v = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.5
        )
        self.assertEqual(delta_v.shape, (self.B, self.C, self.H, self.W))
        self.assertTrue(torch.isfinite(delta_v).all(), "delta_v must be finite (no NaN/Inf).")
        self.assertGreater(torch.count_nonzero(delta_v).item(), 0, "delta_v must be non-zero at sigma_t=0.5.")

    def test_02_fast_path_early_return(self):
        """Outside the [0.20, 0.98] window the correction is exactly zero."""
        # 1. sigma_t = 0.05 (almost clean)
        delta_v_low = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.05
        )
        self.assertEqual(torch.count_nonzero(delta_v_low).item(), 0, 
                         "Must return 0 at sigma_t = 0.05.")

        # 2. sigma_t = 0.99 (almost pure noise)
        delta_v_high = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.99
        )
        self.assertEqual(torch.count_nonzero(delta_v_high).item(), 0, 
                         "Must return 0 at sigma_t = 0.99.")

    def test_03_direct_clean_shift_direction(self):
        """
        Direction of the correction delta_v = -(s_flow / t) * sum((w_r - w) * z0).
        In the backward Euler step z_{t - dt} = z_t - dt * (v_pred + delta_v),
        the term -dt * delta_v = +(dt * s_flow / t) * sum((w_r - w) * z0) is positive and
        pulls the latent towards the highest-reward samples.
        """
        # Sample k=2 has the highest reward
        rewards = torch.tensor([0.0, 0.0, 10.0, 0.0], device=self.device)
        delta_v = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, rewards, sigma_t=0.5
        )
        # delta_v points against sample 2,
        # i.e. <-delta_v, lookahead_z0[2]> > 0
        pull_force = -delta_v[0]  # direction of the backward Euler update
        target_particle = self.lookahead_z0[2]
        dot_product = torch.sum(pull_force * target_particle).item()
        self.assertGreater(dot_product, 0, "-delta_v must point towards the highest-reward sample.")

    def test_04_inverse_time_scaling(self):
        """1/t scaling: |delta_v| at t=0.5 is about 0.8 / 0.5 = 1.6 times |delta_v| at t=0.8."""
        v_05 = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.5
        )
        v_08 = self.guidance.compute_velocity_guidance(
            self.z_t, self.lookahead_z0, self.smoothed_rewards, sigma_t=0.8
        )
        norm_05 = torch.norm(v_05).item()
        norm_08 = torch.norm(v_08).item()
        # The correction scales with 1/t, so it is larger at t=0.5 than at t=0.8
        self.assertGreater(norm_05, norm_08, "The correction at t=0.5 must exceed the one at t=0.8 (1/t scaling).")
        scaling_ratio = norm_05 / norm_08
        self.assertAlmostEqual(scaling_ratio, 0.8 / 0.5, delta=0.1, 
                               msg="The ratio of corrections must be about 1.6 (1/t scaling).")

    def test_05_dynamic_channel_compatibility_4_channel(self):
        """Works unchanged on 4-channel latents (SD3 / SDXL)."""
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
