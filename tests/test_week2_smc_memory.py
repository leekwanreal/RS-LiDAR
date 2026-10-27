import os
import sys
import gc
import unittest
import torch

# Ensure Diffusion-LiDAR-Sampling is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from rs_smc_engine import HarmonizedRSLiDARSMCEngine


class TestHarmonizedRSLiDARSMCEngine(unittest.TestCase):
    """
    Test suite for Task 1.5: Harmonized RS-Twisted-SMC Engine.
    Validates Doob-Tweedie Innovation Residual, 64-bit Seed Replay, Zero VRAM Leak,
    and ESS-triggered Multinomial Resampling.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N, self.C, self.H, self.W = 4, 4, 64, 64
        self.engine = HarmonizedRSLiDARSMCEngine(
            sigma_high=1.0,
            sigma_smc=0.25,
            beta_risk=0.5,
            eta_drift=0.4
        )

    def test_01_vram_and_memory_leak_free(self):
        """
        Test that compute_step has zero persistent VRAM overhead.
        Generates noise via 64-bit seed replay and immediately frees intermediate buffers.
        """
        x_t = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        tweedie_x0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        v_lidar = torch.randn((self.N,), device=self.device)

        def dummy_reward_fn(batch_tensor):
            # batch_tensor is [2N, C, H, W]
            return torch.randn(batch_tensor.shape[0], device=batch_tensor.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            phi_res, v_smc = self.engine.compute_step(
                x_t=x_t,
                tweedie_x0=tweedie_x0,
                v_lidar=v_lidar,
                seed_int=123456789,
                rho_t=0.6,
                reward_fn=dummy_reward_fn
            )

            self.assertEqual(phi_res.shape, (self.N,))
            self.assertEqual(v_smc.shape, (self.N,))

            # After cleanup of returned references, VRAM must return to baseline
            del phi_res, v_smc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()

            self.assertEqual(mem_before, mem_after,
                             f"VRAM leak detected! Before: {mem_before}, After: {mem_after}")
        else:
            # CPU verification: ensure internal engine state remains clean and stateless
            phi_res, v_smc = self.engine.compute_step(
                x_t=x_t,
                tweedie_x0=tweedie_x0,
                v_lidar=v_lidar,
                seed_int=123456789,
                rho_t=0.6,
                reward_fn=dummy_reward_fn
            )
            self.assertEqual(phi_res.shape, (self.N,))
            self.assertEqual(v_smc.shape, (self.N,))
            # Verify engine attributes hold no tensor cache
            for attr_name, val in self.engine.__dict__.items():
                self.assertFalse(isinstance(val, torch.Tensor),
                                 f"Engine must remain stateless; found cached tensor in {attr_name}")

    def test_02_resampling_trigger_on_weight_collapse(self):
        """
        Test that heavily collapsed log_weights trigger ESS < 0.5 * N and reset weights.
        """
        x_t = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        # Mark distinct particle signatures
        for i in range(self.N):
            x_t[i] = float(i + 1)

        log_weights = torch.tensor([10.0, 0.0, 0.0, -5.0], device=self.device)

        # Expected ESS:
        # w_norm[0] = exp(10)/(exp(10)+2+exp(-5)) ≈ 0.9999
        # ess ≈ 1.0001 < 0.5 * 4 = 2.0
        x_t_new, log_weights_new, resampled = self.engine.resample_if_needed(log_weights, x_t)

        self.assertTrue(resampled, "Resampling must be triggered when ESS < 0.5 * N.")
        self.assertEqual(x_t_new.shape, x_t.shape)
        self.assertTrue(torch.all(log_weights_new == 0.0),
                        "Log-weights must be reset to exact zeros (uniform) after resampling.")

        # Particle 0 has ~99.99% probability, so virtually all particles in x_t_new should be particle 0
        most_frequent_val = x_t_new[:, 0, 0, 0]
        self.assertTrue((most_frequent_val == 1.0).sum() >= 3,
                        "Particle 0 with weight 10.0 should dominate the resampled population.")

    def test_03_no_resampling_on_uniform_weights(self):
        """
        Test that uniform log_weights do NOT trigger resampling (ESS = N = 4.0 >= 2.0).
        """
        x_t = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        log_weights = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

        x_t_new, log_weights_new, resampled = self.engine.resample_if_needed(log_weights, x_t)

        self.assertFalse(resampled, "Resampling must NOT be triggered for uniform weights.")
        self.assertTrue(torch.equal(x_t_new, x_t), "Latent particles must remain unchanged.")
        self.assertTrue(torch.equal(log_weights_new, log_weights), "Log-weights must remain unchanged.")

    def test_04_doob_tweedie_innovation_residual_math(self):
        """
        Test theoretical Doob-Tweedie Innovation Residual formula:
        Phi_t^residual = V_t^SMC - eta_drift * rho_t * V_t^LiDAR
        where V_t^SMC = 0.5*(r_pos + r_neg) - (beta_risk / 4) * (0.5*(r_pos - r_neg))^2.
        """
        x_t = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        tweedie_x0 = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        v_lidar = torch.tensor([2.0, 3.0, 1.0, 4.0], device=self.device)

        # Mock deterministic reward returning specific values
        # Positive half: [2.0, 4.0, 6.0, 8.0]
        # Negative half: [2.0, 2.0, 4.0, 6.0]
        def mock_reward_fn(batch_tensor):
            r_pos = torch.tensor([2.0, 4.0, 6.0, 8.0], device=batch_tensor.device)
            r_neg = torch.tensor([2.0, 2.0, 4.0, 6.0], device=batch_tensor.device)
            return torch.cat([r_pos, r_neg], dim=0)

        rho_t = 0.5
        eta_drift = 0.4
        beta_risk = 0.5

        engine = HarmonizedRSLiDARSMCEngine(
            sigma_high=1.0,
            sigma_smc=0.25,
            beta_risk=beta_risk,
            eta_drift=eta_drift
        )

        phi_res, v_smc = engine.compute_step(
            x_t=x_t,
            tweedie_x0=tweedie_x0,
            v_lidar=v_lidar,
            seed_int=42,
            rho_t=rho_t,
            reward_fn=mock_reward_fn
        )

        # Manual theoretical calculation:
        # Particle 0: r_pos=2, r_neg=2 -> diff=0 -> v_smc = 2.0 - 0 = 2.0
        # Particle 1: r_pos=4, r_neg=2 -> mean=3, diff=2 -> half_diff=1 -> v_smc = 3.0 - (0.5/4)*(1^2) = 3.0 - 0.125 = 2.875
        # Particle 2: r_pos=6, r_neg=4 -> mean=5, diff=2 -> half_diff=1 -> v_smc = 5.0 - 0.125 = 4.875
        # Particle 3: r_pos=8, r_neg=6 -> mean=7, diff=2 -> half_diff=1 -> v_smc = 7.0 - 0.125 = 6.875
        expected_v_smc = torch.tensor([2.0, 2.875, 4.875, 6.875], device=self.device)
        self.assertTrue(torch.allclose(v_smc, expected_v_smc, atol=1e-5),
                        f"Expected v_smc: {expected_v_smc}, got: {v_smc}")

        # Subtraction of absorbed drift:
        # drift_sub = eta_drift * rho_t * v_lidar = 0.4 * 0.5 * [2, 3, 1, 4] = 0.2 * [2, 3, 1, 4] = [0.4, 0.6, 0.2, 0.8]
        # phi_res = v_smc - drift_sub = [2.0-0.4, 2.875-0.6, 4.875-0.2, 6.875-0.8] = [1.6, 2.275, 4.675, 6.075]
        expected_phi_res = expected_v_smc - (eta_drift * rho_t * v_lidar)
        self.assertTrue(torch.allclose(phi_res, expected_phi_res, atol=1e-5),
                        f"Expected phi_res: {expected_phi_res}, got: {phi_res}")

    def test_05_seed_replay_determinism_64bit(self):
        """
        Test that 64-bit Seed Replay produces exact bit-for-bit reproducibility.
        """
        x_t = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        tweedie_x0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        v_lidar = torch.randn((self.N,), device=self.device)

        def mock_reward_fn(batch_tensor):
            # Sum pixels to make reward depend directly on perturbations
            return batch_tensor.view(batch_tensor.shape[0], -1).sum(dim=-1)

        seed = 987654321
        phi1, v1 = self.engine.compute_step(x_t, tweedie_x0, v_lidar, seed, 0.5, mock_reward_fn)
        phi2, v2 = self.engine.compute_step(x_t, tweedie_x0, v_lidar, seed, 0.5, mock_reward_fn)

        self.assertTrue(torch.equal(phi1, phi2), "Seed replay must be bit-for-bit deterministic.")
        self.assertTrue(torch.equal(v1, v2), "v_smc must be bit-for-bit deterministic under same seed.")

        # Different seed must produce different perturbation
        phi3, _ = self.engine.compute_step(x_t, tweedie_x0, v_lidar, 11223344, 0.5, mock_reward_fn)
        self.assertFalse(torch.equal(phi1, phi3), "Different seeds must produce different perturbations.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
