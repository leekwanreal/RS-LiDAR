import os
import sys
import unittest
import torch

# Ensure Diffusion-LiDAR-Sampling is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from lookahead_sampling import (
    generate_smoothed_pixel_batch,
    compute_entropic_antithetic_rewards
)


class TestDualOrderAntitheticAndEntropicRisk(unittest.TestCase):
    """
    Test suite for Task 1.1: Dual-Order Antithetic Smoothing & Entropic Risk Scorer
    Theorems 5.1 & 5.2 (Unified Stochastic Control Framework for ICML 2027).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = 8       # Number of particles
        self.C = 3       # Channels
        self.H = 64      # Height
        self.W = 64      # Width
        self.M = 4       # Monte Carlo samples (even)

    def test_01_antithetic_symmetry(self):
        """
        Verify Theorem 5.1 Antithetic Symmetry: sum_{m=1}^M u_m = 0 exactly.
        """
        # Using zero base image to isolate noise tensor u
        images_0 = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        sigma = 0.25

        perturbed = generate_smoothed_pixel_batch(
            images_0, sigma=sigma, M=self.M, noise_mode="independent_antithetic"
        )
        self.assertEqual(perturbed.shape, (self.M, self.N, self.C, self.H, self.W))

        # Reconstruct effective noise u
        u = perturbed / sigma
        half_M = self.M // 2

        # Pairwise antithetic symmetry: u_pos + u_neg = 0
        u_pos = u[:half_M]
        u_neg = u[half_M:]
        pairwise_sum = u_pos + u_neg
        self.assertTrue(
            torch.allclose(pairwise_sum, torch.zeros_like(pairwise_sum), atol=1e-6),
            f"Pairwise antithetic sum must be 0, max abs error: {pairwise_sum.abs().max().item()}"
        )

        # Total ensemble sum: sum_m u_m = 0
        total_sum = u.sum(dim=0)
        self.assertTrue(
            torch.allclose(total_sum, torch.zeros_like(images_0), atol=1e-6),
            f"Total antithetic noise sum must be 0, max abs error: {total_sum.abs().max().item()}"
        )

    def test_02_noise_modes_and_shapes(self):
        """
        Verify noise modes: independent_antithetic, independent_iid, coupled_crn.
        """
        images_0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device).clamp(-0.5, 0.5)
        sigma = 0.2

        # Mode 1: independent_antithetic
        batch_anti = generate_smoothed_pixel_batch(images_0, sigma, M=self.M, noise_mode="independent_antithetic")
        self.assertEqual(batch_anti.shape, (self.M, self.N, self.C, self.H, self.W))

        # Mode 2: independent_iid
        batch_iid = generate_smoothed_pixel_batch(images_0, sigma, M=self.M, noise_mode="independent_iid")
        self.assertEqual(batch_iid.shape, (self.M, self.N, self.C, self.H, self.W))

        # Mode 3: coupled_crn (CRN: same noise across all N particles within each MC draw)
        zeros_img = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        batch_crn = generate_smoothed_pixel_batch(zeros_img, sigma, M=self.M, noise_mode="coupled_crn")
        self.assertEqual(batch_crn.shape, (self.M, self.N, self.C, self.H, self.W))
        # Particle 0 and Particle 1 must receive the identical noise in CRN mode
        for m in range(self.M):
            diff = (batch_crn[m, 0] - batch_crn[m, 1]).abs().max().item()
            self.assertAlmostEqual(diff, 0.0, places=5)

        # Edge case: sigma <= 0 returns unperturbed images
        batch_zero_sigma = generate_smoothed_pixel_batch(images_0, sigma=0.0, M=self.M)
        for m in range(self.M):
            self.assertTrue(torch.allclose(batch_zero_sigma[m], images_0))

    def test_03_entropic_risk_penalty_and_elimination_of_winners_curse(self):
        """
        Verify Theorem 5.2:
        1. r_corrected <= r_mean for beta_risk >= 0.
        2. Stable particles beat noisy/spurious particles of equal sample mean.
        """
        # M=4, N=2
        # Particle 0: True high quality, stable across noise perturbations (mean=1.0, var=0)
        # Particle 1: Spurious outlier / noisy gradient (mean=1.0, pair 0: [2.5, -0.5], pair 1: [1.5, 0.5])
        # Note: half_M=2, so Row 0 and Row 2 form Pair 0 (+u0, -u0); Row 1 and Row 3 form Pair 1 (+u1, -u1)
        rewards_mc = torch.tensor([
            [1.0,  2.5],  # Pos 0 (+u0)
            [1.0,  1.5],  # Pos 1 (+u1)
            [1.0, -0.5],  # Neg 0 (-u0)
            [1.0,  0.5]   # Neg 1 (-u1)
        ], device=self.device, dtype=torch.float32)

        sigma = 0.25
        beta_risk = 0.5
        res = compute_entropic_antithetic_rewards(rewards_mc, sigma=sigma, beta_risk=beta_risk)

        r_corrected = res["r_corrected"]
        grad_energy_sq = res["grad_energy_sq"]
        pair_variance = res["pair_variance"]

        r_mean = rewards_mc.mean(dim=0)

        # 1. Inequality r_corrected <= r_mean
        self.assertTrue(
            torch.all(r_corrected <= r_mean + 1e-7),
            f"r_corrected must be <= r_mean, got {r_corrected} vs {r_mean}"
        )

        # 2. Particle 0 (zero variance) has zero penalty
        self.assertAlmostEqual(r_corrected[0].item(), 1.0, places=5)
        self.assertAlmostEqual(grad_energy_sq[0].item(), 0.0, places=5)

        # 3. Particle 1 (high variance) is penalized
        self.assertLess(r_corrected[1].item(), 1.0)
        self.assertGreater(grad_energy_sq[1].item(), 0.0)

        # 4. Winner's Curse elimination: Particle 0 strictly preferred over Particle 1
        self.assertGreater(
            r_corrected[0].item(), r_corrected[1].item(),
            f"Stable particle {r_corrected[0]} should rank higher than noisy particle {r_corrected[1]}"
        )

    def test_04_gradient_energy_non_negativity(self):
        """
        Verify Theorem 5.1: grad_energy_sq >= 0 and pair_variance >= 0.
        """
        random_rewards = torch.randn((self.M, 20), device=self.device)
        res = compute_entropic_antithetic_rewards(random_rewards, sigma=0.3, beta_risk=0.5)

        self.assertTrue(torch.all(res["grad_energy_sq"] >= 0.0))
        self.assertTrue(torch.all(res["pair_variance"] >= 0.0))

    def test_05_numerical_stability_under_fp32_and_zero_sigma(self):
        """
        Numerical stability: sigma -> 0 and FP16 input handling.
        """
        rewards_fp16 = torch.randn((self.M, 10), device=self.device, dtype=torch.float16)

        # Test with tiny sigma (division by sigma^2 + 1e-8)
        res_tiny_sigma = compute_entropic_antithetic_rewards(rewards_fp16, sigma=1e-12, beta_risk=0.5)
        self.assertFalse(torch.isnan(res_tiny_sigma["r_corrected"]).any())
        self.assertFalse(torch.isinf(res_tiny_sigma["r_corrected"]).any())

        # Test with zero sigma
        res_zero_sigma = compute_entropic_antithetic_rewards(rewards_fp16, sigma=0.0, beta_risk=0.5)
        self.assertFalse(torch.isnan(res_zero_sigma["r_corrected"]).any())
        self.assertFalse(torch.isinf(res_zero_sigma["r_corrected"]).any())

        # Test with M=2 (half_M = 1, curv_var = 0 without Bessel's division by zero)
        rewards_m2 = torch.tensor([[1.0, 2.0], [0.8, 1.8]], device=self.device)
        res_m2 = compute_entropic_antithetic_rewards(rewards_m2, sigma=0.25, beta_risk=0.5)
        self.assertFalse(torch.isnan(res_m2["r_corrected"]).any())

        # Test fallback for odd M (e.g. M=3)
        rewards_m3 = torch.tensor([[1.0], [0.8], [1.2]], device=self.device)
        res_m3 = compute_entropic_antithetic_rewards(rewards_m3, sigma=0.25, beta_risk=0.5)
        self.assertFalse(torch.isnan(res_m3["r_corrected"]).any())

    def test_06_legacy_bit_for_bit_equivalence(self):
        """
        Rule 1 & Rule 2: When beta_risk == 0.0, r_corrected must be identical
        to the standard Monte Carlo sample mean (legacy baseline).
        """
        rewards = torch.randn((self.M, 15), device=self.device)
        res_zero_risk = compute_entropic_antithetic_rewards(rewards, sigma=0.25, beta_risk=0.0)

        legacy_mean = rewards.mean(dim=0)
        self.assertTrue(
            torch.allclose(res_zero_risk["r_corrected"], legacy_mean, atol=1e-6),
            "When beta_risk=0.0, r_corrected must equal the legacy sample mean."
        )


if __name__ == "__main__":
    unittest.main()
