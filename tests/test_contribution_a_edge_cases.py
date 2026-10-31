import os
import sys
import unittest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from lookahead_sampling import (
    generate_smoothed_pixel_batch,
    compute_entropic_antithetic_rewards
)


class TestContributionAEdgeCases(unittest.TestCase):
    """
    Exhaustive Edge Case Verification Suite for Contribution A:
    1. Pixel Clamping & Negative Sample Dynamics at Boundary ([-1, 1])
    2. Odd M Assertion vs Graceful Fallback
    3. Sigma = 0.0 & Zero-Division Avoidance (1e-8 eps)
    4. Beta_risk = 0.0 Pure Mean Equivalence
    5. Minimal Sample Size M=2 (half_M = 1 Bessel Correction NaN Avoidance)
    6. Numerical Precision (FP16, BF16, FP32 upcasting & downcasting)
    7. Constant Reward Landscape (Zero Variance & Zero Gradient Energy)
    8. Winner's Curse Suppression (Spurious Spike vs Stable Reward)
    """

    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("cpu")
        self.N, self.C, self.H, self.W = 4, 3, 32, 32

    def test_01_clamping_and_negative_sample_dynamics(self):
        """Edge Case 1: Pixel clamping behavior on boundary [-1, 1] with negative antithetic pairs."""
        # Case A: Pure interior image x0 in [-0.5, 0.5]
        x_interior = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        perturbed_interior = generate_smoothed_pixel_batch(
            x_interior, sigma=0.25, M=4, noise_mode="independent_antithetic", clamp_range=(-1.0, 1.0)
        )
        # All perturbed pixels must stay in [-1, 1]
        self.assertTrue((perturbed_interior >= -1.0).all() and (perturbed_interior <= 1.0).all())
        
        # Verify exact symmetry in interior when no clamping occurs:
        # positive sample (idx 0) and negative sample (idx 2)
        pos = perturbed_interior[0] - x_interior
        neg = perturbed_interior[2] - x_interior
        diff_interior = (pos + neg).abs().max().item()
        self.assertLess(diff_interior, 1e-6, "Interior perturbations must have exact zero sum.")

        # Case B: Extreme boundary image x0 = +1.0 (all white)
        x_boundary_high = torch.ones((self.N, self.C, self.H, self.W), device=self.device)
        perturbed_high = generate_smoothed_pixel_batch(
            x_boundary_high, sigma=1.0, M=4, noise_mode="independent_antithetic", clamp_range=(-1.0, 1.0)
        )
        self.assertTrue((perturbed_high >= -1.0).all() and (perturbed_high <= 1.0).all())
        # For positive sample u > 0, x0 + sigma*u was > 1.0, so clamped to 1.0
        # For negative sample -u < 0, x0 - sigma*u is < 1.0
        pos_high = perturbed_high[0]
        neg_high = perturbed_high[2]
        self.assertTrue((pos_high <= 1.0).all(), "Positive sample must be bounded by 1.0")
        self.assertTrue((neg_high >= -1.0).all(), "Negative sample must be bounded by -1.0")

        # Case C: Extreme boundary image x0 = -1.0 (all black)
        x_boundary_low = -torch.ones((self.N, self.C, self.H, self.W), device=self.device)
        perturbed_low = generate_smoothed_pixel_batch(
            x_boundary_low, sigma=1.0, M=4, noise_mode="independent_antithetic", clamp_range=(-1.0, 1.0)
        )
        self.assertTrue((perturbed_low >= -1.0).all() and (perturbed_low <= 1.0).all())

    def test_02_odd_m_handling(self):
        """Edge Case 2: Handling of odd M in batch generation and entropic scorer."""
        x0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        
        # In generate_smoothed_pixel_batch with independent_antithetic, M must be even
        with self.assertRaises(AssertionError):
            generate_smoothed_pixel_batch(x0, sigma=0.5, M=3, noise_mode="independent_antithetic")
            
        with self.assertRaises(AssertionError):
            generate_smoothed_pixel_batch(x0, sigma=0.5, M=5, noise_mode="independent_antithetic")

        # In compute_entropic_antithetic_rewards, odd M must safely fallback without crashing
        r_odd = torch.randn((5, self.N), device=self.device)
        res_odd = compute_entropic_antithetic_rewards(r_odd, sigma=0.5, beta_risk=0.5, noise_mode="independent_antithetic")
        self.assertEqual(res_odd["r_corrected"].shape, (self.N,))
        self.assertTrue(torch.isfinite(res_odd["r_corrected"]).all())
        self.assertTrue((res_odd["grad_energy_sq"] >= 0.0).all())

    def test_03_sigma_zero_and_numerical_stability(self):
        """Edge Case 3: Handling sigma = 0.0 and division by zero."""
        x0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        # When sigma = 0.0, output should repeat images without noise
        p_zero = generate_smoothed_pixel_batch(x0, sigma=0.0, M=4)
        for m in range(4):
            self.assertTrue(torch.equal(p_zero[m], x0))

        # In compute_entropic_antithetic_rewards with sigma = 0.0
        r_mc = torch.randn((4, self.N), device=self.device)
        res_sigma_zero = compute_entropic_antithetic_rewards(r_mc, sigma=0.0, beta_risk=0.5)
        self.assertTrue(torch.isfinite(res_sigma_zero["r_corrected"]).all(), "r_corrected must not be NaN/Inf when sigma=0")
        self.assertTrue(torch.isfinite(res_sigma_zero["grad_energy_sq"]).all(), "grad_energy_sq must not be NaN/Inf when sigma=0")

    def test_04_beta_risk_zero_neutrality(self):
        """Edge Case 4: beta_risk = 0.0 matches exact Monte Carlo mean."""
        r_mc = torch.randn((4, self.N), device=self.device) * 5.0
        res = compute_entropic_antithetic_rewards(r_mc, beta_risk=0.0)
        expected_mean = r_mc.mean(dim=0)
        self.assertTrue(torch.allclose(res["r_corrected"], expected_mean, atol=1e-6))

    def test_05_minimal_sample_size_m_equals_2(self):
        """Edge Case 5: M=2 (half_M = 1) avoids Bessel correction 1/(N-1) division by zero NaN."""
        # For M=2, pair_sums has only 1 sample, so pair_sums.var(unbiased=True) would be NaN
        r_mc_2 = torch.tensor([[1.5, 3.0], [2.5, 4.0]], device=self.device)
        res = compute_entropic_antithetic_rewards(r_mc_2, sigma=0.5, beta_risk=0.5)
        self.assertTrue(torch.isfinite(res["r_corrected"]).all())
        self.assertFalse(torch.isnan(res["r_corrected"]).any())
        self.assertFalse(torch.isnan(res["grad_energy_sq"]).any())
        self.assertFalse(torch.isnan(res["pair_variance"]).any())

    def test_06_dtype_fidelity_and_precision(self):
        """Edge Case 6: Numerical precision with float16, bfloat16 and float32."""
        # FP16 input
        r_fp16 = torch.tensor([[1.2, 2.3], [1.8, 2.7]], dtype=torch.float16, device=self.device)
        res_fp16 = compute_entropic_antithetic_rewards(r_fp16, sigma=0.5, beta_risk=0.5)
        self.assertEqual(res_fp16["r_corrected"].dtype, torch.float16)
        self.assertEqual(res_fp16["grad_energy_sq"].dtype, torch.float16)
        self.assertTrue(torch.isfinite(res_fp16["r_corrected"]).all())

        # BF16 input
        r_bf16 = torch.tensor([[1.2, 2.3], [1.8, 2.7]], dtype=torch.bfloat16, device=self.device)
        res_bf16 = compute_entropic_antithetic_rewards(r_bf16, sigma=0.5, beta_risk=0.5)
        self.assertEqual(res_bf16["r_corrected"].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(res_bf16["r_corrected"]).all())

    def test_07_constant_reward_landscape(self):
        """Edge Case 7: Constant reward landscape yields zero gradient energy and zero variance."""
        r_const = torch.full((4, self.N), 7.5, device=self.device)
        res = compute_entropic_antithetic_rewards(r_const, sigma=0.5, beta_risk=0.5)
        self.assertTrue(torch.allclose(res["r_corrected"], torch.full((self.N,), 7.5, device=self.device)))
        self.assertTrue(torch.allclose(res["grad_energy_sq"], torch.zeros(self.N, device=self.device)))
        self.assertTrue(torch.allclose(res["pair_variance"], torch.zeros(self.N, device=self.device)))

    def test_08_winners_curse_suppression(self):
        """Edge Case 8: Suppressing spurious high-variance reward spike vs stable high-reward candidate."""
        # Candidate 0: High raw reward spike with extreme variance [12.0, 2.0, 11.0, 3.0] -> mean = 7.0, high var
        # Candidate 1: Stable high reward [6.8, 6.8, 6.8, 6.8] -> mean = 6.8, zero var
        r_mc = torch.tensor([
            [12.0, 6.8],
            [11.0, 6.8],
            [2.0,  6.8],
            [3.0,  6.8]
        ], device=self.device)
        
        # Raw mean favors candidate 0: 7.0 > 6.8
        raw_mean = r_mc.mean(dim=0)
        self.assertGreater(raw_mean[0].item(), raw_mean[1].item())

        # Entropic risk penalty must penalize candidate 0 below candidate 1
        res = compute_entropic_antithetic_rewards(r_mc, sigma=0.5, beta_risk=0.5)
        self.assertLess(res["r_corrected"][0].item(), res["r_corrected"][1].item())
        self.assertGreater(res["grad_energy_sq"][0].item(), res["grad_energy_sq"][1].item())


if __name__ == "__main__":
    unittest.main()

