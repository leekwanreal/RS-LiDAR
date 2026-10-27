import os
import sys
import unittest
import torch

# Ensure Diffusion-LiDAR-Sampling is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from multi_reward_evaluator import compute_multi_reward_pareto_potential


class TestLocalLipschitzMultiRewardPareto(unittest.TestCase):
    """
    Test suite for Task 1.2: Local-Lipschitz Multi-Reward Pareto Evaluator (Theorem 6).
    Verifies Lipschitz calibration, gradient norm starvation prevention,
    Smooth Tchebycheff Soft-Min Pareto optimality, and numerical stability.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = 4  # Monte Carlo lookahead samples
        self.N = 2  # Particles

    def test_01_gradient_norm_calibration_and_starvation_prevention(self):
        """
        Verify that 100x scale discrepancy between rewards is calibrated by Lambda_j
        so that both rewards achieve Theta(1) normalized variance and equal influence.
        """
        # Reward 1: small scale hovering around 1.0 (e.g. ImageReward)
        # Values across 4 MC samples for 2 particles
        reward_1_mc = torch.tensor([
            [1.2, 0.8],
            [1.1, 0.9],
            [0.8, 1.2],
            [0.9, 1.1]
        ], device=self.device, dtype=torch.float32)

        # Reward 2: large scale hovering around 100.0 (100x larger gradient/variance)
        reward_2_mc = torch.tensor([
            [120.0,  80.0],
            [110.0,  90.0],
            [ 80.0, 120.0],
            [ 90.0, 110.0]
        ], device=self.device, dtype=torch.float32)

        multi_rewards = {
            "image_reward": reward_1_mc,
            "large_metric": reward_2_mc
        }

        potential, diagnostics = compute_multi_reward_pareto_potential(
            multi_rewards,
            sigma=0.25,
            kappa_lip=1.0,
            pareto_rho=0.5,
            tau_p=2.0,
            beta_risk=0.5
        )

        lambda_1 = diagnostics["image_reward_lambda"]
        lambda_2 = diagnostics["large_metric_lambda"]

        print(f"\n[Test 1 Diagnostics] lambda_1 (image_reward): {lambda_1:.4f}, lambda_2 (large_metric): {lambda_2:.4f}")
        print(f"[Test 1 Diagnostics] Ratio lambda_2 / lambda_1: {lambda_2 / lambda_1:.2f}x (expected ~100x)")

        # Assert Lambda_2 is roughly 100x larger than Lambda_1 (+/- 10%)
        ratio = lambda_2 / lambda_1
        self.assertGreater(ratio, 80.0, f"Lambda ratio should be ~100x, got {ratio:.2f}")
        self.assertLess(ratio, 120.0, f"Lambda ratio should be ~100x, got {ratio:.2f}")

        # Assert normalized rewards are on the exact same scale Theta(1)
        r_norm_1 = diagnostics["image_reward_r_norm"]
        r_norm_2 = diagnostics["large_metric_r_norm"]
        self.assertTrue(torch.allclose(r_norm_1.abs(), r_norm_2.abs(), atol=1e-3),
                        "Normalized rewards must have identical unit magnitude!")

    def test_02_pareto_monotonicity_and_dominance(self):
        """
        Verify Pareto monotonicity: if Particle 0 strictly dominates Particle 1
        on all reward objectives, its Pareto potential must be strictly higher.
        """
        # Particle 0 has strictly higher scores across both rewards
        r1 = torch.tensor([
            [2.0, 0.5],
            [1.8, 0.4],
            [2.2, 0.6],
            [2.0, 0.5]
        ], device=self.device, dtype=torch.float32)

        r2 = torch.tensor([
            [150.0, 50.0],
            [140.0, 45.0],
            [160.0, 55.0],
            [150.0, 50.0]
        ], device=self.device, dtype=torch.float32)

        potential, _ = compute_multi_reward_pareto_potential(
            {"r1": r1, "r2": r2},
            sigma=0.25,
            pareto_rho=0.5
        )

        self.assertGreater(
            potential[0].item(), potential[1].item(),
            f"Dominant Particle 0 ({potential[0].item():.4f}) must beat Particle 1 ({potential[1].item():.4f})"
        )

    def test_03_smooth_tchebycheff_bottleneck_protection(self):
        """
        Verify that Smooth Tchebycheff Soft-Min protects against single-metric reward hacking:
        A balanced particle outperforms an extreme reward-hacked particle with a collapsed weak objective.
        """
        # Particle 0 (Balanced): Good on both r1 (8.0) and r2 (8.0)
        # Particle 1 (Reward-Hacked): Super high on r1 (12.0), but collapsed on r2 (-5.0)
        # Particle 2 (Average baseline): Average on r1 (4.0) and r2 (4.0)
        r1 = torch.tensor([[8.0, 12.0, 4.0]] * 4, device=self.device, dtype=torch.float32)
        r2 = torch.tensor([[8.0, -5.0, 4.0]] * 4, device=self.device, dtype=torch.float32)

        # With high pareto_rho (e.g. 0.8), Soft-Min dominates and pulls up the weakest link
        potential, diag = compute_multi_reward_pareto_potential(
            {"r1": r1, "r2": r2},
            sigma=0.25,
            pareto_rho=0.8,
            tau_p=2.0
        )

        print(f"\n[Test 3 Bottleneck] Balanced Particle 0: {potential[0].item():.4f}, Hacked Particle 1: {potential[1].item():.4f}, Baseline P2: {potential[2].item():.4f}")
        self.assertGreater(
            potential[0].item(), potential[1].item(),
            "Soft-Min must penalize reward-hacked particle with collapsed weak objective!"
        )

    def test_04_numerical_stability_under_extreme_negative_values(self):
        """
        Verify logsumexp stability under extreme negative scores (e.g. -100.0)
        where manual exp(-tau * R) would catastrophically overflow to +Inf.
        """
        extreme_r1 = torch.tensor([[-100.0, 50.0]] * 4, device=self.device, dtype=torch.float32)
        extreme_r2 = torch.tensor([[10.0, -80.0]] * 4, device=self.device, dtype=torch.float32)

        potential, _ = compute_multi_reward_pareto_potential(
            {"extreme_1": extreme_r1, "extreme_2": extreme_r2},
            sigma=0.25,
            tau_p=2.0
        )

        self.assertFalse(torch.isnan(potential).any(), "Potential must not contain NaN!")
        self.assertFalse(torch.isinf(potential).any(), "Potential must not contain Inf!")

    def test_05_single_particle_and_single_reward_edge_cases(self):
        """
        Verify graceful execution for single particle (N=1) and single reward (J=1).
        """
        # N=1 edge case
        single_particle_r = torch.tensor([[1.0], [1.1], [0.9], [1.0]], device=self.device, dtype=torch.float32)
        pot_single_p, _ = compute_multi_reward_pareto_potential(
            {"r": single_particle_r},
            sigma=0.25
        )
        self.assertEqual(pot_single_p.shape, (1,))
        self.assertFalse(torch.isnan(pot_single_p).any())

        # J=1 edge case with N=4
        single_reward_mc = torch.randn((4, 4), device=self.device)
        pot_single_j, _ = compute_multi_reward_pareto_potential(
            {"single_reward": single_reward_mc},
            sigma=0.25
        )
        self.assertEqual(pot_single_j.shape, (4,))
        self.assertFalse(torch.isnan(pot_single_j).any())


if __name__ == "__main__":
    unittest.main()
