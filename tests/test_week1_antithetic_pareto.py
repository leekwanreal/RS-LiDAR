import unittest

import torch

from lookahead_sampling import compute_entropic_antithetic_rewards, generate_smoothed_pixel_batch
from multi_reward_evaluator import compute_multi_reward_pareto_potential


class TestWeek1MathAndRegression(unittest.TestCase):
    """
    Antithetic smoothing batches, entropic-risk aggregation and the
    Local-Lipschitz multi-reward Pareto potential, plus a coupled-CRN regression check.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.N, self.C, self.H, self.W = 2, 4, 16, 16  # small shapes, fast on CPU
        self.M = 4
        self.sigma = 0.5
        self.images_0 = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)

    def test_1_antithetic_symmetry(self):
        """Antithetic noise sums to zero over the M samples."""
        torch.manual_seed(42)
        perturbed = generate_smoothed_pixel_batch(
            self.images_0, self.sigma, self.M, noise_mode="independent_antithetic"
        )
        # images_0 = 0, so perturbed equals sigma * u
        u = perturbed / self.sigma
        sum_u = u.sum(dim=0)
        self.assertTrue(
            torch.allclose(sum_u, torch.zeros_like(sum_u), atol=1e-6),
            "The antithetic batch must sum to 0."
        )

    def test_2_donsker_varadhan_penalty(self):
        """The entropic-risk penalty ranks a stable particle above a noisy one with the same mean."""
        # Two particles with the same mean; particle 2 has a much larger variance
        r_mc = torch.tensor([
            [10.0,  10.0],
            [10.0,  50.0],
            [10.0,  10.0],
            [10.0, -30.0]
        ], dtype=torch.float32)  # [M=4, N=2]

        # Particle 1: r_mean = 10, variance = 0
        # Particle 2: r_mean = 10, variance > 0
        stats = compute_entropic_antithetic_rewards(r_mc, sigma=1.0, beta_risk=0.5)
        r_corrected = stats["r_corrected"]

        self.assertAlmostEqual(
            r_corrected[0].item(), 10.0, places=4,
            msg="A zero-variance particle must not be penalized."
        )
        self.assertTrue(
            r_corrected[1].item() < 10.0,
            msg="A high-variance particle must receive the entropic penalty."
        )

    def test_3_local_lipschitz_pareto(self):
        """Lipschitz normalization balances rewards of very different scales."""
        torch.manual_seed(42)
        # Two rewards whose scales differ by 100x
        r1_mc = torch.randn(self.M, self.N)  # mean ~0, std ~1
        r2_mc = r1_mc * 100.0                # mean ~0, std ~100

        rewards_dict = {"reward_small": r1_mc, "reward_large": r2_mc}
        _, diagnostics = compute_multi_reward_pareto_potential(
            rewards_dict, sigma=1.0, kappa_lip=1.0, pareto_rho=0.5
        )

        lambda_small = diagnostics["reward_small_lambda"]
        lambda_large = diagnostics["reward_large_lambda"]

        # lambda of reward_large must be about 100x that of reward_small
        if isinstance(lambda_large, torch.Tensor):
            ratio = (lambda_large / lambda_small).mean().item()
        else:
            ratio = float(lambda_large) / float(lambda_small)

        self.assertTrue(
            90.0 < ratio < 110.0,
            f"Lambda does not match the reward scale ratio: {ratio}"
        )

    def test_4_zero_regression_crn(self):
        """Coupled CRN batch matches the original broadcast implementation bit for bit."""
        # 1. Current implementation in coupled_crn mode
        torch.manual_seed(1024)
        new_crn = generate_smoothed_pixel_batch(
            self.images_0, self.sigma, self.M, noise_mode="coupled_crn"
        )

        # 2. Reference implementation
        torch.manual_seed(1024)
        old_u = torch.randn((self.M, 1, self.C, self.H, self.W)).expand(self.M, self.N, self.C, self.H, self.W)
        old_crn = torch.clamp(self.images_0.unsqueeze(0) + self.sigma * old_u, -1.0, 1.0)

        self.assertTrue(
            torch.equal(new_crn, old_crn),
            "Coupled CRN batch differs from the reference implementation."
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
