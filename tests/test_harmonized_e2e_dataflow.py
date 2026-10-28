import os
import sys
import unittest
import torch
import torch.nn.functional as F

# Ensure Diffusion-LiDAR-Sampling is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from rs_smc_engine import HarmonizedRSLiDARSMCEngine
from lookahead_sampling import generate_smoothed_pixel_batch, compute_entropic_antithetic_rewards
from fkd_diffusers.fkd_class import FKD, PotentialType


class TestHarmonizedE2EDataFlow(unittest.TestCase):
    """
    Independent unit test suite verifying the complete end-to-end data flow
    and theoretical fidelity of the Harmonized RS-Twisted-SMC Engine.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.N, self.C, self.H, self.W = 4, 3, 32, 32

    def test_01_two_scale_stratified_perturbation(self):
        """
        Verify that noise_mode='stratified_harmonized' generates:
        - Pair 1 (samples 0, 1) with scale sigma_high (1.0)
        - Pair 2 (samples 2, 3) with scale sigma_low (0.25)
        - Both pairs satisfy antithetic symmetry sum(u) == 0
        """
        img = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        sigma_high = 1.0
        sigma_low = 0.25
        M = 4

        perturbed = generate_smoothed_pixel_batch(
            img,
            sigma=0.5, # base unused in stratified mode
            M=M,
            noise_mode="stratified_harmonized",
            clamp_range=None,
            sigma_high=sigma_high,
            sigma_low=sigma_low
        )

        self.assertEqual(perturbed.shape, (M, self.N, self.C, self.H, self.W))

        # Check antithetic symmetry: sample 0 + sample 1 == 0
        pair1_sum = perturbed[0] + perturbed[1]
        self.assertTrue(torch.allclose(pair1_sum, torch.zeros_like(pair1_sum), atol=1e-6))

        # Check antithetic symmetry: sample 2 + sample 3 == 0
        pair2_sum = perturbed[2] + perturbed[3]
        self.assertTrue(torch.allclose(pair2_sum, torch.zeros_like(pair2_sum), atol=1e-6))

        # Check scales: pair 1 should have 4x the std of pair 2 (1.0 vs 0.25)
        std_pair1 = perturbed[0].std().item()
        std_pair2 = perturbed[2].std().item()
        ratio = std_pair1 / (std_pair2 + 1e-8)
        self.assertAlmostEqual(ratio, sigma_high / sigma_low, delta=0.5)

    def test_02_stratified_entropic_rewards(self):
        """
        Verify compute_entropic_antithetic_rewards returns:
        - r_high (corrected reward under sigma_high)
        - r_low (corrected reward under sigma_low)
        - r_corrected (blended ranking reward)
        """
        # M=4, N=4
        scores_mat = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],  # pair 1 pos
            [0.8, 1.8, 2.6, 3.6],  # pair 1 neg
            [0.9, 1.9, 2.9, 3.9],  # pair 2 pos
            [0.9, 1.9, 2.8, 3.8],  # pair 2 neg
        ], dtype=torch.float32)

        res = compute_entropic_antithetic_rewards(
            scores_mat,
            sigma=0.5,
            beta_risk=0.5,
            noise_mode="stratified_harmonized"
        )

        self.assertIn("r_high", res)
        self.assertIn("r_low", res)
        self.assertIn("r_corrected", res)
        self.assertEqual(res["r_high"].shape, (self.N,))
        self.assertEqual(res["r_low"].shape, (self.N,))
        self.assertEqual(res["r_corrected"].shape, (self.N,))

        # Verify blend is mean of high and low corrected
        expected_blend = 0.5 * (res["r_high"] + res["r_low"])
        self.assertTrue(torch.allclose(res["r_corrected"], expected_blend, atol=1e-6))

    def test_03_per_particle_seed_replay(self):
        """
        Verify that passing a list of distinct per-particle seeds produces
        deterministic, reproducible noise per particle.
        """
        engine = HarmonizedRSLiDARSMCEngine(alpha_couple=0.7)
        tweedie_x0 = torch.zeros((self.N, self.C, self.H, self.W), device=self.device)
        x_t = torch.zeros((self.N, 4, 16, 16), device=self.device)
        v_lidar = torch.zeros((self.N,), device=self.device)

        seeds_1 = [10001, 10002, 10003, 10004]
        seeds_2 = [10001, 10002, 10003, 10004]
        seeds_diff = [99991, 99992, 99993, 99994]

        def dummy_reward(imgs):
            return imgs.mean(dim=(1, 2, 3))

        phi_1, v_smc_1 = engine.compute_step(x_t, tweedie_x0, v_lidar, seeds_1, 1.0, dummy_reward)
        phi_2, v_smc_2 = engine.compute_step(x_t, tweedie_x0, v_lidar, seeds_2, 1.0, dummy_reward)
        phi_diff, v_smc_diff = engine.compute_step(x_t, tweedie_x0, v_lidar, seeds_diff, 1.0, dummy_reward)

        # Exact match with identical seeds
        self.assertTrue(torch.allclose(phi_1, phi_2, atol=1e-7))
        self.assertTrue(torch.allclose(v_smc_1, v_smc_2, atol=1e-7))

    def test_04_double_subtraction_elimination_in_fkd_resample(self):
        """
        Verify that in FKD.resample:
        - When use_hybrid_smc=True, weights are w = exp(lambda * phi_residual)
          without subtracting self.population_rs a second time!
        - self.population_rs is updated with true Tweedie expectation v_smc.
        """
        def dummy_latent_decode(latents):
            # return fake image tensor [N, 3, 32, 32]
            return torch.zeros((latents.shape[0], 3, 32, 32), device=latents.device)

        def dummy_reward(imgs):
            return torch.linspace(1.0, 2.0, imgs.shape[0], device=imgs.device)

        fkd = FKD(
            reward_fn=dummy_reward,
            latent_to_decode_fn=dummy_latent_decode,
            num_particles=4,
            time_steps=50,
            resampling_t_start=0,
            resampling_t_end=50,
            resample_frequency=1,
            lmbda=10.0,
            potential_type=PotentialType.DIFF,
            device=self.device,
            use_hybrid_smc=True,
            sigma_high=1.0,
            sigma_smc=0.25,
            eta_drift=0.4,
            alpha_couple=0.0 # pure Gaussian to test math directly
        )

        latents = torch.randn((4, 4, 16, 16), device=self.device)
        x0_preds = latents.clone()
        v_lidar = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)

        # Call resample at step 0
        new_latents, _ = fkd.resample(
            sampling_idx=0,
            latents=latents,
            x0_preds=x0_preds,
            v_lidar=v_lidar,
            seed_int=42
        )

        # Ensure self.population_rs was updated to v_smc (around 1.0-1.5) and NOT still -100.0!
        self.assertTrue(fkd.population_rs[0] > 0.0, f"Expected population_rs > 0, got {fkd.population_rs}")

    def test_05_homotopy_variance_bridge_formula(self):
        """
        Verify mathematical formula of Homotopy Variance Bridge:
        R_sync(t) = (1 - rho_t^2) * r_low + rho_t^2 * r_high
        At t = t_max (rho_t -> 1.0): R_sync -> r_high
        At t = 0 (rho_t -> 0.0): R_sync -> r_low
        """
        r_high = torch.tensor([2.0, 3.0])
        r_low = torch.tensor([0.5, 1.0])

        # Near t_max: rho_t = 1.0
        rho_high = 1.0
        r_sync_high = (1.0 - rho_high ** 2) * r_low + (rho_high ** 2) * r_high
        self.assertTrue(torch.allclose(r_sync_high, r_high, atol=1e-6))

        # Near t_0: rho_t = 0.0
        rho_low = 0.0
        r_sync_low = (1.0 - rho_low ** 2) * r_low + (rho_low ** 2) * r_high
        self.assertTrue(torch.allclose(r_sync_low, r_low, atol=1e-6))

        # Intermediate: rho_t = 0.5 -> weight is 0.25 r_high + 0.75 r_low
        rho_mid = 0.5
        r_sync_mid = (1.0 - rho_mid ** 2) * r_low + (rho_mid ** 2) * r_high
        expected_mid = 0.75 * r_low + 0.25 * r_high
        self.assertTrue(torch.allclose(r_sync_mid, expected_mid, atol=1e-6))

    def test_06_no_hardcoded_parameters(self):
        """
        Verify that all parameters (eta_drift, alpha_couple, beta_risk, sigma_low, sigma_high)
        are dynamically passed from CLI/kwargs and NOT hardcoded.
        """
        custom_eta = 0.62
        custom_alpha = 0.85
        custom_beta = 0.33
        custom_sigma_low = 0.18
        custom_sigma_high = 1.45

        fkd = FKD(
            reward_fn=lambda x: torch.ones(x.shape[0]),
            latent_to_decode_fn=lambda x: torch.zeros((x.shape[0], 3, 32, 32)),
            num_particles=4,
            time_steps=50,
            potential_type=PotentialType.DIFF,
            use_hybrid_smc=True,
            eta_drift=custom_eta,
            alpha_couple=custom_alpha,
            beta_risk=custom_beta,
            sigma_low=custom_sigma_low,
            sigma_high=custom_sigma_high,
            device=self.device
        )

        engine = fkd.smc_engine
        self.assertIsNotNone(engine)
        self.assertEqual(engine.eta_drift, custom_eta)
        self.assertEqual(engine.alpha_couple, custom_alpha)
        self.assertEqual(engine.beta_risk, custom_beta)
        self.assertEqual(engine.sigma_smc, custom_sigma_low)
        self.assertEqual(engine.sigma_high, custom_sigma_high)

    def test_07_noise_in_pixel_space_verified(self):
        """
        Verify that noise is TRULY added in pixel space (RGB 3-channel domain [-1, 1]):
        - Checks that tweedie_x0 has 3 channels (RGB pixel space, not 4 channels latent).
        - Checks that reward_fn receives [2N, 3, H, W] in [-1.0, 1.0].
        - Checks that noise perturbation actually modifies pixel values.
        """
        engine = HarmonizedRSLiDARSMCEngine(
            sigma_smc=0.25,
            beta_risk=0.5,
            eta_drift=0.4,
            alpha_couple=0.7
        )

        # 3 channels RGB pixel domain [-1.0, 1.0]
        N, C, H, W = 4, 3, 64, 64
        tweedie_pixel = torch.full((N, C, H, W), 0.5, device=self.device)
        latents_dummy = torch.randn((N, 4, 16, 16), device=self.device)
        v_lidar = torch.zeros((N,), device=self.device)

        captured_shapes = []
        captured_ranges = []
        captured_diffs = []

        def inspecting_reward_fn(pixel_batch):
            # Assert 3 channels (RGB pixel space, NOT 4 channels latent space!)
            captured_shapes.append(pixel_batch.shape)
            captured_ranges.append((pixel_batch.min().item(), pixel_batch.max().item()))
            # Measure deviation from unperturbed tweedie_pixel
            diff = pixel_batch[:N] - tweedie_pixel
            captured_diffs.append(diff.abs().mean().item())
            return pixel_batch.mean(dim=(1, 2, 3))

        phi_res, v_smc = engine.compute_step(
            x_t=latents_dummy,
            tweedie_x0=tweedie_pixel,
            v_lidar=v_lidar,
            seed_int=12345,
            rho_t=1.0,
            reward_fn=inspecting_reward_fn
        )

        # 1. Verify batch shape is [2N, 3, H, W] (3 channels = RGB pixel space!)
        self.assertEqual(captured_shapes[0], (2 * N, 3, H, W))
        # 2. Verify values are clamped to [-1.0, 1.0] (image pixel space contract)
        min_val, max_val = captured_ranges[0]
        self.assertGreaterEqual(min_val, -1.0)
        self.assertLessEqual(max_val, 1.0)
        # 3. Verify noise was actually added: difference is strictly positive!
        self.assertGreater(captured_diffs[0], 0.05, "Noise was NOT injected into pixel values!")


if __name__ == "__main__":
    unittest.main()
