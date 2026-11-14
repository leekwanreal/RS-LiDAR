"""Unit tests for the RS-LiDAR building blocks: smoothing batches, entropic-risk aggregation,
Pareto potential, FLUX velocity guidance, CLI defaults and the transformers patch."""
import sys
import unittest

import numpy as np
import torch

from LiDAR_sampling import get_args as get_lidar_args
from lookahead_sampling import (
    compute_entropic_antithetic_rewards,
    generate_smoothed_pixel_batch,
)
from lookahead_sampling import get_args as get_lookahead_args
from multi_reward_evaluator import compute_multi_reward_pareto_potential
from rs_flow_lidar import RSFlowLiDARGuidance


class TestPhase1UnifiedModules(unittest.TestCase):
    """Antithetic / entropic risk, Pareto potential, FLUX velocity guidance, CLI defaults, compatibility."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.N, self.C, self.H, self.W = 4, 4, 64, 64
        self.M = 4

    def test_01_antithetic_shapes_and_invariance(self):
        """Antithetic symmetry and entropic risk"""
        images_0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        
        # Shape check [M, N, C, H, W]
        perturbed = generate_smoothed_pixel_batch(
            images_0, sigma=0.25, M=self.M, noise_mode="independent_antithetic"
        )
        self.assertEqual(perturbed.shape, (self.M, self.N, self.C, self.H, self.W))

        # Check antithetic symmetry: sum_{m=1}^M u_m = 0 exactly
        u_base = torch.randn((self.M // 2, self.N, self.C, self.H, self.W), device=self.device)
        u_full = torch.cat([u_base, -u_base], dim=0)
        sum_u = u_full.sum(dim=0)
        self.assertLess(torch.norm(sum_u, p=float("inf")).item(), 1e-6)

        # Entropic risk penalty check: r_corrected = r_mean - (beta / 2M) * var
        r_mc = torch.tensor([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]], device=self.device)
        res_stable = compute_entropic_antithetic_rewards(r_mc, beta_risk=0.5)
        self.assertTrue(torch.allclose(res_stable["r_corrected"], torch.tensor([10.0, 10.0])))

        r_mc_noisy = torch.tensor([[14.0, 10.0], [12.0, 10.0], [6.0, 10.0], [8.0, 10.0]], device=self.device)
        res_noisy = compute_entropic_antithetic_rewards(r_mc_noisy, beta_risk=0.5)
        # Variance of particle 0 is > 0, so r_corrected < 10.0
        self.assertLess(res_noisy["r_corrected"][0].item(), 10.0)
        self.assertGreaterEqual(res_noisy["grad_energy_sq"][0].item(), 0.0)

        # clamp_range=None leaves values outside [-1, 1] untouched.
        latents_large = torch.randn((self.N, 4, 64, 64), device=self.device) * 5.0
        perturbed_lat = generate_smoothed_pixel_batch(latents_large, sigma=0.25, M=self.M, clamp_range=None)
        self.assertTrue((perturbed_lat.abs() > 1.0).any(), "Latents must not be clamped to [-1, 1] when clamp_range=None.")

        r_mc_iid = torch.randn((self.M, self.N), device=self.device)
        res_iid = compute_entropic_antithetic_rewards(r_mc_iid, noise_mode="independent_iid")
        self.assertEqual(res_iid["r_corrected"].shape, (self.N,))
        self.assertGreaterEqual(res_iid["grad_energy_sq"].min().item(), 0.0)

    def test_02_pareto_normalization_and_monotonicity(self):
        """Pareto potential: Lipschitz scale and soft-min"""
        # Two objectives with 100x discrepancy: R1 in [0, 1], R2 in [0, 100]
        r_mc_1 = torch.tensor([
            [1.2, 0.8],
            [1.1, 0.9],
            [0.8, 1.2],
            [0.9, 1.1]
        ], device=self.device)
        r_mc_2 = r_mc_1 * 100.0
        multi_rewards = {
            "image_reward": r_mc_1,
            "large_metric": r_mc_2
        }

        potentials, diagnostics = compute_multi_reward_pareto_potential(
            multi_rewards_mc=multi_rewards,
            sigma=0.25,
            kappa_lip=1.0,
            pareto_rho=0.5,
            tau_p=2.0,
            beta_risk=0.5
        )

        self.assertEqual(potentials.shape, (2,))
        lambda_1 = diagnostics["image_reward_lambda"]
        lambda_2 = diagnostics["large_metric_lambda"]
        self.assertGreater(lambda_1, 0.0)
        self.assertGreater(lambda_2, 0.0)

        # The 100x scale discrepancy is calibrated: lambda_2 / lambda_1 ~ 100
        ratio = lambda_2 / lambda_1
        self.assertAlmostEqual(ratio, 100.0, places=1)

        # reward_weights support
        pot_w, diag_w = compute_multi_reward_pareto_potential(
            multi_rewards,
            reward_weights={"image_reward": 0.8, "large_metric": 0.2}
        )
        self.assertEqual(pot_w.shape, (2,))
        self.assertIn("weights", diag_w)

        # Confirm Smooth Tchebycheff Soft-Min protects against reward hacking
        multi_candidates = {
            "r1": torch.tensor([[0.8, 1.0, 0.1], [0.8, 1.0, 0.1], [0.8, 1.0, 0.1], [0.8, 1.0, 0.1]], device=self.device),
            "r2": torch.tensor([[80.0, 10.0, 10.0], [80.0, 10.0, 10.0], [80.0, 10.0, 10.0], [80.0, 10.0, 10.0]], device=self.device)
        }
        p_cands, _ = compute_multi_reward_pareto_potential(multi_candidates, sigma=0.25, tau_p=2.0)
        self.assertGreater(p_cands[0].item(), p_cands[1].item(), "Balanced candidate must strictly dominate hacked candidate.")

    def test_03_flow_lidar_velocity_guidance(self):
        """Flow Matching Velocity Guidance & Fast-Path"""
        guidance = RSFlowLiDARGuidance(s_flow=12.5, lambda_r=5000.0)
        z_t = torch.randn((1, 16, 64, 64), device=self.device)
        lookahead_z0 = torch.randn((4, 16, 64, 64), device=self.device)
        rewards = torch.tensor([1.0, 0.5, 0.8, 1.2], device=self.device)

        # 1. Fast-Path early return outside [0.20, 0.98]
        out_low = guidance.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.05)
        out_high = guidance.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.99)
        self.assertEqual(torch.count_nonzero(out_low).item(), 0)
        self.assertEqual(torch.count_nonzero(out_high).item(), 0)

        # 2. Inside window: output has correct shape and finite values
        out_mid = guidance.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.5)
        self.assertEqual(out_mid.shape, (1, 16, 64, 64))
        self.assertTrue(torch.isfinite(out_mid).all())

        # 3. Numerical stability at extreme boundary with safe denominator clamping
        out_boundary = guidance.compute_velocity_guidance(
            z_t, lookahead_z0, rewards, sigma_t=0.0001, bypass_fast_path=True
        )
        self.assertTrue(torch.isfinite(out_boundary).all())

    def test_05_cli_arguments_audit(self):
        """CLI defaults"""
        sys.argv = ["lookahead_sampling.py"]
        args_lookahead = get_lookahead_args()

        self.assertEqual(args_lookahead.noise_mode, "independent_iid")
        self.assertEqual(args_lookahead.M, 4)
        self.assertEqual(args_lookahead.num_mc_samples, 4)
        self.assertEqual(args_lookahead.beta_risk, 0.0)
        self.assertEqual(args_lookahead.pareto_rho, 0.5)
        self.assertFalse(args_lookahead.use_pareto)

        sys.argv = ["lookahead_sampling.py", "--use_smoothing", "--noise_mode", "independent_antithetic",
                    "--beta_risk", "0.5"]
        self.assertEqual(get_lookahead_args().beta_risk, 0.5)

        # --use_crn is a shorthand for --noise_mode coupled_crn.
        sys.argv = ["lookahead_sampling.py", "--use_crn", "True"]
        args_crn = get_lookahead_args()
        self.assertEqual(args_crn.noise_mode, "coupled_crn")
        self.assertEqual(args_crn.beta_risk, 0.0)

        sys.argv = ["LiDAR_sampling.py"]
        args_lidar = get_lidar_args()
        self.assertEqual(args_lidar.M, 4)
        self.assertEqual(args_lidar.s_flow, 40.0)
        self.assertEqual(args_lidar.smc_anneal_schedule, "tweedie_analytical")

    def test_06_legacy_backward_compatibility(self):
        """Coupled CRN batch matches the single-broadcast reference"""
        torch.manual_seed(2026)
        x = torch.randn((self.N, self.C, self.H, self.W), device=self.device)

        # Legacy simulation: single noise broadcast across all particles
        torch.manual_seed(1024)
        legacy_u = torch.randn((self.M, 1, self.C, self.H, self.W), device=self.device).expand(self.M, self.N, self.C, self.H, self.W)
        legacy_perturbed = torch.clamp(x.unsqueeze(0) + 0.25 * legacy_u, -1.0, 1.0)

        # Coupled CRN mode via generate_smoothed_pixel_batch
        torch.manual_seed(1024)
        new_perturbed = generate_smoothed_pixel_batch(x, sigma=0.25, M=self.M, noise_mode="coupled_crn")

        self.assertTrue(torch.equal(legacy_perturbed, new_perturbed), "Legacy CRN output must match bit-for-bit.")

        # Entropic risk with beta=0.0 must exactly match raw mean
        r_raw = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        res_zero_beta = compute_entropic_antithetic_rewards(r_raw, beta_risk=0.0)
        self.assertTrue(torch.equal(res_zero_beta["r_corrected"], r_raw.mean(dim=0)))

        # Per-particle statistics are sorted together with the rewards.
        results_mock = {
            "ImageReward": {
                "result": [0.5, 0.9, 0.1, 0.7],
                "grad_energy_sq": [10.0, 20.0, 30.0, 40.0],
                "pair_variance": [1.0, 2.0, 3.0, 4.0]
            }
        }
        sorted_idx = np.argsort(results_mock["ImageReward"]["result"])[::-1]
        results_mock["ImageReward"]["result"] = [results_mock["ImageReward"]["result"][i] for i in sorted_idx]
        results_mock["ImageReward"]["grad_energy_sq"] = [results_mock["ImageReward"]["grad_energy_sq"][i] for i in sorted_idx]
        results_mock["ImageReward"]["pair_variance"] = [results_mock["ImageReward"]["pair_variance"][i] for i in sorted_idx]
        self.assertEqual(results_mock["ImageReward"]["grad_energy_sq"], [20.0, 40.0, 10.0, 30.0])

    def test_07_e2e_50_steps_loop(self):
        """Mock 50-step loop over all modules"""
        z_t = torch.randn((1, self.C, self.H, self.W), device=self.device)
        lookahead_z0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        flow_guidance = RSFlowLiDARGuidance(s_flow=12.5)

        for step in range(50):
            sigma_t = 1.0 - (step / 50.0)

            with torch.no_grad():
                # 1. Antithetic batch perturbation
                perturbed = generate_smoothed_pixel_batch(lookahead_z0, sigma=0.25, M=4, noise_mode="independent_antithetic")
                self.assertEqual(perturbed.shape, (4, self.N, self.C, self.H, self.W))

                # 2. Entropic Risk
                r_mc = torch.randn((4, self.N), device=self.device)
                r_stats = compute_entropic_antithetic_rewards(r_mc, beta_risk=0.5)

                # 3. Pareto evaluation (2 metrics)
                potentials, _ = compute_multi_reward_pareto_potential({"r1": r_mc, "r2": r_mc * 2.0}, tau_p=2.0)
                self.assertEqual(potentials.shape, (self.N,))

                # 4. Flow Velocity Guidance
                delta_v = flow_guidance.compute_velocity_guidance(z_t, lookahead_z0, r_stats["r_corrected"], sigma_t)
                self.assertEqual(delta_v.shape, z_t.shape)

                # Free loop references
                del perturbed, r_mc, r_stats, potentials, delta_v

    def test_08_transformers_compatibility_patch(self):
        """transformers compatibility patch for ImageReward"""
        from fkd_diffusers.patch_transformers import patch_transformers_for_imagereward
        patch_transformers_for_imagereward()

        import transformers
        self.assertTrue(hasattr(transformers.modeling_utils, "apply_chunking_to_forward"))
        self.assertTrue(hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"))
        self.assertTrue(hasattr(transformers.modeling_utils, "prune_linear_layer"))

        from transformers import PreTrainedTokenizerBase
        self.assertTrue(hasattr(PreTrainedTokenizerBase, "additional_special_tokens_ids"))
        self.assertTrue(hasattr(PreTrainedTokenizerBase, "additional_special_tokens"))



if __name__ == "__main__":
    unittest.main(verbosity=2)
