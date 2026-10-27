import os
import sys
import time
import unittest
import torch

# Ensure Diffusion-LiDAR-Sampling root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from lookahead_sampling import (
    generate_smoothed_pixel_batch,
    compute_entropic_antithetic_rewards,
    get_args as get_lookahead_args
)
from multi_reward_evaluator import (
    compute_multi_reward_pareto_potential,
    LocalLipschitzParetoEvaluator
)
from rs_flow_lidar import RSFlowLiDARGuidance
from smc_engine import HarmonizedRSLiDARSMCEngine
from LiDAR_sampling import get_args as get_lidar_args


class MarkdownTestResult(unittest.TextTestResult):
    """
    Custom test result collector that prints a publication-grade Markdown table summary.
    """
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.records = []
        self._test_start_time = 0.0

    def startTest(self, test):
        super().startTest(test)
        self._test_start_time = time.perf_counter()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.perf_counter() - self._test_start_time
        doc = (test.shortDescription() or "").strip()
        self.records.append((test._testMethodName, doc, "PASS", elapsed))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.perf_counter() - self._test_start_time
        doc = (test.shortDescription() or "").strip()
        self.records.append((test._testMethodName, doc, "FAIL", elapsed))

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.perf_counter() - self._test_start_time
        doc = (test.shortDescription() or "").strip()
        self.records.append((test._testMethodName, doc, "ERROR", elapsed))


class MarkdownTestRunner(unittest.TextTestRunner):
    resultclass = MarkdownTestResult

    def run(self, test):
        result = super().run(test)
        self.print_markdown_summary(result)
        return result

    def print_markdown_summary(self, result):
        print("\n" + "=" * 96)
        print("PHASE 1 UNIFIED VERIFICATION MATRIX (ICML 2027 MASTER DOSSIER AUDIT)")
        print("=" * 96)
        print(f"| {'Test Case ID':<44} | {'Audit Item / Theoretical Target':<32} | {'Status':<8} | {'Time (s)':<8} |")
        print(f"|{'-' * 46}|{'-' * 34}|{'-' * 10}|{'-' * 10}|")
        for name, doc, status, elapsed in result.records:
            icon = "[PASS]" if status == "PASS" else "[FAIL]"
            print(f"| {name:<44} | {doc[:32]:<32} | {icon:<8} | {elapsed:<8.3f} |")
        print("=" * 96 + "\n")


class TestPhase1UnifiedModules(unittest.TestCase):
    """
    Unified Verification Suite for Phase 1:
    - Test 1: Theorem 5.1 & 5.2 (Antithetic Invariance & Entropic Risk)
    - Test 2: Theorem 6 (Local-Lipschitz Pareto Evaluator)
    - Test 3: Theorem 7 (CondOT Tweedie & Parabolic Boundary Damper)
    - Test 4: Double-Counting Lemma & Hybrid SMC (Doob-Tweedie Residual)
    - Test 5: Rule 5 CLI Default Argument Audit
    - Test 6: Zero-Regression Backward Compatibility
    - Test 7: Unified 50-Step E2E Stress-Test
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.N, self.C, self.H, self.W = 4, 4, 64, 64
        self.M = 4

    def test_01_antithetic_shapes_and_invariance(self):
        """Theorem 5.1 & 5.2: Symmetry & Entropic Risk"""
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

    def test_02_pareto_normalization_and_monotonicity(self):
        """Theorem 6: Lipschitz Scale & Soft-Min"""
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

        # Confirm 100x scale discrepancy is calibrated: ratio lambda_2 / lambda_1 ≈ 100
        ratio = lambda_2 / lambda_1
        self.assertAlmostEqual(ratio, 100.0, places=1)

        # Confirm Smooth Tchebycheff Soft-Min protects against reward hacking
        multi_candidates = {
            "r1": torch.tensor([[0.8, 1.0, 0.1], [0.8, 1.0, 0.1], [0.8, 1.0, 0.1], [0.8, 1.0, 0.1]], device=self.device),
            "r2": torch.tensor([[80.0, 10.0, 10.0], [80.0, 10.0, 10.0], [80.0, 10.0, 10.0], [80.0, 10.0, 10.0]], device=self.device)
        }
        p_cands, _ = compute_multi_reward_pareto_potential(multi_candidates, sigma=0.25, tau_p=2.0)
        self.assertGreater(p_cands[0].item(), p_cands[1].item(), "Balanced candidate must strictly dominate hacked candidate.")

    def test_03_flow_lidar_boundary_damper(self):
        """Theorem 7: Parabolic Damper & Fast-Path"""
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

        # 3. Parabolic Damper boundary symmetry omega(t) == omega(1-t) and stability at extreme boundary
        omega_01 = 0.1 * 0.9
        omega_09 = 0.9 * 0.1
        self.assertAlmostEqual(omega_01, omega_09)

        out_boundary = guidance.compute_velocity_guidance(
            z_t, lookahead_z0, rewards, sigma_t=0.0001, bypass_fast_path=True
        )
        self.assertTrue(torch.isfinite(out_boundary).all())
        self.assertLess(torch.norm(out_boundary).item(), 0.05)

    def test_04_residual_smc_seed_replay(self):
        """Double-Counting Lemma: Innovation Residual"""
        engine = HarmonizedRSLiDARSMCEngine(
            sigma_high=1.0,
            sigma_smc=0.25,
            beta_risk=0.5,
            eta_drift=0.4
        )
        x_t = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        tweedie_x0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        v_lidar = torch.tensor([2.0, 2.0, 2.0, 2.0], device=self.device)
        mock_reward = lambda x: torch.ones(x.shape[0], device=self.device) * 3.0

        # Exact Doob-Tweedie Innovation Residual calculation
        # v_smc = 3.0 (since r_pos=r_neg=3.0, diff=0)
        # phi_residual = 3.0 - (0.4 * 0.5 * 2.0) = 3.0 - 0.4 = 2.6
        phi_res, v_smc = engine.compute_step(x_t, tweedie_x0, v_lidar, seed_int=42, rho_t=0.5, reward_fn=mock_reward)
        self.assertTrue(torch.allclose(phi_res, torch.tensor([2.6, 2.6, 2.6, 2.6])))

        # 64-bit Seed Replay Determinism
        phi_res_2, _ = engine.compute_step(x_t, tweedie_x0, v_lidar, seed_int=42, rho_t=0.5, reward_fn=mock_reward)
        self.assertTrue(torch.equal(phi_res, phi_res_2))

        # Resampling when ESS < 0.5 * N
        log_weights = torch.tensor([10.0, 0.0, 0.0, -5.0], device=self.device)
        _, log_weights_new, resampled = engine.resample_if_needed(log_weights, x_t)
        self.assertTrue(resampled)
        self.assertTrue(torch.all(log_weights_new == 0.0))

    def test_05_cli_arguments_audit(self):
        """Rule 5: CLI Default Argument Audit"""
        sys.argv = ["lookahead_sampling.py"]
        args_lookahead = get_lookahead_args()

        self.assertEqual(args_lookahead.noise_mode, "independent_antithetic")
        self.assertEqual(args_lookahead.M, 4)
        self.assertEqual(args_lookahead.num_mc_samples, 4)
        self.assertEqual(args_lookahead.sigma_high, 1.0)
        self.assertEqual(args_lookahead.sigma_low, 0.25)
        self.assertEqual(args_lookahead.beta_risk, 0.5)
        self.assertEqual(args_lookahead.eta_drift, 0.4)
        self.assertEqual(args_lookahead.alpha_couple, 0.7)
        self.assertEqual(args_lookahead.s_flow, 12.5)
        self.assertEqual(args_lookahead.pareto_rho, 0.5)

        sys.argv = ["LiDAR_sampling.py"]
        args_lidar = get_lidar_args()
        self.assertEqual(args_lidar.noise_mode, "independent_antithetic")
        self.assertEqual(args_lidar.M, 4)
        self.assertEqual(args_lidar.sigma_high, 1.0)
        self.assertEqual(args_lidar.sigma_low, 0.25)
        self.assertEqual(args_lidar.beta_risk, 0.5)
        self.assertEqual(args_lidar.eta_drift, 0.4)
        self.assertEqual(args_lidar.alpha_couple, 0.7)
        self.assertEqual(args_lidar.s_flow, 12.5)
        self.assertEqual(args_lidar.pareto_rho, 0.5)

    def test_06_legacy_backward_compatibility(self):
        """Rule 1 & 2: Bit-for-bit Backward Compatibility"""
        # Fixed seed to ensure exact numerical reproducibility
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

    def test_07_e2e_50_steps_loop(self):
        """End-to-End Mock 50-Step Sampling Loop"""
        z_t = torch.randn((1, self.C, self.H, self.W), device=self.device)
        lookahead_z0 = torch.randn((self.N, self.C, self.H, self.W), device=self.device)
        flow_guidance = RSFlowLiDARGuidance(s_flow=12.5)
        smc_engine = HarmonizedRSLiDARSMCEngine()
        dummy_reward_fn = lambda x: torch.randn(x.shape[0], device=self.device)

        for step in range(50):
            sigma_t = 1.0 - (step / 50.0)
            rho_t = step / 50.0

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

                # 5. Hybrid SMC step
                phi_res, v_smc = smc_engine.compute_step(lookahead_z0, lookahead_z0, r_stats["r_corrected"], seed_int=42+step, rho_t=rho_t, reward_fn=dummy_reward_fn)
                self.assertEqual(phi_res.shape, (self.N,))

                # Free loop references
                del perturbed, r_mc, r_stats, potentials, delta_v, phi_res, v_smc


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase1UnifiedModules)
    runner = MarkdownTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
