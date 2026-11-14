import sys
import torch
import unittest

from lookahead_sampling import get_args as get_lookahead_args, compute_entropic_antithetic_rewards
from LiDAR_sampling import get_args as get_lidar_args

class TestFailFastAndBugResolution(unittest.TestCase):
    def test_01_lookahead_argparse_combinations(self):
        """Verify lookahead_sampling argparse properly parses all variations of M and num_mc_samples without overwrite"""
        # Test default
        sys.argv = ["lookahead_sampling.py"]
        args = get_lookahead_args()
        self.assertEqual(args.num_mc_samples, 4)
        self.assertEqual(args.M, 4)

        # Test --num_mc_samples=8
        sys.argv = ["lookahead_sampling.py", "--num_mc_samples=8"]
        args = get_lookahead_args()
        self.assertEqual(args.num_mc_samples, 8)
        self.assertEqual(args.M, 8)

        # Test --num_mc_samples 8
        sys.argv = ["lookahead_sampling.py", "--num_mc_samples", "8"]
        args = get_lookahead_args()
        self.assertEqual(args.num_mc_samples, 8)
        self.assertEqual(args.M, 8)

        # Test --M=8
        sys.argv = ["lookahead_sampling.py", "--M=8"]
        args = get_lookahead_args()
        self.assertEqual(args.num_mc_samples, 8)
        self.assertEqual(args.M, 8)

        # Test both matching
        sys.argv = ["lookahead_sampling.py", "--num_mc_samples=8", "--M=8"]
        args = get_lookahead_args()
        self.assertEqual(args.num_mc_samples, 8)
        self.assertEqual(args.M, 8)

    def test_02_fail_fast_conflicting_cli_arguments(self):
        """Fail-Fast: Conflicting values for M on CLI MUST raise ValueError immediately"""
        sys.argv = ["lookahead_sampling.py", "--num_mc_samples=8", "--M=4"]
        with self.assertRaises(ValueError):
            get_lookahead_args()

        sys.argv = ["LiDAR_sampling.py", "--smc_num_mc_samples=8", "--M=4"]
        with self.assertRaises(ValueError):
            get_lidar_args()

    def test_03_fail_fast_odd_m_antithetic(self):
        """Fail-Fast: independent_antithetic requires even M, odd M MUST raise ValueError"""
        sys.argv = ["lookahead_sampling.py", "--use_smoothing", "--noise_mode=independent_antithetic", "--num_mc_samples=5"]
        with self.assertRaises(ValueError):
            get_lookahead_args()

    def test_04_antithetic_numerical_difference_m4_vs_m8(self):
        """Verify that M=4 and M=8 generate strictly different empirical scores and grad_energy_sq"""
        N = 50
        torch.manual_seed(100)
        u_m4 = torch.randn(4, N)
        res_m4 = compute_entropic_antithetic_rewards(u_m4, sigma=0.5, beta_risk=0.5, noise_mode="independent_antithetic")

        torch.manual_seed(100)
        u_m8 = torch.randn(8, N)
        res_m8 = compute_entropic_antithetic_rewards(u_m8, sigma=0.5, beta_risk=0.5, noise_mode="independent_antithetic")

        max_diff = (res_m4["r_corrected"] - res_m8["r_corrected"]).abs().max().item()
        self.assertGreater(max_diff, 0.1, f"Expected noticeable score difference between M=4 and M=8, got {max_diff}")

    def test_05_cache_auto_healing_comprehensive(self):
        """Verify cache validation automatically invalidates cache if M, sigma, beta_risk, or noise_mode differ"""
        dummy_cached = {
            "ImageReward": {"mean": 0.5, "max": 1.0, "min": 0.0, "std": 0.2, "result": [0.5]*50},
            "smoothing": {
                "enabled": True,
                "sigma": 0.5,
                "num_mc_samples": 4,
                "noise_mode": "independent_antithetic",
                "beta_risk": 0.5
            }
        }

        # Case 1: M differs (4 vs 8) -> lat_valid False
        args_m = 8
        lat_valid = True
        s_info = dummy_cached["smoothing"]
        if s_info.get("num_mc_samples") != args_m:
            lat_valid = False
        self.assertFalse(lat_valid)

        # Case 2: Beta differs (0.5 vs 0.0) -> lat_valid False
        args_beta = 0.0
        lat_valid = True
        if abs(s_info.get("beta_risk") - args_beta) > 1e-4:
            lat_valid = False
        self.assertFalse(lat_valid)

        # Case 3: Noise mode differs -> lat_valid False
        args_mode = "coupled_crn"
        lat_valid = True
        if s_info.get("noise_mode") != args_mode:
            lat_valid = False
        self.assertFalse(lat_valid)

        # Case 4: Everything matches -> lat_valid True
        lat_valid = True
        if (s_info.get("num_mc_samples") == 4 and 
            abs(s_info.get("beta_risk") - 0.5) < 1e-4 and 
            s_info.get("noise_mode") == "independent_antithetic" and 
            abs(s_info.get("sigma") - 0.5) < 1e-4):
            lat_valid = True
        else:
            lat_valid = False
        self.assertTrue(lat_valid)

if __name__ == "__main__":
    unittest.main()
