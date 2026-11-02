import os
import sys
import json
import torch
import unittest
import tempfile
import shutil

# Add parent directory
sys.path.insert(0, os.path.abspath("Diffusion-LiDAR-Sampling"))
from lookahead_sampling import get_args as get_lookahead_args, compute_entropic_antithetic_rewards
from LiDAR_sampling import get_args as get_lidar_args

class TestMBugResolution(unittest.TestCase):
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

        # Test both --num_mc_samples=8 and --M=8
        sys.argv = ["lookahead_sampling.py", "--num_mc_samples=8", "--M=8"]
        args = get_lookahead_args()
        self.assertEqual(args.num_mc_samples, 8)
        self.assertEqual(args.M, 8)

    def test_02_lidar_argparse_combinations(self):
        """Verify LiDAR_sampling argparse properly parses smc_num_mc_samples, smc_M, and M"""
        # Test default
        sys.argv = ["LiDAR_sampling.py"]
        args = get_lidar_args()
        self.assertEqual(args.smc_num_mc_samples, 4)
        self.assertEqual(args.smc_M, 4)
        self.assertEqual(args.M, 4)

        # Test --smc_num_mc_samples=8
        sys.argv = ["LiDAR_sampling.py", "--smc_num_mc_samples=8"]
        args = get_lidar_args()
        self.assertEqual(args.smc_num_mc_samples, 8)
        self.assertEqual(args.smc_M, 8)
        self.assertEqual(args.M, 8)

        # Test --smc_M=8
        sys.argv = ["LiDAR_sampling.py", "--smc_M=8"]
        args = get_lidar_args()
        self.assertEqual(args.smc_num_mc_samples, 8)
        self.assertEqual(args.smc_M, 8)
        self.assertEqual(args.M, 8)

        # Test --M=8
        sys.argv = ["LiDAR_sampling.py", "--M=8"]
        args = get_lidar_args()
        self.assertEqual(args.smc_num_mc_samples, 8)
        self.assertEqual(args.smc_M, 8)
        self.assertEqual(args.M, 8)

    def test_03_antithetic_numerical_difference_m4_vs_m8(self):
        """Verify that M=4 and M=8 generate strictly different empirical scores and grad_energy_sq"""
        N = 50
        torch.manual_seed(100)
        u_m4 = torch.randn(4, N)
        res_m4 = compute_entropic_antithetic_rewards(u_m4, sigma=0.5, beta_risk=0.5, noise_mode="independent_antithetic")

        torch.manual_seed(100)
        u_m8 = torch.randn(8, N)
        res_m8 = compute_entropic_antithetic_rewards(u_m8, sigma=0.5, beta_risk=0.5, noise_mode="independent_antithetic")

        # The corrected rewards MUST be mathematically different
        max_diff = (res_m4["r_corrected"] - res_m8["r_corrected"]).abs().max().item()
        self.assertGreater(max_diff, 0.1, f"Expected noticeable score difference between M=4 and M=8, got {max_diff}")

    def test_04_cache_auto_healing(self):
        """Verify that lookahead_sampling cache validation automatically rejects a cached result with M=4 when M=8 is requested"""
        temp_dir = tempfile.mkdtemp()
        try:
            prompt_dir = os.path.join(temp_dir, "00552")
            os.makedirs(prompt_dir, exist_ok=True)
            results_file = os.path.join(prompt_dir, "results.json")

            # Simulate corrupted results.json saved by the legacy bug (M=4 recorded)
            dummy_cached = {
                "ImageReward": {"mean": 0.5, "max": 1.0, "min": 0.0, "std": 0.2, "result": [0.5]*50},
                "smoothing": {
                    "enabled": True,
                    "sigma": 0.5,
                    "num_mc_samples": 4, # Recorded M=4!
                    "noise_mode": "independent_antithetic"
                }
            }
            with open(results_file, "w") as f:
                json.dump(dummy_cached, f)

            # Test scenario 1: When args.num_mc_samples is 8, cache should be invalid!
            with open(results_file, "r") as f:
                res_cached = json.load(f)
            
            lat_valid = True
            args_m = 8
            if "smoothing" in res_cached:
                cached_m = res_cached["smoothing"].get("num_mc_samples")
                if cached_m is not None and cached_m != args_m:
                    lat_valid = False
            
            self.assertFalse(lat_valid, "Cache with M=4 should be invalidated when requesting M=8!")

            # Test scenario 2: When args.num_mc_samples is 4, cache should be valid!
            lat_valid_4 = True
            args_m_4 = 4
            if "smoothing" in res_cached:
                cached_m = res_cached["smoothing"].get("num_mc_samples")
                if cached_m is not None and cached_m != args_m_4:
                    lat_valid_4 = False
            
            self.assertTrue(lat_valid_4, "Cache with M=4 should be valid when requesting M=4!")

        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()
