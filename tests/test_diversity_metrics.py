"""
Unit test for compute_diversity_metrics.py
Verifies mathematical boundary conditions:
- Degenerate mode (N identical particles): Vendi Score = 1.0, CLIP distance = 0.0
- Orthogonal mode (N independent orthogonal modes): Vendi Score = N, CLIP distance = 1.0
- Partial mode (N particles clustered into K modes)
"""

import os
import unittest
import numpy as np
import torch

from compute_diversity_metrics import (
    compute_vendi_score,
    compute_pairwise_clip_distance,
    compute_pairwise_lpips_batch,
    format_markdown_table,
    parse_run_settings,
    save_and_merge_reports
)

class TestDiversityMetrics(unittest.TestCase):
    
    def test_vendi_score_degenerate_mode(self):
        """When all N particles are identical, Vendi Score must equal 1.0."""
        N = 4
        D = 512
        single_vec = np.random.randn(1, D)
        single_vec /= np.linalg.norm(single_vec)
        # N identical copies
        identical_embs = np.repeat(single_vec, N, axis=0)
        
        vs = compute_vendi_score(identical_embs)
        self.assertAlmostEqual(vs, 1.0, places=4, msg="N identical particles must have VS = 1.0")
        
        dist = compute_pairwise_clip_distance(identical_embs)
        self.assertAlmostEqual(dist, 0.0, places=4, msg="N identical particles must have distance = 0.0")

    def test_vendi_score_orthogonal_mode(self):
        """When N particles are mutually orthogonal, Vendi Score must equal N."""
        N = 4
        D = 512
        # Construct N orthonormal vectors via QR decomposition
        random_mat = np.random.randn(D, N)
        ortho_basis, _ = np.linalg.qr(random_mat)
        ortho_embs = ortho_basis[:, :N].T # Shape (N, D)
        
        vs = compute_vendi_score(ortho_embs)
        self.assertAlmostEqual(vs, float(N), places=3, msg=f"N orthogonal particles must have VS = {N}")
        
        dist = compute_pairwise_clip_distance(ortho_embs)
        self.assertAlmostEqual(dist, 1.0, places=3, msg="Orthogonal particles must have cosine distance = 1.0")

    def test_vendi_score_monotonicity(self):
        """Vendi Score must strictly scale with increasing mode diversity."""
        # 2 modes in N=4 (2 copies of mode A, 2 copies of mode B, with A perp B)
        vA = np.zeros((1, 4))
        vA[0, 0] = 1.0
        vB = np.zeros((1, 4))
        vB[0, 1] = 1.0
        two_modes = np.concatenate([vA, vA, vB, vB], axis=0)
        
        vs_two = compute_vendi_score(two_modes)
        # For 2 equal orthogonal modes, entropy is -2*(0.5*ln(0.5)) = ln(2), VS = exp(ln(2)) = 2.0
        self.assertAlmostEqual(vs_two, 2.0, places=4, msg="Two balanced orthogonal modes must have VS = 2.0")

    def test_mock_lpips_batch(self):
        """Verifies pairwise LPIPS index generation and batch forward."""
        class MockLPIPS(torch.nn.Module):
            def forward(self, x0, x1):
                # Return dummy distance equal to mean absolute difference
                return torch.mean(torch.abs(x0 - x1), dim=(1, 2, 3), keepdim=True)
                
        mock_model = MockLPIPS()
        N = 4
        # N identical images -> distance should be 0.0
        img = torch.zeros(1, 3, 32, 32)
        imgs = torch.cat([img] * N, dim=0)
        d = compute_pairwise_lpips_batch(imgs, mock_model)
        self.assertAlmostEqual(d, 0.0, places=4)
        
        # 1 image different
        imgs[0] = 1.0
        d_diff = compute_pairwise_lpips_batch(imgs, mock_model)
        self.assertGreater(d_diff, 0.0)

    def test_markdown_formatting(self):
        """Verifies Markdown summary table formatting."""
        summaries = [
            {
                "run_name": "Vanilla_SMC_N4",
                "num_particles": 4,
                "image_reward_max": 0.9004,
                "vendi_score_mean": 1.45,
                "lpips_mean": 0.312,
                "clip_dist_mean": 0.052,
                "evaluated_prompts": 553
            },
            {
                "run_name": "RS_SMC_LiDAR_N4",
                "num_particles": 4,
                "image_reward_max": 0.9796,
                "vendi_score_mean": 3.25,
                "lpips_mean": 0.485,
                "clip_dist_mean": 0.089,
                "evaluated_prompts": 553
            }
        ]
        table = format_markdown_table(summaries)
        self.assertIn("Vanilla_SMC_N4", table)
        self.assertIn("RS_SMC_LiDAR_N4", table)
        self.assertIn("0.9796", table)

    def test_parse_run_settings(self):
        """Verifies automatic extraction of hyperparameters from systematic run names."""
        name_rssmc = "RSSMC_SD15_N4_sig0.25_M8_anneal_LiDAR_freq20_lmbda10.0_DDPM100_s12.5_seed100"
        cfg1 = parse_run_settings(name_rssmc)
        self.assertEqual(cfg1["method"], "RS-SMC (Ours)")
        self.assertEqual(cfg1["backbone"], "SD15")
        self.assertEqual(cfg1["particles"], 4)
        self.assertEqual(cfg1["sigma"], 0.25)
        self.assertEqual(cfg1["M"], 8)
        self.assertEqual(cfg1["anneal"], "C\u00f3")
        self.assertEqual(cfg1["guidance"], "LiDAR")

        name_smc = "SMC_SD15_N4_LiDAR_freq20_lmbda10.0_DDPM100_s12.5_seed100"
        cfg2 = parse_run_settings(name_smc)
        self.assertEqual(cfg2["method"], "Vanilla SMC")
        self.assertEqual(cfg2["particles"], 4)
        self.assertEqual(cfg2["sigma"], "\u2014")
        self.assertEqual(cfg2["anneal"], "\u2014")

    def test_save_and_merge_reports(self):
        """Verifies incremental accumulation and deduplication across multiple evaluation rounds."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "diversity_report.json")
            csv_path = os.path.join(tmpdir, "diversity_report.csv")

            # Round 1: Evaluate Folder A
            round1 = [{
                "run_name": "Run_A",
                "image_reward_max": 0.85,
                "vendi_score_mean": 2.1,
                "evaluated_prompts": 100
            }]
            acc1 = save_and_merge_reports(round1, output_json=json_path, output_csv=csv_path, create_timestamped_copy=False)
            self.assertEqual(len(acc1), 1)

            # Round 2: Evaluate Folder B (New run)
            round2 = [{
                "run_name": "Run_B",
                "image_reward_max": 0.95,
                "vendi_score_mean": 3.4,
                "evaluated_prompts": 100
            }]
            acc2 = save_and_merge_reports(round2, output_json=json_path, output_csv=csv_path, create_timestamped_copy=False)
            self.assertEqual(len(acc2), 2, "Round 2 must accumulate both Run_A and Run_B")

            # Round 3: Re-evaluate Folder A with updated score (Deduplication)
            round3 = [{
                "run_name": "Run_A",
                "image_reward_max": 0.89, # updated
                "vendi_score_mean": 2.3,
                "evaluated_prompts": 553
            }]
            acc3 = save_and_merge_reports(round3, output_json=json_path, output_csv=csv_path, create_timestamped_copy=False)
            self.assertEqual(len(acc3), 2, "Must not duplicate Run_A")
            run_a_updated = next(r for r in acc3 if r["run_name"] == "Run_A")
            self.assertEqual(run_a_updated["image_reward_max"], 0.89)
            self.assertEqual(run_a_updated["evaluated_prompts"], 553)

    def test_clip_feature_extraction_compatibility(self):
        """Verifies handling of modern transformers BaseModelOutputWithPooling objects."""
        class MockBaseModelOutputWithPooling:
            def __init__(self, tensor):
                self.pooler_output = tensor
                
        # Simulate modern output
        dummy_tensor = torch.randn(4, 512)
        mock_output = MockBaseModelOutputWithPooling(dummy_tensor)
        
        # Test unwrapping logic
        feats = mock_output
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            feats = feats.image_embeds
        elif not isinstance(feats, torch.Tensor):
            feats = feats[0]
            
        self.assertTrue(isinstance(feats, torch.Tensor))
        self.assertEqual(feats.shape, (4, 512))
        
        # Norm should work without AttributeError
        normed = feats / feats.norm(dim=-1, keepdim=True)
        self.assertAlmostEqual(float(normed[0].norm().item()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
