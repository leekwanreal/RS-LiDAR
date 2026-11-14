import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import lookahead_sampling
import LiDAR_sampling


class TestFluxMockE2E(unittest.TestCase):
    """
    Mocked end-to-end run of both FLUX.1-dev phases on CPU (no GPU needed):
    1. Phase 1: lookahead_sampling draws 3 lookahead particles (16x128x128), scores and saves them.
    2. Phase 2: LiDAR_sampling loads the lookahead pool, applies velocity guidance (s_flow=40.0) and saves results.
    """

    def setUp(self):
        self.tmp_dir = os.path.join(SCRIPT_DIR, "tmp_mock_e2e")
        self.lookahead_dir = os.path.join(self.tmp_dir, "Lookahead_samples")
        self.target_dir = os.path.join(self.tmp_dir, "Target_samples")
        os.makedirs(self.lookahead_dir, exist_ok=True)
        os.makedirs(self.target_dir, exist_ok=True)

        # Mock prompt metadata
        self.prompt_file = os.path.join(self.tmp_dir, "test_prompts.jsonl")
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "A cute cat on Mars", "evaluation_prompt": "A cute cat on Mars"}\n')
            f.write('{"prompt": "A futuristic city at sunset", "evaluation_prompt": "A futuristic city at sunset"}\n')

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_end_to_end_flux_mock_flow(self):
        # 1. Mock FluxPipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.vae_scale_factor = 16
        mock_pipe.vae.config.scaling_factor = 0.3611
        mock_pipe.vae.config.shift_factor = 0.1159

        # _unpack_latents returns a 4D tensor [B, 16, 128, 128]
        def mock_unpack(latents, h, w, scale):
            b = latents.shape[0]
            return torch.randn(b, 16, 128, 128, dtype=torch.float32)
        mock_pipe._unpack_latents = mock_unpack

        # VAE decode returns a tuple (tensor,)
        mock_pipe.vae.decode.side_effect = lambda chunk, return_dict=False: (torch.randn(chunk.shape[0], 3, 224, 224),)

        # postprocess returns a list of PIL images
        mock_pipe.image_processor.postprocess.side_effect = lambda tensor, output_type="pil": [
            Image.new("RGB", (224, 224), color=(100, 150, 200)) for _ in range(tensor.shape[0])
        ]

        # The pipeline call returns packed latents [B, (128*128)//4, 64] in .images
        mock_pipe.side_effect = lambda *args, **kwargs: MagicMock(images=torch.randn(len(kwargs.get("prompt", [1, 2, 3])), 4096, 64))

        # Mock do_eval returning valid ImageReward scores
        def mock_do_eval(prompt, images, metrics_to_compute):
            n = len(images)
            res = [1.5 + 0.1 * i for i in range(n)]
            return {
                "ImageReward": {
                    "result": res,
                    "mean": float(sum(res) / len(res)),
                    "std": 0.1,
                    "max": float(max(res)),
                    "min": float(min(res)),
                }
            }

        # --- Phase 1 (lookahead sampling) ---
        phase1_args = [
            "lookahead_sampling.py",
            "--model_name", "ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors",
            "--prompt_path", self.prompt_file,
            "--prompt_indices", "0-1",
            "--num_particles", "3",
            "--num_inference_steps", "8",
            "--seed", "100",
            "--guidance_reward_fn", "ImageReward",
            "--metrics_to_compute", "ImageReward",
            "--save_individual_images",
            "--output_dir", self.lookahead_dir,
            "--run_name", "FLUX_100_3_8",
        ]

        # Mock class FluxPipeline
        mock_flux_class = MagicMock()
        mock_flux_class.from_pretrained.return_value = mock_pipe

        with patch("sys.argv", phase1_args), \
             patch("lookahead_sampling.FluxPipeline", mock_flux_class), \
             patch("lookahead_sampling.hf_hub_download", return_value="dummy.safetensors"), \
             patch("lookahead_sampling.do_eval", side_effect=mock_do_eval), \
             patch("fks_utils.do_eval", side_effect=mock_do_eval):
            
            lookahead_args = lookahead_sampling.get_args()
            lookahead_sampling.main(lookahead_args)

        # Check Phase 1 outputs
        latent_file_0 = os.path.join(self.lookahead_dir, "FLUX_100_3_8", "00000", "samples", "latent.pt")
        results_file_0 = os.path.join(self.lookahead_dir, "FLUX_100_3_8", "00000", "results.json")
        self.assertTrue(os.path.exists(latent_file_0), "Phase 1 must save latent.pt for prompt 00000")
        self.assertTrue(os.path.exists(results_file_0), "Phase 1 must save results.json for prompt 00000")

        saved_latents = torch.load(latent_file_0, map_location="cpu")
        if isinstance(saved_latents, list):
            self.assertEqual(len(saved_latents), 3, "Expected exactly 3 lookahead particles")
            self.assertEqual(saved_latents[0].shape, (16, 128, 128), "Each lookahead particle must be [16, 128, 128]")
        else:
            self.assertEqual(saved_latents.shape, (3, 16, 128, 128), "Phase 1 latents must be [3, 16, 128, 128]")

        # --- Phase 2 (target sampling with LiDAR guidance) ---
        phase2_args = [
            "LiDAR_sampling.py",
            "--model_name", "black-forest-labs/FLUX.1-dev",
            "--prompt_path", self.prompt_file,
            "--prompt_indices", "0-1",
            "--method", "LiDAR",
            "--use_rag",
            "--lookahead_path", os.path.join(self.lookahead_dir, "FLUX_100_3_8"),
            "--top_k", "3",
            "--s_flow", "40.0",
            "--lmbda", "5000.0",
            "--guidance_start", "1.0",
            "--guidance_end", "0.95",
            "--num_inference_steps", "28",
            "--num_particles", "1",
            "--run_name", "FLUX_LiDAR_Run",
            "--guidance_reward_fn", "ImageReward",
            "--save_individual_images",
            "--output_dir", self.target_dir,
        ]

        # Mock FluxPipeline_LiDAR for Phase 2
        mock_flux_lidar_pipe = MagicMock()
        mock_flux_lidar_pipe.to.return_value = mock_flux_lidar_pipe
        mock_flux_lidar_pipe.return_value = MagicMock(images=[Image.new("RGB", (224, 224))])

        with patch("sys.argv", phase2_args), \
             patch("fkd_diffusers.fkd_pipeline_flux.FluxPipeline_LiDAR.from_pretrained", return_value=mock_flux_lidar_pipe), \
             patch("LiDAR_sampling.do_eval", side_effect=mock_do_eval), \
             patch("fks_utils.do_eval", side_effect=mock_do_eval):

            lidar_args = LiDAR_sampling.get_args()
            LiDAR_sampling.main(lidar_args)

        # Check Phase 2 outputs
        target_results_0 = os.path.join(self.target_dir, "FLUX_LiDAR_Run", "00000", "results.json")
        target_img_0 = os.path.join(self.target_dir, "FLUX_LiDAR_Run", "00000", "samples", "00000.png")
        final_metrics = os.path.join(self.target_dir, "FLUX_LiDAR_Run", "final_metrics.json")
        self.assertTrue(os.path.exists(target_results_0), "Phase 2 must save results.json for prompt 00000")
        self.assertTrue(os.path.exists(target_img_0), "Phase 2 must save the generated image 00000.png")
        self.assertTrue(os.path.exists(final_metrics), "Phase 2 must save final_metrics.json")
        self.assertTrue(os.path.exists(target_results_0), "Phase 2 must save results.json for prompt 00000")
        self.assertTrue(os.path.exists(target_img_0), "Phase 2 must save the generated image 00000.png")


if __name__ == "__main__":
    unittest.main()
