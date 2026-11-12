import os
import sys
import shutil
import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import lookahead_sampling
import LiDAR_sampling


class TestFluxMockE2E(unittest.TestCase):
    """
    Kiểm thử Mock End-to-End quy trình 2 Phase của FLUX.1-dev trên CPU (không cần GPU A100):
    1. Phase 1: lookahead_sampling sinh 3 hạt lookahead (16x128x128), đánh giá reward và lưu file.
    2. Phase 2: LiDAR_sampling tải tập lookahead, tính velocity guidance s_flow=40.0 và xuất kết quả.
    """

    def setUp(self):
        self.tmp_dir = os.path.join(SCRIPT_DIR, "tmp_mock_e2e")
        self.lookahead_dir = os.path.join(self.tmp_dir, "Lookahead_samples")
        self.target_dir = os.path.join(self.tmp_dir, "Target_samples")
        os.makedirs(self.lookahead_dir, exist_ok=True)
        os.makedirs(self.target_dir, exist_ok=True)

        # Mock prompt metadata nếu chưa có
        self.prompt_file = os.path.join(self.tmp_dir, "test_prompts.jsonl")
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "A cute cat on Mars", "evaluation_prompt": "A cute cat on Mars"}\n')
            f.write('{"prompt": "A futuristic city at sunset", "evaluation_prompt": "A futuristic city at sunset"}\n')

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_end_to_end_flux_mock_flow(self):
        # 1. Tạo mock FluxPipeline
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.vae_scale_factor = 16
        mock_pipe.vae.config.scaling_factor = 0.3611
        mock_pipe.vae.config.shift_factor = 0.1159

        # Unpack latents trả về tensor 4D [B, 16, 128, 128]
        def mock_unpack(latents, h, w, scale):
            b = latents.shape[0]
            return torch.randn(b, 16, 128, 128, dtype=torch.float32)
        mock_pipe._unpack_latents = mock_unpack

        # VAE decode trả về tuple (tensor,)
        mock_pipe.vae.decode.side_effect = lambda chunk, return_dict=False: (torch.randn(chunk.shape[0], 3, 224, 224),)

        # Postprocess trả về danh sách PIL Images
        mock_pipe.image_processor.postprocess.side_effect = lambda tensor, output_type="pil": [
            Image.new("RGB", (224, 224), color=(100, 150, 200)) for _ in range(tensor.shape[0])
        ]

        # Pipe forward call trả về object có .images là latents packed [B, (128*128)//4, 64]
        mock_pipe.side_effect = lambda *args, **kwargs: MagicMock(images=torch.randn(len(kwargs.get("prompt", [1, 2, 3])), 4096, 64))

        # Mock do_eval trả về điểm ImageReward hợp lệ
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

        # --- CHẠY THỬ PHASE 1 (Lookahead Sampling) ---
        print("\n🚀 [MOCK TEST] Khởi chạy Phase 1: lookahead_sampling...")
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

        # Kiểm tra kết quả Phase 1
        latent_file_0 = os.path.join(self.lookahead_dir, "FLUX_100_3_8", "00000", "samples", "latent.pt")
        results_file_0 = os.path.join(self.lookahead_dir, "FLUX_100_3_8", "00000", "results.json")
        self.assertTrue(os.path.exists(latent_file_0), "Phase 1 phải lưu được latent.pt cho prompt 00000")
        self.assertTrue(os.path.exists(results_file_0), "Phase 1 phải lưu được results.json cho prompt 00000")

        saved_latents = torch.load(latent_file_0, map_location="cpu")
        if isinstance(saved_latents, list):
            self.assertEqual(len(saved_latents), 3, "Phải có đúng 3 hạt lookahead")
            self.assertEqual(saved_latents[0].shape, (16, 128, 128), "Mỗi hạt lookahead phải có kích thước [16, 128, 128]")
        else:
            self.assertEqual(saved_latents.shape, (3, 16, 128, 128), "Latents Phase 1 phải có kích thước chuẩn [3, 16, 128, 128]")
        print("✅ [MOCK TEST] Phase 1 hoàn tất xuất sắc: Latents chuẩn [3, 16, 128, 128] và results.json đã được lưu!")

        # --- CHẠY THỬ PHASE 2 (Target Sampling với LiDAR) ---
        print("\n🚀 [MOCK TEST] Khởi chạy Phase 2: LiDAR_sampling...")
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

        # Mock FluxPipeline_LiDAR cho Phase 2
        mock_flux_lidar_pipe = MagicMock()
        mock_flux_lidar_pipe.to.return_value = mock_flux_lidar_pipe
        mock_flux_lidar_pipe.return_value = MagicMock(images=[Image.new("RGB", (224, 224))])

        with patch("sys.argv", phase2_args), \
             patch("fkd_diffusers.fkd_pipeline_flux.FluxPipeline_LiDAR.from_pretrained", return_value=mock_flux_lidar_pipe), \
             patch("LiDAR_sampling.do_eval", side_effect=mock_do_eval), \
             patch("fks_utils.do_eval", side_effect=mock_do_eval):

            lidar_args = LiDAR_sampling.get_args()
            LiDAR_sampling.main(lidar_args)

        # Kiểm tra kết quả Phase 2
        target_results_0 = os.path.join(self.target_dir, "FLUX_LiDAR_Run", "00000", "results.json")
        target_img_0 = os.path.join(self.target_dir, "FLUX_LiDAR_Run", "00000", "samples", "00000.png")
        final_metrics = os.path.join(self.target_dir, "FLUX_LiDAR_Run", "final_metrics.json")
        self.assertTrue(os.path.exists(target_results_0), "Phase 2 phải lưu được results.json cho prompt 00000")
        self.assertTrue(os.path.exists(target_img_0), "Phase 2 phải lưu được ảnh sinh ra 00000.png")
        self.assertTrue(os.path.exists(final_metrics), "Phase 2 phải lưu được final_metrics.json")
        print("✅ [MOCK TEST] Phase 2 hoàn tất xuất sắc: Ảnh, results.json và final_metrics.json đã được lưu!")
        self.assertTrue(os.path.exists(target_results_0), "Phase 2 phải lưu được results.json cho prompt 00000")
        self.assertTrue(os.path.exists(target_img_0), "Phase 2 phải lưu được ảnh sinh ra 00000.png")
        print("✅ [MOCK TEST] Phase 2 hoàn tất xuất sắc: Ảnh và kết quả đánh giá LiDAR đã được lưu!")


if __name__ == "__main__":
    unittest.main()
