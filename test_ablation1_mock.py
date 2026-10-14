"""
========================================================================================
🧪 UNIT TEST GIẢ LẬP: ABLATION 1 SENSITIVE PROMPT LANDSCAPE SLICING (MOCK / CPU)
========================================================================================

Kiểm tra toàn diện pipeline Ablation 1 trên môi trường CPU không cần 2 GPU:
1. Giả lập Mock Pipeline, Mock VAE, Mock ImageReward (tensors giả lập siêu nhẹ).
2. Kiểm tra phân tầng Prompts (Stratified Sampling theo 6 Category GenEval).
3. Kiểm tra Phase 1 Lookahead Sampling & Dual Annotation (r_vanilla vs r_RS).
4. Kiểm tra Phase 2 Target Closed-form Steering (3 nhánh: Unsteered, Vanilla, RS-LiDAR).
5. Kiểm tra 1D Directional Landscape Slicing & Đạo hàm số học vi phân |Df(alpha)|.
6. Giả lập chạy song song 2 Shard (Shard 0 & Shard 1), ghi checkpoint.
7. Kiểm tra hàm merge_shard_checkpoints: Tự động gộp dữ liệu, sắp xếp Top 5,
   xuất ablation1_summary.csv, ablation1_top5_divergence.csv, ablation1_metrics.json,
   và xuất đồ họa Side-by-Side 5-cột panel PNG.
"""

import os
import sys
import shutil
import json
import unittest
from types import SimpleNamespace
from PIL import Image
import numpy as np
import torch

# Đảm bảo import được module chính
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from test_sensitive_prompt_landscape import (
    load_stratified_prompts_per_category,
    run_phase1_lookahead_dual_annotation,
    run_phase2_target_steering,
    compute_1d_landscape_profile,
    compute_prompt_divergence_score,
    merge_shard_checkpoints,
    InMemoryLookaheadDataset,
    pil_to_tensor_neg1_to_1,
    tensor_to_pil,
    extract_single_pil_image,
    extract_pil_images_list,
)


# ======================================================================================
# 🎭 MOCK COMPONENTS (GIẢ LẬP ĐỘC LẬP SIÊU NHẸ)
# ======================================================================================
class MockScheduler:
    def __init__(self):
        self.config = {}

    @classmethod
    def from_config(cls, cfg):
        return cls()


class MockImageProcessor:
    def postprocess(self, tensors, output_type="pil"):
        # tensors: [B, 3, H, W] in [-1.0, 1.0]
        pils = []
        for i in range(tensors.shape[0]):
            arr = (tensors[i].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            pils.append(Image.fromarray(arr))
        return pils


class MockVAE:
    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=0.18215)
        self.dtype = torch.float32

    def decode(self, latents, return_dict=False):
        # latents: [B, 4, 64, 64] -> fake decoded [B, 3, 64, 64]
        b = latents.shape[0]
        # Tạo ảnh giả lập có màu sắc biến đổi theo latent
        out = torch.sin(latents[:, :3, :, :]).clamp(-1.0, 1.0)
        return (out,)


class MockDiffusionPipeline:
    def __init__(self):
        self.scheduler = MockScheduler()
        self.vae = MockVAE()
        self.image_processor = MockImageProcessor()

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt, num_inference_steps=5, guidance_scale=7.5, generator=None, output_type="pil", latents=None, fkd_args=None, **kwargs):
        self.last_guidance_scale = guidance_scale
        self.last_fkd_args = fkd_args
        if isinstance(prompt, list):
            b = len(prompt)
        else:
            b = 1

        if output_type == "latent":
            # Sinh latents giả lập [B, 4, 64, 64]
            if latents is not None:
                imgs = latents
            else:
                imgs = torch.randn((b, 4, 64, 64), dtype=torch.float32)
            return SimpleNamespace(images=imgs)

        # Output PIL images
        pils = []
        for i in range(b):
            # Màu sắc khác nhau tùy theo fkd_args (để phân biệt Unsteered vs Vanilla vs RS-LiDAR)
            if fkd_args is None or fkd_args.get("scale", 0.0) == 0.0:
                color = (120, 120, 180) # Unsteered (Xanh lam nhạt)
            elif "vanilla" in str(fkd_args.get("rag_dataset", "")).lower() or fkd_args.get("lmbda", 0) > 0 and not getattr(fkd_args.get("rag_dataset"), "is_rs", False):
                color = (220, 60, 60)   # Vanilla LiDAR (Đỏ)
            else:
                color = (40, 180, 140)  # RS-LiDAR (Xanh ngọc teal)
            pils.append(Image.new("RGB", (64, 64), color=color))

        # Mô phỏng chuẩn FKDStableDiffusion trả về tuple([StableDiffusionPipelineOutput(...)])
        if fkd_args is not None:
            return tuple([SimpleNamespace(images=pils)])

        return SimpleNamespace(images=pils)


class MockImageRewardModel:
    def score(self, prompt, pil_image):
        if not isinstance(pil_image, Image.Image):
            raise TypeError(f"ImageReward scoring requires PIL.Image or str, got {type(pil_image)}")
        # Trả về điểm giả lập dựa trên độ sáng trung bình và prompt hash
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        mean_val = float(np.mean(arr))
        # Mô phỏng hàm reward có gai nhọn cục bộ đối với một số prompt
        p_hash = sum(ord(c) for c in str(prompt)) % 10
        spike = 1.2 if (p_hash % 2 == 1 and mean_val > 0.5) else 0.0
        return float(mean_val * 2.0 - 0.5 + spike)


# ======================================================================================
# 🧪 TEST CASE SUITE
# ======================================================================================
class TestAblation1Mock(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(_current_dir, "results", "test_ablation1_mock_output")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        self.mock_pipe = MockDiffusionPipeline()
        self.mock_vae = self.mock_pipe.vae
        self.mock_ir = MockImageRewardModel()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_load_stratified_prompts_per_category(self):
        """Kiểm tra chọn lọc phân tầng chuẩn 6 categories GenEval."""
        # 1 prompt per category -> 6 prompts
        prompts_1 = load_stratified_prompts_per_category(
            prompt_path="prompt_files/geneval_metadata.jsonl",
            prompts_per_category=1
        )
        self.assertEqual(len(prompts_1), 6)
        cats_1 = {p["tag"] for p in prompts_1}
        self.assertEqual(len(cats_1), 6)

        # 2 prompts per category -> 12 prompts
        prompts_2 = load_stratified_prompts_per_category(
            prompt_path="prompt_files/geneval_metadata.jsonl",
            prompts_per_category=2
        )
        self.assertEqual(len(prompts_2), 12)
        for p in prompts_2:
            self.assertIn("prompt", p)
            self.assertIn("original_idx", p)
            self.assertIn("tag", p)

    def test_02_phase1_lookahead_dual_annotation(self):
        """Kiểm tra Phase 1 sinh lookahead latents và chấm song song r_vanilla vs r_RS."""
        prompt = "three apples on a wooden table"
        real_idx = 4
        num_lookahead = 10 # 10 hạt cho test nhanh

        ds_vanilla, ds_rs, stats = run_phase1_lookahead_dual_annotation(
            pipe=self.mock_pipe,
            vae=self.mock_vae,
            ir_model=self.mock_ir,
            prompt=prompt,
            real_idx=real_idx,
            num_particles=num_lookahead,
            guidance_steps=3,
            sigma=1.0,
            mc_samples=3,
            use_crn=True,
            seed=42,
            device="cpu"
        )

        self.assertIn(real_idx, ds_vanilla.data)
        self.assertIn(real_idx, ds_rs.data)
        self.assertEqual(ds_vanilla.data[real_idx]["latents"].shape, (num_lookahead, 4, 64, 64))
        self.assertEqual(len(ds_vanilla.data[real_idx]["rewards"]), num_lookahead)
        self.assertEqual(len(ds_rs.data[real_idx]["rewards"]), num_lookahead)
        self.assertIn("r_vanilla_mean", stats)
        self.assertIn("r_rs_mean", stats)

    def test_03_phase2_target_steering(self):
        """Kiểm tra Phase 2 sinh 3 nhánh (Unsteered, Vanilla, RS-LiDAR) với cùng seed."""
        prompt = "a red sports car"
        real_idx = 6

        # Chuẩn bị dummy datasets
        ds_v = InMemoryLookaheadDataset()
        ds_v.set_datapoint(real_idx, torch.randn(10, 4, 64, 64), torch.ones(10) * 0.5, [prompt]*10)
        ds_rs = InMemoryLookaheadDataset()
        ds_rs.is_rs = True
        ds_rs.set_datapoint(real_idx, torch.randn(10, 4, 64, 64), torch.ones(10) * 0.8, [prompt]*10)

        p2_res = run_phase2_target_steering(
            pipe=self.mock_pipe,
            prompt=prompt,
            real_idx=real_idx,
            ds_vanilla=ds_v,
            ds_rs=ds_rs,
            ir_model=self.mock_ir,
            guidance_scale=12.5,
            target_steps=10,
            seed=42,
            device="cpu"
        )

        self.assertIn("img_unsteered", p2_res)
        self.assertIn("img_vanilla", p2_res)
        self.assertIn("img_rs", p2_res)
        self.assertIn("r_vanilla", p2_res)
        self.assertIn("r_rs", p2_res)
        self.assertIn("delta_r", p2_res)
        self.assertIsInstance(p2_res["img_unsteered"], Image.Image)
        self.assertIsInstance(p2_res["img_vanilla"], Image.Image)
        self.assertIsInstance(p2_res["img_rs"], Image.Image)

        # Kiểm tra Text CFG = 12.5 và Particle Guidance scale = 12.5 được truyền đúng chuẩn
        self.assertEqual(self.mock_pipe.last_guidance_scale, 12.5)
        self.assertIsNotNone(self.mock_pipe.last_fkd_args)
        self.assertEqual(self.mock_pipe.last_fkd_args["scale"], 12.5)

    def test_03b_extract_single_pil_image_unwrapping(self):
        """Kiểm tra bóc tách an toàn tuyệt đối từ các cấu trúc lồng nhau của Diffusers output."""
        base_img = Image.new("RGB", (32, 32), color=(100, 150, 200))

        # 1. Tuple of PipelineOutput (FKDStableDiffusion return format)
        nested_tuple = tuple([SimpleNamespace(images=[base_img])])
        self.assertIsInstance(extract_single_pil_image(nested_tuple), Image.Image)

        # 2. PipelineOutput trực tiếp
        pipe_out = SimpleNamespace(images=[base_img])
        self.assertIsInstance(extract_single_pil_image(pipe_out), Image.Image)

        # 3. List of PIL
        self.assertIsInstance(extract_single_pil_image([base_img]), Image.Image)

        # 4. Pure PIL
        self.assertIsInstance(extract_single_pil_image(base_img), Image.Image)

        # 5. Tensor [-1, 1]
        dummy_tensor = torch.zeros(1, 3, 32, 32)
        self.assertIsInstance(extract_single_pil_image(dummy_tensor), Image.Image)

        # 6. NumPy array
        dummy_arr = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertIsInstance(extract_single_pil_image(dummy_arr), Image.Image)

    def test_04_compute_1d_landscape_slicing(self):
        """Kiểm tra cắt lát 1D dọc vector vi sai và tính đạo hàm số học vi phân."""
        img_v = Image.new("RGB", (64, 64), color=(200, 50, 50))
        img_rs = Image.new("RGB", (64, 64), color=(50, 180, 120))
        prompt = "a cat under a wooden chair"

        slicing = compute_1d_landscape_profile(
            img_vanilla=img_v,
            img_rs=img_rs,
            prompt=prompt,
            ir_model=self.mock_ir,
            slicing_points=11, # 11 điểm cho test nhanh
            alpha_max=0.2,
            slicing_sigma=1.0,
            mc_samples=2,
            device="cpu"
        )

        self.assertEqual(len(slicing["alphas"]), 11)
        self.assertEqual(len(slicing["curve_vanilla"]), 11)
        self.assertEqual(len(slicing["curve_rs"]), 11)
        self.assertEqual(len(slicing["deriv_vanilla"]), 10)
        self.assertEqual(len(slicing["deriv_rs"]), 10)
        self.assertGreaterEqual(slicing["l_max_vanilla"], 0.0)
        self.assertGreaterEqual(slicing["l_max_rs"], 0.0)
        self.assertGreater(slicing["theoretical_bound"], 0.0)

    def test_05_sharding_and_merge_end_to_end(self):
        """Giả lập 2 Shards song song, lưu checkpoint, gộp shard, xuất CSV, JSON, PNG."""
        # Tạo 6 prompts đại diện
        test_prompts = [
            {"original_idx": 0, "tag": "single_object", "prompt": "a photo of a bench"},
            {"original_idx": 2, "tag": "two_object", "prompt": "a photo of a bench and a horse"},
            {"original_idx": 4, "tag": "counting", "prompt": "three apples on a wooden table"},
            {"original_idx": 6, "tag": "colors", "prompt": "a red sports car"},
            {"original_idx": 8, "tag": "position", "prompt": "a laptop on top of a desk"},
            {"original_idx": 10, "tag": "color_attr", "prompt": "a red dog and a blue cat"},
        ]

        images_dir = os.path.join(self.test_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        records_shard0 = []
        records_shard1 = []

        for idx, item in enumerate(test_prompts):
            p_id = item["original_idx"]
            p_text = item["prompt"]
            cat = item["tag"]

            # Giả lập tạo ảnh và lưu ra đĩa
            un_path = os.path.join(images_dir, f"{p_id:05d}_unsteered.png")
            vl_path = os.path.join(images_dir, f"{p_id:05d}_vanilla.png")
            rs_path = os.path.join(images_dir, f"{p_id:05d}_rs.png")
            Image.new("RGB", (64, 64), color=(120, 120, 180)).save(un_path)
            Image.new("RGB", (64, 64), color=(220, 60, 60)).save(vl_path)
            Image.new("RGB", (64, 64), color=(40, 180, 140)).save(rs_path)

            alphas = np.linspace(-0.2, 0.2, 11).tolist()
            cv = [float(0.5 + 0.1 * np.sin(i * 1.5)) for i in range(11)]
            crs = [float(0.7 + 0.02 * i) for i in range(11)]
            lv = float(np.max(np.abs(np.diff(cv) / 0.04)))
            lrs = float(np.max(np.abs(np.diff(crs) / 0.04)))

            div_score = compute_prompt_divergence_score(
                delta_r=0.2,
                l_max_vanilla=lv,
                l_max_rs=lrs,
                dist_l2=5.0
            )

            rec = {
                "original_idx": p_id,
                "prompt_idx": p_id,
                "category": cat,
                "prompt": p_text,
                "r_unsteered": 0.35,
                "r_vanilla": 0.50,
                "r_rs": 0.70,
                "delta_r": 0.20,
                "img_unsteered_path": un_path,
                "img_vanilla_path": vl_path,
                "img_rs_path": rs_path,
                "dist_l2": 5.0,
                "l_max_vanilla": lv,
                "l_max_rs": lrs,
                "l_reduction_ratio": lv / (lrs + 1e-6),
                "divergence_score": div_score,
                "slicing": {
                    "alphas": alphas,
                    "curve_vanilla": cv,
                    "curve_rs": crs,
                    "theoretical_bound": 1.2
                }
            }

            if idx % 2 == 0:
                records_shard0.append(rec)
            else:
                records_shard1.append(rec)

        # Ghi checkpoint Shard 0
        ckpt0_path = os.path.join(self.test_dir, "checkpoint_ablation1_shard_0.json")
        with open(ckpt0_path, "w", encoding="utf-8") as f:
            json.dump({"shard_id": 0, "records": records_shard0}, f, indent=2)

        # Ghi checkpoint Shard 1
        ckpt1_path = os.path.join(self.test_dir, "checkpoint_ablation1_shard_1.json")
        with open(ckpt1_path, "w", encoding="utf-8") as f:
            json.dump({"shard_id": 1, "records": records_shard1}, f, indent=2)

        # Gộp shard checkpoints
        merge_shard_checkpoints(self.test_dir)

        # Kiểm tra các file kết quả
        summary_csv = os.path.join(self.test_dir, "ablation1_summary.csv")
        top5_csv = os.path.join(self.test_dir, "ablation1_top5_divergence.csv")
        rescue_csv = os.path.join(self.test_dir, "ablation1_top5_rescue.csv")
        smooth_csv = os.path.join(self.test_dir, "ablation1_top5_smoothness.csv")
        metrics_json = os.path.join(self.test_dir, "ablation1_metrics.json")
        panel_png = os.path.join(self.test_dir, "ablation1_top5_divergence_panel.png")
        rescue_png = os.path.join(self.test_dir, "ablation1_top5_rescue_panel.png")
        smooth_png = os.path.join(self.test_dir, "ablation1_top5_smoothness_panel.png")
        curves_png = os.path.join(self.test_dir, "ablation1_landscape_1d_curves.png")

        self.assertTrue(os.path.exists(summary_csv), "Thiếu file ablation1_summary.csv")
        self.assertTrue(os.path.exists(top5_csv), "Thiếu file ablation1_top5_divergence.csv")
        self.assertTrue(os.path.exists(rescue_csv), "Thiếu file ablation1_top5_rescue.csv")
        self.assertTrue(os.path.exists(smooth_csv), "Thiếu file ablation1_top5_smoothness.csv")
        self.assertTrue(os.path.exists(metrics_json), "Thiếu file ablation1_metrics.json")
        self.assertTrue(os.path.exists(panel_png), "Thiếu file ablation1_top5_divergence_panel.png")
        self.assertTrue(os.path.exists(rescue_png), "Thiếu file ablation1_top5_rescue_panel.png")
        self.assertTrue(os.path.exists(smooth_png), "Thiếu file ablation1_top5_smoothness_panel.png")
        self.assertTrue(os.path.exists(curves_png), "Thiếu file ablation1_landscape_1d_curves.png")

        # Kiểm tra nội dung Top 5 CSV có đủ 5 bản ghi và xếp hạng đúng
        with open(top5_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 6) # 1 header + 5 data rows

        # Kiểm tra file ảnh PNG không rỗng
        self.assertGreater(os.path.getsize(panel_png), 1000)
        self.assertGreater(os.path.getsize(rescue_png), 1000)
        self.assertGreater(os.path.getsize(smooth_png), 1000)
        self.assertGreater(os.path.getsize(curves_png), 1000)

    def test_06_bounded_rescue_divergence_ranking(self):
        """Kiểm tra logic xếp hạng mới: Failure Rescue vs Smoothness Index."""
        from test_sensitive_prompt_landscape import compute_prompt_rescue_score, compute_prompt_smoothness_score

        # Case 1: Prompt 80 (Failure Rescue - Vanilla sụp đổ dưới Unsteered, RS cứu ngoạn mục)
        # R_un = -0.328, R_v = -1.660, R_rs = +1.849, dR = +3.509, dist_l2 = 81.3, Lv = 20.47, Lrs = 1.03
        score_rescue_80 = compute_prompt_rescue_score(
            delta_r=3.509,
            r_unsteered=-0.328,
            r_vanilla=-1.660,
            l_max_vanilla=20.47,
            l_max_rs=1.03,
            dist_l2=81.3,
            tau_dist=500.0
        )

        # Case 2: Prompt 260 (Pink Car Paradox - 2 ảnh gần như giống hệt nhau, dR ~ 0.005, dist_l2 = 40.8, Lv = 1.43, Lrs = 0.03)
        score_rescue_260 = compute_prompt_rescue_score(
            delta_r=0.006,
            r_unsteered=1.642,
            r_vanilla=1.678,
            l_max_vanilla=1.43,
            l_max_rs=0.03,
            dist_l2=40.8,
            tau_dist=500.0
        )

        # Case 3: Prompt 454 (RS thua Vanilla - dR = -0.34)
        score_rescue_454 = compute_prompt_rescue_score(
            delta_r=-0.34,
            r_unsteered=-1.037,
            r_vanilla=-0.469,
            l_max_vanilla=2.16,
            l_max_rs=0.05,
            dist_l2=77.0,
            tau_dist=500.0
        )

        # Rescue Score phải ưu tiên mạnh mẽ Prompt 80 trên Prompt 260 và Prompt 454!
        self.assertGreater(score_rescue_80, score_rescue_260, "Prompt 80 cứu nguy sụp đổ phải có điểm Rescue cao hơn hẳn Prompt 260!")
        self.assertGreater(score_rescue_260, score_rescue_454, "Prompt 260 phải xếp trên Prompt 454 nơi RS bị suy giảm reward!")

        # Trong khi đó, Smoothness Score phải tôn vinh tỷ số làm mịn Lipschitz (Prompt 260 có ratio = 48.98x)
        score_smooth_260 = compute_prompt_smoothness_score(delta_r=0.006, l_max_vanilla=1.43, l_max_rs=0.03)
        self.assertGreater(score_smooth_260, 2.5, "Smoothness Score của Prompt 260 phải cao do tỷ số triệt tiêu độ dốc 48.98x!")

        print("✅ Toàn bộ các kiểm tra Unit Test Giả Lập cho Ablation 1 đều HOÀN THÀNH XUẤT SẮC!")


if __name__ == "__main__":
    unittest.main()
