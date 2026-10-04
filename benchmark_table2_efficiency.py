#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_table2_efficiency.py - Independent Table 2 Efficiency Benchmark Suite for RS-LiDAR vs Vanilla LiDAR
===========================================================================================================

Mục đích:
  Đo đạc chính xác thời gian thực thi phân rã 3 thành phần (khớp Bảng 9) và đỉnh VRAM (Bảng 2)
  của RS-LiDAR và Vanilla LiDAR trên GPU NVIDIA A100:
    1. T_lookahead: Thời gian sinh N hạt lookahead (DPM-5 hoặc DMD-1) và giải mã qua VAE.
    2. T_reward:    Thời gian đánh giá ImageReward (Full-Batch trên toàn bộ M x N ảnh).
    3. T_target:    Thời gian khử nhiễu mục tiêu (Closed-form steering, 4 ảnh đích, KHÔNG CHẤM REWARD PHASE 2).
    4. T_total:     T_lookahead + T_reward + T_target.
    5. Peak VRAM:   torch.cuda.max_memory_allocated() / 1024^3 (GiB).

3 Cấu hình chuẩn Bảng 2:
  - sdv1.5_ddim50:  SD 1.5, Lookahead DPM-5 (N=50), Target DDIM 50 steps (N_tar=4, scale=12.5, eta=0.0)
  - sdv1.5_ddpm100: SD 1.5, Lookahead DPM-5 (N=50), Target DDPM 100 steps (N_tar=4, scale=12.5, eta=1.0)
  - sdxl_dmd1:      SDXL (2.6B), Lookahead DMD-1 (N=100), Target DDPM 100 steps (N_tar=4, scale=8.0, eta=1.0)

Tác giả: RS-LiDAR Research Team
Ngày cập nhật: 2026-10-04
"""

# Standard Library
import os
import sys
import json
import csv
import math
import random
import shutil
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output encoding across Windows/Linux/Colab consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
# Universal compatibility patch for transformers, diffusers, peft, protobuf, and ImageReward
try:
    import google.protobuf
    if not hasattr(google.protobuf, "runtime_version"):
        class _RuntimeVersion:
            DOMAIN = "protobuf"
        google.protobuf.runtime_version = _RuntimeVersion
except Exception:
    pass

try:
    import transformers
    if not hasattr(transformers, "EncoderDecoderCache"):
        class EncoderDecoderCache:
            pass
        transformers.EncoderDecoderCache = EncoderDecoderCache
    import transformers.modeling_utils
    if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
            assert len(args) > 0
            if chunk_size <= 0:
                return forward_fn(*args)
            num_chunks = args[0].shape[chunk_dim] // chunk_size
            chunked_args = [torch.chunk(x, num_chunks, dim=chunk_dim) if isinstance(x, torch.Tensor) else [x] * num_chunks for x in args]
            layer_outputs = [forward_fn(*[x[i] for x in chunked_args]) for i in range(num_chunks)]
            return torch.cat(layer_outputs, dim=chunk_dim)
        transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            if len(heads) == 0:
                return set(), torch.empty(0, dtype=torch.long)
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
            transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
except Exception:
    pass

# Lazy import flag
_DIFFUSERS_AVAILABLE = False
try:
    from diffusers import (
        StableDiffusionPipeline,
        DiffusionPipeline,
        DPMSolverMultistepScheduler,
        DDIMScheduler,
        LCMScheduler,
        UNet2DConditionModel,
    )
    from huggingface_hub import hf_hub_download
    _DIFFUSERS_AVAILABLE = True
except ImportError:
    pass

try:
    from fkd_diffusers import FKDStableDiffusion, FKDStableDiffusionXL
    from fkd_diffusers.image_reward_utils import rm_load
    from fks_utils import do_eval
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from fkd_diffusers import FKDStableDiffusion, FKDStableDiffusionXL
        from fkd_diffusers.image_reward_utils import rm_load
        from fks_utils import do_eval
    except ImportError:
        pass


# ==============================================================================
# 1. Cấu Trúc In-Memory Dataset Để Tránh Đọc/Ghi Ổ Đĩa Trong Quá Trình Benchmark
# ==============================================================================
class InMemoryLookaheadDataset:
    """Dataset trong bộ nhớ RAM phục vụ Closed-form guidance mà không tốn I/O ổ đĩa."""
    def __init__(self, data_dict: Optional[Dict[int, Any]] = None):
        self.data = data_dict or {}

    def set_datapoint(self, prompt_idx: int, latents: torch.Tensor, rewards: torch.Tensor, prompts: List[str]):
        self.data[prompt_idx] = {
            "latents": latents.detach().cpu(),
            "rewards": rewards.detach().cpu() if isinstance(rewards, torch.Tensor) else torch.tensor(rewards),
            "prompts": prompts,
            "prompt_idx": prompt_idx,
        }


# ==============================================================================
# 2. Xử Lý Prompts
# ==============================================================================
def load_prompts(prompt_path: str, max_prompts: int = 3, prompt_indices: Optional[str] = None) -> List[Dict[str, Any]]:
    """Nạp danh sách prompts từ file JSON hoặc JSONL."""
    if not os.path.exists(prompt_path):
        # Fallback thử tìm trong thư mục Diffusion-LiDAR-Sampling
        alt_path = os.path.join("Diffusion-LiDAR-Sampling", prompt_path)
        if os.path.exists(alt_path):
            prompt_path = alt_path
        else:
            # Fallback tạo prompts mẫu nếu không tìm thấy file
            print(f"⚠️ Prompt file '{prompt_path}' không tồn tại. Tạo {max_prompts} prompts chuẩn GenEval mẫu.")
            default_prompts = [
                {"prompt": "A cute small dog wearing sunglasses on a beach", "original_idx": 0},
                {"prompt": "A modern red sports car parked in front of a white villa", "original_idx": 1},
                {"prompt": "A plate of pancakes with strawberries and maple syrup", "original_idx": 2},
                {"prompt": "An astronaut riding a green horse on the surface of mars", "original_idx": 3},
                {"prompt": "A serene lake surrounded by pine trees under a sunset sky", "original_idx": 4},
            ]
            return default_prompts[:max_prompts]

    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

    # Chuẩn hóa key 'prompt' và 'original_idx'
    for idx, item in enumerate(data):
        if "prompt" not in item:
            item["prompt"] = item.get("text", "")
        if "original_idx" not in item:
            item["original_idx"] = idx

    if prompt_indices is not None and prompt_indices.strip() != "":
        s = prompt_indices.strip()
        if "-" in s and "," not in s:
            parts = s.split("-")
            selected_ids = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            selected_ids = [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
        data = [item for item in data if item["original_idx"] in selected_ids]
    else:
        data = data[:max_prompts]

    return data


# ==============================================================================
# 3. Vectorized Full-Batch ImageReward Evaluation
# ==============================================================================
def evaluate_reward_full_batch(
    decoded_tensor: torch.Tensor,
    prompt_str: str,
    ir_model: Any,
    image_processor: Any,
    method: str = "rs-lidar",
    sigma: float = 1.0,
    num_mc: int = 4,
    reward_batch_size: Optional[int] = None,
    device: str = "cuda:0"
) -> List[float]:
    """
    Đánh giá ImageReward toàn diện trong đúng 1 batch forward:
    - Nếu method == 'rs-lidar': Tạo M tensor nhiễu Gaussian sigma, ghép thành (M * N, 3, H, W).
    - Nếu method == 'lidar': Chạy trực tiếp N ảnh sạch (1 * N, 3, H, W).
    - Chạy thẳng qua ir_model.score_batched với batch_size = None (Full Batch).
    - Reshape ma trận (M, N) và lấy kỳ vọng trung bình E[R(x + eps)].
    """
    n_particles = decoded_tensor.shape[0]

    with torch.inference_mode():
        if method == "rs-lidar" and num_mc > 1 and sigma > 0:
            noisy_list = [
                (decoded_tensor + torch.randn_like(decoded_tensor) * sigma).clamp(-1.0, 1.0)
                for _ in range(num_mc)
            ]
            all_images_tensor = torch.cat(noisy_list, dim=0)  # Shape: (M * N, 3, H, W)
        else:
            num_mc = 1
            all_images_tensor = decoded_tensor.clamp(-1.0, 1.0)

        # Chuyển đổi sang list PIL (chuẩn hóa kích thước tương thích ImageReward)
        pil_images = image_processor.postprocess(all_images_tensor, output_type="pil")
        eval_prompts = [prompt_str] * len(pil_images)

        # GỌI ĐÚNG 1 BATCH DUY NHẤT: reward_batch_size = None
        raw_scores = ir_model.score_batched(eval_prompts, pil_images, batch_size=reward_batch_size)

        if num_mc > 1:
            scores_by_m = np.array(raw_scores).reshape(num_mc, n_particles)
            final_scores = np.mean(scores_by_m, axis=0).tolist()
        else:
            final_scores = [float(s) for s in raw_scores]

    return final_scores


# ==============================================================================
# 4. Pipeline Factory Theo Cấu Hình Setting
# ==============================================================================
def setup_models(setting: str, device: str = "cuda:0"):
    """
    Khởi tạo Model Phase 1 (Lookahead), Model Phase 2 (Target) và ImageReward
    theo đúng thông số chuẩn Bảng 2.
    """
    if not _DIFFUSERS_AVAILABLE:
        raise ImportError(
            "Thư viện 'diffusers' hoặc 'huggingface_hub' chưa được cài đặt!\n"
            "Vui lòng cài đặt: pip install diffusers transformers accelerate"
        )
    print(f"\n📦 Đang nạp mô hình cho cấu hình '{setting}' trên {device}...")

    # 1. Nạp ImageReward Model
    print("🔹 Nạp ImageReward-v1.0...")
    ir_model = rm_load("ImageReward-v1.0", device=device)
    ir_model.eval()

    if setting in ["sdv1.5_ddim50", "sdv1.5_ddpm100"]:
        model_id = "runwayml/stable-diffusion-v1-5"

        # Phase 1 Pipeline: DPM-Solver 5 bước (Lookahead)
        print(f"🔹 Khởi tạo Phase 1: {model_id} (DPM-5)...")
        pipe_phase1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        pipe_phase1.scheduler = DPMSolverMultistepScheduler.from_config(pipe_phase1.scheduler.config)

        # Phase 2 Pipeline: FKDStableDiffusion (Target Steering)
        print(f"🔹 Khởi tạo Phase 2: FKDStableDiffusion...")
        pipe_phase2 = FKDStableDiffusion.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        pipe_phase2.scheduler = DDIMScheduler.from_config(pipe_phase2.scheduler.config)

        # Cấu hình siêu tham số
        lookahead_steps = 5
        num_particles_phase1 = 50
        num_target_images = 4

        if setting == "sdv1.5_ddim50":
            target_steps = 50
            target_eta = 0.0      # DDIM xác định
            target_scale = 12.5
        else:  # sdv1.5_ddpm100
            target_steps = 100
            target_eta = 1.0      # DDPM ngẫu nhiên
            target_scale = 12.5

    elif setting == "sdxl_dmd1":
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        # Phase 1 Pipeline: SDXL DMD-1 (1 bước Lookahead)
        print(f"🔹 Khởi tạo Phase 1: SDXL DMD-1 (1 bước)...")
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
        ckpt_path = hf_hub_download(repo_name, ckpt_name)
        unet.load_state_dict(torch.load(ckpt_path, map_location=device))

        pipe_phase1 = DiffusionPipeline.from_pretrained(
            base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe_phase1.scheduler = LCMScheduler.from_config(pipe_phase1.scheduler.config)
        pipe_phase1.vae.to(dtype=torch.float32)

        # Phase 2 Pipeline: FKDStableDiffusionXL (Target Steering)
        print(f"🔹 Khởi tạo Phase 2: FKDStableDiffusionXL...")
        pipe_phase2 = FKDStableDiffusionXL.from_pretrained(
            base_model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe_phase2.scheduler = DDIMScheduler.from_config(pipe_phase2.scheduler.config)

        lookahead_steps = 1
        num_particles_phase1 = 100
        num_target_images = 4
        target_steps = 100
        target_eta = 1.0          # DDPM ngẫu nhiên
        target_scale = 8.0

    else:
        raise ValueError(f"Không hỗ trợ cấu hình setting '{setting}'. Lựa chọn: sdv1.5_ddim50, sdv1.5_ddpm100, sdxl_dmd1")

    config = {
        "lookahead_steps": lookahead_steps,
        "num_particles_phase1": num_particles_phase1,
        "num_target_images": num_target_images,
        "target_steps": target_steps,
        "target_eta": target_eta,
        "target_scale": target_scale,
    }

    return pipe_phase1, pipe_phase2, ir_model, config


# ==============================================================================
# 5. Core Benchmark Runner (Đo đạc 1 prompt đầy đủ 2-Phase)
# ==============================================================================
def benchmark_single_prompt(
    item: Dict[str, Any],
    pipe_phase1: Any,
    pipe_phase2: Any,
    ir_model: Any,
    config: Dict[str, Any],
    method: str = "rs-lidar",
    sigma: float = 1.0,
    num_mc: int = 4,
    reward_batch_size: Optional[int] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Thực thi trọn vẹn quy trình 2-Phase cho 1 prompt và đo vi mô từng thành phần:
      - T_lookahead
      - T_reward (Full batch 1 forward)
      - T_target (Dừng sau khi sinh xong 4 ảnh đích, KHÔNG CHẤM REWARD)
    """
    prompt_str = item["prompt"]
    real_idx = item["original_idx"]
    n_p1 = config["num_particles_phase1"]
    n_tar = config["num_target_images"]
    is_sdxl = "sdxl" in getattr(pipe_phase1, "_name_or_path", "").lower() or hasattr(pipe_phase1, "unet") and pipe_phase1.unet.config.sample_size == 128

    # Đồng bộ hóa GPU và reset bộ nhớ
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 1: Phase 1 Lookahead Sampling (DPM-5 hoặc DMD-1)
    # --------------------------------------------------------------------------
    ev_look_start = torch.cuda.Event(enable_timing=True)
    ev_look_end = torch.cuda.Event(enable_timing=True)

    ev_look_start.record()
    with torch.inference_mode():
        prompt_batch = [prompt_str] * n_p1

        if is_sdxl:
            # SDXL Phase 1 Lookahead
            latents = pipe_phase1(
                prompt_batch,
                num_inference_steps=config["lookahead_steps"],
                output_type="latent"
            ).images  # Tensor (N, 4, 128, 128)

            # VAE Decode an toàn với batch size = 1 để tránh spike VRAM trên SDXL 1024x1024
            latents_scaled = (latents / pipe_phase1.vae.config.scaling_factor) + getattr(pipe_phase1.vae.config, "shift_factor", 0.0)
            decoded_chunks = []
            for v_i in range(0, latents_scaled.shape[0], 1):
                chunk = latents_scaled[v_i : v_i + 1]
                decoded_chunks.append(pipe_phase1.vae.decode(chunk, return_dict=False)[0])
            decoded_tensor = torch.cat(decoded_chunks, dim=0)  # Shape: (N, 3, 1024, 1024)
        else:
            # SD 1.5 Phase 1 Lookahead
            latents = pipe_phase1(
                prompt_batch,
                num_inference_steps=config["lookahead_steps"],
                output_type="latent"
            ).images  # Tensor (N, 4, 64, 64)

            # VAE Decode toàn bộ batch 50 hạt 512x512
            scaled_latents = latents / pipe_phase1.vae.config.scaling_factor
            decoded_tensor = pipe_phase1.vae.decode(scaled_latents, return_dict=False)[0]  # Shape: (N, 3, 512, 512)

    ev_look_end.record()
    torch.cuda.synchronize(device)
    t_lookahead = ev_look_start.elapsed_time(ev_look_end) / 1000.0

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 2: Phase 1 Reward Annotation (Full-Batch 1 Forward)
    # --------------------------------------------------------------------------
    ev_rew_start = torch.cuda.Event(enable_timing=True)
    ev_rew_end = torch.cuda.Event(enable_timing=True)

    ev_rew_start.record()
    smoothed_scores = evaluate_reward_full_batch(
        decoded_tensor=decoded_tensor,
        prompt_str=prompt_str,
        ir_model=ir_model,
        image_processor=pipe_phase1.image_processor,
        method=method,
        sigma=sigma,
        num_mc=num_mc,
        reward_batch_size=reward_batch_size,
        device=device
    )
    ev_rew_end.record()
    torch.cuda.synchronize(device)
    t_reward = ev_rew_start.elapsed_time(ev_rew_end) / 1000.0

    # Đóng gói dữ liệu Lookahead vào In-Memory Dataset cho Phase 2
    lookahead_ds = InMemoryLookaheadDataset()
    lookahead_ds.set_datapoint(
        prompt_idx=real_idx,
        latents=latents,
        rewards=torch.tensor(smoothed_scores, dtype=torch.float32),
        prompts=[prompt_str] * n_p1
    )

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 3: Phase 2 Target Sampling (Closed-Form Guidance)
    # --------------------------------------------------------------------------
    fkd_args = dict(
        lmbda=5000.0,
        scale=config["target_scale"],
        reward_type="ImageReward",
        num_particles=n_tar,             # Sinh đúng 4 ảnh đích
        use_smc=False,
        use_grad=False,
        use_rag=True,
        FK_lmbda=1.0,
        FK_resample_t_start=0,
        FK_resample_t_end=config["target_steps"],
        top_k=n_p1,
        rag_dataset=lookahead_ds,
        adaptive_resampling=False,
        resample_frequency=1,
        time_steps=config["target_steps"],
        resampling_t_end=config["target_steps"],
        guidance_reward_fn="ImageReward",
        potential_type="closed_form",
    )

    ev_tar_start = torch.cuda.Event(enable_timing=True)
    ev_tar_end = torch.cuda.Event(enable_timing=True)

    ev_tar_start.record()
    with torch.inference_mode():
        target_prompts = [prompt_str] * n_tar
        # Sinh 4 ảnh đích với closed-form steering
        target_output = pipe_phase2(
            target_prompts,
            prompt_idx=real_idx,
            num_inference_steps=config["target_steps"],
            eta=config["target_eta"],
            fkd_args=fkd_args,
        )
        # CHỐT CHẶN: DỪNG NGAY SAU KHI SINH XONG ẢNH, KHÔNG CHẤM REWARD PHASE 2!
        _ = target_output.images if hasattr(target_output, "images") else target_output

    ev_tar_end.record()
    torch.cuda.synchronize(device)
    t_target = ev_tar_start.elapsed_time(ev_tar_end) / 1000.0

    # Tổng kết số liệu của prompt
    t_total = t_lookahead + t_reward + t_target
    peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return {
        "prompt_idx": real_idx,
        "prompt": prompt_str,
        "t_lookahead": t_lookahead,
        "t_reward": t_reward,
        "t_target": t_target,
        "t_total": t_total,
        "peak_vram_gib": peak_vram_gib,
        "scores_mean": float(np.mean(smoothed_scores)),
        "scores_max": float(np.max(smoothed_scores)),
    }


# ==============================================================================
# 6. Main Execution Loop & Real-Time Logging
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark Table 2 Efficiency (Time & Peak VRAM) on A100")
    parser.add_argument("--setting", type=str, default="sdv1.5_ddpm100",
                        choices=["sdv1.5_ddim50", "sdv1.5_ddpm100", "sdxl_dmd1"],
                        help="Table 2 benchmark configuration: sdv1.5_ddim50, sdv1.5_ddpm100, sdxl_dmd1")
    parser.add_argument("--method", type=str, default="rs-lidar", choices=["rs-lidar", "lidar"],
                        help="Sampling method: rs-lidar (Randomized Smoothing) or lidar (Vanilla)")
    parser.add_argument("--num_prompts", type=int, default=3, help="Number of prompts to benchmark and average")
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl",
                        help="Path to prompts JSON or JSONL file")
    parser.add_argument("--prompt_indices", type=str, default=None,
                        help="Optional prompt indices (e.g. '0-2', '0,5,10')")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian smoothing std sigma. Default: 1.0 (SD 1.5), 0.5 (SDXL), 0.0 (LiDAR)")
    parser.add_argument("--num_mc_samples", "-M", type=int, default=None,
                        help="Monte Carlo sample count M. Default: 4 (rs-lidar), 1 (lidar)")
    parser.add_argument("--reward_batch_size", type=str, default="None",
                        help="ImageReward batch size. Default: 'None' (Full Batch 1 forward)")
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Perform 1 warm-up prompt run to initialize GPU kernels")
    parser.add_argument("--no_warmup", dest="warmup", action="store_false",
                        help="Disable GPU warm-up")
    parser.add_argument("--output_dir", type=str, default="results/benchmark_efficiency",
                        help="Output directory for CSV and JSON results")
    parser.add_argument("--drive_backup_dir", type=str, default=None,
                        help="Optional Google Drive backup directory to auto-sync CSV and JSON results in real time")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device identifier")

    args = parser.parse_args()

    # Thiết lập device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
        vram_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"🖥️ Thiết bị: {gpu_name} ({vram_total:.1f} GiB VRAM)")
    else:
        print("⚠️ Chạy trên CPU! Thời gian đo sẽ không đại diện cho benchmark A100.")

    # Thiết lập seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Phân giải tham số mặc định theo setting & method
    if args.method == "lidar":
        sigma = 0.0
        num_mc = 1
    else:
        num_mc = args.num_mc_samples if args.num_mc_samples is not None else 4
        if args.sigma is not None:
            sigma = args.sigma
        else:
            sigma = 0.5 if "sdxl" in args.setting else 1.0

    # Phân giải reward_batch_size
    if args.reward_batch_size is None or str(args.reward_batch_size).lower() in ["none", "null", "full"]:
        reward_batch_size = None
    else:
        reward_batch_size = int(args.reward_batch_size)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"🚀 BẮT ĐẦU BENCHMARK TABLE 2 EFFICIENCY (TIME & MEMORY)")
    print(f"{'='*75}")
    print(f"• Cấu hình Setting:    {args.setting}")
    print(f"• Phương pháp Method:  {args.method.upper()}")
    print(f"• Smoothing Sigma:     {sigma}")
    print(f"• Monte Carlo M:       {num_mc}")
    print(f"• Reward Batch Size:   {'FULL BATCH (None)' if reward_batch_size is None else reward_batch_size}")
    print(f"• Số Prompts đo:       {args.num_prompts}")
    print(f"• Output Directory:    {args.output_dir}")
    print(f"{'='*75}\n")

    # 1. Nạp Model & Pipeline
    pipe_phase1, pipe_phase2, ir_model, config = setup_models(args.setting, device=device)

    # 2. Nạp dữ liệu Prompts
    prompts_data = load_prompts(args.prompt_path, max_prompts=args.num_prompts, prompt_indices=args.prompt_indices)
    print(f"📋 Đã chọn {len(prompts_data)} prompts để tiến hành benchmark.")

    # 3. Chạy Warm-Up GPU (Nếu bật)
    if args.warmup and len(prompts_data) > 0:
        print("\n🔥 Đang chạy 1 prompt Warm-up để compile CUDA kernels và nạp weights vào VRAM...")
        _ = benchmark_single_prompt(
            item=prompts_data[0],
            pipe_phase1=pipe_phase1,
            pipe_phase2=pipe_phase2,
            ir_model=ir_model,
            config=config,
            method=args.method,
            sigma=sigma,
            num_mc=num_mc,
            reward_batch_size=reward_batch_size,
            device=device
        )
        print("✅ Warm-up hoàn tất! Bắt đầu bấm giờ chính thức.")

    # 4. Thiết lập File CSV & JSON Real-time Persistence
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"efficiency_{args.setting}_{args.method}_{timestamp}.csv"
    json_filename = f"efficiency_{args.setting}_{args.method}_{timestamp}.json"
    csv_path = os.path.join(args.output_dir, csv_filename)
    json_path = os.path.join(args.output_dir, json_filename)

    csv_fields = [
        "prompt_idx", "prompt", "setting", "method", "sigma", "num_mc",
        "t_lookahead", "t_reward", "t_target", "t_total", "peak_vram_gib"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    results_list = []

    # 5. Vòng Lặp Đo Đạc Chính Thức
    print(f"\n⏱️ Bắt đầu đo đạc {len(prompts_data)} prompts...")
    for idx, item in enumerate(tqdm(prompts_data, desc="Benchmarking")):
        res = benchmark_single_prompt(
            item=item,
            pipe_phase1=pipe_phase1,
            pipe_phase2=pipe_phase2,
            ir_model=ir_model,
            config=config,
            method=args.method,
            sigma=sigma,
            num_mc=num_mc,
            reward_batch_size=reward_batch_size,
            device=device
        )
        results_list.append(res)

        # Ghi tức thì vào CSV sau mỗi prompt (chống mất dữ liệu nếu Colab ngắt kết nối)
        row = {
            "prompt_idx": res["prompt_idx"],
            "prompt": res["prompt"],
            "setting": args.setting,
            "method": args.method,
            "sigma": sigma,
            "num_mc": num_mc,
            "t_lookahead": f"{res['t_lookahead']:.3f}",
            "t_reward": f"{res['t_reward']:.3f}",
            "t_target": f"{res['t_target']:.3f}",
            "t_total": f"{res['t_total']:.3f}",
            "peak_vram_gib": f"{res['peak_vram_gib']:.2f}",
        }
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        # Tự động sao chép sang Google Drive ngay lập tức sau mỗi prompt nếu có cấu hình
        if args.drive_backup_dir:
            try:
                os.makedirs(args.drive_backup_dir, exist_ok=True)
                drive_csv_target = os.path.join(args.drive_backup_dir, csv_filename)
                shutil.copyfile(csv_path, drive_csv_target)
            except Exception as e_drive:
                print(f"⚠️ Cảnh báo: Không thể đồng bộ CSV sang Google Drive ({e_drive})")

        print(f"\n  [Prompt #{res['prompt_idx']:03d}] T_lookahead: {res['t_lookahead']:.2f}s | T_reward: {res['t_reward']:.2f}s | T_target: {res['t_target']:.2f}s | T_total: {res['t_total']:.2f}s | Peak VRAM: {res['peak_vram_gib']:.2f} GiB")

    # 6. Tính Toán Thống Kê Trung Bình & Xuất Báo Cáo
    mean_t_look = float(np.mean([r["t_lookahead"] for r in results_list]))
    mean_t_rew = float(np.mean([r["t_reward"] for r in results_list]))
    mean_t_tar = float(np.mean([r["t_target"] for r in results_list]))
    mean_t_tot = float(np.mean([r["t_total"] for r in results_list]))
    max_peak_vram = float(np.max([r["peak_vram_gib"] for r in results_list]))

    summary = {
        "setting": args.setting,
        "method": args.method,
        "sigma": sigma,
        "num_mc": num_mc,
        "reward_batch_size": "Full-Batch" if reward_batch_size is None else reward_batch_size,
        "num_prompts": len(results_list),
        "mean_t_lookahead_sec": mean_t_look,
        "mean_t_reward_sec": mean_t_rew,
        "mean_t_target_sec": mean_t_tar,
        "mean_t_total_sec": mean_t_tot,
        "max_peak_vram_gib": max_peak_vram,
        "detailed_results": results_list,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # Đồng bộ JSON sang Google Drive
    if args.drive_backup_dir:
        try:
            drive_json_target = os.path.join(args.drive_backup_dir, json_filename)
            shutil.copyfile(json_path, drive_json_target)
            print(f"☁️ Đã đồng bộ toàn bộ kết quả CSV & JSON sang Google Drive tại: {args.drive_backup_dir}")
        except Exception as e_drive:
            print(f"⚠️ Cảnh báo: Không thể đồng bộ JSON sang Google Drive ({e_drive})")

    # 7. In Bảng Tổng Kết Chuẩn Bảng 9 và Bảng 2
    print(f"\n{'='*85}")
    print(f"📊 BẢNG TỔNG KẾT PHÂN RÃ THỜI GIAN (KHỚP TABLE 9 & TABLE 2 ICML 2026)")
    print(f"{'='*85}")
    print(f"Setting: {args.setting} | Method: {args.method.upper()} | Prompts: {len(results_list)} | Batch: {'Full Batch' if reward_batch_size is None else reward_batch_size}")
    print(f"{'-'*85}")
    print(f"• T_lookahead (Sinh N hạt Lookahead):        {mean_t_look:6.2f} giây")
    print(f"• T_reward    (Đánh giá ImageReward M hạt):   {mean_t_rew:6.2f} giây")
    print(f"• T_target    (Khử nhiễu 4 ảnh mục tiêu):     {mean_t_tar:6.2f} giây  (Đã dừng, không chấm Phase 2)")
    print(f"---------------------------------------------------------------------")
    print(f"👉 TỔNG THỜI GIAN TRUNG BÌNH (T_total / prompt): {mean_t_tot:6.2f} giây")
    print(f"👉 ĐỈNH BỘ NHỚ VRAM (Peak Memory Allocated):     {max_peak_vram:6.2f} GiB")
    print(f"{'='*85}")
    print(f"📁 Dữ liệu chi tiết đã lưu tại:")
    print(f"   CSV:  {csv_path}")
    print(f"   JSON: {json_path}")
    if args.drive_backup_dir:
        print(f"   Google Drive: {args.drive_backup_dir}")
    print(f"{'='*85}\n")


if __name__ == "__main__":
    main()
