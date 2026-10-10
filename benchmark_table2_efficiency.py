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
from typing import List, Dict, Any, Optional, Tuple

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

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if os.path.join(_current_dir, "fkd_diffusers") not in sys.path:
    sys.path.insert(0, os.path.join(_current_dir, "fkd_diffusers"))

try:
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
except ImportError:
    try:
        from fkd_pipeline_sdxl import FKDStableDiffusionXL
        from fkd_pipeline_sd import FKDStableDiffusion
    except ImportError:
        FKDStableDiffusionXL = None
        FKDStableDiffusion = None

try:
    from fkd_diffusers.image_reward_utils import rm_load
except ImportError:
    try:
        from image_reward_utils import rm_load
    except ImportError:
        try:
            import ImageReward as RM
            rm_load = RM.load
        except ImportError:
            rm_load = None

try:
    from fks_utils import do_eval
except ImportError:
    do_eval = None


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
) -> Tuple[List[float], float, float]:
    """
    Đánh giá ImageReward toàn diện trong đúng 1 batch forward:
    - BƯỚC 1 (Đổi ảnh): Tạo tensor nhiễu (nếu RS-LiDAR) và đổi sang list PIL (Đo t_convert).
    - BƯỚC 2 (Chấm điểm thuần túy): Chạy qua ir_model.score_batched với batch_size = None (Đo t_score_pure).
    - Reshape ma trận (M, N) và lấy kỳ vọng trung bình E[R(x + eps)].
    """
    n_particles = decoded_tensor.shape[0]

    with torch.inference_mode():
        # 1. Đo riêng thời gian đổi Tensor sang PIL
        ev_conv_start = torch.cuda.Event(enable_timing=True)
        ev_conv_end = torch.cuda.Event(enable_timing=True)
        ev_conv_start.record()

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

        ev_conv_end.record()
        torch.cuda.synchronize(device)
        t_convert = ev_conv_start.elapsed_time(ev_conv_end) / 1000.0

        # 2. Đo riêng thời gian CHẤM ĐIỂM THUẦN TÚY (Pure Reward Scoring Forward)
        ev_score_start = torch.cuda.Event(enable_timing=True)
        ev_score_end = torch.cuda.Event(enable_timing=True)
        ev_score_start.record()

        # GỌI ĐÚNG 1 BATCH DUY NHẤT: reward_batch_size = None
        raw_scores = ir_model.score_batched(eval_prompts, pil_images, batch_size=reward_batch_size)

        ev_score_end.record()
        torch.cuda.synchronize(device)
        t_score_pure = ev_score_start.elapsed_time(ev_score_end) / 1000.0

        if num_mc > 1:
            scores_by_m = np.array(raw_scores).reshape(num_mc, n_particles)
            final_scores = np.mean(scores_by_m, axis=0).tolist()
        else:
            final_scores = [float(s) for s in raw_scores]

    return final_scores, t_score_pure, t_convert


# ==============================================================================
# 4. Pipeline Factory Theo Cấu Hình Setting
# ==============================================================================
def setup_models(setting: str, device: str = "cuda:0", lookahead_steps: Optional[int] = None):
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
    load_fn = rm_load
    if load_fn is None:
        try:
            from fkd_diffusers.image_reward_utils import rm_load as load_fn
        except Exception:
            try:
                from image_reward_utils import rm_load as load_fn
            except Exception:
                import ImageReward as RM
                load_fn = RM.load
    ir_model = load_fn("ImageReward-v1.0", device=device)
    ir_model.eval()

    if setting in ["sdv1.5_ddim50", "sdv1.5_ddpm100"]:
        model_id = "runwayml/stable-diffusion-v1-5"

        # Phase 2 Pipeline: FKDStableDiffusion (Target Steering)
        print(f"🔹 Khởi tạo Model SD 1.5: {model_id}...")
        pipe_phase2 = FKDStableDiffusion.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        pipe_phase2.scheduler = DDIMScheduler.from_config(pipe_phase2.scheduler.config)

        # Phase 1 Pipeline: DPM-Solver (Chia sẻ chung trọng số UNet/VAE/TextEncoder với Phase 2 để tránh nhân đôi VRAM)
        pipe_phase1 = StableDiffusionPipeline(
            vae=pipe_phase2.vae,
            text_encoder=pipe_phase2.text_encoder,
            tokenizer=pipe_phase2.tokenizer,
            unet=pipe_phase2.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(pipe_phase2.scheduler.config),
            safety_checker=None,
            feature_extractor=pipe_phase2.feature_extractor,
        )

        # Cấu hình siêu tham số
        default_lookahead_steps = 5
        actual_lookahead_steps = lookahead_steps if lookahead_steps is not None else default_lookahead_steps
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

        # Phase 1 Pipeline: SDXL DMD-1 (Lookahead)
        print(f"🔹 Khởi tạo Phase 1: SDXL DMD-1...")
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

        default_lookahead_steps = 1
        actual_lookahead_steps = lookahead_steps if lookahead_steps is not None else default_lookahead_steps
        num_particles_phase1 = 100
        num_target_images = 4
        target_steps = 100
        target_eta = 1.0          # DDPM ngẫu nhiên
        target_scale = 8.0

    else:
        raise ValueError(f"Không hỗ trợ cấu hình setting '{setting}'. Lựa chọn: sdv1.5_ddim50, sdv1.5_ddpm100, sdxl_dmd1")

    config = {
        "lookahead_steps": actual_lookahead_steps,
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

    # Đồng bộ hóa GPU và quản lý VRAM động (Offload Phase 2 để dồn toàn lực VRAM cho Phase 1)
    torch.cuda.synchronize(device)
    pipe_phase2.to("cpu")
    pipe_phase1.to(device)
    torch.cuda.empty_cache()
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
            # SDXL Phase 1 Lookahead (Full Batch)
            # Khởi tạo guidance_scale=0.0 và timesteps=[399] hệt như file gốc để tắt CFG (tránh x2 batch size)
            latents = pipe_phase1(
                prompt_batch,
                num_inference_steps=config["lookahead_steps"],
                guidance_scale=0.0,
                timesteps=[399] if config["lookahead_steps"] == 1 else None,
                output_type="latent"
            ).images  # Tensor (N, 4, 128, 128)

            # VAE Decode an toàn với batch size = 1 để tránh spike VRAM trên SDXL 1024x1024
            shift_factor = getattr(pipe_phase1.vae.config, "shift_factor", 0.0)
            shift_factor = 0.0 if shift_factor is None else shift_factor
            latents_scaled = (latents / pipe_phase1.vae.config.scaling_factor) + shift_factor
            latents_scaled = latents_scaled.to(pipe_phase1.vae.dtype) # ép về đúng kiểu dữ liệu của VAE để tránh lỗi
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

            # VAE Decode theo chunk size 10 (loại bỏ cú spike 18GB VRAM không cần thiết)
            scaled_latents = latents / pipe_phase1.vae.config.scaling_factor
            decoded_chunks = []
            vae_chunk = 10
            for v_i in range(0, scaled_latents.shape[0], vae_chunk):
                chunk = scaled_latents[v_i : v_i + vae_chunk]
                decoded_chunks.append(pipe_phase1.vae.decode(chunk, return_dict=False)[0])
            decoded_tensor = torch.cat(decoded_chunks, dim=0)  # Shape: (N, 3, 512, 512)

    ev_look_end.record()
    torch.cuda.synchronize(device)
    t_lookahead = ev_look_start.elapsed_time(ev_look_end) / 1000.0

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 2: Phase 1 Reward Annotation (Chấm Điểm Thuần Túy - Khớp Table 9)
    # --------------------------------------------------------------------------
    smoothed_scores, t_reward, t_convert = evaluate_reward_full_batch(
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

    # Ghi nhận đỉnh VRAM trước Phase 2
    torch.cuda.synchronize(device)
    peak_vram_p1_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Đảo models trong VRAM (Offload Phase 1, Load Phase 2) để mô phỏng pipeline tách rời
    pipe_phase1.to("cpu")
    pipe_phase2.to(device)
    torch.cuda.empty_cache()

    # Reset stats trước Phase 2 để đo riêng biệt Peak VRAM của Target Sampling (khớp Table 2 & Figure 10)
    torch.cuda.reset_peak_memory_stats(device)

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
    vram_phase2_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Tổng kết số liệu của prompt (T_total tính theo chuẩn Table 9: Lookahead + Reward Scoring + Target)
    t_total = t_lookahead + t_reward + t_target
    peak_vram_gib = max(peak_vram_p1_gib, vram_phase2_gib)

    return {
        "prompt_idx": real_idx,
        "prompt": prompt_str,
        "t_lookahead": t_lookahead,
        "t_reward": t_reward,              # Chấm điểm thuần túy
        "t_convert": t_convert,            # Đổi Tensor -> PIL
        "t_reward_total": t_reward + t_convert,
        "t_target": t_target,
        "t_total": t_total,
        "vram_phase2_gib": vram_phase2_gib,
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
    parser.add_argument("--lookahead_steps", "--phase1_steps", "--num_inference_steps", type=int, default=None,
                        help="Number of Phase 1 lookahead generation steps (default: 5 for SD1.5, 1 for SDXL)")
    parser.add_argument("--sweep_steps", type=str, default=None,
                        help="Comma-separated list of Phase 1 steps to benchmark consecutively in a single session (e.g. '2,3,4,5' or '2, 3, 4, 5')")
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

    # Phân giải danh sách bước Phase 1 cần chạy
    if args.sweep_steps is not None and args.sweep_steps.strip() != "":
        steps_to_run = [int(s.strip()) for s in args.sweep_steps.split(",") if s.strip().isdigit()]
    elif args.lookahead_steps is not None:
        steps_to_run = [args.lookahead_steps]
    else:
        default_step = 1 if "sdxl" in args.setting else 5
        steps_to_run = [default_step]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"🚀 BẮT ĐẦU BENCHMARK TABLE 2 EFFICIENCY (TIME & MEMORY)")
    print(f"{'='*75}")
    print(f"• Cấu hình Setting:    {args.setting}")
    print(f"• Phương pháp Method:  {args.method.upper()}")
    print(f"• Phase 1 Steps:       {steps_to_run if len(steps_to_run) > 1 else steps_to_run[0]}")
    print(f"• Smoothing Sigma:     {sigma}")
    print(f"• Monte Carlo M:       {num_mc}")
    print(f"• Reward Batch Size:   {'FULL BATCH (None)' if reward_batch_size is None else reward_batch_size}")
    print(f"• Số Prompts đo:       {args.num_prompts}")
    print(f"• Output Directory:    {args.output_dir}")
    print(f"{'='*75}\n")

    # 1. Nạp Model & Pipeline (Chỉ nạp 1 lần duy nhất trong toàn phiên để tối ưu tốc độ)
    pipe_phase1, pipe_phase2, ir_model, config = setup_models(
        args.setting, device=device, lookahead_steps=steps_to_run[0]
    )

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

    # 4. Thực Thi Đo Đạc Cho Từng Bước (Hỗ trợ chạy đơn lẻ hoặc Sweep 2, 3, 4, 5 steps liên tục)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_summaries = []

    for step_idx, cur_step in enumerate(steps_to_run):
        config["lookahead_steps"] = cur_step
        step_suffix = f"_step{cur_step}" if (len(steps_to_run) > 1 or args.lookahead_steps is not None) else ""
        csv_filename = f"efficiency_{args.setting}_{args.method}{step_suffix}_{timestamp}.csv"
        json_filename = f"efficiency_{args.setting}_{args.method}{step_suffix}_{timestamp}.json"
        csv_path = os.path.join(args.output_dir, csv_filename)
        json_path = os.path.join(args.output_dir, json_filename)

        csv_fields = [
            "prompt_idx", "prompt", "setting", "method", "lookahead_steps", "sigma", "num_mc",
            "t_lookahead", "t_reward", "t_convert", "t_target", "t_total", "vram_phase2_gib", "peak_vram_gib"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

        results_list = []

        print(f"\n⏱️ [{step_idx + 1}/{len(steps_to_run)}] Bắt đầu đo đạc {len(prompts_data)} prompts cho Phase 1 ({cur_step} steps)...")
        for idx, item in enumerate(tqdm(prompts_data, desc=f"Benchmarking (Step={cur_step})")):
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

            row = {
                "prompt_idx": res["prompt_idx"],
                "prompt": res["prompt"],
                "setting": args.setting,
                "method": args.method,
                "lookahead_steps": cur_step,
                "sigma": sigma,
                "num_mc": num_mc,
                "t_lookahead": f"{res['t_lookahead']:.3f}",
                "t_reward": f"{res['t_reward']:.3f}",
                "t_convert": f"{res['t_convert']:.3f}",
                "t_target": f"{res['t_target']:.3f}",
                "t_total": f"{res['t_total']:.3f}",
                "vram_phase2_gib": f"{res['vram_phase2_gib']:.2f}",
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

            print(f"\n  [Step {cur_step} | Prompt #{res['prompt_idx']:03d}] T_look: {res['t_lookahead']:.2f}s | T_rew (Chấm điểm): {res['t_reward']:.2f}s (Convert PIL: {res['t_convert']:.2f}s) | T_tar: {res['t_target']:.2f}s | T_tot: {res['t_total']:.2f}s | VRAM P2: {res['vram_phase2_gib']:.2f} GiB (Peak: {res['peak_vram_gib']:.2f} GiB)")

        # Tính toán thống kê trung bình
        mean_t_look = float(np.mean([r["t_lookahead"] for r in results_list]))
        mean_t_rew = float(np.mean([r["t_reward"] for r in results_list]))
        mean_t_conv = float(np.mean([r["t_convert"] for r in results_list]))
        mean_t_tar = float(np.mean([r["t_target"] for r in results_list]))
        mean_t_tot = float(np.mean([r["t_total"] for r in results_list]))
        mean_vram_p2 = float(np.mean([r["vram_phase2_gib"] for r in results_list]))
        max_peak_vram = float(np.max([r["peak_vram_gib"] for r in results_list]))

        summary = {
            "setting": args.setting,
            "method": args.method,
            "lookahead_steps": cur_step,
            "sigma": sigma,
            "num_mc": num_mc,
            "reward_batch_size": "Full-Batch" if reward_batch_size is None else reward_batch_size,
            "num_prompts": len(results_list),
            "mean_t_lookahead_sec": mean_t_look,
            "mean_t_reward_sec": mean_t_rew,
            "mean_t_convert_sec": mean_t_conv,
            "mean_t_reward_total_sec": mean_t_rew + mean_t_conv,
            "mean_t_target_sec": mean_t_tar,
            "mean_t_total_sec": mean_t_tot,
            "mean_vram_phase2_gib": mean_vram_p2,
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
                print(f"☁️ Đã đồng bộ kết quả Step {cur_step} (JSON & CSV) sang Google Drive tại: {args.drive_backup_dir}")
            except Exception as e_drive:
                print(f"⚠️ Cảnh báo: Không thể đồng bộ JSON sang Google Drive ({e_drive})")

        sweep_summaries.append(summary)

        # In bảng tổng kết của từng bước
        paper_t_look = "5.69s" if "sdv1.5" in args.setting else "4.30s"
        paper_t_rew = "0.65s" if "sdv1.5" in args.setting else "1.30s"
        paper_t_tar = "7.07s" if "ddpm100" in args.setting else ("3.58s" if "ddim50" in args.setting else "50.00s")
        paper_t_tot = "13.41s" if "ddpm100" in args.setting else ("9.92s" if "ddim50" in args.setting else "55.60s")
        paper_vram = "8.90 GiB" if "sdv1.5" in args.setting else "33.84 GiB"

        print(f"\n{'='*95}")
        print(f"📊 BẢNG TỔNG KẾT PHÂN RÃ THỜI GIAN & BỘ NHỚ (PHASE 1: {cur_step} STEPS)")
        print(f"{'='*95}")
        print(f"Setting: {args.setting} | Method: {args.method.upper()} | Lookahead Steps: {cur_step} | Prompts: {len(results_list)} | Batch: {'Full Batch' if reward_batch_size is None else reward_batch_size}")
        print(f"{'-'*95}")
        print(f"• T_lookahead (Sinh N hạt Lookahead):            {mean_t_look:6.2f} giây  (Bài báo DPM-5 / n=50: {paper_t_look})")
        print(f"• T_reward    (Chấm điểm ImageReward thuần túy): {mean_t_rew:6.2f} giây  (Bài báo n=50 / M=1:   {paper_t_rew})")
        print(f"  └─ Phụ phí đổi Tensor sang PIL (Convert):       {mean_t_conv:6.2f} giây")
        print(f"• T_target    (Khử nhiễu 4 ảnh đích Phase 2):    {mean_t_tar:6.2f} giây  (Bài báo Target:       {paper_t_tar})")
        print(f"--------------------------------------------------------------------------------------------")
        print(f"👉 TỔNG THỜI GIAN THEO TABLE 9 (Look + Rew + Tar):   {mean_t_tot:6.2f} giây  (Bài báo Table 9:     {paper_t_tot})")
        print(f"👉 VRAM PHASE 2 (Target Sampling - Khớp Table 2):    {mean_vram_p2:6.2f} GiB  (Bài báo & Fig 10:     {paper_vram})")
        print(f"👉 ĐỈNH BỘ NHỚ TOÀN BỘ (Peak Overall VRAM):          {max_peak_vram:6.2f} GiB")
        print(f"{'='*95}")
        print(f"📁 Dữ liệu chi tiết Step {cur_step} đã lưu tại:")
        print(f"   CSV:  {csv_path}")
        print(f"   JSON: {json_path}")
        if args.drive_backup_dir:
            print(f"   Google Drive: {args.drive_backup_dir}")
        print(f"{'='*95}\n")

    # Nếu chạy Sweep nhiều bước: Xuất file tổng hợp và in bảng Ablation đối chiếu
    if len(steps_to_run) > 1:
        sweep_data = {
            "setting": args.setting,
            "method": args.method,
            "sweep_steps": steps_to_run,
            "sigma": sigma,
            "num_mc": num_mc,
            "num_prompts": len(prompts_data),
            "sweep_results": sweep_summaries
        }
        sweep_json_filename = f"efficiency_sweep_{args.setting}_{args.method}_{timestamp}.json"
        sweep_csv_filename = f"efficiency_sweep_{args.setting}_{args.method}_{timestamp}.csv"
        sweep_json_path = os.path.join(args.output_dir, sweep_json_filename)
        sweep_csv_path = os.path.join(args.output_dir, sweep_csv_filename)

        with open(sweep_json_path, "w", encoding="utf-8") as f:
            json.dump(sweep_data, f, indent=4, ensure_ascii=False)

        sweep_csv_fields = [
            "lookahead_steps", "mean_t_lookahead_sec", "mean_t_reward_sec", "mean_t_convert_sec",
            "mean_t_target_sec", "mean_t_total_sec", "mean_vram_phase2_gib", "max_peak_vram_gib"
        ]
        with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_csv_fields)
            writer.writeheader()
            for sm in sweep_summaries:
                writer.writerow({
                    "lookahead_steps": sm["lookahead_steps"],
                    "mean_t_lookahead_sec": f"{sm['mean_t_lookahead_sec']:.3f}",
                    "mean_t_reward_sec": f"{sm['mean_t_reward_sec']:.3f}",
                    "mean_t_convert_sec": f"{sm['mean_t_convert_sec']:.3f}",
                    "mean_t_target_sec": f"{sm['mean_t_target_sec']:.3f}",
                    "mean_t_total_sec": f"{sm['mean_t_total_sec']:.3f}",
                    "mean_vram_phase2_gib": f"{sm['mean_vram_phase2_gib']:.2f}",
                    "max_peak_vram_gib": f"{sm['max_peak_vram_gib']:.2f}",
                })

        if args.drive_backup_dir:
            try:
                shutil.copyfile(sweep_json_path, os.path.join(args.drive_backup_dir, sweep_json_filename))
                shutil.copyfile(sweep_csv_path, os.path.join(args.drive_backup_dir, sweep_csv_filename))
                print(f"☁️ Đã đồng bộ file tổng hợp SWEEP (JSON & CSV) sang Google Drive tại: {args.drive_backup_dir}")
            except Exception as e_drive:
                print(f"⚠️ Cảnh báo: Không thể đồng bộ Sweep files sang Google Drive ({e_drive})")

        print(f"\n{'='*105}")
        print(f"📊 BẢNG TỔNG HỢP ABLATION LOOKAHEAD STEPS (SWEEP: {steps_to_run})")
        print(f"{'='*105}")
        print(f"Setting: {args.setting} | Method: {args.method.upper()} | Prompts: {len(prompts_data)}")
        print(f"{'-'*105}")
        print(f"{'Lookahead Steps (δ)':<22} | {'T_lookahead':<12} | {'T_reward':<10} | {'T_target':<10} | {'T_total':<10} | {'VRAM P2':<10} | {'Peak VRAM':<10}")
        print(f"{'-'*105}")
        for sm in sweep_summaries:
            st = sm['lookahead_steps']
            st_name = f"δ = {st} (DPM-{st})" if "sdv1.5" in args.setting else f"δ = {st} (DMD-{st})"
            print(f"{st_name:<22} | {sm['mean_t_lookahead_sec']:10.2f}s | {sm['mean_t_reward_sec']:8.2f}s | {sm['mean_t_target_sec']:8.2f}s | {sm['mean_t_total_sec']:8.2f}s | {sm['mean_vram_phase2_gib']:6.2f} GiB | {sm['max_peak_vram_gib']:6.2f} GiB")
        print(f"{'='*105}")
        print(f"📁 File tổng hợp Sweep đã lưu tại:")
        print(f"   CSV:  {sweep_csv_path}")
        print(f"   JSON: {sweep_json_path}")
        print(f"{'='*105}\n")


if __name__ == "__main__":
    main()
