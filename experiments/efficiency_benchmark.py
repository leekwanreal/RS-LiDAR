#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Table 2 efficiency benchmark (wall-clock time and peak VRAM) for RS-LiDAR vs. vanilla LiDAR.

Measures, per prompt, the time breakdown reported in Table 9 and the peak VRAM reported in Table 2:
  1. T_lookahead: generate N lookahead particles (DPM-5 or DMD-1) and decode them with the VAE.
  2. T_reward:    ImageReward scoring of all M x N (noisy) images in a single batch.
  3. T_target:    target denoising with closed-form steering (4 target images; no Phase-2 reward scoring).
  4. T_total:     T_lookahead + T_reward + T_target.
  5. Peak VRAM:   torch.cuda.max_memory_allocated() / 1024^3 (GiB).

Table 2 settings:
  - sdv1.5_ddim50:  SD 1.5, lookahead DPM-5 (N=50), target DDIM 50 steps (N_tar=4, scale=12.5, eta=0.0)
  - sdv1.5_ddpm100: SD 1.5, lookahead DPM-5 (N=50), target DDPM 100 steps (N_tar=4, scale=12.5, eta=1.0)
  - sdxl_dmd1:      SDXL, lookahead DMD-1 (N=100), target DDPM 100 steps (N_tar=4, scale=8.0, eta=1.0)

Usage (from the repository root):
  python -m experiments.efficiency_benchmark --setting sdv1.5_ddpm100 --method rs-lidar
"""

# Standard Library
import os
import sys
import json
import csv
import random
import shutil
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Force UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from tqdm import tqdm
import warnings

# Suppress harmless diffusers warning about float16 pipelines
warnings.filterwarnings("ignore", message=".*Pipelines loaded with.*dtype=torch.float16.*cannot run with.*cpu.*")

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if os.path.join(_repo_root, "fkd_diffusers") not in sys.path:
    sys.path.insert(0, os.path.join(_repo_root, "fkd_diffusers"))

from rslidar_utils import setup_runtime  # noqa: E402

# Compatibility shims for transformers / protobuf / ImageReward (must run before importing diffusers)
setup_runtime()

# Optional diffusers import (checked in setup_models)
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
# 1. In-memory lookahead dataset (avoids disk I/O during timing)
# ==============================================================================
class InMemoryLookaheadDataset:
    """In-memory lookahead dataset consumed by closed-form guidance without disk I/O."""
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
# 2. Prompt loading
# ==============================================================================
def load_prompts(
    prompt_path: str,
    max_prompts: int = 3,
    prompt_indices: Optional[str] = None,
    prompts_per_category: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load prompts from a JSON/JSONL file, optionally with stratified per-category selection."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Prompt file '{prompt_path}' not found. Pass --prompt_path or place the GenEval "
            "metadata file under prompt_files/."
        )

    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

    # Normalize the 'prompt', 'tag' and 'original_idx' keys
    for idx, item in enumerate(data):
        if "prompt" not in item:
            item["prompt"] = item.get("text", "")
        if "original_idx" not in item:
            item["original_idx"] = idx
        if "tag" not in item:
            item["tag"] = item.get("category", "general")

    if prompt_indices is not None and prompt_indices.strip() != "":
        s = prompt_indices.strip()
        if "-" in s and "," not in s:
            parts = s.split("-")
            selected_ids = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            selected_ids = [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
        data = [item for item in data if item["original_idx"] in selected_ids]
    elif prompts_per_category is not None and prompts_per_category > 0:
        # Group prompts by category (tag)
        cat_dict = {}
        for item in data:
            tag = item.get("tag", "general")
            cat_dict.setdefault(tag, []).append(item)

        selected_data = []
        print(f"Stratified selection: {prompts_per_category} prompts per category:")
        for tag, items in cat_dict.items():
            picked = items[:prompts_per_category]
            selected_data.extend(picked)
            indices_str = ", ".join(str(p["original_idx"]) for p in picked)
            print(f"   - [{tag}]: {len(picked)}/{len(items)} prompts (IDs: {indices_str})")
        data = selected_data
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
    Score all images with ImageReward in a single forward batch:
    - Step 1 (conversion): add Gaussian noise (RS-LiDAR only) and convert to PIL (timed as t_convert).
    - Step 2 (pure scoring): ir_model.score_batched with batch_size=None (timed as t_score_pure).
    - Reshape to (M, N) and average to obtain E[R(x + eps)].
    """
    n_particles = decoded_tensor.shape[0]

    with torch.inference_mode():
        # 1. Time the tensor -> PIL conversion separately
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

        # Convert to a list of PIL images (resolution expected by ImageReward)
        pil_images = image_processor.postprocess(all_images_tensor, output_type="pil")
        eval_prompts = [prompt_str] * len(pil_images)

        ev_conv_end.record()
        torch.cuda.synchronize(device)
        t_convert = ev_conv_start.elapsed_time(ev_conv_end) / 1000.0

        # 2. Time pure reward scoring (single forward pass)
        ev_score_start = torch.cuda.Event(enable_timing=True)
        ev_score_end = torch.cuda.Event(enable_timing=True)
        ev_score_start.record()

        # Single batch: reward_batch_size = None
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
# 4. Pipeline factory per setting
# ==============================================================================
def setup_models(setting: str, device: str = "cuda:0", lookahead_steps: Optional[int] = None):
    """
    Build the Phase 1 (lookahead) model, the Phase 2 (target) model and ImageReward
    with the Table 2 hyperparameters.
    """
    if not _DIFFUSERS_AVAILABLE:
        raise ImportError(
            "'diffusers' or 'huggingface_hub' is not installed.\n"
            "Install it with: pip install diffusers transformers accelerate"
        )
    print(f"\nLoading models for setting '{setting}' on {device}...")

    # 1. ImageReward model
    print("Loading ImageReward-v1.0...")
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

        # Phase 2 pipeline: FKDStableDiffusion (target steering)
        print(f"Loading SD 1.5: {model_id}...")
        pipe_phase2 = FKDStableDiffusion.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        pipe_phase2.scheduler = DDIMScheduler.from_config(pipe_phase2.scheduler.config)

        # Phase 1 pipeline: DPM-Solver (shares UNet/VAE/text encoder with Phase 2 to avoid duplicating VRAM)
        pipe_phase1 = StableDiffusionPipeline(
            vae=pipe_phase2.vae,
            text_encoder=pipe_phase2.text_encoder,
            tokenizer=pipe_phase2.tokenizer,
            unet=pipe_phase2.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(pipe_phase2.scheduler.config),
            safety_checker=None,
            feature_extractor=pipe_phase2.feature_extractor,
        )

        # Hyperparameters
        default_lookahead_steps = 5
        actual_lookahead_steps = lookahead_steps if lookahead_steps is not None else default_lookahead_steps
        num_particles_phase1 = 50
        num_target_images = 4

        if setting == "sdv1.5_ddim50":
            target_steps = 50
            target_eta = 0.0      # deterministic DDIM
            target_scale = 12.5
        else:  # sdv1.5_ddpm100
            target_steps = 100
            target_eta = 1.0      # stochastic DDPM
            target_scale = 12.5

    elif setting == "sdxl_dmd1":
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        # Phase 1 pipeline: SDXL DMD-1 (lookahead)
        print("Loading Phase 1: SDXL DMD-1...")
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

        # Phase 2 pipeline: FKDStableDiffusionXL (target steering)
        print("Loading Phase 2: FKDStableDiffusionXL...")
        pipe_phase2 = FKDStableDiffusionXL.from_pretrained(
            base_model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe_phase2.scheduler = DDIMScheduler.from_config(pipe_phase2.scheduler.config)

        default_lookahead_steps = 1
        actual_lookahead_steps = lookahead_steps if lookahead_steps is not None else default_lookahead_steps
        num_particles_phase1 = 100
        num_target_images = 4
        target_steps = 100
        target_eta = 1.0          # stochastic DDPM
        target_scale = 8.0

    else:
        raise ValueError(f"Unsupported setting '{setting}'. Choices: sdv1.5_ddim50, sdv1.5_ddpm100, sdxl_dmd1")

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
# 5. Core benchmark runner (one prompt, both phases)
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
    Run the full two-phase pipeline for one prompt and time each component:
      - T_lookahead
      - T_reward (single full-batch forward)
      - T_target (stops after the 4 target images are generated; no reward scoring)
    """
    prompt_str = item["prompt"]
    real_idx = item["original_idx"]
    n_p1 = config["num_particles_phase1"]
    n_tar = config["num_target_images"]
    is_sdxl = "sdxl" in getattr(pipe_phase1, "_name_or_path", "").lower() or hasattr(pipe_phase1, "unet") and pipe_phase1.unet.config.sample_size == 128

    # Synchronize the GPU before Phase 1
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------------------------------
    # Stage 1: Phase 1 lookahead sampling (DPM-5 or DMD-1)
    # --------------------------------------------------------------------------
    ev_look_start = torch.cuda.Event(enable_timing=True)
    ev_look_end = torch.cuda.Event(enable_timing=True)

    ev_look_start.record()
    with torch.inference_mode():
        prompt_batch = [prompt_str] * n_p1

        if is_sdxl:
            # SDXL Phase 1 lookahead (full batch)
            # guidance_scale=0.0 and timesteps=[399] as in the original DMD2 setup (CFG off, no doubled batch)
            latents = pipe_phase1(
                prompt_batch,
                num_inference_steps=config["lookahead_steps"],
                guidance_scale=0.0,
                timesteps=[399] if config["lookahead_steps"] == 1 else None,
                output_type="latent"
            ).images  # Tensor (N, 4, 128, 128)

            # VAE decode with batch size 1 to avoid a VRAM spike at 1024x1024
            shift_factor = getattr(pipe_phase1.vae.config, "shift_factor", 0.0)
            shift_factor = 0.0 if shift_factor is None else shift_factor
            latents_scaled = (latents / pipe_phase1.vae.config.scaling_factor) + shift_factor
            latents_scaled = latents_scaled.to(pipe_phase1.vae.dtype)  # match the VAE dtype
            decoded_chunks = []
            for v_i in range(0, latents_scaled.shape[0], 1):
                chunk = latents_scaled[v_i : v_i + 1]
                decoded_chunks.append(pipe_phase1.vae.decode(chunk, return_dict=False)[0])
            decoded_tensor = torch.cat(decoded_chunks, dim=0)  # Shape: (N, 3, 1024, 1024)
        else:
            # SD 1.5 Phase 1 lookahead
            latents = pipe_phase1(
                prompt_batch,
                num_inference_steps=config["lookahead_steps"],
                output_type="latent"
            ).images  # Tensor (N, 4, 64, 64)

            # VAE decode in chunks of 10 (avoids an unnecessary ~18 GB VRAM spike)
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
    # Stage 2: Phase 1 reward annotation (pure scoring, matches Table 9)
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

    # Pack the lookahead data into the in-memory dataset for Phase 2
    lookahead_ds = InMemoryLookaheadDataset()
    lookahead_ds.set_datapoint(
        prompt_idx=real_idx,
        latents=latents,
        rewards=torch.tensor(smoothed_scores, dtype=torch.float32),
        prompts=[prompt_str] * n_p1
    )

    # --------------------------------------------------------------------------
    # Stage 3: Phase 2 target sampling (closed-form guidance)
    # --------------------------------------------------------------------------
    fkd_args = dict(
        lmbda=5000.0,
        scale=config["target_scale"],
        reward_type="ImageReward",
        num_particles=n_tar,             # exactly 4 target images
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

    # Record the Phase 1 peak VRAM
    torch.cuda.synchronize(device)
    peak_vram_p1_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Reset stats so that Phase 2 (target sampling) peak VRAM is measured separately (Table 2, Figure 10)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    ev_tar_start = torch.cuda.Event(enable_timing=True)
    ev_tar_end = torch.cuda.Event(enable_timing=True)

    ev_tar_start.record()
    with torch.inference_mode():
        target_prompts = [prompt_str] * n_tar
        # Generate the 4 target images with closed-form steering
        target_output = pipe_phase2(
            target_prompts,
            prompt_idx=real_idx,
            num_inference_steps=config["target_steps"],
            eta=config["target_eta"],
            fkd_args=fkd_args,
        )
        # Stop right after image generation; Phase 2 rewards are not scored
        _ = target_output.images if hasattr(target_output, "images") else target_output

    ev_tar_end.record()
    torch.cuda.synchronize(device)
    t_target = ev_tar_start.elapsed_time(ev_tar_end) / 1000.0
    vram_phase2_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Per-prompt totals (Table 9 convention: lookahead + reward scoring + target)
    t_total = t_lookahead + t_reward + t_target
    peak_vram_gib = max(peak_vram_p1_gib, vram_phase2_gib)

    return {
        "prompt_idx": real_idx,
        "prompt": prompt_str,
        "category": item.get("tag", "general"),
        "t_lookahead": t_lookahead,
        "t_reward": t_reward,              # pure scoring
        "t_convert": t_convert,            # tensor -> PIL conversion
        "t_reward_total": t_reward + t_convert,
        "t_target": t_target,
        "t_total": t_total,
        "vram_phase2_gib": vram_phase2_gib,
        "peak_vram_gib": peak_vram_gib,
        "scores_mean": float(np.mean(smoothed_scores)),
        "scores_max": float(np.max(smoothed_scores)),
    }


# ==============================================================================
# 6. Main loop and real-time logging
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark Table 2 Efficiency (Time & Peak VRAM) on A100")
    parser.add_argument("--setting", type=str, default="sdv1.5_ddpm100",
                        choices=["sdv1.5_ddim50", "sdv1.5_ddpm100", "sdxl_dmd1"],
                        help="Table 2 benchmark configuration: sdv1.5_ddim50, sdv1.5_ddpm100, sdxl_dmd1")
    parser.add_argument("--method", type=str, default="rs-lidar", choices=["rs-lidar", "lidar"],
                        help="Sampling method: rs-lidar (Randomized Smoothing) or lidar (Vanilla)")
    parser.add_argument("--num_prompts", type=int, default=3, help="Number of prompts to benchmark and average")
    parser.add_argument("--prompts_per_category", type=int, default=None,
                        help="Number of representative prompts per category (e.g. 2 -> 12 prompts across 6 GenEval categories)")
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
                        help="Optional backup directory; CSV and JSON results are copied there after every prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device identifier")

    args = parser.parse_args()

    # Device setup
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
        vram_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"Device: {gpu_name} ({vram_total:.1f} GiB VRAM)")
    else:
        print("Warning: running on CPU; timings are not representative of the A100 benchmark.")

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve default parameters from setting and method
    if args.method == "lidar":
        sigma = 0.0
        num_mc = 1
    else:
        num_mc = args.num_mc_samples if args.num_mc_samples is not None else 4
        if "sdxl" in args.setting:
            sigma = 0.5 if (args.sigma is None or args.sigma == 1.0) else args.sigma
        elif args.sigma is not None:
            sigma = args.sigma
        else:
            sigma = 1.0

    # Resolve reward_batch_size
    if args.reward_batch_size is None or str(args.reward_batch_size).lower() in ["none", "null", "full"]:
        reward_batch_size = None
    else:
        reward_batch_size = int(args.reward_batch_size)

    # Resolve the list of Phase 1 step counts to run
    if "sdxl" in args.setting:
        if args.sweep_steps is not None and args.sweep_steps.strip() not in ["1", ""]:
            print(f"[SDXL DMD-1] DMD-1 is a 1-step distilled model (Table 2). Ignoring sweep '{args.sweep_steps}' and running a single 1-step pass.")
        elif args.lookahead_steps is not None and args.lookahead_steps != 1:
            print(f"[SDXL DMD-1] DMD-1 is a 1-step distilled model (Table 2). Using 1 lookahead step instead of {args.lookahead_steps}.")
        steps_to_run = [1]
    elif args.sweep_steps is not None and args.sweep_steps.strip() != "":
        steps_to_run = [int(s.strip()) for s in args.sweep_steps.split(",") if s.strip().isdigit()]
    elif args.lookahead_steps is not None:
        steps_to_run = [args.lookahead_steps]
    else:
        default_step = 5
        steps_to_run = [default_step]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print("TABLE 2 EFFICIENCY BENCHMARK (TIME & MEMORY)")
    print(f"{'='*75}")
    print(f"- Setting:             {args.setting}")
    print(f"- Method:              {args.method.upper()}")
    print(f"- Phase 1 steps:       {steps_to_run if len(steps_to_run) > 1 else steps_to_run[0]}")
    print(f"- Smoothing sigma:     {sigma}")
    print(f"- Monte Carlo M:       {num_mc}")
    print(f"- Reward batch size:   {'FULL BATCH (None)' if reward_batch_size is None else reward_batch_size}")
    if args.prompts_per_category is not None:
        print(f"- Prompt selection:    stratified, {args.prompts_per_category} prompts/category")
    else:
        print(f"- Number of prompts:   {args.num_prompts}")
    print(f"- Output directory:    {args.output_dir}")
    print(f"{'='*75}\n")

    # 1. Load models and pipelines (once per session)
    pipe_phase1, pipe_phase2, ir_model, config = setup_models(
        args.setting, device=device, lookahead_steps=steps_to_run[0]
    )

    # 2. Load prompts
    prompts_data = load_prompts(
        args.prompt_path,
        max_prompts=args.num_prompts,
        prompt_indices=args.prompt_indices,
        prompts_per_category=args.prompts_per_category
    )
    print(f"Selected {len(prompts_data)} prompts for benchmarking.")

    # 3. GPU warm-up (optional)
    if args.warmup and len(prompts_data) > 0:
        print("\nRunning one warm-up prompt to compile CUDA kernels and load weights into VRAM...")
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
        print("Warm-up done. Starting timed runs.")

    # 4. Timed runs for each Phase 1 step count (single run or sweep, e.g. 2,3,4,5 steps)
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
            "prompt_idx", "category", "prompt", "setting", "method", "lookahead_steps", "sigma", "num_mc",
            "t_lookahead", "t_reward", "t_convert", "t_target", "t_total", "vram_phase2_gib", "peak_vram_gib"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

        results_list = []

        print(f"\n[{step_idx + 1}/{len(steps_to_run)}] Benchmarking {len(prompts_data)} prompts with {cur_step} Phase 1 steps...")
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
                "category": res.get("category", ""),
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

            # Copy the CSV to the backup directory after every prompt (if configured)
            if args.drive_backup_dir:
                try:
                    os.makedirs(args.drive_backup_dir, exist_ok=True)
                    drive_csv_target = os.path.join(args.drive_backup_dir, csv_filename)
                    shutil.copyfile(csv_path, drive_csv_target)
                except Exception as e_drive:
                    print(f"Warning: could not copy CSV to the backup directory ({e_drive})")

            cat_str = f" ({res.get('category')})" if res.get('category') else ""
            print(f"\n  [Step {cur_step} | #{res['prompt_idx']:03d}{cat_str}] T_look: {res['t_lookahead']:.2f}s | T_rew (scoring): {res['t_reward']:.2f}s (PIL convert: {res['t_convert']:.2f}s) | T_tar: {res['t_target']:.2f}s | T_tot: {res['t_total']:.2f}s | VRAM P2: {res['vram_phase2_gib']:.2f} GiB (Peak: {res['peak_vram_gib']:.2f} GiB)")

        # Averages
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

        # Copy the JSON to the backup directory
        if args.drive_backup_dir:
            try:
                drive_json_target = os.path.join(args.drive_backup_dir, json_filename)
                shutil.copyfile(json_path, drive_json_target)
                print(f"Copied step {cur_step} results (JSON & CSV) to the backup directory: {args.drive_backup_dir}")
            except Exception as e_drive:
                print(f"Warning: could not copy JSON to the backup directory ({e_drive})")

        sweep_summaries.append(summary)

        # Per-step summary table
        paper_t_look = "5.69s" if "sdv1.5" in args.setting else "4.30s"
        paper_t_rew = "0.65s" if "sdv1.5" in args.setting else "1.30s"
        paper_t_tar = "7.07s" if "ddpm100" in args.setting else ("3.58s" if "ddim50" in args.setting else "50.00s")
        paper_t_tot = "13.41s" if "ddpm100" in args.setting else ("9.92s" if "ddim50" in args.setting else "55.60s")
        paper_vram = "8.90 GiB" if "sdv1.5" in args.setting else "33.84 GiB"

        print(f"\n{'='*95}")
        print(f"TIME & MEMORY BREAKDOWN (PHASE 1: {cur_step} STEPS)")
        print(f"{'='*95}")
        print(f"Setting: {args.setting} | Method: {args.method.upper()} | Lookahead Steps: {cur_step} | Prompts: {len(results_list)} | Batch: {'Full Batch' if reward_batch_size is None else reward_batch_size}")
        print(f"{'-'*95}")
        print(f"- T_lookahead (N lookahead particles):          {mean_t_look:6.2f} s  (paper, DPM-5 / n=50: {paper_t_look})")
        print(f"- T_reward    (pure ImageReward scoring):       {mean_t_rew:6.2f} s  (paper, n=50 / M=1:   {paper_t_rew})")
        print(f"  └─ tensor -> PIL conversion overhead:         {mean_t_conv:6.2f} s")
        print(f"- T_target    (Phase 2 denoising, 4 images):    {mean_t_tar:6.2f} s  (paper, target:       {paper_t_tar})")
        print("--------------------------------------------------------------------------------------------")
        print(f"TOTAL TIME, Table 9 (look + rew + tar):          {mean_t_tot:6.2f} s  (paper, Table 9:     {paper_t_tot})")
        print(f"PHASE 2 VRAM (target sampling, Table 2):         {mean_vram_p2:6.2f} GiB  (paper & Fig. 10:  {paper_vram})")
        print(f"OVERALL PEAK VRAM:                               {max_peak_vram:6.2f} GiB")
        print(f"{'='*95}")
        print(f"Step {cur_step} results saved to:")
        print(f"   CSV:  {csv_path}")
        print(f"   JSON: {json_path}")
        if args.drive_backup_dir:
            print(f"   Backup: {args.drive_backup_dir}")
        print(f"{'='*95}\n")

    # Sweep over several step counts: write the combined files and print the ablation table
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
                print(f"Copied the sweep summary (JSON & CSV) to the backup directory: {args.drive_backup_dir}")
            except Exception as e_drive:
                print(f"Warning: could not copy the sweep files to the backup directory ({e_drive})")

        print(f"\n{'='*105}")
        print(f"LOOKAHEAD STEPS ABLATION (SWEEP: {steps_to_run})")
        print(f"{'='*105}")
        print(f"Setting: {args.setting} | Method: {args.method.upper()} | Prompts: {len(prompts_data)}")
        print(f"{'-'*105}")
        print(f"{'Lookahead steps':<22} | {'T_lookahead':<12} | {'T_reward':<10} | {'T_target':<10} | {'T_total':<10} | {'VRAM P2':<10} | {'Peak VRAM':<10}")
        print(f"{'-'*105}")
        for sm in sweep_summaries:
            st = sm['lookahead_steps']
            st_name = f"delta = {st} (DPM-{st})" if "sdv1.5" in args.setting else f"delta = {st} (DMD-{st})"
            print(f"{st_name:<22} | {sm['mean_t_lookahead_sec']:10.2f}s | {sm['mean_t_reward_sec']:8.2f}s | {sm['mean_t_target_sec']:8.2f}s | {sm['mean_t_total_sec']:8.2f}s | {sm['mean_vram_phase2_gib']:6.2f} GiB | {sm['max_peak_vram_gib']:6.2f} GiB")
        print(f"{'='*105}")
        print("Sweep summary saved to:")
        print(f"   CSV:  {sweep_csv_path}")
        print(f"   JSON: {sweep_json_path}")
        print(f"{'='*105}\n")


if __name__ == "__main__":
    main()
