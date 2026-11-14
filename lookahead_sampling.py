"""Phase 1 of LiDAR / RS-LiDAR: sample a pool of lookahead particles and score them.

For every prompt, N lookahead latents are drawn with a few-step solver, decoded and scored. Each
prompt folder stores `samples/latent.pt` (latents sorted by reward) and `results.json`
(`<metric>.result` per particle); Phase 2 (`LiDAR_sampling.py`) loads them as the dataset of the
closed-form lookahead guidance.

With `--use_smoothing` the stored reward is the randomized-smoothing estimate of RS-LiDAR,

    R_sigma(x) = (1/M) * sum_m R(clip(D(x) + sigma * u_m, -1, 1)),

computed in decoded pixel space D(x). `mc_rewards_loop` evaluates the [M, N] matrix of rewards
of the noisy decodings, with u_m drawn according to `--noise_mode` (I.I.D., Coupled CRN shared
across particles, or antithetic pairs (u, -u)). `compute_entropic_antithetic_rewards` aggregates
it into the Monte Carlo mean, or the entropic-risk corrected mean when `--beta_risk > 0`.
"""
import argparse
import json
import math
import os
import pickle
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from rslidar_utils import load_geneval_metadata, parse_prompt_indices, setup_runtime, str2bool

setup_runtime()

from diffusers import (  # noqa: E402
    AutoPipelineForText2Image,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    FluxPipeline,
    LCMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

sys.path.append("fkd_diffusers")

from fks_utils import do_eval  # noqa: E402

NOISE_MODES = ["independent_iid", "independent_antithetic", "coupled_crn"]


@torch.no_grad()
def generate_smoothed_pixel_batch(
    images_0: torch.Tensor,
    sigma: float,
    M: int = 4,
    noise_mode: str = "independent_iid",
    clamp_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> torch.Tensor:
    """Vectorized smoothing batch: returns clip(images_0 + sigma * u_m) with shape [M, N, C, H, W].

    `independent_antithetic` uses pairs (u, -u) (M must be even), `independent_iid` independent
    noise per particle, and `coupled_crn` one noise draw per m shared by all N particles.
    """
    N, C, H, W = images_0.shape
    device, dtype = images_0.device, images_0.dtype

    if sigma <= 0.0 or M <= 0:
        return images_0.unsqueeze(0).repeat(max(1, M), 1, 1, 1, 1)

    if noise_mode == "independent_antithetic":
        assert M % 2 == 0, f"M must be even for antithetic sampling, got M={M}."
        half_M = M // 2
        u_half = torch.randn((half_M, N, C, H, W), device=device, dtype=dtype)
        u = torch.cat([u_half, -u_half], dim=0)
    elif noise_mode == "independent_iid":
        u = torch.randn((M, N, C, H, W), device=device, dtype=dtype)
    elif noise_mode == "coupled_crn":
        u = torch.randn((M, 1, C, H, W), device=device, dtype=dtype).expand(M, N, C, H, W)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}. Expected one of {NOISE_MODES}.")

    perturbed = images_0.unsqueeze(0) + sigma * u
    if clamp_range is not None:
        perturbed = torch.clamp(perturbed, clamp_range[0], clamp_range[1])
    del u
    return perturbed


def mc_rewards_loop(
    decoded: torch.Tensor,
    sigma: float,
    num_mc_samples: int,
    noise_mode: str,
    score_fn: Callable[[torch.Tensor], Dict[str, List[float]]],
) -> Dict[str, torch.Tensor]:
    """Monte Carlo rewards of the smoothed reward R_sigma, one noise sample at a time (low memory).

    For m = 1..M, scores `clip(decoded + sigma * u_m, -1, 1)` with `score_fn`, which maps a batch of
    N decoded images to `{metric: [N rewards]}`. Returns `{metric: float32 tensor [M, N]}`.

    Noise: antithetic pairs (u_m, -u_m) (all M/2 draws are made up front), a single draw per m shared
    by all particles (`coupled_crn`), or independent draws (`independent_iid`, and antithetic with odd M).
    """
    half_m = num_mc_samples // 2
    is_antithetic = noise_mode == "independent_antithetic" and num_mc_samples % 2 == 0
    u_half = [torch.randn_like(decoded) for _ in range(half_m)] if is_antithetic else None

    rows: Dict[str, List[List[float]]] = {}
    for m in range(num_mc_samples):
        if sigma > 0:
            if noise_mode == "coupled_crn":
                noise = torch.randn((1, *decoded.shape[1:]), device=decoded.device, dtype=decoded.dtype) * sigma
            elif is_antithetic:
                u_m = u_half[m] if m < half_m else -u_half[m - half_m]
                noise = u_m * sigma
            else:
                noise = torch.randn_like(decoded) * sigma
            noisy = (decoded + noise).clamp(-1.0, 1.0)
        else:
            noisy = decoded.clamp(-1.0, 1.0)
        for metric, values in score_fn(noisy).items():
            rows.setdefault(metric, []).append(values)
    return {metric: torch.tensor(values, dtype=torch.float32) for metric, values in rows.items()}


@torch.no_grad()
def compute_entropic_antithetic_rewards(
    rewards_mc: torch.Tensor,
    sigma: float = 1.0,
    beta_risk: float = 0.0,
    noise_mode: str = "independent_iid",
) -> Dict[str, torch.Tensor]:
    """Aggregate an [M, N] matrix of Monte Carlo rewards into the smoothed reward of each particle.

    Returns `r_corrected` (the MC mean, minus the entropic-risk penalty `beta_risk / (2M) * Var` when
    `beta_risk != 0`), `grad_energy_sq` and `pair_variance`. For antithetic pairs the variance is split
    into the half-difference (gradient energy) and half-sum (curvature) terms.
    """
    r_fp32 = rewards_mc.to(torch.float32)
    M, N = r_fp32.shape

    if noise_mode != "independent_antithetic" or M % 2 != 0:
        r_mean = r_fp32.mean(dim=0)
        variance = r_fp32.var(dim=0, unbiased=True) if M > 1 else torch.zeros_like(r_mean)
        r_corrected = r_mean if beta_risk == 0.0 else (r_mean - (beta_risk / (2.0 * float(M))) * variance)
        grad_energy_sq = variance / (sigma ** 2 + 1e-8)
        return {
            "r_corrected": r_corrected.to(rewards_mc.dtype),
            "grad_energy_sq": grad_energy_sq.to(rewards_mc.dtype),
            "pair_variance": variance.to(rewards_mc.dtype)
        }

    half_M = M // 2
    r_pos = r_fp32[:half_M]
    r_neg = r_fp32[half_M:]

    # Half-sum / half-difference decomposition of antithetic pairs.
    pair_sums = 0.5 * (r_pos + r_neg)
    pair_diffs = 0.5 * (r_pos - r_neg)

    grad_energy_sq = torch.mean(pair_diffs ** 2, dim=0) / (sigma ** 2 + 1e-8)
    curv_var = pair_sums.var(dim=0, unbiased=True) if half_M >= 2 else torch.zeros_like(grad_energy_sq)

    total_var = (sigma ** 2) * grad_energy_sq + curv_var
    r_mean = r_fp32.mean(dim=0)
    r_corrected = r_mean if beta_risk == 0.0 else (r_mean - (beta_risk / (2.0 * float(M))) * total_var)

    return {
        "r_corrected": r_corrected.to(rewards_mc.dtype),
        "grad_energy_sq": grad_energy_sq.to(rewards_mc.dtype),
        "pair_variance": total_var.to(rewards_mc.dtype)
    }


def _reward_summary(values: List[float]) -> dict:
    arr = torch.tensor(values, dtype=torch.float32)
    return {
        "result": values,
        "mean": arr.mean().item(),
        "std": arr.std().item() if len(values) > 1 else 0.0,
        "max": arr.max().item(),
        "min": arr.min().item(),
    }


def summarize_smoothed_rewards(multi_rewards_mc: Dict[str, torch.Tensor], args) -> dict:
    """Turn the per-metric [M, N] Monte Carlo rewards into `results.json` entries."""
    results = {}
    for metric, scores_mat in multi_rewards_mc.items():
        if args.noise_mode == "coupled_crn" and args.beta_risk == 0.0:
            results[metric] = _reward_summary(scores_mat.mean(dim=0).tolist())
        else:
            stats = compute_entropic_antithetic_rewards(
                scores_mat, sigma=args.sigma, beta_risk=args.beta_risk, noise_mode=args.noise_mode
            )
            results[metric] = _reward_summary(stats["r_corrected"].tolist())
            results[metric]["grad_energy_sq"] = stats["grad_energy_sq"].tolist()
            results[metric]["pair_variance"] = stats["pair_variance"].tolist()

    if args.use_pareto and len(multi_rewards_mc) > 1:
        # Imported lazily: multi_reward_evaluator imports this module.
        from multi_reward_evaluator import compute_multi_reward_pareto_potential
        pareto_pot, _ = compute_multi_reward_pareto_potential(
            multi_rewards_mc,
            sigma=args.sigma,
            pareto_rho=args.pareto_rho,
            beta_risk=args.beta_risk,
            noise_mode=args.noise_mode,
        )
        results["Pareto"] = _reward_summary(pareto_pot.tolist())
    return results


def load_lookahead_pipeline(model_name, device):
    """Load the few-step lookahead model used in Phase 1."""
    if "runwayml/stable-diffusion-v1-5" in model_name:  # SD1.5, DPM-Solver
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif "mhdang/dpo" in model_name and "xl" not in model_name:  # SD1.5-DPO, DPM-Solver
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe.unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sd1.5-text2image-v1", subfolder="unet", torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif "stabilityai/stable-diffusion-xl-base-1.0" in model_name:  # SDXL, DPM-Solver
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.vae.to(dtype=torch.float32)

    elif "dmd2_sdxl_4step_lora_fp16.safetensors" in model_name:  # SDXL-DMD2, 4 steps
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.load_lora_weights(hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_4step_lora_fp16.safetensors"))
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.vae.to(dtype=torch.float32)

    elif "dmd2_sdxl_1step_unet_fp16.bin" in model_name:  # SDXL-DMD2, 1 step
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(torch.load(hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_1step_unet_fp16.bin"),
                                        map_location=device))
        pipe = DiffusionPipeline.from_pretrained(
            base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.to(dtype=torch.float32)

    elif "latent-consistency/lcm-lora-sdxl" in model_name:  # SDXL-LCM-LoRA, 4 steps
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        pipe.fuse_lora()
        pipe.vae.to(dtype=torch.float32)

    elif "sdxl_lightning_4step_lora.safetensors" in model_name:  # SDXL-Lightning, 4 steps
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"))
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.fuse_lora()
        pipe.vae.to(dtype=torch.float32)

    elif "Hyper-SDXL-1step-Unet.safetensors" in model_name:  # SDXL-Hyper-SD, 1 step
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-Unet.safetensors"),
                                       device=device))
        pipe = DiffusionPipeline.from_pretrained(
            base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.to(dtype=torch.float32)

    elif "Hyper-FLUX.1-dev-8steps-lora.safetensors" in model_name:  # FLUX.1-dev + Hyper-SD LoRA, 8 steps
        base_model_id = "black-forest-labs/FLUX.1-dev"
        print(f"Loading FLUX lookahead model {base_model_id} with ByteDance/Hyper-SD 8-step LoRA")
        pipe = FluxPipeline.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN", True))
        pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"))
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to(device, dtype=torch.float16)

    else:
        raise ValueError(f"Unsupported lookahead model: {model_name}")
    return pipe


def sample_lookahead_latents(pipe, model_name, prompt, num_inference_steps):
    """Run the lookahead solver and return latents (packed for FLUX)."""
    if "dmd2_sdxl_4step_lora_fp16.safetensors" in model_name:
        return pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, timesteps=[999, 749, 499, 249],
                    output_type="latent").images
    if "dmd2_sdxl_1step_unet_fp16.bin" in model_name:
        return pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399],
                    output_type="latent").images
    if "latent-consistency/lcm-lora-sdxl" in model_name or "sdxl_lightning_4step_lora.safetensors" in model_name:
        return pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, output_type="latent").images
    if "Hyper-SDXL-1step-Unet.safetensors" in model_name:
        return pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[800],
                    output_type="latent").images
    if "Hyper-FLUX.1-dev-8steps-lora.safetensors" in model_name:
        return pipe(prompt=prompt, num_inference_steps=8, guidance_scale=3.5, output_type="latent").images
    # SD1.5, SD1.5-DPO and SDXL with a multistep DPM-Solver.
    return pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=num_inference_steps,
                output_type="latent").images


def _cache_mismatches(res_cached, args):
    """List hyper-parameter mismatches between a cached smoothed `results.json` and the current run."""
    s_info = res_cached["smoothing"]
    cached_m = s_info.get("num_mc_samples")
    cached_mode = s_info.get("noise_mode")
    cached_sig = s_info.get("sigma")
    cached_beta = s_info.get("beta_risk")
    reasons = []
    if cached_m is not None and cached_m != args.num_mc_samples:
        reasons.append(f"M: {cached_m} != {args.num_mc_samples}")
    if cached_mode is not None and cached_mode != args.noise_mode:
        reasons.append(f"noise_mode: '{cached_mode}' != '{args.noise_mode}'")
    if cached_sig is not None and abs(cached_sig - args.sigma) > 1e-4:
        reasons.append(f"sigma: {cached_sig} != {args.sigma}")
    if cached_beta is not None and args.beta_risk != 0.0 and abs(cached_beta - args.beta_risk) > 1e-4:
        reasons.append(f"beta_risk: {cached_beta} != {args.beta_risk}")
    return reasons


def main(args):
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if args.gpu_id is not None:
            actual_gpu_id = args.gpu_id if args.gpu_id < num_devices else (args.gpu_id % num_devices)
            torch.cuda.set_device(actual_gpu_id)
            device = f"cuda:{actual_gpu_id}"
            print(f"Assigned process to GPU {actual_gpu_id} (device {actual_gpu_id + 1}/{num_devices}): "
                  f"{torch.cuda.get_device_name(actual_gpu_id)}")
        else:
            actual_gpu_id = 0
            device = "cuda:0"
            torch.cuda.set_device(0)
    else:
        actual_gpu_id = "cpu"
        device = "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    parsed_indices = parse_prompt_indices(args.prompt_indices)
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt, prompt_indices=parsed_indices)

    pipe = load_lookahead_pipeline(args.model_name, device)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    is_flux = "flux" in args.model_name.lower()
    model_prefix = "FLUX_" if is_flux else ""
    prefix = f"{model_prefix}{args.seed}_{args.num_particles}_{args.num_inference_steps}"
    if args.use_smoothing:
        if args.noise_mode == "coupled_crn":
            suffix = "_crn"
        elif args.noise_mode == "independent_iid":
            suffix = "_iid"
        else:
            suffix = f"_{args.noise_mode}"
        prefix += f"_rs_sig{args.sigma}_M{args.num_mc_samples}{suffix}"
    output_dir = os.path.join(args.output_dir, args.run_name or prefix)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    reward_metrics = args.metrics_to_compute.split("#")
    metrics_to_compute = list(reward_metrics)  # may additionally report "Pareto"

    # Load and warm up the reward models once.
    print(f"Initializing reward models ({args.metrics_to_compute})...")
    do_eval(prompt=["test"], images=[Image.new("RGB", (224, 224))], metrics_to_compute=reward_metrics)

    metrics_arr = {metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute}
    n_samples = 0
    average_time = 0

    total_prompts = len(prompt_data)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        print(f"Shard {args.shard_id + 1}/{args.num_shards}: processing prompts {start_idx} to {end_idx - 1} "
              f"(total: {end_idx - start_idx})")
    else:
        start_idx = 0
        end_idx = total_prompts

    is_xl = "xl" in args.model_name.lower() or is_flux
    expected_latent_dim = 128 if is_xl else 64

    for prompt_idx in tqdm(range(start_idx, end_idx), desc=f"Shard-{args.shard_id}"):
        item = prompt_data[prompt_idx]
        real_idx = item.get("original_idx", prompt_idx)
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{real_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        results_file = os.path.join(prompt_path, "results.json")
        latent_file = os.path.join(prompt_path, "samples", "latent.pt")
        if args.resume and not args.overwrite and os.path.exists(results_file) and os.path.exists(latent_file):
            try:
                lat_valid = True
                lat_check = torch.load(latent_file, map_location="cpu")
                check_t = lat_check[0] if isinstance(lat_check, (list, tuple)) else lat_check
                if hasattr(check_t, "shape") and check_t.shape[-1] != expected_latent_dim:
                    lat_valid = False
                    print(f"[Prompt {real_idx:05d}] cached latent.pt is {check_t.shape[-1]}x{check_t.shape[-1]}, "
                          f"expected {expected_latent_dim}x{expected_latent_dim} for {args.model_name}; regenerating.")

                if lat_valid:
                    with open(results_file, "r") as f:
                        res_cached = json.load(f)
                    if args.use_smoothing:
                        if "smoothing" not in res_cached or not res_cached["smoothing"].get("enabled", False):
                            lat_valid = False
                            print(f"[Prompt {real_idx:05d}] cached results.json has unsmoothed rewards but "
                                  f"--use_smoothing is set; recomputing.")
                        else:
                            mismatch_reasons = _cache_mismatches(res_cached, args)
                            if mismatch_reasons:
                                lat_valid = False
                                print(f"[Prompt {real_idx:05d}] cached results.json does not match "
                                      f"({', '.join(mismatch_reasons)}); recomputing.")
                    if lat_valid:
                        for metric in metrics_to_compute:
                            if metric in res_cached:
                                for stat in ("mean", "max", "min", "std"):
                                    metrics_arr[metric][stat] += res_cached[metric][stat]
                        n_samples += 1
                        print(f"[Shard {args.shard_id} | GPU {actual_gpu_id}] Prompt #{real_idx:05d}: "
                              f"valid cached result found, skipping (resume).")
                        continue
            except Exception as e:
                print(f"Error loading cached result for prompt {real_idx}: {e}")

        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        # Optionally reuse latents of a previous Phase 1 run (rewards are recomputed).
        reused_latent_file = None
        cached_reused_latents = None
        if args.reuse_latents_from:
            candidates = [
                os.path.join(args.reuse_latents_from, f"{real_idx:0>5}", "samples", "latent.pt"),
                os.path.join("Lookahead_samples", args.reuse_latents_from, f"{real_idx:0>5}", "samples", "latent.pt"),
                os.path.join(args.output_dir, args.reuse_latents_from, f"{real_idx:0>5}", "samples", "latent.pt"),
            ]
            for c in candidates:
                if not os.path.exists(c):
                    continue
                try:
                    test_lat = torch.load(c, map_location="cpu")
                except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
                    continue
                test_t = test_lat[0] if isinstance(test_lat, (list, tuple)) else test_lat
                if isinstance(test_lat, (list, tuple)):
                    n_avail = len(test_lat)
                else:
                    n_avail = test_lat.shape[0] if hasattr(test_lat, "shape") else 0
                if hasattr(test_t, "shape") and test_t.shape[-1] == expected_latent_dim and n_avail >= args.num_particles:
                    reused_latent_file = c
                    cached_reused_latents = test_lat
                    break
            if reused_latent_file is None:
                raise FileNotFoundError(
                    f"--reuse_latents_from='{args.reuse_latents_from}' was given, but no valid latent.pt was found for "
                    f"prompt #{real_idx:05d} (need {expected_latent_dim}x{expected_latent_dim} latents and "
                    f">= {args.num_particles} particles). Searched: {candidates}"
                )

        start_time = datetime.now()
        with torch.inference_mode():
            if reused_latent_file is not None:
                if prompt_idx == start_idx or prompt_idx % 50 == 0:
                    print(f"[Prompt {real_idx:05d}] reusing latents from {reused_latent_file}")
                if isinstance(cached_reused_latents, (list, tuple)):
                    latents = torch.stack([x.to(device) for x in cached_reused_latents[:args.num_particles]])
                else:
                    latents = cached_reused_latents[:args.num_particles].to(device)
                cached_reused_latents = None
            else:
                latents = sample_lookahead_latents(pipe, args.model_name, prompt, args.num_inference_steps)

            if "xl" in args.model_name or "XL" in args.model_name:
                latents = latents.to(torch.float32)

            if is_flux:
                latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
                scaled_latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                vae_batch_size = 1
                decoded_chunks = []
                for v_i in range(0, scaled_latents.shape[0], vae_batch_size):
                    chunk = scaled_latents[v_i: v_i + vae_batch_size]
                    decoded_chunks.append(pipe.vae.decode(chunk, return_dict=False)[0])
                decoded_tensor = torch.cat(decoded_chunks, dim=0)
            else:
                scaled_latents = latents / pipe.vae.config.scaling_factor
                decoded_tensor = pipe.vae.decode(scaled_latents, return_dict=False)[0]
            if args.save_individual_images or not args.use_smoothing:
                clean_images = pipe.image_processor.postprocess(decoded_tensor, output_type="pil")
            else:
                clean_images = None

        time_taken = datetime.now() - start_time
        images = clean_images

        if args.use_smoothing:
            eval_prompt = prompt[0]

            if args.use_smoothing_loop:
                def score_fn(noisy):
                    noisy_pil = pipe.image_processor.postprocess(noisy, output_type="pil")
                    evals = do_eval(prompt=[eval_prompt] * len(noisy_pil), images=noisy_pil,
                                    metrics_to_compute=reward_metrics)
                    return {metric: evals[metric]["result"] for metric in reward_metrics}

                with torch.inference_mode():
                    multi_rewards_mc = mc_rewards_loop(
                        decoded_tensor, args.sigma, args.num_mc_samples, args.noise_mode, score_fn
                    )
            else:
                num_mc = args.num_mc_samples
                n_particles = decoded_tensor.shape[0]
                with torch.inference_mode():
                    if args.sigma > 0:
                        if args.noise_mode == "coupled_crn":
                            noisy_t_list = [
                                (decoded_tensor + torch.randn((1, *decoded_tensor.shape[1:]), device=decoded_tensor.device,
                                                              dtype=decoded_tensor.dtype) * args.sigma).clamp(-1.0, 1.0)
                                for _ in range(num_mc)
                            ]
                            noisy_t_all = torch.cat(noisy_t_list, dim=0)
                        else:
                            perturbed_all = generate_smoothed_pixel_batch(
                                decoded_tensor, sigma=args.sigma, M=num_mc, noise_mode=args.noise_mode
                            )
                            noisy_t_all = perturbed_all.view(-1, *decoded_tensor.shape[1:])
                    else:
                        noisy_t_all = decoded_tensor.clamp(-1.0, 1.0).repeat(num_mc, 1, 1, 1)

                    noisy_pil_all = pipe.image_processor.postprocess(noisy_t_all, output_type="pil")
                    eval_all = do_eval(prompt=[eval_prompt] * len(noisy_pil_all), images=noisy_pil_all,
                                       metrics_to_compute=reward_metrics)
                multi_rewards_mc = {
                    metric: torch.tensor(eval_all[metric]["result"], dtype=torch.float32).view(num_mc, n_particles)
                    for metric in reward_metrics
                }

            results = summarize_smoothed_rewards(multi_rewards_mc, args)
            if "Pareto" in results and "Pareto" not in metrics_to_compute:
                metrics_to_compute.append("Pareto")
                metrics_arr["Pareto"] = dict(mean=0, max=0, min=0, std=0)
        else:
            results = do_eval(prompt=[prompt[0]] * len(images), images=images, metrics_to_compute=reward_metrics)

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = real_idx
        if args.use_smoothing:
            results["smoothing"] = {
                "enabled": True,
                "sigma": args.sigma,
                "num_mc_samples": args.num_mc_samples,
                "domain": "image",
                "mode": "loop" if args.use_smoothing_loop else "vectorized",
                "noise_mode": args.noise_mode,
                "beta_risk": args.beta_risk,
                "use_crn": args.noise_mode == "coupled_crn",
                "reused_latents_from": reused_latent_file,
            }

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        # Sort particles (and their rewards) by the guidance reward, best first.
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        if images is not None:
            images = [images[i] for i in sorted_idx]
        latents = [latents[i] for i in sorted_idx]
        for metric in metrics_to_compute:
            if metric in results and "result" in results[metric]:
                for key in ("result", "grad_energy_sq", "pair_variance"):
                    if key in results[metric]:
                        results[metric][key] = [results[metric][key][i] for i in sorted_idx]

        for metric in metrics_to_compute:
            for stat in ("mean", "max", "min", "std"):
                metrics_arr[metric][stat] += results[metric][stat]

        for metric in metrics_to_compute:
            print(metric, metrics_arr[metric]["mean"] / n_samples, metrics_arr[metric]["max"] / n_samples)

        sample_path = os.path.join(prompt_path, "samples")
        os.makedirs(sample_path, exist_ok=True)
        torch.save(latents, os.path.join(sample_path, "latent.pt"))

        if args.save_individual_images and images is not None:
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))
            n_show = len(images)
            _, ax = plt.subplots(1, n_show, figsize=(max(4, n_show * 4), 4))
            if n_show == 1:
                ax = [ax]
            for i, image in enumerate(images):
                ax[i].imshow(image)
                ax[i].axis("off")

            plt.suptitle(prompt[0])
            plt.savefig(os.path.join(prompt_path, "grid.png"))
            plt.close()

        with open(os.path.join(prompt_path, "results.json"), "w") as f:
            json.dump(results, f)

    if n_samples > 0:
        for metric in metrics_to_compute:
            for stat in ("mean", "max", "min", "std"):
                metrics_arr[metric][stat] /= n_samples

    metrics_fname = f"final_metrics_shard_{args.shard_id}.json" if args.num_shards > 1 else "final_metrics.json"
    with open(os.path.join(output_dir, metrics_fname), "w") as f:
        json.dump(metrics_arr, f, indent=4)
    print(f"[Shard {args.shard_id} | GPU {actual_gpu_id}] finished prompts {start_idx} -> {end_idx - 1}.")


def get_args():
    parser = argparse.ArgumentParser(description="Phase 1: lookahead sampling with (optionally smoothed) rewards.")
    parser.add_argument("--output_dir", type=str, default="Lookahead_samples")
    parser.add_argument("--save_individual_images", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--num_inference_steps", "--steps", dest="num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--metrics_to_compute", type=str, default="ImageReward#Clip-Score",
                        help="'#'-separated list of metrics")
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt", type=int, default=1000)
    parser.add_argument("--prompt_indices", type=str, default=None,
                        help="Optional subset of prompt indices ('stratified_100', '0-99', '0,5,10' or a JSON file)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of GPU shards to split prompts")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard ID (0 to num_shards-1)")
    parser.add_argument("--gpu_id", type=int, default=None, help="Explicit CUDA device ID to run on (e.g. 0 or 1)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed prompts if results exist")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Force re-run and overwrite existing results")
    parser.add_argument("--use_smoothing", action="store_true", default=False,
                        help="Score lookahead particles with the randomized-smoothing reward (RS-LiDAR)")
    parser.add_argument("--sigma", type=float, default=0.25, help="Pixel-space smoothing noise std sigma")
    parser.add_argument("--num_mc_samples", "--M", dest="num_mc_samples", type=int, default=4,
                        help="Number of Monte Carlo samples M")
    parser.add_argument("--vectorized_smoothing", type=str2bool, nargs="?", const=True, default=False,
                        help="Score all M*N noisy images in one batch instead of looping over M (more memory)")
    parser.add_argument("--loop_smoothing", type=str2bool, nargs="?", const=True, default=None,
                        help="Explicitly enable/disable the loop over M (overrides --vectorized_smoothing)")
    parser.add_argument("--reuse_latents_from", type=str, default=None,
                        help="Run folder (or name under Lookahead_samples) whose latents are reused, e.g. 100_50_5")
    parser.add_argument("--no_reuse_latents", action="store_true", default=False,
                        help="Ignore --reuse_latents_from")
    parser.add_argument("--use_crn", type=str2bool, nargs="?", const=True, default=False,
                        help="Shorthand for --noise_mode coupled_crn")
    parser.add_argument("--noise_mode", type=str, default="independent_iid", choices=NOISE_MODES,
                        help="Monte Carlo noise: 'independent_iid' (default), 'independent_antithetic' (pairs u, -u) "
                             "or 'coupled_crn' (one draw per m shared by all particles)")
    parser.add_argument("--beta_risk", type=float, default=0.0,
                        help="Entropic-risk coefficient; 0 uses the plain Monte Carlo mean")
    parser.add_argument("--pareto_rho", type=float, default=0.5, help="Pareto trade-off weight (with --use_pareto)")
    parser.add_argument("--use_pareto", type=str2bool, nargs="?", const=True, default=False,
                        help="Also store a multi-reward Pareto potential (requires >= 2 metrics)")
    parser.add_argument("--run_name", type=str, default=None, help="Custom output subfolder name")
    args = parser.parse_args()

    if args.use_crn or args.noise_mode == "coupled_crn":
        args.noise_mode = "coupled_crn"

    # FLUX protocol: 3 lookahead particles unless --num_particles is given explicitly.
    if "flux" in args.model_name.lower():
        if not any(arg.startswith("--num_particles") for arg in sys.argv):
            args.num_particles = 3
            print("FLUX protocol: using num_particles = 3 for the Phase 1 lookahead pool.")

    args.M = args.num_mc_samples

    raw_m_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--M=')]
    raw_mc_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--num_mc_samples=')]
    if raw_m_flags and raw_mc_flags and raw_m_flags[0] != raw_mc_flags[0]:
        raise ValueError(f"Conflicting flags: --num_mc_samples={raw_mc_flags[0]} but --M={raw_m_flags[0]}.")

    if args.use_smoothing and args.noise_mode == "independent_antithetic" and args.num_mc_samples % 2 != 0:
        raise ValueError(f"--noise_mode independent_antithetic needs an even number of Monte Carlo samples "
                         f"(pairs u, -u), got M={args.num_mc_samples}.")

    if args.no_reuse_latents:
        args.reuse_latents_from = None
    if args.loop_smoothing is not None:
        args.use_smoothing_loop = args.loop_smoothing
    else:
        args.use_smoothing_loop = not args.vectorized_smoothing

    return args


if __name__ == "__main__":
    main(get_args())
