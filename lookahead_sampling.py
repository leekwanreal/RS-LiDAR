# primary generation script
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import torch
import sys
import types
import importlib.machinery
if "wandb" not in sys.modules:
    try:
        import wandb
    except Exception:
        _mock_wandb = types.ModuleType("wandb")
        _mock_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        _mock_wandb.__file__ = "wandb"
        sys.modules["wandb"] = _mock_wandb

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
except Exception:
    pass

try:
    from diffusers import (
        DDIMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler,
        StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline,
        LCMScheduler, AutoPipelineForText2Image, EulerDiscreteScheduler, FluxPipeline
    )
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except ImportError:
    DDIMScheduler = UNet2DConditionModel = DPMSolverMultistepScheduler = None
    StableDiffusionPipeline = StableDiffusionXLPipeline = DiffusionPipeline = None
    LCMScheduler = AutoPipelineForText2Image = EulerDiscreteScheduler = FluxPipeline = None
    hf_hub_download = load_file = None


import sys
sys.path.append("fkd_diffusers")


try:
    from fks_utils import do_eval
except ImportError:
    do_eval = None
from typing import Dict, List, Optional, Tuple

@torch.no_grad()
def generate_smoothed_pixel_batch(
    images_0: torch.Tensor,
    sigma: float,
    M: int = 4,
    noise_mode: str = "independent_antithetic",
    clamp_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
    **kwargs
) -> torch.Tensor:
    """
    Tạo batch nhiễu Monte Carlo tuân thủ tính đối xứng Antithetic (Theorem 5.1),
    IID độc lập, hoặc Coupled CRN (Common Random Numbers).
    """
    N, C, H, W = images_0.shape
    device, dtype = images_0.device, images_0.dtype

    if sigma <= 0.0 or M <= 0:
        return images_0.unsqueeze(0).repeat(max(1, M), 1, 1, 1, 1)

    if noise_mode == "stratified_harmonized":
        assert M >= 4 and M % 2 == 0, f"M must be even and >= 4 for stratified_harmonized, got M={M}."
        sigma_hi = kwargs.get("sigma_high", 1.0) if kwargs else 1.0
        sigma_lo = kwargs.get("sigma_low", 0.25) if kwargs else 0.25
        u_hi = torch.randn((1, N, C, H, W), device=device, dtype=dtype)
        p_hi = torch.cat([u_hi, -u_hi], dim=0) * sigma_hi
        u_lo = torch.randn((1, N, C, H, W), device=device, dtype=dtype)
        p_lo = torch.cat([u_lo, -u_lo], dim=0) * sigma_lo
        pert = torch.cat([p_hi, p_lo], dim=0)
        if M > 4:
            extra_half = (M - 4) // 2
            u_ex = torch.randn((extra_half, N, C, H, W), device=device, dtype=dtype)
            p_ex = torch.cat([u_ex, -u_ex], dim=0) * sigma_lo
            pert = torch.cat([pert, p_ex], dim=0)
        perturbed = images_0.unsqueeze(0) + pert
        del u_hi, u_lo, pert
        if clamp_range is not None:
            perturbed = torch.clamp(perturbed, clamp_range[0], clamp_range[1])
        return perturbed
    elif noise_mode == "independent_antithetic":
        assert M % 2 == 0, f"M phải là số chẵn cho chế độ antithetic, nhận M={M}."
        half_M = M // 2
        u_half = torch.randn((half_M, N, C, H, W), device=device, dtype=dtype)
        u = torch.cat([u_half, -u_half], dim=0)
    elif noise_mode == "independent_iid":
        u = torch.randn((M, N, C, H, W), device=device, dtype=dtype)
    elif noise_mode == "coupled_crn":
        # Chế độ cũ: cùng 1 nhiễu cho toàn bộ N hạt
        u = torch.randn((M, 1, C, H, W), device=device, dtype=dtype).expand(M, N, C, H, W)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}. Expected 'independent_antithetic', 'independent_iid', 'coupled_crn', or 'stratified_harmonized'.")

    # In-place broadcast addition to save VRAM, explicitly delete u
    perturbed = images_0.unsqueeze(0) + sigma * u
    if clamp_range is not None:
        perturbed = torch.clamp(perturbed, clamp_range[0], clamp_range[1])
    del u
    return perturbed


@torch.no_grad()
def compute_entropic_antithetic_rewards(
    rewards_mc: torch.Tensor,
    sigma: float = 1.0,
    beta_risk: float = 0.5,
    noise_mode: str = "independent_antithetic",
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Phân rã Half-Sum / Half-Difference (Theorem 5.1) và tính Donsker-Varadhan Entropic Risk (Theorem 5.2).
    """
    # Ép kiểu FP32 để chống tràn số / sai số
    r_fp32 = rewards_mc.to(torch.float32)
    M, N = r_fp32.shape

    if noise_mode == "stratified_harmonized" and M >= 4:
        # Pair 1 (high): [0, 1], Pair 2 (low): [2, 3]
        r_hi_pos, r_hi_neg = r_fp32[0], r_fp32[1]
        mean_hi = 0.5 * (r_hi_pos + r_hi_neg)
        diff_hi = 0.5 * (r_hi_pos - r_hi_neg)
        grad_energy_hi = diff_hi ** 2
        r_hi_corr = mean_hi - (beta_risk / 4.0) * grad_energy_hi

        r_lo_pos, r_lo_neg = r_fp32[2], r_fp32[3]
        mean_lo = 0.5 * (r_lo_pos + r_lo_neg)
        diff_lo = 0.5 * (r_lo_pos - r_lo_neg)
        grad_energy_lo = diff_lo ** 2
        r_lo_corr = mean_lo - (beta_risk / 4.0) * grad_energy_lo

        r_corrected = 0.5 * (r_hi_corr + r_lo_corr)
        grad_energy_sq = 0.5 * (grad_energy_hi + grad_energy_lo)
        pair_var = 0.5 * (grad_energy_hi + grad_energy_lo)
        return {
            "r_corrected": r_corrected.to(rewards_mc.dtype),
            "grad_energy_sq": grad_energy_sq.to(rewards_mc.dtype),
            "pair_variance": pair_var.to(rewards_mc.dtype),
            "r_high": r_hi_corr.to(rewards_mc.dtype),
            "r_low": r_lo_corr.to(rewards_mc.dtype),
        }

    if noise_mode != "independent_antithetic" or M % 2 != 0:
        # Fallback chuẩn nếu không phải antithetic hoặc M lẻ
        r_mean = r_fp32.mean(dim=0)
        variance = r_fp32.var(dim=0, unbiased=True) if M > 1 else torch.zeros_like(r_mean)
        r_corrected = r_mean - (beta_risk / (2.0 * float(M))) * variance
        grad_energy_sq = variance / (sigma ** 2 + 1e-8)
        return {
            "r_corrected": r_corrected.to(rewards_mc.dtype),
            "grad_energy_sq": grad_energy_sq.to(rewards_mc.dtype),
            "pair_variance": variance.to(rewards_mc.dtype)
        }

    half_M = M // 2
    r_pos = r_fp32[:half_M]
    r_neg = r_fp32[half_M:]

    # Định lý 5.1: Phân rã Half-Sum và Half-Difference
    pair_sums = 0.5 * (r_pos + r_neg)
    pair_diffs = 0.5 * (r_pos - r_neg)

    grad_energy_sq = torch.mean(pair_diffs ** 2, dim=0) / (sigma ** 2 + 1e-8)
    curv_var = pair_sums.var(dim=0, unbiased=True) if half_M >= 2 else torch.zeros_like(grad_energy_sq)

    # Định lý 5.2: Phạt Donsker-Varadhan Entropic Risk
    total_var = (sigma ** 2) * grad_energy_sq + curv_var
    r_mean = r_fp32.mean(dim=0)  # Tương đương pair_sums.mean(dim=0)
    r_corrected = r_mean - (beta_risk / (2.0 * float(M))) * total_var

    return {
        "r_corrected": r_corrected.to(rewards_mc.dtype),
        "grad_energy_sq": grad_energy_sq.to(rewards_mc.dtype),
        "pair_variance": total_var.to(rewards_mc.dtype)
    }


def parse_prompt_indices(indices_arg):
    if not indices_arg:
        return None
    if isinstance(indices_arg, (list, tuple)):
        return list(indices_arg)
    indices_str = str(indices_arg).strip()
    if indices_str.lower() in ["stratified_100", "100_representative"]:
        for repr_path in ["prompt_files/geneval_100_representative.json", "Diffusion-LiDAR-Sampling/prompt_files/geneval_100_representative.json"]:
            if os.path.exists(repr_path):
                with open(repr_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return [0, 5, 11, 16, 21, 26, 32, 37, 42, 47, 53, 58, 63, 68, 74, 79, 80, 86, 92, 99, 105, 111, 117, 123, 130, 136, 142, 148, 154, 161, 167, 173, 178, 179, 184, 190, 195, 200, 205, 211, 216, 221, 226, 232, 237, 242, 247, 253, 258, 259, 265, 271, 277, 282, 288, 294, 300, 306, 312, 318, 323, 329, 335, 341, 347, 352, 353, 359, 365, 372, 378, 384, 390, 396, 403, 409, 415, 421, 428, 434, 440, 446, 452, 453, 459, 465, 472, 478, 484, 490, 496, 503, 509, 515, 521, 527, 533, 540, 546, 552]
    if os.path.exists(indices_str):
        with open(indices_str, "r", encoding="utf-8") as f:
            return json.load(f)
    if "-" in indices_str and "," not in indices_str:
        parts = indices_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]

# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None, prompt_indices=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    for idx, item in enumerate(data):
        if "original_idx" not in item:
            item["original_idx"] = idx
    if prompt_indices is not None and len(prompt_indices) > 0:
        data = [data[i] for i in prompt_indices if i < len(data)]
    elif max_prompts is not None:
        data = data[:max_prompts]
    return data




def main(args):
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if args.gpu_id is not None:
            actual_gpu_id = args.gpu_id if args.gpu_id < num_devices else (args.gpu_id % num_devices)
            torch.cuda.set_device(actual_gpu_id)
            device = f"cuda:{actual_gpu_id}"
            print(f"🎯 Assigned process to GPU {actual_gpu_id} (device {actual_gpu_id + 1}/{num_devices}): {torch.cuda.get_device_name(actual_gpu_id)}")
        else:
            actual_gpu_id = 0
            device = "cuda:0"
            torch.cuda.set_device(0)
    else:
        actual_gpu_id = "cpu"
        device = "cpu"

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load prompt data
    parsed_indices = parse_prompt_indices(getattr(args, "prompt_indices", None))
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt, prompt_indices=parsed_indices)

    if "runwayml/stable-diffusion-v1-5" in args.model_name: ## SDv1.5-DPM
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif "mhdang/dpo" in args.model_name and "xl" not in args.model_name: ## SDv1.5-DPO-DPM
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # load finetuned model
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif "stabilityai/stable-diffusion-xl-base-1.0" in args.model_name: ## SDXL-DPM
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,use_safetensors=True, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.vae.to(dtype=torch.float32)

    elif "dmd2_sdxl_4step_lora_fp16.safetensors" in args.model_name: ## SDXL-DMD-4
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_lora_fp16.safetensors"
        pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.vae.to(dtype=torch.float32)

    elif "dmd2_sdxl_1step_unet_fp16.bin" in args.model_name: ## SDXL-DMD-1
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device))
        pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.to(dtype=torch.float32)

    elif "latent-consistency/lcm-lora-sdxl" in args.model_name: ## SDXL-LCM-LoRA-4
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter_id = "latent-consistency/lcm-lora-sdxl"
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()
        pipe.vae.to(dtype=torch.float32)

    elif "sdxl_lightning_4step_lora.safetensors" in args.model_name: ## SDXL-Lightning-4
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_lora.safetensors"
        pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.fuse_lora()
        pipe.vae.to(dtype=torch.float32)

    elif "Hyper-SDXL-1step-Unet.safetensors" in args.model_name: ## SDXL-Hyper-SD-1
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device=device))
        pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16,variant="fp16").to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.to(dtype=torch.float32)

    elif "Hyper-FLUX.1-dev-8steps-lora.safetensors" in args.model_name: ## FLUX-Hyper-8
        base_model_id = "black-forest-labs/FLUX.1-dev"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        pipe = FluxPipeline.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN", True))
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to(device, dtype=torch.float16)

    # set device
    if torch.cuda.is_available():
        if getattr(args, "gpu_id", None) is not None:
            actual_gpu = args.gpu_id if args.gpu_id < torch.cuda.device_count() else (args.gpu_id % torch.cuda.device_count())
            torch.cuda.set_device(actual_gpu)
            device = f"cuda:{actual_gpu}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # set output directory
    prefix = f"{args.seed}_{args.num_particles}_{args.num_inference_steps}"
    if getattr(args, "use_smoothing", False):
        if getattr(args, "noise_mode", "independent_antithetic") == "coupled_crn" or getattr(args, "use_crn", False):
            suffix = "_crn"
        elif getattr(args, "noise_mode", "") == "independent_iid":
            suffix = "_iid"
        else:
            beta_str = f"_beta{args.beta_risk}" if getattr(args, "beta_risk", 0.0) != 0.0 else ""
            suffix = f"_antithetic{beta_str}"
        prefix += f"_rs_sig{args.sigma}_M{args.num_mc_samples}{suffix}"
    if getattr(args, "run_name", None):
        output_dir = os.path.join(args.output_dir, args.run_name)
    else:
        output_dir = os.path.join(args.output_dir, f"{prefix}")

    os.makedirs(output_dir, exist_ok=True)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#")

    # cache metric fns
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    metrics_arr = {
        metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute
    }
    n_samples = 0
    average_time = 0

    total_prompts = len(prompt_data)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        print(f"🚀 Shard {args.shard_id + 1}/{args.num_shards}: Processing prompts {start_idx} to {end_idx - 1} (Total: {end_idx - start_idx})")
    else:
        start_idx = 0
        end_idx = total_prompts

    is_xl = "xl" in args.model_name.lower() or "flux" in args.model_name.lower()
    expected_latent_dim = 128 if is_xl else 64

    for prompt_idx in tqdm(range(start_idx, end_idx), desc=f"Shard-{args.shard_id}"):
        item = prompt_data[prompt_idx]
        real_idx = item.get("original_idx", prompt_idx)
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{real_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        results_file = os.path.join(prompt_path, "results.json")
        latent_file = os.path.join(prompt_path, "samples", "latent.pt")
        if args.resume and not getattr(args, "overwrite", False) and os.path.exists(results_file) and os.path.exists(latent_file):
            try:
                lat_valid = True
                lat_check = torch.load(latent_file, map_location="cpu")
                check_t = lat_check[0] if isinstance(lat_check, (list, tuple)) else lat_check
                if hasattr(check_t, "shape") and check_t.shape[-1] != expected_latent_dim:
                    lat_valid = False
                    print(f"⚠️ [Prompt {real_idx:05d}] latent.pt có kích thước {check_t.shape[-1]}x{check_t.shape[-1]} != {expected_latent_dim}x{expected_latent_dim} ({args.model_name}). Bỏ qua cache hỏng và tạo lại!")

                if lat_valid:
                    with open(results_file, "r") as f:
                        res_cached = json.load(f)
                    for metric in metrics_to_compute:
                        if metric in res_cached:
                            metrics_arr[metric]["mean"] += res_cached[metric]["mean"]
                            metrics_arr[metric]["max"] += res_cached[metric]["max"]
                            metrics_arr[metric]["min"] += res_cached[metric]["min"]
                            metrics_arr[metric]["std"] += res_cached[metric]["std"]
                    n_samples += 1
                    print(f"⏩ [Shard {args.shard_id} | GPU {actual_gpu_id}] Prompt #{real_idx:05d}: Đã có sẵn kết quả hợp lệ, tự động bỏ qua (Resume)!")
                    continue
            except Exception as e:
                print(f"Error loading cached result for prompt {real_idx}: {e}")

        # dump metadata
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        # Kiểm tra xem có thể tái sử dụng latent.pt đã chạy từ trước hay không
        reused_latent_file = None
        cached_reused_latents = None
        if getattr(args, "reuse_latents_from", None):
            candidates = [
                os.path.join(args.reuse_latents_from, f"{real_idx:0>5}", "samples", "latent.pt"),
                os.path.join("Lookahead_samples", args.reuse_latents_from, f"{real_idx:0>5}", "samples", "latent.pt"),
                os.path.join(args.output_dir, args.reuse_latents_from, f"{real_idx:0>5}", "samples", "latent.pt"),
            ]
            if not is_xl:
                candidates.append(
                    os.path.join("Lookahead_samples", f"{args.seed}_{args.num_particles}_{args.num_inference_steps}", f"{real_idx:0>5}", "samples", "latent.pt")
                )
                if args.num_inference_steps == 5:
                    if args.num_particles <= 50:
                        candidates.append(
                            os.path.join("Lookahead_samples", "100_50_5", f"{real_idx:0>5}", "samples", "latent.pt")
                        )
                    candidates.append(
                        os.path.join("Lookahead_samples", f"Lookahead_SD15_DPM5_n{args.num_particles}_seed{args.seed}", f"{real_idx:0>5}", "samples", "latent.pt")
                    )
            for c in candidates:
                if os.path.exists(c):
                    try:
                        test_lat = torch.load(c, map_location="cpu")
                        test_t = test_lat[0] if isinstance(test_lat, (list, tuple)) else test_lat
                        n_avail = len(test_lat) if isinstance(test_lat, (list, tuple)) else (test_lat.shape[0] if hasattr(test_lat, "shape") else 0)
                        if hasattr(test_t, "shape") and test_t.shape[-1] == expected_latent_dim and n_avail >= args.num_particles:
                            reused_latent_file = c
                            cached_reused_latents = test_lat
                            break
                    except Exception:
                        continue

        start_time = datetime.now()
        with torch.inference_mode():
            if reused_latent_file is not None:
                if prompt_idx == start_idx or prompt_idx % 50 == 0:
                    print(f"⚡ [Prompt {real_idx:05d}] Tái sử dụng latents có sẵn: {reused_latent_file}")
                if cached_reused_latents is not None:
                    loaded_latents = cached_reused_latents
                else:
                    loaded_latents = torch.load(reused_latent_file, map_location=device)
                if isinstance(loaded_latents, (list, tuple)):
                    latents = torch.stack([x.to(device) for x in loaded_latents[:args.num_particles]])
                else:
                    latents = loaded_latents[:args.num_particles].to(device)
                cached_reused_latents = None
            else:
                if "dmd2_sdxl_4step_lora_fp16.safetensors" in args.model_name:
                    latents = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, timesteps=[999, 749, 499, 249], output_type="latent").images
                elif "dmd2_sdxl_1step_unet_fp16.bin" in args.model_name:
                    latents = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399], output_type="latent").images
                elif "latent-consistency/lcm-lora-sdxl" in args.model_name:
                    latents = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, output_type="latent").images
                elif "sdxl_lightning_4step_lora.safetensors" in args.model_name:
                    latents = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, output_type="latent").images
                elif "Hyper-SDXL-1step-Unet.safetensors" in args.model_name:
                    latents = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[800], output_type="latent").images
                elif "Hyper-FLUX.1-dev-8steps-lora.safetensors" in args.model_name:
                    latents = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=3.5, output_type="latent").images
                else: ## runwayml/stable-diffusion-v1-5 // mhdang/dpo // stabilityai/stable-diffusion-xl-base-1.0
                    latents = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=args.num_inference_steps, output_type="latent").images

            if "xl" in args.model_name or "XL" in args.model_name:
                latents = latents.to(torch.float32)

            if "FLUX" in args.model_name:
                latents_unpacked = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
                latents_unpacked = (latents_unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                vae_batch_size = 1
                decoded_chunks = []
                for v_i in range(0, latents_unpacked.shape[0], vae_batch_size):
                    chunk = latents_unpacked[v_i : v_i + vae_batch_size]
                    decoded_chunks.append(pipe.vae.decode(chunk, return_dict=False)[0])
                decoded_tensor = torch.cat(decoded_chunks, dim=0)
                if args.save_individual_images or not getattr(args, "use_smoothing", False):
                    clean_images = pipe.image_processor.postprocess(decoded_tensor, output_type="pil")
                else:
                    clean_images = None
            else:
                scaled_latents = latents / pipe.vae.config.scaling_factor
                decoded_tensor = pipe.vae.decode(scaled_latents, return_dict=False)[0]
                if args.save_individual_images or not getattr(args, "use_smoothing", False):
                    clean_images = pipe.image_processor.postprocess(decoded_tensor, output_type="pil")
                else:
                    clean_images = None

        end_time = datetime.now()
        time_taken = end_time - start_time

        if getattr(args, "use_smoothing", False):
            # Randomized Smoothing: Monte Carlo expectation E[R(x + eps)] over M noisy samples
            use_loop = getattr(args, "use_smoothing_loop", True)

            if use_loop:
                mc_evals = {metric: [] for metric in metrics_to_compute}
                with torch.inference_mode():
                    half_M = args.num_mc_samples // 2
                    is_antithetic = (getattr(args, "noise_mode", "independent_antithetic") == "independent_antithetic" and args.num_mc_samples % 2 == 0)
                    if is_antithetic:
                        if args.smoothing_domain == "latent" and "FLUX" not in args.model_name:
                            u_half_list = [torch.randn_like(scaled_latents) for _ in range(half_M)]
                        else:
                            u_half_list = [torch.randn_like(decoded_tensor) for _ in range(half_M)]
                    else:
                        u_half_list = None

                    for m_idx in range(args.num_mc_samples):
                        if args.smoothing_domain == "latent" and "FLUX" not in args.model_name:
                            if getattr(args, "noise_mode", "") == "coupled_crn" or getattr(args, "use_crn", False):
                                noise = torch.randn((1, *scaled_latents.shape[1:]), device=scaled_latents.device, dtype=scaled_latents.dtype) * args.sigma
                            elif is_antithetic and u_half_list is not None:
                                u_m = u_half_list[m_idx] if m_idx < half_M else -u_half_list[m_idx - half_M]
                                noise = u_m * args.sigma
                            else:
                                noise = torch.randn_like(scaled_latents) * args.sigma
                            noisy_lat = scaled_latents + noise
                            noisy_t = pipe.vae.decode(noisy_lat, return_dict=False)[0].clamp(-1.0, 1.0)
                        else:
                            if args.sigma > 0:
                                if getattr(args, "noise_mode", "") == "coupled_crn" or getattr(args, "use_crn", False):
                                    noise = torch.randn((1, *decoded_tensor.shape[1:]), device=decoded_tensor.device, dtype=decoded_tensor.dtype) * args.sigma
                                elif is_antithetic and u_half_list is not None:
                                    u_m = u_half_list[m_idx] if m_idx < half_M else -u_half_list[m_idx - half_M]
                                    noise = u_m * args.sigma
                                else:
                                    noise = torch.randn_like(decoded_tensor) * args.sigma
                                noisy_t = (decoded_tensor + noise).clamp(-1.0, 1.0)
                            else:
                                noisy_t = decoded_tensor.clamp(-1.0, 1.0)
                        noisy_pil = pipe.image_processor.postprocess(noisy_t, output_type="pil")
                        eval_prompt = [prompt[0]] * len(noisy_pil)
                        eval_m = do_eval(prompt=eval_prompt, images=noisy_pil, metrics_to_compute=metrics_to_compute)
                        for metric in metrics_to_compute:
                            mc_evals[metric].append(eval_m[metric]["result"])

                    if u_half_list is not None:
                        del u_half_list

                results = {}
                multi_rewards_mc = {}
                for metric in metrics_to_compute:
                    scores_mat = torch.tensor(mc_evals[metric], dtype=torch.float32)
                    multi_rewards_mc[metric] = scores_mat
                    if getattr(args, "noise_mode", "independent_antithetic") == "coupled_crn" and getattr(args, "beta_risk", 0.0) == 0.0:
                        smoothed_scores = scores_mat.mean(dim=0).tolist()
                        scores_arr = torch.tensor(smoothed_scores, dtype=torch.float32)
                        results[metric] = {
                            "result": smoothed_scores,
                            "mean": scores_arr.mean().item(),
                            "std": scores_arr.std().item() if len(smoothed_scores) > 1 else 0.0,
                            "max": scores_arr.max().item(),
                            "min": scores_arr.min().item(),
                        }
                    else:
                        entropic_res = compute_entropic_antithetic_rewards(
                            scores_mat,
                            sigma=args.sigma,
                            beta_risk=getattr(args, "beta_risk", 0.5),
                            noise_mode=getattr(args, "noise_mode", "independent_antithetic"),
                            sigma_high=getattr(args, "sigma_high", 1.0),
                            sigma_low=getattr(args, "sigma_low", 0.25)
                        )
                        smoothed_scores = entropic_res["r_corrected"].tolist()
                        scores_arr = torch.tensor(smoothed_scores, dtype=torch.float32)
                        results[metric] = {
                            "result": smoothed_scores,
                            "mean": scores_arr.mean().item(),
                            "std": scores_arr.std().item() if len(smoothed_scores) > 1 else 0.0,
                            "max": scores_arr.max().item(),
                            "min": scores_arr.min().item(),
                            "grad_energy_sq": entropic_res["grad_energy_sq"].tolist(),
                            "pair_variance": entropic_res["pair_variance"].tolist(),
                        }
                        if "r_high" in entropic_res and "r_low" in entropic_res:
                            results[metric]["result_high"] = entropic_res["r_high"].tolist()
                            results[metric]["result_low"] = entropic_res["r_low"].tolist()
                results["lookahead_seeds"] = [(int(args.seed) * 1000003 + k * 1009) & 0x7FFFFFFF for k in range(n_particles)]
                if getattr(args, "use_pareto", False) and len(metrics_to_compute) > 1:
                    try:
                        from multi_reward_evaluator import compute_multi_reward_pareto_potential
                        pareto_pot, pareto_diag = compute_multi_reward_pareto_potential(
                            multi_rewards_mc,
                            sigma=args.sigma,
                            pareto_rho=getattr(args, "pareto_rho", 0.5),
                            beta_risk=getattr(args, "beta_risk", 0.5),
                            noise_mode=getattr(args, "noise_mode", "independent_antithetic")
                        )
                        pareto_list = pareto_pot.tolist()
                        results["Pareto"] = {
                            "result": pareto_list,
                            "mean": pareto_pot.mean().item(),
                            "std": pareto_pot.std().item() if len(pareto_list) > 1 else 0.0,
                            "max": pareto_pot.max().item(),
                            "min": pareto_pot.min().item(),
                        }
                        if "Pareto" not in metrics_to_compute:
                            metrics_to_compute = list(metrics_to_compute) + ["Pareto"]
                            metrics_arr["Pareto"] = dict(mean=0, max=0, min=0, std=0)
                    except Exception as e:
                        print(f"Warning: Failed to compute Pareto potential: {e}")
                images = clean_images
            else:
                num_mc = args.num_mc_samples
                n_particles = decoded_tensor.shape[0]
                with torch.inference_mode():
                    if args.smoothing_domain == "latent" and "FLUX" not in args.model_name:
                        if getattr(args, "noise_mode", "") == "coupled_crn" or getattr(args, "use_crn", False):
                            noisy_lat_list = [scaled_latents + torch.randn((1, *scaled_latents.shape[1:]), device=scaled_latents.device, dtype=scaled_latents.dtype) * args.sigma for _ in range(num_mc)]
                            noisy_lat_cat = torch.cat(noisy_lat_list, dim=0)
                        else:
                            perturbed_lat = generate_smoothed_pixel_batch(
                                scaled_latents,
                                sigma=args.sigma,
                                M=num_mc,
                                noise_mode=getattr(args, "noise_mode", "independent_antithetic"),
                                clamp_range=None
                            )
                            noisy_lat_cat = perturbed_lat.view(-1, *scaled_latents.shape[1:])
                        noisy_t_all = pipe.vae.decode(noisy_lat_cat, return_dict=False)[0].clamp(-1.0, 1.0)
                    else:
                        if args.sigma > 0:
                            if getattr(args, "noise_mode", "") == "coupled_crn" or getattr(args, "use_crn", False):
                                noisy_t_list = [(decoded_tensor + torch.randn((1, *decoded_tensor.shape[1:]), device=decoded_tensor.device, dtype=decoded_tensor.dtype) * args.sigma).clamp(-1.0, 1.0) for _ in range(num_mc)]
                                noisy_t_all = torch.cat(noisy_t_list, dim=0)
                            else:
                                perturbed_all = generate_smoothed_pixel_batch(decoded_tensor, sigma=args.sigma, M=num_mc, noise_mode=getattr(args, "noise_mode", "independent_antithetic"), sigma_high=getattr(args, "sigma_high", 1.0), sigma_low=getattr(args, "sigma_low", 0.25))
                                noisy_t_all = perturbed_all.view(-1, *decoded_tensor.shape[1:])
                        else:
                            noisy_t_all = decoded_tensor.clamp(-1.0, 1.0).repeat(num_mc, 1, 1, 1)

                    noisy_pil_all = pipe.image_processor.postprocess(noisy_t_all, output_type="pil")
                    eval_prompt_all = [prompt[0]] * len(noisy_pil_all)
                    eval_all = do_eval(prompt=eval_prompt_all, images=noisy_pil_all, metrics_to_compute=metrics_to_compute)

                results = {}
                multi_rewards_mc = {}
                for metric in metrics_to_compute:
                    raw_scores = eval_all[metric]["result"]
                    scores_mat = torch.tensor(raw_scores, dtype=torch.float32).view(num_mc, n_particles)
                    multi_rewards_mc[metric] = scores_mat
                    if getattr(args, "noise_mode", "independent_antithetic") == "coupled_crn" and getattr(args, "beta_risk", 0.0) == 0.0:
                        smoothed_scores = scores_mat.mean(dim=0).tolist()
                        scores_arr = torch.tensor(smoothed_scores, dtype=torch.float32)
                        results[metric] = {
                            "result": smoothed_scores,
                            "mean": scores_arr.mean().item(),
                            "std": scores_arr.std().item() if len(smoothed_scores) > 1 else 0.0,
                            "max": scores_arr.max().item(),
                            "min": scores_arr.min().item(),
                        }
                    else:
                        entropic_res = compute_entropic_antithetic_rewards(
                            scores_mat,
                            sigma=args.sigma,
                            beta_risk=getattr(args, "beta_risk", 0.5),
                            noise_mode=getattr(args, "noise_mode", "independent_antithetic")
                        )
                        smoothed_scores = entropic_res["r_corrected"].tolist()
                        scores_arr = torch.tensor(smoothed_scores, dtype=torch.float32)
                        results[metric] = {
                            "result": smoothed_scores,
                            "mean": scores_arr.mean().item(),
                            "std": scores_arr.std().item() if len(smoothed_scores) > 1 else 0.0,
                            "max": scores_arr.max().item(),
                            "min": scores_arr.min().item(),
                            "grad_energy_sq": entropic_res["grad_energy_sq"].tolist(),
                            "pair_variance": entropic_res["pair_variance"].tolist(),
                        }
                if getattr(args, "use_pareto", False) and len(metrics_to_compute) > 1:
                    try:
                        from multi_reward_evaluator import compute_multi_reward_pareto_potential
                        pareto_pot, pareto_diag = compute_multi_reward_pareto_potential(
                            multi_rewards_mc,
                            sigma=args.sigma,
                            pareto_rho=getattr(args, "pareto_rho", 0.5),
                            beta_risk=getattr(args, "beta_risk", 0.5),
                            noise_mode=getattr(args, "noise_mode", "independent_antithetic")
                        )
                        pareto_list = pareto_pot.tolist()
                        results["Pareto"] = {
                            "result": pareto_list,
                            "mean": pareto_pot.mean().item(),
                            "std": pareto_pot.std().item() if len(pareto_list) > 1 else 0.0,
                            "max": pareto_pot.max().item(),
                            "min": pareto_pot.min().item(),
                        }
                        if "Pareto" not in metrics_to_compute:
                            metrics_to_compute = list(metrics_to_compute) + ["Pareto"]
                            metrics_arr["Pareto"] = dict(mean=0, max=0, min=0, std=0)
                    except Exception as e:
                        print(f"Warning: Failed to compute Pareto potential: {e}")
                images = clean_images
        else:
            images = clean_images
            eval_prompt = [prompt[0]] * len(images)
            results = do_eval(
                prompt=eval_prompt, images=images, metrics_to_compute=metrics_to_compute
            )

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = real_idx
        if getattr(args, "use_smoothing", False):
            results["smoothing"] = {
                "enabled": True,
                "sigma": args.sigma,
                "num_mc_samples": args.num_mc_samples,
                "domain": args.smoothing_domain,
                "mode": "loop" if use_loop else "vectorized",
                "noise_mode": getattr(args, "noise_mode", "independent_antithetic"),
                "beta_risk": getattr(args, "beta_risk", 0.5),
                "use_crn": getattr(args, "use_crn", False) or (getattr(args, "noise_mode", "") == "coupled_crn"),
                "reused_latents_from": reused_latent_file,
            }

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        # sort images by reward
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        if images is not None:
            images = [images[i] for i in sorted_idx]
        latents = [latents[i] for i in sorted_idx]
        for metric in metrics_to_compute:
            if metric in results and "result" in results[metric]:
                results[metric]["result"] = [
                    results[metric]["result"][i] for i in sorted_idx
                ]
                if "grad_energy_sq" in results[metric]:
                    results[metric]["grad_energy_sq"] = [
                        results[metric]["grad_energy_sq"][i] for i in sorted_idx
                    ]
                if "pair_variance" in results[metric]:
                    results[metric]["pair_variance"] = [
                        results[metric]["pair_variance"][i] for i in sorted_idx
                    ]

        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] += results[metric]["mean"]
            metrics_arr[metric]["max"] += results[metric]["max"]
            metrics_arr[metric]["min"] += results[metric]["min"]
            metrics_arr[metric]["std"] += results[metric]["std"]

        for metric in metrics_to_compute:
            print(
                metric,
                metrics_arr[metric]["mean"] / n_samples,
                metrics_arr[metric]["max"] / n_samples,
            )

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
            image_fpath = os.path.join(prompt_path, f"grid.png")
            plt.savefig(image_fpath)
            plt.close()

        with open(os.path.join(prompt_path, "results.json"), "w") as f:
            json.dump(results, f)

    # save final metrics
    if n_samples > 0:
        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] /= n_samples
            metrics_arr[metric]["max"] /= n_samples
            metrics_arr[metric]["min"] /= n_samples
            metrics_arr[metric]["std"] /= n_samples

    metrics_fname = f"final_metrics_shard_{args.shard_id}.json" if args.num_shards > 1 else "final_metrics.json"
    with open(os.path.join(output_dir, metrics_fname), "w") as f:
        json.dump(metrics_arr, f, indent=4)
    print(f"🏁 [Shard {args.shard_id} | GPU {actual_gpu_id}] Đã hoàn thành toàn bộ shard ({start_idx} -> {end_idx - 1})!")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="Lookahead_samples")
    parser.add_argument("--save_individual_images", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--num_inference_steps", "--steps", dest="num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--metrics_to_compute",type=str,default="ImageReward#Clip-Score",help="# separated list of metrics")
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt", type=int, default=1000)
    parser.add_argument("--prompt_indices", type=str, default=None, help="Optional subset of prompt indices (e.g. 'stratified_100', '0-99', '0,5,10', or path to JSON)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of GPU shards to split prompts")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard ID (0 to num_shards-1)")
    parser.add_argument("--gpu_id", type=int, default=None, help="Explicit CUDA device ID to run on (e.g. 0 or 1)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed prompts if results exist")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Force re-run and overwrite existing results")
    parser.add_argument("--use_smoothing", action="store_true", default=False, help="Enable Randomized Smoothing for Phase 1 lookahead rewards")
    parser.add_argument("--sigma", type=float, default=0.25, help="Gaussian noise standard deviation for Randomized Smoothing (default: 0.25)")
    parser.add_argument("--num_mc_samples", type=int, default=4, help="Number of Monte Carlo samples M to compute expected reward (default: 4)")
    parser.add_argument("--M", type=int, default=4, help="Number of Monte Carlo samples M (default: 4)")
    parser.add_argument("--smoothing_domain", type=str, default="image", choices=["image", "latent"], help="Noise injection domain: 'image' (default) or 'latent'")
    parser.add_argument("--vectorized_smoothing", type=str2bool, nargs="?", const=True, default=False, help="Compute smoothing over all M*N samples in a single batch (True) or loop over M MC iterations (False, default to save VRAM and avoid OOM)")
    parser.add_argument("--loop_smoothing", type=str2bool, nargs="?", const=True, default=None, help="Explicitly enable/disable loop-based smoothing over M iterations (default: True, overrides --vectorized_smoothing if set)")
    parser.add_argument("--reuse_latents_from", type=str, default=None, help="Path or folder name under Lookahead_samples to reuse previously generated latents (e.g. 100_50_5)")
    parser.add_argument("--no_reuse_latents", action="store_true", default=False, help="Do not reuse existing latents even if available")
    parser.add_argument("--use_crn", type=str2bool, nargs="?", const=True, default=False, help="Use Common Random Numbers (CRN): broadcast a single noise sample across all N particles within each prompt to eliminate inter-particle ranking variance")
    parser.add_argument("--noise_mode", type=str, default="independent_antithetic", choices=["independent_antithetic", "independent_iid", "coupled_crn", "stratified_harmonized"], help="Monte Carlo noise sampling mode: 'independent_antithetic' (Theorem 5.1), 'independent_iid', or 'coupled_crn' (legacy)")
    parser.add_argument("--beta_risk", type=float, default=0.5, help="Entropic risk penalty parameter beta (Theorem 5.2, default: 0.5; 0.0 disables risk penalty)")
    parser.add_argument("--sigma_high", type=float, default=1.0, help="Upper noise standard deviation limit (default: 1.0)")
    parser.add_argument("--sigma_low", type=float, default=0.25, help="Lower noise standard deviation limit (default: 0.25)")
    parser.add_argument("--eta_drift", type=float, default=0.4, help="Fraction of lookahead drift absorbed in path measure (default: 0.4)")
    parser.add_argument("--alpha_couple", type=float, default=0.7, help="Coupling strength coefficient (default: 0.7)")
    parser.add_argument("--s_flow", type=float, default=12.5, help="Velocity guidance scale for Rectified Flows (default: 12.5)")
    parser.add_argument("--pareto_rho", type=float, default=0.5, help="Pareto trade-off weight parameter (default: 0.5)")
    parser.add_argument("--use_pareto", type=str2bool, nargs="?", const=True, default=False, help="Enable Local-Lipschitz Multi-Reward Pareto Evaluator (Theorem 6)")
    parser.add_argument("--run_name", type=str, default=None, help="Custom output subfolder name")
    args = parser.parse_args()

    if getattr(args, "use_crn", False) or getattr(args, "noise_mode", "") == "coupled_crn":
        args.noise_mode = "coupled_crn"
        if not any(arg.startswith("--beta_risk") for arg in sys.argv):
            args.beta_risk = 0.0

    if args.M is not None:
        args.num_mc_samples = args.M
    if args.no_reuse_latents:
        args.reuse_latents_from = None
    if args.loop_smoothing is not None:
        args.use_smoothing_loop = args.loop_smoothing
    else:
        args.use_smoothing_loop = not args.vectorized_smoothing

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
