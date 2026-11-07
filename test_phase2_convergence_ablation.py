# Phase 2 Target Denoising Convergence Ablation Script
# Measures convergence dynamics of LiDAR vs RS-LiDAR across intermediate steps within a single DDIM trajectory.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import math
import numpy as np
import pandas as pd
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
    from diffusers import DDIMScheduler, UNet2DConditionModel

    import sys
    sys.path.append("fkd_diffusers")

    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

    from fks_utils import do_eval
    from lookahead_datasets import gen_lookahead_samples
except ImportError:
    DDIMScheduler = UNet2DConditionModel = None
    FKDStableDiffusionXL = FKDStableDiffusion = None
    do_eval = gen_lookahead_samples = None

from rs_smc_engine import HarmonizedRSLiDARSMCEngine


def parse_prompt_indices(indices_arg):
    if not indices_arg or str(indices_arg).strip().lower() in ["none", "all", ""]:
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


def parse_checkpoint_steps(steps_arg, total_steps):
    """Parses a comma-separated string or list into a sorted list of unique checkpoint steps."""
    if not steps_arg:
        return [total_steps]
    if isinstance(steps_arg, (list, tuple)):
        raw_steps = [int(s) for s in steps_arg]
    else:
        raw_steps = [int(s.strip()) for s in str(steps_arg).split(",") if s.strip().isdigit()]

    valid_steps = sorted(list(set([s for s in raw_steps if 1 <= s <= total_steps])))
    if total_steps not in valid_steps:
        valid_steps.append(total_steps)
    valid_steps.sort()
    return valid_steps


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


def decode_latents_to_pil(pipe, latents):
    """Decodes latent tensor x_0 (B, 4, H, W) to a list of PIL Images using pipe.vae."""
    with torch.inference_mode():
        needs_upcasting = (
            hasattr(pipe.vae, "config")
            and getattr(pipe.vae.config, "force_upcast", False)
            and pipe.vae.dtype == torch.float16
        )
        lat = latents.clone()
        if needs_upcasting and hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
            lat = lat.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        elif lat.dtype != pipe.vae.dtype:
            lat = lat.to(pipe.vae.dtype)

        scaled = lat / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        do_denormalize = [True] * decoded.shape[0]
        pil_images = pipe.image_processor.postprocess(
            decoded, output_type="pil", do_denormalize=do_denormalize
        )
        return list(pil_images)


def attach_convergence_hook(pipe, eval_checkpoint_steps, total_inference_steps=None):
    """
    Hooks pipe.scheduler.step to capture and decode pred_original_sample (x_0)
    at the specified eval_checkpoint_steps during the single denoising trajectory.
    Completely preserves the underlying diffusion dynamics and guidance updates.
    """
    orig_step_fn = pipe.scheduler.step
    pipe._step_counter = 0
    pipe._checkpoint_predictions = {}
    pipe._eval_checkpoint_steps = set(eval_checkpoint_steps)
    pipe._total_inference_steps = total_inference_steps

    def hooked_step(*step_args, **step_kwargs):
        pipe._step_counter += 1
        current_step = pipe._step_counter

        # Force return_dict=True so step_output has pred_original_sample
        step_kwargs["return_dict"] = True
        step_output = orig_step_fn(*step_args, **step_kwargs)

        if current_step in pipe._eval_checkpoint_steps:
            # For the terminal step, skip redundant VAE decode of pred_original_sample;
            # the pipeline will decode the final latents (prev_sample) into final_images directly.
            if pipe._total_inference_steps is not None and current_step == pipe._total_inference_steps:
                return step_output

            if hasattr(step_output, "pred_original_sample") and step_output.pred_original_sample is not None:
                x0_lat = step_output.pred_original_sample.detach().clone()
            elif isinstance(step_output, (tuple, list)) and len(step_output) > 1 and step_output[1] is not None:
                x0_lat = step_output[1].detach().clone()
            elif hasattr(step_output, "__getitem__") and "pred_original_sample" in step_output:
                x0_lat = step_output["pred_original_sample"].detach().clone()
            else:
                x0_lat = step_output["prev_sample"].detach().clone()

            # Decode intermediate predicted x_0 to PIL images
            pil_imgs = decode_latents_to_pil(pipe, x0_lat)
            pipe._checkpoint_predictions[current_step] = pil_imgs

        return step_output

    pipe.scheduler.step = hooked_step
    return orig_step_fn


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

    # Seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps

    if args.use_smc:
        assert args.resample_frequency > 0
        if args.num_particles <= 1:
            print("⚠️ [CẢNH BÁO] args.num_particles <= 1: Tự động vô hiệu hóa SMC resampling.")
            args.use_smc = False

    # Parse checkpoint steps sweep range
    eval_checkpoint_steps = parse_checkpoint_steps(args.eval_checkpoint_steps, args.num_inference_steps)
    print(f"📈 [CONVERGENCE ABLATION] Denoising Steps: {args.num_inference_steps} (η={args.eta})")
    print(f"🎯 [CHECKPOINT STEPS SWEEP]: {eval_checkpoint_steps} ({len(eval_checkpoint_steps)} checkpoints)")

    # Load prompt data
    parsed_indices = parse_prompt_indices(getattr(args, "prompt_indices", None))
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt, prompt_indices=parsed_indices)

    # Configure pipeline
    if "mhdang/dpo" in args.model_name and "xl" not in args.model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    elif "stabilityai/stable-diffusion-xl-base-1.0" in args.model_name:
        print("Using SDXL")
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
    else:
        pipe = FKDStableDiffusion.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        )

    if "FLUX" not in args.model_name:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Set device & vae slicing
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # Attach non-invasive convergence hook
    attach_convergence_hook(pipe, eval_checkpoint_steps, total_inference_steps=args.num_inference_steps)

    # Output directory setup
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.run_name:
        output_dir = os.path.join(args.output_dir, args.run_name)
    else:
        model_tag = "SDXL" if "xl" in args.model_name.lower() else "SD15"
        method_prefix = "RS" if getattr(args, "use_smc_smoothing", False) or "sig" in getattr(args, "lookahead_path", "") else "Vanilla"
        output_dir = os.path.join(args.output_dir, f"Convergence_{method_prefix}_{model_tag}_steps{args.num_inference_steps}_seed{args.seed}_{cur_time}")

    os.makedirs(output_dir, exist_ok=True)

    # Pre-create subdirectories for each checkpoint step
    for step_k in eval_checkpoint_steps:
        os.makedirs(os.path.join(output_dir, f"step_{step_k}"), exist_ok=True)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#")

    # Cache metric evaluation functions
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    # Track metrics separately for EACH checkpoint step
    metrics_by_step = {
        step_k: {
            metric: dict(mean=0.0, max=0.0, min=0.0, std=0.0)
            for metric in metrics_to_compute
        }
        for step_k in eval_checkpoint_steps
    }
    n_samples = 0
    average_time = 0.0

    # Load lookahead dataset if using RAG / LiDAR guidance
    lookaheads = None
    if args.use_rag:
        lookaheads = gen_lookahead_samples(args.lookahead_path, args.top_k, reward_key=args.guidance_reward_fn)
        if len(lookaheads) == 0:
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"❌ [FAIL-FAST: KHÔNG TÌM THẤY LOOKAHEAD DATASET]\n"
                f"Không tìm thấy dữ liệu lookahead hợp lệ tại: '{args.lookahead_path}'!\n"
                f"{'='*70}"
            )
        first_idx = next(iter(lookaheads.data.keys()))
        first_lat = lookaheads[first_idx]["latents"]
        expected_dim = 128 if ("xl" in args.model_name.lower() or "flux" in args.model_name.lower()) else 64
        if first_lat.shape[-1] != expected_dim:
            raise ValueError(
                f"\n{'='*70}\n"
                f"❌ [LỖI KHÔNG TƯƠNG THÍCH KÍCH THƯỚC LATENT]\n"
                f"Lookahead dataset '{args.lookahead_path}' có spatial dimension là {first_lat.shape[-1]}x{first_lat.shape[-1]},\n"
                f"trong khi mô hình '{args.model_name}' yêu cầu {expected_dim}x{expected_dim}!\n"
                f"{'='*70}"
            )

    total_prompts = len(prompt_data)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        print(f"🚀 Shard {args.shard_id + 1}/{args.num_shards}: Processing prompts {start_idx} to {end_idx - 1} (Total: {end_idx - start_idx})")
    else:
        start_idx = 0
        end_idx = total_prompts

    # Main Denoising Loop
    for prompt_idx in tqdm(range(start_idx, end_idx), desc=f"Shard-{args.shard_id}"):
        item = prompt_data[prompt_idx]
        real_idx = item.get("original_idx", prompt_idx)
        prompt = [item["prompt"]] * args.num_particles

        # Smart Resume Check across ALL checkpoint steps
        all_cached = True
        cached_results_by_step = {}
        if args.resume and not getattr(args, "overwrite", False):
            for step_k in eval_checkpoint_steps:
                step_res_file = os.path.join(output_dir, f"step_{step_k}", f"{real_idx:0>5}", "results.json")
                if os.path.exists(step_res_file):
                    try:
                        with open(step_res_file, "r", encoding="utf-8") as f:
                            cached_results_by_step[step_k] = json.load(f)
                    except Exception:
                        all_cached = False
                        break
                else:
                    all_cached = False
                    break
        else:
            all_cached = False

        if all_cached and len(cached_results_by_step) == len(eval_checkpoint_steps):
            for step_k, res_cached in cached_results_by_step.items():
                for metric in metrics_to_compute:
                    if metric in res_cached:
                        metrics_by_step[step_k][metric]["mean"] += res_cached[metric]["mean"]
                        metrics_by_step[step_k][metric]["max"] += res_cached[metric]["max"]
                        metrics_by_step[step_k][metric]["min"] += res_cached[metric]["min"]
                        metrics_by_step[step_k][metric]["std"] += res_cached[metric]["std"]
            n_samples += 1
            continue

        fkd_args = dict(
            lmbda=args.lmbda,
            scale=args.scale,
            reward_type=args.reward_type,
            num_particles=args.num_particles,
            use_smc=args.use_smc,
            use_grad=args.use_grad,
            use_rag=args.use_rag,
            use_smc_smoothing=getattr(args, "use_smc_smoothing", False),
            use_hybrid_smc=getattr(args, "use_hybrid_smc", False),
            sigma_high=getattr(args, "sigma_high", 1.0),
            sigma_low=getattr(args, "sigma_low", 0.25),
            eta_drift=getattr(args, "eta_drift", 0.4),
            beta_risk=getattr(args, "beta_risk", 0.5),
            alpha_couple=getattr(args, "alpha_couple", 0.7),
            smc_sigma=getattr(args, "smc_sigma", 0.25),
            smc_num_mc_samples=getattr(args, "smc_num_mc_samples", 4),
            smc_smoothing_domain=getattr(args, "smc_smoothing_domain", "image"),
            smc_use_crn=getattr(args, "smc_use_crn", False),
            smc_smoothing_mode=getattr(args, "smc_smoothing_mode", "loop"),
            smc_anneal_sigma=getattr(args, "smc_anneal_sigma", False),
            smc_anneal_schedule=getattr(args, "smc_anneal_schedule", "sqrt_noise"),
            smc_sigma_min=getattr(args, "smc_sigma_min", None),
            smc_terminal_resample=getattr(args, "smc_terminal_resample", True),
            FK_lmbda=args.FK_lmbda,
            FK_resample_t_start=args.FK_resample_t_start,
            FK_resample_t_end=args.FK_resample_t_end,
            top_k=args.top_k,
            rag_dataset=lookaheads,
            adaptive_resampling=args.adaptive_resampling,
            resample_frequency=args.resample_frequency,
            time_steps=args.num_inference_steps,
            resampling_t_start=args.FK_resample_t_start,
            resampling_t_end=args.resample_t_end,
            guidance_reward_fn=args.guidance_reward_fn,
            potential_type=args.potential_type,
        )

        # Reset hook before running prompt
        pipe._step_counter = 0
        pipe._checkpoint_predictions = {}

        start_time = datetime.now()
        raw_final_images = pipe(
            prompt,
            prompt_idx=real_idx,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            fkd_args=fkd_args,
        )
        end_time = datetime.now()
        time_taken = end_time - start_time

        def _extract_pil_images(raw_imgs):
            if hasattr(raw_imgs, "images"):
                return raw_imgs.images
            if isinstance(raw_imgs, (tuple, list)) and len(raw_imgs) > 0:
                first = raw_imgs[0]
                if hasattr(first, "images"):
                    return first.images
                if isinstance(first, (tuple, list)):
                    return _extract_pil_images(first)
                return list(raw_imgs)
            return raw_imgs

        final_images = _extract_pil_images(raw_final_images)

        # Always assign pipeline's final_images (decoded from prev_sample) for terminal step
        # to ensure 100% bitwise & metric alignment with LiDAR_sampling.py
        if args.num_inference_steps in eval_checkpoint_steps:
            pipe._checkpoint_predictions[args.num_inference_steps] = final_images

        eval_prompt = [prompt[0]] * len(final_images) if isinstance(prompt, list) else [prompt] * len(final_images)

        # Evaluate and save predictions across all checkpoint steps
        step_log_strs = []
        for step_k in eval_checkpoint_steps:
            step_prompt_path = os.path.join(output_dir, f"step_{step_k}", f"{real_idx:0>5}")
            os.makedirs(step_prompt_path, exist_ok=True)

            with open(os.path.join(step_prompt_path, "metadata.jsonl"), "w", encoding="utf-8") as f:
                json.dump(item, f)

            imgs_k = pipe._checkpoint_predictions.get(step_k, final_images)

            # Evaluate metrics at step k
            results_k = do_eval(
                prompt=eval_prompt, images=imgs_k, metrics_to_compute=metrics_to_compute
            )
            results_k["step"] = step_k
            results_k["time_taken"] = time_taken.total_seconds()
            results_k["prompt"] = prompt
            results_k["prompt_index"] = real_idx

            # Sort particles by guidance reward at step k
            if args.guidance_reward_fn in results_k:
                guidance_reward = np.array(results_k[args.guidance_reward_fn]["result"])
                sorted_idx = np.argsort(guidance_reward)[::-1]
                imgs_k_sorted = [imgs_k[i] for i in sorted_idx]
                for metric in metrics_to_compute:
                    if metric in results_k and "result" in results_k[metric]:
                        results_k[metric]["result"] = [
                            results_k[metric]["result"][i] for i in sorted_idx
                        ]
            else:
                imgs_k_sorted = imgs_k

            # Save results.json for step k
            with open(os.path.join(step_prompt_path, "results.json"), "w", encoding="utf-8") as f:
                json.dump(results_k, f, indent=4)

            # Save individual images for step k (for GenEval / visualization)
            if args.save_individual_images:
                sample_path = os.path.join(step_prompt_path, "samples")
                os.makedirs(sample_path, exist_ok=True)
                for image_idx, image in enumerate(imgs_k_sorted):
                    image.save(os.path.join(sample_path, f"{image_idx:05}.png"), compress_level=1)

                best_of_n_sample_path = os.path.join(step_prompt_path, "best_of_n_samples")
                os.makedirs(best_of_n_sample_path, exist_ok=True)
                for image_idx, image in enumerate(imgs_k_sorted[:1]):
                    image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"), compress_level=1)

                # Fast PIL grid creation (avoids heavy matplotlib CPU overhead across thousands of steps)
                if imgs_k_sorted:
                    w, h = imgs_k_sorted[0].size
                    grid_img = Image.new("RGB", (w * len(imgs_k_sorted), h))
                    for i_ax, img in enumerate(imgs_k_sorted):
                        grid_img.paste(img, (i_ax * w, 0))
                    grid_img.save(os.path.join(step_prompt_path, "grid.png"), compress_level=1)

            # Accumulate metrics
            for metric in metrics_to_compute:
                if metric in results_k:
                    metrics_by_step[step_k][metric]["mean"] += results_k[metric]["mean"]
                    metrics_by_step[step_k][metric]["max"] += results_k[metric]["max"]
                    metrics_by_step[step_k][metric]["min"] += results_k[metric]["min"]
                    metrics_by_step[step_k][metric]["std"] += results_k[metric]["std"]

            if args.guidance_reward_fn in results_k:
                step_log_strs.append(f"S{step_k}:{results_k[args.guidance_reward_fn]['mean']:.3f}")

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples:.2f}s | Progress {', '.join(step_log_strs)}")

    # Aggregate & Save Final Metrics Across Steps
    if n_samples > 0:
        convergence_summary = []
        for step_k in eval_checkpoint_steps:
            step_dir = os.path.join(output_dir, f"step_{step_k}")
            step_final_metrics = {}
            row_record = {"step": step_k, "n_samples": n_samples}

            for metric in metrics_to_compute:
                step_final_metrics[metric] = {
                    "mean": metrics_by_step[step_k][metric]["mean"] / n_samples,
                    "max": metrics_by_step[step_k][metric]["max"] / n_samples,
                    "min": metrics_by_step[step_k][metric]["min"] / n_samples,
                    "std": metrics_by_step[step_k][metric]["std"] / n_samples,
                }
                row_record[f"{metric}_mean"] = step_final_metrics[metric]["mean"]
                row_record[f"{metric}_max"] = step_final_metrics[metric]["max"]

            # Save per-step final_metrics.json (100% compatible with GenEval summary scripts)
            with open(os.path.join(step_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(step_final_metrics, f, indent=4)

            convergence_summary.append(row_record)

        # Save root convergence_summary.json and .csv
        with open(os.path.join(output_dir, "convergence_summary.json"), "w", encoding="utf-8") as f:
            json.dump(convergence_summary, f, indent=4)

        df_summary = pd.DataFrame(convergence_summary)
        df_summary.to_csv(os.path.join(output_dir, "convergence_summary.csv"), index=False)

        print("\n" + "=" * 70)
        print("📊 [PHASE 2 CONVERGENCE SUMMARY ACROSS CHECKPOINT STEPS]:")
        print("=" * 70)
        print(df_summary.to_string(index=False))
        print("=" * 70)
        print(f"✅ Đã lưu bảng tổng kết hội tụ tại: {os.path.join(output_dir, 'convergence_summary.csv')}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Phase 2 Convergence Ablation for LiDAR vs RS-LiDAR")
    parser.add_argument("--output_dir", type=str, default="Target_samples")
    parser.add_argument("--save_individual_images", action="store_true", default=True, help="Save individual PNGs per step for GenEval and visualization")
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Total DDIM inference steps (default: 50)")
    parser.add_argument("--eval_checkpoint_steps", type=str, default="25,30,35,40,45,50", help="Comma-separated intermediate steps to evaluate (e.g. '25,30,35,40,45,50')")
    parser.add_argument("--use_smc", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_grad", action="store_true")
    parser.add_argument("--use_rag", action="store_true", default=True)

    ## RS-SMC arguments
    parser.add_argument("--use_smc_smoothing", type=str2bool, nargs="?", const=True, default=False, help="Enable Randomized Smoothing for SMC resampling")
    parser.add_argument("--use_rs_smc", type=str2bool, nargs="?", const=True, default=False, help="Alias for --use_smc_smoothing")
    parser.add_argument("--smc_sigma", type=float, default=0.25, help="Gaussian noise standard deviation for RS-SMC (default: 0.25)")
    parser.add_argument("--smc_num_mc_samples", "--smc_M", dest="smc_num_mc_samples", type=int, default=4, help="Number of Monte Carlo samples M for RS-SMC (default: 4)")
    parser.add_argument("--M", dest="M", type=int, default=None, help="Alias for number of MC samples M")
    parser.add_argument("--noise_mode", type=str, default="independent_antithetic")
    parser.add_argument("--beta_risk", type=float, default=0.5)
    parser.add_argument("--sigma_high", type=float, default=1.0)
    parser.add_argument("--sigma_low", type=float, default=0.25)
    parser.add_argument("--eta_drift", type=float, default=0.4)
    parser.add_argument("--alpha_couple", type=float, default=0.7)
    parser.add_argument("--smc_smoothing_domain", type=str, default="image", choices=["image", "latent"])
    parser.add_argument("--smc_use_crn", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--smc_smoothing_mode", type=str, default="loop", choices=["loop", "vectorized"])
    parser.add_argument("--smc_anneal_sigma", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--smc_anneal_schedule", type=str, default="tweedie_analytical")
    parser.add_argument("--smc_sigma_min", type=float, default=None)
    parser.add_argument("--smc_terminal_resample", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--terminal_resample", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--use_hybrid_smc", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lmbda", type=float, default=5000)
    parser.add_argument("--scale", type=float, default=12.5)

    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (default 0.0 for deterministic ODE)")
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#Clip-Score#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=20)
    parser.add_argument("--resample_t_start", type=int, default=1000)
    parser.add_argument("--resample_t_end", type=int, default=200)
    parser.add_argument("--potential_type", type=str, default="diff")

    parser.add_argument("--max_prompt", type=int, default=1000)
    parser.add_argument("--prompt_indices", type=str, default=None, help="Optional subset of prompt indices (e.g. 'stratified_100', '0-99', '0,5,10', or path to JSON)")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--reward_type", type=str, choices=["ImageReward", "Clip-Score-only", "HumanPreference", "AS", "mix"], default="ImageReward")

    parser.add_argument("--lookahead_path", type=str, default="100_50_5")

    ## FK redefine
    parser.add_argument("--FK_lmbda", type=float, default=10.0)
    parser.add_argument("--FK_resample_t_start", type=int, default=20)
    parser.add_argument("--FK_resample_t_end", type=int, default=80)

    parser.add_argument("--run_name", type=str, default="", help="Custom output subfolder name")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed prompts if results exist")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Force re-run and overwrite existing results")
    args = parser.parse_args()

    # Fail-Fast: Check command line flags
    raw_m_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--M=') or a.startswith('--smc_M=')]
    raw_mc_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--smc_num_mc_samples=')]
    if raw_m_flags and raw_mc_flags and raw_m_flags[0] != raw_mc_flags[0]:
        raise ValueError(f"❌ [FAIL-FAST] Xung đột cờ dòng lệnh: --smc_num_mc_samples={raw_mc_flags[0]} vs --M={raw_m_flags[0]}")

    if args.M is not None:
        args.smc_num_mc_samples = args.M
    args.M = args.smc_num_mc_samples
    args.smc_M = args.smc_num_mc_samples
    if args.terminal_resample is not None:
        args.smc_terminal_resample = args.terminal_resample
    if args.use_rs_smc:
        args.use_smc_smoothing = True
    if args.use_smc_smoothing:
        args.use_smc = True

    # Auto-dispatch scale for SDXL vs SD 1.5
    if "xl" in args.model_name.lower():
        if args.scale == 12.5:
            args.scale = 8.0
            print("⚡ Auto-dispatch scale to 8.0 for SDXL")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

