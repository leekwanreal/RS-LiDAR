"""Phase-2 target-denoising convergence ablation (LiDAR vs. RS-LiDAR).

Runs a single DDIM trajectory per prompt with LiDAR / RS-LiDAR closed-form guidance and
evaluates the predicted clean sample x_0 at intermediate denoising steps
(`--eval_checkpoint_steps`). The guidance and sampling path mirror `LiDAR_sampling.py`;
intermediate predictions are captured by a non-invasive hook on `scheduler.step`.

Usage (from the repository root):
    python -m experiments.target_steps_ablation --lookahead_path <lookahead_dir> --eval_checkpoint_steps 25,30,35,40,45,50
"""
import os
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import argparse
from datetime import datetime
import torch
import sys

from rslidar_utils import load_geneval_metadata, parse_prompt_indices, setup_runtime, str2bool

# Compatibility shims (wandb / protobuf / transformers) must run before diffusers and ImageReward are imported.
setup_runtime()

try:
    from diffusers import DDIMScheduler, UNet2DConditionModel

    sys.path.append("fkd_diffusers")

    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

    from fks_utils import do_eval
    from lookahead_datasets import gen_lookahead_samples
except ImportError:
    DDIMScheduler = UNet2DConditionModel = None
    FKDStableDiffusionXL = FKDStableDiffusion = None
    do_eval = gen_lookahead_samples = None


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


def decode_latents_to_pil(pipe, latents):
    """Decodes latent tensor x_0 (B, 4, H, W) to a list of PIL Images using pipe.vae."""
    with torch.inference_mode():
        lat = latents.clone()
        if lat.dtype != pipe.vae.dtype:
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
            print(f"Assigned process to GPU {actual_gpu_id} (device {actual_gpu_id + 1}/{num_devices}): {torch.cuda.get_device_name(actual_gpu_id)}")
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
            print("Warning: num_particles <= 1, disabling SMC resampling.")
            args.use_smc = False

    # Parse checkpoint steps sweep range
    eval_checkpoint_steps = parse_checkpoint_steps(args.eval_checkpoint_steps, args.num_inference_steps)
    print(f"[Convergence ablation] Denoising steps: {args.num_inference_steps} (eta={args.eta})")
    print(f"[Checkpoint steps] {eval_checkpoint_steps} ({len(eval_checkpoint_steps)} checkpoints)")

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
            raise FileNotFoundError(f"No valid lookahead data found at '{args.lookahead_path}'.")
        first_idx = next(iter(lookaheads.data.keys()))
        first_lat = lookaheads[first_idx]["latents"]
        expected_dim = 128 if ("xl" in args.model_name.lower() or "flux" in args.model_name.lower()) else 64
        if first_lat.shape[-1] != expected_dim:
            raise ValueError(
                f"Latent size mismatch: lookahead dataset '{args.lookahead_path}' has spatial size "
                f"{first_lat.shape[-1]}x{first_lat.shape[-1]}, but model '{args.model_name}' expects "
                f"{expected_dim}x{expected_dim}."
            )

    total_prompts = len(prompt_data)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        print(f"Shard {args.shard_id + 1}/{args.num_shards}: Processing prompts {start_idx} to {end_idx - 1} (Total: {end_idx - start_idx})")
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
            use_rag=args.use_rag,
            use_smc_smoothing=getattr(args, "use_smc_smoothing", False),
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

        # For the terminal step use the pipeline's final images (decoded from prev_sample)
        # so results match LiDAR_sampling.py exactly.
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

                # Build the grid with PIL (much cheaper than matplotlib).
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

            # Per-step final_metrics.json (same format as LiDAR_sampling.py outputs)
            with open(os.path.join(step_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(step_final_metrics, f, indent=4)

            convergence_summary.append(row_record)

        # Save root convergence_summary.json and .csv
        with open(os.path.join(output_dir, "convergence_summary.json"), "w", encoding="utf-8") as f:
            json.dump(convergence_summary, f, indent=4)

        df_summary = pd.DataFrame(convergence_summary)
        df_summary.to_csv(os.path.join(output_dir, "convergence_summary.csv"), index=False)

        print("\n" + "=" * 70)
        print("[Phase-2 convergence summary across checkpoint steps]")
        print("=" * 70)
        print(df_summary.to_string(index=False))
        print("=" * 70)
        print(f"Saved convergence summary to {os.path.join(output_dir, 'convergence_summary.csv')}")


def get_args():
    parser = argparse.ArgumentParser(description="Phase 2 Convergence Ablation for LiDAR vs RS-LiDAR")
    parser.add_argument("--output_dir", type=str, default="Target_samples")
    parser.add_argument("--save_individual_images", action="store_true", default=True, help="Save individual PNGs per step for GenEval and visualization")
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Total DDIM inference steps (default: 50)")
    parser.add_argument("--eval_checkpoint_steps", type=str, default="25,30,35,40,45,50", help="Comma-separated intermediate steps to evaluate (e.g. '25,30,35,40,45,50')")
    parser.add_argument("--use_smc", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_rag", action="store_true", default=True)

    ## RS-SMC arguments
    parser.add_argument("--use_smc_smoothing", type=str2bool, nargs="?", const=True, default=False, help="Enable Randomized Smoothing for SMC resampling")
    parser.add_argument("--use_rs_smc", type=str2bool, nargs="?", const=True, default=False, help="Alias for --use_smc_smoothing")
    parser.add_argument("--smc_sigma", type=float, default=0.25, help="Gaussian noise standard deviation for RS-SMC (default: 0.25)")
    parser.add_argument("--smc_num_mc_samples", "--smc_M", dest="smc_num_mc_samples", type=int, default=4, help="Number of Monte Carlo samples M for RS-SMC (default: 4)")
    parser.add_argument("--M", dest="M", type=int, default=None, help="Alias for number of MC samples M")
    parser.add_argument("--noise_mode", type=str, default="independent_antithetic")
    parser.add_argument("--smc_smoothing_domain", type=str, default="image", choices=["image"])
    parser.add_argument("--smc_use_crn", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--smc_smoothing_mode", type=str, default="loop", choices=["loop", "vectorized"])
    parser.add_argument("--smc_anneal_sigma", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--smc_anneal_schedule", type=str, default="tweedie_analytical")
    parser.add_argument("--smc_sigma_min", type=float, default=None)
    parser.add_argument("--smc_terminal_resample", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--terminal_resample", type=str2bool, nargs="?", const=True, default=None)

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lmbda", type=float, default=5000)
    parser.add_argument("--scale", type=float, default=12.5)

    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (default 0.0 for deterministic ODE)")
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#Clip-Score#Clip-Diversity#HumanPreference#AS",
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

    # Reject conflicting Monte Carlo sample-count flags.
    raw_m_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--M=') or a.startswith('--smc_M=')]
    raw_mc_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--smc_num_mc_samples=')]
    if raw_m_flags and raw_mc_flags and raw_m_flags[0] != raw_mc_flags[0]:
        raise ValueError(f"Conflicting flags: --smc_num_mc_samples={raw_mc_flags[0]} vs --M={raw_m_flags[0]}")

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
            print("Using scale 8.0 for SDXL")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
