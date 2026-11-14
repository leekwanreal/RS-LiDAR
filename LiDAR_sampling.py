"""Phase 2 of LiDAR / RS-LiDAR: target sampling steered by the Phase 1 lookahead pool.

The lookahead dataset (`lookahead_datasets.gen_lookahead_samples`) holds, per prompt, K lookahead
latents z_{0,k} with their (optionally randomized-smoothing) rewards. Depending on the flags:

- `--use_rag`: closed-form LiDAR guidance. SD1.5 / SDXL add `get_sample_guide` (in
  `fkd_diffusers/fkd_pipeline_sd.py` / `fkd_pipeline_sdxl.py`) to the noise prediction; FLUX adds the
  velocity guidance of `rs_flow_lidar.RSFlowLiDARGuidance` (`fkd_diffusers/fkd_pipeline_flux.py`).
- `--use_smc` / `--use_smc_smoothing`: Feynman-Kac particle resampling (`fkd_diffusers/fkd_class.py`),
  with RS-SMC scoring each Tweedie estimate with the smoothed reward and an optional sigma(t) schedule.
- otherwise: best-of-N / vanilla sampling.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from rslidar_utils import load_geneval_metadata, parse_prompt_indices, setup_runtime, str2bool

setup_runtime()

sys.path.append("fkd_diffusers")

from diffusers import DDIMScheduler, UNet2DConditionModel  # noqa: E402

from fkd_diffusers.fkd_pipeline_flux import FluxPipeline_LiDAR  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion  # noqa: E402
from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL  # noqa: E402
from fks_utils import do_eval  # noqa: E402
from lookahead_datasets import gen_lookahead_samples  # noqa: E402


def _extract_pil_images(raw_imgs):
    """Return the list of PIL images from a pipeline output (object, tuple or list)."""
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


def load_target_pipeline(model_name):
    """Load the Phase 2 target model with the guidance-enabled pipeline class."""
    if "mhdang/dpo" in model_name and "xl" not in model_name:
        pipe = FKDStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe.unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sd1.5-text2image-v1", subfolder="unet", torch_dtype=torch.float16
        )
    elif "stabilityai/stable-diffusion-xl-base-1.0" in model_name:
        print("Using SDXL")
        pipe = FKDStableDiffusionXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                    torch_dtype=torch.float16)
    elif "flux" in model_name.lower():
        print("Using FLUX.1-dev with lookahead velocity guidance")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        pipe = FluxPipeline_LiDAR.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype)
    else:
        pipe = FKDStableDiffusion.from_pretrained(model_name, torch_dtype=torch.float16)
    if "flux" not in model_name.lower():
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def output_folder_name(args):
    if "flux" in args.model_name.lower():
        model_tag = "FLUX"
    elif "xl" in args.model_name.lower():
        model_tag = "SDXL"
    else:
        model_tag = "SD15"

    if args.use_smc:
        crn_suffix = "_crn" if args.smc_use_crn else ""
        anneal_suffix = "_anneal" if args.smc_anneal_sigma else ""
        term_suffix = "" if args.smc_terminal_resample else "_noTerminal"
        if args.use_smc_smoothing:
            method_tag = (f"{model_tag}_RSSMC_N{args.num_particles}_M{args.smc_num_mc_samples}_sigma{args.smc_sigma}"
                          f"{crn_suffix}{anneal_suffix}{term_suffix}_freq{args.resample_frequency}_lmbda{args.FK_lmbda}")
        else:
            method_tag = (f"{model_tag}_SMC_N{args.num_particles}{term_suffix}"
                          f"_freq{args.resample_frequency}_lmbda{args.FK_lmbda}")
        if args.use_rag:
            method_tag += f"_LiDAR_topk{args.top_k}_lmbda{int(args.lmbda)}"
    elif args.use_rag:
        if model_tag == "FLUX":
            method_tag = f"{model_tag}_LiDAR_N{args.num_particles}_topk{args.top_k}_sflow{args.s_flow}"
        else:
            method_tag = f"{model_tag}_LiDAR_N{args.num_particles}_topk{args.top_k}_lmbda{int(args.lmbda)}"
    else:
        method_tag = f"{model_tag}_BoN_N{args.num_particles}" if args.num_particles > 1 else f"{model_tag}_Vanilla_N1"

    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{method_tag}_steps{args.num_inference_steps}_scale{args.scale}_seed{args.seed}_{cur_time}"


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

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps

    is_flux = "flux" in args.model_name.lower()
    if is_flux:
        # FLUX protocol: one target image per prompt unless --num_particles is given explicitly.
        if not any(arg.startswith("--num_particles") for arg in sys.argv):
            args.num_particles = 1
            print("FLUX protocol: using num_particles = 1 for Phase 2.")
        if "--s_flow" not in sys.argv:
            args.s_flow = 40.0  # LiDAR FLUX setting (shared by LiDAR and RS-LiDAR)

    # FLUX guidance window in flow time t (1 = noise, 0 = data).
    if args.guidance_start is None:
        args.guidance_start = 1.0
    if args.guidance_end is None:
        args.guidance_end = 0.95

    if args.use_smc:
        assert args.resample_frequency > 0
        if args.num_particles <= 1:
            print("num_particles <= 1: SMC resampling is disabled (nothing to resample).")
            args.use_smc = False

    parsed_indices = parse_prompt_indices(args.prompt_indices)
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt, prompt_indices=parsed_indices)

    pipe = load_target_pipeline(args.model_name)
    pipe = pipe.to(device)  # FluxPipeline_LiDAR.to also moves every sub-module and pins the execution device
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    output_dir = os.path.join(args.output_dir, args.run_name or output_folder_name(args))
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    metrics_to_compute = args.metrics_to_compute.split("#")

    # Load and warm up the reward models once.
    do_eval(prompt=["test"], images=[Image.new("RGB", (224, 224))], metrics_to_compute=metrics_to_compute)
    metrics_arr = {metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute}
    n_samples = 0
    average_time = 0

    lookaheads = None
    if args.use_rag:
        lookaheads = gen_lookahead_samples(args.lookahead_path, args.top_k, reward_key=args.guidance_reward_fn)
        if len(lookaheads) == 0:
            raise FileNotFoundError(
                f"--use_rag is set but no lookahead data was found at '{args.lookahead_path}'. "
                f"Check the folder name or run Phase 1 (lookahead_sampling.py) first."
            )
        first_idx = next(iter(lookaheads.data.keys()))
        first_lat = lookaheads[first_idx]["latents"]
        expected_dim = 128 if ("xl" in args.model_name.lower() or is_flux) else 64
        if first_lat.shape[-1] != expected_dim:
            raise ValueError(
                f"Lookahead dataset '{args.lookahead_path}' has {first_lat.shape[-1]}x{first_lat.shape[-1]} latents, "
                f"but '{args.model_name}' needs {expected_dim}x{expected_dim}. Phase 1 was run with a different "
                f"model; re-run it with --overwrite for this model."
            )

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

    for prompt_idx in tqdm(range(start_idx, end_idx), desc=f"Shard-{args.shard_id}"):
        item = prompt_data[prompt_idx]
        real_idx = item.get("original_idx", prompt_idx)
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{real_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        results_file = os.path.join(prompt_path, "results.json")
        if args.resume and not args.overwrite and os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    res_cached = json.load(f)
                for metric in metrics_to_compute:
                    if metric in res_cached:
                        for stat in ("mean", "max", "min", "std"):
                            metrics_arr[metric][stat] += res_cached[metric][stat]
                n_samples += 1
                continue
            except Exception as e:
                print(f"Error loading cached result for prompt {real_idx}: {e}")

        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        fkd_args = dict(
            lmbda=args.lmbda,
            scale=args.scale,
            reward_type=args.reward_type,
            num_particles=args.num_particles,
            use_smc=args.use_smc,
            use_rag=args.use_rag,
            use_smc_smoothing=args.use_smc_smoothing,
            smc_sigma=args.smc_sigma,
            smc_num_mc_samples=args.smc_num_mc_samples,
            smc_use_crn=args.smc_use_crn,
            smc_smoothing_mode=args.smc_smoothing_mode,
            smc_anneal_sigma=args.smc_anneal_sigma,
            smc_anneal_schedule=args.smc_anneal_schedule,
            smc_sigma_min=args.smc_sigma_min,
            smc_terminal_resample=args.smc_terminal_resample,
            FK_lmbda=args.FK_lmbda,
            FK_resample_t_start=args.FK_resample_t_start,
            FK_resample_t_end=args.FK_resample_t_end,
            top_k=args.top_k,
            rag_dataset=lookaheads,
            adaptive_resampling=args.adaptive_resampling,
            resample_frequency=args.resample_frequency,
            time_steps=args.num_inference_steps,
            resampling_t_start=args.guidance_start if is_flux else args.FK_resample_t_start,
            resampling_t_end=args.guidance_end if is_flux else args.resample_t_end,
            guidance_start=args.guidance_start,
            guidance_end=args.guidance_end,
            guidance_reward_fn=args.guidance_reward_fn,
            potential_type=args.potential_type,
            s_flow=args.s_flow,
            flux_debug=args.flux_debug,
        )

        start_time = datetime.now()
        images = pipe(
            prompt,
            prompt_idx=real_idx,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            fkd_args=fkd_args,
        )
        time_taken = datetime.now() - start_time

        images = _extract_pil_images(images)

        eval_prompt = [prompt[0]] * len(images)
        results = do_eval(prompt=eval_prompt, images=images, metrics_to_compute=metrics_to_compute)

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = real_idx

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        # Sort images (and rewards) by the guidance reward, best first.
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]

        for metric in metrics_to_compute:
            results[metric]["result"] = [results[metric]["result"][i] for i in sorted_idx]

        for metric in metrics_to_compute:
            for stat in ("mean", "max", "min", "std"):
                metrics_arr[metric][stat] += results[metric][stat]

        for metric in metrics_to_compute:
            print(metric, metrics_arr[metric]["mean"] / n_samples, metrics_arr[metric]["max"] / n_samples)

        if args.save_individual_images:
            sample_path = os.path.join(prompt_path, "samples")
            os.makedirs(sample_path, exist_ok=True)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))

            n_show = len(images)
            _, ax = plt.subplots(1, n_show, figsize=(n_show * 5, 5))
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


def get_args():
    parser = argparse.ArgumentParser(description="Phase 2: target sampling with LiDAR / RS-LiDAR guidance or RS-SMC.")
    parser.add_argument("--output_dir", type=str, default="Target_samples")
    parser.add_argument("--save_individual_images", action="store_true")
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_smc", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_rag", action="store_true", help="Enable closed-form lookahead (LiDAR) guidance")

    # RS-SMC
    parser.add_argument("--use_smc_smoothing", type=str2bool, nargs="?", const=True, default=False,
                        help="Score SMC particles with the randomized-smoothing reward (RS-SMC)")
    parser.add_argument("--use_rs_smc", type=str2bool, nargs="?", const=True, default=False,
                        help="Alias for --use_smc_smoothing")
    parser.add_argument("--smc_sigma", type=float, default=0.25, help="RS-SMC pixel-space noise std sigma")
    parser.add_argument("--smc_num_mc_samples", "--smc_M", dest="smc_num_mc_samples", type=int, default=4,
                        help="Number of Monte Carlo samples M for RS-SMC")
    parser.add_argument("--M", dest="M", type=int, default=None, help="Alias for --smc_num_mc_samples")
    parser.add_argument("--method", type=str, default=None, choices=["LiDAR", "RS-FlowLiDAR", "Vanilla"],
                        help="FLUX preset: 'LiDAR' / 'RS-FlowLiDAR' (lookahead velocity guidance) or 'Vanilla'")
    parser.add_argument("--s_flow", type=float, default=40.0, help="FLUX velocity guidance scale s_flow")
    parser.add_argument("--guidance_start", type=float, default=None,
                        help="FLUX guidance window start in flow time (default 1.0)")
    parser.add_argument("--guidance_end", type=float, default=None,
                        help="FLUX guidance window end in flow time (default 0.95)")
    parser.add_argument("--flux_debug", action="store_true", default=False,
                        help="FLUX: print per-step guidance diagnostics (t, w_r - w, |dt*dv|/|z|)")
    parser.add_argument("--smc_use_crn", type=str2bool, nargs="?", const=True, default=False,
                        help="RS-SMC: share each noise draw across particles (common random numbers)")
    parser.add_argument("--smc_smoothing_mode", type=str, default="loop", choices=["loop", "vectorized"],
                        help="RS-SMC: loop over M (low memory) or score all M*N images at once")
    parser.add_argument("--smc_anneal_sigma", type=str2bool, nargs="?", const=True, default=False,
                        help="RS-SMC: anneal sigma(t) over the resampling window")
    parser.add_argument("--smc_anneal_schedule", type=str, default="tweedie_analytical",
                        choices=["tweedie_analytical", "cosine", "linear", "sqrt_noise", "tweedie_snr"],
                        help="sigma(t) schedule; 'sqrt_noise' and 'tweedie_snr' are aliases of 'tweedie_analytical'")
    parser.add_argument("--smc_sigma_min", type=float, default=None, help="Minimum floor for the annealed sigma(t)")
    parser.add_argument("--smc_terminal_resample", type=str2bool, nargs="?", const=True, default=True,
                        help="Also resample at the terminal step (set False to keep particle diversity)")
    parser.add_argument("--terminal_resample", type=str2bool, nargs="?", const=True, default=None,
                        help="Alias for --smc_terminal_resample")

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lmbda", type=float, default=5000)
    parser.add_argument("--scale", type=float, default=12.5)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--metrics_to_compute", type=str, default="ImageReward", help="'#'-separated list of metrics")
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=20)
    parser.add_argument("--resample_t_start", type=int, default=1000)
    parser.add_argument("--resample_t_end", type=int, default=200)
    parser.add_argument("--potential_type", type=str, default="diff")

    parser.add_argument("--max_prompt", type=int, default=1000)
    parser.add_argument("--prompt_indices", type=str, default=None,
                        help="Optional subset of prompt indices ('stratified_100', '0-99', '0,5,10' or a JSON file)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of GPU shards to split prompts")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard ID (0 to num_shards-1)")
    parser.add_argument("--gpu_id", type=int, default=None, help="Explicit CUDA device ID to run on (e.g. 0 or 1)")
    parser.add_argument("--reward_type", type=str, choices=["ImageReward", "Clip-Score-only", "HumanPreference", "AS", "mix"],
                        default="ImageReward")

    parser.add_argument("--lookahead_path", type=str, default="100_50_5")

    # Feynman-Kac resampling (SMC)
    parser.add_argument("--FK_lmbda", type=float, default=10.0)
    parser.add_argument("--FK_resample_t_start", type=int, default=20)
    parser.add_argument("--FK_resample_t_end", type=int, default=80)

    parser.add_argument("--run_name", type=str, default="", help="Custom output subfolder name")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed prompts if results exist")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Force re-run and overwrite existing results")
    args = parser.parse_args()

    raw_m_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--M=') or a.startswith('--smc_M=')]
    raw_mc_flags = [a.split('=')[1] for a in sys.argv if a.startswith('--smc_num_mc_samples=')]
    if raw_m_flags and raw_mc_flags and raw_m_flags[0] != raw_mc_flags[0]:
        raise ValueError(f"Conflicting flags: --smc_num_mc_samples={raw_mc_flags[0]} but --M/--smc_M={raw_m_flags[0]}.")

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

    # FLUX presets only fill in defaults; explicit CLI values (e.g. an s_flow sweep) are kept.
    if args.method in ["LiDAR", "RS-FlowLiDAR"]:
        args.use_rag = True
        if not any(a.startswith("--s_flow") for a in sys.argv):
            args.s_flow = 40.0
        if not any(a.startswith("--guidance_start") for a in sys.argv):
            args.guidance_start = 1.0
        if not any(a.startswith("--guidance_end") for a in sys.argv):
            args.guidance_end = 0.95
        args.use_smc = False
    elif args.method == "Vanilla":
        args.use_rag = False
        args.s_flow = 0.0
        args.use_smc = False

    # SDXL defaults (LiDAR Table 2 guidance scale; calibrated RS-SMC sigma).
    if "xl" in args.model_name.lower():
        if args.scale == 12.5:
            args.scale = 8.0
            print("SDXL: using guidance scale 8.0")
        if args.use_smc_smoothing and args.smc_sigma == 0.25:
            args.smc_sigma = 0.15
            print("SDXL: using smc_sigma 0.15")

    return args


if __name__ == "__main__":
    main(get_args())
