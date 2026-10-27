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

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps

    if args.use_smc:
        assert args.resample_frequency > 0
        if args.num_particles <= 1:
            print("⚠️ [CẢNH BÁO] args.num_particles <= 1: Tự động vô hiệu hóa SMC resampling (chuyển sang Vanilla unguided sampling vì N=1 không thể resample).")
            args.use_smc = False

    # load prompt data # configure pipeline
    parsed_indices = parse_prompt_indices(getattr(args, "prompt_indices", None))
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt, prompt_indices=parsed_indices)


    if "mhdang/dpo" in args.model_name and "xl" not in args.model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # load finetuned model
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
    # elif "black-forest-labs/FLUX.1-dev" in args.model_name:
    #     print("Using FLUX")
    #     pipe = FluxPipeline_LiDAR.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    else:
        pipe = FKDStableDiffusion.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        )
    if "FLUX" not in args.model_name:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # set device (preserve assigned device cuda:actual_gpu_id)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # set output directory
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.run_name:
        output_dir = os.path.join(args.output_dir, args.run_name)
    else:
        model_tag = "SDXL" if "xl" in args.model_name.lower() else "SD15"
        if args.use_smc:
            crn_suffix = "_crn" if getattr(args, "smc_use_crn", False) else ""
            anneal_suffix = "_anneal" if getattr(args, "smc_anneal_sigma", False) else ""
            term_suffix = "" if getattr(args, "smc_terminal_resample", True) else "_noTerminal"
            if getattr(args, "use_smc_smoothing", False):
                method_tag = f"{model_tag}_RSSMC_N{args.num_particles}_M{args.smc_num_mc_samples}_sigma{args.smc_sigma}{crn_suffix}{anneal_suffix}{term_suffix}_freq{args.resample_frequency}_lmbda{args.FK_lmbda}"
            else:
                method_tag = f"{model_tag}_SMC_N{args.num_particles}{term_suffix}_freq{args.resample_frequency}_lmbda{args.FK_lmbda}"
            if args.use_rag:
                method_tag += f"_LiDAR_topk{args.top_k}_lmbda{int(args.lmbda)}"
        elif args.use_rag:
            method_tag = f"{model_tag}_LiDAR_N{args.num_particles}_topk{args.top_k}_lmbda{int(args.lmbda)}"
        else:
            method_tag = f"{model_tag}_BoN_N{args.num_particles}" if args.num_particles > 1 else f"{model_tag}_Vanilla_N1"

        prefix = f"{method_tag}_steps{args.num_inference_steps}_scale{args.scale}_seed{args.seed}"
        output_dir = os.path.join(args.output_dir, f"{prefix}_{cur_time}")

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

    lookaheads = None
    if args.use_rag:
        lookaheads = gen_lookahead_samples(args.lookahead_path, args.top_k)
        if len(lookaheads) > 0:
            first_idx = next(iter(lookaheads.data.keys()))
            first_lat = lookaheads[first_idx]["latents"]
            expected_dim = 128 if ("xl" in args.model_name.lower() or "flux" in args.model_name.lower()) else 64
            if first_lat.shape[-1] != expected_dim:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"❌ [LỖI KHÔNG TƯƠNG THÍCH KÍCH THƯỚC LATENT]\n"
                    f"Lookahead dataset '{args.lookahead_path}' có spatial dimension là {first_lat.shape[-1]}x{first_lat.shape[-1]},\n"
                    f"trong khi mô hình hiện tại '{args.model_name}' yêu cầu {expected_dim}x{expected_dim}!\n"
                    f"Nguyên nhân: Phase 1 đã chạy với mô hình khác (hoặc bị tái sử dụng nhầm latents SD 1.5 64x64 cho SDXL 128x128).\n"
                    f"Giải pháp: Vui lòng chạy lại Phase 1 với cờ --overwrite để sinh lookahead latents {expected_dim}x{expected_dim} chuẩn xác!\n"
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

    for prompt_idx in tqdm(range(start_idx, end_idx), desc=f"Shard-{args.shard_id}"):
        item = prompt_data[prompt_idx]
        real_idx = item.get("original_idx", prompt_idx)
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{real_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        results_file = os.path.join(prompt_path, "results.json")
        if args.resume and not getattr(args, "overwrite", False) and os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    res_cached = json.load(f)
                for metric in metrics_to_compute:
                    if metric in res_cached:
                        metrics_arr[metric]["mean"] += res_cached[metric]["mean"]
                        metrics_arr[metric]["max"] += res_cached[metric]["max"]
                        metrics_arr[metric]["min"] += res_cached[metric]["min"]
                        metrics_arr[metric]["std"] += res_cached[metric]["std"]
                n_samples += 1
                continue
            except Exception as e:
                print(f"Error loading cached result for prompt {real_idx}: {e}")

        # dump metadata
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        fkd_args = dict(
            lmbda=args.lmbda,
            scale=args.scale,
            reward_type=args.reward_type,
            num_particles=args.num_particles,
            use_smc=args.use_smc,
            use_grad=args.use_grad,
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

        start_time = datetime.now()
        images = pipe(
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

        images = _extract_pil_images(images)

        eval_prompt = [prompt[0]] * len(images) if isinstance(prompt, list) else [prompt] * len(images)
        results = do_eval(
            prompt=eval_prompt, images=images, metrics_to_compute=metrics_to_compute
        )

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = real_idx

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        # sort images by reward
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]

        for metric in metrics_to_compute:
            results[metric]["result"] = [
                results[metric]["result"][i] for i in sorted_idx
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
    parser.add_argument("--output_dir", type=str, default="Target_samples")
    parser.add_argument("--save_individual_images", action="store_true")
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_smc", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_grad", action="store_true")
    parser.add_argument("--use_rag", action="store_true")

    ## RS-SMC arguments
    parser.add_argument("--use_smc_smoothing", type=str2bool, nargs="?", const=True, default=False, help="Enable Randomized Smoothing for SMC resampling")
    parser.add_argument("--use_rs_smc", type=str2bool, nargs="?", const=True, default=False, help="Alias for --use_smc_smoothing")
    parser.add_argument("--smc_sigma", type=float, default=0.25, help="Gaussian noise standard deviation for RS-SMC (default: 0.25)")
    parser.add_argument("--smc_num_mc_samples", type=int, default=4, help="Number of Monte Carlo samples M for RS-SMC (default: 4)")
    parser.add_argument("--smc_M", type=int, default=4, help="Alias for --smc_num_mc_samples")
    parser.add_argument("--M", type=int, default=4, help="Alias for number of MC samples M")
    parser.add_argument("--noise_mode", type=str, default="independent_antithetic", choices=["independent_antithetic", "independent_iid", "coupled_crn"], help="Monte Carlo noise sampling mode")
    parser.add_argument("--beta_risk", type=float, default=0.5, help="Entropic risk penalty parameter beta (Theorem 5.2, default: 0.5)")
    parser.add_argument("--sigma_high", type=float, default=1.0, help="Upper noise standard deviation limit (default: 1.0)")
    parser.add_argument("--sigma_low", type=float, default=0.25, help="Lower noise standard deviation limit (default: 0.25)")
    parser.add_argument("--eta_drift", type=float, default=0.4, help="Fraction of lookahead drift absorbed in path measure (default: 0.4)")
    parser.add_argument("--alpha_couple", type=float, default=0.7, help="Coupling strength coefficient (default: 0.7)")
    parser.add_argument("--s_flow", type=float, default=12.5, help="Velocity guidance scale for Rectified Flows (default: 12.5)")
    parser.add_argument("--pareto_rho", type=float, default=0.5, help="Pareto trade-off weight parameter (default: 0.5)")
    parser.add_argument("--smc_smoothing_domain", type=str, default="image", choices=["image", "latent"], help="Noise injection domain: 'image' (default) or 'latent'")
    parser.add_argument("--smc_use_crn", type=str2bool, nargs="?", const=True, default=False, help="Use Common Random Numbers for RS-SMC across particles")
    parser.add_argument("--smc_smoothing_mode", type=str, default="loop", choices=["loop", "vectorized"], help="Smoothing execution mode: 'loop' (default, VRAM safe) or 'vectorized'")
    parser.add_argument("--smc_anneal_sigma", type=str2bool, nargs="?", const=True, default=False, help="Enable Tweedie posterior-grounded Annealed Randomized Smoothing for SMC")
    parser.add_argument("--smc_anneal_schedule", type=str, default="tweedie_analytical", choices=["tweedie_analytical", "cosine", "linear", "sqrt_noise", "tweedie_snr"], help="Annealing schedule type: 'tweedie_analytical' (default, optimal), 'cosine', 'linear', or legacy 'sqrt_noise'")
    parser.add_argument("--smc_sigma_min", type=float, default=None, help="Minimum floor for annealed sigma")
    parser.add_argument("--smc_terminal_resample", type=str2bool, nargs="?", const=True, default=True, help="Whether to perform resampling at terminal step (time_steps - 1). Default: True. Set False to preserve particle diversity.")
    parser.add_argument("--terminal_resample", type=str2bool, nargs="?", const=True, default=None, help="Alias for --smc_terminal_resample")

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lmbda", type=float, default=5000)
    parser.add_argument("--scale", type=float, default=12.5)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=20)
    parser.add_argument("--resample_t_start", type=int, default=1000)
    parser.add_argument("--resample_t_end", type=int, default=200)
    parser.add_argument("--potential_type", type=str, default="diff")

    parser.add_argument("--max_prompt", type=int, default=1000)
    parser.add_argument("--prompt_indices", type=str, default=None, help="Optional subset of prompt indices (e.g. 'stratified_100', '0-99', '0,5,10', or path to JSON)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of GPU shards to split prompts")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard ID (0 to num_shards-1)")
    parser.add_argument("--gpu_id", type=int, default=None, help="Explicit CUDA device ID to run on (e.g. 0 or 1)")
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

    if args.smc_M is not None:
        args.smc_num_mc_samples = args.smc_M
    if args.terminal_resample is not None:
        args.smc_terminal_resample = args.terminal_resample
    if args.use_rs_smc:
        args.use_smc_smoothing = True
    if args.use_smc_smoothing:
        args.use_smc = True

    # Auto-dispatch scale and sigma for SDXL vs SD 1.5
    if "xl" in args.model_name.lower():
        if args.scale == 12.5:
            args.scale = 8.0
            print("⚡ Auto-dispatch scale to 8.0 for SDXL (Paper Table 2 standard)")
        if args.use_smc_smoothing and args.smc_sigma == 0.25:
            args.smc_sigma = 0.15
            print("⚡ Auto-dispatch smc_sigma to 0.15 for SDXL (Calibrated [0.10, 0.20] range)")
    else:
        if args.use_smc_smoothing and getattr(args, "smc_sigma", 0.25) == 1.0:
            # If user had legacy 1.0 default, warn and recommend 0.25
            pass

    return args


if __name__ == "__main__":
    args = get_args()
    print(args.seed)
    main(args)
