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

from diffusers import DDIMScheduler, UNet2DConditionModel

import sys
sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion


from fks_utils import do_eval
from lookahead_datasets import gen_lookahead_samples

# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data




def main(args):
    if args.gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}"
        print(f"🎯 Assigned process explicitly to GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps

    if args.use_smc:
        assert args.resample_frequency > 0
        assert args.num_particles > 1

    # load prompt data # configure pipeline
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt)


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

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # set output directory
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{args.num_inference_steps}_{args.eta}_{args.num_particles}_{args.top_k}_{args.resample_t_end}_{args.lmbda}_{args.scale}_{args.seed}"
    if args.run_name:
        output_dir = os.path.join(args.output_dir, args.run_name)
    else:
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

    if args.use_rag:
        lookaheads = gen_lookahead_samples(args.lookahead_path,args.top_k)

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
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        results_file = os.path.join(prompt_path, "results.json")
        if args.resume and os.path.exists(results_file):
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
                print(f"Error loading cached result for prompt {prompt_idx}: {e}")

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
            FK_lmbda=args.FK_lmbda,
            FK_resample_t_start=args.FK_resample_t_start,
            FK_resample_t_end=args.FK_resample_t_end,
            top_k=args.top_k,
            rag_dataset=lookaheads,
            adaptive_resampling=args.adaptive_resampling,
            resample_frequency=args.resample_frequency,
            time_steps=args.num_inference_steps,
            resampling_t_end=args.resample_t_end,
            guidance_reward_fn=args.guidance_reward_fn,
            potential_type=args.potential_type,
        )

        start_time = datetime.now()
        images = pipe(
            prompt,
            prompt_idx=prompt_idx,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            fkd_args=fkd_args,
        )
        end_time = datetime.now()
        time_taken = end_time - start_time

        images = images[0][0]

        results = do_eval(
            prompt=prompt, images=images, metrics_to_compute=metrics_to_compute
        )

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = prompt_idx

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

            _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles * 5, 5))
            if args.num_particles == 1:
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="Target_samples")
    parser.add_argument("--save_individual_images", action="store_true")
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_smc", action="store_true")
    parser.add_argument("--use_grad", action="store_true")
    parser.add_argument("--use_rag", action="store_true")

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--lmbda", type=float, default=5000)
    parser.add_argument("--scale", type=float, default=12.5)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#Clip-Score#Clip-Diversity#HumanPreference#AS",
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
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    print(args.seed)
    main(args)
