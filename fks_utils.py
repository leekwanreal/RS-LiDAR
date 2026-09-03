"""
Utility functions for the FKD pipeline.
"""
import torch

# Compatibility patch for ImageReward with newer transformers versions (>= 4.40)
try:
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
    import transformers
    if not hasattr(transformers, "EncoderDecoderCache"):
        class EncoderDecoderCache:
            pass
        transformers.EncoderDecoderCache = EncoderDecoderCache
except Exception:
    pass

from diffusers import DDIMScheduler

from fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_pipeline_sd import FKDStableDiffusion

from fkd_diffusers.rewards import (
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_AS,
    do_human_preference_score,
)


def get_model(model_name):
    """
    Get the FKD-supported model based on the model name.
    """
    if model_name == "stable-diffusion-xl":
        pipeline = FKDStableDiffusionXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-5":
        pipeline = FKDStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-4":
        pipeline = FKDStableDiffusion.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-2-1":
        pipeline = FKDStableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    return pipeline



def do_eval(*, prompt, images, metrics_to_compute):
    """
    Compute the metrics for the given images and prompt.
    """
    results = {}
    for metric in metrics_to_compute:
        if metric == "Clip-Diversity":
            results[metric] = {}
            (
                results[metric]["result"],
                results[metric]["diversity"],
            ) = do_clip_score_diversity(images=images, prompts=prompt)
            results_arr = torch.tensor(results[metric]["diversity"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "ImageReward":
            results[metric] = {}
            results[metric]["result"] = do_image_reward(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "Clip-Score":
            results[metric] = {}
            results[metric]["result"] = do_clip_score(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()
        elif metric == "HumanPreference":
            results[metric] = {}
            results[metric]["result"] = do_human_preference_score(
                images=images, prompts=prompt
            )

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "AS":
            results[metric] = {}
            as_res = do_AS(images=images, prompts=prompt)
            if as_res is None:
                as_res = [0.0] * len(images)
            results[metric]["result"] = as_res

            results_arr = torch.tensor(as_res, dtype=torch.float32)

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results
