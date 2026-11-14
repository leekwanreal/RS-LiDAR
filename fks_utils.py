"""Reward evaluation shared by Phase 1 and Phase 2.

`do_eval` scores a list of PIL images against their prompt with each requested metric
(ImageReward, Clip-Score, Clip-Diversity, HumanPreference, AS) and returns, per metric,
the per-image `result` list together with its mean / std / max / min. Phase 1 stores this dict
in `results.json`; Phase 2 reads `<metric>.result` as the lookahead rewards.
"""
import torch

from rslidar_utils import setup_runtime

setup_runtime()  # ImageReward compatibility shims; must run before the reward models are imported.

from fkd_diffusers.rewards import (  # noqa: E402
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_AS,
    do_human_preference_score,
)


def get_model(model_name):
    """Load an FKD pipeline (SD 1.4 / 1.5 / 2.1 or SDXL) with a DDIM scheduler."""
    from diffusers import DDIMScheduler
    if model_name == "stable-diffusion-xl":
        from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
        pipeline = FKDStableDiffusionXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-5":
        from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
        pipeline = FKDStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-4":
        from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
        pipeline = FKDStableDiffusion.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-2-1":
        from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
        pipeline = FKDStableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    return pipeline


def do_eval(*, prompt, images, metrics_to_compute):
    """Compute `metrics_to_compute` for `images`; a single prompt is broadcast to all images."""
    if images is None or len(images) == 0:
        return {}

    n_imgs = len(images)
    if isinstance(prompt, str):
        prompt_list = [prompt] * n_imgs
    elif isinstance(prompt, (list, tuple)):
        if len(prompt) == 1:
            prompt_list = list(prompt) * n_imgs
        elif len(prompt) < n_imgs:
            prompt_list = list(prompt) + [prompt[0]] * (n_imgs - len(prompt))
        elif len(prompt) > n_imgs:
            prompt_list = list(prompt)[:n_imgs]
        else:
            prompt_list = list(prompt)
    else:
        prompt_list = [str(prompt)] * n_imgs

    results = {}
    for metric in metrics_to_compute:
        if metric == "Clip-Diversity":
            results[metric] = {}
            res, div = do_clip_score_diversity(images=images, prompts=prompt_list)
            if res is None:
                res = [0.0] * n_imgs
            if div is None:
                div = 0.0
            results[metric]["result"] = res
            results[metric]["diversity"] = div
            results_arr = torch.tensor(div, dtype=torch.float32)

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = 0.0
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "ImageReward":
            results[metric] = {}
            res = do_image_reward(images=images, prompts=prompt_list)
            if res is None:
                res = [0.0] * n_imgs
            results[metric]["result"] = res

            results_arr = torch.tensor(res, dtype=torch.float32)

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item() if n_imgs > 1 else 0.0
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "Clip-Score":
            results[metric] = {}
            res = do_clip_score(images=images, prompts=prompt_list)
            if res is None:
                res = [0.0] * n_imgs
            results[metric]["result"] = res

            results_arr = torch.tensor(res, dtype=torch.float32)

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item() if n_imgs > 1 else 0.0
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "HumanPreference":
            results[metric] = {}
            res = do_human_preference_score(images=images, prompts=prompt_list)
            if res is None:
                res = [0.0] * n_imgs
            results[metric]["result"] = res

            results_arr = torch.tensor(res, dtype=torch.float32)

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item() if n_imgs > 1 else 0.0
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "AS":
            results[metric] = {}
            as_res = do_AS(images=images, prompts=prompt_list)
            if as_res is None:
                as_res = [0.0] * n_imgs
            results[metric]["result"] = as_res

            results_arr = torch.tensor(as_res, dtype=torch.float32)

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item() if n_imgs > 1 else 0.0
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results
