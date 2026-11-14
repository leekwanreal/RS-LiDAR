"""Empirical Lipschitz constant of reward models: LiDAR vs. RS-LiDAR (randomized smoothing).

Estimates the empirical Lipschitz constant L = sup |R(x1) - R(x2)| / ||x1 - x2||_2 via secant slopes:
    - LiDAR:    slope = |R(x_clean) - R(x_pert)| / ||x_clean - x_pert||_2
    - RS-LiDAR: slope = |R_sigma2(x_clean) - R_sigma2(x_pert)| / ||x_clean - x_pert||_2
where R_sigma2 is a Monte Carlo estimate of the Gaussian-smoothed reward, computed for BOTH the clean
and the perturbed image.

Features:
    - Reward models: ImageReward, CLIP-Score, HPS v2.1, Aesthetic Score.
    - Sweep over the smoothing radius (`--sigma2_list 0.0,0.1,0.25,0.5,1.0`) to trace the O(1/sigma)
      decay of the Lipschitz bound.
    - Multi-GPU sharding (`--num_shards/--shard_id`), per-prompt checkpointing with resume,
      and shard merging (`--merge_shards`).
    - Outputs CSV / JSON metrics and 3-panel comparison figures.

Usage (from the repository root):
    python -m experiments.lipschitz_empirical --num_prompts 50 --num_particles 10 --sigma1 0.1
    python -m experiments.lipschitz_empirical --merge_shards --output_dir results/lipschitz_empirical
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
if hasattr(sys.stderr, "reconfigure"):
    try: sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

import types
import json
import argparse
import glob
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch-only transformers; reduce CUDA memory fragmentation.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# transformers compatibility shims (local variant: also stubs DynamicCache / Cache).
try:
    import transformers
    for dummy_cls in ["EncoderDecoderCache", "DynamicCache", "Cache"]:
        if not hasattr(transformers, dummy_cls):
            setattr(transformers, dummy_cls, type(dummy_cls, (), {}))
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

# Stub out wandb when it is not installed (ImageReward imports it).
if "wandb" not in sys.modules:
    try:
        import wandb  # noqa: F401
    except Exception:
        import importlib.machinery
        dummy_wandb = types.ModuleType("wandb")
        dummy_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        dummy_wandb.run = None
        dummy_wandb.init = lambda *args, **kwargs: None
        dummy_wandb.log = lambda *args, **kwargs: None
        sys.modules["wandb"] = dummy_wandb

import torch
import torch.nn.functional as F
try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
except ImportError:
    DPMSolverMultistepScheduler = None
    StableDiffusionPipeline = None

# OpenAI CLIP
try:
    import clip
except ImportError:
    clip = None

# ImageReward
try:
    from fkd_diffusers.image_reward_utils import rm_load
except ImportError:
    try:
        from image_reward_utils import rm_load
    except ImportError:
        try:
            import ImageReward as RM
            rm_load = RM.load
        except Exception:
            rm_load = None

# Multi-metric reward scorers
try:
    from fkd_diffusers.rewards import REWARDS_DICT, CLIPScore, AestheticScore, do_human_preference_score
except ImportError:
    try:
        from rewards import REWARDS_DICT, CLIPScore, AestheticScore, do_human_preference_score
    except ImportError:
        REWARDS_DICT = {}
        CLIPScore = None
        AestheticScore = None
        do_human_preference_score = None


# ======================================================================================
# FAST BATCH REWARD EVALUATION (GPU VECTORIZED)
# ======================================================================================
_fast_batch_initialized = False

@torch.inference_mode()
def fast_batch_clip_and_aesthetic(images, prompt, device="cuda"):
    """
    Compute CLIP-Score and Aesthetic Score for all images in a single batched GPU pass
    ([N, 3, 224, 224]) instead of scoring images one by one.
    """
    global _fast_batch_initialized
    if CLIPScore is None:
        return None, None

    # Lazily load CLIPScore.
    if REWARDS_DICT.get("Clip-Score") is None:
        try:
            dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=os.path.expanduser("~/.cache/clip"), device=dev)
        except Exception:
            return None, None

    # Lazily load AestheticScore.
    if REWARDS_DICT.get("AS") is None and AestheticScore is not None:
        try:
            dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            weight_path = "sac+logos+ava1-l14-linearMSE.pth"
            if not os.path.exists(weight_path):
                try:
                    import urllib.request
                    urllib.request.urlretrieve(
                        "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
                        weight_path
                    )
                except Exception:
                    pass
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location='cpu')
                as_obj = AestheticScore(download_root=os.path.expanduser("~/.cache/clip"), device=dev)
                as_obj.mlp.load_state_dict(state_dict, strict=False)
                as_obj.mlp.to(dev)
                REWARDS_DICT["AS"] = as_obj
        except Exception:
            pass

    clip_obj = REWARDS_DICT.get("Clip-Score")
    as_obj = REWARDS_DICT.get("AS")

    if clip_obj is None:
        return None, None

    try:
        clip_model = clip_obj.clip_model
        preprocess = clip_obj.preprocess
        model_dtype = next(clip_model.parameters()).dtype

        # 1. Stack the N images into a [N, 3, 224, 224] tensor
        tensors = torch.stack([preprocess(img) for img in images]).to(device=device, dtype=model_dtype)

        # 2. Encode all images in one GPU pass
        image_features = F.normalize(clip_model.encode_image(tensors))

        # 3. CLIP-Score: encode the prompt once, then take dot products
        p_str = prompt[0] if isinstance(prompt, (list, tuple)) else str(prompt)
        text_tokens = clip.tokenize(p_str, truncate=True).to(device)
        txt_features = F.normalize(clip_model.encode_text(text_tokens))
        clip_scores = (image_features.float() * txt_features.float()).sum(dim=-1).cpu().tolist()

        # 4. Aesthetic Score: forward the image features through the MLP head
        if as_obj is not None and hasattr(as_obj, "mlp"):
            as_scores = as_obj.mlp(image_features.float()).squeeze(-1).cpu().tolist()
        else:
            as_scores = None

        if not _fast_batch_initialized:
            print(" [Fast-Batch] Vectorized GPU scoring enabled for CLIP-Score and Aesthetic Score.")
            _fast_batch_initialized = True

        return clip_scores, as_scores
    except Exception:
        return None, None


# ======================================================================================
# PROMPT LOADING & SAMPLING UTILITIES
# ======================================================================================
def load_stratified_prompts(prompt_path="prompt_files/geneval_50_stratified.jsonl", max_prompts=-1, seed=42):
    """
    Load GenEval prompts and draw a stratified subset (balanced over task tags) of size max_prompts.
    If max_prompts > 50 (or <= 0) and the default 50-prompt file is used (or the given file is missing),
    switch to the larger local 100- or 553-prompt file under prompt_files/.
    """
    # Only switch files when using the default 50-prompt file or when the given file does not exist
    is_default_50 = os.path.basename(prompt_path) == "geneval_50_stratified.jsonl"
    if (is_default_50 or not os.path.exists(prompt_path)) and (max_prompts > 50 or max_prompts <= 0):
        if max_prompts > 100 or max_prompts <= 0:
            larger_candidates = [
                "prompt_files/geneval_metadata.jsonl",
                "prompt_files/geneval_100_stratified.jsonl",
                "prompt_files/geneval_100_representative.jsonl"
            ]
        else:
            larger_candidates = [
                "prompt_files/geneval_100_stratified.jsonl",
                "prompt_files/geneval_100_representative.jsonl",
                "prompt_files/geneval_metadata.jsonl"
            ]
        for c in larger_candidates:
            if os.path.exists(c):
                prompt_path = c
                break

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Prompt file not found: '{prompt_path}'. Pass --prompts_file or place the file under prompt_files/."
        )

    items = []
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("prompt"):
                        items.append(item)

    if not items:
        fallback_prompts = [
            ("single_object", "a photo of a bench"), ("single_object", "a photo of a potted plant"),
            ("two_object", "a photo of a knife and a zebra"), ("two_object", "a photo of a couch and a horse"),
            ("counting", "three apples on a wooden table"), ("counting", "two cars parked on a street"),
            ("colors", "a red sports car"), ("colors", "a blue ceramic vase"),
            ("position", "a laptop on top of a desk"), ("position", "a cat under a wooden chair"),
            ("color_attr", "a red dog and a blue cat"), ("color_attr", "a green book and a yellow pen")
        ]
        items = [{"tag": t, "prompt": p} for t, p in fallback_prompts]

    # No limit, or the limit covers all prompts
    if max_prompts <= 0 or max_prompts >= len(items):
        print(f"Loaded all {len(items)} prompts from {prompt_path}.")
        return [it["prompt"] for it in items[:max_prompts]] if max_prompts > 0 else [it["prompt"] for it in items]

    # Stratified sampling over task tags
    by_tag = {}
    for it in items:
        t = it.get("tag", "general")
        by_tag.setdefault(t, []).append(it)

    tags = sorted(list(by_tag.keys()))
    rng = random.Random(seed)
    rng.shuffle(tags)

    base = max_prompts // len(tags)
    rem = max_prompts % len(tags)

    chosen = []
    task_dist = {}
    for i, t in enumerate(tags):
        count = base + (1 if i < rem else 0)
        bucket = by_tag[t]
        k = min(count, len(bucket))
        if k > 0:
            sampled_task = rng.sample(bucket, k)
            chosen.extend(sampled_task)
            task_dist[t] = len(sampled_task)

    # Seeded shuffle so that every GPU shard receives a mix of tasks
    rng.shuffle(chosen)
    print(f"Stratified sample of {len(chosen)} prompts (seed={seed}): {task_dist}")
    return [it["prompt"] for it in chosen]


# ======================================================================================
# GENERATION & IMAGE PROCESSING
# ======================================================================================
@torch.inference_mode()
def generate_clean_and_perturbed_images(
    pipe, vae, prompt, num_particles, guidance_steps, guidance_scale,
    sigma1, perturbation_type="gaussian", seed=42, device="cuda"
):
    """
    Generate K clean images x_clean, decode them with the VAE to [-1, 1], then build perturbed images
    x_pert according to perturbation_type (default: additive Gaussian noise with std sigma1).
    Returns:
      - clean_tensors: [K, 3, H, W] tensor in [-1, 1]
      - pert_tensors: [K, 3, H, W] tensor in [-1, 1]
      - delta_x: NumPy array [K] with ||x_clean - x_pert||_2
      - clean_pils: list of K clean PIL images
      - pert_pils: list of K perturbed PIL images
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    # 1. Generate clean latents
    latents = pipe(
        [prompt] * num_particles,
        num_inference_steps=guidance_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent"
    ).images

    # 2. Decode with the VAE to an image tensor in [-1, 1]
    scaled = (latents / vae.config.scaling_factor).to(device=device, dtype=vae.dtype)
    clean_tensors = vae.decode(scaled, return_dict=False)[0].clamp(-1.0, 1.0)

    # 3. Build the perturbed images x_pert
    if perturbation_type == "gaussian":
        noise = torch.randn_like(clean_tensors) * sigma1
        pert_tensors = (clean_tensors + noise).clamp(-1.0, 1.0)
    elif perturbation_type == "blur":
        # 5x5 Gaussian blur via F.conv2d
        channels = clean_tensors.shape[1]
        kernel_size = 5
        sigma_blur = max(0.5, sigma1 * 10.0)
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma_blur**2))
        kernel = (kernel / torch.sum(kernel)).repeat(channels, 1, 1, 1)
        pert_tensors = F.conv2d(clean_tensors, kernel, padding=kernel_size // 2, groups=channels).clamp(-1.0, 1.0)
    else:
        pert_tensors = clean_tensors.clone()

    # 4. Pixel-space Euclidean distance ||x_clean - x_pert||_2
    diff = clean_tensors - pert_tensors
    # Flatten (channels * H * W) per particle k
    delta_x = torch.norm(diff.view(num_particles, -1), p=2, dim=1).cpu().numpy()

    # 5. Convert to PIL
    clean_pils = pipe.image_processor.postprocess(clean_tensors, output_type="pil")
    pert_pils = pipe.image_processor.postprocess(pert_tensors, output_type="pil")

    return clean_tensors, pert_tensors, delta_x, clean_pils, pert_pils


@torch.inference_mode()
def evaluate_reward_all_models(
    images, prompt, ir_model, use_ir, use_clip, use_hps, use_aesthetic, device="cuda"
):
    """
    Score a list of PIL images with every enabled reward model.
    Returns {'ImageReward': np.array, 'CLIP-Score': np.array, 'HPS-v2.1': np.array, 'Aesthetic': np.array}.
    """
    results = {}
    N = len(images)
    p_str = prompt[0] if isinstance(prompt, (list, tuple)) else str(prompt)

    # 1. ImageReward
    if use_ir and ir_model is not None:
        try:
            ir_scores = []
            for img in images:
                sc = ir_model.score(p_str, img)
                ir_scores.append(float(sc))
            results["ImageReward"] = np.array(ir_scores)
        except Exception:
            results["ImageReward"] = None

    # 2. CLIP-Score & Aesthetic Score (Fast-Batch GPU)
    if (use_clip or use_aesthetic):
        try:
            clip_scs, as_scs = fast_batch_clip_and_aesthetic(images, p_str, device=device)
            if use_clip and clip_scs is not None:
                results["CLIP-Score"] = np.array(clip_scs)
            if use_aesthetic and as_scs is not None:
                results["Aesthetic"] = np.array(as_scs)
        except Exception:
            pass

    # 3. HPS v2.1
    if use_hps and do_human_preference_score is not None:
        try:
            h_scs = do_human_preference_score(images=images, prompts=[p_str] * N)
            results["HPS-v2.1"] = np.array(h_scs)
        except Exception:
            results["HPS-v2.1"] = None

    return results


# ======================================================================================
# CORE EXPERIMENT: EMPIRICAL LIPSCHITZ CONSTANT
# ======================================================================================
def run_empirical_lipschitz_test(
    pipe, vae, ir_model, prompt_list, args, device="cuda"
):
    """
    Empirical Lipschitz measurement:
    1. Generate clean images x_clean and perturbed images x_pert.
    2. LiDAR (vanilla) rewards: R(x_clean) and R(x_pert).
    3. RS-LiDAR Monte Carlo rewards:
       - R_sigma2(x_clean) = 1/M sum_{m=1}^M R(x_clean + sigma2 u_m)
       - R_sigma2(x_pert)  = 1/M sum_{m=1}^M R(x_pert + sigma2 u'_m)
    4. Secant slopes:
       - Slope_vanilla = |R(x_clean) - R(x_pert)| / ||x_clean - x_pert||_2
       - Slope_RS      = |R_sigma2(x_clean) - R_sigma2(x_pert)| / ||x_clean - x_pert||_2
    """
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"checkpoint_shard_{args.shard_id}.json")

    # Split prompts across shards
    total_prompts = len(prompt_list)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        active_prompts = prompt_list[start_idx:end_idx]
    else:
        start_idx = 0
        end_idx = total_prompts
        active_prompts = prompt_list

    # Parse the sigma2 values to test
    if args.sigma2_list:
        try:
            sigma2_vals = [float(s.strip()) for s in args.sigma2_list.split(",") if s.strip()]
        except Exception:
            sigma2_vals = [0.0, 0.1, 0.25, 0.5, 1.0]
    else:
        sigma2_vals = [args.sigma2]

    # Make sure at least one sigma2 > 0 is present for the RS-LiDAR comparison
    if not any(s > 0 for s in sigma2_vals):
        sigma2_vals.append(1.0)
    sigma2_vals = sorted(list(set(sigma2_vals)))

    print(f"\n[Shard {args.shard_id}/{args.num_shards}] Running on {len(active_prompts)} prompts (#{start_idx} to #{end_idx-1})")
    print(f"Settings: {args.num_particles} particles/prompt | sigma1={args.sigma1} ({args.perturbation_type}) | M={args.num_mc_samples}")
    print(f"RS-LiDAR smoothing radii sigma2: {sigma2_vals}")

    # Load the checkpoint if present
    saved_data = {"completed_prompts": [], "pairs_data": []}
    if os.path.exists(ckpt_path) and not args.overwrite:
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            print(f"Loaded checkpoint: {len(saved_data.get('completed_prompts', []))} prompts already completed.")
        except Exception as e:
            print(f"Warning: could not read checkpoint ({e}); starting from scratch.")

    completed_set = set(saved_data.get("completed_prompts", []))
    all_pairs_data = saved_data.get("pairs_data", [])

    pbar = tqdm(enumerate(active_prompts), total=len(active_prompts), desc=f"Shard {args.shard_id} Progress")

    for local_idx, prompt in pbar:
        global_idx = start_idx + local_idx
        if str(global_idx) in completed_set or prompt in completed_set:
            continue

        prompt_seed = args.seed + global_idx * 100

        # 1. Generate clean images x_clean and perturbed images x_pert
        clean_tensors, pert_tensors, delta_x, clean_pils, pert_pils = generate_clean_and_perturbed_images(
            pipe=pipe,
            vae=vae,
            prompt=prompt,
            num_particles=args.num_particles,
            guidance_steps=args.guidance_steps,
            guidance_scale=args.guidance_scale,
            sigma1=args.sigma1,
            perturbation_type=args.perturbation_type,
            seed=prompt_seed,
            device=device
        )

        # 2. LiDAR (vanilla) rewards: R(x_clean) and R(x_pert)
        vanilla_clean_rewards = evaluate_reward_all_models(
            clean_pils, prompt, ir_model, args.use_imagereward, args.use_clip, args.use_hps, args.use_aesthetic, device=device
        )
        vanilla_pert_rewards = evaluate_reward_all_models(
            pert_pils, prompt, ir_model, args.use_imagereward, args.use_clip, args.use_hps, args.use_aesthetic, device=device
        )

        # 3. RS-LiDAR Monte Carlo rewards for each sigma2,
        #    computed for BOTH the clean and the perturbed images
        rs_rewards_by_sigma = {}  # {sigma2: {'clean': {model: [K]}, 'pert': {model: [K]}}}

        for s2 in sigma2_vals:
            if s2 == 0.0:
                # sigma2 = 0 is the unsmoothed (vanilla) reward
                rs_rewards_by_sigma[s2] = {
                    "clean": vanilla_clean_rewards,
                    "pert": vanilla_pert_rewards
                }
                continue

            # M Monte Carlo samples for x_clean and x_pert. With args.use_crn, the same noise u_m is used
            # for both (common random numbers), which cancels most of the Monte Carlo variance in the difference.
            use_crn = getattr(args, "use_crn", True)
            mc_clean_accum = {m: np.zeros(args.num_particles) for m in vanilla_clean_rewards if vanilla_clean_rewards[m] is not None}
            mc_clean_counts = {m: 0 for m in mc_clean_accum}
            mc_pert_accum = {m: np.zeros(args.num_particles) for m in vanilla_pert_rewards if vanilla_pert_rewards[m] is not None}
            mc_pert_counts = {m: 0 for m in mc_pert_accum}

            for m_iter in range(args.num_mc_samples):
                # Noise vector u_m
                u_m = torch.randn_like(clean_tensors) * s2

                # 1. Clean image x_clean with noise u_m
                noisy_clean_t = (clean_tensors + u_m).clamp(-1.0, 1.0)
                noisy_clean_pils = pipe.image_processor.postprocess(noisy_clean_t, output_type="pil")
                r_m_clean = evaluate_reward_all_models(
                    noisy_clean_pils, prompt, ir_model, args.use_imagereward, args.use_clip, args.use_hps, args.use_aesthetic, device=device
                )
                for model_k, scores in r_m_clean.items():
                    if scores is not None and model_k in mc_clean_accum:
                        mc_clean_accum[model_k] += scores
                        mc_clean_counts[model_k] += 1

                # 2. Perturbed image x_pert: shared u_m (CRN) or independent u'_m
                u_prime_m = u_m if use_crn else (torch.randn_like(pert_tensors) * s2)
                noisy_pert_t = (pert_tensors + u_prime_m).clamp(-1.0, 1.0)
                noisy_pert_pils = pipe.image_processor.postprocess(noisy_pert_t, output_type="pil")
                r_m_pert = evaluate_reward_all_models(
                    noisy_pert_pils, prompt, ir_model, args.use_imagereward, args.use_clip, args.use_hps, args.use_aesthetic, device=device
                )
                for model_k, scores in r_m_pert.items():
                    if scores is not None and model_k in mc_pert_accum:
                        mc_pert_accum[model_k] += scores
                        mc_pert_counts[model_k] += 1

            rs_clean_scores = {m: (mc_clean_accum[m] / max(1, mc_clean_counts[m])).tolist() for m in mc_clean_accum}
            rs_pert_scores = {m: (mc_pert_accum[m] / max(1, mc_pert_counts[m])).tolist() for m in mc_pert_accum}

            rs_rewards_by_sigma[s2] = {
                "clean": rs_clean_scores,
                "pert": rs_pert_scores
            }

        # 4. Record per-pair data (delta_x, delta_R, slopes)
        for k in range(args.num_particles):
            dx_k = float(delta_x[k])
            if dx_k < 1e-7:
                dx_k = 1e-7  # avoid division by zero

            pair_entry = {
                "prompt_idx": global_idx,
                "particle_idx": k,
                "prompt": prompt,
                "delta_x": dx_k,
                "models": {}
            }

            for model_name in vanilla_clean_rewards:
                if vanilla_clean_rewards[model_name] is None or vanilla_pert_rewards[model_name] is None:
                    continue

                r_clean = float(vanilla_clean_rewards[model_name][k])
                r_pert = float(vanilla_pert_rewards[model_name][k])
                diff_vanilla = abs(r_clean - r_pert)
                slope_vanilla = diff_vanilla / dx_k

                model_entry = {
                    "r_clean_vanilla": r_clean,
                    "r_pert_vanilla": r_pert,
                    "delta_r_vanilla": diff_vanilla,
                    "slope_vanilla": slope_vanilla,
                    "rs_by_sigma": {}
                }

                # Per-sigma2 entries
                for s2, rs_dict in rs_rewards_by_sigma.items():
                    r_clean_rs = float(rs_dict["clean"][model_name][k])
                    r_pert_rs = float(rs_dict["pert"][model_name][k])
                    diff_rs = abs(r_clean_rs - r_pert_rs)
                    slope_rs = diff_rs / dx_k
                    model_entry["rs_by_sigma"][str(s2)] = {
                        "r_clean_rs": r_clean_rs,
                        "r_pert_rs": r_pert_rs,
                        "delta_r_rs": diff_rs,
                        "slope_rs": slope_rs
                    }

                pair_entry["models"][model_name] = model_entry

            all_pairs_data.append(pair_entry)

        # Save the checkpoint after every prompt
        completed_set.add(str(global_idx))
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "completed_prompts": list(completed_set),
                "pairs_data": all_pairs_data
            }, f)

        # Periodically free the GPU cache
        if local_idx % 5 == 0:
            torch.cuda.empty_cache()

    print(f"\n[Shard {args.shard_id}] Completed {len(active_prompts)} prompts. Checkpoint: {ckpt_path}")
    return all_pairs_data


# ======================================================================================
# METRICS AGGREGATION & VISUALIZATION
# ======================================================================================
def aggregate_and_plot_results(all_pairs, output_dir, primary_sigma2=1.0):
    """
    Aggregate all pairs into L_max, L_mean, L_95% and reduction ratios, write
    lipschitz_summary.csv, lipschitz_raw_pairs.csv and lipschitz_metrics.json,
    and draw the 3-panel comparison figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_pairs:
        print("Warning: no data to aggregate.")
        return

    # Reward models present in the data
    available_models = list(all_pairs[0]["models"].keys())
    # sigma2 values present in the data
    sample_model = available_models[0]
    available_sigmas = sorted([float(s) for s in all_pairs[0]["models"][sample_model]["rs_by_sigma"].keys()])

    if primary_sigma2 not in available_sigmas:
        # Fall back to the largest available sigma2 if primary_sigma2 is missing
        non_zero_sigmas = [s for s in available_sigmas if s > 0]
        primary_sigma2 = non_zero_sigmas[-1] if non_zero_sigmas else available_sigmas[0]

    print(f"\nAggregating {len(all_pairs)} pairs ({len(available_models)} models, primary sigma2={primary_sigma2})...")

    summary_rows = []
    metrics_json = {
        "num_pairs": len(all_pairs),
        "primary_sigma2": primary_sigma2,
        "models": {},
        "sigma_sweep": {}
    }

    # Flat per-pair table for CSV export
    raw_flat_rows = []
    for p in all_pairs:
        base_row = {
            "prompt_idx": p["prompt_idx"],
            "particle_idx": p["particle_idx"],
            "prompt": p["prompt"][:80],
            "delta_x": p["delta_x"]
        }
        for m in available_models:
            m_info = p["models"].get(m, {})
            base_row[f"{m}_slope_vanilla"] = m_info.get("slope_vanilla", 0.0)
            base_row[f"{m}_delta_r_vanilla"] = m_info.get("delta_r_vanilla", 0.0)

            # Values at primary_sigma2
            rs_info = m_info.get("rs_by_sigma", {}).get(str(primary_sigma2), {})
            base_row[f"{m}_slope_rs_{primary_sigma2}"] = rs_info.get("slope_rs", 0.0)
            base_row[f"{m}_delta_r_rs_{primary_sigma2}"] = rs_info.get("delta_r_rs", 0.0)
        raw_flat_rows.append(base_row)

    # Write the detailed per-pair CSV
    import pandas as pd
    df_raw = pd.DataFrame(raw_flat_rows)
    raw_csv_path = os.path.join(output_dir, "lipschitz_raw_pairs.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Saved per-pair data: {raw_csv_path}")

    # Per-model statistics at primary_sigma2
    for model_name in available_models:
        vanilla_slopes = []
        rs_slopes = []
        dx_vals = []

        for p in all_pairs:
            m_info = p["models"].get(model_name, {})
            if "slope_vanilla" in m_info:
                vanilla_slopes.append(m_info["slope_vanilla"])
                rs_info = m_info.get("rs_by_sigma", {}).get(str(primary_sigma2), {})
                rs_slopes.append(rs_info.get("slope_rs", 0.0))
                dx_vals.append(p["delta_x"])

        v_arr = np.array(vanilla_slopes)
        rs_arr = np.array(rs_slopes)

        # Summary statistics
        l_max_v = float(np.max(v_arr))
        l_max_rs = float(np.max(rs_arr))
        l_mean_v = float(np.mean(v_arr))
        l_mean_rs = float(np.mean(rs_arr))
        l_med_v = float(np.median(v_arr))
        l_med_rs = float(np.median(rs_arr))
        l_p95_v = float(np.percentile(v_arr, 95))
        l_p95_rs = float(np.percentile(rs_arr, 95))

        ratio_max = l_max_v / max(1e-9, l_max_rs)
        ratio_mean = l_mean_v / max(1e-9, l_mean_rs)
        ratio_p95 = l_p95_v / max(1e-9, l_p95_rs)

        summary_rows.append({
            "Reward Model": model_name,
            "L_max (Vanilla)": f"{l_max_v:.5f}",
            "L_max (RS-LiDAR)": f"{l_max_rs:.5f}",
            "Max Reduction Ratio (↑)": f"{ratio_max:.2f}x",
            "L_mean (Vanilla)": f"{l_mean_v:.5f}",
            "L_mean (RS-LiDAR)": f"{l_mean_rs:.5f}",
            "Mean Reduction Ratio (↑)": f"{ratio_mean:.2f}x",
            "L_95% (Vanilla)": f"{l_p95_v:.5f}",
            "L_95% (RS-LiDAR)": f"{l_p95_rs:.5f}",
            "95% Reduction Ratio (↑)": f"{ratio_p95:.2f}x",
        })

        metrics_json["models"][model_name] = {
            "L_max_vanilla": l_max_v, "L_max_rs": l_max_rs, "reduction_ratio_max": ratio_max,
            "L_mean_vanilla": l_mean_v, "L_mean_rs": l_mean_rs, "reduction_ratio_mean": ratio_mean,
            "L_p95_vanilla": l_p95_v, "L_p95_rs": l_p95_rs, "reduction_ratio_p95": ratio_p95,
            "L_median_vanilla": l_med_v, "L_median_rs": l_med_rs
        }

    # Summary table: CSV and console
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "lipschitz_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 120)
    print("EMPIRICAL LIPSCHITZ CONSTANTS: LiDAR vs. RS-LiDAR")
    print("=" * 120)
    print(df_summary.to_string(index=False))
    print("=" * 120)

    # sigma2 sweep statistics (ablation)
    sweep_table_rows = []
    for s2 in available_sigmas:
        row = {"Sigma2": s2}
        for m in available_models:
            s_slopes = []
            for p in all_pairs:
                rs_dict = p["models"].get(m, {}).get("rs_by_sigma", {}).get(str(s2), {})
                s_slopes.append(rs_dict.get("slope_rs", 0.0))
            arr = np.array(s_slopes)
            row[f"{m}_L_max"] = float(np.max(arr))
            row[f"{m}_L_mean"] = float(np.mean(arr))
        sweep_table_rows.append(row)
        metrics_json["sigma_sweep"][str(s2)] = row

    df_sweep = pd.DataFrame(sweep_table_rows)
    sweep_csv_path = os.path.join(output_dir, "lipschitz_sigma_ablation.csv")
    df_sweep.to_csv(sweep_csv_path, index=False)

    # Save metrics JSON
    with open(os.path.join(output_dir, "lipschitz_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    # ==================================================================================
    # 3-PANEL FIGURE
    # ==================================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    # Panel A: bar chart of L_max
    ax_a = axes[0]
    n_mods = len(available_models)
    indices = np.arange(n_mods)
    width = 0.35

    l_max_v_list = [metrics_json["models"][m]["L_max_vanilla"] for m in available_models]
    l_max_rs_list = [metrics_json["models"][m]["L_max_rs"] for m in available_models]

    ax_a.bar(indices - width/2, l_max_v_list, width, label="LiDAR", color="#E63946", alpha=0.85, edgecolor="black")
    ax_a.bar(indices + width/2, l_max_rs_list, width, label=f"RS-LiDAR ($\\sigma$={primary_sigma2})", color="#2A9D8F", alpha=0.85, edgecolor="black")

    ax_a.set_ylabel("Empirical Lipschitz Constant $L_{\\max}$", fontsize=12, fontweight="bold")
    ax_a.set_title("Panel A: Worst-Case Empirical Bound ($L_{\\max}$)", fontsize=13, fontweight="bold", pad=12)
    ax_a.set_xticks(indices)
    ax_a.set_xticklabels(available_models, fontsize=11, fontweight="bold")
    ax_a.set_ylim(0, max(l_max_v_list) * 1.08)
    ax_a.legend(frameon=True, fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel B: KDE of the slope distribution
    ax_b = axes[1]
    # Primary model for illustration (ImageReward if available, otherwise the first model)
    primary_model = "ImageReward" if "ImageReward" in available_models else available_models[0]
    v_slopes = [p["models"][primary_model]["slope_vanilla"] for p in all_pairs if primary_model in p["models"]]
    rs_slopes = [p["models"][primary_model]["rs_by_sigma"][str(primary_sigma2)]["slope_rs"] for p in all_pairs if primary_model in p["models"]]

    import scipy.stats as stats
    kde_v = stats.gaussian_kde(v_slopes)
    kde_rs = stats.gaussian_kde(rs_slopes)
    max_x = max(np.percentile(v_slopes, 98), np.percentile(rs_slopes, 98)) * 1.2
    x_grid = np.linspace(0, max_x, 300)

    ax_b.plot(x_grid, kde_v(x_grid), label="LiDAR (Heavy-tailed spikes)", color="#E63946", lw=2.5)
    ax_b.fill_between(x_grid, kde_v(x_grid), color="#E63946", alpha=0.25)
    ax_b.plot(x_grid, kde_rs(x_grid), label="RS-LiDAR (Flattened landscape)", color="#2A9D8F", lw=2.5)
    ax_b.fill_between(x_grid, kde_rs(x_grid), color="#2A9D8F", alpha=0.35)

    ax_b.set_xlabel("Secant Slope $\\Delta R / \\Delta x$", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax_b.set_title(f"Panel B: Slope Density Distribution ({primary_model})", fontsize=13, fontweight="bold", pad=12)
    ax_b.legend(frameon=True, fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # Panel C: per-pair scatter against the diagonal y = x
    ax_c = axes[2]
    ax_c.scatter(v_slopes, rs_slopes, color="#457B9D", alpha=0.6, edgecolors="none", s=28, label="Sample Pairs")
    diag_max = max(max(v_slopes), max(rs_slopes)) * 1.05
    ax_c.plot([0, diag_max], [0, diag_max], linestyle="--", color="#E63946", lw=2, label="Parity ($y = x$)")

    ax_c.set_xlabel("LiDAR Slope ($|\\Delta R| / \\Delta x$)", fontsize=12, fontweight="bold")
    ax_c.set_ylabel(f"RS-LiDAR Slope ($\\sigma={primary_sigma2}$)", fontsize=12, fontweight="bold")
    ax_c.set_title("Panel C: Pairwise Slope Contraction", fontsize=13, fontweight="bold", pad=12)
    ax_c.legend(frameon=True, fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.set_xlim(0, diag_max)
    ax_c.set_ylim(0, diag_max)

    # Panel C annotation
    ax_c.text(0.6 * diag_max, 0.2 * diag_max, "Flattening Region\n(RS-LiDAR < LiDAR)",
              fontsize=10, fontweight="bold", color="#2A9D8F", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2A9D8F", alpha=0.9))

    plot_path = os.path.join(output_dir, "lipschitz_comparison_3panel.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-panel figure: {plot_path}")

    # Extra figure: Lipschitz bound vs. sigma2 (when at least 3 sigma values were swept)
    if len(available_sigmas) >= 3:
        fig_abl, ax_abl = plt.subplots(figsize=(8, 5), dpi=300)
        colors = ["#E63946", "#2A9D8F", "#457B9D", "#E76F51"]
        for idx, m in enumerate(available_models):
            y_vals = [metrics_json["sigma_sweep"][str(s)][f"{m}_L_max"] for s in available_sigmas]
            ax_abl.plot(available_sigmas, y_vals, marker="o", lw=2.2, color=colors[idx % len(colors)], label=f"{m} ($L_{{\\max}}$)")

        ax_abl.set_xlabel("Smoothing Radius $\\sigma_2$", fontsize=12, fontweight="bold")
        ax_abl.set_ylabel("Empirical Lipschitz Bound $L_{\\max}$", fontsize=12, fontweight="bold")
        ax_abl.set_title("Decay of Lipschitz Bound vs. Smoothing Scale $\\sigma_2$", fontsize=13, fontweight="bold")
        ax_abl.grid(True, linestyle="--", alpha=0.6)
        ax_abl.legend(frameon=True)
        abl_plot_path = os.path.join(output_dir, "lipschitz_sigma_ablation.png")
        plt.savefig(abl_plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved sigma2 sweep figure: {abl_plot_path}")

    # Additional figures: per-sigma 3-panel plots (0.1, 0.25, 0.5, 1.0)
    if any(s in available_sigmas for s in [0.1, 0.25, 0.5, 1.0]):
        try:
            plot_enhanced_lipschitz_figures(all_pairs, output_dir, available_models, available_sigmas, metrics_json)
        except Exception as e:
            print(f"Warning: could not draw the additional figures: {e}")


def plot_3panel_for_sigma(all_pairs, output_dir, available_models, target_sigma2, filename=None):
    """
    Draw the 3-panel figure for a single sigma2:
    - Panel A: L_max bar chart.
    - Panel B: KDE of the slope distribution for the primary model.
    - Panel C: per-pair scatter against the diagonal y = x (pairwise slope contraction).
    """
    import scipy.stats as stats
    s2_t = float(target_sigma2)
    indices = np.arange(len(available_models))
    width = 0.35

    l_max_v_list = []
    l_max_rs_list = []
    ratios = []
    for m in available_models:
        v_s = [p["models"][m]["slope_vanilla"] for p in all_pairs if m in p["models"]]
        rs_s = [p["models"][m]["rs_by_sigma"][str(s2_t)]["slope_rs"] for p in all_pairs if m in p["models"] and str(s2_t) in p["models"][m].get("rs_by_sigma", {})]
        lm_v = float(np.max(v_s)) if v_s else 0.0
        lm_rs = float(np.max(rs_s)) if rs_s else 0.0
        l_max_v_list.append(lm_v)
        l_max_rs_list.append(lm_rs)
        ratios.append(lm_v / max(1e-9, lm_rs))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    # Panel A: Bar chart L_max
    ax_a = axes[0]
    ax_a.bar(indices - width/2, l_max_v_list, width, label="LiDAR", color="#E63946", alpha=0.9, edgecolor="black")
    ax_a.bar(indices + width/2, l_max_rs_list, width, label=f"RS-LiDAR ($\\sigma$={s2_t})", color="#2A9D8F", alpha=0.9, edgecolor="black")
    ax_a.set_ylabel("Empirical Lipschitz Constant $L_{\\max}$", fontsize=12, fontweight="bold")
    ax_a.set_title(f"Panel A: Worst-Case Bound ($L_{{\\max}}$, $\\sigma={s2_t}$)", fontsize=13, fontweight="bold", pad=12)
    ax_a.set_xticks(indices)
    ax_a.set_xticklabels(available_models, fontsize=11, fontweight="bold")
    ax_a.legend(frameon=True, fontsize=11)
    ax_a.set_ylim(0, max(l_max_v_list) * 1.08)
    ax_a.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel B: KDE
    ax_b = axes[1]
    pm = "ImageReward" if "ImageReward" in available_models else available_models[0]
    v_slopes_pm = [p["models"][pm]["slope_vanilla"] for p in all_pairs if pm in p["models"]]
    rs_slopes_pm = [p["models"][pm]["rs_by_sigma"][str(s2_t)]["slope_rs"] for p in all_pairs if pm in p["models"] and str(s2_t) in p["models"][pm].get("rs_by_sigma", {})]

    kde_v = stats.gaussian_kde(v_slopes_pm)
    kde_rs = stats.gaussian_kde(rs_slopes_pm)
    max_x = max(np.percentile(v_slopes_pm, 98), np.percentile(rs_slopes_pm, 98)) * 1.2
    x_grid = np.linspace(0, max_x, 300)

    ax_b.plot(x_grid, kde_v(x_grid), label="LiDAR (Heavy-tailed spikes)", color="#E63946", lw=2.5)
    ax_b.fill_between(x_grid, kde_v(x_grid), color="#E63946", alpha=0.25)
    ax_b.plot(x_grid, kde_rs(x_grid), label=f"RS-LiDAR ($\\sigma$={s2_t})", color="#2A9D8F", lw=2.5)
    ax_b.fill_between(x_grid, kde_rs(x_grid), color="#2A9D8F", alpha=0.35)
    ax_b.set_xlabel("Secant Slope $|\\Delta R| / \\|\\Delta x\\|_2$", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax_b.set_title(f"Panel B: Slope Density Distribution ({pm})", fontsize=13, fontweight="bold", pad=12)
    ax_b.legend(frameon=True, fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Scatter plot
    ax_c = axes[2]
    ax_c.scatter(v_slopes_pm, rs_slopes_pm, color="#2A9D8F", alpha=0.6, edgecolors="none", s=28, label=f"Pairs ($\\sigma$={s2_t})")
    diag_max = max(max(v_slopes_pm), max(rs_slopes_pm)) * 1.05
    ax_c.plot([0, diag_max], [0, diag_max], linestyle="--", color="#E63946", lw=2, label="Parity ($y = x$)")
    ax_c.set_xlabel("LiDAR Slope ($|\\Delta R| / \\|\\Delta x\\|_2$)", fontsize=12, fontweight="bold")
    ax_c.set_ylabel(f"RS-LiDAR Slope ($\\sigma={s2_t}$)", fontsize=12, fontweight="bold")
    ax_c.set_title(f"Panel C: Pairwise Slope Contraction ($\\sigma={s2_t}$)", fontsize=13, fontweight="bold", pad=12)
    ax_c.legend(frameon=True, fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.set_xlim(0, diag_max)
    ax_c.set_ylim(0, diag_max)

    if filename is None:
        filename = f"lipschitz_comparison_3panel_sigma_{s2_t}.png"
    p_path = os.path.join(output_dir, filename)
    plt.savefig(p_path, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-panel figure (sigma={s2_t}): {p_path}")
    return p_path


def plot_enhanced_lipschitz_figures(all_pairs, output_dir, available_models, available_sigmas, metrics_json):
    """
    Draw one 3-panel Lipschitz figure per sigma in {0.1, 0.25, 0.5, 1.0}.
    """
    # One 3-panel figure per sigma: 0.1, 0.25, 0.5, 1.0
    for s2_target in [0.1, 0.25, 0.5, 1.0]:
        if s2_target in available_sigmas:
            plot_3panel_for_sigma(all_pairs, output_dir, available_models, s2_target)

# ======================================================================================
# MERGE SHARDS
# ======================================================================================
def merge_shard_checkpoints(output_dir, primary_sigma2=1.0):
    """
    Merge all checkpoint_shard_*.json files in output_dir into one dataset, then compute
    the metrics and draw the figures.
    """
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint_shard_*.json")))
    if not ckpts:
        print(f"No checkpoint files found in {output_dir}")
        return

    print(f"Found {len(ckpts)} shard checkpoints: {[os.path.basename(c) for c in ckpts]}")
    merged_pairs = []
    seen_prompts = set()

    for c_path in ckpts:
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            pairs = c_data.get("pairs_data", [])
            for p in pairs:
                p_key = (p.get("prompt_idx"), p.get("particle_idx"))
                if p_key not in seen_prompts:
                    seen_prompts.add(p_key)
                    merged_pairs.append(p)
        except Exception as e:
            print(f"Warning: could not read {c_path}: {e}")

    print(f"Merged {len(merged_pairs)} unique pairs.")
    aggregate_and_plot_results(merged_pairs, output_dir, primary_sigma2=primary_sigma2)


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Empirical Lipschitz constant: LiDAR vs. RS-LiDAR")

    # Model & data
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Diffusion model used to generate images")
    parser.add_argument("--prompts_file", type=str, default="prompt_files/geneval_50_stratified.jsonl", help="Prompt file (.jsonl)")
    parser.add_argument("--num_prompts", type=int, default=50, help="Number of prompts")
    parser.add_argument("--num_particles", type=int, default=10, help="Clean images generated per prompt (K)")
    parser.add_argument("--guidance_steps", type=int, default=5, help="DPM-Solver steps used to generate images")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Lipschitz probe & noise
    parser.add_argument("--sigma1", type=float, default=0.1, help="Std of the perturbation that produces x_pert (probe scale)")
    parser.add_argument("--perturbation_type", type=str, default="gaussian", choices=["gaussian", "blur"], help="Perturbation type")
    parser.add_argument("--sigma2", type=float, default=1.0, help="Primary RS-LiDAR smoothing radius")
    parser.add_argument("--sigma2_list", type=str, default="0.0,0.1,0.25,0.5,1.0", help="Comma-separated sigma2 values to sweep")
    parser.add_argument("--num_mc_samples", "-M", type=int, default=4, help="Monte Carlo samples for the smoothed reward")
    parser.add_argument("--use_crn", action="store_true", default=True, help="Use common random numbers (shared noise) for x_clean and x_pert to reduce Monte Carlo variance")
    parser.add_argument("--no_crn", action="store_false", dest="use_crn", help="Disable CRN: independent noise for x_clean and x_pert")

    # Reward models (--use_* flags and aliases --ImageReward, --ClipScore, --HPS, --Aesthetic, --GenEval)
    parser.add_argument("--reward_models", type=str, default=None, help="Comma-separated reward models (e.g. 'ImageReward,CLIP-Score,HPS')")
    parser.add_argument("--use_imagereward", "--ImageReward", action="store_true", default=True, help="Evaluate ImageReward")
    parser.add_argument("--no_imagereward", action="store_false", dest="use_imagereward")
    parser.add_argument("--use_clip", "--ClipScore", action="store_true", default=True, help="Evaluate CLIP-Score")
    parser.add_argument("--no_clip", action="store_false", dest="use_clip")
    parser.add_argument("--use_hps", "--HPS", action="store_true", default=True, help="Evaluate HPS v2.1")
    parser.add_argument("--no_hps", action="store_false", dest="use_hps")
    parser.add_argument("--use_aesthetic", "--Aesthetic", action="store_true", default=True, help="Evaluate Aesthetic Score")
    parser.add_argument("--no_aesthetic", action="store_false", dest="use_aesthetic")
    parser.add_argument("--use_geneval", "--GenEval", action="store_true", default=False, help="Evaluate GenEval consistency / detection score")

    # Multi-GPU sharding & output
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index (0, 1, ...)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index for this process")
    parser.add_argument("--output_dir", type=str, default="results/lipschitz_empirical", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoints and start from scratch")
    parser.add_argument("--merge_shards", action="store_true", help="Merge shard checkpoints and draw the figures")

    args = parser.parse_args()

    # Parse --reward_models if given
    if args.reward_models:
        models_lower = [m.strip().lower() for m in args.reward_models.split(",") if m.strip()]
        args.use_imagereward = any("image" in m for m in models_lower)
        args.use_clip = any("clip" in m for m in models_lower)
        args.use_hps = any("hps" in m for m in models_lower)
        args.use_aesthetic = any("aesthetic" in m or "as" == m for m in models_lower)
        args.use_geneval = any("geneval" in m for m in models_lower)

    # Merge-only mode
    if args.merge_shards:
        merge_shard_checkpoints(args.output_dir, primary_sigma2=args.sigma2)
        return

    # Device setup
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    # 1. Load prompts
    prompts = load_stratified_prompts(args.prompts_file, max_prompts=args.num_prompts, seed=args.seed)
    print(f"Prepared {len(prompts)} prompts.")

    # 2. Load the Stable Diffusion pipeline (DPM-Solver, few steps)
    print(f"Loading {args.model_name}...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        scheduler=scheduler,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae

    # 3. Load ImageReward if enabled
    ir_model = None
    if args.use_imagereward and rm_load is not None:
        try:
            print("Loading ImageReward...")
            ir_model = rm_load("ImageReward-v1.0")
            print("ImageReward loaded.")
        except Exception as e:
            print(f"Warning: could not load ImageReward ({e}); skipping it.")
            ir_model = None

    # 4. Run the Lipschitz measurement
    all_pairs = run_empirical_lipschitz_test(pipe, vae, ir_model, prompts, args, device=device)

    # 5. Single-shard runs aggregate and plot immediately
    if args.num_shards == 1:
        aggregate_and_plot_results(all_pairs, args.output_dir, primary_sigma2=args.sigma2)


if __name__ == "__main__":
    main()
