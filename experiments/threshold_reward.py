"""
Empirical Lipschitz constant of a threshold (step-function) reward based on OWL-ViT counting.

Reward: 1.0 if the number of detected objects equals the count requested in the prompt, else 0.0.

The experiment:
1. Shows that the threshold reward is not Lipschitz (finite-difference slopes diverge as the
   perturbation radius shrinks).
2. Measures how randomized smoothing (RS-LiDAR) turns it into a Lipschitz-continuous reward.
3. Exports slope and reward distributions, sweep tables over sigma1 / sigma2 / M, and dashboards.

Usage (from the repository root):
  python -m experiments.threshold_reward --num_particles 10
"""

import os
import json
import glob
import argparse
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Environment
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# transformers compatibility shims (local variant: also stubs DynamicCache / Cache)
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

import torch.nn.functional as F
try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
except ImportError:
    DPMSolverMultistepScheduler = None
    StableDiffusionPipeline = None

from transformers import OwlViTProcessor, OwlViTForObjectDetection


# ======================================================================================
# Counting prompt set
# ======================================================================================
COUNTING_PROMPTS = [
    {"prompt": "two apples on a wooden table", "object": "apple", "count": 2},
    {"prompt": "three dogs playing in the park", "object": "dog", "count": 3},
    {"prompt": "four cars parked on the street", "object": "car", "count": 4},
    {"prompt": "one cat sleeping on a sofa", "object": "cat", "count": 1},
    {"prompt": "five birds flying in the sky", "object": "bird", "count": 5},
    {"prompt": "two laptops on the desk", "object": "laptop", "count": 2},
    {"prompt": "three coffee cups on the table", "object": "coffee cup", "count": 3},
    {"prompt": "four chairs around a dining table", "object": "chair", "count": 4},
    {"prompt": "two books on the shelf", "object": "book", "count": 2},
    {"prompt": "three horses in the field", "object": "horse", "count": 3},
    {"prompt": "one airplane in the blue sky", "object": "airplane", "count": 1},
    {"prompt": "two bicycles leaning against a wall", "object": "bicycle", "count": 2},
    {"prompt": "three people walking down the street", "object": "person", "count": 3},
    {"prompt": "four potted plants by the window", "object": "potted plant", "count": 4},
    {"prompt": "five balloons tied to a fence", "object": "balloon", "count": 5},
    {"prompt": "two buses on the road", "object": "bus", "count": 2},
    {"prompt": "three motorcycles in the parking lot", "object": "motorcycle", "count": 3},
    {"prompt": "four boats on the lake", "object": "boat", "count": 4},
    {"prompt": "five trees in the garden", "object": "tree", "count": 5},
    {"prompt": "two ducks in the pond", "object": "duck", "count": 2},
    {"prompt": "three cows in the meadow", "object": "cow", "count": 3},
    {"prompt": "four sheep on the hill", "object": "sheep", "count": 4},
    {"prompt": "one train at the station", "object": "train", "count": 1},
    {"prompt": "two elephants in the zoo", "object": "elephant", "count": 2},
    {"prompt": "three monkeys on a tree", "object": "monkey", "count": 3},
    {"prompt": "four lions in the savanna", "object": "lion", "count": 4},
    {"prompt": "five penguins on the ice", "object": "penguin", "count": 5},
    {"prompt": "two pizzas on the table", "object": "pizza", "count": 2},
    {"prompt": "three burgers on a plate", "object": "burger", "count": 3},
    {"prompt": "four sandwiches in the basket", "object": "sandwich", "count": 4},
    {"prompt": "five hotdogs on the grill", "object": "hotdog", "count": 5},
    {"prompt": "two donuts in a box", "object": "donut", "count": 2},
    {"prompt": "three cakes on display", "object": "cake", "count": 3},
    {"prompt": "four apples in a bowl", "object": "apple", "count": 4},
    {"prompt": "five oranges on the counter", "object": "orange", "count": 5},
    {"prompt": "two bananas in a bunch", "object": "banana", "count": 2},
    {"prompt": "three strawberries on a plate", "object": "strawberry", "count": 3},
    {"prompt": "four grapes on a vine", "object": "grape", "count": 4},
    {"prompt": "five watermelons in a cart", "object": "watermelon", "count": 5},
    {"prompt": "two carrots on a cutting board", "object": "carrot", "count": 2},
    {"prompt": "three potatoes in a sack", "object": "potato", "count": 3},
    {"prompt": "four tomatoes in a basket", "object": "tomato", "count": 4},
    {"prompt": "five onions in a bowl", "object": "onion", "count": 5},
    {"prompt": "two peppers on a grill", "object": "pepper", "count": 2},
    {"prompt": "three mushrooms in a pan", "object": "mushroom", "count": 3},
    {"prompt": "four cucumbers in a salad", "object": "cucumber", "count": 4},
    {"prompt": "five lemons in a basket", "object": "lemon", "count": 5},
    {"prompt": "two smartphones on the desk", "object": "smartphone", "count": 2},
    {"prompt": "three tablets on the table", "object": "tablet", "count": 3},
    {"prompt": "four monitors on the wall", "object": "monitor", "count": 4}
]

# ======================================================================================
# Threshold (binary) reward with OWL-ViT
# ======================================================================================
@torch.inference_mode()
def evaluate_threshold_reward(
    images, target_object, target_count, processor, model, device="cuda"
):
    """
    Score a batch of images with OWL-ViT.
    Returns a float array [N] with 1.0 (correct count) or 0.0 (wrong count).
    """
    N = len(images)
    texts = [[f"a photo of a {target_object}"]] * N
    inputs = processor(text=texts, images=images, return_tensors="pt").to(device)
    
    outputs = model(**inputs)
    
    rewards = []
    for i in range(N):
        # Logits of the i-th image
        logits = outputs.logits[i] # [num_boxes, num_classes]
        probs = logits.sigmoid().max(dim=-1).values
        # Count boxes with probability > 0.1
        count = (probs > 0.1).sum().item()
        # Binary (step) reward
        rewards.append(1.0 if count == target_count else 0.0)
        
    return np.array(rewards)


# ======================================================================================
# Generation and image processing
# ======================================================================================
@torch.inference_mode()
def generate_clean_and_perturbed_images(
    pipe, vae, prompt, num_particles, guidance_steps, guidance_scale,
    sigma1, perturbation_type="gaussian", seed=42, device="cuda"
):
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe(
        [prompt] * num_particles,
        num_inference_steps=guidance_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent"
    ).images

    scaled = (latents / vae.config.scaling_factor).to(device=device, dtype=vae.dtype)
    clean_tensors = vae.decode(scaled, return_dict=False)[0].clamp(-1.0, 1.0)

    if perturbation_type == "gaussian":
        noise = torch.randn_like(clean_tensors) * sigma1
        pert_tensors = (clean_tensors + noise).clamp(-1.0, 1.0)
    elif perturbation_type == "blur":
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

    diff = clean_tensors - pert_tensors
    delta_x = torch.norm(diff.view(num_particles, -1), p=2, dim=1).cpu().numpy()

    clean_pils = pipe.image_processor.postprocess(clean_tensors, output_type="pil")
    pert_pils = pipe.image_processor.postprocess(pert_tensors, output_type="pil")

    return clean_tensors, pert_tensors, delta_x, clean_pils, pert_pils


# ======================================================================================
# Core experiment: empirical Lipschitz constants
# ======================================================================================
def run_empirical_lipschitz_test(
    pipe, vae, owl_processor, owl_model, prompt_list, args, device="cuda"
):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"checkpoint_threshold_shard_{args.shard_id}.json")

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

    # Probe radii sigma1 (sweep)
    primary_sigma1 = getattr(args, "primary_sigma1", None)
    if primary_sigma1 is None:
        primary_sigma1 = args.sigma1

    if getattr(args, "sigma1_list", None):
        try:
            sigma1_vals = [float(s.strip()) for s in args.sigma1_list.split(",") if s.strip()]
        except Exception:
            sigma1_vals = [primary_sigma1]
    else:
        sigma1_vals = [primary_sigma1]

    if primary_sigma1 not in sigma1_vals:
        sigma1_vals.append(primary_sigma1)
    sigma1_vals = sorted(list(set(sigma1_vals)))

    # Smoothing radii sigma2
    if args.sigma2_list:
        try:
            sigma2_vals = [float(s.strip()) for s in args.sigma2_list.split(",") if s.strip()]
        except Exception:
            sigma2_vals = [0.0, 0.1, 0.25, 0.5, 1.0] # default list
    else:
        sigma2_vals = [args.sigma2]

    if not any(s > 0 for s in sigma2_vals):
        sigma2_vals.append(1.0)
    sigma2_vals = sorted(list(set(sigma2_vals)))

    primary_sigma2 = getattr(args, "sigma2", 0.5)
    if primary_sigma2 not in sigma2_vals:
        non_zero = [s for s in sigma2_vals if s > 0]
        primary_sigma2 = non_zero[-1] if non_zero else sigma2_vals[0]

    # Monte Carlo sample counts M (sweep)
    if getattr(args, "m_list", None):
        try:
            m_vals = [int(m.strip()) for m in args.m_list.split(",") if m.strip()]
        except Exception:
            m_vals = [1, 2, 4, 8]
    else:
        m_vals = [args.num_mc_samples]
    m_vals = sorted(list(set(m_vals)))
    max_m = max(m_vals)
    primary_m = getattr(args, "primary_m", 4)
    if primary_m not in m_vals:
        primary_m = m_vals[-1]

    print(f"\n[Shard {args.shard_id}/{args.num_shards}] Threshold reward experiment on {len(active_prompts)} prompts")
    print(f"Parameters: {args.num_particles} particles/prompt | sigma1={primary_sigma1} (sweep={sigma1_vals})")
    print(f"sigma2 sweep: {sigma2_vals} (primary={primary_sigma2}) | M sweep: {m_vals} (max={max_m}, primary={primary_m})")

    saved_data = {"completed_prompts": [], "pairs_data": []}
    if os.path.exists(ckpt_path) and not args.overwrite:
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            print(f"Loaded checkpoint: {len(saved_data.get('completed_prompts', []))} prompts already completed.")
        except Exception as e:
            print(f"Warning: could not read checkpoint: {e}. Starting from scratch.")

    completed_set = set(saved_data.get("completed_prompts", []))
    all_pairs_data = saved_data.get("pairs_data", [])

    pbar = tqdm(enumerate(active_prompts), total=len(active_prompts), desc=f"Shard {args.shard_id} Progress")

    for local_idx, prompt_data in pbar:
        global_idx = start_idx + local_idx
        prompt = prompt_data["prompt"]
        target_obj = prompt_data["object"]
        target_count = prompt_data["count"]
        
        if str(global_idx) in completed_set or prompt in completed_set:
            continue

        prompt_seed = args.seed + global_idx * 100

        # 1. Generate clean images x_clean and perturbed images x_pert (at primary_sigma1)
        clean_tensors, pert_tensors, delta_x, clean_pils, pert_pils = generate_clean_and_perturbed_images(
            pipe=pipe,
            vae=vae,
            prompt=prompt,
            num_particles=args.num_particles,
            guidance_steps=args.guidance_steps,
            guidance_scale=args.guidance_scale,
            sigma1=primary_sigma1,
            perturbation_type=args.perturbation_type,
            seed=prompt_seed,
            device=device
        )

        # 2. Vanilla rewards R(x_clean) and R(x_pert)
        vanilla_clean_rewards = evaluate_threshold_reward(
            clean_pils, target_obj, target_count, owl_processor, owl_model, device=device
        )
        vanilla_pert_rewards = evaluate_threshold_reward(
            pert_pils, target_obj, target_count, owl_processor, owl_model, device=device
        )

        # 3. Monte Carlo RS-LiDAR rewards for each sigma2 and each M
        # Draw max_m samples once and take prefix means for every M in m_vals
        rs_rewards_by_sigma_m = {} 
        use_crn = getattr(args, "use_crn", True)
        primary_u_m_list = [] # noise vectors of primary_sigma2, reused by the sigma1 sweep

        for s2 in sigma2_vals:
            rs_rewards_by_sigma_m[s2] = {}
            if s2 == 0.0:
                for M in m_vals:
                    rs_rewards_by_sigma_m[s2][M] = {
                        "clean": vanilla_clean_rewards.tolist(),
                        "pert": vanilla_pert_rewards.tolist()
                    }
                continue

            clean_samples = []
            pert_samples = []
            cur_u_m_list = []

            for m_iter in range(max_m):
                u_m = torch.randn_like(clean_tensors) * s2
                if s2 == primary_sigma2:
                    cur_u_m_list.append(u_m)

                # Clean image + noise
                noisy_clean_t = (clean_tensors + u_m).clamp(-1.0, 1.0)
                noisy_clean_pils = pipe.image_processor.postprocess(noisy_clean_t, output_type="pil")
                r_m_clean = evaluate_threshold_reward(
                    noisy_clean_pils, target_obj, target_count, owl_processor, owl_model, device=device
                )
                clean_samples.append(r_m_clean)

                # Perturbed image + noise (common random numbers if use_crn=True)
                u_prime_m = u_m if use_crn else (torch.randn_like(pert_tensors) * s2)
                noisy_pert_t = (pert_tensors + u_prime_m).clamp(-1.0, 1.0)
                noisy_pert_pils = pipe.image_processor.postprocess(noisy_pert_t, output_type="pil")
                r_m_pert = evaluate_threshold_reward(
                    noisy_pert_pils, target_obj, target_count, owl_processor, owl_model, device=device
                )
                pert_samples.append(r_m_pert)

            if s2 == primary_sigma2:
                primary_u_m_list = cur_u_m_list

            clean_samples_arr = np.array(clean_samples) # [max_m, num_particles]
            pert_samples_arr = np.array(pert_samples)   # [max_m, num_particles]

            for M in m_vals:
                avg_clean_m = np.mean(clean_samples_arr[:M], axis=0)
                avg_pert_m = np.mean(pert_samples_arr[:M], axis=0)
                rs_rewards_by_sigma_m[s2][M] = {
                    "clean": avg_clean_m.tolist(),
                    "pert": avg_pert_m.tolist()
                }

        # 3b. sigma1 sweep for the primary setting (primary_sigma2, primary_m)
        # Shows that the vanilla slope blows up as sigma1 -> 0 while the RS-LiDAR slope stays bounded
        sigma1_sweep_results = {}
        for s1 in sigma1_vals:
            if s1 == primary_sigma1:
                sigma1_sweep_results[s1] = {
                    "delta_x": delta_x.tolist(),
                    "vanilla_pert": vanilla_pert_rewards.tolist(),
                    "rs_pert": rs_rewards_by_sigma_m[primary_sigma2][primary_m]["pert"]
                }
            else:
                if args.perturbation_type == "gaussian":
                    noise_s1 = torch.randn_like(clean_tensors) * s1
                    pert_tensors_s1 = (clean_tensors + noise_s1).clamp(-1.0, 1.0)
                elif args.perturbation_type == "blur":
                    channels = clean_tensors.shape[1]
                    kernel_size = 5
                    sigma_blur = max(0.5, s1 * 10.0)
                    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
                    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
                    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma_blur**2))
                    kernel = (kernel / torch.sum(kernel)).repeat(channels, 1, 1, 1)
                    pert_tensors_s1 = F.conv2d(clean_tensors, kernel, padding=kernel_size // 2, groups=channels).clamp(-1.0, 1.0)
                else:
                    pert_tensors_s1 = clean_tensors.clone()

                diff_s1 = clean_tensors - pert_tensors_s1
                delta_x_s1 = torch.norm(diff_s1.view(args.num_particles, -1), p=2, dim=1).cpu().numpy()
                pert_pils_s1 = pipe.image_processor.postprocess(pert_tensors_s1, output_type="pil")

                vanilla_pert_s1 = evaluate_threshold_reward(
                    pert_pils_s1, target_obj, target_count, owl_processor, owl_model, device=device
                )

                # RS reward at primary_sigma2 and primary_m
                pert_samples_s1 = []
                for m_idx in range(primary_m):
                    u_m_cur = primary_u_m_list[m_idx] if len(primary_u_m_list) > m_idx else (torch.randn_like(pert_tensors_s1) * primary_sigma2)
                    u_prime_m_cur = u_m_cur if use_crn else (torch.randn_like(pert_tensors_s1) * primary_sigma2)
                    noisy_pert_t_s1 = (pert_tensors_s1 + u_prime_m_cur).clamp(-1.0, 1.0)
                    noisy_pert_pils_s1 = pipe.image_processor.postprocess(noisy_pert_t_s1, output_type="pil")
                    r_m_pert_s1 = evaluate_threshold_reward(
                        noisy_pert_pils_s1, target_obj, target_count, owl_processor, owl_model, device=device
                    )
                    pert_samples_s1.append(r_m_pert_s1)

                avg_pert_s1 = np.mean(pert_samples_s1, axis=0)
                sigma1_sweep_results[s1] = {
                    "delta_x": delta_x_s1.tolist(),
                    "vanilla_pert": vanilla_pert_s1.tolist(),
                    "rs_pert": avg_pert_s1.tolist()
                }

        # 4. Record per-particle results
        for k in range(args.num_particles):
            dx_k = float(delta_x[k])
            if dx_k < 1e-7: dx_k = 1e-7

            r_clean = float(vanilla_clean_rewards[k])
            r_pert = float(vanilla_pert_rewards[k])
            diff_vanilla = abs(r_clean - r_pert)
            slope_vanilla = diff_vanilla / dx_k

            model_entry = {
                "r_clean_vanilla": r_clean,
                "r_pert_vanilla": r_pert,
                "delta_r_vanilla": diff_vanilla,
                "slope_vanilla": slope_vanilla,
                "rs_by_m": {},
                "rs_by_sigma": {},
                "sigma1_sweep": {}
            }

            for M in m_vals:
                model_entry["rs_by_m"][str(M)] = {}
                for s2 in sigma2_vals:
                    r_clean_rs = float(rs_rewards_by_sigma_m[s2][M]["clean"][k])
                    r_pert_rs = float(rs_rewards_by_sigma_m[s2][M]["pert"][k])
                    diff_rs = abs(r_clean_rs - r_pert_rs)
                    slope_rs = diff_rs / dx_k

                    model_entry["rs_by_m"][str(M)][str(s2)] = {
                        "r_clean": r_clean_rs,
                        "r_pert": r_pert_rs,
                        "delta_r": diff_rs,
                        "slope": slope_rs
                    }

            # Backward compatibility: rs_by_sigma maps to primary_m
            model_entry["rs_by_sigma"] = model_entry["rs_by_m"][str(primary_m)]

            # sigma1 sweep results for particle k
            r_clean_rs_p = float(rs_rewards_by_sigma_m[primary_sigma2][primary_m]["clean"][k])
            for s1 in sigma1_vals:
                dx_s1_k = float(sigma1_sweep_results[s1]["delta_x"][k])
                if dx_s1_k < 1e-7: dx_s1_k = 1e-7
                r_pert_v_s1 = float(sigma1_sweep_results[s1]["vanilla_pert"][k])
                diff_v_s1 = abs(r_clean - r_pert_v_s1)
                slope_v_s1 = diff_v_s1 / dx_s1_k

                r_pert_rs_s1 = float(sigma1_sweep_results[s1]["rs_pert"][k])
                diff_rs_s1 = abs(r_clean_rs_p - r_pert_rs_s1)
                slope_rs_s1 = diff_rs_s1 / dx_s1_k

                model_entry["sigma1_sweep"][str(s1)] = {
                    "delta_x": dx_s1_k,
                    "r_clean_vanilla": r_clean,
                    "r_pert_vanilla": r_pert_v_s1,
                    "delta_r_vanilla": diff_v_s1,
                    "slope_vanilla": slope_v_s1,
                    "r_clean_rs": r_clean_rs_p,
                    "r_pert_rs": r_pert_rs_s1,
                    "delta_r_rs": diff_rs_s1,
                    "slope_rs": slope_rs_s1
                }

            pair_entry = {
                "prompt_idx": global_idx,
                "particle_idx": k,
                "prompt": prompt,
                "delta_x": dx_k,
                "models": {"OWL-ViT-Count": model_entry}
            }
            all_pairs_data.append(pair_entry)

        # Checkpoint
        completed_set.add(str(global_idx))
        completed_set.add(prompt)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"completed_prompts": list(completed_set), "pairs_data": all_pairs_data}, f, indent=2)

    print(f"\nShard {args.shard_id} done. Saved {len(all_pairs_data)} pairs to {ckpt_path}.")


# ======================================================================================
# Utility: render a table as a matplotlib image
# ======================================================================================
def render_table_as_image(table_data, col_labels, title, output_path, col_widths=None):
    """Render a table as a high-resolution PNG with zebra striping."""
    fig_w = max(10, len(col_labels) * 1.35)
    fig_h = max(2.8, len(table_data) * 0.45 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12, color='#111827')

    tab = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.0, 1.4)

    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor('#d1d5db')
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor('#1e3a8a') # Navy header
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 1:
                cell.set_facecolor('#f9fafb') # Light gray zebra
            else:
                cell.set_facecolor('#ffffff')
            # Bold first column
            if col == 0 or (col == 1 and "OWL" in table_data[0][0]):
                cell.set_text_props(fontweight='semibold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================================================
# Aggregation, tables and plots
# ======================================================================================
def merge_and_plot_results(output_dir, primary_sigma2=0.5, primary_m=4, primary_sigma1=0.05):
    ckpt_files = glob.glob(os.path.join(output_dir, "checkpoint_threshold_shard_*.json"))
    if not ckpt_files:
        print("Warning: no checkpoint_threshold_shard_*.json files found to merge.")
        return

    merged_pairs = []
    seen = set()
    for c_path in ckpt_files:
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            for p in c_data.get("pairs_data", []):
                p_key = (p.get("prompt_idx"), p.get("particle_idx"))
                if p_key not in seen:
                    seen.add(p_key)
                    merged_pairs.append(p)
        except Exception as e:
            print(f"Warning: could not read {c_path}: {e}")

    print(f"Merged {len(merged_pairs)} unique pairs.")

    # 1. Discover the M, sigma2 and sigma1 values present in the data
    available_ms = set()
    available_sigmas = set()
    available_sigma1s = set()

    for p in merged_pairs:
        m_data = p["models"].get("OWL-ViT-Count")
        if not m_data: continue
        if "rs_by_m" in m_data:
            for M_str in m_data["rs_by_m"].keys():
                available_ms.add(int(M_str))
                for s2_str in m_data["rs_by_m"][M_str].keys():
                    available_sigmas.add(float(s2_str))
        elif "rs_by_sigma" in m_data:
            available_ms.add(4)
            for s2_str in m_data["rs_by_sigma"].keys():
                available_sigmas.add(float(s2_str))

        if "sigma1_sweep" in m_data:
            for s1_str in m_data["sigma1_sweep"].keys():
                available_sigma1s.add(float(s1_str))

    available_ms = sorted(list(available_ms)) if available_ms else [1, 2, 4, 8]
    available_sigmas = sorted(list(available_sigmas)) if available_sigmas else [0.0, 0.1, 0.25, 0.5, 1.0]
    available_sigma1s = sorted(list(available_sigma1s))

    # Resolve primary_sigma2, primary_m and primary_sigma1
    if primary_sigma2 not in available_sigmas:
        non_zero = [s for s in available_sigmas if s > 0]
        primary_sigma2 = non_zero[-1] if non_zero else available_sigmas[0]

    if primary_m not in available_ms:
        primary_m = available_ms[-1]

    if available_sigma1s and (primary_sigma1 not in available_sigma1s):
        primary_sigma1 = available_sigma1s[-1]

    # 2. Collect vanilla and RS data over the (M, sigma2) grid
    vanilla_slopes = []
    vanilla_rewards = []
    data_matrix = {M: {s: {"slopes": [], "rewards": []} for s in available_sigmas} for M in available_ms}

    for p in merged_pairs:
        m_data = p["models"].get("OWL-ViT-Count")
        if not m_data: continue
        vanilla_slopes.append(m_data["slope_vanilla"])
        vanilla_rewards.append(m_data["r_clean_vanilla"])

        has_rs_by_m = "rs_by_m" in m_data
        for M in available_ms:
            for s2 in available_sigmas:
                s2_key = str(s2)
                if s2_key == "0.0": s2_key = str(float(s2))
                
                entry = None
                if has_rs_by_m and str(M) in m_data["rs_by_m"]:
                    entry = m_data["rs_by_m"][str(M)].get(s2_key)
                elif "rs_by_sigma" in m_data:
                    entry = m_data["rs_by_sigma"].get(s2_key)

                if entry:
                    data_matrix[M][s2]["slopes"].append(entry["slope"])
                    data_matrix[M][s2]["rewards"].append(entry["r_clean"])

    vanilla_slopes = np.array(vanilla_slopes)
    vanilla_rewards = np.array(vanilla_rewards)

    # Boundary samples: pairs that cross the reward threshold (slope_vanilla > 0)
    boundary_mask = (vanilla_slopes > 0)
    num_boundary = int(np.sum(boundary_mask))
    num_total = len(vanilla_slopes)
    pct_boundary = (num_boundary / num_total * 100.0) if num_total > 0 else 0.0

    # Statistics over all samples
    L_mean_vanilla = float(np.mean(vanilla_slopes)) if len(vanilla_slopes) > 0 else 0.0
    L_med_vanilla = float(np.median(vanilla_slopes)) if len(vanilla_slopes) > 0 else 0.0
    L_95_vanilla = float(np.percentile(vanilla_slopes, 95)) if len(vanilla_slopes) > 0 else 0.0
    L_max_vanilla = float(np.max(vanilla_slopes)) if len(vanilla_slopes) > 0 else 0.0

    # Statistics over boundary samples
    b_vanilla_slopes = vanilla_slopes[boundary_mask]
    L_mean_b_vanilla = float(np.mean(b_vanilla_slopes)) if len(b_vanilla_slopes) > 0 else 0.0
    L_med_b_vanilla = float(np.median(b_vanilla_slopes)) if len(b_vanilla_slopes) > 0 else 0.0
    L_95_b_vanilla = float(np.percentile(b_vanilla_slopes, 95)) if len(b_vanilla_slopes) > 0 else 0.0
    L_max_b_vanilla = float(np.max(b_vanilla_slopes)) if len(b_vanilla_slopes) > 0 else 0.0

    # 3. Statistics for every (M, sigma2) cell
    stats_matrix = {M: {} for M in available_ms}
    for M in available_ms:
        for s2 in available_sigmas:
            arr_slopes = np.array(data_matrix[M][s2]["slopes"])
            arr_rewards = np.array(data_matrix[M][s2]["rewards"])
            arr_b_slopes = arr_slopes[boundary_mask] if (len(arr_slopes) == len(boundary_mask) and num_boundary > 0) else np.array([])
            
            stats_matrix[M][s2] = {
                "L_mean": float(np.mean(arr_slopes)) if len(arr_slopes) > 0 else 0.0,
                "L_median": float(np.median(arr_slopes)) if len(arr_slopes) > 0 else 0.0,
                "L_95": float(np.percentile(arr_slopes, 95)) if len(arr_slopes) > 0 else 0.0,
                "L_max": float(np.max(arr_slopes)) if len(arr_slopes) > 0 else 0.0,
                "slopes": arr_slopes,
                "rewards": arr_rewards,
                "boundary": {
                    "L_mean": float(np.mean(arr_b_slopes)) if len(arr_b_slopes) > 0 else 0.0,
                    "L_median": float(np.median(arr_b_slopes)) if len(arr_b_slopes) > 0 else 0.0,
                    "L_95": float(np.percentile(arr_b_slopes, 95)) if len(arr_b_slopes) > 0 else 0.0,
                    "L_max": float(np.max(arr_b_slopes)) if len(arr_b_slopes) > 0 else 0.0,
                    "count": int(len(arr_b_slopes))
                }
            }

    # ==================================================================================
    # Table 1A: all samples at the primary smoothing radius
    # ==================================================================================
    table1_rows_md = []
    table1_img_data = []
    table1_headers_md = [
        "Reward Model", "M",
        "$L_{\\text{mean}}$ (Vanilla $\\to$ RS)", "$L_{\\text{mean}}$ Reduction (↑)",
        "$L_{\\text{median}}$ (Vanilla $\\to$ RS)", "$L_{\\text{median}}$ Reduction (↑)",
        "$L_{95\\%}$ (Vanilla $\\to$ RS)", "$L_{95\\%}$ Reduction (↑)",
        "$L_{\\max}$ (Vanilla $\\to$ RS)", "$L_{\\max}$ Reduction (↑)"
    ]
    table1_headers_img = [
        "Model", "M",
        "L_mean (Van -> RS)", "L_mean red.",
        "L_med (Van -> RS)", "L_med red.",
        "L_95% (Van -> RS)", "L_95% red.",
        "L_max (Van -> RS)", "L_max red."
    ]

    for idx_m, M in enumerate(available_ms):
        st = stats_matrix[M][primary_sigma2]
        r_mean = f"{(L_mean_vanilla / st['L_mean']):.2f}x" if st['L_mean'] > 0 else "N/A"
        r_med = f"{(L_med_vanilla / st['L_median']):.2f}x" if st['L_median'] > 0 else "N/A"
        r_95 = f"{(L_95_vanilla / st['L_95']):.2f}x" if st['L_95'] > 0 else "N/A"
        r_max = f"{(L_max_vanilla / st['L_max']):.2f}x" if st['L_max'] > 0 else "N/A"

        model_name = "**OWL-ViT-Count**" if idx_m == 0 else ""
        model_name_img = "OWL-ViT-Count" if idx_m == 0 else ""

        table1_rows_md.append([
            model_name,
            f"**M = {M}**",
            f"{L_mean_vanilla:.6f} $\\to$ {st['L_mean']:.6f}",
            f"**{r_mean}**",
            f"{L_med_vanilla:.6f} $\\to$ {st['L_median']:.6f}",
            f"**{r_med}**",
            f"{L_95_vanilla:.6f} $\\to$ {st['L_95']:.6f}",
            f"**{r_95}**",
            f"{L_max_vanilla:.6f} $\\to$ {st['L_max']:.6f}",
            f"**{r_max}**"
        ])
        table1_img_data.append([
            model_name_img,
            f"M = {M}",
            f"{L_mean_vanilla:.5f} -> {st['L_mean']:.5f}",
            r_mean,
            f"{L_med_vanilla:.5f} -> {st['L_median']:.5f}",
            r_med,
            f"{L_95_vanilla:.5f} -> {st['L_95']:.5f}",
            r_95,
            f"{L_max_vanilla:.5f} -> {st['L_max']:.5f}",
            r_max
        ])

    # ==================================================================================
    # Table 1B: boundary samples at the primary smoothing radius
    # ==================================================================================
    table1b_rows_md = []
    table1b_img_data = []

    for idx_m, M in enumerate(available_ms):
        st_b = stats_matrix[M][primary_sigma2]["boundary"]
        r_b_mean = f"{(L_mean_b_vanilla / st_b['L_mean']):.2f}x" if st_b['L_mean'] > 0 else "N/A"
        r_b_med = f"{(L_med_b_vanilla / st_b['L_median']):.2f}x" if st_b['L_median'] > 0 else "N/A"
        r_b_95 = f"{(L_95_b_vanilla / st_b['L_95']):.2f}x" if st_b['L_95'] > 0 else "N/A"
        r_b_max = f"{(L_max_b_vanilla / st_b['L_max']):.2f}x" if st_b['L_max'] > 0 else "N/A"

        model_name_b = f"**Boundary (N={num_boundary})**" if idx_m == 0 else ""
        model_name_b_img = f"Boundary (N={num_boundary})" if idx_m == 0 else ""

        table1b_rows_md.append([
            model_name_b,
            f"**M = {M}**",
            f"{L_mean_b_vanilla:.6f} $\\to$ {st_b['L_mean']:.6f}",
            f"**{r_b_mean}**",
            f"{L_med_b_vanilla:.6f} $\\to$ {st_b['L_median']:.6f}",
            f"**{r_b_med}**",
            f"{L_95_b_vanilla:.6f} $\\to$ {st_b['L_95']:.6f}",
            f"**{r_b_95}**",
            f"{L_max_b_vanilla:.6f} $\\to$ {st_b['L_max']:.6f}",
            f"**{r_b_max}**"
        ])
        table1b_img_data.append([
            model_name_b_img,
            f"M = {M}",
            f"{L_mean_b_vanilla:.5f} -> {st_b['L_mean']:.5f}",
            r_b_mean,
            f"{L_med_b_vanilla:.5f} -> {st_b['L_median']:.5f}",
            r_b_med,
            f"{L_95_b_vanilla:.5f} -> {st_b['L_95']:.5f}",
            r_b_95,
            f"{L_max_b_vanilla:.5f} -> {st_b['L_max']:.5f}",
            r_b_max
        ])

    # ==================================================================================
    # Table 2: sigma2 sweep for each M (all samples)
    # ==================================================================================
    table2_headers_md = ["$\\sigma_2$"]
    table2_headers_img = ["sigma2"]
    for M in available_ms:
        table2_headers_md.append(f"$L_{{\\text{{mean}}}} (M={M})$")
        table2_headers_img.append(f"L_mean(M={M})")
    for M in available_ms:
        table2_headers_md.append(f"$L_{{\\max}} (M={M})$")
        table2_headers_img.append(f"L_max(M={M})")

    table2_rows_md = []
    table2_img_data = []

    for s2 in available_sigmas:
        s2_label_md = f"**{s2:.2f} (Vanilla)**" if s2 == 0.0 else f"**{s2:.2f}**"
        s2_label_img = f"{s2:.2f} (Vanilla)" if s2 == 0.0 else f"{s2:.2f}"

        row_md = [s2_label_md]
        row_img = [s2_label_img]

        # L_mean columns
        for M in available_ms:
            val_mean = stats_matrix[M][s2]["L_mean"]
            row_md.append(f"{val_mean:.6f}")
            row_img.append(f"{val_mean:.6f}")

        # L_max columns
        for M in available_ms:
            val_max = stats_matrix[M][s2]["L_max"]
            row_md.append(f"{val_max:.6f}")
            row_img.append(f"{val_max:.6f}")

        table2_rows_md.append(row_md)
        table2_img_data.append(row_img)

    # ==================================================================================
    # Table 2B: sigma2 sweep for each M (boundary samples)
    # ==================================================================================
    table2b_rows_md = []
    for s2 in available_sigmas:
        s2_label_md = f"**{s2:.2f} (Vanilla)**" if s2 == 0.0 else f"**{s2:.2f}**"
        row_b_md = [s2_label_md]
        for M in available_ms:
            val_b_mean = stats_matrix[M][s2]["boundary"]["L_mean"]
            row_b_md.append(f"{val_b_mean:.6f}")
        for M in available_ms:
            val_b_max = stats_matrix[M][s2]["boundary"]["L_max"]
            row_b_md.append(f"{val_b_max:.6f}")
        table2b_rows_md.append(row_b_md)

    # ==================================================================================
    # Table 3: probe radius sweep sigma1 (singularity)
    # ==================================================================================
    sigma1_stats = {}
    table3_rows_md = []
    table3_img_data = []
    table3_headers_md = [
        "Radius $\\sigma_1$",
        "Distance $\\|\\Delta x\\|_2$",
        "$L_{\\max}$ (Vanilla $\\to$ RS)", "$L_{\\max}$ Reduction (↑)",
        "$L_{\\text{mean}}$ (Vanilla $\\to$ RS)", "$L_{\\text{mean}}$ Reduction (↑)",
        "$L_{\\text{boundary}}$ (Vanilla $\\to$ RS)", "$L_{\\text{boundary}}$ Reduction (↑)"
    ]
    table3_headers_img = [
        "sigma1", "||dx||_2",
        "L_max (Van -> RS)", "L_max red.",
        "L_mean (Van -> RS)", "L_mean red.",
        "L_bnd (Van -> RS)", "L_bnd red."
    ]

    if available_sigma1s:
        for s1 in available_sigma1s:
            s1_k = str(s1)
            s1_v_slopes = []
            s1_rs_slopes = []
            s1_dx_list = []
            s1_b_v_slopes = []
            s1_b_rs_slopes = []

            for p in merged_pairs:
                m_data = p["models"].get("OWL-ViT-Count")
                if not m_data or "sigma1_sweep" not in m_data: continue
                ent = m_data["sigma1_sweep"].get(s1_k)
                if ent:
                    sl_v = ent["slope_vanilla"]
                    sl_rs = ent["slope_rs"]
                    s1_v_slopes.append(sl_v)
                    s1_rs_slopes.append(sl_rs)
                    s1_dx_list.append(ent["delta_x"])
                    if ent["delta_r_vanilla"] > 0:
                        s1_b_v_slopes.append(sl_v)
                        s1_b_rs_slopes.append(sl_rs)

            s1_v_slopes = np.array(s1_v_slopes)
            s1_rs_slopes = np.array(s1_rs_slopes)
            s1_b_v_slopes = np.array(s1_b_v_slopes)
            s1_b_rs_slopes = np.array(s1_b_rs_slopes)
            mean_dx = float(np.mean(s1_dx_list)) if s1_dx_list else 0.0

            l_max_v = float(np.max(s1_v_slopes)) if len(s1_v_slopes) > 0 else 0.0
            l_max_rs = float(np.max(s1_rs_slopes)) if len(s1_rs_slopes) > 0 else 0.0
            r_max = f"{(l_max_v / l_max_rs):.2f}x" if l_max_rs > 0 else "N/A"

            l_mean_v = float(np.mean(s1_v_slopes)) if len(s1_v_slopes) > 0 else 0.0
            l_mean_rs = float(np.mean(s1_rs_slopes)) if len(s1_rs_slopes) > 0 else 0.0
            r_mean = f"{(l_mean_v / l_mean_rs):.2f}x" if l_mean_rs > 0 else "N/A"

            l_b_v = float(np.mean(s1_b_v_slopes)) if len(s1_b_v_slopes) > 0 else 0.0
            l_b_rs = float(np.mean(s1_b_rs_slopes)) if len(s1_b_rs_slopes) > 0 else 0.0
            r_b = f"{(l_b_v / l_b_rs):.2f}x" if l_b_rs > 0 else "N/A"

            sigma1_stats[str(s1)] = {
                "mean_dx": mean_dx,
                "l_max_vanilla": l_max_v, "l_max_rs": l_max_rs, "ratio_max": r_max,
                "l_mean_vanilla": l_mean_v, "l_mean_rs": l_mean_rs, "ratio_mean": r_mean,
                "l_boundary_vanilla": l_b_v, "l_boundary_rs": l_b_rs, "ratio_boundary": r_b,
                "boundary_count": len(s1_b_v_slopes)
            }

            is_p = (s1 == primary_sigma1)
            table3_rows_md.append([
                f"**{s1}**" + (" *(Primary)*" if is_p else ""),
                f"{mean_dx:.2f}",
                f"{l_max_v:.6f} $\\to$ {l_max_rs:.6f}", f"**{r_max}**",
                f"{l_mean_v:.6f} $\\to$ {l_mean_rs:.6f}", f"**{r_mean}**",
                f"{l_b_v:.6f} $\\to$ {l_b_rs:.6f}", f"**{r_b}**"
            ])
            table3_img_data.append([
                f"{s1}" + ("*" if is_p else ""),
                f"{mean_dx:.2f}",
                f"{l_max_v:.5f} -> {l_max_rs:.5f}", r_max,
                f"{l_mean_v:.5f} -> {l_mean_rs:.5f}", r_mean,
                f"{l_b_v:.5f} -> {l_b_rs:.5f}", r_b
            ])

    # Console output
    print("\n" + "=" * 90)
    print(f"TABLE 1A: ALL SAMPLES (N={num_total}) AT sigma={primary_sigma2}")
    print("=" * 90)
    print(f"Vanilla LiDAR (All) - L_max: {L_max_vanilla:.6f} | L_mean: {L_mean_vanilla:.6f} | L_median: {L_med_vanilla:.6f}")
    for row in table1_img_data:
        print(f"- {row[1]:<8} | L_mean: {row[2]:<24} (reduction {row[3]:<6}) | L_max: {row[8]:<24} (reduction {row[9]:<6})")

    print("\n" + "=" * 90)
    print(f"TABLE 1B: BOUNDARY SAMPLES (N={num_boundary}/{num_total} = {pct_boundary:.1f}%) AT sigma={primary_sigma2}")
    print("=" * 90)
    print(f"Vanilla LiDAR (Boundary) - L_max: {L_max_b_vanilla:.6f} | L_mean: {L_mean_b_vanilla:.6f} | L_median: {L_med_b_vanilla:.6f}")
    for row in table1b_img_data:
        print(f"- {row[1]:<8} | L_mean: {row[2]:<24} (reduction {row[3]:<6}) | L_max: {row[8]:<24} (reduction {row[9]:<6})")
    print("=" * 90)

    if table3_img_data:
        print(f"\nTABLE 3: PROBE RADIUS sigma1 SWEEP (PRIMARY sigma2={primary_sigma2}, M={primary_m})")
        print("=" * 90)
        for row in table3_img_data:
            print(f"- sigma1={row[0]:<8} | dx={row[1]:<8} | L_max: {row[2]:<24} ({row[3]:<5}) | L_bnd: {row[6]:<24} ({row[7]:<5})")
        print("=" * 90 + "\n")

    # ==================================================================================
    # Full Markdown report
    # ==================================================================================
    md_output_path = os.path.join(output_dir, "threshold_m_ablation_tables.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("# Lipschitz Experiment: Threshold Reward (Zero-shot Counting via OWL-ViT)\n\n")
        f.write("> **Project**: RS-LiDAR | **Reward**: Threshold Reward (Zero-shot Object Counting)\n")
        f.write(f"> **Setup**: primary radius $\\sigma_2 = {primary_sigma2}$, sweep $\\sigma_2 \\in \\{{{', '.join([str(s) for s in available_sigmas])}\\}}$ | samples $M \\in \\{{{', '.join([str(m) for m in available_ms])}\\}}$ | total image pairs: {len(merged_pairs)}\n")
        f.write(f"> **Boundary-crossing samples**: {num_boundary} / {num_total} pairs ({pct_boundary:.1f}%)\n\n")
        f.write("---\n\n")

        # Table 1A
        f.write(f"### 2.1. Table 1A: All Samples (N={num_total}) at $\\sigma_2 = {primary_sigma2}$\n\n")
        f.write("| " + " | ".join(table1_headers_md) + " |\n")
        f.write("| " + " | ".join([":---:" if i > 0 else ":---" for i in range(len(table1_headers_md))]) + " |\n")
        for row in table1_rows_md:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        # Table 1B
        f.write(f"### 2.2. Table 1B: Boundary Samples (N={num_boundary}) at $\\sigma_2 = {primary_sigma2}$\n\n")
        f.write("> *Note: boundary samples are pairs with $R_{\\text{Vanilla}}(x_{\\text{clean}}) \\ne R_{\\text{Vanilla}}(x_{\\text{pert}})$, i.e. pairs that cross the binary threshold.*\n\n")
        f.write("| " + " | ".join(table1_headers_md) + " |\n")
        f.write("| " + " | ".join([":---:" if i > 0 else ":---" for i in range(len(table1_headers_md))]) + " |\n")
        for row in table1b_rows_md:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        # Table 2
        f.write("### 2.3. Table 2: $\\sigma_2$ Sweep for Each $M$ (All Samples)\n\n")
        f.write("| " + " | ".join(table2_headers_md) + " |\n")
        f.write("| " + " | ".join([":---:" for _ in range(len(table2_headers_md))]) + " |\n")
        for row in table2_rows_md:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        # Table 2B
        f.write("### 2.4. Table 2B: $\\sigma_2$ Sweep for Each $M$ (Boundary Samples)\n\n")
        f.write("| " + " | ".join(table2_headers_md) + " |\n")
        f.write("| " + " | ".join([":---:" for _ in range(len(table2_headers_md))]) + " |\n")
        for row in table2b_rows_md:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        # Table 3
        if table3_rows_md:
            f.write(f"### 2.5. Table 3: Probe Radius Sweep $\\sigma_1 \\to 0$ at the Primary Setting ($\\sigma_2 = {primary_sigma2}, M = {primary_m}$)\n\n")
            f.write("> *Singularity: as $\\sigma_1 \\to 0$, $\\|\\Delta x\\|_2 \\to 0$, so $L_{\\max}^{\\text{Vanilla}} \\to \\infty$, while $L_{\\max}^{\\text{RS}}$ stays bounded as predicted by Theorem 1.*\n\n")
            f.write("| " + " | ".join(table3_headers_md) + " |\n")
            f.write("| " + " | ".join([":---:" for _ in range(len(table3_headers_md))]) + " |\n")
            for row in table3_rows_md:
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n---\n\n")

        f.write("### 3. Observations\n")
        f.write(f"1. **Lipschitz constant on boundary samples**: over all samples, the vanilla $L_{{\\text{{mean}}}}$ is dominated by pairs that do not cross the threshold (zero slope). On boundary samples (N={num_boundary}), $L_{{\\text{{boundary, mean}}}}$ reaches **{L_mean_b_vanilla:.4f}**, and RS-LiDAR reduces this slope by **{(L_mean_b_vanilla / stats_matrix[primary_m][primary_sigma2]['boundary']['L_mean']):.2f}x** (at $M={primary_m}$).\n")
        f.write("2. **Bounded slope as $\\sigma_1 \\to 0$**: when the probe radius $\\sigma_1$ shrinks, the vanilla reward exhibits a singularity ($L \\to \\infty$), whereas RS-LiDAR keeps the slope bounded, consistent with Theorem 1 (Lipschitz smoothing).\n")

    print(f"Saved the full Markdown tables to: {md_output_path}")

    # Metrics JSON
    json_metrics_path = os.path.join(output_dir, "threshold_m_ablation_metrics.json")
    with open(json_metrics_path, "w", encoding="utf-8") as f:
        json_export = {
            "vanilla": {
                "all": {
                    "L_mean": L_mean_vanilla, "L_median": L_med_vanilla,
                    "L_95": L_95_vanilla, "L_max": L_max_vanilla
                },
                "boundary": {
                    "L_mean": L_mean_b_vanilla, "L_median": L_med_b_vanilla,
                    "L_95": L_95_b_vanilla, "L_max": L_max_b_vanilla,
                    "count": num_boundary, "percent": pct_boundary
                }
            },
            "primary_sigma2": primary_sigma2,
            "primary_m": primary_m,
            "primary_sigma1": primary_sigma1,
            "available_ms": available_ms,
            "available_sigmas": available_sigmas,
            "available_sigma1s": available_sigma1s,
            "matrix": {},
            "sigma1_sweep": sigma1_stats
        }
        for M in available_ms:
            json_export["matrix"][str(M)] = {}
            for s2 in available_sigmas:
                json_export["matrix"][str(M)][str(s2)] = {
                    "all": {
                        "L_mean": stats_matrix[M][s2]["L_mean"],
                        "L_median": stats_matrix[M][s2]["L_median"],
                        "L_95": stats_matrix[M][s2]["L_95"],
                        "L_max": stats_matrix[M][s2]["L_max"]
                    },
                    "boundary": stats_matrix[M][s2]["boundary"]
                }
        json.dump(json_export, f, indent=2)
    print(f"Saved detailed JSON metrics to: {json_metrics_path}")

    # ==================================================================================
    # Render tables as PNG images
    # ==================================================================================
    table1_img_path = os.path.join(output_dir, "threshold_table_primary_sigma.png")
    render_table_as_image(
        table1_img_data, table1_headers_img,
        f"Threshold Reward: All Samples Lipschitz Benchmark (sigma = {primary_sigma2})",
        table1_img_path
    )
    print(f"Saved Table 1A image (all samples) to: {table1_img_path}")

    table1b_img_path = os.path.join(output_dir, "threshold_table_boundary.png")
    render_table_as_image(
        table1b_img_data, table1_headers_img,
        f"Threshold Reward: Boundary Samples Lipschitz Benchmark (sigma = {primary_sigma2}, N={num_boundary})",
        table1b_img_path
    )
    print(f"Saved Table 1B image (boundary samples) to: {table1b_img_path}")

    table2_img_path = os.path.join(output_dir, "threshold_table_sweep_matrix.png")
    render_table_as_image(
        table2_img_data, table2_headers_img,
        "Threshold Reward: Lipschitz Matrix Sweep (sigma vs. M)",
        table2_img_path
    )
    print(f"Saved Table 2 image to: {table2_img_path}")

    # Table 3 image and sigma1 sweep plot (if data are available)
    if available_sigma1s and table3_img_data:
        table3_img_path = os.path.join(output_dir, "threshold_table_sigma1_sweep.png")
        render_table_as_image(
            table3_img_data, table3_headers_img,
            f"Threshold Reward: Probe Radius sigma1 Singularity Sweep (sigma2 = {primary_sigma2}, M = {primary_m})",
            table3_img_path
        )
        print(f"Saved Table 3 image (sigma1 sweep) to: {table3_img_path}")

        if len(available_sigma1s) > 1:
            plt.figure(figsize=(14, 5.5))
            plt.subplot(1, 2, 1)
            y_max_v = [sigma1_stats[str(s)]["l_max_vanilla"] for s in available_sigma1s]
            y_max_rs = [sigma1_stats[str(s)]["l_max_rs"] for s in available_sigma1s]
            plt.plot(available_sigma1s, y_max_v, 'r-o', lw=2.2, label='Vanilla LiDAR $L_{max}$')
            plt.plot(available_sigma1s, y_max_rs, 'b-s', lw=2.2, label=f'RS-LiDAR $L_{{max}}$ ($\\sigma_2$={primary_sigma2}, M={primary_m})')
            plt.xscale('log')
            plt.xlabel('Probe Radius $\\sigma_1$ (Log Scale)', fontsize=11, fontweight='bold')
            plt.ylabel('Lipschitz $L_{max}$', fontsize=11, fontweight='bold')
            plt.title('Singularity Explosion: $L_{max}$ vs. $\\sigma_1 \\to 0$', fontsize=12, fontweight='bold')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(frameon=True)

            plt.subplot(1, 2, 2)
            y_b_v = [sigma1_stats[str(s)]["l_boundary_vanilla"] for s in available_sigma1s]
            y_b_rs = [sigma1_stats[str(s)]["l_boundary_rs"] for s in available_sigma1s]
            plt.plot(available_sigma1s, y_b_v, 'r-o', lw=2.2, label='Vanilla Boundary Slope')
            plt.plot(available_sigma1s, y_b_rs, 'b-s', lw=2.2, label=f'RS-LiDAR Boundary Slope ($\\sigma_2$={primary_sigma2}, M={primary_m})')
            plt.xscale('log')
            plt.xlabel('Probe Radius $\\sigma_1$ (Log Scale)', fontsize=11, fontweight='bold')
            plt.ylabel('Boundary Lipschitz $L_{boundary}$', fontsize=11, fontweight='bold')
            plt.title('Boundary Slope: Vanilla $\\to \\infty$ vs. RS Bounded', fontsize=12, fontweight='bold')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(frameon=True)

            plt.tight_layout()
            sigma1_plot_path = os.path.join(output_dir, "threshold_sigma1_ablation.png")
            plt.savefig(sigma1_plot_path, dpi=300)
            plt.close()
            print(f"Saved the sigma1 ablation plot to: {sigma1_plot_path}")

    # ==================================================================================
    # Plot 1: ablation over sigma2 (one line per M)
    # ==================================================================================
    plt.figure(figsize=(14, 5.5))
    colors_m = {1: '#f87171', 2: '#fbbf24', 4: '#38bdf8', 8: '#1d4ed8'}
    markers_m = {1: 'o', 2: '^', 4: 's', 8: 'D'}

    # Subplot 1: L_mean over sigma2
    plt.subplot(1, 2, 1)
    for M in available_ms:
        means = [stats_matrix[M][s2]["L_mean"] for s2 in available_sigmas]
        c = colors_m.get(M, '#6366f1')
        mk = markers_m.get(M, 'o')
        plt.plot(available_sigmas, means, marker=mk, color=c, lw=2.2, label=f'RS-LiDAR (M={M})')
    plt.axhline(L_mean_vanilla, color='#dc2626', linestyle='--', lw=1.8, label='Vanilla $L_{mean}$')
    plt.xlabel('Smoothing Radius $\\sigma_2$', fontsize=11, fontweight='bold')
    plt.ylabel('Lipschitz $L_{mean}$', fontsize=11, fontweight='bold')
    plt.title('Landscape Smoothing: $L_{mean}$ vs. $\\sigma_2$', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)

    # Subplot 2: L_max over sigma2
    plt.subplot(1, 2, 2)
    for M in available_ms:
        maxs = [stats_matrix[M][s2]["L_max"] for s2 in available_sigmas]
        c = colors_m.get(M, '#6366f1')
        mk = markers_m.get(M, 'o')
        plt.plot(available_sigmas, maxs, marker=mk, color=c, lw=2.2, label=f'RS-LiDAR (M={M})')
    plt.axhline(L_max_vanilla, color='#dc2626', linestyle='--', lw=1.8, label='Vanilla $L_{max}$')
    plt.xlabel('Smoothing Radius $\\sigma_2$', fontsize=11, fontweight='bold')
    plt.ylabel('Lipschitz $L_{max}$', fontsize=11, fontweight='bold')
    plt.title('Extremal Bound: $L_{max}$ vs. $\\sigma_2$', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)

    plt.tight_layout()
    ablation_sigma_path = os.path.join(output_dir, "threshold_sigma_ablation.png")
    plt.savefig(ablation_sigma_path, dpi=300)
    plt.close()
    print(f"Saved the sigma2 ablation line plot to: {ablation_sigma_path}")

    # ==================================================================================
    # Plot 2: convergence in M
    # ==================================================================================
    if len(available_ms) > 1:
        plt.figure(figsize=(7, 5))
        m_means = [stats_matrix[M][primary_sigma2]["L_mean"] for M in available_ms]
        m_maxs = [stats_matrix[M][primary_sigma2]["L_max"] for M in available_ms]
        plt.plot(available_ms, m_means, marker='s', color='#1e3a8a', lw=2.2, label=f'$L_{{mean}}$ ($\\sigma$={primary_sigma2})')
        plt.plot(available_ms, m_maxs, marker='o', color='#0284c7', lw=2.2, label=f'$L_{{max}}$ ($\\sigma$={primary_sigma2})')
        plt.axhline(L_mean_vanilla, color='#dc2626', linestyle='--', label='Vanilla $L_{mean}$')
        plt.axhline(L_max_vanilla, color='#ea580c', linestyle=':', label='Vanilla $L_{max}$')
        plt.xlabel('Monte Carlo Samples $M$', fontsize=11, fontweight='bold')
        plt.ylabel('Empirical Lipschitz', fontsize=11, fontweight='bold')
        plt.title(f'Monte Carlo $M$ Convergence Rate ($\\sigma$ = {primary_sigma2})', fontsize=12, fontweight='bold')
        plt.xticks(available_ms)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        m_curves_path = os.path.join(output_dir, "threshold_m_ablation_curves.png")
        plt.savefig(m_curves_path, dpi=300)
        plt.close()
        print(f"Saved the M-convergence plot to: {m_curves_path}")

    # ==================================================================================
    # Plot 3: reward distribution and slope scatter for each sigma (incl. the primary one)
    # ==================================================================================
    non_zero_sigmas = [s for s in available_sigmas if s > 0]
    for s2 in non_zero_sigmas:
        is_primary = (s2 == primary_sigma2)
        primary_tag = "_primary" if is_primary else ""
        s2_suffix = f"sigma_{s2}{primary_tag}"

        rs_slopes_cur = stats_matrix[primary_m][s2]["slopes"]
        rs_rewards_cur = stats_matrix[primary_m][s2]["rewards"]

        # 3.1. Reward distribution
        plt.figure(figsize=(7, 5))
        plt.hist(vanilla_rewards, bins=10, alpha=0.55, color='#dc2626', label='Vanilla (Discrete 0/1)', density=True)
        plt.hist(rs_rewards_cur, bins=20, alpha=0.55, color='#2563eb', label=f'RS-LiDAR ($\\sigma$={s2}, M={primary_m})', density=True)
        plt.xlabel('Reward Value', fontsize=11, fontweight='bold')
        plt.ylabel('Density', fontsize=11, fontweight='bold')
        plt.title(f'Reward Distribution Smoothing ($\\sigma$ = {s2}, M = {primary_m})', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        reward_dist_path = os.path.join(output_dir, f"threshold_reward_dist_{s2_suffix}.png")
        plt.savefig(reward_dist_path, dpi=300)
        plt.close()

        # 3.2. Slope scatter
        plt.figure(figsize=(6, 6))
        max_scatter_val = max(
            np.percentile(vanilla_slopes, 99) if len(vanilla_slopes) > 0 else 1.0,
            np.percentile(rs_slopes_cur, 99) if len(rs_slopes_cur) > 0 else 1.0
        )
        if max_scatter_val <= 0: max_scatter_val = 0.05
        plt.scatter(vanilla_slopes, rs_slopes_cur, alpha=0.55, color='#7c3aed', s=16)
        plt.plot([0, max_scatter_val], [0, max_scatter_val], 'k--', lw=1.5, label='y = x (No Reduction)')
        plt.xlabel('Vanilla Slope', fontsize=11, fontweight='bold')
        plt.ylabel(f'RS-LiDAR Slope ($\\sigma$={s2}, M={primary_m})', fontsize=11, fontweight='bold')
        plt.xlim(0, max_scatter_val)
        plt.ylim(0, max_scatter_val)
        plt.title(f'Slope Contraction Scatter ($\\sigma$ = {s2}, M = {primary_m})', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        slope_scatter_path = os.path.join(output_dir, f"threshold_slope_scatter_{s2_suffix}.png")
        plt.savefig(slope_scatter_path, dpi=300)
        plt.close()

        # ==============================================================================
        # 4. Combined 2x2 dashboard
        # ==============================================================================
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(2, 2)

        # Panel 1 (top-left): Table 1 (primary sigma, all M)
        ax_tab = fig.add_subplot(gs[0, 0])
        ax_tab.axis('off')
        ax_tab.set_title(f"A. Lipschitz Benchmark Table ($\\sigma$ = {primary_sigma2})", fontsize=13, fontweight='bold', pad=12)
        tab_cell = ax_tab.table(cellText=table1_img_data, colLabels=table1_headers_img, loc='center', cellLoc='center')
        tab_cell.auto_set_font_size(False)
        tab_cell.set_fontsize(8.5)
        tab_cell.scale(1.0, 1.45)
        for (r, c), cl in tab_cell.get_celld().items():
            cl.set_edgecolor('#d1d5db')
            if r == 0:
                cl.set_facecolor('#1e3a8a')
                cl.set_text_props(color='white', fontweight='bold')
            else:
                cl.set_facecolor('#f9fafb' if r % 2 == 1 else '#ffffff')

        # Panel 2 (top-right): L_mean vs. sigma2
        ax_line = fig.add_subplot(gs[0, 1])
        for M in available_ms:
            means = [stats_matrix[M][s_val]["L_mean"] for s_val in available_sigmas]
            ax_line.plot(available_sigmas, means, marker=markers_m.get(M, 'o'),
                         color=colors_m.get(M, '#6366f1'), lw=2.0, label=f'RS-LiDAR (M={M})')
        ax_line.axhline(L_mean_vanilla, color='#dc2626', linestyle='--', lw=1.6, label='Vanilla $L_{mean}$')
        ax_line.set_xlabel('Smoothing Radius $\\sigma_2$', fontsize=11, fontweight='bold')
        ax_line.set_ylabel('Lipschitz $L_{mean}$', fontsize=11, fontweight='bold')
        ax_line.set_title('B. Landscape Flattening vs. $\\sigma_2$', fontsize=13, fontweight='bold')
        ax_line.grid(True, linestyle='--', alpha=0.5)
        ax_line.legend()

        # Panel 3 (bottom-left): reward distribution
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_dist.hist(vanilla_rewards, bins=10, alpha=0.55, color='#dc2626', label='Vanilla (Discrete 0/1)', density=True)
        ax_dist.hist(rs_rewards_cur, bins=20, alpha=0.55, color='#2563eb', label=f'RS-LiDAR ($\\sigma$={s2}, M={primary_m})', density=True)
        ax_dist.set_xlabel('Reward Value', fontsize=11, fontweight='bold')
        ax_dist.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_dist.set_title(f'C. Reward Distribution Smoothing ($\\sigma$ = {s2}, M = {primary_m})', fontsize=13, fontweight='bold')
        ax_dist.legend()
        ax_dist.grid(True, linestyle='--', alpha=0.4)

        # Panel 4 (bottom-right): slope scatter
        ax_scat = fig.add_subplot(gs[1, 1])
        ax_scat.scatter(vanilla_slopes, rs_slopes_cur, alpha=0.55, color='#7c3aed', s=16)
        ax_scat.plot([0, max_scatter_val], [0, max_scatter_val], 'k--', lw=1.5, label='y = x')
        ax_scat.set_xlabel('Vanilla Slope', fontsize=11, fontweight='bold')
        ax_scat.set_ylabel(f'RS-LiDAR Slope ($\\sigma$={s2}, M={primary_m})', fontsize=11, fontweight='bold')
        ax_scat.set_xlim(0, max_scatter_val)
        ax_scat.set_ylim(0, max_scatter_val)
        ax_scat.set_title(f'D. Slope Contraction Scatter ($\\sigma$ = {s2}, M = {primary_m})', fontsize=13, fontweight='bold')
        ax_scat.legend()
        ax_scat.grid(True, linestyle='--', alpha=0.4)

        fig.suptitle(f"Threshold Reward Comprehensive Analysis Dashboard (RS-LiDAR $\\sigma$={s2}, M={primary_m})", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        combined_path = os.path.join(output_dir, f"threshold_lipschitz_combined_{s2_suffix}.png")
        plt.savefig(combined_path, dpi=300)
        plt.close()
        print(f"Saved the combined dashboard to: {combined_path}")

    print(f"All tables and plots exported to: {output_dir}")


# ======================================================================================
# Entry point
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Empirical Lipschitz constant of a threshold reward (OWL-ViT counting)")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_particles", type=int, default=10)
    parser.add_argument("--guidance_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma1", type=float, default=0.05)
    parser.add_argument("--sigma1_list", type=str, default="0.001,0.005,0.01,0.02,0.05", help="Comma-separated probe radii sigma1 to sweep for the primary setting")
    parser.add_argument("--primary_sigma1", type=float, default=None, help="Primary probe radius (defaults to --sigma1)")
    parser.add_argument("--perturbation_type", type=str, default="gaussian", choices=["gaussian", "blur"])
    parser.add_argument("--sigma2", type=float, default=0.5)
    parser.add_argument("--sigma2_list", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--num_mc_samples", "-M", type=int, default=4)
    parser.add_argument("--m_list", type=str, default="1,2,4,8", help="Comma-separated Monte Carlo sample counts M to sweep")
    parser.add_argument("--primary_m", type=int, default=4, help="Reference M used for the main comparison")
    parser.add_argument("--use_crn", action="store_true", default=True)
    parser.add_argument("--no_crn", action="store_false", dest="use_crn")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/New_Rewards/Threshold_Rewards")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge_shards", action="store_true")
    args = parser.parse_args()

    if args.merge_shards:
        merge_and_plot_results(
            args.output_dir,
            primary_sigma2=args.sigma2,
            primary_m=args.primary_m,
            primary_sigma1=(args.primary_sigma1 if args.primary_sigma1 is not None else args.sigma1)
        )
        return

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    # 1. Stable Diffusion pipeline
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

    # 2. OWL-ViT
    print("Loading OWL-ViT (zero-shot object detection)...")
    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    owl_model.eval()

    # 3. Measurement
    run_empirical_lipschitz_test(
        pipe, vae, owl_processor, owl_model, COUNTING_PROMPTS, args, device=device
    )

if __name__ == "__main__":
    main()

