"""
========================================================================================
🧪 MODULE THỰC NGHIỆM ĐỘC LẬP: ĐO ĐẠC HỆ SỐ LIPSCHITZ CHO THRESHOLD REWARD (OWL-ViT)
========================================================================================

Module này chuyên biệt hóa cho Thí nghiệm 1 (Threshold Reward - Hàm Bước Nhảy).
Mục đích:
1. Chứng minh hàm Threshold Reward (đếm vật thể đúng/sai) không có tính Lipschitz (độ dốc -> ∞).
2. Đo đạc tác dụng của Randomized Smoothing (RS-LiDAR) biến Threshold Reward thành Lipschitz-continuous.
3. Xuất biểu đồ phân phối độ dốc (Slope Distribution) và phân phối phần thưởng (Reward Distribution).
"""

import os
import sys
import json
import glob
import argparse
import math
import torch
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Cấu hình môi trường
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Compatibility shims cho transformers
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
# 📚 PROMPT DATASET DÀNH RIÊNG CHO ĐẾM SỐ LƯỢNG (COUNTING)
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
# 🧠 HÀM THRESHOLD REWARD (BINARY) DÙNG OWL-ViT
# ======================================================================================
@torch.inference_mode()
def evaluate_threshold_reward(
    images, target_object, target_count, processor, model, device="cuda"
):
    """
    Đánh giá Batch ảnh với OWL-ViT.
    Trả về mảng float [N] có giá trị 1.0 (Đúng số lượng) hoặc 0.0 (Sai số lượng).
    """
    N = len(images)
    texts = [[f"a photo of a {target_object}"]] * N
    inputs = processor(text=texts, images=images, return_tensors="pt").to(device)
    
    outputs = model(**inputs)
    
    rewards = []
    for i in range(N):
        # Lấy logits cho ảnh thứ i
        logits = outputs.logits[i] # [num_boxes, num_classes]
        probs = logits.sigmoid().max(dim=-1).values
        # Đếm số box có prob > 0.1
        count = (probs > 0.1).sum().item()
        # Binary reward (Bước nhảy)
        rewards.append(1.0 if count == target_count else 0.0)
        
    return np.array(rewards)


# ======================================================================================
# 🖼️ GENERATION & IMAGE PROCESSING
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
# 🔬 CORE EXPERIMENTAL PIPELINE: MEASURING EMPIRICAL LIPSCHITZ CONSTANT
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

    # Đọc danh sách sigma2
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

    print(f"\n🚀 [Shard {args.shard_id}/{args.num_shards}] Bắt đầu thực nghiệm Threshold trên {len(active_prompts)} prompts")
    print(f"⚙️ Tham số: {args.num_particles} hạt/prompt | sigma1={args.sigma1} | M={args.num_mc_samples}")

    saved_data = {"completed_prompts": [], "pairs_data": []}
    if os.path.exists(ckpt_path) and not args.overwrite:
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            print(f"🔄 Đã nạp checkpoint: Đã hoàn thành {len(saved_data.get('completed_prompts', []))} prompts trước đó.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc checkpoint: {e}. Chạy mới từ đầu.")

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

        # 1. Sinh ảnh sạch x_clean và ảnh biến dạng x_pert
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

        # 2. Đánh giá Reward Vanilla: R(x_clean) và R(x_pert)
        vanilla_clean_rewards = evaluate_threshold_reward(
            clean_pils, target_obj, target_count, owl_processor, owl_model, device=device
        )
        vanilla_pert_rewards = evaluate_threshold_reward(
            pert_pils, target_obj, target_count, owl_processor, owl_model, device=device
        )

        # 3. Đánh giá Reward RS-LiDAR Monte Carlo cho từng giá trị sigma2
        rs_rewards_by_sigma = {} 

        for s2 in sigma2_vals:
            if s2 == 0.0:
                rs_rewards_by_sigma[s2] = {
                    "clean": vanilla_clean_rewards.tolist(),
                    "pert": vanilla_pert_rewards.tolist()
                }
                continue

            use_crn = getattr(args, "use_crn", True)
            mc_clean_accum = np.zeros(args.num_particles)
            mc_pert_accum = np.zeros(args.num_particles)

            for m_iter in range(args.num_mc_samples):
                u_m = torch.randn_like(clean_tensors) * s2

                # Ảnh sạch + nhiễu
                noisy_clean_t = (clean_tensors + u_m).clamp(-1.0, 1.0)
                noisy_clean_pils = pipe.image_processor.postprocess(noisy_clean_t, output_type="pil")
                r_m_clean = evaluate_threshold_reward(
                    noisy_clean_pils, target_obj, target_count, owl_processor, owl_model, device=device
                )
                mc_clean_accum += r_m_clean

                # Ảnh biến dạng + nhiễu
                u_prime_m = u_m if use_crn else (torch.randn_like(pert_tensors) * s2)
                noisy_pert_t = (pert_tensors + u_prime_m).clamp(-1.0, 1.0)
                noisy_pert_pils = pipe.image_processor.postprocess(noisy_pert_t, output_type="pil")
                r_m_pert = evaluate_threshold_reward(
                    noisy_pert_pils, target_obj, target_count, owl_processor, owl_model, device=device
                )
                mc_pert_accum += r_m_pert

            rs_rewards_by_sigma[s2] = {
                "clean": (mc_clean_accum / args.num_mc_samples).tolist(),
                "pert": (mc_pert_accum / args.num_mc_samples).tolist()
            }

        # 4. Ghi nhận dữ liệu
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
                "rs_by_sigma": {}
            }

            for s2, rs_dict in rs_rewards_by_sigma.items():
                r_clean_rs = float(rs_dict["clean"][k])
                r_pert_rs = float(rs_dict["pert"][k])
                diff_rs = abs(r_clean_rs - r_pert_rs)
                slope_rs = diff_rs / dx_k

                model_entry["rs_by_sigma"][str(s2)] = {
                    "r_clean": r_clean_rs,
                    "r_pert": r_pert_rs,
                    "delta_r": diff_rs,
                    "slope": slope_rs
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

    print(f"\n✅ Shard {args.shard_id} hoàn tất đo đạc. Lưu {len(all_pairs_data)} cặp tại {ckpt_path}.")


# ======================================================================================
# 📊 AGGREGATION & PLOTTING
# ======================================================================================
def merge_and_plot_results(output_dir, primary_sigma2):
    ckpt_files = glob.glob(os.path.join(output_dir, "checkpoint_threshold_shard_*.json"))
    if not ckpt_files:
        print("⚠️ Không tìm thấy file checkpoint_threshold_shard_*.json nào để gộp.")
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
            print(f"⚠️ Lỗi đọc file {c_path}: {e}")

    print(f"✅ Đã gộp thành công {len(merged_pairs)} cặp mẫu duy nhất.")

    # 1. Trích xuất dữ liệu cho tất cả các sigma
    vanilla_slopes = []
    vanilla_rewards = []
    
    # Lấy danh sách tất cả các sigma2 có trong dữ liệu
    available_sigmas = set()
    for p in merged_pairs:
        m_data = p["models"].get("OWL-ViT-Count")
        if m_data and "rs_by_sigma" in m_data:
            for s2 in m_data["rs_by_sigma"].keys():
                available_sigmas.add(float(s2))
    
    available_sigmas = sorted(list(available_sigmas))
    if len(available_sigmas) == 0:
        print("⚠️ Không có dữ liệu RS-LiDAR.")
        return

    # Dict lưu trữ slope theo từng sigma
    rs_slopes_by_sigma = {s: [] for s in available_sigmas}
    rs_rewards_by_sigma = {s: [] for s in available_sigmas}

    for p in merged_pairs:
        m_data = p["models"].get("OWL-ViT-Count")
        if not m_data: continue
        
        vanilla_slopes.append(m_data["slope_vanilla"])
        vanilla_rewards.append(m_data["r_clean_vanilla"])
        
        for s2 in available_sigmas:
            s2_key = str(s2)
            if s2_key == "0.0":
                s2_key = str(float(s2))
            
            if s2_key in m_data["rs_by_sigma"]:
                rs_slopes_by_sigma[s2].append(m_data["rs_by_sigma"][s2_key]["slope"])
                rs_rewards_by_sigma[s2].append(m_data["rs_by_sigma"][s2_key]["r_clean"])

    vanilla_slopes = np.array(vanilla_slopes)
    vanilla_rewards = np.array(vanilla_rewards)
    
    L_max_vanilla = float(np.max(vanilla_slopes)) if len(vanilla_slopes)>0 else 0
    L_mean_vanilla = float(np.mean(vanilla_slopes)) if len(vanilla_slopes)>0 else 0

    # Tính metric cho từng sigma
    L_max_rs_list = []
    L_mean_rs_list = []
    
    for s2 in available_sigmas:
        arr = np.array(rs_slopes_by_sigma[s2])
        L_max_rs_list.append(float(np.max(arr)) if len(arr) > 0 else 0)
        L_mean_rs_list.append(float(np.mean(arr)) if len(arr) > 0 else 0)

    # Lấy primary_sigma2 (hoặc lớn nhất nếu không có)
    if primary_sigma2 not in available_sigmas:
        non_zero = [s for s in available_sigmas if s > 0]
        primary_sigma2 = non_zero[-1] if non_zero else available_sigmas[0]
        
    idx_primary = available_sigmas.index(primary_sigma2)
    L_max_rs_primary = L_max_rs_list[idx_primary]
    L_mean_rs_primary = L_mean_rs_list[idx_primary]
    rs_slopes_primary = np.array(rs_slopes_by_sigma[primary_sigma2])
    rs_rewards_primary = np.array(rs_rewards_by_sigma[primary_sigma2])

    print("=" * 60)
    print("🏆 BẢNG TỔNG HỢP HỆ SỐ LI    # 3. Vẽ Ablation curve over sigma2 (chung cho tất cả)
    if len(available_sigmas) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot(available_sigmas, L_max_rs_list, marker='o', color='skyblue', lw=2, label='RS-LiDAR $L_{max}$')
        plt.plot(available_sigmas, L_mean_rs_list, marker='s', color='navy', lw=2, label='RS-LiDAR $L_{mean}$')
        plt.axhline(L_max_vanilla, color='salmon', linestyle='--', label='Vanilla $L_{max}$')
        plt.axhline(L_mean_vanilla, color='darkred', linestyle='--', label='Vanilla $L_{mean}$')
        plt.xlabel('Smoothing Radius $\\sigma_2$')
        plt.ylabel('Lipschitz Constant')
        plt.title('Lipschitz Bound vs. Smoothing Radius $\\sigma_2$')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "threshold_sigma_ablation.png"), dpi=300)
        plt.close()
        print(f"📈 Đã lưu đồ thị Ablation tại threshold_sigma_ablation.png")

    # 4. Lặp qua từng mức sigma để vẽ biểu đồ riêng lẻ
    non_zero_sigmas = [s for s in available_sigmas if s > 0]
    bar_width = 0.4
    x_pos = np.arange(2)

    for s2 in non_zero_sigmas:
        is_primary = (s2 == primary_sigma2)
        primary_text = " (Primary)" if is_primary else ""
        file_suffix = f"sigma_{s2}{'_primary' if is_primary else ''}"
        
        idx_s2 = available_sigmas.index(s2)
        L_max_rs_s2 = L_max_rs_list[idx_s2]
        L_mean_rs_s2 = L_mean_rs_list[idx_s2]
        rs_slopes_s2 = np.array(rs_slopes_by_sigma[s2])
        rs_rewards_s2 = np.array(rs_rewards_by_sigma[s2])

        # 4.1 Vẽ Plot tổng hợp chung (2x2) cho s2
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        
        ax_lmax = fig.add_subplot(gs[0, 0])
        ax_lmean = fig.add_subplot(gs[0, 1])
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_scatter = fig.add_subplot(gs[1, 1])
        
        fig.suptitle(f"Threshold Reward Analysis - RS-LiDAR σ={s2}{primary_text}", fontsize=16, fontweight='bold')

        # Subplot 1: Bar Chart L_max
        ax_lmax.bar(x_pos, [L_max_vanilla, L_max_rs_s2], bar_width, color=['salmon', 'skyblue'])
        ax_lmax.set_xticks(x_pos)
        ax_lmax.set_xticklabels(['Vanilla', f'RS-LiDAR\n(σ={s2})'])
        ax_lmax.set_ylabel('Empirical Lipschitz $L_{max}$')
        ax_lmax.set_title(f'Lipschitz $L_{max}$ Reduction')
        
        # Subplot 2: Bar Chart L_mean
        ax_lmean.bar(x_pos, [L_mean_vanilla, L_mean_rs_s2], bar_width, color=['darkred', 'navy'])
        ax_lmean.set_xticks(x_pos)
        ax_lmean.set_xticklabels(['Vanilla', f'RS-LiDAR\n(σ={s2})'])
        ax_lmean.set_ylabel('Empirical Lipschitz $L_{mean}$')
        ax_lmean.set_title(f'Lipschitz $L_{mean}$ Reduction')

        # Subplot 3: Reward Distribution
        ax_dist.hist(vanilla_rewards, bins=10, alpha=0.5, color='red', label='Vanilla (Discrete 0/1)', density=True)
        ax_dist.hist(rs_rewards_s2, bins=20, alpha=0.5, color='blue', label=f'RS-LiDAR (σ={s2})', density=True)
        ax_dist.set_xlabel('Reward Value')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title('Reward Distribution Smoothing')
        ax_dist.legend()

        # Subplot 4: Slope Scatter
        ax_scatter.scatter(vanilla_slopes, rs_slopes_s2, alpha=0.5, color='purple', s=10)
        max_val = max(np.percentile(vanilla_slopes, 99) if len(vanilla_slopes)>0 else 1,
                      np.percentile(rs_slopes_s2, 99) if len(rs_slopes_s2)>0 else 1)
        ax_scatter.plot([0, max_val], [0, max_val], 'k--', label='y = x')
        ax_scatter.set_xlabel('Vanilla Slope')
        ax_scatter.set_ylabel(f'RS-LiDAR Slope (σ={s2})')
        ax_scatter.set_xlim(0, max_val)
        ax_scatter.set_ylim(0, max_val)
        ax_scatter.set_title('Slope Reduction (Scatter)')
        ax_scatter.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        combined_path = os.path.join(output_dir, f"threshold_lipschitz_combined_{file_suffix}.png")
        plt.savefig(combined_path, dpi=300)
        plt.close()
        
        # 4.2 LƯU TỪNG BIỂU ĐỒ RIÊNG LẺ
        # 1. Bar Chart L_max
        plt.figure(figsize=(6, 5))
        plt.bar(x_pos, [L_max_vanilla, L_max_rs_s2], bar_width, color=['salmon', 'skyblue'])
        plt.xticks(x_pos, ['Vanilla', f'RS-LiDAR (σ={s2})'])
        plt.ylabel('Empirical Lipschitz $L_{max}$')
        plt.title(f'Lipschitz $L_{max}$ Reduction{primary_text}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"threshold_lmax_bar_{file_suffix}.png"), dpi=300)
        plt.close()

        # 2. Bar Chart L_mean
        plt.figure(figsize=(6, 5))
        plt.bar(x_pos, [L_mean_vanilla, L_mean_rs_s2], bar_width, color=['darkred', 'navy'])
        plt.xticks(x_pos, ['Vanilla', f'RS-LiDAR (σ={s2})'])
        plt.ylabel('Empirical Lipschitz $L_{mean}$')
        plt.title(f'Lipschitz $L_{mean}$ Reduction{primary_text}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"threshold_lmean_bar_{file_suffix}.png"), dpi=300)
        plt.close()

        # 3. Reward Distribution
        plt.figure(figsize=(7, 5))
        plt.hist(vanilla_rewards, bins=10, alpha=0.5, color='red', label='Vanilla (Discrete 0/1)', density=True)
        plt.hist(rs_rewards_s2, bins=20, alpha=0.5, color='blue', label=f'RS-LiDAR (σ={s2})', density=True)
        plt.xlabel('Reward Value')
        plt.ylabel('Density')
        plt.title(f'Reward Distribution Smoothing{primary_text}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"threshold_reward_dist_{file_suffix}.png"), dpi=300)
        plt.close()

        # 4. Slope Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(vanilla_slopes, rs_slopes_s2, alpha=0.5, color='purple', s=10)
        plt.plot([0, max_val], [0, max_val], 'k--', label='y = x')
        plt.xlabel('Vanilla Slope')
        plt.ylabel(f'RS-LiDAR Slope (σ={s2})')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(f'Slope Reduction (Scatter){primary_text}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"threshold_slope_scatter_{file_suffix}.png"), dpi=300)
        plt.close()
        
    print(f"📈 Đã lưu tất cả các biểu đồ cho từng mức sigma vào thư mục {output_dir}")

# ======================================================================================
# 🏁 MAIN ENTRY POINT
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Đo đạc Hệ số Lipschitz cho Threshold Reward bằng OWL-ViT")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_particles", type=int, default=10)
    parser.add_argument("--guidance_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma1", type=float, default=0.1)
    parser.add_argument("--perturbation_type", type=str, default="gaussian", choices=["gaussian", "blur"])
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--sigma2_list", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--num_mc_samples", "-M", type=int, default=4)
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
        merge_and_plot_results(args.output_dir, args.sigma2)
        return

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"🎮 Đang sử dụng GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    # 1. Khởi tạo Stable Diffusion Pipeline
    print(f"📥 Đang nạp mô hình {args.model_name}...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        scheduler=scheduler,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae

    # 2. Khởi tạo OWL-ViT
    print("📥 Đang nạp mô hình OWL-ViT (Zero-shot Object Detection)...")
    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    owl_model.eval()

    # 3. Chạy đo đạc
    run_empirical_lipschitz_test(
        pipe, vae, owl_processor, owl_model, COUNTING_PROMPTS, args, device=device
    )

if __name__ == "__main__":
    main()
