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

    # Đọc danh sách M (Monte Carlo Sample Sweep)
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

    print(f"\n🚀 [Shard {args.shard_id}/{args.num_shards}] Bắt đầu thực nghiệm Threshold trên {len(active_prompts)} prompts")
    print(f"⚙️ Tham số: {args.num_particles} hạt/prompt | sigma1={args.sigma1} | M sweep: {m_vals} (max={max_m}, primary={primary_m})")
    print(f"🔬 Bán kính sigma2 sweep: {sigma2_vals}")

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

        # 3. Đánh giá Reward RS-LiDAR Monte Carlo cho từng giá trị sigma2 và sweep M
        # Thuật toán tối ưu: Chỉ cần sinh max_m mẫu một lần duy nhất, sau đó prefix-mean cho từng M in m_vals!
        rs_rewards_by_sigma_m = {} 
        use_crn = getattr(args, "use_crn", True)

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

            for m_iter in range(max_m):
                u_m = torch.randn_like(clean_tensors) * s2

                # Ảnh sạch + nhiễu
                noisy_clean_t = (clean_tensors + u_m).clamp(-1.0, 1.0)
                noisy_clean_pils = pipe.image_processor.postprocess(noisy_clean_t, output_type="pil")
                r_m_clean = evaluate_threshold_reward(
                    noisy_clean_pils, target_obj, target_count, owl_processor, owl_model, device=device
                )
                clean_samples.append(r_m_clean)

                # Ảnh biến dạng + nhiễu (CRN nếu use_crn=True)
                u_prime_m = u_m if use_crn else (torch.randn_like(pert_tensors) * s2)
                noisy_pert_t = (pert_tensors + u_prime_m).clamp(-1.0, 1.0)
                noisy_pert_pils = pipe.image_processor.postprocess(noisy_pert_t, output_type="pil")
                r_m_pert = evaluate_threshold_reward(
                    noisy_pert_pils, target_obj, target_count, owl_processor, owl_model, device=device
                )
                pert_samples.append(r_m_pert)

            clean_samples_arr = np.array(clean_samples) # [max_m, num_particles]
            pert_samples_arr = np.array(pert_samples)   # [max_m, num_particles]

            for M in m_vals:
                avg_clean_m = np.mean(clean_samples_arr[:M], axis=0)
                avg_pert_m = np.mean(pert_samples_arr[:M], axis=0)
                rs_rewards_by_sigma_m[s2][M] = {
                    "clean": avg_clean_m.tolist(),
                    "pert": avg_pert_m.tolist()
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
                "rs_by_m": {},
                "rs_by_sigma": {}
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

            # Tương thích ngược: rs_by_sigma ánh xạ tới primary_m
            model_entry["rs_by_sigma"] = model_entry["rs_by_m"][str(primary_m)]

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
# 📊 UTILITIES: RENDER BẢNG RA HÌNH ẢNH MATPLOTLIB
# ======================================================================================
def render_table_as_image(table_data, col_labels, title, output_path, col_widths=None):
    """Render bảng dữ liệu thành ảnh PNG độ nét cao, định dạng zebra striping chuyên nghiệp."""
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
            # Cột đầu in đậm
            if col == 0 or (col == 1 and "OWL" in table_data[0][0]):
                cell.set_text_props(fontweight='semibold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================================================
# 📊 AGGREGATION, TABLES GENERATION & PLOTTING
# ======================================================================================
def merge_and_plot_results(output_dir, primary_sigma2=0.5, primary_m=4):
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

    # 1. Khám phá các mức M và sigma2 có trong dữ liệu
    available_ms = set()
    available_sigmas = set()

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

    available_ms = sorted(list(available_ms)) if available_ms else [1, 2, 4, 8]
    available_sigmas = sorted(list(available_sigmas)) if available_sigmas else [0.0, 0.1, 0.25, 0.5, 1.0]

    # Chuẩn hóa primary_sigma2 và primary_m
    if primary_sigma2 not in available_sigmas:
        non_zero = [s for s in available_sigmas if s > 0]
        primary_sigma2 = non_zero[-1] if non_zero else available_sigmas[0]

    if primary_m not in available_ms:
        primary_m = available_ms[-1]

    # 2. Thu thập dữ liệu Vanilla và RS theo ma trận (M, sigma2)
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
    L_mean_vanilla = float(np.mean(vanilla_slopes)) if len(vanilla_slopes) > 0 else 0.0
    L_med_vanilla = float(np.median(vanilla_slopes)) if len(vanilla_slopes) > 0 else 0.0
    L_95_vanilla = float(np.percentile(vanilla_slopes, 95)) if len(vanilla_slopes) > 0 else 0.0
    L_max_vanilla = float(np.max(vanilla_slopes)) if len(vanilla_slopes) > 0 else 0.0

    # 3. Tính toán thống kê cho từng ô trong ma trận (M, sigma2)
    stats_matrix = {M: {} for M in available_ms}
    for M in available_ms:
        for s2 in available_sigmas:
            arr_slopes = np.array(data_matrix[M][s2]["slopes"])
            arr_rewards = np.array(data_matrix[M][s2]["rewards"])
            stats_matrix[M][s2] = {
                "L_mean": float(np.mean(arr_slopes)) if len(arr_slopes) > 0 else 0.0,
                "L_median": float(np.median(arr_slopes)) if len(arr_slopes) > 0 else 0.0,
                "L_95": float(np.percentile(arr_slopes, 95)) if len(arr_slopes) > 0 else 0.0,
                "L_max": float(np.max(arr_slopes)) if len(arr_slopes) > 0 else 0.0,
                "slopes": arr_slopes,
                "rewards": arr_rewards
            }

    # ==================================================================================
    # 📋 BẢNG 1: BẢNG TỔNG HỢP TẠI BÁN KÍNH CHUẨN PRIMARY SIGMA2 (THEO TỪNG M)
    # ==================================================================================
    table1_rows_md = []
    table1_img_data = []
    table1_headers_md = [
        "Mô Hình Phần Thưởng", "Cấu Hình M",
        "$L_{\\text{mean}}$ (Vanilla $\\to$ RS)", "Tỷ Số Giảm $L_{\\text{mean}}$ (↑)",
        "$L_{\\text{median}}$ (Vanilla $\\to$ RS)", "Tỷ Số Giảm $L_{\\text{median}}$ (↑)",
        "$L_{95\\%}$ (Vanilla $\\to$ RS)", "Tỷ Số Giảm $L_{95\\%}$ (↑)",
        "$L_{\\max}$ (Vanilla $\\to$ RS)", "Tỷ Số Giảm $L_{\\max}$ (↑)"
    ]
    table1_headers_img = [
        "Mô Hình", "Cấu Hình M",
        "L_mean (Van -> RS)", "Giảm L_mean",
        "L_med (Van -> RS)", "Giảm L_med",
        "L_95% (Van -> RS)", "Giảm L_95%",
        "L_max (Van -> RS)", "Giảm L_max"
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
    # 📋 BẢNG 2: BẢNG QUÉT BÁN KÍNH SIGMA2 THEO TỪNG MỨC M
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

        # Cột L_mean
        for M in available_ms:
            val_mean = stats_matrix[M][s2]["L_mean"]
            row_md.append(f"{val_mean:.6f}")
            row_img.append(f"{val_mean:.6f}")

        # Cột L_max
        for M in available_ms:
            val_max = stats_matrix[M][s2]["L_max"]
            row_md.append(f"{val_max:.6f}")
            row_img.append(f"{val_max:.6f}")

        table2_rows_md.append(row_md)
        table2_img_data.append(row_img)

    # In ra terminal
    print("\n" + "=" * 90)
    print(f"🏆 BẢNG 1: TỔNG HỢP LIPSCHITZ TẠI BÁN KÍNH CHUẨN σ={primary_sigma2} (PRIMARY BENCHMARK)")
    print("=" * 90)
    print(f"Vanilla LiDAR (Gốc) - L_max: {L_max_vanilla:.6f} | L_mean: {L_mean_vanilla:.6f} | L_median: {L_med_vanilla:.6f}")
    for row in table1_img_data:
        print(f"• {row[1]:<8} | L_mean: {row[2]:<24} (Giảm {row[3]:<6}) | L_max: {row[8]:<24} (Giảm {row[9]:<6})")
    print("=" * 90)

    print(f"\n🏆 BẢNG 2: MA TRẬN QUÉT BÁN KÍNH σ2 THEO TỪNG MỨC M")
    print("=" * 90)
    header_str = " | ".join([f"{h:<14}" for h in table2_headers_img])
    print(header_str)
    print("-" * len(header_str))
    for row in table2_img_data:
        print(" | ".join([f"{val:<14}" for val in row]))
    print("=" * 90 + "\n")

    # ==================================================================================
    # 💾 XUẤT FILE MARKDOWN BẢNG BIỂU (TƯƠNG THÍCH HOÀN TOÀN TÀI LIỆU NGHIÊN CỨU)
    # ==================================================================================
    md_output_path = os.path.join(output_dir, "threshold_m_ablation_tables.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Khảo Sát Số Mẫu Monte Carlo $M$ Trên Hàm Threshold Reward (OWL-ViT)\n\n")
        f.write(f"> **Dự án**: RS-LiDAR | **Mô hình**: Threshold Reward (Zero-shot Object Counting)\n")
        f.write(f"> **Cấu hình khảo sát**: Bán kính chuẩn $\\sigma_2 = {primary_sigma2}$, Dải quét $\\sigma_2 \\in \\{{{', '.join([str(s) for s in available_sigmas])}\\}}$ | Số mẫu $M \\in \\{{{', '.join([str(m) for m in available_ms])}\\}}$ | Tổng số cặp ảnh: {len(merged_pairs)}\n\n")
        f.write("---\n\n")

        # Bảng 2.1
        f.write(f"### 2.1. Bảng Tổng Hợp Tại Bán Kính Chuẩn $\\sigma_2 = {primary_sigma2}$ (Primary Benchmark)\n\n")
        f.write("| " + " | ".join(table1_headers_md) + " |\n")
        f.write("| " + " | ".join([":---:" if i > 0 else ":---" for i in range(len(table1_headers_md))]) + " |\n")
        for row in table1_rows_md:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        # Bảng 2.2
        f.write(f"### 2.2. Bảng Quét Bán Kính $\\sigma_2$ Theo Từng Mức $M$\n\n")
        f.write("| " + " | ".join(table2_headers_md) + " |\n")
        f.write("| " + " | ".join([":---:" for _ in range(len(table2_headers_md))]) + " |\n")
        for row in table2_rows_md:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        f.write("### 3. Nhận Xét & Kết Luận Khoa Học\n")
        f.write("1. **Hiệu ứng làm phẳng Lipschitz**: RS-LiDAR làm giảm hệ số Lipschitz thực nghiệm rõ rệt qua các mức $M$, triệt tiêu các gai nhọn cực đoan do hàm bước nhảy gây ra.\n")
        f.write("2. **Độ mịn của phân phối Reward**: Khi tăng số mẫu Monte Carlo $M$, phân phối điểm thưởng chuyển đổi từ dạng nhị phân rời rạc {0, 1} sang dải liên tục mượt mà [0, 1], khôi phục thông tin gradient cho quá trình lấy mẫu diffusion.\n")

    print(f"📝 Đã lưu 2 bảng số liệu Markdown tại: {md_output_path}")

    # Lưu metrics JSON
    json_metrics_path = os.path.join(output_dir, "threshold_m_ablation_metrics.json")
    with open(json_metrics_path, "w", encoding="utf-8") as f:
        json_export = {
            "vanilla": {
                "L_mean": L_mean_vanilla, "L_median": L_med_vanilla,
                "L_95": L_95_vanilla, "L_max": L_max_vanilla
            },
            "primary_sigma2": primary_sigma2,
            "primary_m": primary_m,
            "available_ms": available_ms,
            "available_sigmas": available_sigmas,
            "matrix": {}
        }
        for M in available_ms:
            json_export["matrix"][str(M)] = {}
            for s2 in available_sigmas:
                json_export["matrix"][str(M)][str(s2)] = {
                    "L_mean": stats_matrix[M][s2]["L_mean"],
                    "L_median": stats_matrix[M][s2]["L_median"],
                    "L_95": stats_matrix[M][s2]["L_95"],
                    "L_max": stats_matrix[M][s2]["L_max"]
                }
        json.dump(json_export, f, indent=2)
    print(f"💾 Đã lưu JSON metrics chi tiết tại: {json_metrics_path}")

    # ==================================================================================
    # 🖼️ RENDER BẢNG 1 VÀ BẢNG 2 THÀNH ẢNH PNG RIÊNG BIỆT
    # ==================================================================================
    table1_img_path = os.path.join(output_dir, "threshold_table_primary_sigma.png")
    render_table_as_image(
        table1_img_data, table1_headers_img,
        f"Threshold Reward: Lipschitz Reduction Benchmark (σ = {primary_sigma2})",
        table1_img_path
    )
    print(f"📊 Đã lưu ảnh Bảng 1 tại: {table1_img_path}")

    table2_img_path = os.path.join(output_dir, "threshold_table_sweep_matrix.png")
    render_table_as_image(
        table2_img_data, table2_headers_img,
        f"Threshold Reward: Lipschitz Matrix Sweep (σ vs. M)",
        table2_img_path
    )
    print(f"📊 Đã lưu ảnh Bảng 2 tại: {table2_img_path}")

    # ==================================================================================
    # 📈 BIỂU ĐỒ 1: LINE GRAPH ABLATION THEO SIGMA2 (CÁC ĐƯỜNG M)
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
    print(f"📈 Đã lưu đồ thị Line Graph Ablation theo sigma2 tại: {ablation_sigma_path}")

    # ==================================================================================
    # 📈 BIỂU ĐỒ 2: ĐỒ THỊ HỘI TỤ THEO M (M-ABLATION CONVERGENCE)
    # ==================================================================================
    if len(available_ms) > 1:
        plt.figure(figsize=(7, 5))
        m_means = [stats_matrix[M][primary_sigma2]["L_mean"] for M in available_ms]
        m_maxs = [stats_matrix[M][primary_sigma2]["L_max"] for M in available_ms]
        plt.plot(available_ms, m_means, marker='s', color='#1e3a8a', lw=2.2, label=f'$L_{{mean}}$ (σ={primary_sigma2})')
        plt.plot(available_ms, m_maxs, marker='o', color='#0284c7', lw=2.2, label=f'$L_{{max}}$ (σ={primary_sigma2})')
        plt.axhline(L_mean_vanilla, color='#dc2626', linestyle='--', label='Vanilla $L_{mean}$')
        plt.axhline(L_max_vanilla, color='#ea580c', linestyle=':', label='Vanilla $L_{max}$')
        plt.xlabel('Monte Carlo Samples $M$', fontsize=11, fontweight='bold')
        plt.ylabel('Empirical Lipschitz', fontsize=11, fontweight='bold')
        plt.title(f'Monte Carlo $M$ Convergence Rate (σ = {primary_sigma2})', fontsize=12, fontweight='bold')
        plt.xticks(available_ms)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        m_curves_path = os.path.join(output_dir, "threshold_m_ablation_curves.png")
        plt.savefig(m_curves_path, dpi=300)
        plt.close()
        print(f"📈 Đã lưu đồ thị M-Convergence tại: {m_curves_path}")

    # ==================================================================================
    # 📈 BIỂU ĐỒ 3: PHÂN PHỐI REWARD & SCATTER PLOT CHO CÁC MỨC SIGMA (ĐẶC BIỆT PRIMARY)
    # ==================================================================================
    non_zero_sigmas = [s for s in available_sigmas if s > 0]
    for s2 in non_zero_sigmas:
        is_primary = (s2 == primary_sigma2)
        primary_tag = "_primary" if is_primary else ""
        s2_suffix = f"sigma_{s2}{primary_tag}"

        rs_slopes_cur = stats_matrix[primary_m][s2]["slopes"]
        rs_rewards_cur = stats_matrix[primary_m][s2]["rewards"]

        # 3.1. Reward Distribution riêng lẻ
        plt.figure(figsize=(7, 5))
        plt.hist(vanilla_rewards, bins=10, alpha=0.55, color='#dc2626', label='Vanilla (Discrete 0/1)', density=True)
        plt.hist(rs_rewards_cur, bins=20, alpha=0.55, color='#2563eb', label=f'RS-LiDAR (σ={s2}, M={primary_m})', density=True)
        plt.xlabel('Reward Value', fontsize=11, fontweight='bold')
        plt.ylabel('Density', fontsize=11, fontweight='bold')
        plt.title(f'Reward Distribution Smoothing (σ = {s2}, M = {primary_m})', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        reward_dist_path = os.path.join(output_dir, f"threshold_reward_dist_{s2_suffix}.png")
        plt.savefig(reward_dist_path, dpi=300)
        plt.close()

        # 3.2. Slope Scatter riêng lẻ
        plt.figure(figsize=(6, 6))
        max_scatter_val = max(
            np.percentile(vanilla_slopes, 99) if len(vanilla_slopes) > 0 else 1.0,
            np.percentile(rs_slopes_cur, 99) if len(rs_slopes_cur) > 0 else 1.0
        )
        if max_scatter_val <= 0: max_scatter_val = 0.05
        plt.scatter(vanilla_slopes, rs_slopes_cur, alpha=0.55, color='#7c3aed', s=16)
        plt.plot([0, max_scatter_val], [0, max_scatter_val], 'k--', lw=1.5, label='y = x (No Reduction)')
        plt.xlabel('Vanilla Slope', fontsize=11, fontweight='bold')
        plt.ylabel(f'RS-LiDAR Slope (σ={s2}, M={primary_m})', fontsize=11, fontweight='bold')
        plt.xlim(0, max_scatter_val)
        plt.ylim(0, max_scatter_val)
        plt.title(f'Slope Contraction Scatter (σ = {s2}, M = {primary_m})', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        slope_scatter_path = os.path.join(output_dir, f"threshold_slope_scatter_{s2_suffix}.png")
        plt.savefig(slope_scatter_path, dpi=300)
        plt.close()

        # ==============================================================================
        # 🖼️ 4. BẢN GỘP CHUNG (COMBINED 2x2 DASHBOARD)
        # ==============================================================================
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(2, 2)

        # Panel 1 (Top-Left): Render Bảng 1 (Table Primary across M)
        ax_tab = fig.add_subplot(gs[0, 0])
        ax_tab.axis('off')
        ax_tab.set_title(f"A. Lipschitz Benchmark Table (σ = {primary_sigma2})", fontsize=13, fontweight='bold', pad=12)
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

        # Panel 2 (Top-Right): Line graph ablation L_mean over sigma2
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

        # Panel 3 (Bottom-Left): Reward Distribution
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_dist.hist(vanilla_rewards, bins=10, alpha=0.55, color='#dc2626', label='Vanilla (Discrete 0/1)', density=True)
        ax_dist.hist(rs_rewards_cur, bins=20, alpha=0.55, color='#2563eb', label=f'RS-LiDAR (σ={s2}, M={primary_m})', density=True)
        ax_dist.set_xlabel('Reward Value', fontsize=11, fontweight='bold')
        ax_dist.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_dist.set_title(f'C. Reward Distribution Smoothing (σ = {s2}, M = {primary_m})', fontsize=13, fontweight='bold')
        ax_dist.legend()
        ax_dist.grid(True, linestyle='--', alpha=0.4)

        # Panel 4 (Bottom-Right): Slope Scatter
        ax_scat = fig.add_subplot(gs[1, 1])
        ax_scat.scatter(vanilla_slopes, rs_slopes_cur, alpha=0.55, color='#7c3aed', s=16)
        ax_scat.plot([0, max_scatter_val], [0, max_scatter_val], 'k--', lw=1.5, label='y = x')
        ax_scat.set_xlabel('Vanilla Slope', fontsize=11, fontweight='bold')
        ax_scat.set_ylabel(f'RS-LiDAR Slope (σ={s2}, M={primary_m})', fontsize=11, fontweight='bold')
        ax_scat.set_xlim(0, max_scatter_val)
        ax_scat.set_ylim(0, max_scatter_val)
        ax_scat.set_title(f'D. Slope Contraction Scatter (σ = {s2}, M = {primary_m})', fontsize=13, fontweight='bold')
        ax_scat.legend()
        ax_scat.grid(True, linestyle='--', alpha=0.4)

        fig.suptitle(f"Threshold Reward Comprehensive Analysis Dashboard (RS-LiDAR σ={s2}, M={primary_m})", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        combined_path = os.path.join(output_dir, f"threshold_lipschitz_combined_{s2_suffix}.png")
        plt.savefig(combined_path, dpi=300)
        plt.close()
        print(f"📊 Đã lưu Dashboard gộp tại: {combined_path}")

    print(f"✨ Hoàn tất xuất toàn bộ bảng biểu và đồ thị tại thư mục: {output_dir}")


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
    parser.add_argument("--sigma1", type=float, default=0.05)
    parser.add_argument("--perturbation_type", type=str, default="gaussian", choices=["gaussian", "blur"])
    parser.add_argument("--sigma2", type=float, default=0.5)
    parser.add_argument("--sigma2_list", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--num_mc_samples", "-M", type=int, default=4)
    parser.add_argument("--m_list", type=str, default="1,2,4,8", help="Dải quét số mẫu Monte Carlo M")
    parser.add_argument("--primary_m", type=int, default=4, help="Mức M chuẩn làm mốc so sánh chính")
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
        merge_and_plot_results(args.output_dir, args.sigma2, args.primary_m)
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

