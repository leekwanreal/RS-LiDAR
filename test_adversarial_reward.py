"""
========================================================================================
🧪 MODULE THỰC NGHIỆM ĐỘC LẬP: ĐO ĐẠC HỆ SỐ LIPSCHITZ & GRADIENT CHO ADVERSARIAL REWARD
========================================================================================

Module này chuyên biệt hóa cho Thí nghiệm 2 (Adversarially Vulnerable Models - ResNet-50).
Mục đích:
1. Khảo sát hàm phần thưởng liên tục có hằng số Lipschitz cục bộ khổng lồ L_local >> 1.
2. Chứng minh Vanilla guidance bị bẫy bởi gradient đột biến (Adversarial Spikes & Misdirection).
3. Đo đạc tác dụng của Randomized Smoothing (RS-LiDAR): san phẳng landscape, dập tắt gai nhọn,
   ép gradient norm về cận lý thuyết Theorem 1, và bảo toàn đặc trưng ngữ nghĩa bền vững.
4. Xuất Bảng Lipschitz Benchmark, Breakdown 10 classes, Saliency Heatmaps, và 1D Landscape Slice.
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
import torchvision.models as models

try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
except ImportError:
    DPMSolverMultistepScheduler = None
    StableDiffusionPipeline = None

# (Đã loại bỏ CLIP theo yêu cầu để tập trung vào kiểm chứng toán học thuần túy của hàm phần thưởng)


# ======================================================================================
# 📚 PROMPT DATASET DÀNH CHO 10 SUPER-CATEGORIES CỦA IMAGENET (50 PROMPTS)
# ======================================================================================
ADVERSARIAL_REWARD_PROMPTS = [
    # === 1. Canine: Golden Retriever (target_class_id = 207) ===
    {"prompt": "a golden retriever sitting in a park", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a cute golden retriever puppy looking at camera", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a golden retriever running through autumn leaves", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a golden retriever lying on green grass", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a happy golden retriever with tongue out in sunlight", "target_class": 207, "class_name": "golden retriever"},

    # === 2. Feline: Tabby Cat (target_class_id = 281) ===
    {"prompt": "a tabby cat sleeping on a windowsill", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "a tabby cat sitting on a sofa", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "a striped tabby cat on a wooden floor", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "a cute tabby kitten playing with yarn", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "close-up portrait of a striped tabby cat", "target_class": 281, "class_name": "tabby cat"},

    # === 3. Ursidae: Giant Panda (target_class_id = 388 - Goodfellow et al. 2015) ===
    {"prompt": "a giant panda sitting and eating bamboo", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "a cute giant panda bear in a bamboo forest", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "close-up portrait of a giant panda looking forward", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "a giant panda resting on a wooden log", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "a baby giant panda playing on green grass", "target_class": 388, "class_name": "giant panda"},

    # === 4. Avian: Flamingo (target_class_id = 130) ===
    {"prompt": "a vibrant pink flamingo standing in shallow water", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "a flock of pink flamingos in a tropical lagoon", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "close-up portrait of an elegant pink flamingo", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "a pink flamingo resting on one leg in the sun", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "a beautiful flamingo reflected in calm water", "target_class": 130, "class_name": "flamingo"},

    # === 5. Vehicle: Sports Car (target_class_id = 817) ===
    {"prompt": "a sleek red sports car parked on a city street", "target_class": 817, "class_name": "sports car"},
    {"prompt": "a modern luxury sports car driving on highway", "target_class": 817, "class_name": "sports car"},
    {"prompt": "front view of a shiny black sports car", "target_class": 817, "class_name": "sports car"},
    {"prompt": "a yellow sports car parked on a mountain road", "target_class": 817, "class_name": "sports car"},
    {"prompt": "a red sports car under bright studio lights", "target_class": 817, "class_name": "sports car"},

    # === 6. Aviation: Airliner (target_class_id = 404) ===
    {"prompt": "a commercial airliner flying through blue sky with clouds", "target_class": 404, "class_name": "airliner"},
    {"prompt": "a passenger airliner approaching runway for landing", "target_class": 404, "class_name": "airliner"},
    {"prompt": "a large passenger jet airliner parked at an airport", "target_class": 404, "class_name": "airliner"},
    {"prompt": "side view of a white passenger airliner in flight", "target_class": 404, "class_name": "airliner"},
    {"prompt": "an airliner taking off into a golden sunset", "target_class": 404, "class_name": "airliner"},

    # === 7. Flora: Daisy (target_class_id = 985) ===
    {"prompt": "a field of white daisies in bright sunlight", "target_class": 985, "class_name": "daisy"},
    {"prompt": "a single white daisy flower close-up in morning dew", "target_class": 985, "class_name": "daisy"},
    {"prompt": "white daisy flowers blooming in a lush garden", "target_class": 985, "class_name": "daisy"},
    {"prompt": "macro photography of a daisy with yellow center", "target_class": 985, "class_name": "daisy"},
    {"prompt": "wild white daisies growing in green grass", "target_class": 985, "class_name": "daisy"},

    # === 8. Food: Strawberry (target_class_id = 949) ===
    {"prompt": "a bowl of fresh ripe red strawberries", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "close-up of a single juicy red strawberry with green leaves", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "freshly picked red strawberries on a wooden table", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "a pile of sweet red strawberries in sunlight", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "ripe red strawberries growing in a garden patch", "target_class": 949, "class_name": "strawberry"},

    # === 9. Electronics: Laptop (target_class_id = 620) ===
    {"prompt": "a modern silver laptop open on a wooden desk", "target_class": 620, "class_name": "laptop"},
    {"prompt": "a sleek laptop with glowing screen in a modern cafe", "target_class": 620, "class_name": "laptop"},
    {"prompt": "a laptop computer next to a cup of coffee on desk", "target_class": 620, "class_name": "laptop"},
    {"prompt": "a slim laptop on an office desk beside a window", "target_class": 620, "class_name": "laptop"},
    {"prompt": "close-up view of a modern laptop keyboard and screen", "target_class": 620, "class_name": "laptop"},

    # === 10. Landscape: Volcano (target_class_id = 980) ===
    {"prompt": "a majestic snow-capped volcano under blue sky", "target_class": 980, "class_name": "volcano"},
    {"prompt": "an active volcano emitting smoke and ash into the sky", "target_class": 980, "class_name": "volcano"},
    {"prompt": "a dramatic volcano mountain landscape at sunrise", "target_class": 980, "class_name": "volcano"},
    {"prompt": "a towering volcano rising above a peaceful lake", "target_class": 980, "class_name": "volcano"},
    {"prompt": "aerial view of a volcanic mountain crater with clouds", "target_class": 980, "class_name": "volcano"},
]


# ======================================================================================
# 🧠 HÀM REWARD ADVERSARIAL VỚI RESNET-50
# ======================================================================================
class ResNet50AdversarialReward:
    """
    Reward function dựa trên Logit/Softmax của mô hình ResNet-50 được tiền huấn luyện trên ImageNet.
    Có hằng số Lipschitz cục bộ lớn và nhạy cảm cao với nhiễu đối nghịch.
    """
    def __init__(self, device="cuda", reward_mode="logit"):
        self.device = device
        self.reward_mode = reward_mode
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights).to(device)
        except Exception:
            self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(self, tensors):
        """
        Nhận tensors ở dải [-1.0, 1.0], chuyển sang [0.0, 1.0],
        resize về 224x224 và chuẩn hóa theo chuẩn ImageNet.
        """
        # Map [-1, 1] -> [0, 1]
        x = (tensors + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        # Interpolate to 224x224
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Normalize
        x = (x - self.mean) / self.std
        return x

    def compute_reward(self, tensors, target_class_id):
        """
        Đánh giá reward cho batch tensor [N, 3, H, W] trên target_class_id.
        """
        x = self.preprocess(tensors)
        logits = self.model(x)

        if self.reward_mode == "logit":
            return logits[:, target_class_id]
        elif self.reward_mode == "softmax":
            probs = F.softmax(logits, dim=-1)
            return probs[:, target_class_id]
        elif self.reward_mode == "margin":
            target_logit = logits[:, target_class_id]
            # Max other logit
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, target_class_id] = False
            other_max = logits[mask].view(logits.size(0), -1).max(dim=-1)[0]
            return target_logit - other_max
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def compute_gradient(self, tensors, target_class_id):
        """
        Tính toán đạo hàm chính xác dR/dx bằng PyTorch Autograd.
        Trả về tensor gradient [N, 3, H, W] và chuẩn norm L2 [N].
        """
        x_in = tensors.clone().detach().requires_grad_(True)
        r = self.compute_reward(x_in, target_class_id)
        loss = r.sum()
        grads = torch.autograd.grad(loss, x_in, create_graph=False, retain_graph=False)[0]
        grad_norms = grads.view(tensors.size(0), -1).norm(p=2, dim=-1)
        return grads.detach(), grad_norms.detach()

    def compute_rs_gradient(self, tensors, target_class_id, sigma2, num_mc_samples=4, use_crn=True):
        """
        Tính gradient của hàm Smoothed Reward:
        ∇R_σ(x) = E_eps [ ∇R(x + eps) ]
        """
        if sigma2 == 0.0 or num_mc_samples <= 0:
            return self.compute_gradient(tensors, target_class_id)

        mc_grads = []
        for m in range(num_mc_samples):
            if use_crn:
                noise = torch.randn((1, *tensors.shape[1:]), device=tensors.device, dtype=tensors.dtype) * sigma2
            else:
                noise = torch.randn_like(tensors) * sigma2
            noisy_t = (tensors + noise).clamp(-1.0, 1.0)
            g_m, _ = self.compute_gradient(noisy_t, target_class_id)
            mc_grads.append(g_m)

        rs_grads = torch.stack(mc_grads, dim=0).mean(dim=0)
        rs_grad_norms = rs_grads.view(tensors.size(0), -1).norm(p=2, dim=-1)
        return rs_grads, rs_grad_norms

    def compute_gradcam(self, tensors, target_class_id):
        """
        Tính Grad-CAM heatmap tại layer4 của ResNet-50 cho target_class_id.
        Layer4 cung cấp các đặc trưng ngữ nghĩa mức cao (high-level semantic features).
        Trả về:
            cam: Tensor kích thước [N, H, W] chuẩn hóa về [0.0, 1.0].
        """
        with torch.enable_grad():
            x = self.preprocess(tensors)
            x.requires_grad_(True)

            # Forward từng block của ResNet-50 để trích xuất feature map tại layer4
            feat = self.model.conv1(x)
            feat = self.model.bn1(feat)
            feat = self.model.relu(feat)
            feat = self.model.maxpool(feat)
            feat = self.model.layer1(feat)
            feat = self.model.layer2(feat)
            feat = self.model.layer3(feat)
            feat = self.model.layer4(feat)  # [N, 2048, 7, 7]
            feat.retain_grad()

            pooled = self.model.avgpool(feat)
            flattened = torch.flatten(pooled, 1)
            logits = self.model.fc(flattened)

            if self.reward_mode == "logit":
                target_score = logits[:, target_class_id]
            elif self.reward_mode == "softmax":
                probs = F.softmax(logits, dim=-1)
                target_score = probs[:, target_class_id]
            elif self.reward_mode == "margin":
                target_logit = logits[:, target_class_id]
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[:, target_class_id] = False
                other_max = logits[mask].view(logits.size(0), -1).max(dim=-1)[0]
                target_score = target_logit - other_max
            else:
                target_score = logits[:, target_class_id]

            loss = target_score.sum()
            self.model.zero_grad()
            loss.backward(retain_graph=False)

            gradients = feat.grad  # [N, 2048, 7, 7]
            if gradients is None:
                gradients = torch.ones_like(feat)
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # [N, 2048, 1, 1]
            cam = F.relu((weights * feat).sum(dim=1, keepdim=True))  # [N, 1, 7, 7]

            # Upsample về kích thước tensor gốc [N, 1, H, W]
            cam = F.interpolate(cam, size=tensors.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            # Min-max normalization theo từng sample trong batch
            cam_min = cam.view(cam.size(0), -1).min(dim=-1)[0].view(-1, 1, 1)
            cam_max = cam.view(cam.size(0), -1).max(dim=-1)[0].view(-1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            return cam.detach()

    def compute_rs_gradcam(self, tensors, target_class_id, sigma2=0.5, num_mc_samples=8, use_crn=True):
        """
        Tính RS-LiDAR Smoothed Grad-CAM:
        CAM_σ(x) = E_eps [ Grad-CAM(x + eps) ]
        """
        if sigma2 == 0.0 or num_mc_samples <= 0:
            return self.compute_gradcam(tensors, target_class_id)

        cams = []
        for m in range(num_mc_samples):
            if use_crn:
                noise = torch.randn((1, *tensors.shape[1:]), device=tensors.device, dtype=tensors.dtype) * sigma2
            else:
                noise = torch.randn_like(tensors) * sigma2
            noisy_t = (tensors + noise).clamp(-1.0, 1.0)
            cam_m = self.compute_gradcam(noisy_t, target_class_id)
            cams.append(cam_m)

        rs_cam = torch.stack(cams, dim=0).mean(dim=0)
        cam_min = rs_cam.view(rs_cam.size(0), -1).min(dim=-1)[0].view(-1, 1, 1)
        cam_max = rs_cam.view(rs_cam.size(0), -1).max(dim=-1)[0].view(-1, 1, 1)
        rs_cam = (rs_cam - cam_min) / (cam_max - cam_min + 1e-8)
        return rs_cam


# ======================================================================================
# 🖼️ GENERATION & IMAGE PROCESSING
# ======================================================================================
@torch.inference_mode()
def generate_clean_and_perturbed_images(
    pipe, vae, prompt, num_particles, guidance_steps, guidance_scale,
    sigma1, perturbation_type="gaussian", seed=42, device="cuda"
):
    """
    Sinh batch ảnh sạch x_clean và ảnh biến dạng vi nhiễu x_pert.
    """
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
# 🔬 CORE EXPERIMENTAL PIPELINE: MEASURING EMPIRICAL LIPSCHITZ & GRADIENTS
# ======================================================================================
def run_empirical_lipschitz_test(
    pipe, vae, resnet_reward, prompt_list, args, device="cuda"
):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"checkpoint_adversarial_shard_{args.shard_id}.json")

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

    # Đọc danh sách sigma1 (Quét probe radius)
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

    # Đọc danh sách sigma2
    if args.sigma2_list:
        try:
            sigma2_vals = [float(s.strip()) for s in args.sigma2_list.split(",") if s.strip()]
        except Exception:
            sigma2_vals = [0.0, 0.1, 0.25, 0.5, 1.0]
    else:
        sigma2_vals = [args.sigma2]

    if not any(s > 0 for s in sigma2_vals):
        sigma2_vals.append(1.0)
    sigma2_vals = sorted(list(set(sigma2_vals)))

    primary_sigma2 = getattr(args, "sigma2", 0.5)
    if primary_sigma2 not in sigma2_vals:
        non_zero = [s for s in sigma2_vals if s > 0]
        primary_sigma2 = non_zero[-1] if non_zero else sigma2_vals[0]

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

    print(f"\n🚀 [Shard {args.shard_id}/{args.num_shards}] Bắt đầu thực nghiệm Adversarial Reward trên {len(active_prompts)} prompts")
    print(f"⚙️ Tham số: {args.num_particles} hạt/prompt | sigma1={primary_sigma1} (sweep={sigma1_vals})")
    print(f"🔬 Bán kính sigma2 sweep: {sigma2_vals} (primary={primary_sigma2}) | M sweep: {m_vals} (max={max_m}, primary={primary_m})")

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
        target_class = prompt_data["target_class"]
        class_name = prompt_data["class_name"]

        if str(global_idx) in completed_set or prompt in completed_set:
            continue

        prompt_seed = args.seed + global_idx * 100

        # 1. Sinh ảnh sạch x_clean và ảnh biến dạng x_pert (theo primary_sigma1)
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

        # 2. Đánh giá Reward Vanilla: R(x_clean) và R(x_pert)
        with torch.no_grad():
            r_clean_vanilla = resnet_reward.compute_reward(clean_tensors, target_class).cpu().numpy()
            r_pert_vanilla = resnet_reward.compute_reward(pert_tensors, target_class).cpu().numpy()

        # 3. Tính Autograd Gradient Norm cho Vanilla
        _, grad_norm_vanilla = resnet_reward.compute_gradient(clean_tensors, target_class)
        grad_norm_vanilla = grad_norm_vanilla.cpu().numpy()

        # 4. Tính Autograd Gradient Norm cho RS-LiDAR tại primary_sigma2
        rs_grad_norms_by_m = {}
        for M in m_vals:
            _, g_norm_rs = resnet_reward.compute_rs_gradient(
                clean_tensors, target_class, primary_sigma2, num_mc_samples=M, use_crn=getattr(args, "use_crn", True)
            )
            rs_grad_norms_by_m[M] = g_norm_rs.cpu().numpy().tolist()

        # 5. Đánh giá Reward RS-LiDAR Monte Carlo cho từng sigma2 và sweep M (Prefix-mean tối ưu)
        rs_rewards_by_sigma_m = {}
        use_crn = getattr(args, "use_crn", True)
        primary_u_m_list = []

        with torch.no_grad():
            for s2 in sigma2_vals:
                rs_rewards_by_sigma_m[s2] = {}
                if s2 == 0.0:
                    for M in m_vals:
                        rs_rewards_by_sigma_m[s2][M] = {
                            "clean": r_clean_vanilla.tolist(),
                            "pert": r_pert_vanilla.tolist()
                        }
                    continue

                clean_samples = []
                pert_samples = []

                for m_iter in range(max_m):
                    u_m = torch.randn_like(clean_tensors) * s2
                    if s2 == primary_sigma2:
                        primary_u_m_list.append(u_m)

                    noisy_clean_t = (clean_tensors + u_m).clamp(-1.0, 1.0)
                    r_m_clean = resnet_reward.compute_reward(noisy_clean_t, target_class).cpu().numpy()
                    clean_samples.append(r_m_clean)

                    u_prime_m = u_m if use_crn else (torch.randn_like(pert_tensors) * s2)
                    noisy_pert_t = (pert_tensors + u_prime_m).clamp(-1.0, 1.0)
                    r_m_pert = resnet_reward.compute_reward(noisy_pert_t, target_class).cpu().numpy()
                    pert_samples.append(r_m_pert)

                clean_samples = np.array(clean_samples) # [max_m, num_particles]
                pert_samples = np.array(pert_samples)

                for M in m_vals:
                    smoothed_clean_M = np.mean(clean_samples[:M], axis=0).tolist()
                    smoothed_pert_M = np.mean(pert_samples[:M], axis=0).tolist()
                    rs_rewards_by_sigma_m[s2][M] = {
                        "clean": smoothed_clean_M,
                        "pert": smoothed_pert_M
                    }

        # 6. Đánh giá quét bán kính probe sigma1 (đo bùng nổ đạo hàm vi phân)
        sigma1_sweep_results = {}
        with torch.no_grad():
            for s1 in sigma1_vals:
                if s1 == primary_sigma1:
                    pert_t_s1 = pert_tensors
                    delta_x_s1 = delta_x
                    r_pert_v_s1 = r_pert_vanilla
                else:
                    noise_s1 = torch.randn_like(clean_tensors) * s1
                    pert_t_s1 = (clean_tensors + noise_s1).clamp(-1.0, 1.0)
                    delta_x_s1 = torch.norm((clean_tensors - pert_t_s1).view(args.num_particles, -1), p=2, dim=1).cpu().numpy()
                    r_pert_v_s1 = resnet_reward.compute_reward(pert_t_s1, target_class).cpu().numpy()

                # Đo RS cho sigma1 này tại primary_sigma2 và primary_m
                pert_samples_s1 = []
                for m_iter in range(primary_m):
                    u_cand = primary_u_m_list[m_iter] if m_iter < len(primary_u_m_list) else (torch.randn_like(pert_t_s1) * primary_sigma2)
                    noisy_pert_s1 = (pert_t_s1 + u_cand).clamp(-1.0, 1.0)
                    r_m_s1 = resnet_reward.compute_reward(noisy_pert_s1, target_class).cpu().numpy()
                    pert_samples_s1.append(r_m_s1)

                r_pert_rs_s1 = np.mean(pert_samples_s1, axis=0).tolist()
                r_clean_rs_s1 = rs_rewards_by_sigma_m[primary_sigma2][primary_m]["clean"]

                sigma1_sweep_results[s1] = {
                    "delta_x": delta_x_s1.tolist(),
                    "r_pert_vanilla": r_pert_v_s1.tolist(),
                    "r_pert_rs": r_pert_rs_s1,
                    "r_clean_rs": r_clean_rs_s1
                }

        # 7. Lưu trữ dữ liệu cho từng hạt particle
        for p_idx in range(args.num_particles):
            pair_entry = {
                "prompt_idx": global_idx,
                "particle_idx": p_idx,
                "prompt": prompt,
                "target_class": target_class,
                "class_name": class_name,
                "delta_x": float(delta_x[p_idx]),
                "models": {
                    "ResNet50-Adversarial": {
                        "r_clean_vanilla": float(r_clean_vanilla[p_idx]),
                        "r_pert_vanilla": float(r_pert_vanilla[p_idx]),
                        "delta_r_vanilla": float(abs(r_clean_vanilla[p_idx] - r_pert_vanilla[p_idx])),
                        "slope_vanilla": float(abs(r_clean_vanilla[p_idx] - r_pert_vanilla[p_idx]) / max(delta_x[p_idx], 1e-8)),
                        "grad_norm_vanilla": float(grad_norm_vanilla[p_idx]),
                        "rs_grad_norms_by_m": {str(M): float(rs_grad_norms_by_m[M][p_idx]) for M in m_vals},
                        "rs_by_m": {
                            str(M): {
                                str(s2): {
                                    "r_clean_rs": float(rs_rewards_by_sigma_m[s2][M]["clean"][p_idx]),
                                    "r_pert_rs": float(rs_rewards_by_sigma_m[s2][M]["pert"][p_idx]),
                                    "delta_r_rs": float(abs(rs_rewards_by_sigma_m[s2][M]["clean"][p_idx] - rs_rewards_by_sigma_m[s2][M]["pert"][p_idx])),
                                    "slope_rs": float(abs(rs_rewards_by_sigma_m[s2][M]["clean"][p_idx] - rs_rewards_by_sigma_m[s2][M]["pert"][p_idx]) / max(delta_x[p_idx], 1e-8))
                                }
                                for s2 in sigma2_vals
                            }
                            for M in m_vals
                        },
                        "sigma1_sweep": {
                            str(s1): {
                                "delta_x": float(sigma1_sweep_results[s1]["delta_x"][p_idx]),
                                "slope_vanilla": float(abs(r_clean_vanilla[p_idx] - sigma1_sweep_results[s1]["r_pert_vanilla"][p_idx]) / max(sigma1_sweep_results[s1]["delta_x"][p_idx], 1e-8)),
                                "slope_rs": float(abs(sigma1_sweep_results[s1]["r_clean_rs"][p_idx] - sigma1_sweep_results[s1]["r_pert_rs"][p_idx]) / max(sigma1_sweep_results[s1]["delta_x"][p_idx], 1e-8))
                            }
                            for s1 in sigma1_vals
                        }
                    }
                }
            }
            all_pairs_data.append(pair_entry)

        completed_set.add(str(global_idx))
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"completed_prompts": list(completed_set), "pairs_data": all_pairs_data}, f, indent=2)

    print(f"\n✅ Shard {args.shard_id} hoàn tất đo đạc. Đã lưu {len(all_pairs_data)} cặp mẫu tại {ckpt_path}.")


# ======================================================================================
# 🎨 CASE STUDY VISUALIZATIONS: SALIENCY MAPS & 1D LANDSCAPE SLICING
# ======================================================================================
def generate_case_study_saliency(pipe, resnet_reward, output_dir, device="cuda"):
    """
    Sinh Saliency Map trực quan hóa qua Grad-CAM (Layer 4):
    So sánh Grad-CAM của Vanilla (phân tán trên các mẫu hoa văn nền và artifact đối kháng)
    vs Smoothed Grad-CAM của RS-LiDAR (tập trung sắc nét vào vùng đặc trưng thực thể ngữ nghĩa).
    """
    os.makedirs(output_dir, exist_ok=True)
    case_studies = [
        {"prompt": "portrait of a giant panda looking forward", "target_class": 388, "title": "Giant Panda (Class 388)"},
        {"prompt": "portrait of a golden retriever", "target_class": 207, "title": "Golden Retriever (Class 207)"},
        {"prompt": "front view of a shiny black sports car", "target_class": 817, "title": "Sports Car (Class 817)"},
    ]

    fig, axes = plt.subplots(len(case_studies), 3, figsize=(14, 4.4 * len(case_studies)), dpi=300)
    if len(case_studies) == 1:
        axes = [axes]

    print("\n🎨 Đang tạo biểu đồ Grad-CAM Saliency Maps (Layer 4) cho các Case Studies...")

    for row_idx, item in enumerate(case_studies):
        prompt = item["prompt"]
        target_class = item["target_class"]
        title = item["title"]

        generator = torch.Generator(device=device).manual_seed(42 + row_idx * 7)
        with torch.no_grad():
            latents = pipe(
                [prompt],
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
                output_type="latent"
            ).images
            scaled = (latents / pipe.vae.config.scaling_factor).to(device=device, dtype=pipe.vae.dtype)
            clean_tensor = pipe.vae.decode(scaled, return_dict=False)[0].detach().clamp(-1.0, 1.0)

        # 1. Vanilla Grad-CAM (Layer 4)
        cam_v = resnet_reward.compute_gradcam(clean_tensor, target_class)[0].cpu().numpy()

        # 2. RS Smoothed Grad-CAM (Layer 4)
        cam_rs = resnet_reward.compute_rs_gradcam(clean_tensor, target_class, sigma2=0.5, num_mc_samples=8, use_crn=True)[0].cpu().numpy()

        # Saliency Focus Ratio Metrics:
        # A. Central Energy Ratio (%) trong vùng [H/4:3H/4, W/4:3W/4]
        H, W = cam_v.shape
        focus_v = float(cam_v[H//4:3*H//4, W//4:3*W//4].sum() / (cam_v.sum() + 1e-8) * 100.0)
        focus_rs = float(cam_rs[H//4:3*H//4, W//4:3*W//4].sum() / (cam_rs.sum() + 1e-8) * 100.0)

        # B. Top-20% Energy Concentration (%)
        flat_v = np.sort(cam_v.flatten())[::-1]
        k_v = int(len(flat_v) * 0.20)
        top20_v = float(flat_v[:k_v].sum() / (flat_v.sum() + 1e-8) * 100.0)

        flat_rs = np.sort(cam_rs.flatten())[::-1]
        k_rs = int(len(flat_rs) * 0.20)
        top20_rs = float(flat_rs[:k_rs].sum() / (flat_rs.sum() + 1e-8) * 100.0)

        clean_pil = pipe.image_processor.postprocess(clean_tensor.detach(), output_type="pil")[0]

        # Cột 1: Clean Image
        axes[row_idx][0].imshow(clean_pil)
        axes[row_idx][0].set_title(f"{title}\nClean Sample (SD 1.5)", fontsize=11, fontweight="bold")
        axes[row_idx][0].axis("off")

        # Cột 2: Vanilla Grad-CAM Overlay
        axes[row_idx][1].imshow(clean_pil)
        axes[row_idx][1].imshow(cam_v, cmap="jet", alpha=0.55)
        axes[row_idx][1].set_title(
            f"Vanilla Grad-CAM (Layer 4)\nCentral: {focus_v:.1f}% | Top-20%: {top20_v:.1f}%",
            fontsize=10.5, fontweight="bold", color="#dc2626"
        )
        axes[row_idx][1].axis("off")

        # Cột 3: RS Smoothed Grad-CAM Overlay
        axes[row_idx][2].imshow(clean_pil)
        axes[row_idx][2].imshow(cam_rs, cmap="jet", alpha=0.55)
        axes[row_idx][2].set_title(
            f"RS-LiDAR Smoothed Grad-CAM (σ=0.5, M=8)\nCentral: {focus_rs:.1f}% | Top-20%: {top20_rs:.1f}%",
            fontsize=10.5, fontweight="bold", color="#1d4ed8"
        )
        axes[row_idx][2].axis("off")

    fig.suptitle("Visual Saliency via Grad-CAM (Layer 4): Vanilla vs. RS-LiDAR Smoothed Attribution", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = os.path.join(output_dir, "adversarial_visual_saliency_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Đã lưu ảnh Visual Grad-CAM Saliency Grid tại: {out_path}")


def generate_1d_landscape_slice(pipe, resnet_reward, output_dir, device="cuda", primary_sigma2=0.5, primary_m=4):
    """
    Cắt lát cảnh quan 1D qua 2 góc nhìn bổ trợ:
    Panel A (Macro Interpolation): Cắt lát 200 điểm nội suy giữa 2 ảnh sạch x_A và x_B,
            sử dụng Common Random Numbers (CRN) cố định tập mẫu nhiễu Monte Carlo dọc lát cắt
            để đảm bảo không bị sai số lấy mẫu ngẫu nhiên (sampling jitter).
    Panel B (Local Adversarial Direction): Cắt lát vi mô theo hướng gradient đối kháng sign(∇R)
            quanh ảnh sạch x_A, chứng minh Vanilla có độ dốc biến thiên mạnh còn RS-LiDAR được san phẳng theo Theorem 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n📈 Đang thực hiện 1D Landscape Slicing (Macro Interpolation + Local Adversarial Direction)...")

    prompt = "portrait of a giant panda looking forward"
    target_class = 388

    generator1 = torch.Generator(device=device).manual_seed(101)
    generator2 = torch.Generator(device=device).manual_seed(202)

    lat1 = pipe([prompt], num_inference_steps=20, generator=generator1, output_type="latent").images
    lat2 = pipe([prompt], num_inference_steps=20, generator=generator2, output_type="latent").images

    scaled1 = (lat1 / pipe.vae.config.scaling_factor).to(device=device, dtype=pipe.vae.dtype)
    scaled2 = (lat2 / pipe.vae.config.scaling_factor).to(device=device, dtype=pipe.vae.dtype)

    with torch.no_grad():
        img1 = pipe.vae.decode(scaled1, return_dict=False)[0].clamp(-1.0, 1.0)
        img2 = pipe.vae.decode(scaled2, return_dict=False)[0].clamp(-1.0, 1.0)

    # Cố định tập vector nhiễu CRN dọc lát cắt để triệt tiêu sampling jitter
    slice_m = max(primary_m, 16)
    torch.manual_seed(42)
    noise_bank = [torch.randn_like(img1) * primary_sigma2 for _ in range(slice_m)]

    # 1. Panel A: Macro Interpolation x_A -> x_B
    alphas = np.linspace(0.0, 1.0, 200)
    vanilla_rewards_macro = []
    rs_rewards_macro = []

    with torch.no_grad():
        for a in alphas:
            x_alpha = ((1.0 - a) * img1 + a * img2).clamp(-1.0, 1.0)
            r_vanilla = resnet_reward.compute_reward(x_alpha, target_class).item()
            vanilla_rewards_macro.append(r_vanilla)

            mc_r = []
            for u in noise_bank:
                noisy = (x_alpha + u).clamp(-1.0, 1.0)
                mc_r.append(resnet_reward.compute_reward(noisy, target_class).item())
            rs_rewards_macro.append(float(np.mean(mc_r)))

    # 2. Panel B: Local Adversarial Perturbation Direction
    grad_A, _ = resnet_reward.compute_gradient(img1, target_class)
    adv_dir = torch.sign(grad_A) # Direction of fastest increase / spike
    deltas = np.linspace(-0.05, 0.05, 100)
    vanilla_rewards_adv = []
    rs_rewards_adv = []

    with torch.no_grad():
        for d in deltas:
            x_delta = (img1 + d * adv_dir).clamp(-1.0, 1.0)
            r_vanilla = resnet_reward.compute_reward(x_delta, target_class).item()
            vanilla_rewards_adv.append(r_vanilla)

            mc_r = []
            for u in noise_bank:
                noisy = (x_delta + u).clamp(-1.0, 1.0)
                mc_r.append(resnet_reward.compute_reward(noisy, target_class).item())
            rs_rewards_adv.append(float(np.mean(mc_r)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=300)

    # Panel A plot
    axes[0].plot(alphas, vanilla_rewards_macro, color="#dc2626", lw=1.8, label="Vanilla ResNet-50 Logit R(x)")
    axes[0].plot(alphas, rs_rewards_macro, color="#2563eb", lw=2.2, label=f"RS-LiDAR Logit R_σ(x) (σ={primary_sigma2}, M={slice_m})")
    axes[0].set_xlabel("Interpolation Path α: x(α) = (1 - α) x_A + α x_B", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Reward Value (Target Class Logit)", fontsize=10.5, fontweight="bold")
    axes[0].set_title("A. Macro Interpolation Path (CRN Stabilized)", fontsize=11.5, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, fontsize=9.5)

    # Panel B plot
    axes[1].plot(deltas, vanilla_rewards_adv, color="#dc2626", lw=1.8, label="Vanilla ResNet-50 Logit R(x)")
    axes[1].plot(deltas, rs_rewards_adv, color="#2563eb", lw=2.2, label=f"RS-LiDAR Logit R_σ(x) (σ={primary_sigma2}, M={slice_m})")
    axes[1].set_xlabel("Adversarial Perturbation Step δ: x(δ) = x_A + δ · sign(∇R)", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Reward Value (Target Class Logit)", fontsize=10.5, fontweight="bold")
    axes[1].set_title("B. Local Slice along Adversarial Gradient Direction", fontsize=11.5, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, fontsize=9.5)

    fig.suptitle("1D Reward Landscape Slicing: Vanilla vs. RS-LiDAR Smoothed Landscape", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "adversarial_1d_landscape_slice.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Đã lưu đồ thị 1D Landscape Slicing tại: {out_path}")


def generate_case_study_cohen_certificates(pipe, resnet_reward, output_dir, device="cuda", sigma2=0.5, num_noise_samples=100):
    """
    Thực nghiệm C (Mục 4.3 trong Plan): Cohen Certified Robustness Radius
    Đo đạc bán kính bền vững được chứng thực toán học theo Cohen et al. (2019):
    R_cert = σ₂ * Φ⁻¹(p_A) với p_A = Pr[ argmax f(x + ε) == target_class ]
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n🛡️ Đang đo đạc Certified Robustness Radius (Cohen Certificate) cho các Case Studies...")
    case_studies = [
        {"prompt": "portrait of a giant panda looking forward", "target_class": 388, "title": "Giant Panda"},
        {"prompt": "portrait of a golden retriever", "target_class": 207, "title": "Golden Retriever"},
        {"prompt": "front view of a shiny black sports car", "target_class": 817, "title": "Sports Car"},
    ]

    cohen_records = []
    for row_idx, item in enumerate(case_studies):
        prompt = item["prompt"]
        target_class = item["target_class"]
        title = item["title"]

        generator = torch.Generator(device=device).manual_seed(42 + row_idx * 7)
        with torch.no_grad():
            latents = pipe([prompt], num_inference_steps=25, generator=generator, output_type="latent").images
            scaled = (latents / pipe.vae.config.scaling_factor).to(device=device, dtype=pipe.vae.dtype)
            clean_tensor = pipe.vae.decode(scaled, return_dict=False)[0].detach().clamp(-1.0, 1.0)

            # Đánh giá 100 mẫu Gaussian noise theo batch 25 để tối ưu tốc độ
            correct_count = 0
            batch_size = 25
            for b_start in range(0, num_noise_samples, batch_size):
                curr_b = min(batch_size, num_noise_samples - b_start)
                noise = torch.randn((curr_b, *clean_tensor.shape[1:]), device=clean_tensor.device, dtype=clean_tensor.dtype) * sigma2
                noisy_batch = (clean_tensor.expand(curr_b, -1, -1, -1) + noise).clamp(-1.0, 1.0)
                prep = resnet_reward.preprocess(noisy_batch)
                preds = resnet_reward.model(prep).argmax(dim=-1)
                correct_count += (preds == target_class).sum().item()

            p_A = correct_count / num_noise_samples
            if correct_count >= num_noise_samples:
                p_A_lower = float(0.001 ** (1.0 / num_noise_samples))
            elif correct_count > 0:
                p_A_lower = float(scipy.stats.beta.ppf(0.001, correct_count, num_noise_samples - correct_count + 1))
            else:
                p_A_lower = 0.0

            # Giới hạn p_A an toàn tránh inf từ scipy.stats.norm.ppf(1.0)
            p_A_safe = min(max(p_A, 0.0), 1.0 - 1e-4)
            if p_A > 0.5:
                r_cert = float(sigma2 * scipy.stats.norm.ppf(p_A_safe))
                r_cert_cp = float(sigma2 * scipy.stats.norm.ppf(p_A_lower)) if p_A_lower > 0.5 else 0.0
            else:
                r_cert = 0.0
                r_cert_cp = 0.0

            cohen_records.append({
                "class_name": title,
                "target_class": target_class,
                "p_A": p_A,
                "p_A_lower": p_A_lower,
                "r_cert": r_cert,
                "r_cert_cp": r_cert_cp
            })

    # Vẽ biểu đồ Bar Chart so sánh Cohen Certified Radius
    classes = [r["class_name"] for r in cohen_records]
    r_certs = [r["r_cert"] for r in cohen_records]
    r_cert_cps = [r["r_cert_cp"] for r in cohen_records]
    p_As = [r["p_A"] * 100 for r in cohen_records]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    bars = ax.bar(classes, r_certs, color='#2563eb', alpha=0.85, width=0.45, label=f'RS-LiDAR Empirical Radius (σ={sigma2})')
    ax.axhline(0.0, color='#dc2626', linestyle='--', lw=2.0, label='Vanilla Certified Radius = 0.0 (Uncertified)')
    for bar, p_val, r_cp in zip(bars, p_As, r_cert_cps):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015,
                f"R_emp = {yval:.3f}\nR_cert = {r_cp:.3f}\n(Top-1 = {p_val:.0f}%)",
                ha='center', va='bottom', fontsize=9.0, fontweight='semibold')

    ax.set_ylabel("Cohen Certified $\\ell_2$ Radius", fontsize=11, fontweight="bold")
    ax.set_title(f"Cohen Certified Robustness Radius (Randomized Smoothing σ={sigma2})", fontsize=12, fontweight="bold")
    max_h = max([b.get_height() for b in bars] + [0.1])
    ax.set_ylim(0, max(max_h * 1.45, 0.6))
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    cohen_plot_path = os.path.join(output_dir, "adversarial_cohen_certified_radius.png")
    plt.savefig(cohen_plot_path, dpi=300)
    plt.close()
    print(f"🛡️ Đã lưu đồ thị Cohen Certified Radius tại: {cohen_plot_path}")

    # Xuất Markdown bảng Cohen Certificates
    md_cohen_path = os.path.join(output_dir, "adversarial_cohen_certificates.md")
    with open(md_cohen_path, "w", encoding="utf-8") as f:
        f.write("# 🛡️ Cohen Certified Robustness Radius (Randomized Smoothing)\n\n")
        f.write(f"Theo Cohen et al. (ICML 2019): Bán kính chứng nhận bền vững $\\ell_2$ tại $\\sigma_2 = {sigma2}$:\n\n")
        f.write("$$R_{\\text{cert}} = \\sigma_2 \\cdot \\Phi^{-1}(\\underline{p_A})$$\n\n")
        f.write("| Case Study | Target Class | Vanilla $R_{\\text{cert}}$ | RS-LiDAR $p_A$ (Top-1) | RS-LiDAR $R_{\\text{cert}}$ (Empirical) | RS-LiDAR $R_{\\text{cert}}^{\\text{CP}}$ (99.9% Bound) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for rec in cohen_records:
            f.write(f"| **{rec['class_name']}** | {rec['target_class']} | 0.000 (Uncertified) | {rec['p_A']*100:.1f}% | {rec['r_cert']:.4f} | {rec['r_cert_cp']:.4f} |\n")
        f.write("\n> *Vanilla classifier không có bảo đảm chứng nhận bền vững (R_cert = 0). Dưới RS-LiDAR, mô hình được chứng thực toán học không thể bị lật nhãn bởi bất kỳ nhiễu đối kháng nào nằm trong quả cầu bán kính R_cert.*\n")
    print(f"📝 Đã xuất Markdown Cohen Certificates tại: {md_cohen_path}")



# ======================================================================================
# 📊 UTILITIES: RENDER BẢNG RA HÌNH ẢNH MATPLOTLIB
# ======================================================================================
def render_table_as_image(table_data, col_labels, title, output_path, col_widths=None):
    """Render bảng dữ liệu thành ảnh PNG độ nét cao, zebra striping chuẩn hội nghị."""
    fig_w = max(10, len(col_labels) * 1.45)
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
            cell.set_facecolor('#1e3a8a')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 1:
                cell.set_facecolor('#f9fafb')
            else:
                cell.set_facecolor('#ffffff')
            if col == 0:
                cell.set_text_props(fontweight='semibold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================================================
# 📊 AGGREGATION, TABLES GENERATION & PLOTTING
# ======================================================================================
def merge_and_plot_results(output_dir, primary_sigma2=0.5, primary_m=4, primary_sigma1=0.05):
    ckpt_files = glob.glob(os.path.join(output_dir, "checkpoint_adversarial_shard_*.json"))
    if not ckpt_files:
        print("⚠️ Không tìm thấy file checkpoint_adversarial_shard_*.json nào để gộp.")
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

    # Khám phá tham số
    available_ms = set()
    available_sigmas = set()
    available_sigma1s = set()

    for p in merged_pairs:
        m_data = p["models"].get("ResNet50-Adversarial")
        if not m_data: continue
        if "rs_by_m" in m_data:
            for M_str in m_data["rs_by_m"].keys():
                available_ms.add(int(M_str))
                for s2_str in m_data["rs_by_m"][M_str].keys():
                    available_sigmas.add(float(s2_str))
        if "sigma1_sweep" in m_data:
            for s1_str in m_data["sigma1_sweep"].keys():
                available_sigma1s.add(float(s1_str))

    available_ms = sorted(list(available_ms)) if available_ms else [1, 2, 4, 8]
    available_sigmas = sorted(list(available_sigmas)) if available_sigmas else [0.0, 0.1, 0.25, 0.5, 1.0]
    available_sigma1s = sorted(list(available_sigma1s))

    if primary_sigma2 not in available_sigmas:
        non_zero = [s for s in available_sigmas if s > 0]
        primary_sigma2 = non_zero[-1] if non_zero else available_sigmas[0]

    if primary_m not in available_ms:
        primary_m = available_ms[-1]

    # Thu thập số liệu cơ bản
    vanilla_slopes = []
    vanilla_grad_norms = []
    vanilla_clean_rewards = []
    rs_clean_primary = []
    rs_slopes_primary = []

    for p in merged_pairs:
        m_data = p["models"]["ResNet50-Adversarial"]
        vanilla_slopes.append(m_data["slope_vanilla"])
        vanilla_grad_norms.append(m_data.get("grad_norm_vanilla", 0.0))
        vanilla_clean_rewards.append(m_data["r_clean_vanilla"])
        rs_clean_primary.append(m_data["rs_by_m"][str(primary_m)][str(primary_sigma2)]["r_clean_rs"])
        rs_slopes_primary.append(m_data["rs_by_m"][str(primary_m)][str(primary_sigma2)]["slope_rs"])

    vanilla_slopes = np.array(vanilla_slopes)
    vanilla_grad_norms = np.array(vanilla_grad_norms)
    vanilla_clean_rewards = np.array(vanilla_clean_rewards)
    rs_clean_primary = np.array(rs_clean_primary)
    rs_slopes_primary = np.array(rs_slopes_primary)

    L_mean_vanilla = float(np.mean(vanilla_slopes))
    L_max_vanilla = float(np.max(vanilla_slopes))
    L_95_vanilla = float(np.percentile(vanilla_slopes, 95))
    G_mean_vanilla = float(np.mean(vanilla_grad_norms))
    G_max_vanilla = float(np.max(vanilla_grad_norms))

    # Tính toán ma trận stats theo M và sigma2
    stats_matrix = {M: {} for M in available_ms}
    for M in available_ms:
        for s2 in available_sigmas:
            s_list = []
            g_list = []
            for p in merged_pairs:
                m_data = p["models"]["ResNet50-Adversarial"]
                entry = m_data["rs_by_m"][str(M)][str(s2)]
                s_list.append(entry["slope_rs"])
                # Gradient norm nếu có
                if "rs_grad_norms_by_m" in m_data and str(M) in m_data["rs_grad_norms_by_m"]:
                    g_list.append(m_data["rs_grad_norms_by_m"][str(M)])
            s_arr = np.array(s_list)
            g_arr = np.array(g_list) if g_list else np.array([0.0])
            stats_matrix[M][s2] = {
                "L_mean": float(np.mean(s_arr)),
                "L_max": float(np.max(s_arr)),
                "L_95": float(np.percentile(s_arr, 95)),
                "G_mean": float(np.mean(g_arr)),
                "G_max": float(np.max(g_arr)),
                "contraction": float(L_mean_vanilla / max(np.mean(s_arr), 1e-8))
            }

    # ==================================================================================
    # 📊 BẢNG 1: LIPSCHITZ & GRADIENT NORM BENCHMARK (Table 1 Main Text)
    # ==================================================================================
    table1_headers = ["Method / Setting", "L_mean", "L_max", "L_95%", "||∇R||_mean", "Contraction"]
    table1_rows = []

    # Row 1: Vanilla
    table1_rows.append([
        "Vanilla ResNet-50",
        f"{L_mean_vanilla:.6f}",
        f"{L_max_vanilla:.6f}",
        f"{L_95_vanilla:.6f}",
        f"{G_mean_vanilla:.2f}",
        "1.00× (Baseline)"
    ])

    for M in available_ms:
        s_info = stats_matrix[M][primary_sigma2]
        table1_rows.append([
            f"RS-LiDAR (σ={primary_sigma2}, M={M})",
            f"{s_info['L_mean']:.6f}",
            f"{s_info['L_max']:.6f}",
            f"{s_info['L_95']:.6f}",
            f"{s_info['G_mean']:.2f}",
            f"{s_info['contraction']:.2f}×"
        ])

    table1_img_path = os.path.join(output_dir, "adversarial_table_benchmark.png")
    render_table_as_image(table1_rows, table1_headers, f"Adversarial Reward: Lipschitz & Gradient Benchmark (σ = {primary_sigma2})", table1_img_path)
    print(f"📊 Đã lưu ảnh Bảng 1 tại: {table1_img_path}")

    # ==================================================================================
    # 📊 BẢNG 2: BREAKDOWN THEO 10 SUPER-CATEGORIES
    # ==================================================================================
    class_groups = {}
    for p in merged_pairs:
        c_name = p.get("class_name", "unknown")
        if c_name not in class_groups:
            class_groups[c_name] = {"vanilla_slopes": [], "rs_slopes": []}
        m_data = p["models"]["ResNet50-Adversarial"]
        class_groups[c_name]["vanilla_slopes"].append(m_data["slope_vanilla"])
        class_groups[c_name]["rs_slopes"].append(m_data["rs_by_m"][str(primary_m)][str(primary_sigma2)]["slope_rs"])

    table_cat_headers = ["Super-Category (Class)", "Vanilla L_mean", f"RS-LiDAR (M={primary_m})", "Contraction Ratio"]
    table_cat_rows = []
    for c_name, data in class_groups.items():
        v_mean = np.mean(data["vanilla_slopes"])
        rs_mean = np.mean(data["rs_slopes"])
        ratio = v_mean / max(rs_mean, 1e-8)
        table_cat_rows.append([
            c_name.capitalize(),
            f"{v_mean:.6f}",
            f"{rs_mean:.6f}",
            f"{ratio:.2f}×"
        ])

    table_cat_path = os.path.join(output_dir, "adversarial_table_categories_breakdown.png")
    render_table_as_image(table_cat_rows, table_cat_headers, "Adversarial Reward: 10 Super-Categories Robustness Breakdown", table_cat_path)
    print(f"📊 Đã lưu ảnh Bảng Breakdown 10 classes tại: {table_cat_path}")

    # ==================================================================================
    # 📊 BẢNG 3: KHẢO SÁT BÁN KÍNH VI NHIỄU SIGMA1 -> 0 (Spike Singularity & Bound)
    # ==================================================================================
    sigma1_stats = {}
    table3_headers = ["Probe Radius σ1", "Mean ||Δx||", "Vanilla L_max → RS", "Max Contraction", "Vanilla L_mean → RS", "Mean Contraction"]
    table3_rows = []
    if available_sigma1s:
        for s1 in available_sigma1s:
            s1_key = str(s1)
            v_slopes_s1 = []
            rs_slopes_s1 = []
            dx_s1_list = []
            for p in merged_pairs:
                m_data = p["models"]["ResNet50-Adversarial"]
                if "sigma1_sweep" in m_data and s1_key in m_data["sigma1_sweep"]:
                    entry = m_data["sigma1_sweep"][s1_key]
                    v_slopes_s1.append(entry["slope_vanilla"])
                    rs_slopes_s1.append(entry["slope_rs"])
                    dx_s1_list.append(entry["delta_x"])

            if v_slopes_s1:
                v_slopes_s1 = np.array(v_slopes_s1)
                rs_slopes_s1 = np.array(rs_slopes_s1)
                mean_dx = float(np.mean(dx_s1_list))
                l_max_v = float(np.max(v_slopes_s1))
                l_max_rs = float(np.max(rs_slopes_s1))
                l_mean_v = float(np.mean(v_slopes_s1))
                l_mean_rs = float(np.mean(rs_slopes_s1))
                r_max = l_max_v / max(l_max_rs, 1e-8)
                r_mean = l_mean_v / max(l_mean_rs, 1e-8)

                sigma1_stats[s1_key] = {
                    "mean_dx": mean_dx,
                    "l_max_vanilla": l_max_v, "l_max_rs": l_max_rs, "ratio_max": r_max,
                    "l_mean_vanilla": l_mean_v, "l_mean_rs": l_mean_rs, "ratio_mean": r_mean
                }
                is_p = (s1 == primary_sigma1)
                p_tag = " *(Primary)*" if is_p else ""
                table3_rows.append([
                    f"{s1}{p_tag}",
                    f"{mean_dx:.3f}",
                    f"{l_max_v:.4f} → {l_max_rs:.4f}",
                    f"{r_max:.2f}×",
                    f"{l_mean_v:.4f} → {l_mean_rs:.4f}",
                    f"{r_mean:.2f}×"
                ])

        table3_img_path = os.path.join(output_dir, "adversarial_table_sigma1_sweep.png")
        render_table_as_image(table3_rows, table3_headers, f"Adversarial Reward: Probe Radius σ1 Sweep (σ2 = {primary_sigma2}, M = {primary_m})", table3_img_path)
        print(f"📊 Đã lưu ảnh Bảng 3 (sigma1 sweep) tại: {table3_img_path}")

        # Vẽ đồ thị đường quét sigma1 ablation
        if len(available_sigma1s) > 1:
            plt.figure(figsize=(14, 5.5), dpi=300)
            plt.subplot(1, 2, 1)
            y_max_v = [sigma1_stats[str(s)]["l_max_vanilla"] for s in available_sigma1s]
            y_max_rs = [sigma1_stats[str(s)]["l_max_rs"] for s in available_sigma1s]
            plt.plot(available_sigma1s, y_max_v, 'r-o', lw=2.2, label='Vanilla ResNet-50 $L_{max}$')
            plt.plot(available_sigma1s, y_max_rs, 'b-s', lw=2.2, label=f'RS-LiDAR $L_{{max}}$ (σ2={primary_sigma2}, M={primary_m})')
            plt.xscale('log')
            plt.xlabel('Probe Radius $\\sigma_1$ (Log Scale)', fontsize=11, fontweight='bold')
            plt.ylabel('Lipschitz $L_{max}$', fontsize=11, fontweight='bold')
            plt.title('Spike Explosion: $L_{max}$ vs. $\\sigma_1 \\to 0$', fontsize=12, fontweight='bold')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(frameon=True)

            plt.subplot(1, 2, 2)
            y_mean_v = [sigma1_stats[str(s)]["l_mean_vanilla"] for s in available_sigma1s]
            y_mean_rs = [sigma1_stats[str(s)]["l_mean_rs"] for s in available_sigma1s]
            plt.plot(available_sigma1s, y_mean_v, 'r-o', lw=2.2, label='Vanilla $L_{mean}$')
            plt.plot(available_sigma1s, y_mean_rs, 'b-s', lw=2.2, label=f'RS-LiDAR $L_{{mean}}$ (σ2={primary_sigma2}, M={primary_m})')
            plt.xscale('log')
            plt.xlabel('Probe Radius $\\sigma_1$ (Log Scale)', fontsize=11, fontweight='bold')
            plt.ylabel('Lipschitz $L_{mean}$', fontsize=11, fontweight='bold')
            plt.title('Mean Slope: Vanilla vs. RS-LiDAR Bounded', fontsize=12, fontweight='bold')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.legend(frameon=True)

            plt.tight_layout()
            sigma1_plot_path = os.path.join(output_dir, "adversarial_sigma1_ablation.png")
            plt.savefig(sigma1_plot_path, dpi=300)
            plt.close()
            print(f"📈 Đã lưu đồ thị sigma1 ablation tại: {sigma1_plot_path}")

    # ==================================================================================
    # 📝 XUẤT CÁC BẢNG BÁO CÁO MARKDOWN
    # ==================================================================================
    tbl1_md_path = os.path.join(output_dir, "adversarial_lipschitz_benchmark_table.md")
    with open(tbl1_md_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 Adversarial Reward Lipschitz & Gradient Benchmark (ResNet-50)\n\n")
        f.write(f"### 1. Bảng 1: Benchmark Chính Tại Bán Kính Làm Mịn Chuẩn $\\sigma_2 = {primary_sigma2}$\n\n")
        f.write("| " + " | ".join(table1_headers) + " |\n")
        f.write("| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(table1_headers))]) + " |\n")
        for row in table1_rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n---\n\n")

        if table3_rows:
            f.write(f"### 2. Bảng 3: Khảo Sát Bán Kính Vi Nhiễu $\\sigma_1 \\to 0$ Tại $\\sigma_2 = {primary_sigma2}, M = {primary_m}$\n\n")
            f.write("> *Khảo sát hiện tượng bùng nổ gai đối kháng: Khi $\\sigma_1 \\to 0$, khoảng cách vi phân $\\|\\Delta x\\|_2 \\to 0$, khiến độ dốc cục bộ của Vanilla tăng vọt do các gai nhọn đối kháng tần số cao. RS-LiDAR dập tắt triệt để các gai này, giữ độ dốc luôn bị chặn theo Theorem 1.*\n\n")
            f.write("| " + " | ".join(table3_headers) + " |\n")
            f.write("| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(table3_headers))]) + " |\n")
            for row in table3_rows:
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
    print(f"📝 Đã xuất Markdown Bảng 1 & 3 tại: {tbl1_md_path}")

    tbl2_md_path = os.path.join(output_dir, "adversarial_class_breakdown_table.md")
    with open(tbl2_md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Adversarial Reward Breakdown: 10 ImageNet Super-Categories\n\n")
        f.write(f"### Bảng 2: Phân Tích Độ Bền Vững & Tỷ Lệ Co Rút Trên 10 Nhóm Lớp Mục Tiêu ($\\sigma_2 = {primary_sigma2}, M = {primary_m}$)\n\n")
        f.write("| " + " | ".join(table_cat_headers) + " |\n")
        f.write("| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(table_cat_headers))]) + " |\n")
        for row in table_cat_rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")
    print(f"📝 Đã xuất Markdown Bảng 2 tại: {tbl2_md_path}")

    # ==================================================================================
    # 📈 BIỂU ĐỒ LINE GRAPH ABLATION THEO SIGMA2 & CONVERGENCE M
    # ==================================================================================
    plt.figure(figsize=(14, 5.5), dpi=300)
    plt.subplot(1, 2, 1)
    colors_m = {1: '#f87171', 2: '#fbbf24', 4: '#38bdf8', 8: '#1d4ed8'}
    for M in available_ms:
        means = [stats_matrix[M][s2]["L_mean"] for s2 in available_sigmas]
        plt.plot(available_sigmas, means, marker='o', color=colors_m.get(M, '#6366f1'), lw=2.2, label=f"RS (M={M})")
    plt.axhline(L_mean_vanilla, color="#dc2626", linestyle="--", lw=1.8, label="Vanilla $L_{mean}$")
    plt.xlabel("Smoothing Radius $\\sigma_2$", fontsize=11, fontweight="bold")
    plt.ylabel("Lipschitz $L_{mean}$", fontsize=11, fontweight="bold")
    plt.title("Ablation: $L_{mean}$ Contraction vs. $\\sigma_2$", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    plt.subplot(1, 2, 2)
    for M in available_ms:
        maxs = [stats_matrix[M][s2]["L_max"] for s2 in available_sigmas]
        plt.plot(available_sigmas, maxs, marker='s', color=colors_m.get(M, '#6366f1'), lw=2.2, label=f"RS (M={M})")
    plt.axhline(L_max_vanilla, color="#dc2626", linestyle="--", lw=1.8, label="Vanilla $L_{max}$")
    plt.xlabel("Smoothing Radius $\\sigma_2$", fontsize=11, fontweight="bold")
    plt.ylabel("Lipschitz $L_{max}$", fontsize=11, fontweight="bold")
    plt.title("Extremal Bound: $L_{max}$ vs. $\\sigma_2$", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    plt.tight_layout()
    ablation_sigma_path = os.path.join(output_dir, "adversarial_sigma_ablation.png")
    plt.savefig(ablation_sigma_path, dpi=300)
    plt.close()
    print(f"📈 Đã lưu đồ thị Line Graph Ablation tại: {ablation_sigma_path}")

    # ==================================================================================
    # 📈 BIỂU ĐỒ PHÂN PHỐI GRADIENT NORM (KDE)
    # ==================================================================================
    plt.figure(figsize=(9, 5), dpi=300)
    rs_primary_grads = []
    for p in merged_pairs:
        m_data = p["models"]["ResNet50-Adversarial"]
        if "rs_grad_norms_by_m" in m_data and str(primary_m) in m_data["rs_grad_norms_by_m"]:
            rs_primary_grads.append(m_data["rs_grad_norms_by_m"][str(primary_m)])
    rs_primary_grads = np.array(rs_primary_grads) if rs_primary_grads else np.array([1.0])

    plt.hist(vanilla_grad_norms, bins=25, alpha=0.55, color="#dc2626", density=True, label="Vanilla Gradient Norm ||∇R||")
    plt.hist(rs_primary_grads, bins=25, alpha=0.65, color="#2563eb", density=True, label=f"RS-LiDAR Smoothed ||∇R_σ|| (σ={primary_sigma2}, M={primary_m})")
    plt.xlabel("Gradient Norm Magnitude", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title("Gradient Norm Suppression: Vanilla Spikes vs. RS Bounded Norm", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    grad_plot_path = os.path.join(output_dir, "adversarial_gradient_norm_dist.png")
    plt.savefig(grad_plot_path, dpi=300)
    plt.close()
    print(f"📈 Đã lưu đồ thị Gradient Norm Distribution tại: {grad_plot_path}")

    # ==================================================================================
    # 📊 DASHBOARD 6-PANEL TỔNG HỢP (Combined Dashboard)
    # ==================================================================================
    fig = plt.figure(figsize=(19, 11), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)

    # Panel 1: Table snippet
    ax_tab = fig.add_subplot(gs[0, 0])
    ax_tab.axis('off')
    ax_tab.set_title("A. Lipschitz Benchmark Summary", fontsize=12, fontweight='bold', pad=8)
    table_snippet_headers = table1_headers
    table_snippet_rows = table1_rows[:5]
    col_widths = [0.26, 0.14, 0.14, 0.14, 0.15, 0.17]
    tab = ax_tab.table(cellText=table_snippet_rows, colLabels=table_snippet_headers, colWidths=col_widths, loc='center', cellLoc='center', bbox=[-0.02, 0.05, 1.04, 0.88])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#d1d5db')
        if r == 0:
            cell.set_facecolor('#1e3a8a')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#f9fafb')

    # Panel 2: L_mean vs sigma2
    ax_line = fig.add_subplot(gs[0, 1])
    for M in available_ms:
        means = [stats_matrix[M][s2]["L_mean"] for s2 in available_sigmas]
        ax_line.plot(available_sigmas, means, marker='o', lw=2.0, color=colors_m.get(M, '#6366f1'), label=f'M={M}')
    ax_line.axhline(L_mean_vanilla, color='#dc2626', linestyle='--', lw=1.6, label='Vanilla')
    ax_line.set_xlabel('Smoothing Radius $\\sigma_2$', fontsize=10.5, fontweight='bold')
    ax_line.set_ylabel('$L_{mean}$', fontsize=10.5, fontweight='bold')
    ax_line.set_title('B. Lipschitz Contraction Curve', fontsize=12, fontweight='bold')
    ax_line.grid(True, linestyle='--', alpha=0.4)
    ax_line.legend(fontsize=9)

    # Panel 3: Gradient Norm Distribution
    ax_grad = fig.add_subplot(gs[0, 2])
    ax_grad.hist(vanilla_grad_norms, bins=20, alpha=0.55, color='#dc2626', density=True, label='Vanilla ||∇R||')
    ax_grad.hist(rs_primary_grads, bins=20, alpha=0.65, color='#2563eb', density=True, label=f'RS ||∇R_σ||')
    ax_grad.set_xlabel('Gradient Norm', fontsize=10.5, fontweight='bold')
    ax_grad.set_ylabel('Density', fontsize=10.5, fontweight='bold')
    ax_grad.set_title('C. Gradient Norm Distribution', fontsize=12, fontweight='bold')
    ax_grad.grid(True, linestyle='--', alpha=0.4)
    ax_grad.legend(fontsize=9)

    # Panel 4: Reward Distribution Smoothing
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.hist(vanilla_clean_rewards, bins=20, alpha=0.55, color='#dc2626', density=True, label='Vanilla Logits')
    ax_dist.hist(rs_clean_primary, bins=20, alpha=0.65, color='#2563eb', density=True, label='RS Logits')
    ax_dist.set_xlabel('Target Class Logit Value', fontsize=10.5, fontweight='bold')
    ax_dist.set_ylabel('Density', fontsize=10.5, fontweight='bold')
    ax_dist.set_title('D. Reward Distribution Smoothing', fontsize=12, fontweight='bold')
    ax_dist.grid(True, linestyle='--', alpha=0.4)
    ax_dist.legend(fontsize=9)

    # Panel 5: Slope Contraction Scatter
    ax_scat = fig.add_subplot(gs[1, 1])
    max_val = max(np.percentile(vanilla_slopes, 98), np.percentile(rs_slopes_primary, 98))
    ax_scat.scatter(vanilla_slopes, rs_slopes_primary, alpha=0.55, color='#7c3aed', s=16)
    ax_scat.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='y = x (No reduction)')
    ax_scat.set_xlabel('Vanilla Slope', fontsize=10.5, fontweight='bold')
    ax_scat.set_ylabel(f'RS Slope (M={primary_m})', fontsize=10.5, fontweight='bold')
    ax_scat.set_xlim(0, max_val)
    ax_scat.set_ylim(0, max_val)
    ax_scat.set_title('E. Slope Contraction Scatter', fontsize=12, fontweight='bold')
    ax_scat.grid(True, linestyle='--', alpha=0.4)
    ax_scat.legend(fontsize=9)

    # Panel 6: Super-Categories Contraction Bar Chart
    ax_cat = fig.add_subplot(gs[1, 2])
    cat_names = [c.capitalize() for c in class_groups.keys()][:8]
    cat_ratios = [np.mean(class_groups[c.lower()]["vanilla_slopes"]) / max(np.mean(class_groups[c.lower()]["rs_slopes"]), 1e-8) for c in cat_names]
    ax_cat.barh(cat_names, cat_ratios, color='#0284c7', alpha=0.85)
    ax_cat.set_xlabel('Contraction Ratio (× reduction)', fontsize=10.5, fontweight='bold')
    ax_cat.set_title('F. Contraction Across Classes', fontsize=12, fontweight='bold')
    ax_cat.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(f"Adversarial Reward Analysis Dashboard (ResNet-50 | RS-LiDAR σ={primary_sigma2}, M={primary_m})", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    dash_path = os.path.join(output_dir, "adversarial_lipschitz_combined_dashboard.png")
    plt.savefig(dash_path, dpi=300)
    plt.close()
    print(f"📊 Đã lưu Dashboard gộp tại: {dash_path}")

    # Xuất metrics json
    summary_path = os.path.join(output_dir, "summary_adversarial_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": len(merged_pairs),
            "L_mean_vanilla": L_mean_vanilla,
            "L_max_vanilla": L_max_vanilla,
            "L_95_vanilla": L_95_vanilla,
            "grad_norm_mean_vanilla": G_mean_vanilla,
            "stats_matrix": stats_matrix,
            "sigma1_sweep": sigma1_stats
        }, f, indent=4)
    print(f"📁 Đã xuất summary metrics tại: {summary_path}")


# ======================================================================================
# 🏁 MAIN ENTRY POINT
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Đo đạc Hệ số Lipschitz & Gradient cho Adversarial Reward bằng ResNet-50")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_particles", type=int, default=10)
    parser.add_argument("--guidance_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma1", type=float, default=0.05)
    parser.add_argument("--sigma1_list", type=str, default="0.001,0.005,0.01,0.02,0.05", help="Dải quét probe radius sigma1")
    parser.add_argument("--primary_sigma1", type=float, default=None, help="Bán kính vi nhiễu chính (mặc định theo sigma1)")
    parser.add_argument("--perturbation_type", type=str, default="gaussian", choices=["gaussian", "blur"])
    parser.add_argument("--sigma2", type=float, default=0.5)
    parser.add_argument("--sigma2_list", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--num_mc_samples", "-M", type=int, default=4)
    parser.add_argument("--m_list", type=str, default="1,2,4,8", help="Dải quét số mẫu Monte Carlo M")
    parser.add_argument("--primary_m", type=int, default=4, help="Mức M chuẩn làm mốc so sánh chính")
    parser.add_argument("--reward_mode", type=str, default="logit", choices=["logit", "softmax", "margin"])
    parser.add_argument("--use_crn", action="store_true", default=True, help="Sử dụng Common Random Numbers (CRN) triệt tiêu phương sai Monte Carlo")
    parser.add_argument("--no_crn", action="store_false", dest="use_crn", help="Tắt Common Random Numbers (dùng mẫu độc lập)")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/New_Rewards/Adversarial_Rewards")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--run_case_studies", action="store_true", help="Tự động sinh Saliency Maps & 1D Landscape Slicing")
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

    # 2. Khởi tạo ResNet-50 Adversarial Reward
    print("📥 Đang nạp mô hình ResNet-50 (ImageNet Pretrained)...")
    resnet_reward = ResNet50AdversarialReward(device=device, reward_mode=args.reward_mode)

    # 3. Chạy thực nghiệm đo đạc chính
    run_empirical_lipschitz_test(
        pipe, vae, resnet_reward, ADVERSARIAL_REWARD_PROMPTS, args, device=device
    )

    # 5. Chạy Case Studies trực quan hóa (nếu được yêu cầu hoặc shard 0 hoàn thành)
    if args.run_case_studies and args.shard_id == 0:
        try:
            generate_case_study_saliency(pipe, resnet_reward, args.output_dir, device=device)
            generate_1d_landscape_slice(pipe, resnet_reward, args.output_dir, device=device, primary_sigma2=args.sigma2, primary_m=args.primary_m)
            generate_case_study_cohen_certificates(pipe, resnet_reward, args.output_dir, device=device, sigma2=args.sigma2)
        except Exception as e:
            print(f"⚠️ Lỗi sinh Case Studies: {e}")

    # 6. Tự động merge và vẽ biểu đồ nếu chạy 1 shard
    if args.num_shards == 1:
        merge_and_plot_results(
            args.output_dir,
            primary_sigma2=args.sigma2,
            primary_m=args.primary_m,
            primary_sigma1=(args.primary_sigma1 if args.primary_sigma1 is not None else args.sigma1)
        )


if __name__ == "__main__":
    main()
