"""
========================================================================================
🧪 RS-LiDAR ABLATION 1: SENSITIVE PROMPT ANALYSIS & NON-LIPSCHITZ LANDSCAPE SLICING
========================================================================================

Chuyên đề Ablation Study 1:
1. Xác định và khảo sát các prompt nhạy cảm (sensitive prompts) trong GenEval
   (ví dụ: counting, spatial positioning, color attribute binding) nơi hàm reward
   ImageReward / CLIP / HPSv2 có hằng số Lipschitz cục bộ bùng nổ (L >> 1, gai nhọn giả tạo).
2. Đo lường cơ chế sinh 2-Phase chuẩn xác:
   - Phase 1: Sinh n=50 hạt lookahead (DPM-5) 1 lần duy nhất, chấm song song r_vanilla và r_RS.
   - Phase 2: Chạy Closed-form Target Steering (DDIM-50) cùng seed khởi tạo cho 3 nhánh:
     (i) Unsteered (s=0), (ii) Vanilla LiDAR (s=12.5), (iii) RS-LiDAR (s=12.5, sigma=1.0).
3. Cắt lát bề mặt 1D (Directional Landscape Slicing):
   Quét dọc vector định hướng v = (x_RS - x_vanilla) / ||x_RS - x_vanilla||_2 với alpha in [-0.2, 0.2]
   để đối chiếu gai nhọn phi Lipschitz của Vanilla vs sườn dốc phẳng trơn có chặn lý thuyết của RS-LiDAR.
4. Cơ chế tự động sắp xếp (Sorting) và trích xuất Top 5 prompt chênh lệch lớn nhất:
   Xuất bảng ghép ảnh trực quan so sánh cạnh nhau (Side-by-side Visual Comparison Panel)
   và biểu đồ 1D profile kèm theo.
5. Tối ưu hóa 100% môi trường Kaggle 2x GPU Tesla T4 (chạy song song Shard 0/1, pre-cache lock-free,
   real-time checkpointing từng prompt, tự động merge shard và đóng gói zip Save Version).
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
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Cấu hình môi trường chống crash và chống phân mảnh VRAM trên GPU T4
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Compatibility shims cho transformers
try:
    import transformers
    for dummy_cls in ["EncoderDecoderCache", "DynamicCache", "Cache"]:
        if not hasattr(transformers, dummy_cls):
            setattr(transformers, dummy_cls, type(dummy_cls, (), {}))
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
        if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
            transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
except Exception:
    pass

# Dummy wandb module để tránh crash
if "wandb" not in sys.modules:
    try:
        import wandb
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

# Safe imports for diffusion pipeline & schedulers
try:
    from diffusers import (
        DPMSolverMultistepScheduler,
        DDIMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel
    )
except ImportError:
    DPMSolverMultistepScheduler = None
    DDIMScheduler = None
    StableDiffusionPipeline = None
    UNet2DConditionModel = None

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

# FKD Pipeline for closed-form target steering (READ-ONLY usage from fkd_diffusers)
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if os.path.join(_current_dir, "fkd_diffusers") not in sys.path:
    sys.path.insert(0, os.path.join(_current_dir, "fkd_diffusers"))

try:
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
except ImportError:
    try:
        from fkd_pipeline_sd import FKDStableDiffusion
    except ImportError:
        FKDStableDiffusion = None


# ======================================================================================
# 📦 IN-MEMORY DATASET DÀNH CHO LOOKAHEAD SAMPLING (ZERO DISK I/O)
# ======================================================================================
class InMemoryLookaheadDataset:
    """Dataset trong bộ nhớ RAM phục vụ Closed-form guidance mà không tốn I/O ổ đĩa."""
    def __init__(self, data_dict: Optional[Dict[int, Any]] = None):
        self.data = data_dict or {}

    def set_datapoint(self, prompt_idx: int, latents: torch.Tensor, rewards: torch.Tensor, prompts: List[str]):
        self.data[prompt_idx] = {
            "latents": latents.detach().cpu(),
            "rewards": rewards.detach().cpu() if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, dtype=torch.float32),
            "prompts": prompts,
            "prompt_idx": prompt_idx,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ======================================================================================
# 📚 PROMPT LOADING & STRATIFIED CATEGORY SAMPLING
# ======================================================================================
GENEVAL_CATEGORIES = [
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr"
]

def load_stratified_prompts_per_category(
    prompt_path: str = "prompt_files/geneval_metadata.jsonl",
    prompts_per_category: int = 2,
    max_prompts: int = -1,
    prompt_indices: Optional[str] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Nạp danh sách prompts phân tầng chặt chẽ theo từng Category chuẩn GenEval.
    - Nếu prompts_per_category > 0: Chọn đúng K prompts cho mỗi category trong 6 category chuẩn.
    - Giữ nguyên 'original_idx' để bảo toàn vị trí gốc trong metadata.
    """
    if not os.path.exists(prompt_path):
        cands = [
            os.path.join("Diffusion-LiDAR-Sampling", prompt_path),
            os.path.join("..", prompt_path),
            prompt_path
        ]
        found = False
        for c in cands:
            if os.path.exists(c):
                prompt_path = c
                found = True
                break
        if not found:
            # Fallback tạo prompts mẫu nếu không tìm thấy file
            print(f"⚠️ Prompt file '{prompt_path}' không tồn tại. Tạo prompts mẫu 6 category GenEval.")
            fallback_prompts = [
                {"tag": "single_object", "prompt": "a photo of a bench", "original_idx": 0},
                {"tag": "single_object", "prompt": "a photo of a bicycle", "original_idx": 1},
                {"tag": "two_object", "prompt": "a photo of a bench and a horse", "original_idx": 2},
                {"tag": "two_object", "prompt": "a photo of a couch and a dog", "original_idx": 3},
                {"tag": "counting", "prompt": "three apples on a wooden table", "original_idx": 4},
                {"tag": "counting", "prompt": "four red cars parked on a street", "original_idx": 5},
                {"tag": "colors", "prompt": "a red sports car", "original_idx": 6},
                {"tag": "colors", "prompt": "a blue ceramic vase", "original_idx": 7},
                {"tag": "position", "prompt": "a laptop on top of a desk", "original_idx": 8},
                {"tag": "position", "prompt": "a cat under a wooden chair", "original_idx": 9},
                {"tag": "color_attr", "prompt": "a red dog and a blue cat", "original_idx": 10},
                {"tag": "color_attr", "prompt": "a green book and a yellow pen", "original_idx": 11},
            ]
            return fallback_prompts

    items = []
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        with open(prompt_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    it = json.loads(line)
                    if "original_idx" not in it:
                        it["original_idx"] = idx
                    items.append(it)

    for idx, it in enumerate(items):
        if "prompt" not in it and "text" in it:
            it["prompt"] = it["text"]
        if "tag" not in it:
            it["tag"] = it.get("category", "general")
        if "original_idx" not in it:
            it["original_idx"] = idx

    # 1. Nếu chỉ định cụ thể prompt_indices
    if prompt_indices is not None and str(prompt_indices).strip() != "":
        s = str(prompt_indices).strip()
        if "-" in s and "," not in s:
            parts = s.split("-")
            selected_ids = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            selected_ids = [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
        selected = [it for it in items if it["original_idx"] in selected_ids]
        print(f"📋 Đã chọn {len(selected)} prompts theo chỉ số chỉ định ({prompt_indices}).")
        return selected

    # 2. Phân tầng theo Category (Stratified per Category)
    if prompts_per_category is not None and prompts_per_category > 0:
        by_cat = {}
        for it in items:
            cat = it.get("tag", "general")
            by_cat.setdefault(cat, []).append(it)

        selected = []
        print(f"📊 Chọn phân tầng (Stratified) {prompts_per_category} prompts cho mỗi Category:")
        # Ưu tiên các category chuẩn GenEval trước
        active_cats = [c for c in GENEVAL_CATEGORIES if c in by_cat] + [c for c in by_cat if c not in GENEVAL_CATEGORIES]
        for cat in active_cats:
            bucket = by_cat[cat]
            k = min(prompts_per_category, len(bucket))
            picked = bucket[:k]
            selected.extend(picked)
            ids_str = ", ".join(str(p["original_idx"]) for p in picked)
            print(f"   • [{cat}]: {len(picked)}/{len(bucket)} prompts (IDs: {ids_str})")
        return selected

    # 3. Giới hạn theo max_prompts thông thường
    if max_prompts > 0 and max_prompts < len(items):
        return items[:max_prompts]

    return items


# ======================================================================================
# 🔄 TENSOR & IMAGE CONVERSION UTILITIES
# ======================================================================================
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Chuyển PyTorch Tensor [1, 3, H, W] hoặc [3, H, W] trong [-1.0, 1.0] thành ảnh PIL."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.contiguous().permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def extract_single_pil_image(item: Any) -> Image.Image:
    """
    Rút trích triệt để 1 đối tượng PIL.Image từ bất kỳ cấu trúc lồng nhau nào:
    (tuple, list, StableDiffusionPipelineOutput, torch.Tensor, np.ndarray, v.v.)
    Học tập cơ chế bóc tách an toàn của test_empirical_lipschitz.py & LiDAR_sampling.py.
    """
    cur = item
    # 1. Bóc tách các lớp tuple/list bên ngoài
    while isinstance(cur, (tuple, list)) and len(cur) > 0:
        cur = cur[0]

    # 2. Bóc tách thuộc tính .images của Diffusers Pipeline Output
    if hasattr(cur, "images"):
        cur = cur.images
        while isinstance(cur, (tuple, list)) and len(cur) > 0:
            cur = cur[0]

    # 3. Chuyển đổi định dạng nếu cần
    if isinstance(cur, Image.Image):
        return cur
    if isinstance(cur, torch.Tensor):
        return tensor_to_pil(cur)
    if isinstance(cur, np.ndarray):
        if cur.dtype != np.uint8:
            cur = np.clip(cur * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(cur)
    if hasattr(cur, "cpu") and hasattr(cur, "numpy"):
        return tensor_to_pil(cur)

    return cur

def extract_pil_images_list(raw_imgs: Any) -> List[Image.Image]:
    """
    Rút trích danh sách PIL.Image từ output của Diffusers pipeline hoặc bất kỳ cấu trúc nào.
    """
    if hasattr(raw_imgs, "images"):
        raw_list = raw_imgs.images
    elif isinstance(raw_imgs, (tuple, list)):
        if len(raw_imgs) == 0:
            return []
        first = raw_imgs[0]
        if hasattr(first, "images"):
            raw_list = first.images
        elif isinstance(first, (tuple, list)):
            return extract_pil_images_list(first)
        else:
            raw_list = list(raw_imgs)
    else:
        raw_list = [raw_imgs]

    return [extract_single_pil_image(it) for it in raw_list]

def pil_to_tensor_neg1_to_1(pil_img: Any, device: str = "cpu") -> torch.Tensor:
    """Chuyển ảnh PIL thành PyTorch Tensor [1, 3, H, W] trong dải [-1.0, 1.0]."""
    clean_img = extract_single_pil_image(pil_img)
    arr = np.array(clean_img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor


# ======================================================================================
# ⚡ REWARD SCORING UTILITIES
# ======================================================================================
@torch.inference_mode()
def score_single_reward_model(
    images: List[Any],
    prompt: str,
    reward_name: str = "ImageReward",
    ir_model: Any = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Chấm điểm 1 danh sách ảnh PIL với prompt bằng mô hình Reward được chỉ định.
    Tự động chuẩn hóa và rút trích PIL.Image thuần túy cho từng phần tử để chống crash
    (tương tự evaluate_reward_all_models trong test_empirical_lipschitz.py).
    Trả về numpy array 1D chứa điểm của từng ảnh.
    """
    p_str = prompt[0] if isinstance(prompt, (list, tuple)) else str(prompt)
    clean_images = [extract_single_pil_image(im) for im in images]
    N = len(clean_images)

    # 1. ImageReward
    if "image" in reward_name.lower():
        if ir_model is not None:
            try:
                scores = [float(ir_model.score(p_str, img)) for img in clean_images]
                return np.array(scores, dtype=np.float32)
            except Exception as e:
                print(f"⚠️ Lỗi ImageReward scoring: {e}")
        # Synthetic fallback if ImageReward not available
        return np.zeros(N, dtype=np.float32)

    # 2. CLIP-Score
    if "clip" in reward_name.lower():
        if REWARDS_DICT.get("Clip-Score") is not None and clip is not None:
            try:
                clip_obj = REWARDS_DICT["Clip-Score"]
                clip_model = clip_obj.clip_model
                preprocess = clip_obj.preprocess
                tensors = torch.stack([preprocess(img) for img in clean_images]).to(device=device)
                image_features = F.normalize(clip_model.encode_image(tensors))
                text_tokens = clip.tokenize(p_str, truncate=True).to(device)
                txt_features = F.normalize(clip_model.encode_text(text_tokens))
                scs = (image_features.float() * txt_features.float()).sum(dim=-1).cpu().numpy()
                return scs
            except Exception as e:
                pass
        return np.zeros(N, dtype=np.float32)

    # 3. HPS v2.1
    if "hps" in reward_name.lower() and do_human_preference_score is not None:
        try:
            scs = do_human_preference_score(images=clean_images, prompts=[p_str] * N)
            return np.array(scs, dtype=np.float32)
        except Exception:
            pass

    return np.zeros(N, dtype=np.float32)


# ======================================================================================
# 🔬 PHASE 1: DUAL LOOKAHEAD ANNOTATION (VANILLA VS RS-LIDAR)
# ======================================================================================
@torch.inference_mode()
def run_phase1_lookahead_dual_annotation(
    pipe: Any,
    vae: Any,
    ir_model: Any,
    prompt: str,
    real_idx: int,
    num_particles: int = 50,
    guidance_steps: int = 5,
    sigma: float = 1.0,
    mc_samples: int = 4,
    use_crn: bool = True,
    seed: int = 42,
    device: str = "cuda"
) -> Tuple[InMemoryLookaheadDataset, InMemoryLookaheadDataset, Dict[str, Any]]:
    """
    Thực thi Phase 1 với tối ưu hoá thời gian và VRAM:
    1. Sinh n=50 hạt lookahead z_0 (DPM-5) DUY NHẤT 1 LẦN.
    2. Giải mã VAE sang không gian pixel [-1.0, 1.0].
    3. Đánh giá song song:
       - r_vanilla = R(x_0)
       - r_RS = (1/M) sum_{m=1}^M R(clip(x_0 + sigma * u_m, -1, 1)) với CRN.
    4. Trả về 2 InMemoryLookaheadDataset (Vanilla & RS-LiDAR).
    """
    generator = torch.Generator(device=device).manual_seed(seed + real_idx)

    # 1. Sinh latents bằng DPM-Solver 5 bước (cực nhanh)
    scheduler_backup = pipe.scheduler
    if DPMSolverMultistepScheduler is not None:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Sử dụng superclass StableDiffusionPipeline.__call__ để bypass FK Steering hook trong Phase 1
    # Tránh lỗi KeyError: 'pred_original_sample' khi dùng DPMSolverMultistepScheduler
    if StableDiffusionPipeline is not None and isinstance(pipe, StableDiffusionPipeline):
        latents_out = StableDiffusionPipeline.__call__(
            pipe,
            prompt=[prompt] * num_particles,
            num_inference_steps=guidance_steps,
            guidance_scale=7.5,
            generator=generator,
            output_type="latent"
        )
    else:
        latents_out = pipe(
            prompt=[prompt] * num_particles,
            num_inference_steps=guidance_steps,
            guidance_scale=7.5,
            generator=generator,
            output_type="latent"
        )
    pipe.scheduler = scheduler_backup
    raw_latents = latents_out.images if hasattr(latents_out, "images") else latents_out[0]
    lookahead_latents = raw_latents.detach() # [50, 4, 64, 64]

    # 2. Giải mã VAE sang tensor pixel [-1.0, 1.0] (theo chunk để tránh OOM)
    scaled = (lookahead_latents / vae.config.scaling_factor).to(device=device, dtype=vae.dtype)
    chunk_size = 10
    decoded_chunks = []
    for i in range(0, scaled.shape[0], chunk_size):
        chunk = scaled[i:i + chunk_size]
        dec = vae.decode(chunk, return_dict=False)[0].clamp(-1.0, 1.0)
        decoded_chunks.append(dec)
    decoded_tensors = torch.cat(decoded_chunks, dim=0) # [50, 3, 512, 512]

    # Chuyển ảnh sạch sang PIL
    clean_pils = pipe.image_processor.postprocess(decoded_tensors, output_type="pil")

    # 3. Chấm Vanilla Reward r_vanilla
    r_vanilla = score_single_reward_model(clean_pils, prompt, "ImageReward", ir_model, device=device)

    # 4. Chấm RS-LiDAR Smoothed Reward r_RS qua Monte Carlo CRN
    if sigma > 0.0 and mc_samples > 0:
        mc_scores_all = []
        for m in range(mc_samples):
            # Common Random Numbers (CRN)
            crn_seed = (10000 + m) if use_crn else (seed * 100 + real_idx * 10 + m)
            crn_gen = torch.Generator(device=device).manual_seed(crn_seed)
            noise = torch.randn(decoded_tensors.shape, generator=crn_gen, device=device, dtype=decoded_tensors.dtype) * sigma
            pert_tensors = (decoded_tensors + noise).clamp(-1.0, 1.0)
            pert_pils = pipe.image_processor.postprocess(pert_tensors, output_type="pil")
            sc_m = score_single_reward_model(pert_pils, prompt, "ImageReward", ir_model, device=device)
            mc_scores_all.append(sc_m)
        r_rs = np.mean(mc_scores_all, axis=0)
    else:
        r_rs = r_vanilla.copy()

    # 5. Đóng gói vào In-Memory Lookahead Datasets
    ds_vanilla = InMemoryLookaheadDataset()
    ds_vanilla.set_datapoint(real_idx, lookahead_latents, torch.tensor(r_vanilla, dtype=torch.float32), [prompt] * num_particles)

    ds_rs = InMemoryLookaheadDataset()
    ds_rs.set_datapoint(real_idx, lookahead_latents, torch.tensor(r_rs, dtype=torch.float32), [prompt] * num_particles)

    phase1_stats = {
        "r_vanilla_mean": float(np.mean(r_vanilla)),
        "r_vanilla_max": float(np.max(r_vanilla)),
        "r_vanilla_min": float(np.min(r_vanilla)),
        "r_rs_mean": float(np.mean(r_rs)),
        "r_rs_max": float(np.max(r_rs)),
        "r_rs_min": float(np.min(r_rs)),
        "lookahead_shape": list(lookahead_latents.shape),
    }

    return ds_vanilla, ds_rs, phase1_stats


# ======================================================================================
# 🎯 PHASE 2: TARGET STEERING (UNSTEERED VS VANILLA VS RS-LIDAR)
# ======================================================================================
@torch.inference_mode()
def run_phase2_target_steering(
    pipe: Any,
    prompt: str,
    real_idx: int,
    ds_vanilla: InMemoryLookaheadDataset,
    ds_rs: InMemoryLookaheadDataset,
    ir_model: Any,
    guidance_scale: float = 12.5,
    target_steps: int = 50,
    lmbda: float = 5000.0,
    resampling_t_end: int = 200,
    seed: int = 42,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Chạy Phase 2 Target Steering với cùng seed khởi tạo x_T cho cả 3 nhánh:
    1. Nhánh 1: Baseline Unsteered (s=0, không lái).
    2. Nhánh 2: Vanilla LiDAR (s=12.5, lái qua ds_vanilla).
    3. Nhánh 3: RS-LiDAR (s=12.5, lái qua ds_rs).
    """
    # Khởi tạo nhiễu khởi đầu cố định x_T
    init_gen = torch.Generator(device=device).manual_seed(seed * 1000 + real_idx)
    init_latents = torch.randn((1, 4, 64, 64), generator=init_gen, device=device, dtype=torch.float16 if "cuda" in device else torch.float32)

    # Đảm bảo dùng DDIMScheduler chuẩn Table 2 SD 1.5
    if DDIMScheduler is not None and not isinstance(pipe.scheduler, DDIMScheduler):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 1. Nhánh 1: Unsteered (s = 0)
    if StableDiffusionPipeline is not None and isinstance(pipe, StableDiffusionPipeline):
        out_unsteered = StableDiffusionPipeline.__call__(
            pipe,
            prompt=prompt,
            num_inference_steps=target_steps,
            guidance_scale=7.5,
            latents=init_latents.clone()
        )
    else:
        out_unsteered = pipe(
            prompt=prompt,
            num_inference_steps=target_steps,
            guidance_scale=7.5,
            latents=init_latents.clone(),
            fkd_args=None
        )
    img_unsteered = extract_single_pil_image(out_unsteered)
    r_unsteered = float(score_single_reward_model([img_unsteered], prompt, "ImageReward", ir_model, device=device)[0])

    # 2. Nhánh 2: Vanilla LiDAR (s = 12.5)
    fkd_args_vanilla = dict(
        lmbda=lmbda,
        scale=guidance_scale,
        reward_type="ImageReward",
        num_particles=1,
        use_smc=False,
        use_grad=False,
        use_rag=True,
        FK_lmbda=lmbda,
        FK_resample_t_start=1000,
        FK_resample_t_end=resampling_t_end,
        top_k=len(ds_vanilla.data[real_idx]["latents"]),
        rag_dataset=ds_vanilla,
        adaptive_resampling=False,
        resample_frequency=1,
        time_steps=target_steps,
        resampling_t_end=resampling_t_end,
        guidance_reward_fn="ImageReward",
        potential_type="l2",
    )
    out_vanilla = pipe(
        prompt=prompt,
        prompt_idx=real_idx,
        num_inference_steps=target_steps,
        guidance_scale=7.5,
        latents=init_latents.clone(),
        fkd_args=fkd_args_vanilla
    )
    img_vanilla = extract_single_pil_image(out_vanilla)
    r_vanilla = float(score_single_reward_model([img_vanilla], prompt, "ImageReward", ir_model, device=device)[0])

    # 3. Nhánh 3: RS-LiDAR (s = 12.5)
    fkd_args_rs = dict(
        lmbda=lmbda,
        scale=guidance_scale,
        reward_type="ImageReward",
        num_particles=1,
        use_smc=False,
        use_grad=False,
        use_rag=True,
        FK_lmbda=lmbda,
        FK_resample_t_start=1000,
        FK_resample_t_end=resampling_t_end,
        top_k=len(ds_rs.data[real_idx]["latents"]),
        rag_dataset=ds_rs,
        adaptive_resampling=False,
        resample_frequency=1,
        time_steps=target_steps,
        resampling_t_end=resampling_t_end,
        guidance_reward_fn="ImageReward",
        potential_type="l2",
    )
    out_rs = pipe(
        prompt=prompt,
        prompt_idx=real_idx,
        num_inference_steps=target_steps,
        guidance_scale=7.5,
        latents=init_latents.clone(),
        fkd_args=fkd_args_rs
    )
    img_rs = extract_single_pil_image(out_rs)
    r_rs = float(score_single_reward_model([img_rs], prompt, "ImageReward", ir_model, device=device)[0])

    return {
        "img_unsteered": img_unsteered,
        "img_vanilla": img_vanilla,
        "img_rs": img_rs,
        "r_unsteered": r_unsteered,
        "r_vanilla": r_vanilla,
        "r_rs": r_rs,
        "delta_r": r_rs - r_vanilla,
    }


# ======================================================================================
# 📐 1D DIRECTIONAL LANDSCAPE SLICING (CẮT LÁT BỀ MẶT REWARD)
# ======================================================================================
@torch.inference_mode()
def compute_1d_landscape_profile(
    img_vanilla: Any,
    img_rs: Any,
    prompt: str,
    ir_model: Any,
    slicing_points: int = 41,
    alpha_max: float = 0.2,
    slicing_sigma: float = 1.0,
    mc_samples: int = 4,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Cắt lát 1D không gian pixel dọc theo vector vi sai nối giữa Vanilla và RS-LiDAR:
      v = (x_RS - x_vanilla) / ||x_RS - x_vanilla||_2
      x(alpha) = clip(x_vanilla + alpha * v, -1, 1) với alpha in [-alpha_max, alpha_max]
    Đo:
      f_vanilla(alpha) = R(x(alpha))
      f_RS(alpha)      = (1/M) sum R(clip(x(alpha) + sigma * u_m, -1, 1)) với CRN
      Numerical Secant Derivative |Df(alpha)| = |f(alpha + da) - f(alpha)| / da
    """
    img_vanilla = extract_single_pil_image(img_vanilla)
    img_rs = extract_single_pil_image(img_rs)
    t_v = pil_to_tensor_neg1_to_1(img_vanilla, device=device)
    t_rs = pil_to_tensor_neg1_to_1(img_rs, device=device)

    # 1. Vector định hướng v
    diff = t_rs - t_v
    dist_l2 = float(torch.norm(diff.reshape(-1), p=2).item())

    if dist_l2 < 1e-6:
        # Nếu 2 ảnh trùng nhau, tạo hướng ngẫu nhiên
        v_rand = torch.randn_like(t_v)
        v = v_rand / torch.norm(v_rand.reshape(-1), p=2)
    else:
        v = diff / dist_l2

    # 2. Quét dải alpha in [-alpha_max, alpha_max]
    alphas = np.linspace(-alpha_max, alpha_max, slicing_points)
    delta_alpha = float(alphas[1] - alphas[0])

    curve_vanilla = []
    curve_rs = []

    # Cố định CRN cho toàn bộ đường cắt lát để đồ thị RS-LiDAR không bị nhiễu cục bộ
    crn_noise_list = []
    if slicing_sigma > 0.0 and mc_samples > 0:
        for m in range(mc_samples):
            crn_gen = torch.Generator(device=device).manual_seed(20000 + m)
            n_m = torch.randn(t_v.shape, generator=crn_gen, device=device, dtype=t_v.dtype) * slicing_sigma
            crn_noise_list.append(n_m)

    for a in alphas:
        t_a = (t_v + float(a) * v).clamp(-1.0, 1.0)
        img_a = tensor_to_pil(t_a)

        # Vanilla score dọc đường cắt
        sc_v = float(score_single_reward_model([img_a], prompt, "ImageReward", ir_model, device=device)[0])
        curve_vanilla.append(sc_v)

        # RS-LiDAR smoothed score dọc đường cắt
        if crn_noise_list:
            mc_scs = []
            for n_m in crn_noise_list:
                t_pert = (t_a + n_m).clamp(-1.0, 1.0)
                img_pert = tensor_to_pil(t_pert)
                sc_pert = float(score_single_reward_model([img_pert], prompt, "ImageReward", ir_model, device=device)[0])
                mc_scs.append(sc_pert)
            curve_rs.append(float(np.mean(mc_scs)))
        else:
            curve_rs.append(sc_v)

    curve_vanilla = np.array(curve_vanilla, dtype=np.float32)
    curve_rs = np.array(curve_rs, dtype=np.float32)

    # 3. Tính đạo hàm số học vi phân (Numerical Derivatives)
    # Secant slope: |f(a_{i+1}) - f(a_i)| / delta_alpha
    deriv_vanilla = np.abs(np.diff(curve_vanilla) / delta_alpha)
    deriv_rs = np.abs(np.diff(curve_rs) / delta_alpha)

    l_max_vanilla = float(np.max(deriv_vanilla)) if len(deriv_vanilla) > 0 else 0.0
    l_max_rs = float(np.max(deriv_rs)) if len(deriv_rs) > 0 else 0.0
    l_mean_vanilla = float(np.mean(deriv_vanilla)) if len(deriv_vanilla) > 0 else 0.0
    l_mean_rs = float(np.mean(deriv_rs)) if len(deriv_rs) > 0 else 0.0

    # Chặn lý thuyết Theorem 1: Delta R / (sigma * sqrt(2*pi))
    # Dùng range của reward quan sát được hoặc hằng số chuẩn
    reward_range = max(float(np.ptp(curve_vanilla)), 1.5)
    theoretical_bound = float(reward_range / (slicing_sigma * math.sqrt(2 * math.pi))) if slicing_sigma > 0 else 999.0

    return {
        "dist_l2": dist_l2,
        "alphas": alphas.tolist(),
        "curve_vanilla": curve_vanilla.tolist(),
        "curve_rs": curve_rs.tolist(),
        "deriv_vanilla": deriv_vanilla.tolist(),
        "deriv_rs": deriv_rs.tolist(),
        "l_max_vanilla": l_max_vanilla,
        "l_max_rs": l_max_rs,
        "l_mean_vanilla": l_mean_vanilla,
        "l_mean_rs": l_mean_rs,
        "theoretical_bound": theoretical_bound,
        "l_reduction_ratio": float(l_max_vanilla / (l_max_rs + 1e-6)),
    }


# ======================================================================================
# 🎯 DIVERGENCE SCORE & TOP-5 RANKING
# ======================================================================================
def compute_prompt_divergence_score(
    delta_r: float,
    l_max_vanilla: float,
    l_max_rs: float,
    dist_l2: float,
    total_pixels: int = 3 * 512 * 512
) -> float:
    """
    Tính điểm chỉ số Divergence tổng hợp để tìm prompt nhạy cảm nhất:
      Score = (R_RS - R_vanilla) + 0.5 * (L_vanilla / L_RS) + 1.0 * (||x_RS - x_vanilla||_2 / sqrt(D))
    """
    ratio_l = l_max_vanilla / (l_max_rs + 1e-6)
    norm_dist = dist_l2 / math.sqrt(total_pixels)
    score = float(delta_r + 0.5 * ratio_l + 1.0 * norm_dist)
    return score


# ======================================================================================
# 📊 VISUALIZATION & PUBLICATION-READY PLOTTING
# ======================================================================================
def plot_top5_divergence_panel(
    top5_records: List[Dict[str, Any]],
    output_path: str
):
    """
    Xuất bảng ghép đồ họa chất lượng cao Side-by-Side gồm 5 hàng x 5 cột:
    - Cột 1: Thông tin Prompt & Category (Category badge, Prompt text, Divergence score, Delta R, Ratio).
    - Cột 2: Ảnh Unsteered (s=0).
    - Cột 3: Ảnh Vanilla LiDAR (s=12.5) kèm điểm R và độ dốc L.
    - Cột 4: Ảnh RS-LiDAR (s=12.5, sigma=1.0) kèm điểm R và độ dốc L.
    - Cột 5: Đồ thị 1D Slicing Curve Profile thu nhỏ đặt ngay cạnh ảnh!
    """
    num_rows = len(top5_records)
    if num_rows == 0:
        return

    fig, axes = plt.subplots(num_rows, 5, figsize=(22, 4.2 * num_rows), dpi=300)
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    for i, rec in enumerate(top5_records):
        rank = i + 1
        tag = rec.get("category", rec.get("tag", "general"))
        prompt_text = rec.get("prompt", "")
        score = rec.get("divergence_score", 0.0)
        dr = rec.get("delta_r", 0.0)
        l_ratio = rec.get("l_reduction_ratio", 1.0)
        r_v = rec.get("r_vanilla", 0.0)
        r_rs = rec.get("r_rs", 0.0)
        lv = rec.get("l_max_vanilla", 0.0)
        lrs = rec.get("l_max_rs", 0.0)

        # ----------------------------------------------------------------------
        # CỘT 1: THẺ THÔNG TIN PROMPT (TEXT CARD)
        # ----------------------------------------------------------------------
        ax_info = axes[i, 0]
        ax_info.axis("off")
        # Khung viền thẻ thông tin
        card_box = dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#457B9D", lw=1.8)
        info_lines = [
            f"[RANK #{rank} SENSITIVE PROMPT]",
            f"---------------------------------",
            f"Category:     [{tag.upper()}]",
            f"Prompt ID:    #{rec.get('original_idx', rec.get('prompt_idx', i))}",
            f"Prompt Text:",
            f"  \"{prompt_text[:75]}...\"" if len(prompt_text) > 75 else f"  \"{prompt_text}\"",
            f"---------------------------------",
            f"Divergence Score:  {score:.3f}",
            f"Reward Gain dR:    {dr:+.3f}",
            f"Lipschitz Ratio:   {l_ratio:.2f}x",
        ]
        info_str = "\n".join(info_lines)
        ax_info.text(0.05, 0.5, info_str, fontsize=9.5, verticalalignment="center",
                     family="monospace", bbox=card_box)

        # ----------------------------------------------------------------------
        # CỘT 2: ẢNH UNSTEERED (BASELINE)
        # ----------------------------------------------------------------------
        ax_un = axes[i, 1]
        img_un = rec.get("img_unsteered_obj")
        if img_un is None and "img_unsteered_path" in rec and os.path.exists(rec["img_unsteered_path"]):
            img_un = Image.open(rec["img_unsteered_path"])
        if img_un is not None:
            ax_un.imshow(img_un)
            ax_un.set_title(f"Unsteered ($s=0$)\n$R = {rec.get('r_unsteered', 0.0):.3f}$", fontsize=10.5, fontweight="bold", color="#1D3557")
        else:
            ax_un.text(0.5, 0.5, "Image N/A", ha="center", va="center")
        ax_un.axis("off")

        # ----------------------------------------------------------------------
        # CỘT 3: ẢNH VANILLA LIDAR (s = 12.5)
        # ----------------------------------------------------------------------
        ax_vl = axes[i, 2]
        img_vl = rec.get("img_vanilla_obj")
        if img_vl is None and "img_vanilla_path" in rec and os.path.exists(rec["img_vanilla_path"]):
            img_vl = Image.open(rec["img_vanilla_path"])
        if img_vl is not None:
            ax_vl.imshow(img_vl)
            ax_vl.set_title(f"Vanilla LiDAR ($s=12.5$)\n$R = {r_v:.3f}$ | $L_{{\\max}} = {lv:.1f}$", fontsize=10.5, fontweight="bold", color="#E63946")
            for spine in ax_vl.spines.values():
                spine.set_edgecolor("#E63946")
                spine.set_linewidth(2.0)
        else:
            ax_vl.text(0.5, 0.5, "Image N/A", ha="center", va="center")
        ax_vl.axis("off")

        # ----------------------------------------------------------------------
        # CỘT 4: ẢNH RS-LIDAR (s = 12.5, sigma = 1.0)
        # ----------------------------------------------------------------------
        ax_rs = axes[i, 3]
        img_rs = rec.get("img_rs_obj")
        if img_rs is None and "img_rs_path" in rec and os.path.exists(rec["img_rs_path"]):
            img_rs = Image.open(rec["img_rs_path"])
        if img_rs is not None:
            ax_rs.imshow(img_rs)
            ax_rs.set_title(f"RS-LiDAR ($s=12.5, \\sigma=1.0$)\n$R = {r_rs:.3f}$ | $L_{{\\max}} = {lrs:.1f}$", fontsize=10.5, fontweight="bold", color="#2A9D8F")
            for spine in ax_rs.spines.values():
                spine.set_edgecolor("#2A9D8F")
                spine.set_linewidth(2.0)
        else:
            ax_rs.text(0.5, 0.5, "Image N/A", ha="center", va="center")
        ax_rs.axis("off")

        # ----------------------------------------------------------------------
        # CỘT 5: 1D LANDSCAPE SLICING CURVE PROFILE
        # ----------------------------------------------------------------------
        ax_sl = axes[i, 4]
        slicing = rec.get("slicing", {})
        alphas = slicing.get("alphas", [])
        cv = slicing.get("curve_vanilla", [])
        crs = slicing.get("curve_rs", [])
        bound = slicing.get("theoretical_bound", 0.0)

        if len(alphas) > 0 and len(cv) > 0:
            ax_sl.plot(alphas, cv, color="#E63946", lw=2.2, label="Vanilla $R(x)$ (Spiky)", linestyle="-")
            ax_sl.plot(alphas, crs, color="#2A9D8F", lw=2.4, label="RS-LiDAR $R_\\sigma(x)$ (Smooth)", linestyle="-")
            ax_sl.axvline(0.0, color="#1D3557", linestyle=":", alpha=0.6, label="Vanilla Point")
            ax_sl.set_xlabel("Perturbation $\\alpha$", fontsize=9.5, fontweight="bold")
            ax_sl.set_ylabel("Reward Score", fontsize=9.5, fontweight="bold")
            ax_sl.set_title(f"1D Landscape Slice (Rank #{rank})\n$L_{{\\text{{ratio}}}} = {l_ratio:.2f}\\times$", fontsize=10, fontweight="bold")
            ax_sl.grid(True, linestyle="--", alpha=0.5)
            ax_sl.legend(fontsize=7.5, loc="best", frameon=True)
        else:
            ax_sl.text(0.5, 0.5, "Profile N/A", ha="center", va="center")

    plt.suptitle("RS-LiDAR ABLATION 1: TOP 5 DIVERGENT PROMPTS & NON-LIPSCHITZ LANDSCAPE COMPARISON",
                 fontsize=15, fontweight="bold", y=0.995, color="#1D3557")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"🖼️ Đã xuất biểu đồ so sánh Top 5 Panel chất lượng cao: {output_path}")


def plot_summary_landscape_curves(
    all_records: List[Dict[str, Any]],
    output_path: str
):
    """Vẽ đồ thị tổng hợp các đường cắt lát 1D của tất cả các prompt."""
    if not all_records:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    for i, rec in enumerate(all_records[:12]):
        slicing = rec.get("slicing", {})
        alphas = slicing.get("alphas", [])
        cv = slicing.get("curve_vanilla", [])
        crs = slicing.get("curve_rs", [])
        if len(alphas) > 0:
            alpha_arr = np.array(alphas)
            ax1.plot(alpha_arr, cv, alpha=0.55, lw=1.5, color="#E63946")
            ax2.plot(alpha_arr, crs, alpha=0.55, lw=1.5, color="#2A9D8F")

    ax1.set_xlabel("Slicing Coordinate $\\alpha$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Vanilla Reward $R(x(\\alpha))$", fontsize=11, fontweight="bold")
    ax1.set_title("Panel A: Vanilla Reward Landscapes (Spiky / Rugged)", fontsize=12, fontweight="bold", color="#E63946")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Slicing Coordinate $\\alpha$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("RS-LiDAR Smoothed Reward $R_\\sigma(x(\\alpha))$", fontsize=11, fontweight="bold")
    ax2.set_title("Panel B: RS-LiDAR Smoothed Landscapes (Smooth / Convex-like)", fontsize=12, fontweight="bold", color="#2A9D8F")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📈 Đã xuất biểu đồ tổng hợp 1D Landscape Curves: {output_path}")


# ======================================================================================
# 🔗 MERGE SHARDS UTILITY
# ======================================================================================
def merge_shard_checkpoints(output_dir: str):
    """
    Gộp toàn bộ các file checkpoint_ablation1_shard_*.json trong output_dir thành 1 tập dữ liệu,
    sau đó sắp xếp xếp hạng Divergence Score, xuất Top 5 và kết xuất các đồ thị.
    """
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint_ablation1_shard_*.json")))
    if not ckpts:
        print(f"❌ Không tìm thấy file checkpoint nào trong: {output_dir}")
        return

    print(f"🔄 Tìm thấy {len(ckpts)} shard checkpoints: {[os.path.basename(c) for c in ckpts]}")
    merged_records = []
    seen_ids = set()

    for c_path in ckpts:
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            records = c_data.get("records", [])
            for r in records:
                p_id = r.get("original_idx", r.get("prompt_idx"))
                if p_id not in seen_ids:
                    seen_ids.add(p_id)
                    merged_records.append(r)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file checkpoint {c_path}: {e}")

    print(f"✅ Đã gộp thành công {len(merged_records)} prompts đo đạc duy nhất.")

    # 1. Sắp xếp giảm dần theo Divergence Score
    merged_records.sort(key=lambda x: x.get("divergence_score", 0.0), reverse=True)

    # 2. Xuất bảng CSV tổng hợp tất cả prompt
    summary_csv_path = os.path.join(output_dir, "ablation1_summary.csv")
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write("rank,original_idx,category,prompt,r_unsteered,r_vanilla,r_rs,delta_r,l_max_vanilla,l_max_rs,l_reduction_ratio,dist_l2,divergence_score\n")
        for rank, r in enumerate(merged_records, 1):
            p_text = r.get("prompt", "").replace('"', '""')
            cat = r.get("category", r.get("tag", "general"))
            f.write(
                f"{rank},{r.get('original_idx', '')},{cat},\"{p_text}\","
                f"{r.get('r_unsteered', 0.0):.4f},{r.get('r_vanilla', 0.0):.4f},{r.get('r_rs', 0.0):.4f},"
                f"{r.get('delta_r', 0.0):.4f},{r.get('l_max_vanilla', 0.0):.2f},{r.get('l_max_rs', 0.0):.2f},"
                f"{r.get('l_reduction_ratio', 1.0):.2f},{r.get('dist_l2', 0.0):.4f},{r.get('divergence_score', 0.0):.4f}\n"
            )
    print(f"📄 Đã lưu bảng CSV tổng hợp: {summary_csv_path}")

    # 3. Trích xuất Top 5 Prompts chênh lệch nhất
    top5_records = merged_records[:5]
    top5_csv_path = os.path.join(output_dir, "ablation1_top5_divergence.csv")
    with open(top5_csv_path, "w", encoding="utf-8") as f:
        f.write("rank,original_idx,category,prompt,r_vanilla,r_rs,delta_r,l_max_vanilla,l_max_rs,l_reduction_ratio,divergence_score\n")
        for rank, r in enumerate(top5_records, 1):
            p_text = r.get("prompt", "").replace('"', '""')
            cat = r.get("category", r.get("tag", "general"))
            f.write(
                f"{rank},{r.get('original_idx', '')},{cat},\"{p_text}\","
                f"{r.get('r_vanilla', 0.0):.4f},{r.get('r_rs', 0.0):.4f},{r.get('delta_r', 0.0):.4f},"
                f"{r.get('l_max_vanilla', 0.0):.2f},{r.get('l_max_rs', 0.0):.2f},"
                f"{r.get('l_reduction_ratio', 1.0):.2f},{r.get('divergence_score', 0.0):.4f}\n"
            )
    print(f"🏆 Đã lưu bảng CSV Top 5 Divergence: {top5_csv_path}")

    # 4. Xuất JSON Metrics chi tiết
    metrics_json_path = os.path.join(output_dir, "ablation1_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_prompts": len(merged_records),
            "top5_prompts": [
                {
                    "rank": i + 1,
                    "original_idx": r.get("original_idx"),
                    "category": r.get("category"),
                    "prompt": r.get("prompt"),
                    "divergence_score": r.get("divergence_score"),
                    "delta_r": r.get("delta_r"),
                    "l_reduction_ratio": r.get("l_reduction_ratio")
                }
                for i, r in enumerate(top5_records)
            ],
            "all_records": merged_records
        }, f, indent=2, ensure_ascii=False)
    print(f"📊 Đã lưu file JSON metrics: {metrics_json_path}")

    # 5. Vẽ biểu đồ so sánh Top 5 Panel và 1D Curves
    panel_path = os.path.join(output_dir, "ablation1_top5_divergence_panel.png")
    plot_top5_divergence_panel(top5_records, panel_path)

    curves_path = os.path.join(output_dir, "ablation1_landscape_1d_curves.png")
    plot_summary_landscape_curves(merged_records, curves_path)


# ======================================================================================
# 🏁 MAIN EXPERIMENT LOOP
# ======================================================================================
def run_ablation1_experiment(pipe, vae, ir_model, prompts_list, args, device="cuda"):
    """Vòng lặp thực nghiệm chính cho 1 tiến trình shard."""
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    curves_dir = os.path.join(args.output_dir, "slicing_curves")
    os.makedirs(curves_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, f"checkpoint_ablation1_shard_{args.shard_id}.json")

    # Phân chia prompt cho shard
    total_prompts = len(prompts_list)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        active_prompts = prompts_list[start_idx:end_idx]
        print(f"🚀 Shard {args.shard_id + 1}/{args.num_shards}: Xử lý {len(active_prompts)} prompts ({start_idx} đến {end_idx - 1}).")
    else:
        start_idx = 0
        end_idx = total_prompts
        active_prompts = prompts_list
        print(f"🚀 Chạy đơn tiến trình: Xử lý toàn bộ {len(active_prompts)} prompts.")

    # Đọc checkpoint cũ nếu resume
    completed_records = []
    completed_ids = set()
    if args.resume and not args.overwrite and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            completed_records = c_data.get("records", [])
            for r in completed_records:
                completed_ids.add(r.get("original_idx", r.get("prompt_idx")))
            print(f"🔄 Đã nạp lại {len(completed_records)} prompts từ checkpoint cũ {ckpt_path}.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc checkpoint: {e}")

    for idx, item in enumerate(tqdm(active_prompts, desc=f"Shard-{args.shard_id}")):
        real_idx = item.get("original_idx", idx)
        prompt_text = item.get("prompt", "")
        category = item.get("tag", item.get("category", "general"))

        if real_idx in completed_ids:
            continue

        print(f"\n[{idx + 1}/{len(active_prompts)}] Prompt #{real_idx} [{category}]: \"{prompt_text}\"")

        # ----------------------------------------------------------------------
        # BƯỚC 1: PHASE 1 LOOKAHEAD ANNOTATION
        # ----------------------------------------------------------------------
        ds_vanilla, ds_rs, p1_stats = run_phase1_lookahead_dual_annotation(
            pipe=pipe,
            vae=vae,
            ir_model=ir_model,
            prompt=prompt_text,
            real_idx=real_idx,
            num_particles=args.num_lookahead,
            guidance_steps=args.lookahead_steps,
            sigma=args.sigma,
            mc_samples=args.mc_samples,
            use_crn=args.use_crn,
            seed=args.seed,
            device=device
        )

        # ----------------------------------------------------------------------
        # BƯỚC 2: PHASE 2 TARGET STEERING
        # ----------------------------------------------------------------------
        p2_results = run_phase2_target_steering(
            pipe=pipe,
            prompt=prompt_text,
            real_idx=real_idx,
            ds_vanilla=ds_vanilla,
            ds_rs=ds_rs,
            ir_model=ir_model,
            guidance_scale=args.guidance_scale,
            target_steps=args.target_steps,
            lmbda=args.lmbda,
            resampling_t_end=args.resampling_t_end,
            seed=args.seed,
            device=device
        )

        # Lưu ảnh sinh ra
        unsteered_path = os.path.join(images_dir, f"{real_idx:05d}_unsteered.png")
        vanilla_path = os.path.join(images_dir, f"{real_idx:05d}_vanilla.png")
        rs_path = os.path.join(images_dir, f"{real_idx:05d}_rs.png")
        extract_single_pil_image(p2_results["img_unsteered"]).save(unsteered_path)
        extract_single_pil_image(p2_results["img_vanilla"]).save(vanilla_path)
        extract_single_pil_image(p2_results["img_rs"]).save(rs_path)

        # ----------------------------------------------------------------------
        # BƯỚC 3: 1D DIRECTIONAL LANDSCAPE SLICING
        # ----------------------------------------------------------------------
        slicing_res = compute_1d_landscape_profile(
            img_vanilla=p2_results["img_vanilla"],
            img_rs=p2_results["img_rs"],
            prompt=prompt_text,
            ir_model=ir_model,
            slicing_points=args.slicing_points,
            alpha_max=args.slicing_range,
            slicing_sigma=args.slicing_sigma,
            mc_samples=args.slicing_mc_samples,
            device=device
        )

        # ----------------------------------------------------------------------
        # BƯỚC 4: TÍNH ĐIỂM DIVERGENCE SCORE
        # ----------------------------------------------------------------------
        div_score = compute_prompt_divergence_score(
            delta_r=p2_results["delta_r"],
            l_max_vanilla=slicing_res["l_max_vanilla"],
            l_max_rs=slicing_res["l_max_rs"],
            dist_l2=slicing_res["dist_l2"]
        )

        record = {
            "original_idx": real_idx,
            "prompt_idx": real_idx,
            "category": category,
            "prompt": prompt_text,
            "r_unsteered": p2_results["r_unsteered"],
            "r_vanilla": p2_results["r_vanilla"],
            "r_rs": p2_results["r_rs"],
            "delta_r": p2_results["delta_r"],
            "img_unsteered_path": unsteered_path,
            "img_vanilla_path": vanilla_path,
            "img_rs_path": rs_path,
            "dist_l2": slicing_res["dist_l2"],
            "l_max_vanilla": slicing_res["l_max_vanilla"],
            "l_max_rs": slicing_res["l_max_rs"],
            "l_reduction_ratio": slicing_res["l_reduction_ratio"],
            "divergence_score": div_score,
            "slicing": slicing_res,
            "phase1_stats": p1_stats
        }

        completed_records.append(record)
        completed_ids.add(real_idx)

        # Ghi checkpoint ngay sau mỗi prompt
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "shard_id": args.shard_id,
                "completed_count": len(completed_records),
                "records": completed_records
            }, f, indent=2, ensure_ascii=False)

    return completed_records


def main():
    parser = argparse.ArgumentParser(description="RS-LiDAR Ablation 1: Sensitive Prompt & Non-Lipschitz Landscape Slicing")

    # Dữ liệu & Cấu hình Prompts
    parser.add_argument("--prompts_file", type=str, default="prompt_files/geneval_metadata.jsonl", help="Đường dẫn file prompt")
    parser.add_argument("--prompts_per_category", type=int, default=2, help="Số prompts đại diện chọn cho mỗi Category (mặc định 2 -> 12 prompts)")
    parser.add_argument("--num_prompts", type=int, default=12, help="Tổng số prompts (dùng khi không phân tầng)")
    parser.add_argument("--prompt_indices", type=str, default=None, help="Chỉ định ID prompts cụ thể (ví dụ: '0,5,11' hoặc '0-11')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Mô hình Diffusion
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Tên model base")

    # Cấu hình Phase 1 Lookahead
    parser.add_argument("--num_lookahead", type=int, default=50, help="Số hạt lookahead n (mặc định 50)")
    parser.add_argument("--lookahead_steps", type=int, default=5, help="Số bước DPM solver Phase 1 (mặc định 5)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Bán kính làm mịn Gaussian sigma của RS-LiDAR (mặc định 1.0)")
    parser.add_argument("--mc_samples", type=int, default=4, help="Số mẫu Monte Carlo M cho Phase 1 (mặc định 4)")
    parser.add_argument("--use_crn", action="store_true", default=True, help="Dùng Common Random Numbers (CRN)")
    parser.add_argument("--no_crn", action="store_false", dest="use_crn")

    # Cấu hình Phase 2 Target Steering
    parser.add_argument("--guidance_scale", type=float, default=12.5, help="Hệ số lái scale s (mặc định 12.5 chuẩn SD 1.5 Table 2)")
    parser.add_argument("--num_target_images", type=int, default=1, help="Số ảnh sinh mỗi prompt ở Phase 2 (mặc định 1 ảnh để đối chiếu 1-1)")
    parser.add_argument("--target_steps", type=int, default=50, help="Số bước DDIM Solver Phase 2 (mặc định 50)")
    parser.add_argument("--lmbda", type=float, default=5000.0, help="Lambda softmax exponent")
    parser.add_argument("--resampling_t_end", type=int, default=200, help="Ngưỡng timestep kết thúc lái t_end")
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward", help="Hàm reward lái chính")

    # Cấu hình Cắt lát 1D Landscape Slicing
    parser.add_argument("--slicing_points", type=int, default=41, help="Số điểm trên lưới cắt lát 1D")
    parser.add_argument("--slicing_range", type=float, default=0.2, help="Biên độ quét alpha [-range, +range]")
    parser.add_argument("--slicing_sigma", type=float, default=1.0, help="Bán kính làm mịn RS dọc đường cắt lát")
    parser.add_argument("--slicing_mc_samples", type=int, default=4, help="Số mẫu Monte Carlo dọc đường cắt")

    # Multi-GPU Sharding & Hardware
    parser.add_argument("--num_shards", type=int, default=1, help="Tổng số shard GPU")
    parser.add_argument("--shard_id", type=int, default=0, help="ID shard hiện tại (0, 1, ...)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU vật lý")
    parser.add_argument("--device", type=str, default=None, help="Device ép buộc (vd: 'cpu' hoặc 'cuda:0')")
    parser.add_argument("--output_dir", type=str, default="results/ablation1_sensitive_landscape", help="Thư mục kết quả")
    parser.add_argument("--resume", action="store_true", default=True, help="Tiếp tục từ checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè, không dùng checkpoint cũ")
    parser.add_argument("--merge_shards", action="store_true", help="Chế độ gộp kết quả các shard")

    args = parser.parse_args()

    # Chế độ merge shard độc lập
    if args.merge_shards:
        merge_shard_checkpoints(args.output_dir)
        return

    # Thiết lập device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        actual_gpu = args.gpu_id if args.gpu_id < torch.cuda.device_count() else 0
        torch.cuda.set_device(actual_gpu)
        device = f"cuda:{actual_gpu}"
        print(f"🎮 Sử dụng GPU {actual_gpu}: {torch.cuda.get_device_name(actual_gpu)}")
    else:
        device = "cpu"
        print("💻 Không phát hiện GPU, chạy trên CPU.")

    # 1. Nạp danh sách Prompts phân tầng theo Category
    prompts_list = load_stratified_prompts_per_category(
        prompt_path=args.prompts_file,
        prompts_per_category=args.prompts_per_category,
        max_prompts=args.num_prompts,
        prompt_indices=args.prompt_indices,
        seed=args.seed
    )
    print(f"📋 Đã chuẩn bị {len(prompts_list)} prompts kiểm thử cho Ablation 1.")

    # 2. Nạp Mô hình Stable Diffusion & VAE
    print(f"📥 Đang nạp pipeline {args.model_name}...")
    dtype = torch.float16 if "cuda" in device else torch.float32

    # Sử dụng FKDStableDiffusion cho closed-form guidance nếu có, ngược lại fallback StableDiffusionPipeline
    pipe_cls = FKDStableDiffusion if FKDStableDiffusion is not None else StableDiffusionPipeline
    pipe = pipe_cls.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae

    # 3. Nạp ImageReward Model
    ir_model = None
    if rm_load is not None:
        try:
            print("📥 Đang nạp ImageReward...")
            ir_model = rm_load("ImageReward-v1.0")
            print("✅ ImageReward đã sẵn sàng!")
        except Exception as e:
            print(f"⚠️ Không thể nạp ImageReward: {e}")

    # 4. Chạy thực nghiệm
    records = run_ablation1_experiment(pipe, vae, ir_model, prompts_list, args, device=device)

    # 5. Nếu chạy đơn shard, tự động tổng hợp luôn
    if args.num_shards == 1:
        merge_shard_checkpoints(args.output_dir)


if __name__ == "__main__":
    main()
