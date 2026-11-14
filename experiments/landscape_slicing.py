"""RS-LiDAR ablation: sensitive prompts and non-Lipschitz reward-landscape slicing.

1. Selects GenEval prompts (e.g. counting, position, color attribution) on which reward models such as
   ImageReward show large local Lipschitz constants (sharp, spurious spikes).
2. Two-phase generation:
   - Phase 1: generate n=50 lookahead particles (DPM-5) once and score them with both the vanilla
     reward r_vanilla and the smoothed reward r_RS.
   - Phase 2: closed-form target steering (DDIM-50) from the same initial noise for three branches:
     (i) unsteered, (ii) LiDAR (s=12.5), (iii) RS-LiDAR (s=12.5, sigma=1.0).
3. 1D directional landscape slicing between the LiDAR and RS-LiDAR outputs, comparing the spiky vanilla
   reward with the smoothed reward, whose slope is bounded by theory.
4. Ranks prompts (failure-rescue and smoothness scores), exports the top-K prompts as CSV/JSON and a
   side-by-side panel with the 1D slicing profiles.
5. Multi-GPU sharding, per-prompt checkpointing and shard merging.

Usage (from the repository root):
    python -m experiments.landscape_slicing --prompts_per_category 2 --output_dir results/ablation1_sensitive_landscape
    python -m experiments.landscape_slicing --merge_shards --output_dir results/ablation1_sensitive_landscape
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
import shutil
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
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

# FKD pipeline for closed-form target steering. Make the repository root and fkd_diffusers/ importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, "fkd_diffusers") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "fkd_diffusers"))

try:
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
except ImportError:
    try:
        from fkd_pipeline_sd import FKDStableDiffusion
    except ImportError:
        FKDStableDiffusion = None


# ======================================================================================
# IN-MEMORY LOOKAHEAD DATASET (NO DISK I/O)
# ======================================================================================
class InMemoryLookaheadDataset:
    """In-memory lookahead dataset for closed-form guidance (avoids disk I/O)."""
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
# PROMPT LOADING & STRATIFIED CATEGORY SAMPLING
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
    categories: Optional[str] = None,
    prompts_per_category: int = 2,
    max_prompts: int = -1,
    prompt_indices: Optional[str] = None,
    anchor_indices: Optional[str] = "80",
    random_sample: bool = True,
    seed: int = 42,
    curated_selection: bool = False
) -> List[Dict[str, Any]]:
    """
    Load prompts and select them per GenEval category.
    - categories: optional comma-separated category filter (e.g. 'two_object,colors,color_attr').
    - anchor_indices: prompts that are always included first (e.g. '80' - bench and ball).
    - random_sample: if True, sample the remaining prompts at random with the given seed.
    - prompts_per_category > 0: select exactly K prompts per selected category.
    - 'original_idx' is preserved so that results map back to the metadata file.
    """
    if not os.path.exists(prompt_path):
        repo_path = os.path.join(_REPO_ROOT, prompt_path)
        if not os.path.exists(repo_path):
            raise FileNotFoundError(
                f"Prompt file not found: '{prompt_path}'. Pass --prompts_file or place the file under prompt_files/."
            )
        prompt_path = repo_path

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

    # 0. Explicit prompt_indices take precedence over everything else
    if prompt_indices is not None and str(prompt_indices).strip() not in ["", "none", "None"]:
        s = str(prompt_indices).strip()
        if "-" in s and "," not in s:
            parts = s.split("-")
            selected_ids = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            selected_ids = [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
        id_to_item = {it["original_idx"]: it for it in items if "original_idx" in it}
        selected = [id_to_item[i] for i in selected_ids if i in id_to_item]
        print(f"[Explicit selection] Selected {len(selected)} prompts ({prompt_indices}):")
        for rank, s in enumerate(selected, 1):
            print(f"   {rank}. #{s['original_idx']} [{s.get('tag')}]: \"{s.get('prompt')}\"")
        return selected

    # 1. Curated selection (host-guest layouts and high-divergence prompts)
    CURATED_SUPERSTAR_INDICES = [80, 98, 95, 93, 97, 109, 110, 125, 138, 142, 156, 457, 464]
    if curated_selection:
        id_to_item = {it["original_idx"]: it for it in items if "original_idx" in it}
        selected = [id_to_item[i] for i in CURATED_SUPERSTAR_INDICES if i in id_to_item]
        print(f"[Curated selection] Selected {len(selected)} prompts with host-guest layouts and strong color bindings:")
        for rank, s in enumerate(selected, 1):
            print(f"   {rank}. #{s['original_idx']} [{s.get('tag')}]: \"{s.get('prompt')}\"")
        return selected

    # 2. Stratified per category
    if prompts_per_category is not None and prompts_per_category > 0:
        by_cat = {}
        for it in items:
            cat = it.get("tag", "general")
            by_cat.setdefault(cat, []).append(it)

        # Category filter (optional)
        if categories is not None and str(categories).strip().lower() not in ["", "none", "all"]:
            specified_cats = [c.strip() for c in str(categories).split(",") if c.strip()]
            active_cats = [c for c in specified_cats if c in by_cat]
            if not active_cats:
                print(f"Warning: none of the categories {specified_cats} exist in the metadata; using all categories.")
                active_cats = [c for c in GENEVAL_CATEGORIES if c in by_cat] + [c for c in by_cat if c not in GENEVAL_CATEGORIES]
            else:
                print(f"Category filter: {active_cats}")
        else:
            active_cats = [c for c in GENEVAL_CATEGORIES if c in by_cat] + [c for c in by_cat if c not in GENEVAL_CATEGORIES]

        # Anchor prompts (default includes #80 - bench and ball)
        anchor_ids = set()
        if anchor_indices is not None and str(anchor_indices).strip() not in ["", "none", "None"]:
            anchor_ids = {int(x.strip()) for x in str(anchor_indices).split(",") if x.strip().isdigit()}

        anchors = [it for it in items if it["original_idx"] in anchor_ids]
        picked_ids = set(anchor_ids)

        anchors_by_cat = {c: [] for c in active_cats}
        for a in anchors:
            cat = a.get("tag", "general")
            if cat in anchors_by_cat:
                anchors_by_cat[cat].append(a)

        rng = random.Random(seed)
        selected = []
        # Anchor prompts always come first
        selected.extend(anchors)

        print(f"Stratified selection of {prompts_per_category} prompts/category (random_sample={random_sample}, seed={seed}):")
        if anchors:
            anchor_desc = ", ".join(f"#{a['original_idx']} [{a.get('tag')}]: \"{a.get('prompt')}\"" for a in anchors)
            print(f"   Anchor prompts (always included): {anchor_desc}")

        for cat in active_cats:
            already = len(anchors_by_cat.get(cat, []))
            needed = max(0, prompts_per_category - already)
            available = [it for it in by_cat[cat] if it["original_idx"] not in picked_ids]

            if random_sample:
                picked = rng.sample(available, min(needed, len(available)))
            else:
                picked = available[:needed]

            for p in picked:
                picked_ids.add(p["original_idx"])
                selected.append(p)

            total_cat = already + len(picked)
            ids_str = ", ".join(str(p["original_idx"]) for p in anchors_by_cat.get(cat, []) + picked)
            print(f"   - [{cat}]: {total_cat}/{len(by_cat[cat])} prompts (IDs: {ids_str})")

        return selected

    # 3. Plain max_prompts limit
    if max_prompts > 0 and max_prompts < len(items):
        return items[:max_prompts]

    return items


# ======================================================================================
# TENSOR & IMAGE CONVERSION UTILITIES
# ======================================================================================
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] or [3, H, W] tensor in [-1, 1] to a PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.contiguous().permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def extract_single_pil_image(item: Any) -> Image.Image:
    """
    Extract a single PIL.Image from any nested structure
    (tuple, list, StableDiffusionPipelineOutput, torch.Tensor, np.ndarray, ...).
    """
    cur = item
    # 1. Unwrap outer tuples/lists
    while isinstance(cur, (tuple, list)) and len(cur) > 0:
        cur = cur[0]

    # 2. Unwrap the .images attribute of a diffusers pipeline output
    if hasattr(cur, "images"):
        cur = cur.images
        while isinstance(cur, (tuple, list)) and len(cur) > 0:
            cur = cur[0]

    # 3. Convert to PIL if needed
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
    Extract a list of PIL images from a diffusers pipeline output or any nested structure.
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
    """Convert a PIL image to a [1, 3, H, W] tensor in [-1, 1]."""
    clean_img = extract_single_pil_image(pil_img)
    arr = np.array(clean_img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor


# ======================================================================================
# REWARD SCORING UTILITIES
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
    Score a list of images for a prompt with the given reward model.
    Every element is first normalized to a plain PIL.Image.
    Returns a 1D numpy array with one score per image.
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
                print(f"Warning: ImageReward scoring failed: {e}")
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
            except Exception:
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
# PHASE 1: DUAL LOOKAHEAD ANNOTATION (LiDAR vs. RS-LiDAR)
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
    Phase 1:
    1. Generate n=50 lookahead particles z_0 (DPM-5) once.
    2. Decode them with the VAE to pixel space [-1, 1].
    3. Score them with both rewards:
       - r_vanilla = R(x_0)
       - r_RS = (1/M) sum_{m=1}^M R(clip(x_0 + sigma * u_m, -1, 1)) with CRN.
    4. Return two InMemoryLookaheadDatasets (LiDAR and RS-LiDAR).
    """
    generator = torch.Generator(device=device).manual_seed(seed + real_idx)

    # 1. Generate latents with a 5-step DPM-Solver
    scheduler_backup = pipe.scheduler
    if DPMSolverMultistepScheduler is not None:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Call the base StableDiffusionPipeline.__call__ to bypass the FK steering hook in Phase 1
    # (avoids KeyError: 'pred_original_sample' with DPMSolverMultistepScheduler).
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
    lookahead_latents = raw_latents.detach()  # [50, 4, 64, 64]

    # 2. Decode with the VAE to pixel space [-1, 1] (in chunks to avoid OOM)
    scaled = (lookahead_latents / vae.config.scaling_factor).to(device=device, dtype=vae.dtype)
    chunk_size = 10
    decoded_chunks = []
    for i in range(0, scaled.shape[0], chunk_size):
        chunk = scaled[i:i + chunk_size]
        dec = vae.decode(chunk, return_dict=False)[0].clamp(-1.0, 1.0)
        decoded_chunks.append(dec)
    decoded_tensors = torch.cat(decoded_chunks, dim=0)  # [50, 3, 512, 512]

    # Clean images as PIL
    clean_pils = pipe.image_processor.postprocess(decoded_tensors, output_type="pil")

    # 3. Vanilla reward r_vanilla
    r_vanilla = score_single_reward_model(clean_pils, prompt, "ImageReward", ir_model, device=device)

    # 4. RS-LiDAR smoothed reward r_RS (Monte Carlo with CRN)
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

    # 5. Pack into in-memory lookahead datasets
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
# PHASE 2: TARGET STEERING (UNSTEERED vs. LiDAR vs. RS-LiDAR)
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
    particle_scale: Optional[float] = None,
    target_steps: int = 50,
    lmbda: float = 5000.0,
    resampling_t_end: int = 200,
    seed: int = 42,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Phase-2 target steering from the same initial noise x_T for three branches:
    1. Unsteered baseline (CFG=12.5, fkd_args=None, no LiDAR guidance).
    2. LiDAR (CFG=12.5, particle_scale s=12.5, guided by ds_vanilla).
    3. RS-LiDAR (CFG=12.5, particle_scale s=12.5, guided by ds_rs).
    """
    if particle_scale is None:
        particle_scale = guidance_scale

    # Fixed initial noise x_T
    init_gen = torch.Generator(device=device).manual_seed(seed * 1000 + real_idx)
    init_latents = torch.randn((1, 4, 64, 64), generator=init_gen, device=device, dtype=torch.float16 if "cuda" in device else torch.float32)

    # Use the DDIM scheduler (SD 1.5 Table 2 setting)
    if DDIMScheduler is not None and not isinstance(pipe.scheduler, DDIMScheduler):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 1. Unsteered branch (text CFG = guidance_scale, fkd_args = None)
    if StableDiffusionPipeline is not None and isinstance(pipe, StableDiffusionPipeline):
        out_unsteered = StableDiffusionPipeline.__call__(
            pipe,
            prompt=prompt,
            num_inference_steps=target_steps,
            guidance_scale=guidance_scale,
            latents=init_latents.clone()
        )
    else:
        out_unsteered = pipe(
            prompt=prompt,
            num_inference_steps=target_steps,
            guidance_scale=guidance_scale,
            latents=init_latents.clone(),
            fkd_args=None
        )
    img_unsteered = extract_single_pil_image(out_unsteered)
    r_unsteered = float(score_single_reward_model([img_unsteered], prompt, "ImageReward", ir_model, device=device)[0])

    # 2. LiDAR branch (text CFG = guidance_scale, particle scale s = particle_scale)
    fkd_args_vanilla = dict(
        lmbda=lmbda,
        scale=particle_scale,
        reward_type="ImageReward",
        num_particles=1,
        use_smc=False,
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
        guidance_scale=guidance_scale,
        latents=init_latents.clone(),
        fkd_args=fkd_args_vanilla
    )
    img_vanilla = extract_single_pil_image(out_vanilla)
    r_vanilla = float(score_single_reward_model([img_vanilla], prompt, "ImageReward", ir_model, device=device)[0])

    # 3. RS-LiDAR branch (text CFG = guidance_scale, particle scale s = particle_scale)
    fkd_args_rs = dict(
        lmbda=lmbda,
        scale=particle_scale,
        reward_type="ImageReward",
        num_particles=1,
        use_smc=False,
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
        guidance_scale=guidance_scale,
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
# 1D DIRECTIONAL LANDSCAPE SLICING
# ======================================================================================
def generate_full_span_grid(slicing_points: int = 1000, margin: float = 0.2) -> np.ndarray:
    """
    Build a grid u in [-margin, 1 + margin] with about slicing_points (K) points.
    Guarantees, for any K (41, 61, 81, 100, 500, 1000, ...):
    1. u = 0.0 (LiDAR output) is exactly a grid point.
    2. u = 1.0 (RS-LiDAR output) is exactly a grid point.
    """
    margin = float(margin)
    total_span = 1.0 + 2.0 * margin
    ratio_mid = 1.0 / total_span

    n_mid_intervals = max(2, int(round(slicing_points * ratio_mid)))
    n_side_intervals = max(1, (slicing_points - n_mid_intervals) // 2)

    u_left = np.linspace(-margin, 0.0, n_side_intervals, endpoint=False)
    u_mid = np.linspace(0.0, 1.0, n_mid_intervals + 1, endpoint=True)
    u_mid[0] = 0.0
    u_mid[-1] = 1.0
    u_right = np.linspace(1.0, 1.0 + margin, n_side_intervals + 1, endpoint=True)[1:]

    u_vals = np.concatenate([u_left, u_mid, u_right])
    assert 0.0 in u_vals, "u=0.0 (LiDAR) is missing from the grid"
    assert 1.0 in u_vals, "u=1.0 (RS-LiDAR) is missing from the grid"
    return u_vals


@torch.inference_mode()
def compute_1d_landscape_profile(
    img_vanilla: Any,
    img_rs: Any,
    prompt: str,
    ir_model: Any,
    slicing_points: int = 1000,
    alpha_max: float = 0.2,
    slicing_sigma: float = 1.0,
    mc_samples: int = 4,
    slicing_mode: str = "full_span",
    slicing_margin: float = 0.2,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    1D slicing in pixel space:
    - slicing_mode == 'full_span': the full path from LiDAR (u=0) to RS-LiDAR (u=1), extended on both ends:
        x(u) = clip(x_vanilla + u * (x_RS - x_vanilla), -1, 1), u in [-margin, 1 + margin]
      u=0.0 and u=1.0 are always exact grid points.
    - slicing_mode == 'local': a local sweep around the LiDAR output:
        x(alpha) = clip(x_vanilla + alpha * v, -1, 1), alpha in [-alpha_max, alpha_max]
    Measures:
      f_vanilla(u) = R(x(u))
      f_RS(u)      = (1/M) sum R(clip(x(u) + sigma * u_m, -1, 1)) with CRN
      Numerical Secant Derivative |Df(u)| = |f(u_{i+1}) - f(u_i)| / delta_x
    """
    img_vanilla = extract_single_pil_image(img_vanilla)
    img_rs = extract_single_pil_image(img_rs)
    t_v = pil_to_tensor_neg1_to_1(img_vanilla, device=device)
    t_rs = pil_to_tensor_neg1_to_1(img_rs, device=device)

    # 1. Difference vector between the RS-LiDAR and LiDAR outputs
    diff = t_rs - t_v
    dist_l2 = float(torch.norm(diff.reshape(-1), p=2).item())

    if dist_l2 < 1e-6:
        v_rand = torch.randn_like(t_v)
        v = v_rand / torch.norm(v_rand.reshape(-1), p=2)
    else:
        v = diff / dist_l2

    # 2. Slicing grid
    if slicing_mode == "full_span":
        alphas = generate_full_span_grid(slicing_points=slicing_points, margin=slicing_margin)
    else:
        alphas = np.linspace(-alpha_max, alpha_max, slicing_points)

    curve_vanilla = []
    curve_rs = []

    # Fix the CRN noise along the whole slice so that the RS-LiDAR curve has no local Monte Carlo jitter
    crn_noise_list = []
    if slicing_sigma > 0.0 and mc_samples > 0:
        for m in range(mc_samples):
            crn_gen = torch.Generator(device=device).manual_seed(20000 + m)
            n_m = torch.randn(t_v.shape, generator=crn_gen, device=device, dtype=t_v.dtype) * slicing_sigma
            crn_noise_list.append(n_m)

    for a in alphas:
        a_f = float(a)
        if slicing_mode == "full_span":
            # Use the original tensors at the endpoints to avoid floating-point drift
            if abs(a_f - 0.0) < 1e-9:
                t_a = t_v
            elif abs(a_f - 1.0) < 1e-9:
                t_a = t_rs
            else:
                t_a = (t_v + a_f * diff).clamp(-1.0, 1.0)
        else:
            t_a = (t_v + a_f * v).clamp(-1.0, 1.0)

        img_a = tensor_to_pil(t_a)

        # Vanilla reward along the slice
        sc_v = float(score_single_reward_model([img_a], prompt, "ImageReward", ir_model, device=device)[0])
        curve_vanilla.append(sc_v)

        # RS-LiDAR smoothed reward along the slice
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

    # 3. Numerical secant derivatives
    if slicing_mode == "full_span":
        delta_u = np.diff(alphas)
        delta_x = delta_u * dist_l2 if dist_l2 > 1e-6 else delta_u
        deriv_vanilla = np.abs(np.diff(curve_vanilla) / delta_x)
        deriv_rs = np.abs(np.diff(curve_rs) / delta_x)
    else:
        delta_alpha = float(alphas[1] - alphas[0]) if len(alphas) > 1 else 1.0
        deriv_vanilla = np.abs(np.diff(curve_vanilla) / delta_alpha)
        deriv_rs = np.abs(np.diff(curve_rs) / delta_alpha)

    l_max_vanilla = float(np.max(deriv_vanilla)) if len(deriv_vanilla) > 0 else 0.0
    l_max_rs = float(np.max(deriv_rs)) if len(deriv_rs) > 0 else 0.0
    l_mean_vanilla = float(np.mean(deriv_vanilla)) if len(deriv_vanilla) > 0 else 0.0
    l_mean_rs = float(np.mean(deriv_rs)) if len(deriv_rs) > 0 else 0.0

    # Theoretical bound (Theorem 1): Delta R / (sigma * sqrt(2*pi))
    reward_range = max(float(np.ptp(curve_vanilla)), 1.5)
    theoretical_bound = float(reward_range / (slicing_sigma * math.sqrt(2 * math.pi))) if slicing_sigma > 0 else 999.0

    return {
        "dist_l2": dist_l2,
        "alphas": alphas.tolist(),
        "u_vals": alphas.tolist(),
        "slicing_mode": slicing_mode,
        "slicing_margin": float(slicing_margin) if slicing_mode == "full_span" else float(alpha_max),
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
# DIVERGENCE SCORES & TOP-K RANKING
# ======================================================================================
def compute_prompt_rescue_score(
    delta_r: float,
    r_unsteered: float,
    r_vanilla: float,
    l_max_vanilla: float,
    l_max_rs: float,
    dist_l2: float,
    tau_dist: float = 500.0
) -> float:
    """
    Failure-rescue divergence index (bounded terms):
      Score = Delta R
            + 1.5 * max(0, R_unsteered - R_vanilla)
            + 1.0 * min(1.0, ||dx||_2 / tau_dist)
            + 0.5 * tanh((L_vanilla - L_RS) / (L_vanilla + 1e-6))
    Delta R is kept signed, so prompts where RS-LiDAR lowers the reward are penalized and the
    top-K table only contains genuine rescues.
    """
    term_gain = float(delta_r)
    term_rescue = 1.5 * max(0.0, float(r_unsteered - r_vanilla))
    term_visual = 1.0 * min(1.0, float(dist_l2) / max(tau_dist, 1e-6))
    term_smooth = 0.5 * math.tanh((float(l_max_vanilla) - float(l_max_rs)) / (float(l_max_vanilla) + 1e-6))
    return float(term_gain + term_rescue + term_visual + term_smooth)


def compute_prompt_smoothness_score(
    delta_r: float,
    l_max_vanilla: float,
    l_max_rs: float
) -> float:
    """
    Lipschitz smoothness divergence index:
      Score = tanh((L_vanilla - L_RS) / (L_vanilla + 1e-6)) * ln(1 + max(0, L_vanilla / (L_RS + 1e-6)))
            + 0.2 * max(0, Delta R)
    """
    tanh_diff = math.tanh((float(l_max_vanilla) - float(l_max_rs)) / (float(l_max_vanilla) + 1e-6))
    ratio = float(l_max_vanilla) / (float(l_max_rs) + 1e-6)
    log_ratio = math.log(1.0 + max(0.0, ratio))
    return float(tanh_diff * log_ratio + 0.2 * max(0.0, float(delta_r)))


def compute_prompt_divergence_score(
    delta_r: float,
    l_max_vanilla: float,
    l_max_rs: float,
    dist_l2: float,
    r_unsteered: Optional[float] = None,
    r_vanilla: Optional[float] = None,
    total_pixels: int = 3 * 512 * 512,
    tau_dist: float = 500.0
) -> float:
    """
    Combined score: the failure-rescue score when r_unsteered and r_vanilla are given,
    otherwise a bounded visual + smoothness score.
    """
    if r_unsteered is not None and r_vanilla is not None:
        return compute_prompt_rescue_score(
            delta_r=delta_r,
            r_unsteered=r_unsteered,
            r_vanilla=r_vanilla,
            l_max_vanilla=l_max_vanilla,
            l_max_rs=l_max_rs,
            dist_l2=dist_l2,
            tau_dist=tau_dist
        )
    term_gain = float(delta_r)
    term_visual = 1.0 * min(1.0, float(dist_l2) / max(tau_dist, 1e-6))
    term_smooth = 0.5 * math.tanh((float(l_max_vanilla) - float(l_max_rs)) / (float(l_max_vanilla) + 1e-6))
    return float(term_gain + term_visual + term_smooth)


# ======================================================================================
# VISUALIZATION
# ======================================================================================
def plot_topk_divergence_panel(
    topk_records: List[Dict[str, Any]],
    output_path: str,
    panel_title: str = "RS-LiDAR vs. Vanilla LiDAR: Qualitative and 1D Landscape Comparison"
):
    """
    Side-by-side panel with K rows x 5 columns:
    - Column 1: unsteered baseline image with its reward R.
    - Column 2: LiDAR image with R and L_mean (red frame).
    - Column 3: RS-LiDAR image with R, L_mean and the L ratio (green frame).
    - Column 4: spacer (width_ratio=0.16) so captions do not overlap the images.
    - Column 5: 1D slicing profile (full span or local).
    """
    num_rows = len(topk_records)
    if num_rows == 0:
        return

    fig = plt.figure(figsize=(20.5, 5.3 * num_rows), dpi=300)
    gs = fig.add_gridspec(
        num_rows, 5,
        width_ratios=[1.0, 1.0, 1.0, 0.16, 1.12],
        wspace=0.15, hspace=0.52,
        top=0.94, bottom=0.06, left=0.04, right=0.97
    )

    for i, rec in enumerate(topk_records):
        idx = rec.get("original_idx", rec.get("prompt_idx", i))
        p_str = rec.get("prompt", "")
        dr = rec.get("delta_r", 0.0)
        l_ratio = rec.get("l_reduction_ratio", 1.0)
        r_u = rec.get("r_unsteered", 0.0)
        r_v = rec.get("r_vanilla", 0.0)
        r_rs = rec.get("r_rs", 0.0)

        # L_mean and the L ratio based on the mean slope
        slicing = rec.get("slicing", {})
        l_mean_v = rec.get("l_mean_vanilla", slicing.get("l_mean_vanilla", 0.0))
        l_mean_rs = rec.get("l_mean_rs", slicing.get("l_mean_rs", 0.0))
        l_ratio_mean = float(l_mean_v / (l_mean_rs + 1e-6)) if l_mean_rs > 0 else l_ratio

        # Load an image from the record path, falling back to default locations inside the repository
        def _load_image_safe(p_rec, fname):
            if p_rec and os.path.exists(p_rec):
                try:
                    return Image.open(p_rec)
                except Exception:
                    pass
            # Fallback locations
            local_candidates = [
                os.path.join(_REPO_ROOT, "results", "ablation1_sensitive_landscape", "images", fname),
                os.path.join(_REPO_ROOT, "figures", "ablation1", "images", fname)
            ]
            for c in local_candidates:
                if os.path.exists(c):
                    try:
                        return Image.open(c)
                    except Exception:
                        pass
            return None

        # ----------------------------------------------------------------------
        # COLUMN 0: UNSTEERED BASELINE
        # ----------------------------------------------------------------------
        ax_un = fig.add_subplot(gs[i, 0])
        img_un = rec.get("img_unsteered_obj")
        if img_un is None:
            img_un = _load_image_safe(rec.get("img_unsteered_path"), f"{idx:05d}_unsteered.png")
        if img_un is not None:
            ax_un.imshow(img_un)
            ax_un.set_title(f"Unsteered Baseline\n$R = {r_u:.3f}$", fontsize=11.5, fontweight="bold", color="#1D3557", pad=8)
        else:
            ax_un.text(0.5, 0.5, "Image N/A", ha="center", va="center")
        ax_un.axis("off")

        # ----------------------------------------------------------------------
        # COLUMN 1: LiDAR
        # ----------------------------------------------------------------------
        ax_vl = fig.add_subplot(gs[i, 1])
        img_vl = rec.get("img_vanilla_obj")
        if img_vl is None:
            img_vl = _load_image_safe(rec.get("img_vanilla_path"), f"{idx:05d}_vanilla.png")
        if img_vl is not None:
            ax_vl.imshow(img_vl)
            ax_vl.set_title(f"Vanilla LiDAR\n$R = {r_v:.3f}$ | $L_{{\\text{{mean}}}} = {l_mean_v:.4f}$", fontsize=11.5, fontweight="bold", color="#E63946", pad=8)
            for spine in ax_vl.spines.values():
                spine.set_edgecolor("#E63946")
                spine.set_linewidth(2.2)
        else:
            ax_vl.text(0.5, 0.5, "Image N/A", ha="center", va="center")
        ax_vl.axis("off")

        # ----------------------------------------------------------------------
        # COLUMN 2: RS-LiDAR
        # ----------------------------------------------------------------------
        ax_rs = fig.add_subplot(gs[i, 2])
        img_rs = rec.get("img_rs_obj")
        if img_rs is None:
            img_rs = _load_image_safe(rec.get("img_rs_path"), f"{idx:05d}_rs.png")
        if img_rs is not None:
            ax_rs.imshow(img_rs)
            ax_rs.set_title(f"RS-LiDAR\n$R = {r_rs:.3f}$ | $L_{{\\text{{mean}}}} = {l_mean_rs:.4f}$ | $L_{{\\text{{ratio}}}} = {l_ratio_mean:.1f}\\times$", fontsize=11.5, fontweight="bold", color="#2A9D8F", pad=8)
            for spine in ax_rs.spines.values():
                spine.set_edgecolor("#2A9D8F")
                spine.set_linewidth(2.2)
        else:
            ax_rs.text(0.5, 0.5, "Image N/A", ha="center", va="center")
        ax_rs.axis("off")

        # ----------------------------------------------------------------------
        # CAPTION: PROMPT AND KEY METRICS
        # ----------------------------------------------------------------------
        caption_text = f"Prompt #{idx}: \"{p_str}\"  (\u0394R = {dr:+.3f}, Mean Smoothness Gain: {l_ratio_mean:.1f}\u00d7)"
        ax_vl.text(
            0.5, -0.16, caption_text, transform=ax_vl.transAxes,
            fontsize=11.5, fontweight="bold", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.2)
        )

        # ----------------------------------------------------------------------
        # COLUMN 3: SPACER
        # ----------------------------------------------------------------------
        ax_sp = fig.add_subplot(gs[i, 3])
        ax_sp.axis("off")

        # ----------------------------------------------------------------------
        # COLUMN 4: 1D LANDSCAPE SLICE
        # ----------------------------------------------------------------------
        ax_sl = fig.add_subplot(gs[i, 4])
        slicing = rec.get("slicing", {})
        alphas = slicing.get("alphas", [])
        cv = np.array(slicing.get("curve_vanilla", []))
        crs = np.array(slicing.get("curve_rs", []))
        s_mode = slicing.get("slicing_mode", "full_span")

        if len(alphas) > 0 and len(cv) > 0:
            is_full_span = (s_mode == "full_span") or (len(alphas) > 1 and max(alphas) > 0.5)

            if is_full_span:
                ax_sl.plot(alphas, cv, color="#E63946", lw=2.4, label="Vanilla $R(x)$", linestyle="-")
                ax_sl.plot(alphas, crs, color="#2A9D8F", lw=2.6, label="RS-LiDAR $R_\\sigma(x)$", linestyle="-")
                ax_sl.axvline(0.0, color="#E63946", linestyle="--", alpha=0.7, label="Vanilla Point ($u=0$)")
                ax_sl.axvline(1.0, color="#2A9D8F", linestyle="--", alpha=0.7, label="RS-LiDAR Point ($u=1$)")
                ax_sl.scatter([0.0], [r_v], color="#E63946", s=35, zorder=5)
                ax_sl.scatter([1.0], [r_rs], color="#2A9D8F", s=35, zorder=5)
                ax_sl.set_xlabel("Trajectory $u$ (Vanilla: $0 \\rightarrow$ RS: $1$)", fontsize=10.5, fontweight="bold")
            else:
                ax_sl.plot(alphas, cv, color="#E63946", lw=2.4, label="Vanilla $R(x)$", linestyle="-")
                ax_sl.plot(alphas, crs, color="#2A9D8F", lw=2.6, label="RS-LiDAR $R_\\sigma(x)$", linestyle="-")
                ax_sl.axvline(0.0, color="#1D3557", linestyle=":", alpha=0.6, label="Vanilla Point")
                ax_sl.scatter([0.0], [r_v], color="#E63946", s=35, zorder=5)
                ax_sl.set_xlabel("Perturbation $\\alpha$", fontsize=10.5, fontweight="bold")

            # Dynamic auto-fit y-limits with padding
            ptp = max(float(np.ptp(cv)), float(np.ptp(crs)), 0.5)
            y_min = min(float(np.min(cv)), float(np.min(crs)), r_v, r_rs) - 0.08 * ptp
            y_max = max(float(np.max(cv)), float(np.max(crs)), r_v, r_rs) + 0.08 * ptp
            ax_sl.set_ylim(y_min, y_max)

            ax_sl.set_ylabel("Reward Score", fontsize=10.5, fontweight="bold", labelpad=8)
            ax_sl.set_title(f"1D Landscape Slice (Idx #{idx})\n$L_{{\\text{{ratio}}}} = {l_ratio_mean:.1f}\\times$", fontsize=11, fontweight="bold")
            ax_sl.grid(True, linestyle="--", alpha=0.5)
            ax_sl.legend(fontsize=8.5, loc="best", frameon=True)
        else:
            ax_sl.text(0.5, 0.5, "Profile N/A", ha="center", va="center")

    plt.suptitle(panel_title, fontsize=16, fontweight="bold", y=0.985, color="#1D3557")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved top-{num_rows} comparison panel: {output_path}")

# Backward-compatible alias
plot_top5_divergence_panel = plot_topk_divergence_panel


def plot_summary_landscape_curves(
    all_records: List[Dict[str, Any]],
    output_path: str
):
    """Plot the 1D slicing curves of all prompts."""
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
    print(f"Saved 1D landscape curves: {output_path}")


# ======================================================================================
# MERGE SHARDS
# ======================================================================================
def merge_shard_checkpoints(output_dir: str, top_k: int = 10):
    """
    Merge all checkpoint_ablation1_shard_*.json files in output_dir, compute the rescue and
    smoothness divergence indices, and export the top-K rescue / smoothness tables and figures.
    """
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint_ablation1_shard_*.json")))
    if not ckpts:
        print(f"No checkpoint files found in {output_dir}")
        return

    print(f"Found {len(ckpts)} shard checkpoints: {[os.path.basename(c) for c in ckpts]}")
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
            print(f"Warning: could not read checkpoint {c_path}: {e}")

    print(f"Merged {len(merged_records)} unique prompts.")

    # Recompute the rescue and smoothness scores for every record
    for r in merged_records:
        r_un = r.get("r_unsteered", 0.0)
        r_vl = r.get("r_vanilla", 0.0)
        r_rs = r.get("r_rs", 0.0)
        dr = r.get("delta_r", r_rs - r_vl)
        lv = r.get("l_max_vanilla", 0.0)
        lrs = r.get("l_max_rs", 0.0)
        dl2 = r.get("dist_l2", 0.0)

        resc_score = compute_prompt_rescue_score(
            delta_r=dr,
            r_unsteered=r_un,
            r_vanilla=r_vl,
            l_max_vanilla=lv,
            l_max_rs=lrs,
            dist_l2=dl2
        )
        smooth_score = compute_prompt_smoothness_score(
            delta_r=dr,
            l_max_vanilla=lv,
            l_max_rs=lrs
        )
        r["rescue_score"] = resc_score
        r["smoothness_score"] = smooth_score
        r["divergence_score"] = resc_score  # kept for backward compatibility

    # 1. Sort by rescue score (main summary table)
    merged_records.sort(key=lambda x: x.get("rescue_score", 0.0), reverse=True)

    # 2. Summary CSV over all prompts
    summary_csv_path = os.path.join(output_dir, "ablation1_summary.csv")
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write("rank,original_idx,category,prompt,r_unsteered,r_vanilla,r_rs,delta_r,l_max_vanilla,l_max_rs,l_reduction_ratio,dist_l2,rescue_score,smoothness_score,divergence_score\n")
        for rank, r in enumerate(merged_records, 1):
            p_text = r.get("prompt", "").replace('"', '""')
            cat = r.get("category", r.get("tag", "general"))
            f.write(
                f"{rank},{r.get('original_idx', '')},{cat},\"{p_text}\","
                f"{r.get('r_unsteered', 0.0):.4f},{r.get('r_vanilla', 0.0):.4f},{r.get('r_rs', 0.0):.4f},"
                f"{r.get('delta_r', 0.0):.4f},{r.get('l_max_vanilla', 0.0):.2f},{r.get('l_max_rs', 0.0):.2f},"
                f"{r.get('l_reduction_ratio', 1.0):.2f},{r.get('dist_l2', 0.0):.4f},"
                f"{r.get('rescue_score', 0.0):.4f},{r.get('smoothness_score', 0.0):.4f},{r.get('divergence_score', 0.0):.4f}\n"
            )
    print(f"Saved summary CSV: {summary_csv_path}")

    # 3. Top-K failure-rescue prompts
    top_rescue = sorted(merged_records, key=lambda x: x.get("rescue_score", 0.0), reverse=True)[:top_k]
    top_rescue_csv = os.path.join(output_dir, f"ablation1_top{top_k}_rescue.csv")
    with open(top_rescue_csv, "w", encoding="utf-8") as f:
        f.write("rank,original_idx,category,prompt,r_unsteered,r_vanilla,r_rs,delta_r,l_max_vanilla,l_max_rs,l_reduction_ratio,dist_l2,rescue_score\n")
        for rank, r in enumerate(top_rescue, 1):
            p_text = r.get("prompt", "").replace('"', '""')
            cat = r.get("category", r.get("tag", "general"))
            f.write(
                f"{rank},{r.get('original_idx', '')},{cat},\"{p_text}\","
                f"{r.get('r_unsteered', 0.0):.4f},{r.get('r_vanilla', 0.0):.4f},{r.get('r_rs', 0.0):.4f},{r.get('delta_r', 0.0):.4f},"
                f"{r.get('l_max_vanilla', 0.0):.2f},{r.get('l_max_rs', 0.0):.2f},"
                f"{r.get('l_reduction_ratio', 1.0):.2f},{r.get('dist_l2', 0.0):.4f},{r.get('rescue_score', 0.0):.4f}\n"
            )
    print(f"Saved top-{top_k} rescue CSV: {top_rescue_csv}")

    # Backward-compatible file names
    alias_csv_path = os.path.join(output_dir, "ablation1_top5_rescue.csv")
    shutil.copyfile(top_rescue_csv, alias_csv_path)
    shutil.copyfile(top_rescue_csv, os.path.join(output_dir, "ablation1_top5_divergence.csv"))

    # 4. Top-K Lipschitz smoothness prompts
    top_smoothness = sorted(merged_records, key=lambda x: x.get("smoothness_score", 0.0), reverse=True)[:top_k]
    top_smooth_csv = os.path.join(output_dir, f"ablation1_top{top_k}_smoothness.csv")
    with open(top_smooth_csv, "w", encoding="utf-8") as f:
        f.write("rank,original_idx,category,prompt,r_vanilla,r_rs,delta_r,l_max_vanilla,l_max_rs,l_reduction_ratio,smoothness_score\n")
        for rank, r in enumerate(top_smoothness, 1):
            p_text = r.get("prompt", "").replace('"', '""')
            cat = r.get("category", r.get("tag", "general"))
            f.write(
                f"{rank},{r.get('original_idx', '')},{cat},\"{p_text}\","
                f"{r.get('r_vanilla', 0.0):.4f},{r.get('r_rs', 0.0):.4f},{r.get('delta_r', 0.0):.4f},"
                f"{r.get('l_max_vanilla', 0.0):.2f},{r.get('l_max_rs', 0.0):.2f},"
                f"{r.get('l_reduction_ratio', 1.0):.2f},{r.get('smoothness_score', 0.0):.4f}\n"
            )
    print(f"Saved top-{top_k} smoothness CSV: {top_smooth_csv}")

    alias_smooth_csv = os.path.join(output_dir, "ablation1_top5_smoothness.csv")
    shutil.copyfile(top_smooth_csv, alias_smooth_csv)

    # 5. Detailed metrics JSON
    metrics_json_path = os.path.join(output_dir, "ablation1_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_prompts": len(merged_records),
            f"top{top_k}_rescue_prompts": [
                {
                    "rank": i + 1,
                    "original_idx": r.get("original_idx"),
                    "category": r.get("category"),
                    "prompt": r.get("prompt"),
                    "rescue_score": r.get("rescue_score"),
                    "delta_r": r.get("delta_r"),
                    "l_reduction_ratio": r.get("l_reduction_ratio")
                }
                for i, r in enumerate(top_rescue)
            ],
            f"top{top_k}_smoothness_prompts": [
                {
                    "rank": i + 1,
                    "original_idx": r.get("original_idx"),
                    "category": r.get("category"),
                    "prompt": r.get("prompt"),
                    "smoothness_score": r.get("smoothness_score"),
                    "l_reduction_ratio": r.get("l_reduction_ratio"),
                    "delta_r": r.get("delta_r")
                }
                for i, r in enumerate(top_smoothness)
            ],
            "top5_rescue_prompts": [
                {
                    "rank": i + 1,
                    "original_idx": r.get("original_idx"),
                    "category": r.get("category"),
                    "prompt": r.get("prompt"),
                    "rescue_score": r.get("rescue_score"),
                    "delta_r": r.get("delta_r"),
                    "l_reduction_ratio": r.get("l_reduction_ratio")
                }
                for i, r in enumerate(top_rescue[:5])
            ],
            "top5_smoothness_prompts": [
                {
                    "rank": i + 1,
                    "original_idx": r.get("original_idx"),
                    "category": r.get("category"),
                    "prompt": r.get("prompt"),
                    "smoothness_score": r.get("smoothness_score"),
                    "l_reduction_ratio": r.get("l_reduction_ratio"),
                    "delta_r": r.get("delta_r")
                }
                for i, r in enumerate(top_smoothness[:5])
            ],
            "all_records": merged_records
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics JSON: {metrics_json_path}")

    # 6. Top-K comparison panels (rescue and smoothness)
    panel_rescue_path = os.path.join(output_dir, f"ablation1_top{top_k}_rescue_panel.png")
    plot_topk_divergence_panel(
        top_rescue,
        panel_rescue_path,
        panel_title=f"RS-LiDAR ABLATION 1: TOP {len(top_rescue)} FAILURE-RESCUE PROMPTS (VANILLA COLLAPSE)"
    )

    panel_smooth_path = os.path.join(output_dir, f"ablation1_top{top_k}_smoothness_panel.png")
    plot_topk_divergence_panel(
        top_smoothness,
        panel_smooth_path,
        panel_title=f"RS-LiDAR ABLATION 1: TOP {len(top_smoothness)} LIPSCHITZ SMOOTHNESS PROMPTS (THEORETICAL CERTIFICATION)"
    )

    # Backward-compatible panel file names
    alias_panel_path = os.path.join(output_dir, "ablation1_top5_rescue_panel.png")
    shutil.copyfile(panel_rescue_path, alias_panel_path)
    shutil.copyfile(panel_rescue_path, os.path.join(output_dir, "ablation1_top5_divergence_panel.png"))

    alias_smooth_panel = os.path.join(output_dir, "ablation1_top5_smoothness_panel.png")
    shutil.copyfile(panel_smooth_path, alias_smooth_panel)

    # Also write comparison_selected_3_prompts_panel.png
    panel_3prompts = os.path.join(output_dir, "comparison_selected_3_prompts_panel.png")
    shutil.copyfile(panel_rescue_path, panel_3prompts)

    curves_path = os.path.join(output_dir, "ablation1_landscape_1d_curves.png")
    plot_summary_landscape_curves(merged_records, curves_path)


# ======================================================================================
# MAIN EXPERIMENT LOOP
# ======================================================================================
def run_ablation1_experiment(pipe, vae, ir_model, prompts_list, args, device="cuda"):
    """Main experiment loop for one shard."""
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    curves_dir = os.path.join(args.output_dir, "slicing_curves")
    os.makedirs(curves_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, f"checkpoint_ablation1_shard_{args.shard_id}.json")

    # Split prompts across shards
    total_prompts = len(prompts_list)
    if args.num_shards > 1:
        prompts_per_shard = math.ceil(total_prompts / args.num_shards)
        start_idx = args.shard_id * prompts_per_shard
        end_idx = min(start_idx + prompts_per_shard, total_prompts)
        active_prompts = prompts_list[start_idx:end_idx]
        print(f"Shard {args.shard_id + 1}/{args.num_shards}: processing {len(active_prompts)} prompts ({start_idx} to {end_idx - 1}).")
    else:
        start_idx = 0
        end_idx = total_prompts
        active_prompts = prompts_list
        print(f"Single process: processing all {len(active_prompts)} prompts.")

    # Load the existing checkpoint when resuming
    completed_records = []
    completed_ids = set()
    if args.resume and not args.overwrite and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            completed_records = c_data.get("records", [])
            for r in completed_records:
                completed_ids.add(r.get("original_idx", r.get("prompt_idx")))
            print(f"Resumed {len(completed_records)} prompts from {ckpt_path}.")
        except Exception as e:
            print(f"Warning: could not read checkpoint: {e}")

    # Force re-running selected prompts regardless of the checkpoint
    if getattr(args, "force_rerun_indices", None):
        s_force = str(args.force_rerun_indices).strip()
        if s_force and s_force.lower() not in ["none", ""]:
            force_ids = {int(x.strip()) for x in s_force.split(",") if x.strip().isdigit()}
            if force_ids:
                old_len = len(completed_records)
                completed_records = [r for r in completed_records if r.get("original_idx", r.get("prompt_idx")) not in force_ids]
                completed_ids = completed_ids - force_ids
                dropped_cnt = old_len - len(completed_records)
                if dropped_cnt > 0:
                    print(f"[Force rerun] Removed {dropped_cnt} prompts ({sorted(list(force_ids))}) from the checkpoint.")

    for idx, item in enumerate(tqdm(active_prompts, desc=f"Shard-{args.shard_id}")):
        real_idx = item.get("original_idx", idx)
        prompt_text = item.get("prompt", "")
        category = item.get("tag", item.get("category", "general"))

        if real_idx in completed_ids:
            continue

        print(f"\n[{idx + 1}/{len(active_prompts)}] Prompt #{real_idx} [{category}]: \"{prompt_text}\"")

        # ----------------------------------------------------------------------
        # STEP 1: PHASE-1 LOOKAHEAD ANNOTATION
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
        # STEP 2: PHASE-2 TARGET STEERING
        # ----------------------------------------------------------------------
        p2_results = run_phase2_target_steering(
            pipe=pipe,
            prompt=prompt_text,
            real_idx=real_idx,
            ds_vanilla=ds_vanilla,
            ds_rs=ds_rs,
            ir_model=ir_model,
            guidance_scale=args.guidance_scale,
            particle_scale=getattr(args, "particle_scale", None),
            target_steps=args.target_steps,
            lmbda=args.lmbda,
            resampling_t_end=args.resampling_t_end,
            seed=args.seed,
            device=device
        )

        # Save the generated images
        unsteered_path = os.path.join(images_dir, f"{real_idx:05d}_unsteered.png")
        vanilla_path = os.path.join(images_dir, f"{real_idx:05d}_vanilla.png")
        rs_path = os.path.join(images_dir, f"{real_idx:05d}_rs.png")
        extract_single_pil_image(p2_results["img_unsteered"]).save(unsteered_path)
        extract_single_pil_image(p2_results["img_vanilla"]).save(vanilla_path)
        extract_single_pil_image(p2_results["img_rs"]).save(rs_path)

        # ----------------------------------------------------------------------
        # STEP 3: 1D DIRECTIONAL LANDSCAPE SLICING
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
            slicing_mode=getattr(args, "slicing_mode", "full_span"),
            slicing_margin=getattr(args, "slicing_margin", 0.2),
            device=device
        )

        # ----------------------------------------------------------------------
        # STEP 4: DIVERGENCE SCORES (RESCUE & SMOOTHNESS)
        # ----------------------------------------------------------------------
        resc_score = compute_prompt_rescue_score(
            delta_r=p2_results["delta_r"],
            r_unsteered=p2_results["r_unsteered"],
            r_vanilla=p2_results["r_vanilla"],
            l_max_vanilla=slicing_res["l_max_vanilla"],
            l_max_rs=slicing_res["l_max_rs"],
            dist_l2=slicing_res["dist_l2"]
        )
        smooth_score = compute_prompt_smoothness_score(
            delta_r=p2_results["delta_r"],
            l_max_vanilla=slicing_res["l_max_vanilla"],
            l_max_rs=slicing_res["l_max_rs"]
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
            "rescue_score": resc_score,
            "smoothness_score": smooth_score,
            "divergence_score": resc_score,
            "slicing": slicing_res,
            "phase1_stats": p1_stats
        }

        completed_records.append(record)
        completed_ids.add(real_idx)

        # Write the checkpoint after every prompt
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "shard_id": args.shard_id,
                "completed_count": len(completed_records),
                "records": completed_records
            }, f, indent=2, ensure_ascii=False)

    return completed_records


def main():
    parser = argparse.ArgumentParser(description="RS-LiDAR Ablation 1: Sensitive Prompt & Non-Lipschitz Landscape Slicing")

    # Prompts
    parser.add_argument("--prompts_file", type=str, default="prompt_files/geneval_metadata.jsonl", help="Prompt file (.jsonl or .json)")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories to use (e.g. 'two_object,colors,color_attr' or 'all')")
    parser.add_argument("--prompts_per_category", type=int, default=2, help="Prompts selected per category")
    parser.add_argument("--anchor_indices", type=str, default="80,98", help="Prompt IDs that are always included first (default '80,98': bench and ball, couch and horse)")
    parser.add_argument("--force_rerun_indices", type=str, default="80,98", help="Prompt IDs that are always re-run, ignoring the checkpoint (default '80,98')")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top-ranked prompts in the panels and CSVs")
    parser.add_argument("--random_sample", action="store_true", default=True, help="Randomly sample the remaining prompts with the given seed")
    parser.add_argument("--no_random_sample", action="store_false", dest="random_sample")
    parser.add_argument("--num_prompts", type=int, default=12, help="Total number of prompts (when not stratifying)")
    parser.add_argument("--prompt_indices", type=str, default=None, help="Explicit prompt IDs (e.g. '0,5,11' or '0-11')")
    parser.add_argument("--curated_selection", action="store_true", default=False, help="Use the curated set of host-guest / high-divergence prompts (bench and ball, couch and horse, book and laptop, ...)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Diffusion model
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model")

    # Phase 1 (lookahead)
    parser.add_argument("--num_lookahead", type=int, default=50, help="Number of lookahead particles n")
    parser.add_argument("--lookahead_steps", type=int, default=5, help="DPM-Solver steps in Phase 1")
    parser.add_argument("--sigma", type=float, default=1.0, help="RS-LiDAR Gaussian smoothing radius sigma")
    parser.add_argument("--mc_samples", type=int, default=4, help="Monte Carlo samples M in Phase 1")
    parser.add_argument("--use_crn", action="store_true", default=True, help="Use common random numbers (CRN)")
    parser.add_argument("--no_crn", action="store_false", dest="use_crn")

    # Phase 2 (target steering)
    parser.add_argument("--guidance_scale", type=float, default=12.5, help="Text classifier-free guidance scale in Phase 2 (SD 1.5 Table 2 setting)")
    parser.add_argument("--particle_scale", type=float, default=None, help="Lookahead particle guidance scale s (default: same as --guidance_scale)")
    parser.add_argument("--num_target_images", type=int, default=1, help="Images generated per prompt in Phase 2")
    parser.add_argument("--target_steps", type=int, default=50, help="DDIM steps in Phase 2")
    parser.add_argument("--lmbda", type=float, default=5000.0, help="Lambda softmax exponent")
    parser.add_argument("--resampling_t_end", type=int, default=200, help="Timestep at which guidance stops (t_end)")
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward", help="Guidance reward function")

    # 1D landscape slicing
    parser.add_argument("--slicing_points", type=int, default=1000, help="Number of points on the 1D slicing grid")
    parser.add_argument("--slicing_mode", type=str, default="full_span", choices=["full_span", "local"], help="'full_span' (LiDAR to RS-LiDAR path) or 'local'")
    parser.add_argument("--slicing_margin", type=float, default=0.2, help="Extension on both ends in full_span mode: u in [-margin, 1 + margin]")
    parser.add_argument("--slicing_range", type=float, default=0.2, help="Sweep range alpha in [-range, +range] in local mode")
    parser.add_argument("--slicing_sigma", type=float, default=1.0, help="RS smoothing radius along the slice")
    parser.add_argument("--slicing_mc_samples", type=int, default=4, help="Monte Carlo samples along the slice")

    # Multi-GPU Sharding & Hardware
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index (0, 1, ...)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index")
    parser.add_argument("--device", type=str, default=None, help="Force a device (e.g. 'cpu' or 'cuda:0')")
    parser.add_argument("--output_dir", type=str, default="results/ablation1_sensitive_landscape", help="Output directory")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from the checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--merge_shards", action="store_true", help="Merge shard results")

    args = parser.parse_args()

    # Merge-only mode
    if args.merge_shards:
        merge_shard_checkpoints(args.output_dir, top_k=args.top_k)
        return

    # Device setup
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        actual_gpu = args.gpu_id if args.gpu_id < torch.cuda.device_count() else 0
        torch.cuda.set_device(actual_gpu)
        device = f"cuda:{actual_gpu}"
        print(f"Using GPU {actual_gpu}: {torch.cuda.get_device_name(actual_gpu)}")
    else:
        device = "cpu"
        print("No GPU found; running on CPU.")

    # 1. Load prompts (stratified per category)
    prompts_list = load_stratified_prompts_per_category(
        prompt_path=args.prompts_file,
        categories=args.categories,
        prompts_per_category=args.prompts_per_category,
        max_prompts=args.num_prompts,
        prompt_indices=args.prompt_indices,
        anchor_indices=getattr(args, "anchor_indices", "80,98"),
        random_sample=getattr(args, "random_sample", True),
        seed=args.seed,
        curated_selection=getattr(args, "curated_selection", False)
    )
    print(f"Prepared {len(prompts_list)} prompts.")

    # 2. Load Stable Diffusion and the VAE
    print(f"Loading pipeline {args.model_name}...")
    dtype = torch.float16 if "cuda" in device else torch.float32

    # Use FKDStableDiffusion for closed-form guidance when available, otherwise StableDiffusionPipeline
    pipe_cls = FKDStableDiffusion if FKDStableDiffusion is not None else StableDiffusionPipeline
    pipe = pipe_cls.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae

    # 3. Load ImageReward
    ir_model = None
    if rm_load is not None:
        try:
            print("Loading ImageReward...")
            ir_model = rm_load("ImageReward-v1.0")
            print("ImageReward loaded.")
        except Exception as e:
            print(f"Warning: could not load ImageReward: {e}")

    # 4. Run the experiment
    run_ablation1_experiment(pipe, vae, ir_model, prompts_list, args, device=device)

    # 5. Single-shard runs merge immediately
    if args.num_shards == 1:
        merge_shard_checkpoints(args.output_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
