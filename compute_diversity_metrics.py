"""
compute_diversity_metrics.py - In-Memory Multi-Folder Diversity Evaluation Tool

Computes Intra-Prompt Diversity Metrics across generated particle samples:
1. Intra-Prompt Pairwise LPIPS (Perceptual Manifold / Visual Layout)
2. Intra-Prompt Vendi Score (Effective Number of Modes via Von Neumann Entropy on CLIP Gram Matrix)
3. Intra-Prompt Pairwise CLIP Cosine Distance (Semantic Diversity)
4. ImageReward Quality (extracted from results.json) for Quality-Diversity Pareto Frontier Analysis

Features:
- Multithreaded In-Memory RAM Caching: Preloads entire image datasets into RAM to eliminate Drive I/O bottlenecks.
- Multi-Folder Comparison: Evaluates multiple runs side-by-side and formats publication-grade Markdown tables.
- Dual Interface: Standalone CLI and importable Python API for Jupyter/Colab notebooks.
"""

import os
import glob
import json
import time
import math
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

# Graceful optional imports
try:
    import lpips
except ImportError:
    lpips = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor, CLIPModel = None, None


# ==============================================================================
# 1. CORE MATHEMATICAL DIVERSITY SOLVERS
# ==============================================================================

def compute_vendi_score(embeddings: np.ndarray) -> float:
    """
    Computes the Von Neumann Entropy-based Vendi Score (Friedman & Dieng, NeurIPS 2022).
    
    Interpreted as the 'Effective Number of Modes' among N particles.
    For N identical particles, VS = 1.0.
    For N mutually orthogonal / independent particles, VS = N.
    
    Args:
        embeddings: np.ndarray of shape (N, D), assumed to be L2-normalized.
    Returns:
        float: Vendi Score in range [1.0, N].
    """
    N = embeddings.shape[0]
    if N <= 1:
        return 1.0
        
    # Ensure L2 normalization
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    
    # Normalized Gram matrix: K/N
    K = np.dot(normalized, normalized.T) / N
    
    # Eigenvalues of symmetric matrix (Hermitian / real symmetric)
    eigenvalues = np.linalg.eigvalsh(K)
    
    # Numerical safeguard: clamp small negative eigenvalues to 0
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    
    # Von Neumann entropy: H = - sum(lambda_k * ln(lambda_k))
    entropy = -float(np.sum(eigenvalues * np.log(eigenvalues)))
    
    # Vendi Score = exp(H)
    return float(np.exp(entropy))


def compute_pairwise_clip_distance(embeddings: np.ndarray) -> float:
    """
    Computes the average pairwise cosine distance (1 - cos(v_i, v_j)) among N particles.
    
    Args:
        embeddings: np.ndarray of shape (N, D), assumed to be L2-normalized.
    Returns:
        float: Mean pairwise distance in [0.0, 2.0].
    """
    N = embeddings.shape[0]
    if N < 2:
        return 0.0
        
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    
    cos_sim = np.dot(normed, normed.T)
    # Extract strictly upper triangular elements
    triu_indices = np.triu_indices(N, k=1)
    pairwise_distances = 1.0 - cos_sim[triu_indices]
    
    return float(np.mean(pairwise_distances))


def compute_pairwise_lpips_batch(
    images_tensor: torch.Tensor, 
    loss_fn: nn.Module
) -> float:
    """
    Computes average pairwise LPIPS distance across all C(N, 2) particle pairs in a single forward pass.
    
    Args:
        images_tensor: torch.Tensor of shape (N, 3, H, W) normalized to [-1.0, 1.0].
        loss_fn: LPIPS loss model on active device.
    Returns:
        float: Mean perceptual distance.
    """
    N = images_tensor.shape[0]
    if N < 2:
        return 0.0
        
    idx_i, idx_j = [], []
    for i in range(N):
        for j in range(i + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            
    t_i = images_tensor[idx_i] # (Pairs, 3, H, W)
    t_j = images_tensor[idx_j] # (Pairs, 3, H, W)
    
    with torch.inference_mode():
        d = loss_fn(t_i, t_j).view(-1).cpu().numpy()
        
    return float(np.mean(d))


# ==============================================================================
# 2. IN-MEMORY RAM DATASET & MULTITHREADED PRELOADER
# ==============================================================================

class InMemPromptItem:
    """Represents a single prompt's particles and reward metadata preloaded in RAM."""
    def __init__(
        self,
        prompt_idx: int,
        prompt_dir_name: str,
        pil_images: List[Image.Image],
        lpips_tensor: torch.Tensor,
        reward_max: Optional[float] = None,
        reward_mean: Optional[float] = None,
        prompt_text: str = ""
    ):
        self.prompt_idx = prompt_idx
        self.prompt_dir_name = prompt_dir_name
        self.pil_images = pil_images
        self.lpips_tensor = lpips_tensor
        self.reward_max = reward_max
        self.reward_mean = reward_mean
        self.prompt_text = prompt_text
        self.num_particles = len(pil_images)


class InMemRunDataset:
    """Container holding all preloaded prompt items for a single experimental run."""
    def __init__(self, run_name: str, run_path: str):
        self.run_name = run_name
        self.run_path = run_path
        self.items: List[InMemPromptItem] = []
        
    def __len__(self):
        return len(self.items)


def _load_single_prompt_folder(
    p_dir: str, 
    lpips_transform: T.Compose
) -> Optional[InMemPromptItem]:
    """Helper to load a single prompt directory from disk/Drive into RAM."""
    samples_dir = os.path.join(p_dir, "samples")
    if not os.path.exists(samples_dir):
        return None
        
    png_files = sorted([
        os.path.join(samples_dir, f) for f in os.listdir(samples_dir)
        if f.endswith(".png") and not f.startswith(".")
    ])
    if len(png_files) < 2:
        return None
        
    dir_basename = os.path.basename(p_dir)
    try:
        prompt_idx = int(dir_basename)
    except ValueError:
        prompt_idx = -1
        
    # Load and resize images to 256x256 into RAM
    pil_images = []
    tensor_list = []
    for f in png_files:
        try:
            with Image.open(f) as img:
                img_rgb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                pil_images.append(img_rgb)
                tensor_list.append(lpips_transform(img_rgb))
        except Exception:
            continue
            
    if len(pil_images) < 2:
        return None
        
    lpips_tensor = torch.stack(tensor_list, dim=0) # Shape: (N, 3, 256, 256)
    
    # Read rewards from results.json if present
    reward_max, reward_mean, prompt_text = None, None, ""
    results_path = os.path.join(p_dir, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as rf:
                res_data = json.load(rf)
                if "ImageReward" in res_data:
                    ir_info = res_data["ImageReward"]
                    reward_max = float(ir_info.get("max", np.nan))
                    reward_mean = float(ir_info.get("mean", np.nan))
                if "prompt" in res_data:
                    prompt_text = str(res_data["prompt"])
        except Exception:
            pass
            
    return InMemPromptItem(
        prompt_idx=prompt_idx,
        prompt_dir_name=dir_basename,
        pil_images=pil_images,
        lpips_tensor=lpips_tensor,
        reward_max=reward_max,
        reward_mean=reward_mean,
        prompt_text=prompt_text
    )


def preload_run_to_ram(
    run_dir: str, 
    max_prompts: int = 553,
    max_workers: int = 8,
    verbose: bool = True
) -> InMemRunDataset:
    """
    Multithreaded loader that sucks all prompt particle images from Drive/SSD into RAM.
    
    Args:
        run_dir: Path to Target_samples/<RUN_NAME>
        max_prompts: Maximum prompts to load (default: 553 for GenEval)
        max_workers: Parallel I/O threads (default: 8)
        verbose: Print progress log
    Returns:
        InMemRunDataset: Dataset fully resident in RAM.
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    dataset = InMemRunDataset(run_name=run_name, run_path=run_dir)
    
    # Find all prompt subdirectories (e.g. 00000, 00001, ...)
    candidate_dirs = [
        d for d in glob.glob(os.path.join(run_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ]
    candidate_dirs = sorted(candidate_dirs, key=lambda x: int(os.path.basename(x)))
    if max_prompts is not None and max_prompts > 0:
        candidate_dirs = candidate_dirs[:max_prompts]
        
    if len(candidate_dirs) == 0:
        if verbose:
            print(f"⚠️ [RAM Preloader]: Không tìm thấy thư mục prompt con nào trong {run_dir}!")
        return dataset
        
    if verbose:
        print(f"⚡ [RAM Preloader]: Đang nạp song song {len(candidate_dirs)} prompts từ {run_name} vào RAM...")
        
    lpips_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1.0, 1.0]
    ])
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_prompt_folder, d, lpips_transform) for d in candidate_dirs]
        for f in futures:
            item = f.result()
            if item is not None:
                dataset.items.append(item)
                
    dataset.items.sort(key=lambda x: x.prompt_idx)
    elapsed = time.time() - t0
    
    if verbose:
        n_loaded = len(dataset)
        sample_n = dataset.items[0].num_particles if n_loaded > 0 else 0
        mem_mb = (n_loaded * sample_n * 3 * 256 * 256 * 4) / (1024 * 1024) if n_loaded > 0 else 0
        print(f"  ✅ Đã nạp thành công {n_loaded} prompts (N={sample_n} hạt) vào RAM trong {elapsed:.2f}s (~{mem_mb:.1f} MB RAM).")
        
    return dataset


# ==============================================================================
# 3. GPU DIVERSITY EVALUATOR ENGINE
# ==============================================================================

class DiversityEvaluator:
    """Manages LPIPS and CLIP models on GPU to compute metrics across in-memory datasets."""
    
    def __init__(
        self, 
        device: Optional[str] = None,
        load_lpips: bool = True,
        load_clip: bool = True
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None
        
        # Load AlexNet LPIPS
        if load_lpips:
            if lpips is None:
                print("⚠️ Thư viện 'lpips' chưa được cài đặt. Vui lòng chạy: pip install lpips")
            else:
                print(f"📦 Đang nạp mô hình AlexNet LPIPS trên thiết bị {self.device}...")
                self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
                
        # Load CLIP ViT-B/32
        if load_clip:
            if CLIPModel is None or CLIPProcessor is None:
                print("⚠️ Thư viện 'transformers' chưa sẵn sàng cho CLIP.")
            else:
                print(f"📦 Đang nạp mô hình CLIP ViT-B/32 trên thiết bị {self.device}...")
                clip_id = "openai/clip-vit-base-patch32"
                self.clip_model = CLIPModel.from_pretrained(clip_id).to(self.device).eval()
                self.clip_processor = CLIPProcessor.from_pretrained(clip_id)

    def evaluate_dataset(
        self, 
        dataset: InMemRunDataset,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Runs batched GPU evaluation on the preloaded in-memory dataset.
        
        Returns dict containing aggregated metrics and per-prompt arrays.
        """
        if len(dataset) == 0:
            return {"error": "Dataset rỗng", "run_name": dataset.run_name}
            
        t0 = time.time()
        vendi_scores: List[float] = []
        clip_distances: List[float] = []
        lpips_scores: List[float] = []
        reward_max_list: List[float] = []
        reward_mean_list: List[float] = []
        
        sample_n = dataset.items[0].num_particles
        
        for item in dataset.items:
            # 1. Compute Pairwise LPIPS
            if self.lpips_model is not None:
                tensor_gpu = item.lpips_tensor.to(self.device)
                d_lpips = compute_pairwise_lpips_batch(tensor_gpu, self.lpips_model)
                lpips_scores.append(d_lpips)
                
            # 2. Compute CLIP features, Vendi Score and Pairwise CLIP Distance
            if self.clip_model is not None and self.clip_processor is not None:
                inputs = self.clip_processor(images=item.pil_images, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    feats = self.clip_model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    embs = feats.cpu().numpy()
                    
                vs = compute_vendi_score(embs)
                d_clip = compute_pairwise_clip_distance(embs)
                vendi_scores.append(vs)
                clip_distances.append(d_clip)
                
            # 3. Collect Rewards
            if item.reward_max is not None and not math.isnan(item.reward_max):
                reward_max_list.append(item.reward_max)
            if item.reward_mean is not None and not math.isnan(item.reward_mean):
                reward_mean_list.append(item.reward_mean)
                
        elapsed = time.time() - t0
        
        summary = {
            "run_name": dataset.run_name,
            "run_path": dataset.run_path,
            "num_particles": sample_n,
            "evaluated_prompts": len(dataset),
            "eval_time_sec": elapsed,
            "vendi_score_mean": float(np.mean(vendi_scores)) if vendi_scores else None,
            "vendi_score_std": float(np.std(vendi_scores)) if vendi_scores else None,
            "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
            "lpips_std": float(np.std(lpips_scores)) if lpips_scores else None,
            "clip_dist_mean": float(np.mean(clip_distances)) if clip_distances else None,
            "clip_dist_std": float(np.std(clip_distances)) if clip_distances else None,
            "image_reward_max": float(np.mean(reward_max_list)) if reward_max_list else None,
            "image_reward_mean": float(np.mean(reward_mean_list)) if reward_mean_list else None,
        }
        
        # Parse and enrich structured setting parameters
        summary.update(parse_run_settings(dataset.run_name, sample_n))
        
        if verbose:
            print(f"  🏁 Đánh giá xong {dataset.run_name} trong {elapsed:.2f}s:")
            if summary['image_reward_max'] is not None:
                print(f"     • ImageReward (Max):    {summary['image_reward_max']:.4f}")
            if summary['vendi_score_mean'] is not None:
                print(f"     • Vendi Score (Modes):  {summary['vendi_score_mean']:.3f} / {sample_n}.0 (±{summary['vendi_score_std']:.3f})")
            if summary['lpips_mean'] is not None:
                print(f"     • Pairwise LPIPS:       {summary['lpips_mean']:.4f} (±{summary['lpips_std']:.4f})")
            if summary['clip_dist_mean'] is not None:
                print(f"     • Pairwise CLIP Dist:   {summary['clip_dist_mean']:.4f}")
                
        return summary


# ==============================================================================
# 4. MARKDOWN & JSON COMPARISON REPORTERS
# ==============================================================================

def parse_run_settings(run_name: str, num_particles: Optional[int] = None) -> Dict[str, Any]:
    """
    Parses systematic run name into structured hyperparameter columns.
    Extracts: method, backbone, particles, sigma, M, anneal, guidance.
    """
    settings = {
        "method": "Custom",
        "backbone": "SD15",
        "particles": num_particles if num_particles is not None else 4,
        "sigma": "—",
        "M": "—",
        "anneal": "Không",
        "guidance": "None",
    }
    
    # 1. Method
    if "RSSMC" in run_name or "RS_SMC" in run_name:
        settings["method"] = "RS-SMC (Ours)"
    elif "SMC" in run_name:
        settings["method"] = "Vanilla SMC"
    elif "LiDAR" in run_name:
        settings["method"] = "LiDAR"
        
    # 2. Backbone
    if "SDXL" in run_name or "sdxl" in run_name:
        settings["backbone"] = "SDXL"
    elif "SD15" in run_name or "sd15" in run_name:
        settings["backbone"] = "SD15"
        
    # 3. Particles N
    match_n = re.search(r"[-_]N(\d+)", run_name)
    if match_n:
        settings["particles"] = int(match_n.group(1))
    elif num_particles is not None:
        settings["particles"] = num_particles
        
    # 4. Sigma
    match_sig = re.search(r"[-_]sig([0-9\.]+)", run_name)
    if match_sig:
        try:
            settings["sigma"] = float(match_sig.group(1))
        except ValueError:
            settings["sigma"] = match_sig.group(1)
            
    # 5. M (Monte Carlo samples)
    match_m = re.search(r"[-_]M(\d+)", run_name)
    if match_m:
        settings["M"] = int(match_m.group(1))
        
    # 6. Anneal
    if "anneal" in run_name.lower():
        settings["anneal"] = "Có"
    else:
        settings["anneal"] = "Không" if settings["method"] == "RS-SMC (Ours)" else "—"
        
    # 7. Guidance
    if "RSLiDAR" in run_name or "RS_LiDAR" in run_name:
        settings["guidance"] = "RS-LiDAR"
    elif "LiDAR" in run_name:
        settings["guidance"] = "LiDAR"
        
    return settings


def format_markdown_table(summaries: List[Dict[str, Any]]) -> str:
    """Formats multiple evaluation results into a clean GitHub Markdown comparison table."""
    lines = []
    lines.append("| STT | Phương Pháp | $N$ | $\\sigma$ | $M$ | Anneal | ImageReward (↑) | Vendi Score (↑) | Pairwise LPIPS (↑) | CLIP Dist (↑) | Prompts | Thư Mục |")
    lines.append("| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |")
    
    for idx, s in enumerate(summaries, 1):
        if "error" in s:
            lines.append(f"| {idx} | `{s['run_name']}` | — | — | — | — | ERROR: {s['error']} | — | — | — | 0 | `{s['run_name']}` |")
            continue
            
        r_name = s['run_name']
        if "method" not in s:
            s.update(parse_run_settings(r_name, s.get('num_particles')))
            
        method = s.get('method', 'Custom')
        n_part = s.get('particles', s.get('num_particles', '—'))
        sigma = s.get('sigma', '—')
        m_val = s.get('M', '—')
        anneal = s.get('anneal', '—')
        
        sigma_str = f"{sigma:.2f}" if isinstance(sigma, float) else str(sigma)
        
        ir_str = f"**`{s['image_reward_max']:.4f}`**" if s.get('image_reward_max') is not None else "—"
        vs_str = f"**`{s['vendi_score_mean']:.3f}`** / {n_part}" if s.get('vendi_score_mean') is not None else "—"
        lpips_str = f"`{s['lpips_mean']:.4f}`" if s.get('lpips_mean') is not None else "—"
        clip_str = f"`{s['clip_dist_mean']:.4f}`" if s.get('clip_dist_mean') is not None else "—"
        n_prompts = s.get('evaluated_prompts', 0)
        
        if len(r_name) > 38:
            short_name = r_name[:17] + "..." + r_name[-17:]
        else:
            short_name = r_name
            
        lines.append(
            f"| **{idx}** | **{method}** | {n_part} | {sigma_str} | {m_val} | {anneal} | "
            f"{ir_str} | {vs_str} | {lpips_str} | {clip_str} | {n_prompts} | `{short_name}` |"
        )
        
    return "\n".join(lines)


def save_and_merge_reports(
    new_summaries: List[Dict[str, Any]],
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    create_timestamped_copy: bool = True
) -> List[Dict[str, Any]]:
    """
    Saves and merges evaluation results with existing report files on Drive/disk.
    - Incremental Accumulation: Preserves all previous run settings and evaluations.
    - Deduplication: If the same run_name is evaluated again, updates it with the latest data.
    - Timestamped Snapshot: Automatically saves a timestamped backup in a 'history/' folder.
    
    Returns:
        List[Dict[str, Any]]: Complete accumulated list of all runs.
    """
    merged_dict = {}
    
    # 1. Load existing JSON if present
    if output_json and os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as jf:
                existing_data = json.load(jf)
                if isinstance(existing_data, list):
                    for item in existing_data:
                        if isinstance(item, dict) and "run_name" in item:
                            merged_dict[item["run_name"]] = item
            print(f"🔄 [Smart Accumulator]: Đã tải {len(merged_dict)} kết quả tích lũy từ file báo cáo cũ.")
        except Exception as e:
            print(f"⚠️ [Smart Accumulator]: Không thể nạp JSON cũ ({e}), sẽ tạo mới.")
            
    # 2. Merge new summaries (updating existing or appending new)
    for s in new_summaries:
        if "run_name" in s:
            if "method" not in s:
                s.update(parse_run_settings(s["run_name"], s.get("num_particles")))
            merged_dict[s["run_name"]] = s
            
    accumulated_list = list(merged_dict.values())
    
    # Sort accumulated runs cleanly (by method, particles, then run_name)
    accumulated_list.sort(key=lambda x: (
        str(x.get("method", "")), 
        int(x.get("particles", 0)) if str(x.get("particles", "")).isdigit() else 0, 
        str(x.get("run_name", ""))
    ))
    
    # 3. Save Master JSON
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as jf:
            json.dump(accumulated_list, jf, indent=4)
        print(f"💾 [Master JSON]: Đã lưu {len(accumulated_list)} kết quả tích lũy tại: {output_json}")
        
    # 4. Save Master CSV & Timestamped Snapshot
    if output_csv:
        import pandas as pd
        out_dir = os.path.dirname(os.path.abspath(output_csv))
        os.makedirs(out_dir, exist_ok=True)
        
        # Define clean column order
        priority_cols = [
            "run_name", "method", "backbone", "particles", "sigma", "M", "anneal", "guidance",
            "image_reward_max", "image_reward_mean",
            "vendi_score_mean", "vendi_score_std",
            "lpips_mean", "lpips_std",
            "clip_dist_mean", "clip_dist_std",
            "evaluated_prompts", "eval_time_sec", "run_path"
        ]
        
        df = pd.DataFrame(accumulated_list)
        ordered_cols = [c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]
        df = df[ordered_cols]
        df.to_csv(output_csv, index=False)
        print(f"💾 [Master CSV]: Đã lưu bảng tổng hợp tích lũy tại: {output_csv}")
        
        # 5. Save Timestamped Snapshot to history/
        if create_timestamped_copy:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            history_dir = os.path.join(out_dir, "history")
            os.makedirs(history_dir, exist_ok=True)
            ts_csv = os.path.join(history_dir, f"diversity_summary_{timestamp}.csv")
            df.to_csv(ts_csv, index=False)
            print(f"🕒 [Snapshot Backup]: Đã lưu bản sao lưu lịch sử tại: {ts_csv}")
            
    return accumulated_list


def auto_resolve_target_dir(dir_query: str, base_drive: Optional[str] = None) -> str:
    """Helper to auto-resolve relative names, wildcards, or Colab Drive paths."""
    if os.path.isabs(dir_query) and os.path.exists(dir_query):
        return dir_query
        
    candidates = [
        dir_query,
        os.path.join("Target_samples", dir_query),
        os.path.join("Diffusion-LiDAR-Sampling", "Target_samples", dir_query),
    ]
    
    if base_drive:
        candidates.extend([
            os.path.join(base_drive, "Target_samples", dir_query),
            os.path.join(base_drive, "RS-LiDAR", "Target_samples", dir_query),
        ])
        
    for cand in candidates:
        matched = glob.glob(cand)
        if matched and os.path.exists(matched[0]):
            return matched[0]
            
    return dir_query


def evaluate_multiple_folders(
    folder_paths: List[str],
    max_prompts: int = 553,
    max_workers: int = 8,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Main API: Preloads multiple folders into RAM and evaluates all diversity metrics on GPU.
    """
    evaluator = DiversityEvaluator(device=device, load_lpips=True, load_clip=True)
    summaries = []
    
    print("\n" + "=" * 78)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ ĐA DẠNG MẪU (MULTI-FOLDER IN-MEMORY DIVERSITY)")
    print(f"📁 Tổng số thư mục cần đánh giá: {len(folder_paths)}")
    print("=" * 78)
    
    for idx, f_path in enumerate(folder_paths, 1):
        resolved_path = auto_resolve_target_dir(f_path)
        print(f"\n📂 [{idx}/{len(folder_paths)}] Đang xử lý thư mục: {resolved_path}")
        
        # 1. Preload folder into RAM
        dataset = preload_run_to_ram(
            run_dir=resolved_path,
            max_prompts=max_prompts,
            max_workers=max_workers,
            verbose=verbose
        )
        
        # 2. Evaluate on GPU from RAM
        summary = evaluator.evaluate_dataset(dataset, verbose=verbose)
        summaries.append(summary)
        
    print("\n" + "=" * 78)
    print("📊 BẢNG TỔNG HỢP SO SÁNH CHẤT LƯỢNG (IMAGEREWARD) VS ĐỘ ĐA DẠNG (DIVERSITY):")
    print("=" * 78 + "\n")
    md_table = format_markdown_table(summaries)
    print(md_table + "\n")
    print("=" * 78)
    
    # Save & Merge outputs if specified (Smart Accumulator Mode)
    if output_json or output_csv:
        all_accumulated = save_and_merge_reports(
            new_summaries=summaries,
            output_json=output_json,
            output_csv=output_csv,
            create_timestamped_copy=True
        )
        return all_accumulated
    else:
        return summaries


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Folder In-Memory Diversity Evaluator for SMC / RS-SMC")
    parser.add_argument(
        "--dirs", 
        nargs="+", 
        required=True, 
        help="List of run directories or folder names in Target_samples/"
    )
    parser.add_argument(
        "--max_prompts", 
        type=int, 
        default=553, 
        help="Maximum prompts to evaluate (default: 553)"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=8, 
        help="Number of parallel I/O threads for RAM preloading"
    )
    parser.add_argument(
        "--base_drive", 
        type=str, 
        default=None, 
        help="Base Google Drive directory path if running on Colab"
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        default="diversity_results.json", 
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None, 
        help="Optional output path for CSV summary"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Force compute device ('cuda' or 'cpu')"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_multiple_folders(
        folder_paths=args.dirs,
        max_prompts=args.max_prompts,
        max_workers=args.max_workers,
        output_json=args.output_json,
        output_csv=args.output_csv,
        device=args.device,
        verbose=True
    )
