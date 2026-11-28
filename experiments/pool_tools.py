"""Helpers for experiments that reuse cached Phase 1 lookahead pools (SD 1.5).

A pool is a folder written by `lookahead_sampling.py`: one `<idx:05d>/` folder per prompt with
`samples/latent.pt` (list of K latents [4, 64, 64]) and `results.json`, whose `<metric>.result` lists are
in the same order as the latents (sorted by the guidance reward, descending). These helpers were first
written as notebook cells (E2, F1, F8); they live here so that the G experiments can be tested.

Model-dependent helpers (VAE, reward models) import their libraries lazily, so this module can be
imported and tested on a CPU machine without downloading any model.
"""
import glob
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_METADATA = "prompt_files/geneval_metadata.jsonl"


def prompt_dir(pool_dir, idx):
    return os.path.join(pool_dir, f"{int(idx):05d}")


def list_prompt_indices(pool_dir):
    """Prompt indices that have both `results.json` and `samples/latent.pt` (or a link to it)."""
    if not pool_dir or not os.path.isdir(pool_dir):
        return []
    out = []
    for d in os.listdir(pool_dir):
        if d.isdigit():
            p = os.path.join(pool_dir, d)
            if os.path.exists(os.path.join(p, "results.json")) and os.path.exists(
                    os.path.join(p, "samples", "latent.pt")):
                out.append(int(d))
    return sorted(out)


def load_results(pool_dir, idx):
    with open(os.path.join(prompt_dir(pool_dir, idx), "results.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_pool(pool_dir, idx, key="ImageReward"):
    """Latents [K, 4, 64, 64] (dtype as saved) and rewards [K] (float64) of prompt `idx`, same order."""
    lat = torch.load(os.path.join(prompt_dir(pool_dir, idx), "samples", "latent.pt"), map_location="cpu")
    lat = torch.stack(list(lat)) if isinstance(lat, (list, tuple)) else lat
    res = load_results(pool_dir, idx)
    rewards = np.asarray(res[key]["result"], dtype=np.float64)
    if len(rewards) != lat.shape[0]:
        raise ValueError(f"prompt {idx}: {lat.shape[0]} latents but {len(rewards)} rewards in {pool_dir}")
    return lat, rewards


def match_candidates(lat_ref, lat_other, rel_tol=1e-3):
    """Permutation `perm` with lat_other[j] == lat_ref[perm[j]], or None if the two sets differ.

    Same test as the E2/F8 notebooks: every latent of `lat_other` must lie within
    rel_tol * (mean latent norm) of a latent of `lat_ref`, and the matching must be one-to-one.
    """
    a = lat_ref.flatten(1).float()
    b = lat_other.flatten(1).float()
    if a.shape != b.shape:
        return None
    dist = torch.cdist(b, a, compute_mode="donot_use_mm_for_euclid_dist")
    dmin, perm = dist.min(dim=1)
    if bool((dmin > rel_tol * a.norm(dim=1).mean()).any()):
        return None
    perm = perm.numpy()
    if len(set(perm.tolist())) != len(perm):
        return None
    return perm


def reorder_to_ref(values_other, perm):
    """Values given in `lat_other` order, returned in `lat_ref` order (see `match_candidates`)."""
    values_other = np.asarray(values_other)
    out = np.empty_like(values_other)
    out[perm] = values_other
    return out


def summarize_values(v):
    """The {result, mean, std, max, min} dict that `results.json` stores for each metric."""
    v = np.asarray(v, dtype=np.float64)
    return {"result": v.tolist(), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "max": float(v.max()), "min": float(v.min())}


def write_derived_pool(src_dir, dst_dir, idx, rewards, meta=None, link=True):
    """Write prompt `idx` of a derived pool: same latents as `src_dir`, new rewards under "ImageReward".

    `LiDAR_sampling.py` reads the guidance rewards from the "ImageReward" key, so a derived pool steers
    Phase 2 with `rewards` (given in the latent order of `src_dir`). The source metrics are kept under
    "source_results"; `meta` is stored under "derived". The latents are linked (or copied if links fail).
    """
    res = load_results(src_dir, idx)
    rewards = np.asarray(rewards, dtype=np.float64)
    if len(rewards) != len(res["prompt"]):
        raise ValueError(f"prompt {idx}: {len(rewards)} rewards for {len(res['prompt'])} candidates")
    d = prompt_dir(dst_dir, idx)
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    src_lat = os.path.abspath(os.path.join(prompt_dir(src_dir, idx), "samples", "latent.pt"))
    dst_lat = os.path.join(d, "samples", "latent.pt")
    if not os.path.lexists(dst_lat):
        linked = False
        if link:
            try:
                os.symlink(src_lat, dst_lat)
                linked = True
            except OSError:
                linked = False
        if not linked:
            shutil.copy2(src_lat, dst_lat)
    out = {"ImageReward": summarize_values(rewards), "prompt": res["prompt"], "prompt_index": int(idx),
           "source_results": {k: v for k, v in res.items() if isinstance(v, dict) and "result" in v}}
    if meta:
        out["derived"] = meta
    with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    return d


def load_metadata(path=DEFAULT_METADATA):
    """GenEval prompt metadata, one dict per prompt index."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_pil(x):
    """[N, 3, H, W] tensor in [-1, 1] to PIL images, with the rounding of VaeImageProcessor.postprocess."""
    from PIL import Image
    x = (x.detach().float().cpu() / 2 + 0.5).clamp(0, 1)
    arr = (x.permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
    return [Image.fromarray(a) for a in arr]


def pil_to_tensor(images, size=None):
    """PIL images to a [N, 3, H, W] float32 tensor in [-1, 1] (optionally resized to `size` x `size`)."""
    from PIL import Image
    out = []
    for im in images:
        im = im.convert("RGB")
        if size is not None and im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out.append(a * 2 - 1)
    return torch.stack(out)


def load_vae(model_name=DEFAULT_MODEL, device="cuda", dtype=torch.float16):
    """The SD VAE alone (as in the F1 notebook), without loading the UNet or text encoder."""
    from rslidar_utils import setup_runtime
    setup_runtime()  # compatibility shims (e.g. EncoderDecoderCache for peft); must run before diffusers is imported
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype).to(device).eval()
    return vae


@torch.inference_mode()
def decode(vae, lat, batch_size=10, clamp=False):
    """Latents [K, 4, 64, 64] to float32 images [K, 3, 512, 512] on the CPU.

    Phase 1 smooths the unclamped decoder output and clamps after adding noise
    (`lookahead_sampling.mc_rewards_loop`), so `clamp=False` reproduces its inputs.
    """
    outs = []
    for s in range(0, lat.shape[0], batch_size):
        z = lat[s:s + batch_size].to(vae.device, vae.dtype) / vae.config.scaling_factor
        x = vae.decode(z, return_dict=False)[0].float()
        outs.append((x.clamp(-1, 1) if clamp else x).cpu())
    return torch.cat(outs)


def make_reward_fn(metrics=("ImageReward",), chunk=64):
    """score(prompt, x[N,3,H,W] in [-1,1]) -> {metric: float64[N]}, through PIL and `fks_utils.do_eval`.

    This is the path Phase 1 uses to score clean and perturbed candidates.
    """
    from fks_utils import do_eval

    def score(prompt, x):
        out = {m: [] for m in metrics}
        for s in range(0, x.shape[0], chunk):
            pils = to_pil(x[s:s + chunk])
            ev = do_eval(prompt=[prompt] * len(pils), images=pils, metrics_to_compute=list(metrics))
            for m in metrics:
                out[m].extend(ev[m]["result"])
        return {m: np.asarray(v, dtype=np.float64) for m, v in out.items()}

    return score


def flip_pairs(raw_pool, rs_pool, indices, workers=8, log_every=0):
    """Per prompt: index of A (top under the raw reward) and B (top under the smoothed reward), raw order.

    Returns {idx: (a, b, raw_rewards, smoothed_rewards_in_raw_order)}; prompts whose pools do not share
    latents are skipped (reported under key "unmatched" of the second return value). Prompts are read
    by `workers` threads (file reads dominate on a mounted Drive); the result does not depend on it.
    """
    def one(idx):
        lat_a, r_raw = load_pool(raw_pool, idx)
        lat_b, r_rs = load_pool(rs_pool, idx)
        perm = match_candidates(lat_a, lat_b)
        if perm is None:
            return int(idx), None
        s = reorder_to_ref(r_rs, perm)
        return int(idx), (int(np.argmax(r_raw)), int(np.argmax(s)), r_raw, s)

    indices = list(indices)
    pairs, unmatched = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for n, (idx, v) in enumerate(ex.map(one, indices), 1):
            if v is None:
                unmatched.append(idx)
            else:
                pairs[idx] = v
            if log_every and (n % log_every == 0 or n == len(indices)):
                print(f"  pairing pools: {n}/{len(indices)} prompts ({time.time() - t0:.0f} s)", flush=True)
    return pairs, {"unmatched": unmatched}


def _copy_atomic(src, dst):
    """Copy `src` to `dst` through a temporary file, so that an existing `dst` is always complete.

    Returns the number of bytes copied (0 if `dst` already existed). Raises FileNotFoundError if `src` is missing.
    """
    if os.path.exists(dst):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.part{threading.get_ident()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return os.path.getsize(dst)


def stage_local(src_dir, dst_dir, rel_files=("results.json", "samples/latent.pt"), rel_globs=(), indices=None,
                workers=32, log_every=100):
    """Copy the per-prompt files of a pool or run folder to a local folder, many files at a time.

    On Colab, reading a mounted Google Drive file by file is slow (about a second per file), so the
    experiments read a local copy instead. `rel_files` are fixed paths inside each `<idx:05d>/` folder,
    `rel_globs` are patterns expanded there (e.g. "samples/*.png"). Files already copied are kept, so an
    interrupted copy resumes. Missing source files are reported, not raised. Returns `dst_dir`.
    """
    if indices is None:
        indices = sorted(int(d) for d in os.listdir(src_dir) if d.isdigit())
    indices = [int(i) for i in indices]

    def job(idx):
        s, d = prompt_dir(src_dir, idx), prompt_dir(dst_dir, idx)
        n_bytes, missing = 0, []
        for rel in rel_files:
            try:
                n_bytes += _copy_atomic(os.path.join(s, rel), os.path.join(d, rel))
            except FileNotFoundError:
                missing.append(f"{idx:05d}/{rel}")
        for pattern in rel_globs:
            for src in glob.glob(os.path.join(s, pattern)):
                n_bytes += _copy_atomic(src, os.path.join(d, os.path.relpath(src, s)))
        return n_bytes, missing

    t0, total, missing = time.time(), 0, []
    name = os.path.basename(os.path.normpath(src_dir))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for n, (b, m) in enumerate(ex.map(job, indices), 1):
            total += b
            missing += m
            if log_every and (n % log_every == 0 or n == len(indices)):
                print(f"  staging {name}: {n}/{len(indices)} prompts, {total / 2**20:.0f} MB copied "
                      f"({time.time() - t0:.0f} s)", flush=True)
    if missing:
        print(f"  ⚠️ {len(missing)} missing source files in {name}, e.g. {missing[:3]}", flush=True)
    return dst_dir
