"""G1 / X2: how close the final images are to the lookahead candidate that guidance followed.

LiDAR steers toward A (top under the raw reward), RS-LiDAR toward B (top under the smoothed reward).
For every prompt with A != B, this script measures the distance between each final image of the two
seed-100 runs and the two candidates, in pixel L2 over [-1, 1] and in LPIPS (AlexNet, 256 px as in
`compute_diversity_metrics.py`), with a random other candidate as reference. It also reports the norm
of the smoothing perturbation, ||clip(x + sigma u) - x||, so that the copy error can be compared with the
smoothing radius.

Usage (from the repository root):
  python -m experiments.copy_distance --raw_pool ... --rs_pool ... \
      --lidar_run Target_samples/E1_LiDAR_SD15_n50_DDPM100_lmbda5000_seed100 \
      --rs_run Target_samples/E1_RSLiDAR_SD15_n50_sig1.0_M4_DDPM100_lmbda5000_seed100 \
      --out_dir <drive>/extra_experiments/G1_mechanism/copy_distance
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def l2_dist(x, y):
    """Per-pair pixel L2 distance between [N,3,H,W] and [N or 1,3,H,W] tensors in [-1, 1]."""
    return (x - y).flatten(1).norm(dim=1)


def load_final_images(run_dir, idx, size=None):
    from experiments import pool_tools as pt
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(run_dir, f"{int(idx):05d}", "samples", "*.png")))
    if not paths:
        return None
    return pt.pil_to_tensor([Image.open(p) for p in paths], size=size)


def distances(finals, cands, lpips_fn=None, lpips_size=256):
    """Mean over final images of the distance to each candidate: {name: {"l2": float, "lpips": float}}."""
    import torch.nn.functional as F
    out = {}
    for name, c in cands.items():
        c1 = c.unsqueeze(0)
        rec = {"l2": float(l2_dist(finals, c1).mean())}
        if lpips_fn is not None:
            dev = next(lpips_fn.parameters()).device
            f_s = F.interpolate(finals, size=(lpips_size, lpips_size), mode="bilinear", align_corners=False)
            c_s = F.interpolate(c1, size=(lpips_size, lpips_size), mode="bilinear", align_corners=False)
            with torch.inference_mode():
                d = lpips_fn(f_s.to(dev), c_s.expand_as(f_s).to(dev)).flatten()
            rec["lpips"] = float(d.mean())
        out[name] = rec
    return out


def summarize(records, sigma):
    keys = ["LiDAR_to_A", "LiDAR_to_B", "LiDAR_to_rand", "RS_to_B", "RS_to_A", "RS_to_rand"]
    s = {"n_prompts": len(records), "sigma": sigma}
    if not records:
        return s
    for metric in ("l2", "lpips"):
        if metric not in records[0]["dist"]["LiDAR_to_A"]:
            continue
        s[metric] = {}
        for k in keys:
            v = np.array([r["dist"][k][metric] for r in records])
            s[metric][k] = {"mean": float(v.mean()), "median": float(np.median(v))}
        lid_a = np.array([r["dist"]["LiDAR_to_A"][metric] for r in records])
        lid_b = np.array([r["dist"]["LiDAR_to_B"][metric] for r in records])
        rs_a = np.array([r["dist"]["RS_to_A"][metric] for r in records])
        rs_b = np.array([r["dist"]["RS_to_B"][metric] for r in records])
        s[metric]["frac_LiDAR_closer_to_A"] = float((lid_a < lid_b).mean())
        s[metric]["frac_RS_closer_to_B"] = float((rs_b < rs_a).mean())
    pn = np.array([r["perturbation_norm"] for r in records])
    s["perturbation_norm"] = {"mean": float(pn.mean()), "median": float(np.median(pn))}
    s["l2_copy_over_perturbation"] = {
        "LiDAR_to_A": s["l2"]["LiDAR_to_A"]["mean"] / s["perturbation_norm"]["mean"],
        "RS_to_B": s["l2"]["RS_to_B"]["mean"] / s["perturbation_norm"]["mean"]}
    return s


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw_pool", required=True)
    p.add_argument("--rs_pool", required=True)
    p.add_argument("--lidar_run", required=True, help="Phase 2 run of LiDAR on raw_pool (seed 100)")
    p.add_argument("--rs_run", required=True, help="Phase 2 run of RS-LiDAR on rs_pool (seed 100)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sigma", type=float, default=1.0, help="smoothing radius for the perturbation norm")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from experiments import pool_tools as pt

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = sorted(set(pt.list_prompt_indices(args.raw_pool)) & set(pt.list_prompt_indices(args.rs_pool)))
    pairs, info = pt.flip_pairs(args.raw_pool, args.rs_pool, indices)
    flips = [i for i, (a, b, _, _) in pairs.items() if a != b]
    if args.max_prompts:
        flips = flips[:args.max_prompts]
    print(f"A != B on {len(flips)} prompts (unmatched pools: {len(info['unmatched'])})")

    vae = pt.load_vae(args.model_name, device)
    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    records, missing = [], []
    for idx in flips:
        a, b, _, _ = pairs[idx]
        f_lid = load_final_images(args.lidar_run, idx)
        f_rs = load_final_images(args.rs_run, idx)
        if f_lid is None or f_rs is None:
            missing.append(idx)
            continue
        lat, _ = pt.load_pool(args.raw_pool, idx)
        rng = np.random.default_rng([args.seed, idx])
        others = [k for k in range(lat.shape[0]) if k not in (a, b)]
        c = int(rng.choice(others))
        x = pt.decode(vae, lat[[a, b, c]], clamp=True)
        h = f_lid.shape[-1]
        if x.shape[-1] != h:
            x = torch.nn.functional.interpolate(x, size=(h, h), mode="bilinear", align_corners=False)
        d_lid = distances(f_lid, {"A": x[0], "B": x[1], "rand": x[2]}, lpips_fn)
        d_rs = distances(f_rs, {"A": x[0], "B": x[1], "rand": x[2]}, lpips_fn)
        gen = torch.Generator().manual_seed(args.seed + idx)
        u = torch.randn(x[0].shape, generator=gen)
        pnorm = float(((x[0] + args.sigma * u).clamp(-1, 1) - x[0]).norm())
        records.append({"prompt_index": idx, "a": a, "b": b, "rand": c, "perturbation_norm": pnorm,
                        "dist": {"LiDAR_to_A": d_lid["A"], "LiDAR_to_B": d_lid["B"], "LiDAR_to_rand": d_lid["rand"],
                                 "RS_to_A": d_rs["A"], "RS_to_B": d_rs["B"], "RS_to_rand": d_rs["rand"]}})

    summary = summarize(records, args.sigma)
    summary.update({"config": vars(args), "missing_final_images": missing, "unmatched": info["unmatched"],
                    "D": int(3 * 512 * 512), "sigma_sqrt_D": float(args.sigma * np.sqrt(3 * 512 * 512))})
    with open(os.path.join(args.out_dir, "copy_distance_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    cols = ["LiDAR_to_A", "LiDAR_to_B", "LiDAR_to_rand", "RS_to_B", "RS_to_A", "RS_to_rand"]
    metrics = ["l2"] + (["lpips"] if lpips_fn is not None else [])
    head = ["prompt_index", "perturbation_norm"] + [f"{c}_{m}" for c in cols for m in metrics]
    rows = [",".join(head)]
    for r in records:
        vals = [str(r["prompt_index"]), f"{r['perturbation_norm']:.3f}"]
        vals += [f"{r['dist'][c][m]:.5f}" for c in cols for m in metrics]
        rows.append(",".join(vals))
    with open(os.path.join(args.out_dir, "copy_distance_per_prompt.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
