"""G1 / X1: reward of the two top candidates under perturbations of growing size.

For every prompt on which smoothing changes the top lookahead candidate (A = top under the raw reward R,
B = top under the smoothed reward), this script measures the mean of R(clip(x + eps * u)) for
eps in a grid, with the same Gaussian draws u for A and B (paired), plus R after a Gaussian blur.
If A's advantage over B rests on a reward component confined to a small neighborhood of A, the
advantage disappears already at eps much smaller than the smoothing radius sigma = 1.

Pre-registered reading (fixed before the run, see the G1 notebook):
  "spike-like"  : mean gap R(A) - R(B) at eps = 0.1 is at most 50% of the gap at eps = 0;
  "gradual"     : otherwise (the gap shrinks slowly and changes sign only near eps ~ sigma).

Usage (from the repository root, on a GPU):
  python -m experiments.perturbation_profile \
      --raw_pool Lookahead_samples/Lookahead_SD15_DPM5_n50_seed100 \
      --rs_pool Lookahead_samples/RSLookahead_SD15_DPM5_n50_sig1.0_M4_seed100 \
      --out_dir <drive>/extra_experiments/G1_mechanism/profile
"""
import argparse
import json
import math
import os

import numpy as np
import torch

DEFAULT_EPS = "0,0.03,0.1,0.25,0.5,1.0"
DEFAULT_BLUR = "0.5,1,2,4"
SPIKE_EPS = 0.1
SPIKE_FRACTION = 0.5


def perturb(x, eps, u):
    """clip(x + eps * u) for x [C,H,W] and u [n,C,H,W]; the projection used by the smoothed score."""
    return (x.unsqueeze(0) + eps * u).clamp(-1.0, 1.0)


def profile_pair(x_a, x_b, eps_list, num_noise, gen, score_fn):
    """{eps: (R_a [n_eps], R_b [n_eps])} with the same draws u for A and B and for every eps.

    eps = 0 is scored once on the clipped image. `score_fn(x[N,3,H,W]) -> float64[N]`.
    """
    u = torch.randn((num_noise,) + tuple(x_a.shape), generator=gen, dtype=torch.float32)
    out = {}
    for eps in eps_list:
        if eps == 0:
            xa, xb = x_a.clamp(-1, 1).unsqueeze(0), x_b.clamp(-1, 1).unsqueeze(0)
        else:
            xa, xb = perturb(x_a, eps, u), perturb(x_b, eps, u)
        r = np.asarray(score_fn(torch.cat([xa, xb])), dtype=np.float64)
        n = xa.shape[0]
        out[float(eps)] = (r[:n], r[n:])
    return out


def gaussian_blur(x, sigma_px):
    """Gaussian blur of [N,3,H,W] images with standard deviation `sigma_px` pixels."""
    import torchvision.transforms.functional as TF
    k = 2 * int(math.ceil(3 * sigma_px)) + 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma_px, sigma_px])


def blur_scores(x_a, x_b, radii_px, score_fn):
    """{radius: (R(blur(A)), R(blur(B)))}."""
    out = {}
    xs = torch.stack([x_a, x_b]).clamp(-1, 1)
    for r in radii_px:
        s = np.asarray(score_fn(gaussian_blur(xs, float(r)).clamp(-1, 1)), dtype=np.float64)
        out[float(r)] = (float(s[0]), float(s[1]))
    return out


def summarize(records, eps_list, radii):
    """Aggregate per-prompt records into the summary used for the paper (means over prompts)."""
    summary = {"n_prompts": len(records), "eps": {}, "blur": {}}
    if not records:
        return summary
    gap0 = None
    for eps in eps_list:
        ra = np.array([np.mean(r["profile"][str(float(eps))]["A"]) for r in records])
        rb = np.array([np.mean(r["profile"][str(float(eps))]["B"]) for r in records])
        gap = ra - rb
        if eps == 0:
            gap0 = float(gap.mean())
        summary["eps"][str(float(eps))] = {
            "mean_R_A": float(ra.mean()), "mean_R_B": float(rb.mean()), "mean_gap": float(gap.mean()),
            "median_gap": float(np.median(gap)), "frac_A_above_B": float((gap > 0).mean()),
        }
    for r in radii:
        ga = np.array([rec["blur"][str(float(r))]["A"] for rec in records])
        gb = np.array([rec["blur"][str(float(r))]["B"] for rec in records])
        summary["blur"][str(float(r))] = {"mean_R_A": float(ga.mean()), "mean_R_B": float(gb.mean()),
                                          "mean_gap": float((ga - gb).mean()),
                                          "frac_A_above_B": float((ga > gb).mean())}
    if gap0 is not None and str(float(SPIKE_EPS)) in summary["eps"]:
        g = summary["eps"][str(float(SPIKE_EPS))]["mean_gap"]
        summary["gap_ratio_at_spike_eps"] = g / gap0 if gap0 else None
        summary["reading"] = ("spike-like" if gap0 > 0 and g <= SPIKE_FRACTION * gap0 else "gradual")
        summary["reading_rule"] = (f"spike-like if mean gap at eps={SPIKE_EPS} <= {SPIKE_FRACTION} x gap at eps=0 "
                                   "(pre-registered)")
    stored = np.array([rec["stored_R_A"] for rec in records])
    recomputed = np.array([np.mean(rec["profile"]["0.0"]["A"]) for rec in records]) if "0.0" in records[0][
        "profile"] else None
    if recomputed is not None:
        summary["sanity_median_abs_diff_R_A_eps0_vs_pool"] = float(np.median(np.abs(stored - recomputed)))
    return summary


def parse_floats(s):
    return [float(v) for v in str(s).split(",") if v.strip() != ""]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw_pool", required=True, help="LiDAR pool (unsmoothed ImageReward)")
    p.add_argument("--rs_pool", required=True, help="RS pool on the same latents (smoothed ImageReward)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--eps", default=DEFAULT_EPS, help="comma-separated perturbation sizes")
    p.add_argument("--num_noise", type=int, default=16, help="Gaussian draws per eps (shared by A and B)")
    p.add_argument("--blur_px", default=DEFAULT_BLUR, help="comma-separated Gaussian blur radii (pixels)")
    p.add_argument("--seed", type=int, default=2026, help="noise of prompt i uses Generator(seed + i)")
    p.add_argument("--include_no_flip", action="store_true", help="also profile prompts with A = B (skipped)")
    p.add_argument("--max_prompts", type=int, default=None, help="limit for smoke tests")
    p.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--metadata", default="prompt_files/geneval_metadata.jsonl")
    p.add_argument("--chunk", type=int, default=34, help="images per reward call")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from experiments import pool_tools as pt

    eps_list, radii = parse_floats(args.eps), parse_floats(args.blur_px)
    per_dir = os.path.join(args.out_dir, "per_prompt")
    os.makedirs(per_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = pt.load_metadata(args.metadata)

    indices = sorted(set(pt.list_prompt_indices(args.raw_pool)) & set(pt.list_prompt_indices(args.rs_pool)))
    pairs, info = pt.flip_pairs(args.raw_pool, args.rs_pool, indices, log_every=100)
    flips = [i for i, (a, b, _, _) in pairs.items() if a != b]
    todo = sorted(pairs) if args.include_no_flip else flips
    if args.max_prompts:
        todo = todo[:args.max_prompts]
    print(f"prompts: {len(indices)} | paired: {len(pairs)} | unmatched: {len(info['unmatched'])} | "
          f"A != B: {len(flips)} | to profile: {len(todo)}", flush=True)

    vae = pt.load_vae(args.model_name, device)
    reward = pt.make_reward_fn(("ImageReward",), chunk=args.chunk)
    for n, idx in enumerate(todo):
        f = os.path.join(per_dir, f"{idx:05d}.json")
        if args.resume and os.path.exists(f):
            continue
        a, b, r_raw, s = pairs[idx]
        lat, _ = pt.load_pool(args.raw_pool, idx)
        x = pt.decode(vae, lat[[a, b]], clamp=False)
        prompt = pt.load_results(args.raw_pool, idx)["prompt"][0]

        def score_fn(t, _p=prompt):
            return reward(_p, t)["ImageReward"]

        gen = torch.Generator().manual_seed(args.seed + idx)
        prof = profile_pair(x[0], x[1], eps_list, args.num_noise, gen, score_fn)
        blur = blur_scores(x[0], x[1], radii, score_fn)
        rec = {"prompt_index": idx, "tag": meta[idx].get("tag") if idx < len(meta) else None,
               "a": a, "b": b, "flip": a != b,
               "stored_R_A": float(r_raw[a]), "stored_R_B": float(r_raw[b]),
               "stored_Rsigma_A": float(s[a]), "stored_Rsigma_B": float(s[b]),
               "profile": {str(k): {"A": v[0].tolist(), "B": v[1].tolist()} for k, v in prof.items()},
               "blur": {str(k): {"A": v[0], "B": v[1]} for k, v in blur.items()}}
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        if n % 25 == 0:
            g0 = rec["profile"]["0.0"] if "0.0" in rec["profile"] else None
            print(f"[{n + 1}/{len(todo)}] prompt {idx}: stored R(A)={rec['stored_R_A']:.3f}"
                  + (f", recomputed {np.mean(g0['A']):.3f}" if g0 else ""), flush=True)

    records = []
    for idx in todo:
        f = os.path.join(per_dir, f"{idx:05d}.json")
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fh:
                records.append(json.load(fh))
    summary = summarize([r for r in records if r["flip"]], eps_list, radii)
    summary.update({"config": vars(args), "n_paired": len(pairs), "n_flip": len(flips),
                    "unmatched": info["unmatched"]})
    with open(os.path.join(args.out_dir, "profile_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    rows = ["prompt_index,tag,eps,mean_R_A,mean_R_B,gap"]
    for r in records:
        for k, v in r["profile"].items():
            ra, rb = float(np.mean(v["A"])), float(np.mean(v["B"]))
            rows.append(f"{r['prompt_index']},{r['tag']},{k},{ra:.5f},{rb:.5f},{ra - rb:.5f}")
    with open(os.path.join(args.out_dir, "profile_per_prompt.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    rows = ["prompt_index,tag,blur_px,R_A,R_B,gap"]
    for r in records:
        for k, v in r["blur"].items():
            rows.append(f"{r['prompt_index']},{r['tag']},{k},{v['A']:.5f},{v['B']:.5f},{v['A'] - v['B']:.5f}")
    with open(os.path.join(args.out_dir, "blur_per_prompt.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
