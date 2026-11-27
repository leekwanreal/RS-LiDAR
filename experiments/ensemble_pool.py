"""G3 / X4: a reward-ensemble pool for the LiDAR baseline.

Every lookahead candidate of the seed-100 LiDAR pool is scored by ImageReward, HPSv2.1 and CLIPScore on
the clean decoded image. Within each prompt every metric is standardized over the K candidates (z-score)
and the ensemble score is the mean of the z-scores. The result is written as a derived pool (same latents,
ensemble score under "ImageReward"), so that `LiDAR_sampling.py` runs LiDAR guided by the ensemble with
the Table 1 settings and no code change. The ImageReward of the pool is reused (it was computed on the
same clean images in Phase 1) unless --recompute_ir is given.

Note for the paper: with this baseline, CLIPScore and HPS are part of the guidance and are therefore no
longer held out; GenEval remains held out.

Usage (from the repository root, on a GPU):
  python -m experiments.ensemble_pool --raw_pool Lookahead_samples/Lookahead_SD15_DPM5_n50_seed100 \
      --dst_pool /content/derived_lookahead/G3_ensemble/Ensemble --resume
"""
import argparse
import json
import os

import numpy as np
import torch

DEFAULT_METRICS = "ImageReward,HumanPreference,Clip-Score"


def zscore_ensemble(metric_scores, eps=1e-8):
    """Mean over metrics of the within-prompt z-score of each metric. metric_scores: {name: float[K]}."""
    zs = []
    for v in metric_scores.values():
        v = np.asarray(v, dtype=np.float64)
        sd = v.std()
        zs.append((v - v.mean()) / (sd if sd > eps else 1.0))
    return np.mean(zs, axis=0)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw_pool", required=True, help="LiDAR pool (unsmoothed)")
    p.add_argument("--dst_pool", required=True, help="derived pool to write")
    p.add_argument("--metrics", default=DEFAULT_METRICS, help="comma-separated do_eval metric names")
    p.add_argument("--recompute_ir", action="store_true", help="rescore ImageReward instead of reusing the pool's")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--no_link", action="store_true", help="copy latent.pt instead of linking it")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from experiments import pool_tools as pt
    from rslidar_utils import read_smoothing_config

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    indices = pt.list_prompt_indices(args.raw_pool)
    if not indices:
        raise SystemExit(f"no prompts in {args.raw_pool}")
    if read_smoothing_config(pt.load_results(args.raw_pool, indices[0])):
        raise SystemExit(f"{args.raw_pool} holds smoothed rewards; the ensemble needs the raw LiDAR pool")
    if args.max_prompts:
        indices = indices[:args.max_prompts]
    os.makedirs(args.dst_pool, exist_ok=True)

    to_score = [m for m in metrics if args.recompute_ir or m != "ImageReward"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = pt.load_vae(args.model_name, device) if to_score else None
    reward = pt.make_reward_fn(tuple(to_score), chunk=args.chunk) if to_score else None
    for n, idx in enumerate(indices):
        if args.resume and os.path.exists(os.path.join(pt.prompt_dir(args.dst_pool, idx), "results.json")):
            continue
        lat, r_ir = pt.load_pool(args.raw_pool, idx)
        res = pt.load_results(args.raw_pool, idx)
        comp = {}
        if to_score:
            x = pt.decode(vae, lat, clamp=True)
            comp.update(reward(res["prompt"][0], x))
        if "ImageReward" in metrics and "ImageReward" not in comp:
            comp["ImageReward"] = r_ir
        comp = {m: np.asarray(comp[m], dtype=np.float64) for m in metrics}
        ens = zscore_ensemble(comp)
        meta = {"method": "mean of within-prompt z-scores", "metrics": metrics,
                "components": {m: v.tolist() for m, v in comp.items()},
                "imagereward_reused_from_pool": "ImageReward" in metrics and not args.recompute_ir}
        pt.write_derived_pool(args.raw_pool, args.dst_pool, idx, ens, meta=meta, link=not args.no_link)
        if n % 50 == 0:
            print(f"[{n + 1}/{len(indices)}] prompt {idx}: ensemble top = {int(np.argmax(ens))}, "
                  f"IR top = {int(np.argmax(r_ir))}")

    done = pt.list_prompt_indices(args.dst_pool)
    same = 0
    for idx in done:
        _, r_ir = pt.load_pool(args.raw_pool, idx)
        _, ens = pt.load_pool(args.dst_pool, idx)
        same += int(np.argmax(r_ir) == np.argmax(ens))
    summary = {"n_prompts": len(done), "frac_same_top_as_ImageReward": same / max(len(done), 1),
               "config": vars(args)}
    with open(os.path.join(args.dst_pool, "ensemble_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
