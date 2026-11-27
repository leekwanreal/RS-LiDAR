"""G2 / X3: how stable is the top candidate of the smoothed score with M = 4 perturbations?

Each of the K lookahead candidates of a prompt is scored with 64 independent Gaussian perturbations
(sigma = 1, i.i.d., clip to [-1, 1], as in Phase 1). The 64 draws are split into 16 disjoint groups of
M = 4, each group playing the role of one RS-LiDAR run. We report how often the top candidate of an M = 4
estimate equals the top candidate of the M = 64 estimate, how often it is among its top 3, the Kendall
tau between the two rankings, and the empirical gap between the two best candidates relative to the
standard error of an M = 4 estimate (the quantities that enter Proposition 3).

Prompts are processed in a fixed random order (seed --order_seed), so a run stopped early is still a
random subset of the 553 prompts. Scores are saved per prompt, so the run can be resumed.

Usage (from the repository root, on a GPU):
  python -m experiments.rank_stability --raw_pool Lookahead_samples/Lookahead_SD15_DPM5_n50_seed100 \
      --rs_pool Lookahead_samples/RSLookahead_SD15_DPM5_n50_sig1.0_M4_seed100 \
      --out_dir <drive>/extra_experiments/G2_rank_stability --resume
  python -m experiments.rank_stability ... --analyze      # summary only, from the saved scores
"""
import argparse
import json
import os

import numpy as np
import torch


def kendall_tau(x, y):
    """Kendall tau-a between two score vectors (O(K^2), fine for K = 50)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), k=1)
    return float((dx[iu] * dy[iu]).sum() / len(iu[0]))


def analyze_prompt(S, subset_size=4, b_index=None):
    """Statistics of one prompt from its score matrix S [K, n_draws]."""
    K, n = S.shape
    m_all = S.mean(1)
    order = np.argsort(-m_all, kind="stable")
    top, second = int(order[0]), int(order[1])
    top3 = set(order[:3].tolist())
    groups = n // subset_size
    tops, taus, swapped = [], [], []
    for g in range(groups):
        m = S[:, g * subset_size:(g + 1) * subset_size].mean(1)
        tops.append(int(np.argmax(m)))
        taus.append(kendall_tau(m, m_all))
        swapped.append(bool(m[second] > m[top]))
    tops = np.array(tops)
    diff = S[top] - S[second]
    se = float(diff.std(ddof=1) / np.sqrt(subset_size)) if n > 1 else float("nan")
    out = {
        "top_all": top,
        "agree_top1": float((tops == top).mean()),
        "in_top3": float(np.mean([t in top3 for t in tops])),
        "all_groups_agree": bool((tops == top).all()),
        "kendall_tau": float(np.mean(taus)),
        "p_swap_top2": float(np.mean(swapped)),
        "gap_top2": float(m_all[top] - m_all[second]),
        "se_gap_M": se,
        "gap_over_se": float((m_all[top] - m_all[second]) / se) if se and se > 0 else float("inf"),
        "per_draw_std_top": float(S[top].std(ddof=1)),
    }
    if b_index is not None:
        out["b_is_top_all"] = bool(int(b_index) == top)
        out["b_in_top3"] = bool(int(b_index) in top3)
    return out


def analyze(scores, subset_size=4, b_indices=None):
    """Aggregate `analyze_prompt` over prompts. scores: {idx: S[K, n_draws]}; b_indices: {idx: B}."""
    per = {}
    for idx, S in scores.items():
        b = None if b_indices is None else b_indices.get(idx)
        per[idx] = analyze_prompt(np.asarray(S, dtype=np.float64), subset_size, b)
    if not per:
        return {"n_prompts": 0}, per
    keys = ["agree_top1", "in_top3", "kendall_tau", "p_swap_top2", "gap_top2", "se_gap_M", "per_draw_std_top"]
    summ = {"n_prompts": len(per), "subset_size": subset_size}
    for k in keys:
        v = np.array([p[k] for p in per.values()], dtype=np.float64)
        summ[k] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))}
    summ["frac_prompts_all_groups_agree"] = float(np.mean([p["all_groups_agree"] for p in per.values()]))
    g = np.array([p["gap_over_se"] for p in per.values()], dtype=np.float64)
    summ["gap_over_se"] = {"median": float(np.median(g)), "frac_above_2": float((g > 2).mean())}
    if b_indices is not None:
        bs = [p for p in per.values() if "b_is_top_all" in p]
        if bs:
            summ["frac_B_is_top_all"] = float(np.mean([p["b_is_top_all"] for p in bs]))
            summ["frac_B_in_top3"] = float(np.mean([p["b_in_top3"] for p in bs]))
    return summ, per


def score_candidates(x, sigma, num_noise, gen, score_fn):
    """S [K, num_noise]: score of clip(x_k + sigma u_{k,j}) with i.i.d. u for every candidate and draw."""
    K = x.shape[0]
    S = np.empty((K, num_noise), dtype=np.float64)
    for k in range(K):
        u = torch.randn((num_noise,) + tuple(x.shape[1:]), generator=gen, dtype=torch.float32)
        S[k] = np.asarray(score_fn((x[k].unsqueeze(0) + sigma * u).clamp(-1, 1)), dtype=np.float64)
    return S


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw_pool", required=True)
    p.add_argument("--rs_pool", default=None, help="RS pool (same latents) to locate B, the RS-LiDAR choice")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--num_noise", type=int, default=64)
    p.add_argument("--subset_size", type=int, default=4)
    p.add_argument("--order_seed", type=int, default=2026)
    p.add_argument("--seed", type=int, default=2026, help="noise of prompt i uses Generator(seed + i)")
    p.add_argument("--max_prompts", type=int, default=553)
    p.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--analyze", action="store_true", help="only aggregate the saved scores")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from experiments import pool_tools as pt

    sdir = os.path.join(args.out_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    indices = pt.list_prompt_indices(args.raw_pool)
    order = [int(i) for i in np.random.default_rng(args.order_seed).permutation(indices)][:args.max_prompts]

    b_indices = None
    if args.rs_pool:
        pairs, info = pt.flip_pairs(args.raw_pool, args.rs_pool,
                                    [i for i in order if i in set(pt.list_prompt_indices(args.rs_pool))])
        b_indices = {i: v[1] for i, v in pairs.items()}
        print(f"B located on {len(b_indices)} prompts (unmatched: {len(info['unmatched'])})")

    if not args.analyze:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = pt.load_vae(args.model_name, device)
        reward = pt.make_reward_fn(("ImageReward",), chunk=args.chunk)
        for n, idx in enumerate(order):
            f = os.path.join(sdir, f"{idx:05d}.npy")
            if args.resume and os.path.exists(f):
                continue
            lat, _ = pt.load_pool(args.raw_pool, idx)
            x = pt.decode(vae, lat, clamp=False)
            prompt = pt.load_results(args.raw_pool, idx)["prompt"][0]
            gen = torch.Generator().manual_seed(args.seed + idx)
            S = score_candidates(x, args.sigma, args.num_noise, gen,
                                 lambda t, _p=prompt: reward(_p, t)["ImageReward"])
            np.save(f, S.astype(np.float32))
            if n % 10 == 0:
                print(f"[{n + 1}/{len(order)}] prompt {idx} done")

    scores = {}
    for idx in order:
        f = os.path.join(sdir, f"{idx:05d}.npy")
        if os.path.exists(f):
            scores[idx] = np.load(f)
    summ, per = analyze(scores, args.subset_size, b_indices)
    summ["config"] = vars(args)
    with open(os.path.join(args.out_dir, "rank_stability_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2)
    keys = ["agree_top1", "in_top3", "kendall_tau", "p_swap_top2", "gap_top2", "se_gap_M", "gap_over_se"]
    rows = ["prompt_index," + ",".join(keys) + ",b_is_top_all"]
    for idx, p in per.items():
        rows.append(f"{idx}," + ",".join(f"{p[k]:.5f}" for k in keys) + f",{p.get('b_is_top_all', '')}")
    with open(os.path.join(args.out_dir, "rank_stability_per_prompt.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
