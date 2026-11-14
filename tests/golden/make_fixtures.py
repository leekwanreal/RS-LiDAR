"""Record numerical reference outputs of the core algorithms.

Run once from the repository root *before* a refactor:

    python tests/golden/make_fixtures.py

`tests/test_golden_equivalence.py` recomputes the same quantities with the current code
and checks that they match these references, so refactors cannot silently change results.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fkd_diffusers"))
os.chdir(ROOT)

from golden_inputs import (  # noqa: E402
    flux_inputs,
    pixel_images,
    reference_smoothing_loop,
    rewards_mc,
    sd_guide_inputs,
    smc_reward_fn,
    toy_eval_fn,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.pt")


def main():
    from diffusers import DDIMScheduler, DDPMScheduler

    import LiDAR_sampling
    import lookahead_sampling
    from fkd_class import FKD
    from fkd_pipeline_sd import FKDStableDiffusion
    from fkd_pipeline_sdxl import FKDStableDiffusionXL
    from multi_reward_evaluator import compute_multi_reward_pareto_potential
    from rs_flow_lidar import RSFlowLiDARGuidance

    fx = {}

    # Phase 1: vectorized smoothing batch and reward aggregation.
    images = pixel_images()
    for mode in ["independent_iid", "independent_antithetic", "coupled_crn"]:
        torch.manual_seed(1)
        fx[f"smooth_batch/{mode}"] = lookahead_sampling.generate_smoothed_pixel_batch(
            images, sigma=0.5, M=4, noise_mode=mode)
        for beta in [0.0, 0.5]:
            fx[f"entropic/{mode}/beta{beta}"] = lookahead_sampling.compute_entropic_antithetic_rewards(
                rewards_mc(), sigma=0.5, beta_risk=beta, noise_mode=mode)["r_corrected"]
        fx[f"smooth_loop/{mode}"] = reference_smoothing_loop(images, sigma=0.5, M=4, noise_mode=mode,
                                                             eval_fn=toy_eval_fn)

    # Phase 2: closed-form LiDAR guidance (SD and SDXL).
    sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          num_train_timesteps=1000)
    for name, cls, ch, hw in [("sd", FKDStableDiffusion, 4, 8), ("sdxl", FKDStableDiffusionXL, 4, 16)]:
        pipe = cls.__new__(cls)
        object.__setattr__(pipe, "_internal_dict", {})
        object.__setattr__(pipe, "scheduler", sched)
        latents, lookahead, rewards = sd_guide_inputs(ch, hw)
        subset = {"latents": lookahead.unsqueeze(0), "rewards": rewards.unsqueeze(0),
                  "rewards_high": rewards.unsqueeze(0), "rewards_low": rewards.unsqueeze(0)}
        for t in [999, 600, 300]:
            fx[f"guide/{name}/t{t}"] = pipe.get_sample_guide(subset, torch.tensor(t), latents.clone(), 5000.0).float()

    # RS-SMC: sigma schedules and a full resampling trajectory.
    ddpm = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                         num_train_timesteps=1000)
    ddpm.set_timesteps(100)
    configs = {
        "vanilla": dict(use_smc_smoothing=False),
        "static": dict(use_smc_smoothing=True, smc_sigma=0.25, smc_num_mc_samples=4),
        "static_crn": dict(use_smc_smoothing=True, smc_sigma=0.25, smc_num_mc_samples=4, smc_use_crn=True),
        "static_vec": dict(use_smc_smoothing=True, smc_sigma=0.25, smc_num_mc_samples=4,
                           smc_smoothing_mode="vectorized"),
    }
    for sch in ["tweedie_analytical", "cosine", "linear", "sqrt_noise"]:
        configs[f"anneal_{sch}"] = dict(use_smc_smoothing=True, smc_sigma=0.25, smc_num_mc_samples=4,
                                        smc_anneal_sigma=True, smc_anneal_schedule=sch, smc_sigma_min=0.01)
    for cname, extra in configs.items():
        fkd = FKD(potential_type="diff", lmbda=10.0, num_particles=4, adaptive_resampling=False,
                  resample_frequency=20, resampling_t_start=20, resampling_t_end=80, time_steps=100,
                  reward_fn=smc_reward_fn, latent_to_decode_fn=torch.tanh, device=torch.device("cpu"),
                  scheduler=ddpm, **extra)
        fx[f"smc_sigma/{cname}"] = torch.tensor([fkd.get_effective_smc_sigma(i) for i in range(100)])
        torch.manual_seed(7)
        latents = torch.randn(4, 3, 8, 8)
        for i in range(100):
            x0 = latents * 0.9
            latents, _ = fkd.resample(sampling_idx=i, latents=latents, x0_preds=x0)
        fx[f"smc_final/{cname}"] = latents.clone()
        fx[f"smc_rs/{cname}"] = fkd.population_rs.clone()

    # FLUX velocity guidance.
    guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0)
    z_t, z0, rew = flux_inputs()
    for t in [1.0, 0.988, 0.97, 0.9]:
        fx[f"flux/t{t}"] = guider.compute_velocity_guidance(z_t, z0, rew, sigma_t=t, t_start=1.0, t_end=0.95)

    # Pareto potential (kept feature).
    torch.manual_seed(3)
    multi = {"ImageReward": torch.randn(4, 6), "Clip-Score": torch.randn(4, 6) * 0.1}
    pot, _ = compute_multi_reward_pareto_potential(multi, sigma=0.5, pareto_rho=0.5, beta_risk=0.0,
                                                    noise_mode="independent_iid")
    fx["pareto/iid"] = pot

    # Prompt helpers.
    fx["prompts/stratified_100"] = torch.tensor(LiDAR_sampling.parse_prompt_indices("stratified_100"))
    fx["prompts/0-4"] = torch.tensor(LiDAR_sampling.parse_prompt_indices("0-4"))
    fx["prompts/list"] = torch.tensor(LiDAR_sampling.parse_prompt_indices("1,5,9"))
    meta = LiDAR_sampling.load_geneval_metadata("prompt_files/geneval_metadata.jsonl", max_prompts=10,
                                                prompt_indices=[3, 1])
    fx["prompts/meta_idx"] = torch.tensor([m["original_idx"] for m in meta])
    fx["prompts/meta_text"] = [m["prompt"] for m in meta]

    torch.save(fx, OUT)
    print(f"Saved {len(fx)} reference entries to {OUT}")


if __name__ == "__main__":
    main()
