"""Golden equivalence: the refactored code must reproduce the pre-refactor reference outputs.

References were recorded by `tests/golden/make_fixtures.py` on the code before the ICML clean-up.
Everything must match exactly, except the closed-form guidance, where the removed
`(1 - rho^2) * r + rho^2 * r` interpolation is replaced by `r` (float32 rounding, <= 2.4e-7 per reward).
"""
import os
import sys
import unittest

import torch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
sys.path.insert(0, GOLDEN_DIR)

from golden_inputs import (  # noqa: E402
    flux_inputs,
    pixel_images,
    rewards_mc,
    sd_guide_inputs,
    smc_reward_fn,
    toy_eval_fn,
)

FX = torch.load(os.path.join(GOLDEN_DIR, "fixtures.pt"), weights_only=False)
MODES = ["independent_iid", "independent_antithetic", "coupled_crn"]


def _make_guidance_pipe(cls, scheduler):
    pipe = cls.__new__(cls)
    object.__setattr__(pipe, "_internal_dict", {})
    object.__setattr__(pipe, "scheduler", scheduler)
    return pipe


class TestGoldenEquivalence(unittest.TestCase):
    def test_phase1_smoothing_batch_and_aggregation(self):
        import lookahead_sampling
        images = pixel_images()
        for mode in MODES:
            torch.manual_seed(1)
            out = lookahead_sampling.generate_smoothed_pixel_batch(images, sigma=0.5, M=4, noise_mode=mode)
            self.assertTrue(torch.equal(out, FX[f"smooth_batch/{mode}"]), mode)
            for beta in [0.0, 0.5]:
                r = lookahead_sampling.compute_entropic_antithetic_rewards(
                    rewards_mc(), sigma=0.5, beta_risk=beta, noise_mode=mode)["r_corrected"]
                self.assertTrue(torch.equal(r, FX[f"entropic/{mode}/beta{beta}"]), (mode, beta))

    def test_phase1_smoothing_loop(self):
        import lookahead_sampling
        images = pixel_images()
        for mode in MODES:
            torch.manual_seed(11)
            out = lookahead_sampling.mc_rewards_loop(
                images, sigma=0.5, num_mc_samples=4, noise_mode=mode,
                score_fn=lambda x: {"r": toy_eval_fn(x)})["r"]
            self.assertTrue(torch.equal(out, FX[f"smooth_loop/{mode}"]), mode)

    def test_phase2_closed_form_guidance(self):
        from diffusers import DDIMScheduler
        from fkd_pipeline_sd import FKDStableDiffusion
        from fkd_pipeline_sdxl import FKDStableDiffusionXL
        sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              num_train_timesteps=1000)
        for name, cls, ch, hw in [("sd", FKDStableDiffusion, 4, 8), ("sdxl", FKDStableDiffusionXL, 4, 16)]:
            pipe = _make_guidance_pipe(cls, sched)
            latents, lookahead, rewards = sd_guide_inputs(ch, hw)
            subset = {"latents": lookahead.unsqueeze(0), "rewards": rewards.unsqueeze(0)}
            for t in [999, 600, 300]:
                new = pipe.get_sample_guide(subset, torch.tensor(t), latents.clone(), 5000.0).float()
                ref = FX[f"guide/{name}/t{t}"]
                rel = ((new - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                self.assertLess(rel, 1e-2, f"{name} t={t} relative error {rel}")

    def test_rs_smc(self):
        from diffusers import DDPMScheduler
        from fkd_class import FKD
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
            sig = torch.tensor([fkd.get_effective_smc_sigma(i) for i in range(100)])
            self.assertTrue(torch.equal(sig, FX[f"smc_sigma/{cname}"]), cname)
            torch.manual_seed(7)
            latents = torch.randn(4, 3, 8, 8)
            for i in range(100):
                latents, _ = fkd.resample(sampling_idx=i, latents=latents, x0_preds=latents * 0.9)
            self.assertTrue(torch.equal(latents, FX[f"smc_final/{cname}"]), cname)
            self.assertTrue(torch.equal(fkd.population_rs, FX[f"smc_rs/{cname}"]), cname)

    def test_flux_velocity_guidance(self):
        from rs_flow_lidar import RSFlowLiDARGuidance
        guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0)
        z_t, z0, rew = flux_inputs()
        for t in [1.0, 0.988, 0.97, 0.9]:
            out = guider.compute_velocity_guidance(z_t, z0, rew, sigma_t=t, t_start=1.0, t_end=0.95)
            self.assertTrue(torch.equal(out, FX[f"flux/t{t}"]), t)

    def test_pareto_potential(self):
        from multi_reward_evaluator import compute_multi_reward_pareto_potential
        torch.manual_seed(3)
        multi = {"ImageReward": torch.randn(4, 6), "Clip-Score": torch.randn(4, 6) * 0.1}
        pot, _ = compute_multi_reward_pareto_potential(multi, sigma=0.5, pareto_rho=0.5, beta_risk=0.0,
                                                        noise_mode="independent_iid")
        self.assertTrue(torch.equal(pot, FX["pareto/iid"]))

    def test_prompt_helpers(self):
        from rslidar_utils import load_geneval_metadata, parse_prompt_indices
        self.assertEqual(parse_prompt_indices("stratified_100"), FX["prompts/stratified_100"].tolist())
        self.assertEqual(parse_prompt_indices("0-4"), FX["prompts/0-4"].tolist())
        self.assertEqual(parse_prompt_indices("1,5,9"), FX["prompts/list"].tolist())
        meta = load_geneval_metadata("prompt_files/geneval_metadata.jsonl", max_prompts=10, prompt_indices=[3, 1])
        self.assertEqual([m["original_idx"] for m in meta], FX["prompts/meta_idx"].tolist())
        self.assertEqual([m["prompt"] for m in meta], FX["prompts/meta_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
