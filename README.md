# Randomized Smoothing for Provably Robust Test-Time Generative Alignment

Code for **RS-TTA**: randomized smoothing of reward evaluations for derivative-free test-time
alignment of diffusion and flow models. The smoothed reward

$$\bar R_\sigma(\mathbf x) = \frac{1}{M}\sum_{m=1}^{M} R\big(\Pi_{[-1,1]}(\mathbf x + \sigma\mathbf u_m)\big),\qquad \mathbf u_m\sim\mathcal N(\mathbf 0,\mathbf I),$$

is evaluated on decoded images and plugged into two samplers:

| Component | What it does | Where |
|---|---|---|
| **RS-LiDAR** | Smoothed rewards for lookahead samples (Phase 1), closed-form lookahead guidance (Phase 2) | `lookahead_sampling.py`, `LiDAR_sampling.py`, `fkd_diffusers/fkd_pipeline_{sd,sdxl}.py` (`get_sample_guide`) |
| **RS-SMC** | Smoothed Feynman–Kac potentials with σ(t) annealed by the Tweedie posterior scale ψ(t) | `fkd_diffusers/fkd_class.py` (`get_effective_smc_sigma`, `resample`) |
| **RS-FlowLiDAR** | Lookahead guidance for rectified-flow models (FLUX.1-dev), Δv = −(s/t)·Σ(wʳ − w)·ẑ₀ | `rs_flow_lidar.py`, `fkd_diffusers/fkd_pipeline_flux.py` |

The implementation builds on the official code of LiDAR ("Lookahead Sample Reward Guidance for
Test-Time Scaling of Diffusion Models", Kim et al., ICML 2026,
<https://github.com/aailab-kaist/Diffusion-LiDAR-Sampling>), released under the Apache-2.0 license
(see `LICENSE`). Vanilla LiDAR is recovered by omitting `--use_smoothing`.

## Repository layout

```
lookahead_sampling.py      Phase 1: lookahead samples + (smoothed) reward annotation
LiDAR_sampling.py          Phase 2: LiDAR / RS-LiDAR guidance, SMC / RS-SMC, FLUX (RS-FlowLiDAR)
lookahead_datasets.py      Loads Phase 1 outputs (latents + rewards) for Phase 2
rs_flow_lidar.py           Velocity guidance for flow matching
multi_reward_evaluator.py  Multi-reward (Pareto) smoothed potential (optional, --use_pareto)
fks_utils.py               Reward evaluation (ImageReward, CLIP, HPS v2.1, aesthetic)
compute_diversity_metrics.py  Vendi score / LPIPS / CLIP diversity of generated sets
run_geneval_eval.py        GenEval evaluation through the official GenEval repository
rslidar_utils.py           Shared CLI helpers and runtime compatibility shims
fkd_diffusers/             SD 1.5 / SDXL / FLUX pipelines, SMC core, reward wrappers
experiments/               Scripts for the analyses and ablations in the paper
prompt_files/              GenEval prompts (553) and the stratified subsets
tests/                     Unit tests, golden-equivalence and CLI tests
```

## Installation

Python 3.10+, one CUDA GPU (A100 40 GB used in the paper; SD 1.5 needs ~9 GiB, SDXL ~34 GiB, FLUX ~22 GiB).

```bash
pip install torch torchvision                  # CUDA build matching your driver
pip install -r requirements.txt                # SD 1.5 / SDXL experiments
# FLUX experiments use a separate environment:
pip install -r requirements-flux.txt
```

FLUX.1-dev is a gated model: accept its license on Hugging Face and export `HF_TOKEN`.
All commands below are run from the repository root. Phase 1 writes to `Lookahead_samples/<run_name>/`,
Phase 2 to `Target_samples/<run_name>/` (one folder per GenEval prompt, plus `final_metrics.json`).

## Reproducing the results

### RS-LiDAR (main benchmark, SD 1.5 and SDXL)

```bash
# Phase 1 — DPM-5 lookahead, n = 50 candidates, smoothed ImageReward (sigma = 1.0, M = 8, i.i.d. noise)
python lookahead_sampling.py --model_name runwayml/stable-diffusion-v1-5 \
  --num_particles 50 --num_inference_steps 5 --seed 100 \
  --guidance_reward_fn ImageReward --metrics_to_compute ImageReward \
  --use_smoothing --sigma 1.0 --num_mc_samples 8 --noise_mode independent_iid \
  --run_name RS_SD15_DPM5_n50_sig1.0_M8

# Phase 2 — DDPM-100 target sampler (use --num_inference_steps 50 --eta 0.0 for DDIM-50), 4 images per prompt
python LiDAR_sampling.py --model_name runwayml/stable-diffusion-v1-5 \
  --num_particles 4 --num_inference_steps 100 --eta 1.0 --seed 100 \
  --use_rag --lookahead_path RS_SD15_DPM5_n50_sig1.0_M8 --top_k 50 \
  --scale 12.5 --lmbda 5000 --resample_t_end 200 \
  --guidance_reward_fn ImageReward --metrics_to_compute "ImageReward#Clip-Score#HumanPreference" \
  --save_individual_images --run_name RS_SD15_DDPM100
```

SDXL uses the DMD-1 lookahead with n = 100 and sigma = 0.25:
`lookahead_sampling.py --model_name dmd2_sdxl_1step_unet_fp16.bin --num_particles 100 --num_inference_steps 1 --sigma 0.25 ...`
followed by `LiDAR_sampling.py --model_name stabilityai/stable-diffusion-xl-base-1.0 --top_k 100 --scale 8.0 ...`.
`--prompt_indices` restricts a run to a subset (e.g. `stratified_100`, `0-99`, `1,5,9`).

### Noise-topology ablation (Phase 1)

| Variant | Phase 1 flags |
|---|---|
| I.I.D. | `--noise_mode independent_iid` |
| Coupled CRN | `--use_crn` (same as `--noise_mode coupled_crn`) |
| Antithetic | `--noise_mode independent_antithetic` (even `--num_mc_samples`) |
| Entropic Risk | `--noise_mode independent_antithetic --beta_risk <beta>` |

### RS-SMC (multi-particle benchmark)

```bash
python LiDAR_sampling.py --model_name runwayml/stable-diffusion-v1-5 \
  --num_particles 4 --num_inference_steps 100 --eta 1.0 --scale 12.5 --seed 100 \
  --use_smc --use_smc_smoothing --smc_sigma 0.25 --smc_num_mc_samples 4 \
  --smc_anneal_sigma --smc_anneal_schedule tweedie_analytical --smc_sigma_min 0.01 \
  --resample_frequency 20 --FK_lmbda 10.0 --FK_resample_t_start 20 --FK_resample_t_end 80 \
  --metrics_to_compute ImageReward --save_individual_images --run_name RSSMC_SD15_N4
```

Standard SMC: drop the `--use_smc_smoothing ...` flags. LiDAR + RS-SMC: add
`--use_rag --lookahead_path <phase1_run> --top_k 50 --lmbda 5000`. σ(t) schedule ablation:
`--smc_anneal_schedule {linear,cosine,tweedie_analytical}`, or omit `--smc_anneal_sigma` for a static σ.

### Analyses

```bash
# Empirical Lipschitz contraction (553 prompts x 10 particles = 5,530 pairs, sigma = 1.0, M = 4, CRN)
python -m experiments.lipschitz_empirical --prompts_file prompt_files/geneval_metadata.jsonl \
  --num_prompts 553 --num_particles 10 --sigma1 0.1 --sigma2 1.0 --sigma2_list 0.0,0.1,0.25,0.5,1.0 \
  --num_mc_samples 4 --reward_models ImageReward,CLIP-Score,HPS,Aesthetic --use_crn \
  --output_dir results/lipschitz

# 1D reward-landscape slices and qualitative comparison (e.g. prompt 98, "a photo of a couch and a horse")
python -m experiments.landscape_slicing --prompt_indices 98 --num_lookahead 50 --lookahead_steps 5 \
  --target_steps 50 --sigma 1.0 --mc_samples 16 --output_dir results/landscape

# Target-sampler step ablation (evaluates intermediate step budgets of one run)
python -m experiments.target_steps_ablation --model_name runwayml/stable-diffusion-v1-5 \
  --use_rag --lookahead_path RS_SD15_DPM5_n50_sig1.0_M8 --num_inference_steps 50 --eta 0.0 \
  --eval_checkpoint_steps 25,30,35,40,45,50 --run_name RS_SD15_steps

# Latency / peak-VRAM breakdown (lookahead, reward annotation, target sampling)
python -m experiments.efficiency_benchmark --setting sdv1.5_ddpm100 --method rs-lidar
```

`experiments/adversarial_reward.py`, `experiments/threshold_reward.py` and
`experiments/lidar_weaknesses.py` contain the supplementary robustness studies (run with `--help`).

### FLUX.1-dev (RS-FlowLiDAR)

```bash
python lookahead_sampling.py --model_name ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors \
  --num_particles 3 --num_inference_steps 8 --seed 100 \
  --guidance_reward_fn ImageReward --metrics_to_compute ImageReward \
  --use_smoothing --sigma 0.25 --num_mc_samples 4 --noise_mode independent_iid
python LiDAR_sampling.py --model_name black-forest-labs/FLUX.1-dev --method RS-FlowLiDAR \
  --lookahead_path FLUX_100_3_8_rs_sig0.25_M4_iid --top_k 3 --s_flow <s> \
  --guidance_start 1.0 --guidance_end 0.95 --num_inference_steps 28 --num_particles 1 \
  --save_individual_images --run_name FLUX_RSFlowLiDAR
```

`--flux_debug` prints per-step guidance diagnostics (t, wʳ − w, ‖Δt·Δv‖/‖z‖).

## Evaluation

ImageReward, CLIP score and HPS v2.1 are computed during Phase 2 (`--metrics_to_compute`) and
aggregated in `Target_samples/<run_name>/final_metrics.json`. GenEval is computed with the official
evaluator:

```bash
python run_geneval_eval.py --target_dir Target_samples/<run_name>
```

## Tests

```bash
python -m pytest -q tests
```

`tests/test_golden_equivalence.py` checks the core algorithms (smoothing, entropic aggregation,
closed-form guidance, RS-SMC schedules and resampling, FLUX guidance, Pareto potential) against
recorded reference outputs; `tests/test_cli_compat.py` checks the documented command-line interfaces.
