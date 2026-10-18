# [ICML 2026 Spotlight] [Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](https://arxiv.org/pdf/2602.03211) 

**Authors: [Yeongmin Kim](https://sites.google.com/view/yeongmin-space/), Donghyeok Shin, Byeonghu Na, Minsang Park, Richard Lee Kim, and Il-Chul Moon**   

---

## 🌟 Extended Framework: RS-LiDAR & RS-SMC

This repository contains the official implementation of **LiDAR** alongside novel research extensions:
1. **RS-LiDAR (Randomized Smoothing for Lookahead Sample Reward Guidance)**: Mitigating non-Lipschitz reward landscapes and spurious reward peaks via Gaussian convolution in pixel space.
2. **RS-SMC (Randomized Smoothing Sequential Monte Carlo with Tweedie Analytical Annealing)**: Overcoming Tweedie approximation drift at intermediate diffusion timesteps during particle resampling (Table 8 benchmark).

---

## 🚀 Key Features & Methods

### 1. Phase 1 & Phase 2 Lookahead Guidance (Vanilla LiDAR)
- **Phase 1 (Lookahead Sampling)**: Pre-samples candidate trajectories using fast ODE/SDE solvers (e.g. DPM-Solver 5–8 steps) and computes preference rewards.
- **Phase 2 (Closed-Form Steering)**: Steers the diffusion generation process at steps $t > 200$ using an exact closed-form gradient derived from the lookahead potential.

```bash
# Phase 1: Lookahead sampling (50 particles, 5 steps)
python lookahead_sampling.py --seed=100 --num_particles=50 --num_inference_steps=5 --save_individual_images=True

# Phase 2: LiDAR target sampling with closed-form guidance
python LiDAR_sampling.py --use_rag --scale=12.5 --resample_t_end=200 --lookahead_path=100_50_5 --save_individual_images
```

---

### 2. Sequential Monte Carlo (SMC) & RS-SMC (Table 8 Replication)

Table 8 studies multi-particle Sequential Monte Carlo (Feynman-Kac steering) with $N \in \{1, 2, 4, 8\}$ particles. 
Particles are evaluated on Tweedie predictions $\hat{x}_0(x_t) = \mathbb{E}[x_0 \mid x_t]$ at intermediate steps ($i \in [20, 80]$):

$$\Delta R = R(\hat{x}_0^{(t)}) - R(\hat{x}_0^{(t-\Delta t)})$$

#### The Challenge: Tweedie Approximation Drift & Reward Ruggedness
- At high noise levels ($t \in [20, 80]$), Tweedie estimations are blurry conditional expectations.
- Raw reward models (ImageReward, CLIP, HPSv2) have steep local Lipschitz gradients and adversarial peaks, causing **Premature Pruning** (dropping good candidates) or **Spurious Collapse** (replicating noisy artifacts).

#### The Solution: RS-SMC with Analytical Tweedie Annealing
RS-SMC smooths the reward landscape via Monte Carlo Gaussian perturbations:

$$\hat{R}_\sigma(\hat{x}_0) = \frac{1}{M} \sum_{m=1}^M R\left(\text{clamp}(\hat{x}_0 + \sigma \epsilon_m, -1, 1)\right), \quad \epsilon_m \sim \mathcal{N}(0, I)$$

To eliminate over-smoothing, the noise radius $\sigma(t)$ anneals according to the **exact analytical Tweedie posterior standard deviation**:

$$\psi(t) = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}$$

$$\sigma(t) = \sigma_{\text{init}} \cdot \frac{\psi(t) - \psi(t_{\text{end}})}{\psi(t_{\text{start}}) - \psi(t_{\text{end}})}, \quad \forall t \in [t_{\text{start}}, t_{\text{end}}]$$

- **Calibrated Initial Radii**: $\sigma_{\text{init}} = 0.25$ for SD 1.5, $\sigma_{\text{init}} = 0.15$ for SDXL (avoiding pixel clipping saturation in $[-1, 1]$).
- **Strict Terminal Zero-Noise Evaluation**: At steps $i \ge t_{\text{end}}$ and terminal step $i = 99$, $\sigma(t) \equiv 0.0$, guaranteeing 100% clean, sharp images for final Best-of-N particle selection.

---

## 💻 CLI Usage Guide

### A. Stable Diffusion v1.5

```bash
# 1. Vanilla SMC (Table 8 Baseline, N=4)
# Pass --smc_terminal_resample=False to avoid step 99 particle collapse and match Table 8 SMC unselected mean
python LiDAR_sampling.py --model_name="runwayml/stable-diffusion-v1-5" \
  --num_particles=4 --num_inference_steps=100 --eta=1.0 --scale=12.5 \
  --use_smc --resample_frequency=20 --FK_lmbda=10.0 \
  --FK_resample_t_start=20 --FK_resample_t_end=80 \
  --smc_terminal_resample=False \
  --output_dir=Target_samples --save_individual_images

# 2. Annealed RS-SMC (Tweedie Analytical Schedule, N=4, M=4)
python LiDAR_sampling.py --model_name="runwayml/stable-diffusion-v1-5" \
  --num_particles=4 --num_inference_steps=100 --eta=1.0 --scale=12.5 \
  --use_rs_smc --smc_sigma=0.25 --smc_num_mc_samples=4 \
  --smc_anneal_sigma --smc_anneal_schedule=tweedie_analytical \
  --resample_frequency=20 --FK_lmbda=10.0 \
  --output_dir=Target_samples --save_individual_images
```

### B. Stable Diffusion XL (SDXL Base 1.0)

```bash
# Annealed RS-SMC on SDXL (Auto-dispatches scale=8.0, smc_sigma=0.15)
python LiDAR_sampling.py --model_name="stabilityai/stable-diffusion-xl-base-1.0" \
  --num_particles=4 --num_inference_steps=100 --eta=1.0 \
  --use_rs_smc --smc_num_mc_samples=4 \
  --smc_anneal_sigma --smc_anneal_schedule=tweedie_analytical \
  --resample_frequency=20 --FK_lmbda=10.0 \
  --output_dir=Target_samples --save_individual_images
```

---

## 📓 Google Colab Notebooks

Interactive, 1-click replication workflows with automatic Google Drive persistence and systematic logging:
- [`colab/RS_SMC_Table8_Replication_Colab.ipynb`](./colab/RS_SMC_Table8_Replication_Colab.ipynb): Complete Table 8 sweep ($N \in \{2, 4, 8\}$) comparing Vanilla SMC, Static RS-SMC, and Annealed RS-SMC.
- [`colab/LiDAR_Table2_Replication_Colab.ipynb`](./colab/LiDAR_Table2_Replication_Colab.ipynb): Full Table 2 replication comparing Vanilla, Best-of-N, DATE, and LiDAR.

---

## 🧪 Unit & Regression Testing

To verify numerical correctness, zero-regression on Vanilla SMC, and monotonic Tweedie variance decay:

```bash
python test_rs_smc_unit.py
```

All 12 unit tests validate:
1. Exact zero-regression when `use_smc_smoothing=False`.
2. Monte Carlo expectation clamping in $[-1, 1]$.
3. Common Random Numbers (CRN) invariance.
4. Shape compatibility across SD 1.5 ($64 \times 64$) and SDXL ($128 \times 128$).
5. CLI auto-dispatch rules.
6. Systematic directory naming convention.
7. SD 1.5 analytical monotonic sigma decay ($0.250 \to 0.094 \to 0.033 \to 0.000$).
8. SDXL analytical monotonic sigma decay ($0.150 \to 0.057 \to 0.020 \to 0.000$).
9. Constant $\sigma$ preservation when annealing is disabled.
10. Safe auto-bypass for $N=1$ single-particle runs.
11. Strict zero-noise clean evaluation at terminal step $i=99$.
12. Dynamic scheduler `alphas_cumprod` analytical integration.

---

## 📚 Acknowledgements
This codebase builds upon and is inspired by:
- **Diffusers**: https://github.com/huggingface/diffusers
- **ImageReward**: https://github.com/THUDM/ImageReward
- **FK-Diffusion-Steering**: https://github.com/zacharyhorvitz/Fk-Diffusion-Steering
- **DATE**: https://github.com/aailab-kaist/DATE

---

## 📖 Citation
```bibtex
@article{kim2026lookahead,
  title={Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models},
  author={Kim, Yeongmin and Shin, Donghyeok and Na, Byeonghu and Park, Minsang and Kim, Richard Lee and Moon, Il-Chul},
  journal={arXiv preprint arXiv:2602.03211},
  year={2026}
}
```
