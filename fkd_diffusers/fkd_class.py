"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple, Any
import logging


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float = 10.0,
        num_particles: int = 4,
        adaptive_resampling: bool = False,
        resample_frequency: int = 20,
        resampling_t_start: int = 20,
        resampling_t_end: int = 80,
        time_steps: int = 100,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: Optional[torch.device] = None,
        use_smc_smoothing: bool = False,
        smc_sigma: float = 0.25,
        smc_num_mc_samples: int = 4,
        smc_smoothing_domain: str = "image",
        smc_use_crn: bool = False,
        smc_smoothing_mode: str = "loop",
        smc_anneal_sigma: bool = False,
        smc_anneal_schedule: str = "tweedie_analytical",
        smc_sigma_min: Optional[float] = None,
        smc_terminal_resample: bool = True,
        scheduler: Optional[Any] = None,
        **kwargs,
    ) -> None:
        # Support FK_ prefixed argument overrides from CLI
        if "FK_lmbda" in kwargs and kwargs["FK_lmbda"] is not None:
            lmbda = kwargs["FK_lmbda"]
        if "FK_resample_t_start" in kwargs and kwargs["FK_resample_t_start"] is not None:
            resampling_t_start = kwargs["FK_resample_t_start"]
        if "FK_resample_t_end" in kwargs and kwargs["FK_resample_t_end"] is not None:
            resampling_t_end = kwargs["FK_resample_t_end"]

        # Support RS-SMC aliases from CLI / kwargs
        if "use_rs_smc" in kwargs and kwargs["use_rs_smc"] is not None:
            use_smc_smoothing = kwargs["use_rs_smc"]
        if "smc_M" in kwargs and kwargs["smc_M"] is not None:
            smc_num_mc_samples = kwargs["smc_M"]
        if "sigma" in kwargs and kwargs["sigma"] is not None and "smc_sigma" not in kwargs:
            smc_sigma = kwargs["sigma"]
        if "num_mc_samples" in kwargs and kwargs["num_mc_samples"] is not None and "smc_num_mc_samples" not in kwargs:
            smc_num_mc_samples = kwargs["num_mc_samples"]
        if "smc_anneal_sigma" in kwargs and kwargs["smc_anneal_sigma"] is not None:
            smc_anneal_sigma = kwargs["smc_anneal_sigma"]
        if "smc_anneal_schedule" in kwargs and kwargs["smc_anneal_schedule"] is not None:
            smc_anneal_schedule = kwargs["smc_anneal_schedule"]
        if "smc_sigma_min" in kwargs and kwargs["smc_sigma_min"] is not None:
            smc_sigma_min = kwargs["smc_sigma_min"]

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn
        self.scheduler = scheduler or kwargs.get("scheduler", None)

        # RS-SMC Hyperparameters
        self.use_smc_smoothing = use_smc_smoothing
        self.smc_sigma = smc_sigma
        self.smc_num_mc_samples = smc_num_mc_samples
        self.smc_smoothing_domain = smc_smoothing_domain
        self.smc_use_crn = smc_use_crn
        self.smc_smoothing_mode = smc_smoothing_mode
        self.smc_anneal_sigma = smc_anneal_sigma
        self.smc_anneal_schedule = smc_anneal_schedule
        self.smc_sigma_min = smc_sigma_min
        self.smc_terminal_resample = kwargs.get(
            "smc_terminal_resample",
            kwargs.get("terminal_resample", smc_terminal_resample),
        )

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

    def get_effective_smc_sigma(self, sampling_idx: int) -> float:
        """
        Calculate the effective smoothing radius sigma(t) at the current diffusion sampling step.
        If smc_anneal_sigma is False, returns constant self.smc_sigma (Zero Regression).
        When smc_anneal_sigma is True, computes analytical annealed decay grounded in Tweedie posterior variance.
        """
        if not getattr(self, "smc_anneal_sigma", False):
            return float(self.smc_sigma)

        t_start = getattr(self, "resampling_t_start", 20)
        t_end = getattr(self, "resampling_t_end", 80)
        total_span = max(1, t_end - t_start)

        # Terminal step (sampling_idx == time_steps - 1) or beyond t_end must be clean (zero-noise)
        # to ensure Best-of-N final selection and completed samples are never corrupted by artificial noise.
        if sampling_idx >= self.time_steps - 1 or sampling_idx >= t_end:
            min_floor = float(getattr(self, "smc_sigma_min", 0.0) or 0.0)
            return min_floor

        # Progress from 1.0 (at t_start) down to 0.0 (at t_end)
        progress = float(np.clip((t_end - sampling_idx) / total_span, 0.0, 1.0))

        schedule = getattr(self, "smc_anneal_schedule", "tweedie_analytical")

        # Optimal Schedule 1: Analytical Tweedie posterior standard deviation ratio
        # Computed directly from scheduler.alphas_cumprod without any hard-coded constants!
        decay_factor = None
        if schedule in ["tweedie_analytical", "tweedie_snr", "sqrt_noise"]:
            scheduler = getattr(self, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "alphas_cumprod") and hasattr(scheduler, "timesteps"):
                try:
                    timesteps = scheduler.timesteps
                    if len(timesteps) > sampling_idx and len(timesteps) > t_start:
                        curr_t = timesteps[sampling_idx]
                        t_start_val = timesteps[t_start]
                        t_end_val = timesteps[min(t_end, len(timesteps) - 1)]

                        a_curr = float(scheduler.alphas_cumprod[curr_t])
                        a_start = float(scheduler.alphas_cumprod[t_start_val])
                        a_end = float(scheduler.alphas_cumprod[t_end_val])

                        psi_curr = np.sqrt(max(0.0, 1.0 - a_curr) / max(1e-8, a_curr))
                        psi_start = np.sqrt(max(0.0, 1.0 - a_start) / max(1e-8, a_start))
                        psi_end = np.sqrt(max(0.0, 1.0 - a_end) / max(1e-8, a_end))

                        decay_factor = float(np.clip((psi_curr - psi_end) / max(1e-6, psi_start - psi_end), 0.0, 1.0))
                except Exception:
                    decay_factor = None

            if decay_factor is None:
                # Closed-form power law progress^2 matches Tweedie posterior variance ratio (1.0 -> 0.44 -> 0.11 -> 0.0)
                decay_factor = float(progress ** 2.0)
        elif schedule == "cosine":
            # Standard Cosine Annealing: smooth concave decay from 1.0 to 0.0 with zero derivatives at both ends
            decay_factor = float(0.5 * (1.0 + np.cos(np.pi * (1.0 - progress))))
        elif schedule == "linear":
            # Linear decay from 1.0 to 0.0
            decay_factor = float(progress)
        else:
            decay_factor = float(progress ** 2.0)

        effective_sigma = float(self.smc_sigma * decay_factor)

        # Apply user-specified minimum floor if provided
        if getattr(self, "smc_sigma_min", None) is not None:
            effective_sigma = max(float(self.smc_sigma_min), effective_sigma)

        return effective_sigma

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Check if resampling is within the allowed range and conditions
        resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        if getattr(self, "smc_terminal_resample", True):
            resampling_interval = np.append(resampling_interval, self.time_steps - 1)

        if sampling_idx not in resampling_interval:
            return latents, None

        # Decode latents to population images and compute rewards
        population_images = self.latent_to_decode_fn(x0_preds)

        if not getattr(self, "use_smc_smoothing", False):
            # Vanilla SMC: Original raw reward evaluation (Zero Regression)
            rs_candidates = self.reward_fn(population_images)
        else:
            # RS-SMC: Randomized Smoothing over Tweedie estimate with dynamic sigma(t)
            curr_sigma = self.get_effective_smc_sigma(sampling_idx)
            if self.smc_smoothing_domain == "image":
                if self.smc_smoothing_mode == "loop":
                    mc_rewards = []
                    with torch.inference_mode():
                        for _ in range(self.smc_num_mc_samples):
                            if curr_sigma > 0:
                                if self.smc_use_crn:
                                    noise = torch.randn((1, *population_images.shape[1:]), device=population_images.device, dtype=population_images.dtype) * curr_sigma
                                else:
                                    noise = torch.randn_like(population_images) * curr_sigma
                                noisy_imgs = (population_images + noise).clamp(-1.0, 1.0)
                            else:
                                noisy_imgs = population_images.clamp(-1.0, 1.0)
                            r_m = self.reward_fn(noisy_imgs)
                            mc_rewards.append(r_m)
                    rs_candidates = torch.stack(mc_rewards, dim=0).mean(dim=0)
                else: # vectorized
                    with torch.inference_mode():
                        if curr_sigma > 0:
                            if self.smc_use_crn:
                                noise = torch.randn((self.smc_num_mc_samples, 1, *population_images.shape[1:]), device=population_images.device, dtype=population_images.dtype) * curr_sigma
                                noisy_imgs = (population_images.unsqueeze(0) + noise).clamp(-1.0, 1.0).view(-1, *population_images.shape[1:])
                            else:
                                noise = torch.randn((self.smc_num_mc_samples, *population_images.shape), device=population_images.device, dtype=population_images.dtype) * curr_sigma
                                noisy_imgs = (population_images.unsqueeze(0) + noise).clamp(-1.0, 1.0).view(-1, *population_images.shape[1:])
                        else:
                            noisy_imgs = population_images.repeat(self.smc_num_mc_samples, 1, 1, 1)
                        raw_rewards = self.reward_fn(noisy_imgs)
                        rs_candidates = raw_rewards.view(self.smc_num_mc_samples, self.num_particles).mean(dim=0)
            elif self.smc_smoothing_domain == "latent":
                mc_rewards = []
                with torch.inference_mode():
                    for _ in range(self.smc_num_mc_samples):
                        if curr_sigma > 0:
                            if self.smc_use_crn:
                                noise = torch.randn((1, *x0_preds.shape[1:]), device=x0_preds.device, dtype=x0_preds.dtype) * curr_sigma
                            else:
                                noise = torch.randn_like(x0_preds) * curr_sigma
                            noisy_latents = x0_preds + noise
                        else:
                            noisy_latents = x0_preds
                        noisy_imgs = self.latent_to_decode_fn(noisy_latents)
                        r_m = self.reward_fn(noisy_imgs)
                        mc_rewards.append(r_m)
                rs_candidates = torch.stack(mc_rewards, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown smc_smoothing_domain: {self.smc_smoothing_domain}")

        # Ensure device and dtype alignment
        if self.population_rs.device != rs_candidates.device or self.population_rs.dtype != rs_candidates.dtype:
            self.population_rs = self.population_rs.to(device=rs_candidates.device, dtype=rs_candidates.dtype)
        if self.product_of_potentials.device != rs_candidates.device:
            self.product_of_potentials = self.product_of_potentials.to(device=rs_candidates.device)

        # Compute importance weights
        if self.potential_type == PotentialType.MAX:
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        if sampling_idx == self.time_steps - 1:
            if (
                self.potential_type == PotentialType.MAX
                or self.potential_type == PotentialType.ADD
                or self.potential_type == PotentialType.RT
            ):
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            # compute effective sample size
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]

                # Resample population images
                resampled_images = population_images[indices]

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            # Resample indices based on weights
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]

            # Resample population images
            resampled_images = population_images[indices]

            # Update product of potentials; used for max and add potentials
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )

        return resampled_latents, resampled_images


if __name__ == "__main__":

    # Demonstration of FKD resampling step
    import matplotlib.pyplot as plt
    import random

    # set seed
    random.seed(0)

    # 1x1 pixel images
    num_particles = 8
    x0s = torch.rand(num_particles, 1, 1)

    # reward darker images
    reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

    # Define the FKD steering mechanism
    fkds = FKD(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
        reward_fn=lambda x: reward_function(x),
        device=torch.device('cpu'),
    )

    # Define the sampling index
    sampling_idx = 0

    # Perform resampling
    resampled_latents, resampled_images = fkds.resample(
        sampling_idx=sampling_idx,
        latents=x0s,
        x0_preds=x0s,
    )

    plt.rc('text', usetex=True)
    fig, axs = plt.subplots(2, num_particles)

    axs[0, 0].set_title('Initial')
    axs[1, 0].set_title('Resampled')

    for i in range(num_particles):
        axs[0, i].imshow(x0s[i].detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[1, i].imshow(
            resampled_images[i].detach().numpy(), cmap='gray', vmin=0, vmax=1
        )

        axs[1, i].axis('off')
        axs[0, i].axis('off')

    out_path = 'resampled_examples.png'
    plt.savefig(out_path)
    print('Saved resampled examples to:', out_path)
