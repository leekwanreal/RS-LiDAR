"""Feynman-Kac (FK) steering with randomized-smoothing rewards (RS-SMC).

`FKD.resample` is called after every denoising step. At the resampling steps it decodes the Tweedie
estimates x0_hat of the N particles, scores them, and resamples particles with weights
exp(lmbda * potential) (FK steering, Singhal et al. 2025).

With `use_smc_smoothing=True` (RS-SMC) the reward of a particle is the randomized-smoothing estimate
    R_sigma(x0_hat) = (1/M) * sum_m R(clip(D(x0_hat) + sigma(t) * u_m, -1, 1)),
where `get_effective_smc_sigma` gives the smoothing radius sigma(t): constant, or annealed over the
resampling window (`tweedie_analytical`, `cosine`, `linear`).
"""

from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    """FK steering state for one batch of particles; call `.resample()` at every diffusion step.

    Args:
        potential_type: one of `PotentialType` ("diff", "max", "add", "rt").
        lmbda: inverse temperature of the resampling weights.
        num_particles: number of particles N.
        adaptive_resampling: resample only when the effective sample size drops below N / 2.
        resample_frequency, resampling_t_start, resampling_t_end: resampling step grid.
        time_steps: total number of sampling steps.
        reward_fn: maps decoded images [N, 3, H, W] in [-1, 1] to rewards [N].
        reward_min_value: initial reward of every particle (relevant for the "max" potential).
        latent_to_decode_fn: decodes Tweedie latents to images.
        device: device of the reward state.
        use_smc_smoothing: score particles with the smoothed reward (RS-SMC).
        smc_sigma, smc_num_mc_samples: smoothing radius sigma and Monte Carlo samples M.
        smc_use_crn: share each noise draw across particles (common random numbers).
        smc_smoothing_mode: "loop" over M, or "vectorized" (all M * N images at once).
        smc_anneal_sigma, smc_anneal_schedule, smc_sigma_min: sigma(t) schedule (see
            `get_effective_smc_sigma`); "sqrt_noise" and "tweedie_snr" are aliases of "tweedie_analytical".
        smc_terminal_resample: also resample at the last step.
        scheduler: diffusion scheduler (its `alphas_cumprod` / `timesteps` drive "tweedie_analytical").
        **kwargs: `FK_lmbda`, `FK_resample_t_start`, `FK_resample_t_end` override the corresponding
            arguments; `use_rs_smc`, `smc_M`, `sigma`, `num_mc_samples`, `terminal_resample` are aliases.
            Other keys (the full CLI `fkd_args` dict is passed) are ignored.
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
        smc_use_crn: bool = False,
        smc_smoothing_mode: str = "loop",
        smc_anneal_sigma: bool = False,
        smc_anneal_schedule: str = "tweedie_analytical",
        smc_sigma_min: Optional[float] = None,
        smc_terminal_resample: bool = True,
        scheduler: Optional[Any] = None,
        **kwargs,
    ) -> None:
        if kwargs.get("FK_lmbda") is not None:
            lmbda = kwargs["FK_lmbda"]
        if kwargs.get("FK_resample_t_start") is not None:
            resampling_t_start = kwargs["FK_resample_t_start"]
        if kwargs.get("FK_resample_t_end") is not None:
            resampling_t_end = kwargs["FK_resample_t_end"]

        if kwargs.get("use_rs_smc") is not None:
            use_smc_smoothing = kwargs["use_rs_smc"]
        if kwargs.get("smc_M") is not None:
            smc_num_mc_samples = kwargs["smc_M"]
        if kwargs.get("sigma") is not None:
            smc_sigma = kwargs["sigma"]
        if kwargs.get("num_mc_samples") is not None:
            smc_num_mc_samples = kwargs["num_mc_samples"]

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
        self.scheduler = scheduler

        self.use_smc_smoothing = use_smc_smoothing
        self.smc_sigma = smc_sigma
        self.smc_num_mc_samples = smc_num_mc_samples
        self.smc_use_crn = smc_use_crn
        self.smc_smoothing_mode = smc_smoothing_mode
        self.smc_anneal_sigma = smc_anneal_sigma
        self.smc_anneal_schedule = smc_anneal_schedule
        self.smc_sigma_min = smc_sigma_min
        self.smc_terminal_resample = kwargs.get("terminal_resample", smc_terminal_resample)

        self.device = device
        self.population_rs = torch.ones(self.num_particles, device=self.device) * reward_min_value
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

    def get_effective_smc_sigma(self, sampling_idx: int) -> float:
        """Smoothing radius sigma(t) used by RS-SMC at sampling step `sampling_idx`.

        Without annealing this is the constant `smc_sigma`. With annealing, sigma(t) = smc_sigma * decay(t)
        on the resampling window [t_start, t_end] (decay from 1 to 0), floored by `smc_sigma_min`, and the
        floor (0 by default) from t_end and at the last step:
          - tweedie_analytical: decay = (psi_t - psi_end) / (psi_start - psi_end), psi_t = sqrt((1 - abar_t) / abar_t)
            the noise-to-signal scale of the Tweedie estimate (falls back to progress^2 without a scheduler);
          - cosine: 0.5 * (1 + cos(pi * (1 - progress)));  linear: progress,
        where progress = (t_end - t) / (t_end - t_start) is clipped to [0, 1].
        """
        if not self.smc_anneal_sigma:
            return float(self.smc_sigma)

        t_start = self.resampling_t_start
        t_end = self.resampling_t_end
        total_span = max(1, t_end - t_start)

        # Final samples are scored without noise.
        if sampling_idx >= self.time_steps - 1 or sampling_idx >= t_end:
            return float(self.smc_sigma_min or 0.0)

        progress = float(np.clip((t_end - sampling_idx) / total_span, 0.0, 1.0))
        schedule = self.smc_anneal_schedule

        decay_factor = None
        if schedule in ["tweedie_analytical", "tweedie_snr", "sqrt_noise"]:
            scheduler = self.scheduler
            if scheduler is not None and hasattr(scheduler, "alphas_cumprod") and hasattr(scheduler, "timesteps"):
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

            if decay_factor is None:
                decay_factor = float(progress ** 2.0)
        elif schedule == "cosine":
            decay_factor = float(0.5 * (1.0 + np.cos(np.pi * (1.0 - progress))))
        elif schedule == "linear":
            decay_factor = float(progress)
        else:
            decay_factor = float(progress ** 2.0)

        effective_sigma = float(self.smc_sigma * decay_factor)
        if self.smc_sigma_min is not None:
            effective_sigma = max(float(self.smc_sigma_min), effective_sigma)
        return effective_sigma

    def _evaluate_reward_batched(self, imgs: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        """Evaluate `reward_fn` on a large batch in chunks, halving the chunk size on CUDA OOM."""
        total = imgs.shape[0]
        if total <= chunk_size:
            try:
                res = self.reward_fn(imgs)
                return res if isinstance(res, torch.Tensor) else torch.tensor(res, device=imgs.device)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)

        rewards_list = []
        for i in range(0, total, chunk_size):
            chunk = imgs[i : i + chunk_size]
            try:
                r_c = self.reward_fn(chunk)
                r_tensor = r_c if isinstance(r_c, torch.Tensor) else torch.tensor(r_c, device=imgs.device)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sub_chunk_size = max(1, chunk_size // 2)
                sub_rewards = []
                for j in range(0, chunk.shape[0], sub_chunk_size):
                    sub = chunk[j : j + sub_chunk_size]
                    sub_r = self.reward_fn(sub)
                    sub_tensor = sub_r if isinstance(sub_r, torch.Tensor) else torch.tensor(sub_r, device=imgs.device)
                    sub_rewards.append(sub_tensor)
                r_tensor = torch.cat(sub_rewards, dim=0)
            rewards_list.append(r_tensor)

        return torch.cat(rewards_list, dim=0)

    def _smoothed_rewards(self, population_images: torch.Tensor, sigma: float) -> torch.Tensor:
        """RS-SMC reward: Monte Carlo mean of `reward_fn` on clip(images + sigma * u_m, -1, 1), m = 1..M."""
        if sigma == 0:
            # sigma(t) = 0 (annealed schedule): the M noisy copies would be identical.
            return self.reward_fn(population_images.clamp(-1.0, 1.0))

        shape = population_images.shape
        if self.smc_smoothing_mode == "loop":
            mc_rewards = []
            with torch.inference_mode():
                for _ in range(self.smc_num_mc_samples):
                    if self.smc_use_crn:
                        noise = torch.randn((1, *shape[1:]), device=population_images.device,
                                            dtype=population_images.dtype) * sigma
                    else:
                        noise = torch.randn_like(population_images) * sigma
                    noisy_imgs = (population_images + noise).clamp(-1.0, 1.0)
                    mc_rewards.append(self.reward_fn(noisy_imgs))
            return torch.stack(mc_rewards, dim=0).mean(dim=0)

        with torch.inference_mode():
            if self.smc_use_crn:
                noise = torch.randn((self.smc_num_mc_samples, 1, *shape[1:]), device=population_images.device,
                                    dtype=population_images.dtype) * sigma
            else:
                noise = torch.randn((self.smc_num_mc_samples, *shape), device=population_images.device,
                                    dtype=population_images.dtype) * sigma
            noisy_imgs = (population_images.unsqueeze(0) + noise).clamp(-1.0, 1.0).view(-1, *shape[1:])
            raw_rewards = self._evaluate_reward_batched(noisy_imgs)
            return raw_rewards.view(self.smc_num_mc_samples, self.num_particles).mean(dim=0)

    def resample(
        self,
        *,
        sampling_idx: int,
        latents: torch.Tensor,
        x0_preds: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Resample particles at the scheduled steps.

        Args:
            sampling_idx: current sampling step index.
            latents: current noisy latents x_t [N, ...].
            x0_preds: Tweedie estimates x0_hat of the particles [N, ...].

        Returns:
            (resampled latents, resampled decoded images), or (latents, None) at steps without resampling.
        """
        resampling_interval = np.arange(self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency)
        if self.smc_terminal_resample:
            resampling_interval = np.append(resampling_interval, self.time_steps - 1)

        if sampling_idx not in resampling_interval:
            return latents, None

        population_images = self.latent_to_decode_fn(x0_preds)

        if not self.use_smc_smoothing:
            rs_candidates = self.reward_fn(population_images)
        else:
            rs_candidates = self._smoothed_rewards(population_images, self.get_effective_smc_sigma(sampling_idx))

        if self.population_rs.device != rs_candidates.device or self.population_rs.dtype != rs_candidates.dtype:
            self.population_rs = self.population_rs.to(device=rs_candidates.device, dtype=rs_candidates.dtype)
        if self.product_of_potentials.device != rs_candidates.device:
            self.product_of_potentials = self.product_of_potentials.to(device=rs_candidates.device)

        # Potentials (importance weights).
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
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                indices = torch.multinomial(w, num_samples=self.num_particles, replacement=True)
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]
                resampled_images = population_images[indices]
                # Product of potentials, used by the max / add / rt potentials.
                self.product_of_potentials = self.product_of_potentials[indices] * w[indices]
            else:
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            indices = torch.multinomial(w, num_samples=self.num_particles, replacement=True)
            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]
            resampled_images = population_images[indices]
            self.product_of_potentials = self.product_of_potentials[indices] * w[indices]

        return resampled_latents, resampled_images
