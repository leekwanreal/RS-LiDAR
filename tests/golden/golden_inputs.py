"""Deterministic inputs shared by the golden-fixture generator and the golden equivalence test."""
import torch


def pixel_images():
    g = torch.Generator().manual_seed(0)
    return (torch.rand(5, 3, 16, 16, generator=g) * 2 - 1)


def rewards_mc():
    g = torch.Generator().manual_seed(2)
    return torch.randn(4, 6, generator=g)


def toy_eval_fn(images):
    """Stand-in reward: deterministic, non-linear function of each image."""
    return (images.mean(dim=(1, 2, 3)) * 3.0 + torch.sin(images[:, 0].mean(dim=(1, 2)) * 5.0)).tolist()


def smc_reward_fn(images):
    return images.mean(dim=(1, 2, 3)) * 2.0 + torch.cos(images[:, 0].std(dim=(1, 2)) * 4.0)


def sd_guide_inputs(ch, hw):
    g = torch.Generator().manual_seed(4)
    latents = torch.randn(2, ch, hw, hw, generator=g)
    lookahead = torch.randn(5, ch, hw, hw, generator=g)
    rewards = torch.tensor([0.31, 0.29, 0.35, 0.30, 0.28])
    return latents, lookahead, rewards


def flux_inputs():
    g = torch.Generator().manual_seed(5)
    z_t = torch.randn(1, 16, 8, 8, generator=g)
    z0 = torch.randn(3, 16, 8, 8, generator=g) * 0.8
    return z_t, z0, torch.tensor([0.30, 0.31, 0.29])


def reference_smoothing_loop(decoded, sigma, M, noise_mode, eval_fn):
    """Frozen copy of the pre-refactor Phase 1 smoothing loop (image domain, `use_smoothing_loop=True`).

    Returns the [M, N] matrix of rewards on noisy images, reproducing the exact RNG call order of
    `lookahead_sampling.main` before the refactor. `eval_fn(images) -> list[float]` replaces
    `postprocess + do_eval`.
    """
    torch.manual_seed(11)
    half_M = M // 2
    is_antithetic = noise_mode == "independent_antithetic" and M % 2 == 0
    u_half_list = [torch.randn_like(decoded) for _ in range(half_M)] if is_antithetic else None
    rows = []
    for m_idx in range(M):
        if sigma > 0:
            if noise_mode == "coupled_crn":
                noise = torch.randn((1, *decoded.shape[1:]), dtype=decoded.dtype) * sigma
            elif is_antithetic:
                u_m = u_half_list[m_idx] if m_idx < half_M else -u_half_list[m_idx - half_M]
                noise = u_m * sigma
            else:
                noise = torch.randn_like(decoded) * sigma
            noisy = (decoded + noise).clamp(-1.0, 1.0)
        else:
            noisy = decoded.clamp(-1.0, 1.0)
        rows.append(eval_fn(noisy))
    return torch.tensor(rows, dtype=torch.float32)
