"""SMC resampling weights: the 1e10 clamp of the default path vs `fk_stable_weights` (E7, FK_lmbda = 30)."""
import pytest
import torch

from fkd_diffusers.fkd_class import FKD, PotentialType


def _weights(rewards, lmbda, stable, monkeypatch, potential=PotentialType.DIFF):
    """Normalized weights passed to torch.multinomial at resampling step 20 (previous rewards are 0)."""
    captured = {}
    real = torch.multinomial

    def spy(w, *a, **k):
        captured["w"] = w.detach().clone()
        return real(w, *a, **k)

    monkeypatch.setattr(torch, "multinomial", spy)
    r = torch.tensor(rewards, dtype=torch.float32)
    fkd = FKD(potential_type=potential, lmbda=lmbda, num_particles=len(rewards), resample_frequency=20,
              resampling_t_start=20, resampling_t_end=80, time_steps=100, reward_fn=lambda x: r,
              latent_to_decode_fn=lambda x: x, device=torch.device("cpu"), fk_stable_weights=stable)
    x = torch.arange(len(rewards), dtype=torch.float32).view(-1, 1)
    fkd.resample(sampling_idx=20, latents=x, x0_preds=x)
    w = captured["w"].double()
    return w / w.sum()


def test_same_weights_when_clamp_not_hit(monkeypatch):
    rewards, lmbda = [0.1, 0.3, 0.2, 0.0], 10.0
    w_default = _weights(rewards, lmbda, False, monkeypatch)
    w_stable = _weights(rewards, lmbda, True, monkeypatch)
    assert torch.allclose(w_default, w_stable, rtol=1e-5, atol=1e-8)
    expected = torch.softmax(lmbda * torch.tensor(rewards, dtype=torch.float64), 0)
    assert torch.allclose(w_stable, expected, rtol=1e-5)


def test_clamp_flattens_default_but_not_stable(monkeypatch, capsys):
    rewards, lmbda = [1.0, 0.9, 0.0, 0.0], 30.0   # lmbda * d = 30, 27 > ln(1e10) ~ 23
    w_default = _weights(rewards, lmbda, False, monkeypatch)
    assert "FK weight clamp at step 20: 2/4" in capsys.readouterr().out
    assert torch.isclose(w_default[0], w_default[1])                 # clamp: both particles equally likely
    w_stable = _weights(rewards, lmbda, True, monkeypatch)
    assert torch.isclose(w_stable[0] / w_stable[1], torch.exp(torch.tensor(3.0, dtype=torch.float64)), rtol=1e-4)
    assert w_stable.sum().isclose(torch.tensor(1.0, dtype=torch.float64))


def test_stable_weights_finite_for_large_lambda(monkeypatch):
    w = _weights([5.0, -5.0, 0.0, 1.0], 1000.0, True, monkeypatch)
    assert torch.isfinite(w).all() and w[0] > 0.999


def test_stable_weights_only_for_diff_potential():
    with pytest.raises(ValueError):
        FKD(potential_type=PotentialType.MAX, reward_fn=lambda x: x, device=torch.device("cpu"),
            fk_stable_weights=True)
