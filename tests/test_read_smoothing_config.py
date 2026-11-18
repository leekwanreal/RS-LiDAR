"""read_smoothing_config: one normalized view of the smoothing metadata written by every code version."""
from rslidar_utils import read_smoothing_config


def test_pre_2026_10_27_iid():
    # Form written before noise_mode / beta_risk existed (e.g. the Table 1 RS pool on Drive).
    res = {"smoothing": {"enabled": True, "sigma": 1.0, "num_mc_samples": 4, "domain": "image",
                         "mode": "loop", "use_crn": False, "reused_latents_from": None}}
    sm = read_smoothing_config(res)
    assert sm == {"sigma": 1.0, "num_mc_samples": 4, "domain": "image", "noise_mode": "independent_iid",
                  "beta_risk": 0.0, "legacy_metadata": True}


def test_pre_2026_10_27_crn_and_latent_domain():
    res = {"smoothing": {"sigma": 0.5, "num_mc_samples": 8, "domain": "latent", "use_crn": True}}
    sm = read_smoothing_config(res)
    assert sm["noise_mode"] == "coupled_crn"
    assert sm["domain"] == "latent"
    assert sm["beta_risk"] == 0.0 and sm["legacy_metadata"] is True


def test_2026_10_27_to_11_09_antithetic_default_is_kept():
    # This period defaulted to antithetic + entropic risk; the metadata says so and must not be overridden.
    res = {"smoothing": {"sigma": 1.0, "num_mc_samples": 4, "domain": "image",
                         "noise_mode": "independent_antithetic", "beta_risk": 0.5, "use_crn": False}}
    sm = read_smoothing_config(res)
    assert sm["noise_mode"] == "independent_antithetic"
    assert sm["beta_risk"] == 0.5
    assert sm["legacy_metadata"] is False


def test_current_form():
    res = {"smoothing": {"enabled": True, "sigma": 1.0, "num_mc_samples": 4, "domain": "image",
                         "mode": "loop", "noise_mode": "independent_iid", "beta_risk": 0.0,
                         "use_crn": False, "reused_latents_from": "x/latent.pt"}}
    sm = read_smoothing_config(res)
    assert sm == {"sigma": 1.0, "num_mc_samples": 4, "domain": "image", "noise_mode": "independent_iid",
                  "beta_risk": 0.0, "legacy_metadata": False}


def test_missing_or_incomplete_metadata():
    assert read_smoothing_config({}) == {}
    assert read_smoothing_config(None) == {}
    assert read_smoothing_config({"smoothing": None}) == {}
    assert read_smoothing_config({"smoothing": {"sigma": 1.0}}) == {}
