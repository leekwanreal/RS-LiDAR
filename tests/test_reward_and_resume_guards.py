"""Guards added after the multi-reward audit.

- HPS / CLIP must raise instead of silently scoring 0 when their model is unavailable.
- A cached smoothed Phase-1 result is invalid when beta_risk or the reused candidate pool differ.
"""
import types
import unittest

import lookahead_sampling as ls
from fkd_diffusers import rewards


def _args(**kw):
    base = dict(num_mc_samples=4, noise_mode="independent_iid", sigma=1.0, beta_risk=0.0,
                reuse_latents_from=None, no_reuse_latents=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _cached(**kw):
    s = dict(num_mc_samples=4, noise_mode="independent_iid", sigma=1.0, beta_risk=0.0, reused_latents_from=None)
    s.update(kw)
    return {"smoothing": s}


POOL = "Lookahead_SD15_Step5_n50_seed100"


class TestRewardGuards(unittest.TestCase):
    def test_hps_raises_without_hpsv2(self):
        saved = rewards.hpsv2
        try:
            rewards.hpsv2 = None
            with self.assertRaises(ImportError):
                rewards.do_human_preference_score(images=[object()], prompts=["p"])
        finally:
            rewards.hpsv2 = saved

    def test_clip_raises_without_clip(self):
        saved = rewards.clip
        try:
            rewards.clip = None
            with self.assertRaises(ImportError):
                rewards.do_clip_score(images=[object()], prompts=["p"])
        finally:
            rewards.clip = saved


class TestCacheMismatches(unittest.TestCase):
    def test_matching_cache_is_reused(self):
        self.assertEqual(ls._cache_mismatches(_cached(), _args()), [])

    def test_beta_risk_mismatch_detected_when_current_is_zero(self):
        self.assertTrue(ls._cache_mismatches(_cached(beta_risk=0.5), _args(beta_risk=0.0)))

    def test_same_reused_pool_is_accepted(self):
        cached = _cached(reused_latents_from=f"/content/drive/MyDrive/RS-LiDAR/Lookahead_samples/{POOL}/00003/samples/latent.pt")
        args = _args(reuse_latents_from=f"/content/RS-LiDAR/Lookahead_samples/{POOL}")
        self.assertEqual(ls._cache_mismatches(cached, args), [])

    def test_fresh_or_other_pool_is_rejected(self):
        args = _args(reuse_latents_from=POOL)
        self.assertTrue(ls._cache_mismatches(_cached(reused_latents_from=None), args))
        other = "Lookahead_samples/OtherPool/00003/samples/latent.pt"
        self.assertTrue(ls._cache_mismatches(_cached(reused_latents_from=other), args))

    def test_no_pool_check_without_reuse(self):
        cached = _cached(reused_latents_from=f"Lookahead_samples/{POOL}/00003/samples/latent.pt")
        self.assertEqual(ls._cache_mismatches(cached, _args()), [])


if __name__ == "__main__":
    unittest.main()
