"""CPU tests for the pure parts of the G experiments (perturbation profile, rank stability, ensemble)."""
import unittest

import numpy as np
import torch

from experiments import perturbation_profile as pp
from experiments import rank_stability as rs
from experiments import ensemble_pool as ep


def mean_score(x):
    return x.flatten(1).mean(1).double().numpy()


class TestPerturbationProfile(unittest.TestCase):
    def test_eps0_and_shared_noise(self):
        x_a = torch.full((3, 8, 8), 0.1)
        x_b = torch.full((3, 8, 8), -0.2)
        seen = []

        def score(t):
            seen.append(t.clone())
            return mean_score(t)

        out = pp.profile_pair(x_a, x_b, [0.0, 0.05], 4, torch.Generator().manual_seed(1), score)
        self.assertEqual(len(out[0.0][0]), 1)
        self.assertAlmostEqual(out[0.0][0][0], 0.1, places=5)
        self.assertAlmostEqual(out[0.0][1][0], -0.2, places=5)
        batch = seen[1]                     # eps = 0.05: [A draws, B draws], no clipping at this size
        da, db = batch[:4] - x_a, batch[4:] - x_b
        self.assertTrue(torch.allclose(da, db, atol=1e-6))

    def test_deterministic_with_seed(self):
        x_a, x_b = torch.zeros(3, 4, 4), torch.ones(3, 4, 4) * 0.5
        o1 = pp.profile_pair(x_a, x_b, [0.3], 3, torch.Generator().manual_seed(7), mean_score)
        o2 = pp.profile_pair(x_a, x_b, [0.3], 3, torch.Generator().manual_seed(7), mean_score)
        np.testing.assert_allclose(o1[0.3][0], o2[0.3][0])

    def test_perturb_clips(self):
        x = torch.full((3, 2, 2), 0.95)
        u = torch.ones(2, 3, 2, 2)
        self.assertLessEqual(float(pp.perturb(x, 1.0, u).max()), 1.0)

    def _records(self, gaps):
        recs = []
        for i in range(5):
            prof = {str(float(e)): {"A": [1.0 + g], "B": [1.0]} for e, g in gaps.items()}
            recs.append({"profile": prof, "blur": {"1.0": {"A": 1.0, "B": 0.9}}, "stored_R_A": 1.0 + gaps[0.0],
                         "flip": True})
        return recs

    def test_summary_reading(self):
        eps = [0.0, 0.1, 1.0]
        s = pp.summarize(self._records({0.0: 0.4, 0.1: 0.1, 1.0: -0.1}), eps, [1.0])
        self.assertEqual(s["reading"], "spike-like")
        s = pp.summarize(self._records({0.0: 0.4, 0.1: 0.35, 1.0: -0.1}), eps, [1.0])
        self.assertEqual(s["reading"], "gradual")
        self.assertAlmostEqual(s["sanity_median_abs_diff_R_A_eps0_vs_pool"], 0.0)

    def test_blur(self):
        try:
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest("torchvision missing")
        x = torch.rand(3, 16, 16) * 2 - 1
        out = pp.blur_scores(x, x, [1.0], mean_score)
        self.assertAlmostEqual(out[1.0][0], out[1.0][1])


class TestRankStability(unittest.TestCase):
    def test_kendall(self):
        x = np.arange(10.0)
        self.assertAlmostEqual(rs.kendall_tau(x, x), 1.0)
        self.assertAlmostEqual(rs.kendall_tau(x, -x), -1.0)

    def test_separated_scores_are_stable(self):
        rng = np.random.default_rng(0)
        S = np.arange(10.0)[:, None] * 10 + rng.normal(scale=0.01, size=(10, 64))
        p = rs.analyze_prompt(S, 4, b_index=9)
        self.assertEqual(p["top_all"], 9)
        self.assertEqual(p["agree_top1"], 1.0)
        self.assertEqual(p["in_top3"], 1.0)
        self.assertTrue(p["b_is_top_all"])
        self.assertEqual(p["p_swap_top2"], 0.0)

    def test_tied_scores_are_unstable(self):
        rng = np.random.default_rng(1)
        S = rng.normal(size=(20, 64))
        p = rs.analyze_prompt(S, 4)
        self.assertLess(p["agree_top1"], 1.0)

    def test_analyze_aggregate(self):
        rng = np.random.default_rng(2)
        scores = {i: np.arange(5.0)[:, None] + rng.normal(scale=0.01, size=(5, 64)) for i in range(3)}
        summ, per = rs.analyze(scores, 4, {0: 4, 1: 4, 2: 0})
        self.assertEqual(summ["n_prompts"], 3)
        self.assertAlmostEqual(summ["frac_B_is_top_all"], 2 / 3)

    def test_score_candidates_shape(self):
        x = torch.zeros(3, 3, 4, 4)
        S = rs.score_candidates(x, 1.0, 8, torch.Generator().manual_seed(0), mean_score)
        self.assertEqual(S.shape, (3, 8))


class TestEnsemble(unittest.TestCase):
    def test_zscore_mean_zero_and_agreement(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        e = ep.zscore_ensemble({"ImageReward": a, "HPS": a * 0.01 + 0.2, "CLIP": a * 5})
        self.assertAlmostEqual(float(e.mean()), 0.0, places=10)
        self.assertEqual(int(np.argmax(e)), 3)

    def test_constant_metric(self):
        e = ep.zscore_ensemble({"x": np.ones(4), "y": np.array([0.0, 1.0, 0.0, 0.0])})
        self.assertEqual(int(np.argmax(e)), 1)
        self.assertTrue(np.isfinite(e).all())


if __name__ == "__main__":
    unittest.main()
