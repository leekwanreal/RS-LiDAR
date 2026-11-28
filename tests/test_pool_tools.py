"""CPU tests for experiments/pool_tools.py (no model download)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from experiments import pool_tools as pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_pool(root, name, idx, latents, rewards, smoothing=None):
    d = os.path.join(root, name, f"{idx:05d}")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    torch.save([latents[k] for k in range(latents.shape[0])], os.path.join(d, "samples", "latent.pt"))
    res = {"ImageReward": pt.summarize_values(rewards), "prompt": ["a photo of a cat"] * latents.shape[0],
           "prompt_index": idx}
    if smoothing:
        res["smoothing"] = smoothing
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(res, f)
    return os.path.join(root, name)


class TestPoolTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pooltools_")
        g = torch.Generator().manual_seed(0)
        self.K = 6
        self.lat = torch.randn(self.K, 4, 8, 8, generator=g).half()
        self.r = np.sort(np.random.default_rng(0).normal(size=self.K))[::-1].copy()
        self.raw = make_pool(self.tmp, "raw", 3, self.lat, self.r)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_and_list(self):
        self.assertEqual(pt.list_prompt_indices(self.raw), [3])
        lat, r = pt.load_pool(self.raw, 3)
        self.assertTrue(torch.equal(lat, self.lat))
        np.testing.assert_allclose(r, self.r)

    def test_match_candidates_recovers_permutation(self):
        perm_true = np.array([2, 0, 5, 1, 4, 3])
        other = self.lat[perm_true]
        perm = pt.match_candidates(self.lat, other)
        np.testing.assert_array_equal(perm, perm_true)
        vals_other = np.arange(self.K, dtype=float) * 10
        back = pt.reorder_to_ref(vals_other, perm)
        for j in range(self.K):
            self.assertEqual(back[perm[j]], vals_other[j])

    def test_match_candidates_rejects_different_latents(self):
        other = self.lat.clone()
        other[0] = other[0] + 5
        self.assertIsNone(pt.match_candidates(self.lat, other))

    def test_derived_pool_readable_by_phase2_loader(self):
        from lookahead_datasets import gen_lookahead_samples
        new = np.arange(self.K, dtype=float)
        dst = os.path.join(self.tmp, "derived")
        pt.write_derived_pool(self.raw, dst, 3, new, meta={"m": 1}, link=False)
        lat, r = pt.load_pool(dst, 3)
        self.assertTrue(torch.equal(lat, self.lat))
        np.testing.assert_allclose(r, new)
        ds = gen_lookahead_samples(dst, top_k=self.K)
        item = ds.data[3]
        for k in range(self.K):
            j = int(torch.where((self.lat.float() - item["latents"][k].float()).flatten(1).norm(dim=1) == 0)[0][0])
            self.assertAlmostEqual(float(item["rewards"][k]), new[j])
        with open(os.path.join(dst, "00003", "results.json")) as f:
            res = json.load(f)
        self.assertNotIn("smoothing", res)
        self.assertEqual(res["derived"], {"m": 1})
        self.assertIn("ImageReward", res["source_results"])

    def test_derived_pool_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            pt.write_derived_pool(self.raw, os.path.join(self.tmp, "bad"), 3, np.zeros(self.K + 1), link=False)

    def test_flip_pairs(self):
        perm_true = np.array([1, 0, 2, 3, 4, 5])
        s_raw_order = np.array([0.1, 0.9, 0.2, 0.3, 0.0, -0.1])   # smoothed reward in raw order: top is 1
        rs_lat = self.lat[perm_true]
        rs_r = s_raw_order[perm_true]
        rs = make_pool(self.tmp, "rs", 3, rs_lat, rs_r, smoothing={"sigma": 1.0, "num_mc_samples": 4})
        pairs, info = pt.flip_pairs(self.raw, rs, [3])
        a, b, r_raw, s = pairs[3]
        self.assertEqual(a, int(np.argmax(self.r)))
        self.assertEqual(b, 1)
        np.testing.assert_allclose(s, s_raw_order)
        self.assertEqual(info["unmatched"], [])

    def test_flip_pairs_same_result_with_threads_and_unmatched_kept(self):
        g = torch.Generator().manual_seed(1)
        for idx in (4, 5, 6):
            lat = torch.randn(self.K, 4, 8, 8, generator=g).half()
            make_pool(self.tmp, "raw", idx, lat, self.r)
            other = lat.clone()
            if idx == 5:
                other[0] = other[0] + 5  # different latents -> unmatched
            make_pool(self.tmp, "rs", idx, other[[1, 0, 2, 3, 4, 5]], self.r[::-1].copy())
        make_pool(self.tmp, "rs", 3, self.lat, self.r)
        idxs = [3, 4, 5, 6]
        p1, i1 = pt.flip_pairs(self.raw, os.path.join(self.tmp, "rs"), idxs, workers=1)
        p8, i8 = pt.flip_pairs(self.raw, os.path.join(self.tmp, "rs"), idxs, workers=8, log_every=2)
        self.assertEqual(list(p1), list(p8))
        self.assertEqual(list(p1), [3, 4, 6])
        self.assertEqual(i1, i8)
        self.assertEqual(i8["unmatched"], [5])
        for k in p1:
            self.assertEqual(p1[k][:2], p8[k][:2])
            np.testing.assert_array_equal(p1[k][3], p8[k][3])

    def test_stage_local_copies_resumes_and_reports_missing(self):
        make_pool(self.tmp, "raw", 7, self.lat, self.r)
        os.remove(os.path.join(self.raw, "00007", "samples", "latent.pt"))  # missing source file
        for name in ("00000.png", "00001.png"):
            with open(os.path.join(self.raw, "00003", "samples", name), "wb") as f:
                f.write(b"png" + name.encode())
        dst = os.path.join(self.tmp, "local", "raw")
        out = pt.stage_local(self.raw, dst, rel_globs=("samples/*.png",), workers=4)
        self.assertEqual(out, dst)
        lat, r = pt.load_pool(dst, 3)
        self.assertTrue(torch.equal(lat, self.lat))
        np.testing.assert_allclose(r, self.r)
        with open(os.path.join(dst, "00003", "samples", "00001.png"), "rb") as f:
            self.assertEqual(f.read(), b"png00001.png")
        self.assertTrue(os.path.exists(os.path.join(dst, "00007", "results.json")))
        self.assertFalse(os.path.exists(os.path.join(dst, "00007", "samples", "latent.pt")))
        self.assertEqual(pt.list_prompt_indices(dst), [3])
        leftovers = [f for _, _, fs in os.walk(dst) for f in fs if ".part" in f]
        self.assertEqual(leftovers, [])
        # resume: an existing complete file is kept, not copied again
        marker = os.path.join(dst, "00003", "results.json")
        with open(marker, "w") as f:
            f.write("{}")
        pt.stage_local(self.raw, dst, indices=[3], workers=2)
        with open(marker) as f:
            self.assertEqual(f.read(), "{}")

    def test_pil_roundtrip(self):
        x = torch.rand(2, 3, 16, 16) * 2 - 1
        y = pt.pil_to_tensor(pt.to_pil(x))
        self.assertLessEqual(float((x - y).abs().max()), 2 / 255 + 1e-6)


class TestCLIHelp(unittest.TestCase):
    def test_help_of_new_modules(self):
        for mod in ("experiments.perturbation_profile", "experiments.copy_distance",
                    "experiments.rank_stability", "experiments.ensemble_pool"):
            out = subprocess.run([sys.executable, "-m", mod, "--help"], cwd=ROOT, capture_output=True, text=True,
                                 timeout=120)
            self.assertEqual(out.returncode, 0, msg=f"{mod}: {out.stderr[-500:]}")
            self.assertIn("--out_dir" if mod != "experiments.ensemble_pool" else "--dst_pool", out.stdout)


if __name__ == "__main__":
    unittest.main()
