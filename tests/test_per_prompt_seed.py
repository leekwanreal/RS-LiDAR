"""`--per_prompt_seed`: off by default, and when on, each prompt's noise depends only on (seed, prompt index).

The sampling loops of `lookahead_sampling.py` and `LiDAR_sampling.py` draw all noise from the global torch
RNG. Without re-seeding, the noise of prompt i depends on how much randomness earlier prompts consumed
(e.g. smoothing perturbations) and on which prompts `--resume` skipped. The tests below replay that loop.
"""
import ast
import os
import unittest

import torch

from rslidar_utils import prompt_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag_default(script, flag):
    tree = ast.parse(open(os.path.join(ROOT, script), encoding="utf-8").read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):
            return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg in ("default", "const")}
    return None


def _run(prompt_ids, per_prompt_seed, extra_draws, skip=(), seed=42):
    """Replay a sampling loop: per prompt, draw the 'initial noise', then `extra_draws` more numbers."""
    torch.manual_seed(seed)
    noise = {}
    for i in prompt_ids:
        if i in skip:  # --resume: completed prompts are skipped without touching the RNG
            continue
        if per_prompt_seed:
            torch.manual_seed(prompt_seed(seed, i))
        noise[i] = torch.randn(4, 8, 8)
        torch.randn(extra_draws)  # e.g. Monte Carlo smoothing perturbations
    return noise


class TestPerPromptSeed(unittest.TestCase):
    def test_flag_is_off_by_default_in_both_entrypoints(self):
        for script in ("lookahead_sampling.py", "LiDAR_sampling.py"):
            self.assertEqual(_flag_default(script, "--per_prompt_seed"), {"default": False, "const": True}, script)

    def test_prompt_seed(self):
        self.assertEqual(prompt_seed(42, 0), 42)
        self.assertEqual(prompt_seed(42, 552), 594)
        self.assertEqual(prompt_seed("100", "7"), 107)

    def test_noise_paired_across_methods_and_resume(self):
        ids = list(range(10))
        lidar = _run(ids, True, extra_draws=0)
        rs = _run(ids, True, extra_draws=4 * 3 * 64 * 64)
        resumed = _run(ids, True, extra_draws=0, skip={0, 1, 2, 5})
        for i in ids:
            self.assertTrue(torch.equal(lidar[i], rs[i]), f"prompt {i}: methods not paired")
        for i in resumed:
            self.assertTrue(torch.equal(lidar[i], resumed[i]), f"prompt {i}: resume changed the noise")

    def test_legacy_global_seed_is_unpaired(self):
        # Documents the behaviour that the flag fixes; the default path is unchanged.
        ids = list(range(5))
        lidar = _run(ids, False, extra_draws=0)
        rs = _run(ids, False, extra_draws=100)
        self.assertTrue(torch.equal(lidar[0], rs[0]))
        self.assertFalse(torch.equal(lidar[1], rs[1]))

    def test_global_seed_matches_legacy_when_flag_off(self):
        torch.manual_seed(42)
        legacy = torch.randn(4, 8, 8)
        self.assertTrue(torch.equal(_run([0], False, extra_draws=0)[0], legacy))


if __name__ == "__main__":
    unittest.main()
