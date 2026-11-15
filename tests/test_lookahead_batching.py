import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

import lookahead_sampling
from lookahead_sampling import sample_lookahead_latents_batched

FLUX_LOOKAHEAD = "ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors"


def _pipe():
    pipe = MagicMock()
    pipe.side_effect = lambda *a, **k: MagicMock(images=torch.randn(len(k["prompt"]), 4096, 64))
    return pipe


class TestLookaheadBatching(unittest.TestCase):
    def test_default_is_single_call(self):
        for bs in (0, -1, 50, 64):
            pipe = _pipe()
            out = sample_lookahead_latents_batched(pipe, FLUX_LOOKAHEAD, ["p"] * 50, 8, bs)
            self.assertEqual(pipe.call_count, 1)
            self.assertEqual(out.shape, (50, 4096, 64))

    def test_chunks_cover_all_particles_in_order(self):
        pipe = _pipe()
        out = sample_lookahead_latents_batched(pipe, FLUX_LOOKAHEAD, ["p"] * 12, 8, 5)
        self.assertEqual([len(c.kwargs["prompt"]) for c in pipe.call_args_list], [5, 5, 2])
        self.assertEqual(out.shape, (12, 4096, 64))

    def test_same_draws_as_sequential_calls(self):
        torch.manual_seed(0)
        out = sample_lookahead_latents_batched(_pipe(), FLUX_LOOKAHEAD, ["p"] * 7, 8, 3)
        torch.manual_seed(0)
        ref = torch.cat([torch.randn(n, 4096, 64) for n in (3, 3, 1)])
        self.assertTrue(torch.equal(out, ref))

    def test_cli_flag_default_zero(self):
        with patch.object(sys, "argv", ["lookahead_sampling.py"]):
            self.assertEqual(lookahead_sampling.get_args().lookahead_batch_size, 0)
        with patch.object(sys, "argv", ["lookahead_sampling.py", "--lookahead_batch_size", "5"]):
            self.assertEqual(lookahead_sampling.get_args().lookahead_batch_size, 5)


if __name__ == "__main__":
    unittest.main()
