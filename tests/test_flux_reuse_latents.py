import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

import lookahead_sampling

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLUX_LOOKAHEAD = "ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors"


def _mock_flux_pipe():
    pipe = MagicMock()
    pipe.to.return_value = pipe
    pipe.vae_scale_factor = 16
    pipe.vae.config.scaling_factor = 0.3611
    pipe.vae.config.shift_factor = 0.1159

    def unpack(latents, h, w, scale):
        b, _, _ = latents.shape  # like diffusers: fails on already-unpacked 4D latents
        return torch.randn(b, 16, 128, 128)

    pipe._unpack_latents = MagicMock(side_effect=unpack)
    pipe.vae.decode.side_effect = lambda chunk, return_dict=False: (torch.rand(chunk.shape[0], 3, 64, 64) * 2 - 1,)
    pipe.image_processor.postprocess.side_effect = lambda t, output_type="pil": [
        Image.new("RGB", (64, 64)) for _ in range(t.shape[0])
    ]
    pipe.side_effect = lambda *a, **k: MagicMock(images=torch.randn(len(k["prompt"]), 4096, 64))
    return pipe


def _mock_do_eval(prompt, images, metrics_to_compute):
    res = [0.1 * i for i in range(len(images))]
    return {"ImageReward": {"result": res, "mean": sum(res) / len(res), "std": 0.1, "max": max(res), "min": min(res)}}


class TestFluxReuseLatents(unittest.TestCase):
    """RS Phase 1 with --reuse_latents_from must rescore the LiDAR FLUX pool, not regenerate or re-unpack it."""

    def setUp(self):
        self.tmp_dir = os.path.join(SCRIPT_DIR, "tmp_flux_reuse")
        self.out_dir = os.path.join(self.tmp_dir, "Lookahead_samples")
        os.makedirs(self.out_dir, exist_ok=True)
        self.prompt_file = os.path.join(self.tmp_dir, "prompts.jsonl")
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "a cat"}\n{"prompt": "a dog"}\n')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, pipe, extra):
        argv = ["lookahead_sampling.py", "--model_name", FLUX_LOOKAHEAD, "--prompt_path", self.prompt_file,
                "--prompt_indices", "0-1", "--num_particles", "3", "--num_inference_steps", "8", "--seed", "100",
                "--guidance_reward_fn", "ImageReward", "--metrics_to_compute", "ImageReward",
                "--output_dir", self.out_dir] + extra
        flux_cls = MagicMock()
        flux_cls.from_pretrained.return_value = pipe
        with patch("sys.argv", argv), \
             patch("lookahead_sampling.FluxPipeline", flux_cls), \
             patch("lookahead_sampling.hf_hub_download", return_value="dummy.safetensors"), \
             patch("lookahead_sampling.do_eval", side_effect=_mock_do_eval):
            lookahead_sampling.main(lookahead_sampling.get_args())

    def test_rs_rescoring_reuses_lidar_pool(self):
        self._run(_mock_flux_pipe(), ["--run_name", "FLUX_100_3_8"])

        rs_pipe = _mock_flux_pipe()
        self._run(rs_pipe, ["--run_name", "RS", "--use_smoothing", "--sigma", "1.0", "--num_mc_samples", "2",
                            "--noise_mode", "independent_iid",
                            "--reuse_latents_from", os.path.join(self.out_dir, "FLUX_100_3_8")])

        self.assertEqual(rs_pipe.call_count, 0, "reuse must not run the lookahead sampler")
        self.assertEqual(rs_pipe._unpack_latents.call_count, 0, "reused latents are already unpacked")
        for p in ("00000", "00001"):
            lidar = torch.stack(torch.load(os.path.join(self.out_dir, "FLUX_100_3_8", p, "samples", "latent.pt"),
                                           map_location="cpu"))
            rs = torch.stack(torch.load(os.path.join(self.out_dir, "RS", p, "samples", "latent.pt"),
                                        map_location="cpu"))
            self.assertEqual(rs.shape, (3, 16, 128, 128))
            for x in rs:  # same pool (order may change after re-sorting by the smoothed reward)
                self.assertTrue(any(torch.equal(x, y) for y in lidar))


if __name__ == "__main__":
    unittest.main()
