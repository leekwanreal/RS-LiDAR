import unittest
import torch

from fkd_diffusers.fkd_pipeline_flux import FluxPipeline_LiDAR
from rs_flow_lidar import RSFlowLiDARGuidance


class MockRagDataset:
    def __init__(self, K=4, C=16, H=128, W=128):
        self.data = {
            0: {
                "latents": torch.randn((K, C, H, W)),
                "rewards": torch.tensor([1.0, 0.5, 3.2, 0.8]),
            }
        }


class MockTransformer(torch.nn.Module):
    """Fake transformer: returns zero velocity with the shape of hidden_states (the pipeline requires a transformer)."""
    def forward(self, hidden_states, **kwargs):
        return (torch.zeros_like(hidden_states),)


MOCK_EMBEDS = dict(prompt_embeds=torch.zeros((1, 8, 16)), pooled_prompt_embeds=torch.zeros((1, 8)))


class TestFluxPipelineLiDARDataFlow(unittest.TestCase):
    """
    End-to-end data-flow tests for FluxPipeline_LiDAR:
    1. Accepts a 16-channel 128x128 lookahead pool.
    2. FLUX guidance window [0.95, 1.0] vs the default window [0.20, 0.98].
    3. The velocity correction delta_v is added to v_pred.
    4. Euler steps go backward in time (dt > 0).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.pipe = FluxPipeline_LiDAR()
        self.pipe.transformer = MockTransformer()
        self.K = 4
        self.C = 16
        self.H = 128
        self.W = 128
        self.rag_dataset = MockRagDataset(self.K, self.C, self.H, self.W)

    def test_01_guidance_active_in_table3_window(self):
        """For t in [0.95, 1.0] the guidance is active (FLUX window)."""
        fkd_args = {
            "use_rag": True,
            "rag_dataset": self.rag_dataset,
            "top_k": self.K,
            "s_flow": 40.0,
            "lmbda": 5000.0,
            "guidance_start": 1.0,
            "guidance_end": 0.95,
        }
        
        # One Euler step at t=0.98 (inside [0.95, 1.0])
        timesteps = [1.0, 0.96]  # dt = 0.04 > 0
        latents_init = torch.randn((1, self.C, self.H, self.W))
        
        out = self.pipe(
            prompt="A cinematic photo of an astronaut on Mars",
            prompt_idx=0,
            height=1024,
            width=1024,
            num_inference_steps=2,
            timesteps=timesteps,
            latents=latents_init.clone(),
            fkd_args=fkd_args,
            **MOCK_EMBEDS,
        )
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out, "images"))

    def test_02_guidance_inactive_outside_window(self):
        """For t < 0.95, delta_v = 0 and the ODE is unguided."""
        guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0, t_start=1.0, t_end=0.95)
        z_t = torch.randn((1, 16, 128, 128))
        lookahead_z0 = torch.randn((4, 16, 128, 128))
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # t = 0.80 (outside [0.95, 1.0])
        delta_v = guider.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.80)
        self.assertEqual(torch.count_nonzero(delta_v).item(), 0, "Outside [0.95, 1.0], delta_v must be exactly 0.")

        # t = 0.98 (inside [0.95, 1.0])
        delta_v_active = guider.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.98)
        self.assertGreater(torch.count_nonzero(delta_v_active).item(), 0, "Inside [0.95, 1.0], delta_v must be non-zero.")

    def test_03_attraction_toward_best_candidate(self):
        """-delta_v has a positive inner product with the highest-reward lookahead sample."""
        guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0, t_start=1.0, t_end=0.95)
        z_t = torch.randn((1, 16, 128, 128))
        lookahead_z0 = torch.randn((4, 16, 128, 128))
        # Sample 2 has the highest reward
        rewards = torch.tensor([0.0, 0.0, 15.0, 0.0])

        delta_v = guider.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.98)
        pull_force = -delta_v[0]
        best_candidate = lookahead_z0[2]
        dot = torch.sum(pull_force * best_candidate).item()
        self.assertGreater(dot, 0, "The Euler step must pull towards the highest-reward candidate.")

    def test_04_integer_timestep_safety(self):
        """Diffusion-style window values (resampling_t_start=20, resampling_t_end=200) are clamped safely."""
        fkd_args = {
            "use_rag": True,
            "rag_dataset": self.rag_dataset,
            "top_k": self.K,
            "s_flow": 40.0,
            "lmbda": 5000.0,
            "resampling_t_start": 20,   # diffusion integer format
            "resampling_t_end": 200,   # diffusion integer format
        }
        timesteps = [1.0, 0.96]
        latents_init = torch.randn((1, self.C, self.H, self.W))
        
        out = self.pipe(
            prompt="Test prompt",
            prompt_idx=0,
            height=1024,
            width=1024,
            num_inference_steps=2,
            timesteps=timesteps,
            latents=latents_init.clone(),
            fkd_args=fkd_args,
            **MOCK_EMBEDS,
        )
        self.assertIsNotNone(out)
        self.assertEqual(self.pipe.flow_guidance_module.t_start, 1.0)
        self.assertEqual(self.pipe.flow_guidance_module.t_end, 0.95)

    def test_05_bf16_inputs_use_fp32_potential(self):
        """bf16 latents must give the same w_r - w as fp32 (bf16 would round ||.||^2 ~ 2.4e5 to one value)."""
        guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0, t_start=1.0, t_end=0.95)
        torch.manual_seed(0)
        t = 0.964
        z0 = torch.randn((3, 16, 128, 128)) * 0.8
        z_t = (1 - t) * torch.randn((1, 16, 128, 128)) * 0.8 + t * torch.randn((1, 16, 128, 128))
        rewards = torch.tensor([0.30, 0.31, 0.29])
        dv32, d32 = guider.compute_velocity_guidance(z_t, z0, rewards, sigma_t=t, return_diagnostics=True)
        dv16, d16 = guider.compute_velocity_guidance(z_t.bfloat16(), z0.bfloat16(), rewards, sigma_t=t, return_diagnostics=True)
        self.assertTrue(torch.allclose(d32["w_diff"], d16["w_diff"], atol=1e-2), f"fp32 {d32['w_diff']} vs bf16 {d16['w_diff']}")
        self.assertEqual(dv16.dtype, torch.bfloat16, "delta_v must have the dtype of z_t.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

