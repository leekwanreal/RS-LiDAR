import unittest
from unittest.mock import MagicMock
import torch

from fkd_diffusers.fkd_pipeline_flux import FluxPipeline_LiDAR
from diffusers import FlowMatchEulerDiscreteScheduler


class TestFluxPipelineBugfixes(unittest.TestCase):
    """
    Regression tests for FluxPipeline_LiDAR:
    1. Timestep scaling at the last step (t = 1.0): t_scaled must be 0.001, not 1.0.
    2. t_norm at the last step (t = 1.0) must be 0.001, outside [0.95, 1.0].
    3. scheduler.set_begin_index(0) is called before the loop.
    4. vae_scale_factor defaults to 8 (16 would corrupt the spatial latents).
    5. Fallback Euler dt: scaled_dt = dt / 1000.0 khi is_discrete_1000.
    6. With s_flow = 0.0 the trajectory is unchanged (delta_v = 0).
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_01_timestep_scaling_and_guidance_at_final_step(self):
        """
        With 28 FlowMatchEulerDiscreteScheduler steps, the last timestep is t=1.0.
        The transformer must receive t_scaled = 0.001 (1.0 / 1000.0), not 1.0,
        and t_norm = 0.001 at the last step, so no guidance is applied.
        """
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = FlowMatchEulerDiscreteScheduler()
        # FLUX timestep schedule with dynamic shifting, ending at 1.0
        pipe.scheduler.timesteps = torch.linspace(1000.0, 1.0, 28)
        self.assertEqual(pipe.scheduler.timesteps[-1].item(), 1.0)

        # Mock transformer recording every t_scaled it receives
        captured_timesteps = []
        captured_guidance = []

        trans_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        mock_transformer = MagicMock()
        mock_transformer.config.guidance_embeds = True
        mock_transformer.parameters.return_value = [trans_param]

        def transformer_forward(**kwargs):
            t_input = kwargs.get("timestep")
            captured_timesteps.append(t_input.item() if isinstance(t_input, torch.Tensor) else float(t_input))
            captured_guidance.append(kwargs.get("guidance"))
            # Zero velocity
            h = kwargs.get("hidden_states")
            return (torch.zeros_like(h),)

        mock_transformer.side_effect = transformer_forward
        pipe.transformer = mock_transformer

        # Mock guidance module to check whether guidance is called at the last step
        mock_guidance_module = MagicMock()
        mock_guidance_module.compute_velocity_guidance.return_value = torch.zeros(1, 16, 128, 128)
        pipe.flow_guidance_module = mock_guidance_module

        # Mock prompt encode
        pipe.encode_prompt = MagicMock(return_value=(
            torch.randn(1, 512, 64),  # prompt_embeds
            torch.randn(1, 768),       # pooled_prompt_embeds
            torch.zeros(512, 3),       # text_ids
        ))

        # Mock VAE
        pipe.vae = MagicMock()
        pipe.vae.config.scaling_factor = 0.3611
        pipe.vae.config.shift_factor = 0.1159
        pipe.vae.config.block_out_channels = [128, 256, 512, 512]
        pipe.vae.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        pipe.vae.decode.return_value = (torch.zeros(1, 3, 224, 224),)

        mock_rag_dataset = MagicMock()
        mock_rag_dataset.data = {
            0: {
                "latents": torch.randn(3, 16, 128, 128),
                "rewards": torch.tensor([1.0, 2.0, 3.0]),
            }
        }

        fkd_args = {
            "use_rag": True,
            "rag_dataset": mock_rag_dataset,
            "top_k": 3,
            "s_flow": 40.0,
            "lmbda": 5000.0,
            "guidance_start": 1.0,
            "guidance_end": 0.95,
        }

        target_timesteps = torch.linspace(1000.0, 1.0, 28)
        self.assertEqual(target_timesteps[-1].item(), 1.0)

        pipe(
            prompt="A wooden bench in a park",
            prompt_idx=0,
            height=1024,
            width=1024,
            num_inference_steps=28,
            timesteps=target_timesteps,
            output_type="latent",
            fkd_args=fkd_args,
        )

        # 1. All 28 steps are run
        self.assertEqual(len(captured_timesteps), 28)

        # 2. First step (t = 1000.0) -> t_scaled = 1.0
        self.assertAlmostEqual(captured_timesteps[0], 1.0, places=4)

        # 3. Last step (t = 1.0) -> t_scaled = 0.001 (1.0 / 1000.0)
        self.assertAlmostEqual(captured_timesteps[-1], 0.001, places=4,
                               msg=f"Last step received t_scaled={captured_timesteps[-1]} instead of 0.001")

        # 4. The guidance module is not called at the last step (t_norm = 0.001 < 0.95);
        # it is only called inside [0.95, 1.0], i.e. at the first steps
        self.assertGreater(len(mock_guidance_module.compute_velocity_guidance.call_args_list), 0)
        for call_args in mock_guidance_module.compute_velocity_guidance.call_args_list:
            sigma_t_called = call_args.kwargs.get("sigma_t")
            self.assertGreaterEqual(sigma_t_called, 0.95,
                                    msg=f"Guidance must not be called for sigma_t < 0.95, got {sigma_t_called}")
            self.assertLessEqual(sigma_t_called, 1.0,
                                 msg=f"Guidance must not be called for sigma_t > 1.0, got {sigma_t_called}")

    def test_02_set_begin_index_called(self):
        """scheduler.set_begin_index(0) is called before the loop, as in diffusers."""
        pipe = FluxPipeline_LiDAR()
        mock_sched = MagicMock()
        mock_sched.order = 1
        mock_sched.timesteps = torch.tensor([1000.0, 500.0])
        mock_sched.step.side_effect = lambda v, t, l, return_dict=False: (l,)
        pipe.scheduler = mock_sched

        # Minimal mock transformer & prompt
        pipe.transformer = MagicMock()
        pipe.transformer.config.guidance_embeds = False
        pipe.transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        pipe.transformer.side_effect = lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
        pipe.encode_prompt = MagicMock(return_value=(torch.zeros(1, 10, 64), torch.zeros(1, 10), torch.zeros(10, 3)))

        fkd_args = {"use_rag": True, "top_k": 1, "s_flow": 0.0}
        pipe(prompt="test", prompt_idx=0, num_inference_steps=2, fkd_args=fkd_args, output_type="latent")

        mock_sched.set_begin_index.assert_called_once_with(0)

    def test_03_vae_scale_factor_default(self):
        """vae_scale_factor defaults to 8 (FLUX), not 16."""
        pipe = FluxPipeline_LiDAR()
        self.assertEqual(pipe.vae_scale_factor, 8)

    def test_04_euler_fallback_scaled_dt(self):
        """The plain Euler fallback uses scaled_dt = dt / 1000.0 for timesteps in [1, 1000]."""
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = None  # force the plain Euler fallback

        pipe.transformer = MagicMock()
        pipe.transformer.config.guidance_embeds = False
        pipe.transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        # Constant model velocity v_pred = 1.0
        pipe.transformer.side_effect = lambda **kwargs: (torch.ones_like(kwargs["hidden_states"]),)
        pipe.encode_prompt = MagicMock(return_value=(torch.zeros(1, 10, 64), torch.zeros(1, 10), torch.zeros(10, 3)))

        # Two timesteps 1000.0 and 900.0 -> dt = 100.0 on the [1, 1000] scale;
        # scaled_dt must be 100.0 / 1000.0 = 0.1, not 100.0.
        timesteps = torch.tensor([1000.0, 900.0])
        init_latents = torch.zeros(1, 4096, 64)

        fkd_args = {"use_rag": True, "top_k": 1, "s_flow": 0.0}
        out = pipe(
            prompt="test",
            prompt_idx=0,
            timesteps=timesteps,
            latents=init_latents.clone(),
            fkd_args=fkd_args,
            output_type="latent",
        )

        out_latent = out.images
        # latents = init_latents - sum(scaled_dt * v_pred) = 0 - (0.1 + 0.9) * 1.0 = -1.0
        self.assertAlmostEqual(out_latent.mean().item(), -1.0, places=3,
                               msg=f"Euler fallback dt was not divided by 1000: mean={out_latent.mean().item()}")

    def test_05_s_flow_zero_clean_trajectory(self):
        """With s_flow = 0.0 the trajectory is not affected by the lookahead candidates."""
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = FlowMatchEulerDiscreteScheduler()
        pipe.scheduler.set_timesteps(4)

        # Mock transformer
        pipe.transformer = MagicMock()
        pipe.transformer.config.guidance_embeds = False
        pipe.transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        pipe.transformer.side_effect = lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
        pipe.encode_prompt = MagicMock(return_value=(torch.zeros(1, 10, 64), torch.zeros(1, 10), torch.zeros(10, 3)))

        mock_rag_dataset = MagicMock()
        mock_rag_dataset.data = {
            0: {
                "latents": torch.ones(3, 16, 128, 128) * 100.0,  # Giant outlier latents
                "rewards": torch.tensor([10.0, 20.0, 30.0]),
            }
        }

        fkd_args = {
            "use_rag": True,
            "rag_dataset": mock_rag_dataset,
            "top_k": 3,
            "s_flow": 0.0,  # Zero guidance scale
            "lmbda": 5000.0,
            "guidance_start": 1.0,
            "guidance_end": 0.95,
        }

        init_lat = torch.zeros(1, 4096, 64)
        out = pipe(
            prompt="test",
            prompt_idx=0,
            num_inference_steps=4,
            latents=init_lat.clone(),
            fkd_args=fkd_args,
            output_type="latent",
        )

        # Zero transformer output and s_flow = 0.0: latents must stay 0
        self.assertTrue(torch.allclose(out.images, init_lat, atol=1e-5),
                        "With s_flow = 0.0 the latents must not be changed by the lookahead pool.")

    def test_06_default_scheduler_timesteps_and_current_timestep_reset(self):
        """With timesteps=None, retrieve_timesteps works without warnings and _current_timestep is reset."""
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = FlowMatchEulerDiscreteScheduler()

        captured_timesteps = []
        trans_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        mock_transformer = MagicMock()
        mock_transformer.config.guidance_embeds = True
        mock_transformer.parameters.return_value = [trans_param]
        def transformer_forward(**kwargs):
            t_input = kwargs.get("timestep")
            captured_timesteps.append(t_input.item() if isinstance(t_input, torch.Tensor) else float(t_input))
            return (torch.zeros_like(kwargs.get("hidden_states")),)
        mock_transformer.side_effect = transformer_forward
        pipe.transformer = mock_transformer

        pipe.encode_prompt = MagicMock(return_value=(
            torch.randn(1, 512, 64),
            torch.randn(1, 768),
            torch.zeros(512, 3),
        ))

        fkd_args = {
            "use_rag": True,
            "top_k": 1,
            "s_flow": 0.0,
            "guidance_start": 1.0,
            "guidance_end": 0.95,
        }

        pipe(
            prompt="A wooden bench in a park",
            prompt_idx=0,
            height=1024,
            width=1024,
            num_inference_steps=28,
            timesteps=None,  # generated by FlowMatchEulerDiscreteScheduler
            output_type="latent",
            fkd_args=fkd_args,
        )

        self.assertEqual(len(captured_timesteps), 28)
        self.assertAlmostEqual(captured_timesteps[0], 1.0, places=4)
        # The last step must be < 0.05 (0.001 or 0.0357), not 1.0
        self.assertLess(captured_timesteps[-1], 0.05)
        # _current_timestep is reset to None
        self.assertIsNone(pipe._current_timestep)

    def test_07_multiple_prompts_scheduler_step_index_reset(self):
        """Consecutive prompts do not raise IndexError in the scheduler (step_index is reset)."""
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True)

        pipe.transformer = MagicMock()
        pipe.transformer.config.guidance_embeds = False
        pipe.transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        pipe.transformer.side_effect = lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
        pipe.encode_prompt = MagicMock(return_value=(torch.zeros(1, 10, 64), torch.zeros(1, 10), torch.zeros(10, 3)))

        fkd_args = {"use_rag": True, "top_k": 1, "s_flow": 0.0}

        # Prompt 0
        out0 = pipe(prompt="prompt 0", prompt_idx=0, num_inference_steps=28, fkd_args=fkd_args, output_type="latent")
        self.assertIsNotNone(out0)

        # Second prompt: without the reset, step_index would be 28 and sched.step would raise IndexError
        out1 = pipe(prompt="prompt 1", prompt_idx=1, num_inference_steps=28, fkd_args=fkd_args, output_type="latent")
        self.assertIsNotNone(out1)
        self.assertEqual(pipe.scheduler.step_index, 28)

    def test_08_vanilla_call_does_not_pass_timesteps_to_super(self):
        """Vanilla mode (use_rag=False) does not pass timesteps to FluxPipeline.__call__."""
        from diffusers import FluxPipeline
        pipe = FluxPipeline_LiDAR()

        with unittest.mock.patch.object(FluxPipeline, "__call__", autospec=True, return_value=MagicMock()) as mock_call:
            pipe(prompt="a bench", num_inference_steps=28, timesteps=[1000.0, 500.0], fkd_args={"use_rag": False})

        self.assertTrue(mock_call.called)
        called_kwargs = mock_call.call_args[1]
        self.assertNotIn("timesteps", called_kwargs, "timesteps must not be passed to super().__call__.")
        self.assertIn("prompt", called_kwargs)
        self.assertEqual(called_kwargs["num_inference_steps"], 28)

    def test_09_device_property_and_execution_device_resolution(self):
        """pipe.device and pipe._execution_device resolve to the device of the transformer / sub-modules."""
        pipe = FluxPipeline_LiDAR()
        mock_param = torch.nn.Parameter(torch.zeros(2, 2))
        pipe.transformer = torch.nn.Linear(2, 2)
        pipe.transformer.weight = mock_param

        # Initially on CPU
        self.assertEqual(pipe.device.type, "cpu")
        self.assertEqual(pipe._execution_device.type, "cpu")

        # After pinning _execution_device_cache
        pipe._execution_device_cache = torch.device("cuda:0")
        self.assertEqual(pipe.device, torch.device("cuda:0"))
        self.assertEqual(pipe._execution_device, torch.device("cuda:0"))

    def test_10_vanilla_call_preserves_device_alignment(self):
        """In vanilla mode, latents and prompt_embeds are moved to the pipeline device before calling super()."""
        from diffusers import FluxPipeline
        pipe = FluxPipeline_LiDAR()
        pipe._execution_device_cache = torch.device("cpu")

        test_latents = torch.zeros(1, 16, 128, 128)
        with unittest.mock.patch.object(FluxPipeline, "__call__", autospec=True, return_value=MagicMock()) as mock_call:
            pipe(prompt="a bench", latents=test_latents, fkd_args={"use_rag": False})

        self.assertTrue(mock_call.called)
        called_kwargs = mock_call.call_args[1]
        self.assertIn("latents", called_kwargs)
        self.assertEqual(called_kwargs["latents"].device, pipe.device)


if __name__ == "__main__":
    unittest.main(verbosity=2)
