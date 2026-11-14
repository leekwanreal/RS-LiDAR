import os
import sys
import unittest
from unittest.mock import MagicMock
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from fkd_diffusers.fkd_pipeline_flux import FluxPipeline_LiDAR
from diffusers import FlowMatchEulerDiscreteScheduler


class TestFluxPipelineBugfixes(unittest.TestCase):
    """
    Kiểm thử kiểm tra triệt để các lỗi đã sửa trong FluxPipeline_LiDAR:
    1. Timestep scaling ở bước cuối (t = 1.0): t_scaled phải là 0.001, KHÔNG ĐƯỢC là 1.0!
    2. t_norm ở bước cuối (t = 1.0): t_norm phải là 0.001, KHÔNG ĐƯỢC rơi vào [0.95, 1.0]!
    3. Scheduler.set_begin_index(0) được gọi trước vòng lặp.
    4. vae_scale_factor mặc định chuẩn 8 (chứ không phải 16 làm hỏng spatial latents).
    5. Fallback Euler dt: scaled_dt = dt / 1000.0 khi is_discrete_1000.
    6. Khi s_flow = 0.0, ảnh không bị burn và vận tốc lái delta_v = 0.
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_01_timestep_scaling_and_guidance_at_final_step(self):
        """
        Kiểm tra: với 28 bước từ FlowMatchEulerDiscreteScheduler, bước cuối có t=1.0.
        Transformer phải nhận t_scaled = 0.001 (1.0 / 1000.0) chứ không phải 1.0.
        Và t_norm ở bước cuối = 0.001, không bị kích hoạt guidance.
        """
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = FlowMatchEulerDiscreteScheduler()
        # Mô phỏng lịch trình timesteps chính xác của FLUX với dynamic shifting kết thúc tại 1.0
        pipe.scheduler.timesteps = torch.linspace(1000.0, 1.0, 28)
        self.assertEqual(pipe.scheduler.timesteps[-1].item(), 1.0)

        # Mock transformer để ghi lại tất cả các giá trị t_scaled truyền vào
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
            # Trả về zero velocity
            h = kwargs.get("hidden_states")
            return (torch.zeros_like(h),)

        mock_transformer.side_effect = transformer_forward
        pipe.transformer = mock_transformer

        # Mock guidance module để kiểm tra xem guidance có bị gọi ở bước cuối hay không
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

        out = pipe(
            prompt="A wooden bench in a park",
            prompt_idx=0,
            height=1024,
            width=1024,
            num_inference_steps=28,
            timesteps=target_timesteps,
            output_type="latent",
            fkd_args=fkd_args,
        )

        # 1. Khẳng định số bước chạy đủ 28
        self.assertEqual(len(captured_timesteps), 28)

        # 2. Khẳng định bước đầu tiên (t = 1000.0) -> t_scaled = 1.0
        self.assertAlmostEqual(captured_timesteps[0], 1.0, places=4)

        # 3. Khẳng định CRITICAL BUG FIX: bước cuối cùng (t = 1.0) -> t_scaled = 0.001 (1.0 / 1000.0)
        # TRƯỚC FIX: t_scaled = 1.0 (do `1.0 > 1.0` là False)!
        self.assertAlmostEqual(captured_timesteps[-1], 0.001, places=4,
                               msg=f"LỖI CỐT LÕI: Bước cuối nhận t_scaled={captured_timesteps[-1]} thay vì 0.001!")

        # 4. Khẳng định guidance module KHÔNG được gọi ở bước cuối (t_norm = 0.001 < 0.95)
        # Chỉ được gọi trong khoảng [0.95, 1.0], tức là ở các bước đầu tiên
        self.assertGreater(len(mock_guidance_module.compute_velocity_guidance.call_args_list), 0)
        for call_args in mock_guidance_module.compute_velocity_guidance.call_args_list:
            sigma_t_called = call_args.kwargs.get("sigma_t")
            self.assertGreaterEqual(sigma_t_called, 0.95,
                                    msg=f"Guidance không được gọi khi sigma_t < 0.95! Nhận: {sigma_t_called}")
            self.assertLessEqual(sigma_t_called, 1.0,
                                 msg=f"Guidance không được gọi khi sigma_t > 1.0! Nhận: {sigma_t_called}")

    def test_02_set_begin_index_called(self):
        """Khẳng định: scheduler.set_begin_index(0) được gọi đúng chuẩn Diffusers trước vòng lặp."""
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
        """Khẳng định: vae_scale_factor mặc định là 8 (chuẩn FLUX), không phải 16."""
        pipe = FluxPipeline_LiDAR()
        self.assertEqual(pipe.vae_scale_factor, 8)

    def test_04_euler_fallback_scaled_dt(self):
        """Khẳng định: Fallback vi phân Euler lùi sử dụng scaled_dt = dt / 1000.0 khi timesteps ở thang đo [1, 1000]."""
        pipe = FluxPipeline_LiDAR()
        pipe.scheduler = None  # Cưỡng chế chạy vào nhánh fallback Euler

        pipe.transformer = MagicMock()
        pipe.transformer.config.guidance_embeds = False
        pipe.transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        # Model dự đoán vận tốc không đổi v_pred = 1.0
        pipe.transformer.side_effect = lambda **kwargs: (torch.ones_like(kwargs["hidden_states"]),)
        pipe.encode_prompt = MagicMock(return_value=(torch.zeros(1, 10, 64), torch.zeros(1, 10), torch.zeros(10, 3)))

        # 2 timesteps: 1000.0 và 900.0 -> dt = 100.0. Thang đo 1000.
        # scaled_dt phải là 100.0 / 1000.0 = 0.1, KHÔNG PHẢI 100.0!
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
        # TRƯỚC FIX: latents = 0 - (100.0 + 900.0) * 1.0 = -1000.0 (Bị nổ latents 1000x)!
        self.assertAlmostEqual(out_latent.mean().item(), -1.0, places=3,
                               msg=f"Fallback Euler dt không được chia cho 1000: mean={out_latent.mean().item()}")

    def test_05_s_flow_zero_clean_trajectory(self):
        """Khẳng định: Khi s_flow = 0.0, trajectory hoàn toàn không bị ảnh hưởng bởi lookahead candidates."""
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

        # Vì transformer output là 0 và s_flow = 0.0, latents phải giữ nguyên là 0
        self.assertTrue(torch.allclose(out.images, init_lat, atol=1e-5),
                        "Khi s_flow = 0.0, latents không được bị ô nhiễm bởi lookahead!")

    def test_06_default_scheduler_timesteps_and_current_timestep_reset(self):
        """Khẳng định: Khi không truyền timesteps (timesteps=None), retrieve_timesteps hoạt động trơn tru không cảnh báo và _current_timestep được dọn dẹp sạch."""
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

        out = pipe(
            prompt="A wooden bench in a park",
            prompt_idx=0,
            height=1024,
            width=1024,
            num_inference_steps=28,
            timesteps=None,  # Tự động sinh qua FlowMatchEulerDiscreteScheduler
            output_type="latent",
            fkd_args=fkd_args,
        )

        self.assertEqual(len(captured_timesteps), 28)
        self.assertAlmostEqual(captured_timesteps[0], 1.0, places=4)
        # Bước cuối phải < 0.05 (chuẩn 0.001 hoặc 0.0357) và không phải 1.0
        self.assertLess(captured_timesteps[-1], 0.05)
        # Khẳng định _current_timestep đã được dọn sạch về None
        self.assertIsNone(pipe._current_timestep)

    def test_07_multiple_prompts_scheduler_step_index_reset(self):
        """Khẳng định: Chạy liên tiếp nhiều prompts không bị lỗi IndexError ở scheduler (step_index được reset)."""
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

        # Prompt 1: Trước fix, step_index là 28 nên sched.step ném IndexError ở ngay bước 0!
        out1 = pipe(prompt="prompt 1", prompt_idx=1, num_inference_steps=28, fkd_args=fkd_args, output_type="latent")
        self.assertIsNotNone(out1)
        self.assertEqual(pipe.scheduler.step_index, 28)


if __name__ == "__main__":
    unittest.main(verbosity=2)
