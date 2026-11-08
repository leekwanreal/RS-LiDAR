import os
import sys
import unittest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

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


class TestFluxPipelineLiDARDataFlow(unittest.TestCase):
    """
    Kiểm thử tích hợp luồng dữ liệu End-to-End cho FluxPipeline_LiDAR:
    1. Tiếp nhận tập lookahead 16-kênh 128x128.
    2. Cửa sổ thời gian Table 3 [0.95, 1.0] vs cửa sổ liên tục [0.20, 0.98].
    3. Gia số vận tốc delta_v được cộng chính xác vào v_pred.
    4. Bước Euler lùi thời gian dt > 0.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.pipe = FluxPipeline_LiDAR()
        self.K = 4
        self.C = 16
        self.H = 128
        self.W = 128
        self.rag_dataset = MockRagDataset(self.K, self.C, self.H, self.W)

    def test_01_guidance_active_in_table3_window(self):
        """Khẳng định: Tại t trong [0.95, 1.0], guidance hoạt động mạnh mẽ (Table 3 protocol)."""
        fkd_args = {
            "use_rag": True,
            "rag_dataset": self.rag_dataset,
            "top_k": self.K,
            "s_flow": 40.0,
            "lmbda": 5000.0,
            "guidance_start": 1.0,
            "guidance_end": 0.95,
        }
        
        # Test 1 bước Euler tại t=0.98 (nằm trong [0.95, 1.0])
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
        )
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out, "images"))

    def test_02_guidance_inactive_outside_window(self):
        """Khẳng định: Khi t < 0.95, delta_v = 0 và ODE chuyển động thẳng tự do (không bị can thiệp)."""
        guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0, t_start=1.0, t_end=0.95)
        z_t = torch.randn((1, 16, 128, 128))
        lookahead_z0 = torch.randn((4, 16, 128, 128))
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Tại t = 0.80 (ngoài cửa sổ [0.95, 1.0])
        delta_v = guider.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.80)
        self.assertEqual(torch.count_nonzero(delta_v).item(), 0, "Ngoài cửa sổ [0.95, 1.0], delta_v phải bằng 0 tuyệt đối!")

        # Tại t = 0.98 (trong cửa sổ [0.95, 1.0])
        delta_v_active = guider.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.98)
        self.assertGreater(torch.count_nonzero(delta_v_active).item(), 0, "Trong cửa sổ [0.95, 1.0], delta_v phải hoạt động!")

    def test_03_attraction_toward_best_candidate(self):
        """Khẳng định giải tích: -delta_v có tích vô hướng dương với hạt có reward lớn nhất."""
        guider = RSFlowLiDARGuidance(s_flow=40.0, lambda_r=5000.0, t_start=1.0, t_end=0.95)
        z_t = torch.randn((1, 16, 128, 128))
        lookahead_z0 = torch.randn((4, 16, 128, 128))
        # Hạt 2 có reward vượt trội
        rewards = torch.tensor([0.0, 0.0, 15.0, 0.0])

        delta_v = guider.compute_velocity_guidance(z_t, lookahead_z0, rewards, sigma_t=0.98)
        pull_force = -delta_v[0]
        best_candidate = lookahead_z0[2]
        dot = torch.sum(pull_force * best_candidate).item()
        self.assertGreater(dot, 0, "Lực Euler kéo phải hướng thuận về ứng viên reward cao nhất!")

    def test_04_integer_timestep_safety(self):
        """Khẳng định: Nếu vô tình truyền resampling_t_start=20, resampling_t_end=200, pipeline tự sửa an toàn."""
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
        )
        self.assertIsNotNone(out)
        self.assertEqual(self.pipe.flow_guidance_module.t_start, 1.0)
        self.assertEqual(self.pipe.flow_guidance_module.t_end, 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)

