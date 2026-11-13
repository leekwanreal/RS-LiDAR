# Copyright 2026 RS-LiDAR Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

try:
    from diffusers import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput
except ImportError:
    # Graceful fallback for non-GPU / lightweight local test environments
    class FluxPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FluxPipelineOutput:
        def __init__(self, images):
            self.images = images

from rs_flow_lidar import RSFlowLiDARGuidance


class FluxPipeline_LiDAR(FluxPipeline):
    r"""
    Pipeline lấy mẫu Flow Matching cho FLUX.1 tích hợp cơ chế dẫn đường LiDAR và RS-LiDAR.
    Công thức gia số vận tốc lái:
        \Delta v_t(z_t) = - (s_flow / t) * \sum_{k=1}^K (w_k^r - w_k) z_{0,k}
    Kết hợp đồng bộ với Classifier-Free Guidance (CFG) và bộ giải Euler lùi thời gian.
    """

    _optional_components = ["image_encoder", "feature_extractor"]
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"

    def __init__(
        self,
        scheduler: Any = None,
        vae: Any = None,
        text_encoder: Any = None,
        tokenizer: Any = None,
        text_encoder_2: Any = None,
        tokenizer_2: Any = None,
        transformer: Any = None,
        image_encoder: Any = None,
        feature_extractor: Any = None,
    ):
        if hasattr(super(), "__init__") and scheduler is not None and transformer is not None:
            try:
                super().__init__(
                    scheduler=scheduler,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    transformer=transformer,
                    image_encoder=image_encoder,
                    feature_extractor=feature_extractor,
                )
            except Exception:
                pass
        self.flow_guidance_module = RSFlowLiDARGuidance(s_flow=12.5, lambda_r=5000.0)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            from diffusers import FluxPipeline
            if cls is not FluxPipeline and issubclass(cls, FluxPipeline):
                pipe = FluxPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                pipe.__class__ = cls
                if not hasattr(pipe, "flow_guidance_module"):
                    pipe.flow_guidance_module = RSFlowLiDARGuidance(s_flow=12.5, lambda_r=5000.0)
                return pipe
        except Exception:
            pass
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def to(self, *args, **kwargs):
        r"""
        Cưỡng chế chuyển toàn bộ sub-modules của FLUX (transformer, text encoders, vae)
        lên đúng accelerator device (cuda) và dtype mong muốn.
        """
        device = None
        dtype = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        elif len(args) >= 2:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
                device = args[1]
            else:
                device = args[0]
                dtype = args[1]
        device = kwargs.get("device", device)
        dtype = kwargs.get("dtype", dtype)

        target_device = torch.device(device) if device is not None else getattr(self, "device", None)
        
        for sub_name in ["transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"]:
            sub = getattr(self, sub_name, None)
            if sub is not None and isinstance(sub, torch.nn.Module):
                try:
                    if target_device is not None and dtype is not None:
                        sub.to(device=target_device, dtype=dtype)
                    elif target_device is not None:
                        sub.to(device=target_device)
                    elif dtype is not None:
                        sub.to(dtype=dtype)
                except Exception:
                    pass

        try:
            super().to(*args, **kwargs)
        except Exception:
            pass
        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_idx: int = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 28,
        timesteps: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        fkd_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        r"""
        Vòng lặp lấy mẫu Euler Flow Matching có dẫn đường lookahead.
        """
        # Nếu không có fkd_args hoặc không bật use_rag, gọi pipeline mặc định
        if fkd_args is None or not fkd_args.get("use_rag", False):
            if hasattr(super(), "__call__"):
                return super().__call__(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    output_type=output_type,
                    return_dict=return_dict,
                    **kwargs,
                )

        # Trích xuất dữ liệu Lookahead từ rag_dataset
        rag_dataset = fkd_args.get("rag_dataset", None)
        lookahead_latents = None
        rewards = None

        if rag_dataset is not None and hasattr(rag_dataset, "data") and prompt_idx is not None:
            if prompt_idx in rag_dataset.data:
                subset = rag_dataset.data[prompt_idx]
                raw_lats = subset.get("latents", None)
                if raw_lats is not None:
                    lookahead_latents = raw_lats[:fkd_args.get("top_k", 50)]
                raw_rew = subset.get("rewards", None)
                if raw_rew is not None:
                    rewards = raw_rew[:fkd_args.get("top_k", 50)]

        # Cấu hình scale vận tốc
        s_flow = fkd_args.get("s_flow", 12.5)
        lmbda = fkd_args.get("lmbda", 5000.0)
        self.flow_guidance_module.s_flow = s_flow
        self.flow_guidance_module.lambda_r = lmbda

        device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Chuẩn bị sigmas và mu cho dynamic shifting trong FLUX scheduler
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        if hasattr(self, "scheduler") and hasattr(self.scheduler, "config"):
            if getattr(self.scheduler.config, "use_flow_sigmas", False):
                sigmas = None

        mu = None
        if hasattr(self, "scheduler") and hasattr(self.scheduler, "config") and getattr(self.scheduler.config, "use_dynamic_shifting", False):
            image_seq_len = (height // 16) * (width // 16)
            base_seq_len = getattr(self.scheduler.config, "base_image_seq_len", 256)
            max_seq_len = getattr(self.scheduler.config, "max_image_seq_len", 4096)
            base_shift = getattr(self.scheduler.config, "base_shift", 0.5)
            max_shift = getattr(self.scheduler.config, "max_shift", 1.15)
            try:
                from diffusers.pipelines.flux.pipeline_flux import calculate_shift
                mu = calculate_shift(
                    image_seq_len,
                    base_seq_len=base_seq_len,
                    max_seq_len=max_seq_len,
                    base_shift=base_shift,
                    max_shift=max_shift,
                )
            except Exception:
                m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                b = base_shift - m * base_seq_len
                mu = image_seq_len * m + b

        # Khởi tạo timesteps theo chuẩn FlowMatchEulerDiscreteScheduler
        if hasattr(self, "scheduler") and hasattr(self.scheduler, "set_timesteps"):
            try:
                from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
                timesteps, num_inference_steps = retrieve_timesteps(
                    self.scheduler,
                    num_inference_steps,
                    device,
                    sigmas=sigmas,
                    mu=mu,
                )
            except Exception:
                try:
                    self.scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
                except TypeError:
                    try:
                        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
                    except TypeError:
                        self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        elif timesteps is None:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        num_channels = 16
        latent_h = height // 8
        latent_w = width // 8
        device = getattr(self, "_execution_device", getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        # Mã hóa Prompt Embeddings cho FLUX nếu có encoder
        text_ids = None
        latent_image_ids = None
        if hasattr(self, "encode_prompt") and prompt_embeds is None:
            try:
                enc_out = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )
                if isinstance(enc_out, tuple):
                    if len(enc_out) >= 3:
                        prompt_embeds, pooled_prompt_embeds, text_ids = enc_out[:3]
                    elif len(enc_out) == 2:
                        prompt_embeds, pooled_prompt_embeds = enc_out
            except Exception as e:
                pass

        # Đảm bảo device đồng bộ cho embeddings
        if prompt_embeds is not None and hasattr(prompt_embeds, "to"):
            prompt_embeds = prompt_embeds.to(device=device)
        if pooled_prompt_embeds is not None and hasattr(pooled_prompt_embeds, "to"):
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device)
        if text_ids is not None and hasattr(text_ids, "to"):
            text_ids = text_ids.to(device=device)

        # Chuẩn bị Latents chuẩn xác cho FLUX
        if latents is None:
            if hasattr(self, "prepare_latents"):
                try:
                    lat_res = self.prepare_latents(
                        batch_size=batch_size * num_images_per_prompt,
                        num_channels_latents=num_channels,
                        height=height,
                        width=width,
                        dtype=prompt_embeds.dtype if prompt_embeds is not None else torch.float32,
                        device=device,
                        generator=generator,
                        latents=latents,
                    )
                    if isinstance(lat_res, tuple):
                        latents, latent_image_ids = lat_res[0], lat_res[1]
                    else:
                        latents = lat_res
                except Exception:
                    latents = torch.randn((batch_size, num_channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
            else:
                latents = torch.randn((batch_size, num_channels, latent_h, latent_w), generator=generator, device=device, dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
        else:
            latents = latents.to(device=device)

        if latent_image_ids is not None and hasattr(latent_image_ids, "to"):
            latent_image_ids = latent_image_ids.to(device=device)

        # Chuẩn bị Lookahead tensors
        if lookahead_latents is not None:
            lookahead_latents = lookahead_latents.to(device=device, dtype=latents.dtype)
        if rewards is not None:
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            else:
                rewards = rewards.to(device=device, dtype=torch.float32)

        # Cửa sổ thời gian áp dụng guidance (Fast-Path)
        t_start = fkd_args.get("guidance_start", fkd_args.get("resampling_t_start", 1.0))
        t_end = fkd_args.get("guidance_end", fkd_args.get("resampling_t_end", 0.95))
        # Phòng hộ nếu vô tình truyền nhầm tham số diffusion dạng số nguyên (> 1.0)
        if t_start > 1.0:
            t_start = 1.0
        if t_end > 1.0:
            t_end = 0.95

        self.flow_guidance_module.t_start = t_start
        self.flow_guidance_module.t_end = t_end

        # -------------------------------------------------------------
        # Vòng lặp giải vi phân Euler ngược thời gian (t = 1.0 -> 0.0)
        # -------------------------------------------------------------
        sched = getattr(self, "scheduler", None)
        sched_order = getattr(sched, "order", 1) if sched is not None else 1
        num_warmup_steps = max(len(timesteps) - num_inference_steps * sched_order, 0) if sched is not None else 0
        has_prog = hasattr(self, "progress_bar")

        def _run_denoising_loop(prog_bar=None):
            nonlocal latents
            for i, t in enumerate(timesteps):
                t_curr = t.item() if isinstance(t, torch.Tensor) else float(t)
                t_next = timesteps[i + 1].item() if (i + 1 < len(timesteps) and isinstance(timesteps[i + 1], torch.Tensor)) else (float(timesteps[i + 1]) if i + 1 < len(timesteps) else 0.0)
                dt = t_curr - t_next  # dt > 0

                # 1. Dự đoán vận tốc mô hình v_pred (CFG)
                # Với FLUX: biến đổi qua DiT transformer
                if hasattr(self, "transformer"):
                    # Chuyển đổi latents nếu cần đóng gói (pack)
                    if hasattr(self, "_pack_latents") and latents.ndim == 4:
                        packed_latents = self._pack_latents(latents, batch_size, num_channels, latent_h, latent_w)
                    else:
                        packed_latents = latents

                    # Chuẩn bị timestep và guidance vector chuẩn FLUX
                    if prompt_embeds is not None and callable(self.transformer):
                        try:
                            guidance_vec = torch.tensor([guidance_scale], device=latents.device, dtype=torch.float32).expand(latents.shape[0]) if (hasattr(self.transformer, "config") and getattr(self.transformer.config, "guidance_embeds", False)) else None
                            timestep_input = t.expand(latents.shape[0]).to(latents.dtype) if isinstance(t, torch.Tensor) else torch.tensor([t], device=latents.device, dtype=latents.dtype).expand(latents.shape[0])
                            t_scaled = timestep_input / 1000.0 if timestep_input.max() > 1.0 else timestep_input

                            v_pred = self.transformer(
                                hidden_states=packed_latents,
                                timestep=t_scaled,
                                guidance=guidance_vec,
                                pooled_projections=pooled_prompt_embeds,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]
                        except Exception as e:
                            print(f"⚠️ [Transformer Step {i}] Forward warning: {e}")
                            v_pred = packed_latents
                    else:
                        v_pred = packed_latents
                else:
                    v_pred = torch.zeros_like(latents)

                # 2. Tính toán Vector Gia Số Vận Tốc Lái Lookahead (LiDAR / RS-LiDAR)
                t_norm = t_curr / 1000.0 if t_curr > 1.0 else t_curr
                if (lookahead_latents is not None and rewards is not None and 
                    t_end <= t_norm <= t_start):
                    
                    spatial_latents = latents
                    if hasattr(self, "_unpack_latents") and latents.ndim == 3:
                        spatial_latents = self._unpack_latents(latents, height, width, getattr(self, "vae_scale_factor", 16))

                    delta_v = self.flow_guidance_module.compute_velocity_guidance(
                        z_t=spatial_latents,
                        lookahead_z0=lookahead_latents,
                        smoothed_rewards=rewards,
                        sigma_t=t_norm,
                        t_start=t_start,
                        t_end=t_end,
                    )
                    
                    if hasattr(self, "_pack_latents") and latents.ndim == 3:
                        delta_v_packed = self._pack_latents(delta_v, batch_size, num_channels, latent_h, latent_w)
                        v_pred = v_pred + delta_v_packed
                    else:
                        v_pred = v_pred + delta_v

                # 3. Bước cập nhật tích phân Euler hoặc qua Scheduler
                if sched is not None and hasattr(sched, "step"):
                    try:
                        latents = sched.step(v_pred, t, latents, return_dict=False)[0]
                    except Exception:
                        latents = latents - dt * v_pred
                else:
                    latents = latents - dt * v_pred

                if prog_bar is not None and hasattr(prog_bar, "update"):
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % sched_order == 0):
                        prog_bar.update()

        if has_prog:
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                _run_denoising_loop(progress_bar)
        else:
            _run_denoising_loop(None)

        # Giải mã ảnh sạch ở bước cuối
        if hasattr(self, "_unpack_latents") and latents.ndim == 3:
            latents = self._unpack_latents(latents, height, width, getattr(self, "vae_scale_factor", 16))

        if hasattr(self, "vae"):
            scaled_latents = (latents / self.vae.config.scaling_factor) + getattr(self.vae.config, "shift_factor", 0.0)
            decoded_chunks = []
            for idx in range(scaled_latents.shape[0]):
                chunk = scaled_latents[idx : idx + 1]
                dec = self.vae.decode(chunk, return_dict=False)[0]
                decoded_chunks.append(dec)
            decoded_tensor = torch.cat(decoded_chunks, dim=0)
            if hasattr(self, "image_processor"):
                images = self.image_processor.postprocess(decoded_tensor, output_type=output_type)
            else:
                images = decoded_tensor
        else:
            images = latents

        if not return_dict:
            return (images,)
        return FluxPipelineOutput(images=images)
