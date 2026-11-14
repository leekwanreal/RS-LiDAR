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
"""FLUX.1 flow-matching pipeline with LiDAR / RS-LiDAR lookahead velocity guidance.

`FluxPipeline_LiDAR.__call__` runs the Euler flow-matching sampler and, inside the guidance window
[guidance_end, guidance_start] of flow time t, adds the velocity correction of
`rs_flow_lidar.RSFlowLiDARGuidance.compute_velocity_guidance`,
    delta_v_t = -(s_flow / t) * sum_k (w^r_k - w_k) z_{0,k},
computed from the Phase 1 lookahead latents z_{0,k} and their (optionally smoothed) rewards.
Without `fkd_args["use_rag"]` it defers to the stock diffusers `FluxPipeline`.
"""
import contextlib
import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput, calculate_shift, retrieve_timesteps

from rs_flow_lidar import RSFlowLiDARGuidance


class FluxPipeline_LiDAR(FluxPipeline):
    r"""FLUX.1 pipeline whose Euler flow-matching loop adds the lookahead velocity guidance."""

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
        # Without components (unit tests) the diffusers constructor is skipped.
        if scheduler is not None and transformer is not None:
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
        if not hasattr(self, "_internal_dict"):
            self._internal_dict = {}
        if not hasattr(self, "vae_scale_factor"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) and hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels") else 8
        self.flow_guidance_module = RSFlowLiDARGuidance(s_flow=12.5, lambda_r=5000.0)
        self._execution_device_cache = None

    @property
    def device(self) -> torch.device:
        r"""Execution device: the pinned device if set, else the first accelerator sub-module device.

        Overrides the diffusers lookup, which can fall back to CPU when the module registry is empty.
        """
        if getattr(self, "_execution_device_cache", None) is not None:
            return self._execution_device_cache
        for sub_name in ["transformer", "text_encoder_2", "text_encoder", "vae"]:
            sub = getattr(self, sub_name, None)
            if sub is not None and isinstance(sub, torch.nn.Module):
                dev = getattr(sub, "device", None)
                if dev is not None and dev.type not in ("cpu", "meta"):
                    return dev
                params = list(sub.parameters())
                if len(params) > 0 and params[0].device.type not in ("cpu", "meta"):
                    return params[0].device
        for sub_name in ["transformer", "text_encoder_2", "text_encoder", "vae"]:
            sub = getattr(self, sub_name, None)
            if sub is not None and isinstance(sub, torch.nn.Module):
                dev = getattr(sub, "device", None)
                if dev is not None:
                    return dev
                params = list(sub.parameters())
                if len(params) > 0:
                    return params[0].device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _execution_device(self) -> torch.device:
        return self.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load a stock `FluxPipeline` and re-class it (keeps the diffusers loading logic unchanged)."""
        pipe = FluxPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        pipe.__class__ = cls
        if not hasattr(pipe, "vae_scale_factor"):
            pipe.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1) if getattr(pipe, "vae", None) and hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "block_out_channels") else 8
        if not hasattr(pipe, "flow_guidance_module"):
            pipe.flow_guidance_module = RSFlowLiDARGuidance(s_flow=12.5, lambda_r=5000.0)
        return pipe

    def to(self, *args, **kwargs):
        r"""Move every FLUX sub-module (transformer, text encoders, VAE) and pin the execution device."""
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
        if target_device is not None:
            self._execution_device_cache = target_device

        for sub_name in ["transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"]:
            sub = getattr(self, sub_name, None)
            if sub is not None and isinstance(sub, torch.nn.Module):
                if target_device is not None and dtype is not None:
                    sub.to(device=target_device, dtype=dtype)
                elif target_device is not None:
                    sub.to(device=target_device)
                elif dtype is not None:
                    sub.to(dtype=dtype)

        super().to(*args, **kwargs)
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
        r"""Euler flow-matching sampling with lookahead velocity guidance (stock FLUX without `use_rag`)."""
        if fkd_args is None or not fkd_args.get("use_rag", False):
            if hasattr(super(), "__call__"):
                call_sig = inspect.signature(super().__call__).parameters
                call_candidates = {
                    "prompt": prompt,
                    "prompt_2": prompt_2,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "num_images_per_prompt": num_images_per_prompt,
                    "generator": generator,
                    "latents": latents.to(self.device) if (latents is not None and hasattr(latents, "to")) else latents,
                    "prompt_embeds": prompt_embeds.to(self.device) if (prompt_embeds is not None and hasattr(prompt_embeds, "to")) else prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds.to(self.device) if (pooled_prompt_embeds is not None and hasattr(pooled_prompt_embeds, "to")) else pooled_prompt_embeds,
                    "output_type": output_type,
                    "return_dict": return_dict,
                    **kwargs,
                }
                if "timesteps" in call_sig and timesteps is not None:
                    call_candidates["timesteps"] = timesteps
                elif "sigmas" in call_sig and kwargs.get("sigmas", None) is not None:
                    call_candidates["sigmas"] = kwargs["sigmas"]

                filtered_kwargs = {k: v for k, v in call_candidates.items() if k in call_sig}
                return super().__call__(**filtered_kwargs)

        # Lookahead latents and rewards of this prompt.
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

        # Guidance strength.
        s_flow = fkd_args.get("s_flow", 12.5)
        lmbda = fkd_args.get("lmbda", 5000.0)
        self.flow_guidance_module.s_flow = s_flow
        self.flow_guidance_module.lambda_r = lmbda

        # Device and dtype from the transformer (or the VAE).
        trans_param = None
        if hasattr(self, "transformer") and self.transformer is not None and hasattr(self.transformer, "parameters"):
            params = list(self.transformer.parameters())
            if len(params) > 0:
                trans_param = params[0]

        vae_param = None
        if hasattr(self, "vae") and self.vae is not None and hasattr(self.vae, "parameters"):
            params = list(self.vae.parameters())
            if len(params) > 0:
                vae_param = params[0]

        if trans_param is not None and hasattr(trans_param, "device") and hasattr(trans_param, "dtype"):
            device = trans_param.device
            dtype = trans_param.dtype
        elif vae_param is not None and hasattr(vae_param, "device") and hasattr(vae_param, "dtype"):
            device = vae_param.device
            dtype = vae_param.dtype
        else:
            device = self.device
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

        # Sigmas and the resolution-dependent shift mu of the FLUX scheduler.
        if timesteps is not None:
            sigmas = None
        else:
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
            mu = calculate_shift(
                image_seq_len,
                base_seq_len=base_seq_len,
                max_seq_len=max_seq_len,
                base_shift=base_shift,
                max_shift=max_shift,
            )

        # Timesteps of the FlowMatchEulerDiscreteScheduler (in [1, 1000]).
        if hasattr(self, "scheduler") and hasattr(self.scheduler, "set_timesteps"):
            try:
                if timesteps is not None:
                    timesteps, num_inference_steps = retrieve_timesteps(
                        self.scheduler,
                        num_inference_steps,
                        device,
                        timesteps=timesteps,
                    )
                else:
                    timesteps, num_inference_steps = retrieve_timesteps(
                        self.scheduler,
                        num_inference_steps,
                        device,
                        sigmas=sigmas,
                        mu=mu,
                    )
            except ValueError:
                # Scheduler without support for custom timesteps / sigmas.
                try:
                    if timesteps is not None:
                        self.scheduler.set_timesteps(timesteps=timesteps, device=device)
                    elif sigmas is not None:
                        self.scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
                    else:
                        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
                except TypeError:
                    try:
                        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
                    except TypeError:
                        self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        elif timesteps is None:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

        # FlowMatchEulerDiscreteScheduler timesteps are in [1, 1000]; flow time t = timestep / 1000.
        is_discrete_1000 = bool((timesteps.max() > 1.0) if isinstance(timesteps, torch.Tensor) else (max(timesteps) > 1.0))

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        num_channels = 16
        latent_h = height // 8
        latent_w = width // 8

        # Text embeddings.
        text_ids = None
        latent_image_ids = None
        if hasattr(self, "encode_prompt") and prompt_embeds is None:
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

        if prompt_embeds is not None and hasattr(prompt_embeds, "to"):
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if pooled_prompt_embeds is not None and hasattr(pooled_prompt_embeds, "to"):
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        if text_ids is not None and hasattr(text_ids, "to"):
            text_ids = text_ids.to(device=device)

        # Initial latents.
        if latents is None:
            if hasattr(self, "prepare_latents"):
                lat_res = self.prepare_latents(
                    batch_size=batch_size * num_images_per_prompt,
                    num_channels_latents=num_channels,
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                    latents=latents,
                )
                if isinstance(lat_res, tuple):
                    latents, latent_image_ids = lat_res[0], lat_res[1]
                else:
                    latents = lat_res
            else:
                latents = torch.randn((batch_size, num_channels, latent_h, latent_w), generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        if hasattr(self, "_pack_latents") and latents.ndim == 4:
            latents = self._pack_latents(latents, batch_size, num_channels, latent_h, latent_w)
        if latent_image_ids is None and hasattr(self, "_prepare_latent_image_ids"):
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 16, width // 16, device, dtype)

        if latent_image_ids is not None and hasattr(latent_image_ids, "to"):
            latent_image_ids = latent_image_ids.to(device=device)

        # Lookahead tensors.
        if lookahead_latents is not None:
            lookahead_latents = lookahead_latents.to(device=device, dtype=dtype)
        if rewards is not None:
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            else:
                rewards = rewards.to(device=device, dtype=torch.float32)

        # Guidance window in flow time.
        t_start = fkd_args.get("guidance_start", fkd_args.get("resampling_t_start", 1.0))
        t_end = fkd_args.get("guidance_end", fkd_args.get("resampling_t_end", 0.95))
        # Guard against diffusion-style integer step indices passed as the window.
        if t_start > 1.0:
            t_start = 1.0
        if t_end > 1.0:
            t_end = 0.95

        self.flow_guidance_module.t_start = t_start
        self.flow_guidance_module.t_end = t_end

        # FLUX distilled guidance embedding.
        if hasattr(self, "transformer") and hasattr(self.transformer, "config") and getattr(self.transformer.config, "guidance_embeds", False):
            guidance_vec = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
        else:
            guidance_vec = None

        # Euler integration from t = 1 (noise) to t = 0 (data).
        sched = getattr(self, "scheduler", None)
        sched_order = getattr(sched, "order", 1) if sched is not None else 1
        num_warmup_steps = max(len(timesteps) - num_inference_steps * sched_order, 0) if sched is not None else 0
        has_prog = hasattr(self, "progress_bar")

        # Reset the scheduler step counter (as in diffusers' FluxPipeline).
        if sched is not None:
            if hasattr(sched, "_step_index"):
                sched._step_index = None
            if hasattr(sched, "set_begin_index"):
                sched.set_begin_index(0)

        def _run_denoising_loop(prog_bar=None):
            nonlocal latents
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                t_curr = t.item() if isinstance(t, torch.Tensor) else float(t)
                t_next = timesteps[i + 1].item() if (i + 1 < len(timesteps) and isinstance(timesteps[i + 1], torch.Tensor)) else (float(timesteps[i + 1]) if i + 1 < len(timesteps) else 0.0)
                dt = t_curr - t_next  # dt > 0

                # 1. Model velocity v_pred.
                if hasattr(self, "transformer") and self.transformer is not None:
                    if hasattr(self, "_pack_latents") and latents.ndim == 4:
                        packed_latents = self._pack_latents(latents, batch_size, num_channels, latent_h, latent_w)
                    else:
                        packed_latents = latents

                    trans_param = next(iter(self.transformer.parameters()), None) if hasattr(self.transformer, "parameters") else None
                    t_dev = trans_param.device if (trans_param is not None and hasattr(trans_param, "device")) else device
                    t_dtype = trans_param.dtype if (trans_param is not None and hasattr(trans_param, "dtype")) else dtype

                    packed_latents = packed_latents.to(device=t_dev, dtype=t_dtype)
                    p_emb = prompt_embeds.to(device=t_dev, dtype=t_dtype) if prompt_embeds is not None else None
                    pooled_emb = pooled_prompt_embeds.to(device=t_dev, dtype=t_dtype) if pooled_prompt_embeds is not None else None
                    txt_ids = text_ids.to(device=t_dev) if text_ids is not None else None
                    img_ids = latent_image_ids.to(device=t_dev) if latent_image_ids is not None else None
                    g_vec = guidance_vec.to(device=t_dev, dtype=torch.float32) if guidance_vec is not None else None

                    if p_emb is not None and callable(self.transformer):
                        timestep_input = t.expand(latents.shape[0]).to(device=t_dev, dtype=t_dtype) if isinstance(t, torch.Tensor) else torch.tensor([t], device=t_dev, dtype=t_dtype).expand(latents.shape[0])
                        t_scaled = timestep_input / 1000.0 if is_discrete_1000 else timestep_input

                        with (self.transformer.cache_context("cond") if hasattr(self.transformer, "cache_context") else contextlib.nullcontext()):
                            v_pred = self.transformer(
                                hidden_states=packed_latents,
                                timestep=t_scaled,
                                guidance=g_vec,
                                pooled_projections=pooled_emb,
                                encoder_hidden_states=p_emb,
                                txt_ids=txt_ids,
                                img_ids=img_ids,
                                joint_attention_kwargs=getattr(self, "joint_attention_kwargs", None),
                                return_dict=False,
                            )[0]
                    else:
                        raise RuntimeError("FLUX LiDAR: prompt_embeds are missing or the transformer is not callable.")
                else:
                    raise RuntimeError("FLUX LiDAR: the pipeline has no transformer.")

                # 2. Lookahead velocity guidance inside the window.
                t_norm = t_curr / 1000.0 if is_discrete_1000 else t_curr
                if (lookahead_latents is not None and rewards is not None and
                        t_end <= t_norm <= t_start):
                    spatial_latents = latents
                    if hasattr(self, "_unpack_latents") and latents.ndim == 3:
                        spatial_latents = self._unpack_latents(latents, height, width, getattr(self, "vae_scale_factor", 8))

                    flux_debug = bool(fkd_args.get("flux_debug", False))
                    g_kwargs = dict(
                        z_t=spatial_latents,
                        lookahead_z0=lookahead_latents,
                        smoothed_rewards=rewards,
                        sigma_t=t_norm,
                        t_start=t_start,
                        t_end=t_end,
                    )
                    if flux_debug:
                        delta_v, diag = self.flow_guidance_module.compute_velocity_guidance(**g_kwargs, return_diagnostics=True)
                    else:
                        delta_v, diag = self.flow_guidance_module.compute_velocity_guidance(**g_kwargs), {}
                    if flux_debug and diag.get("w_diff") is not None:
                        scaled_dt_dbg = (dt / 1000.0) if is_discrete_1000 else dt
                        z_norm = spatial_latents.float().norm().item()
                        kick = scaled_dt_dbg * delta_v.float().norm().item()
                        print(f"   [flux_debug] step {i:02d} t={t_norm:.4f} dt={scaled_dt_dbg:.4f} "
                              f"w_r-w={[round(x, 3) for x in diag['w_diff'][0].tolist()]} "
                              f"|dt*dv|/|z|={kick / max(z_norm, 1e-8):.4f}")

                    if hasattr(self, "_pack_latents") and latents.ndim == 3:
                        delta_v_packed = self._pack_latents(delta_v, batch_size, num_channels, latent_h, latent_w)
                        v_pred = v_pred + delta_v_packed.to(dtype=v_pred.dtype, device=v_pred.device)
                    else:
                        v_pred = v_pred + delta_v.to(dtype=v_pred.dtype, device=v_pred.device)

                # 3. Euler step (scheduler, or plain Euler without one).
                latents_dtype = latents.dtype
                if sched is not None and hasattr(sched, "step"):
                    latents = sched.step(v_pred, t, latents, return_dict=False)[0]
                else:
                    scaled_dt = (dt / 1000.0) if is_discrete_1000 else dt
                    latents = latents - scaled_dt * v_pred

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if prog_bar is not None and hasattr(prog_bar, "update"):
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % sched_order == 0):
                        prog_bar.update()

        if has_prog:
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                _run_denoising_loop(progress_bar)
        else:
            _run_denoising_loop(None)

        self._current_timestep = None

        # Decode, or return the packed latents for output_type == "latent".
        if output_type == "latent":
            images = latents
        else:
            if hasattr(self, "_unpack_latents") and latents.ndim == 3:
                latents = self._unpack_latents(latents, height, width, getattr(self, "vae_scale_factor", 8))

            if hasattr(self, "vae") and self.vae is not None:
                vae_param = next(iter(self.vae.parameters()), None) if hasattr(self.vae, "parameters") else None
                vae_dev = vae_param.device if (vae_param is not None and hasattr(vae_param, "device")) else device
                vae_dtype = vae_param.dtype if (vae_param is not None and hasattr(vae_param, "dtype")) else dtype

                scaling = getattr(self.vae.config, "scaling_factor", 0.3611) if hasattr(self.vae, "config") else 0.3611
                shift = getattr(self.vae.config, "shift_factor", 0.1159) if hasattr(self.vae, "config") else 0.1159
                scaled_latents = ((latents / scaling) + shift).to(device=vae_dev, dtype=vae_dtype)
                decoded_chunks = []
                for idx in range(scaled_latents.shape[0]):
                    chunk = scaled_latents[idx : idx + 1]
                    dec = self.vae.decode(chunk, return_dict=False)[0]
                    decoded_chunks.append(dec)
                decoded_tensor = torch.cat(decoded_chunks, dim=0)
                if hasattr(self, "image_processor") and self.image_processor is not None:
                    images = self.image_processor.postprocess(decoded_tensor, output_type=output_type)
                else:
                    images = decoded_tensor
            else:
                images = latents

        if not return_dict:
            return (images,)
        return FluxPipelineOutput(images=images)
