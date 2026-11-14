"""ImageReward model with batched scoring, used as the reward R in Phases 1 and 2.

Adapted from ImageReward (Xu et al., 2023; https://github.com/THUDM/ImageReward), which builds on
the CLIP and improved-aesthetic-predictor code bases. `rm_load` mirrors `ImageReward.load` but
returns `IRSMC`, whose `score_batched` scores all particles of a prompt in one forward pass.
"""
import importlib.machinery
import os
import sys
import types
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

# ImageReward imports `wandb` at module level; provide an empty stub when it is not installed.
if "wandb" not in sys.modules:
    try:
        import wandb  # noqa: F401
    except ImportError:
        _mock_wandb = types.ModuleType("wandb")
        _mock_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        _mock_wandb.__file__ = "wandb"
        sys.modules["wandb"] = _mock_wandb

# Restore transformers utilities that the ImageReward BLIP code imports (must precede `import ImageReward`).
from fkd_diffusers.patch_transformers import patch_transformers_for_imagereward  # noqa: E402

patch_transformers_for_imagereward()

import ImageReward as RM  # noqa: E402
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain  # noqa: E402

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class IRSMC(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score_batched_old(self, prompts, images):
        # batch
        results = []
        for i, prompt in enumerate(prompts):
            results.append(self.score(prompt, images[i]))

        return results

    def score_gard(self, prompt_ids, prompt_attention_mask, image):
        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :]  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards

    def score(self, prompt, image):
        if type(image).__name__ == 'list':
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.'
            )

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()

    @torch.inference_mode()
    def score_batched(self, prompts, images, batch_size=None):
        if images is None or len(images) == 0:
            return []

        # Defensive normalization: ensure prompts is a list matching images length
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        elif isinstance(prompts, (list, tuple)):
            prompts = list(prompts)
            if len(prompts) == 1 and len(images) > 1:
                prompts = prompts * len(images)
            elif len(prompts) < len(images):
                prompts = prompts + [prompts[0]] * (len(images) - len(prompts))
            elif len(prompts) > len(images):
                prompts = prompts[:len(images)]
        else:
            prompts = [str(prompts)] * len(images)

        assert isinstance(images, list)

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images_tensor = [
            self.preprocess(image).unsqueeze(0).to(self.device) for image in images
        ]
        images_tensor = torch.cat(images_tensor, 0).to(self.device)

        # Single batch direct inference (true 1-batch execution as in paper)
        if batch_size is not None and 0 < batch_size < images_tensor.shape[0]:
            # Only chunk if caller explicitly requests a smaller batch size (e.g. low-VRAM GPU)
            embed_chunks = []
            for b_i in range(0, images_tensor.shape[0], batch_size):
                chunk = images_tensor[b_i : b_i + batch_size]
                embed_chunks.append(self.blip.visual_encoder(chunk))
            image_embeds = torch.cat(embed_chunks, dim=0)

            txt_feature_chunks = []
            for b_i in range(0, image_embeds.shape[0], batch_size):
                c_embed = image_embeds[b_i : b_i + batch_size]
                c_ids = text_input.input_ids[b_i : b_i + batch_size]
                c_mask = text_input.attention_mask[b_i : b_i + batch_size]
                c_atts = torch.ones(c_embed.size()[:-1], dtype=torch.long, device=self.device)
                text_out = self.blip.text_encoder(
                    c_ids,
                    attention_mask=c_mask,
                    encoder_hidden_states=c_embed,
                    encoder_attention_mask=c_atts,
                    return_dict=True,
                )
                txt_feature_chunks.append(text_out.last_hidden_state[:, 0, :].float())
            txt_features = torch.cat(txt_feature_chunks, dim=0)
        else:
            # 1 single batch: direct forward through visual encoder and text cross-attention
            image_embeds = self.blip.visual_encoder(images_tensor)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_features = text_output.last_hidden_state[:, 0, :].float()

        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rewards.view(txt_features.shape[0]).detach().cpu().numpy().tolist()

    def inference_rank(self, prompt, generations_list):
        text_input = self.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.'
                )
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return (
            indices.detach().cpu().numpy().tolist(),
            rewards.detach().cpu().numpy().tolist(),
        )


def rm_load(
    name: str = "ImageReward-v1.0",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    cache_root = download_root or os.path.expanduser("~/.cache/ImageReward")
    os.makedirs(cache_root, exist_ok=True)
    if name in RM.utils._MODELS:
        model_path = os.path.join(cache_root, "ImageReward.pt")
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
            import urllib.request
            try:
                urllib.request.urlretrieve("https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt", model_path)
            except OSError:
                model_path = RM.ImageReward_download(
                    RM.utils._MODELS[name],
                    cache_root,
                )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found;")

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        import urllib.request
        urllib.request.urlretrieve("https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt", model_path)

    print('load checkpoint from %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')

    # med_config
    if med_config is None:
        med_path = os.path.join(cache_root, "med_config.json")
        if not os.path.exists(med_path) or os.path.getsize(med_path) < 10:
            import urllib.request
            try:
                urllib.request.urlretrieve("https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json", med_path)
                med_config = med_path
            except OSError:
                med_config = RM.ImageReward_download(
                    "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
                    cache_root,
                )
        else:
            med_config = med_path

    model = IRSMC(device=device, med_config=med_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()
    return model
