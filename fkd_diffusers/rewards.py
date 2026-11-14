"""Reward models used for lookahead scoring, SMC potentials and evaluation.

ImageReward (default guidance reward), CLIP score / CLIP diversity, HPSv2 (HumanPreference) and the
LAION aesthetic predictor (AS). Loaded models are cached in `REWARDS_DICT`. `clip`, `hpsv2` and
`filelock` are optional: when missing, the corresponding metric returns zeros.
"""
import os
import shutil
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from filelock import FileLock
except ImportError:
    class FileLock:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

try:
    import clip
except ImportError:
    clip = None

try:
    import hpsv2
    # Some hpsv2 wheels miss bpe_simple_vocab_16e6.txt.gz; copy it from CLIP or download it.
    try:
        _hps_dir = os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip")
        _bpe_target = os.path.join(_hps_dir, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(_bpe_target):
            os.makedirs(_hps_dir, exist_ok=True)
            if clip is not None:
                _clip_bpe = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
                if os.path.exists(_clip_bpe):
                    shutil.copyfile(_clip_bpe, _bpe_target)
            if not os.path.exists(_bpe_target):
                urllib.request.urlretrieve(
                    "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz",
                    _bpe_target,
                )
    except OSError as e:
        print(f"Warning: could not provide the hpsv2 BPE vocabulary: {e}")
except ImportError:
    hpsv2 = None

from fkd_diffusers.image_reward_utils import rm_load  # noqa: E402

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "AS": None,
    "hps": None
}


def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    """Score `images` with the guidance reward `reward_name`."""
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    else:
        raise ValueError(f"Unknown metric: {reward_name}")


# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    global REWARDS_DICT
    if images is None or len(images) == 0:
        return []
    if hpsv2 is None:
        return [0.0] * len(images)
    import numpy as np
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    try:
        prompt_texts = []
        for i in range(len(images)):
            if isinstance(prompts, (list, tuple)):
                p = prompts[i] if i < len(prompts) else prompts[0]
            else:
                p = prompts
            while isinstance(p, (list, tuple)):
                p = p[0]
            prompt_texts.append(str(p))

        cache_dir = os.path.expanduser("~/.cache/hpsv2")
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, "hpsv2_download.lock")
        hps_weight_file = os.path.join(cache_dir, "HPS_v2.1_compressed.pt")
        if not os.path.exists(hps_weight_file):
            with FileLock(lock_path):
                if not os.path.exists(hps_weight_file):
                    try:
                        from PIL import Image as _PILImg
                        _dummy = _PILImg.new("RGB", (224, 224))
                        _ = hpsv2.score(_dummy, "warmup", hps_version="v2.1")
                    except Exception as e:
                        print(f"Warning: HPSv2 warm-up (weight download) failed: {e}")

        scores = []
        # Fast path: when all images share one prompt, score the whole batch in a single call.
        if len(prompt_texts) > 0 and len(set(prompt_texts)) <= 1:
            try:
                raw_res = hpsv2.score(images, prompt_texts[0], hps_version="v2.1")
                if isinstance(raw_res, (list, tuple, np.ndarray, torch.Tensor)):
                    scores = [float(x) for x in raw_res]
                else:
                    scores = [float(raw_res)]
            except Exception:
                scores = []

        # Fall back to per-image scoring if the batched call failed or returned the wrong count.
        if len(scores) != len(images):
            scores = []
            for i, img in enumerate(images):
                p_text = prompt_texts[i]
                score = hpsv2.score(img, p_text, hps_version="v2.1")
                if isinstance(score, (list, tuple, np.ndarray, torch.Tensor)):
                    val = float(score[0])
                else:
                    val = float(score)
                scores.append(val)
        return scores
    except Exception as e:
        print(f"Warning computing HPS on {device}: {e}")
        return [0.0] * len(images)


# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if images is None or len(images) == 0:
        return [], 0.0
    if clip is None:
        return [0.0] * len(images), 0.0
    dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    try:
        if REWARDS_DICT["Clip-Score"] is None:
            REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=os.path.expanduser("~/.cache/clip"), device=dev)
        with torch.no_grad():
            arr_clip_result = []
            arr_img_features = []
            for i, prompt in enumerate(prompts):
                clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                    prompt, images[i], return_feature=True
                )

                arr_clip_result.append(clip_result.item())
                arr_img_features.append(feature_vect['image'])

        # calculate diversity by computing pairwise similarity between image features
        diversity = torch.zeros(len(images), len(images))
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
                diversity[j, i] = diversity[i, j]
        n_samples = len(images)
        diversity = diversity.sum() / (n_samples * (n_samples - 1)) if n_samples > 1 else 0.0

        return arr_clip_result, diversity.item() if isinstance(diversity, torch.Tensor) else float(diversity)
    except Exception as e:
        print(f"Warning computing CLIP diversity on {dev}: {e}")
        return [0.0] * len(images), 0.0


# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if images is None or len(images) == 0:
        return []

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

    dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0", device=dev)

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
    return image_reward_result


# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if images is None or len(images) == 0:
        return []
    if clip is None:
        return [0.0] * len(images)
    dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    try:
        if REWARDS_DICT["Clip-Score"] is None:
            REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=os.path.expanduser("~/.cache/clip"), device=dev)
        with torch.no_grad():
            clip_result = [
                REWARDS_DICT["Clip-Score"].score(prompts[i] if i < len(prompts) else prompts[0], images[i])
                for i in range(len(images))
            ]
        return clip_result
    except Exception as e:
        print(f"Warning computing CLIP score: {e}")
        return [0.0] * len(images)


def do_AS(*, images, prompts):
    global REWARDS_DICT
    dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    if REWARDS_DICT["AS"] is None:
        weight_path = "sac+logos+ava1-l14-linearMSE.pth"
        if not os.path.exists(weight_path):
            with FileLock("sac_aesthetic_download.lock"):
                if not os.path.exists(weight_path):
                    try:
                        urllib.request.urlretrieve(
                            "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
                            weight_path
                        )
                    except Exception as e:
                        print(f"Warning downloading Aesthetic weights: {e}")
        try:
            state_dict = torch.load(weight_path, map_location='cpu')
            REWARDS_DICT["AS"] = AestheticScore(download_root=os.path.expanduser("~/.cache/clip"), device=dev)
            REWARDS_DICT["AS"].mlp.load_state_dict(state_dict, strict=False)
            REWARDS_DICT["AS"].mlp.to(dev)
        except Exception as e:
            print(f"Warning loading Aesthetic predictor: {e}")
            return None
    try:
        with torch.no_grad():
            as_result = [
                REWARDS_DICT["AS"].score(images[i])
                for i in range(len(images))
            ]
        return as_result
    except Exception as e:
        print(f"Warning computing AS score: {e}")
        return [0.0] * len(images)


class CLIPScore(nn.Module):
    def __init__(self, download_root=None, device='cpu'):
        super().__init__()
        self.device = device
        cache_dir = download_root or os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, "clip_vit_l14.lock")
        
        with FileLock(lock_path):
            self.clip_model, self.preprocess = clip.load(
                "ViT-L/14", device=self.device, jit=False, download_root=cache_dir
            )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode with exact dtype matching
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        model_dtype = next(self.clip_model.parameters()).dtype
        image = image.to(dtype=model_dtype)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()


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
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScore(nn.Module):
    def __init__(self, download_root=None, device='cpu'):
        super().__init__()
        self.device = device
        cache_dir = download_root or os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, "clip_vit_l14.lock")
        with FileLock(lock_path):
            self.clip_model, self.preprocess = clip.load(
                "ViT-L/14", device=self.device, jit=False, download_root=cache_dir
            )
        self.mlp = MLP(768)

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(self.clip_model)

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, pil_image):

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image)).float()

        # score
        rewards = self.mlp(image_features)

        return rewards.detach().cpu().numpy().item()