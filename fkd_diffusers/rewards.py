import os
import shutil
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2

# Auto-fix missing bpe_simple_vocab_16e6.txt.gz in hpsv2 wheel on Python 3.11/3.12
try:
    _hps_dir = os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip")
    _bpe_target = os.path.join(_hps_dir, "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(_bpe_target):
        os.makedirs(_hps_dir, exist_ok=True)
        _clip_bpe = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(_clip_bpe):
            shutil.copyfile(_clip_bpe, _bpe_target)
        else:
            urllib.request.urlretrieve(
                "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz",
                _bpe_target,
            )
except Exception as _e:
    pass

from image_reward_utils import rm_load

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "AS": None,
    "hps": None
}


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score", diff=False):
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts, diff=diff)
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    else:
        raise ValueError(f"Unknown metric: {reward_name}")


# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    global REWARDS_DICT
    import numpy as np
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    try:
        if REWARDS_DICT["hps"] is None:
            cache_dir = os.path.expanduser("~/.cache/hpsv2")
            os.makedirs(cache_dir, exist_ok=True)
            ckpt_path = os.path.join(cache_dir, "HPS_v2.1_compressed.pt")
            if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000000:
                try:
                    urllib.request.urlretrieve(
                        "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt",
                        ckpt_path,
                    )
                except Exception as _e_dl:
                    pass
            try:
                from hpsv2.img_score import hpsv2_load
                import inspect
                sig = inspect.signature(hpsv2_load)
                if "device" in sig.parameters:
                    model, tokenizer = hpsv2_load(hps_version="v2.1", device=device)
                else:
                    model, tokenizer = hpsv2_load(hps_version="v2.1")
                if hasattr(model, "to"):
                    model = model.to(device)
                REWARDS_DICT["hps"] = (model, tokenizer)
            except Exception as _load_err:
                pass

        if REWARDS_DICT["hps"] is not None:
            from hpsv2.img_score import hpsv2_score_
            model, tokenizer = REWARDS_DICT["hps"]
            scores = []
            for i, image in enumerate(images):
                score = hpsv2_score_(image, prompts[i], hps_version="v2.1", model=model, tokenizer=tokenizer)
                if isinstance(score, (list, tuple, np.ndarray, torch.Tensor)):
                    val = float(score[0])
                else:
                    val = float(score)
                scores.append(val)
            return scores

        # Standard official hpsv2 fallback
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        if isinstance(scores, list):
            return [float(score) for score in scores]
        else:
            return [float(scores)]
    except Exception as e:
        print(f"Warning computing HPS on {device}: {e}")
        return [0.0] * len(images)


# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
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
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()


# Compute ImageReward
def do_image_reward(*, images, prompts, diff=False):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    if diff:
        image_reward_result = REWARDS_DICT["ImageReward"].differentiable_score_batched(prompts, images)
    else:

        with torch.no_grad():
            image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
    return image_reward_result


# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


def do_AS(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["AS"] is None:
        weight_path = "sac+logos+ava1-l14-linearMSE.pth"
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
            REWARDS_DICT["AS"] = AestheticScore(download_root=".", device="cuda" if torch.cuda.is_available() else "cpu")
            REWARDS_DICT["AS"].mlp.load_state_dict(state_dict, strict=False)
            REWARDS_DICT["AS"].mlp.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"Warning initializing AestheticScore: {e}")
            return [0.0] * len(images)
    try:
        with torch.no_grad():
            as_result = [
                REWARDS_DICT["AS"].score(images[i])
                for i, prompt in enumerate(prompts)
            ]
        return as_result
    except Exception as e:
        print(f"Warning computing AS score: {e}")
        return [0.0] * len(images)



class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
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
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False,
                                                     download_root=download_root)
        self.mlp = MLP(768)

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, pil_image):

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image)).float()

        # score
        rewards = self.mlp(image_features)

        return rewards.detach().cpu().numpy().item()