
import os
import json
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import random
import clip


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


class gen_lookahead_samples(Dataset):
    def __init__(self, directory, top_k):
        print("top_k:", top_k)
        self.dataset_path = f"Lookahead_samples/{directory}"
        data = {}
        if os.path.exists(self.dataset_path):
            prompt_dirs = sorted([d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d)) and d.isdigit()])
            for d in prompt_dirs:
                latent_path = f"{self.dataset_path}/{d}/samples/latent.pt"
                results_path = f"{self.dataset_path}/{d}/results.json"
                if not os.path.exists(latent_path) or not os.path.exists(results_path):
                    continue
                try:
                    latent = torch.load(latent_path, map_location="cpu")
                    with open(results_path, "r") as f:
                        label = json.load(f)
                    reward = label["ImageReward"]["result"]
                    prompt = label["prompt"]

                    random.seed(42)
                    k = min(top_k, len(latent))
                    indices = random.sample(range(len(latent)), k)
                    latents = [latent[i] for i in indices]
                    rewards = [reward[i] for i in indices]
                    prompts = [prompt[i] for i in indices]
                    prompt_idx = int(d)
                    datapoint = {
                        "latents": torch.stack(latents),
                        "rewards": torch.tensor(rewards),
                        "prompts": prompts,
                        "prompt_idx": prompt_idx
                    }
                    data[prompt_idx] = datapoint
                except Exception as e:
                    print(f"Error loading lookahead sample for prompt {d}: {e}")
        self.data = data
        print(f"Loaded {len(self.data)} lookahead prompt datasets.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __contains__(self, idx):
        return idx in self.data
