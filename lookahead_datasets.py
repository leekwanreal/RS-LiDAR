"""Phase 1 lookahead pool loader used by Phase 2 (`LiDAR_sampling.py`).

`gen_lookahead_samples(directory, top_k)` reads every `<prompt_idx>/samples/latent.pt` and
`<prompt_idx>/results.json` written by `lookahead_sampling.py`, and keeps, per prompt, a random
subset of `top_k` lookahead latents with their rewards (`<reward_key>.result`, i.e. the smoothed
rewards when Phase 1 ran with `--use_smoothing`).
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset


class gen_lookahead_samples(Dataset):
    def __init__(self, directory, top_k, reward_key="ImageReward"):
        self.reward_key = reward_key
        print("top_k:", top_k, "| reward_key:", reward_key)
        clean_dir = directory.rstrip("/\\")
        basename = os.path.basename(clean_dir)
        candidates = [
            clean_dir,
            f"Lookahead_samples/{clean_dir}",
            f"Lookahead_samples/{basename}",
            f"Lookahead_samples/{basename}_100p",
        ]
        # Prefer the first candidate that contains prompt folders, else the first existing one.
        chosen_path = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                p_dirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d)) and d.isdigit()]
                if len(p_dirs) > 0:
                    chosen_path = cand
                    break
                elif chosen_path is None:
                    chosen_path = cand
        if chosen_path is not None:
            self.dataset_path = chosen_path
            print(f"Lookahead dataset found at: {self.dataset_path}")
        else:
            self.dataset_path = f"Lookahead_samples/{basename}"
            print(f"Lookahead dataset '{directory}' not found. Defaulting to: {self.dataset_path}")
        data = {}
        if os.path.exists(self.dataset_path):
            prompt_dirs = sorted([d for d in os.listdir(self.dataset_path)
                                  if os.path.isdir(os.path.join(self.dataset_path, d)) and d.isdigit()])
            for d in prompt_dirs:
                latent_path = f"{self.dataset_path}/{d}/samples/latent.pt"
                results_path = f"{self.dataset_path}/{d}/results.json"
                if not os.path.exists(latent_path) or not os.path.exists(results_path):
                    continue
                try:
                    latent = torch.load(latent_path, map_location="cpu")
                    with open(results_path, "r") as f:
                        label = json.load(f)
                    if self.reward_key in label and "result" in label[self.reward_key]:
                        metric_key = self.reward_key
                    elif "ImageReward" in label and "result" in label["ImageReward"]:
                        metric_key = "ImageReward"
                    else:
                        first_metric = [k for k in label.keys() if isinstance(label[k], dict) and "result" in label[k]]
                        metric_key = first_metric[0] if first_metric else None

                    if metric_key is not None:
                        reward = label[metric_key]["result"]
                    else:
                        reward = [0.0] * len(latent)

                    prompt = label["prompt"]

                    random.seed(42)
                    k = min(top_k, len(latent))
                    indices = random.sample(range(len(latent)), k)
                    latents = [latent[i] for i in indices]
                    rewards = [reward[i] for i in indices]
                    prompts = [prompt[i] for i in indices]
                    prompt_idx = int(d)
                    data[prompt_idx] = {
                        "latents": torch.stack(latents),
                        "rewards": torch.tensor(rewards),
                        "prompts": prompts,
                        "prompt_idx": prompt_idx,
                    }
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
