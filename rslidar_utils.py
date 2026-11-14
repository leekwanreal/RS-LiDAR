"""Shared helpers for the RS-LiDAR command-line entrypoints.

- `setup_runtime()` applies small, import-time compatibility shims (optional `wandb`,
  older `protobuf`, newer `transformers`) that the ImageReward / BLIP reward model needs.
  It must run before `diffusers` or `ImageReward` are imported.
- `parse_prompt_indices` / `load_geneval_metadata` select GenEval prompts while keeping each
  prompt's original index (`original_idx`), so output folders always match GenEval metadata.
- `str2bool` is the argparse converter used by boolean flags such as `--use_crn True`.
"""
import argparse
import importlib.machinery
import json
import os
import sys
import types

# Fixed stratified subset of 100 GenEval prompts (all six task categories). Used when
# `prompt_files/geneval_100_representative.json` is not available.
STRATIFIED_100 = [
    0, 5, 11, 16, 21, 26, 32, 37, 42, 47, 53, 58, 63, 68, 74, 79, 80, 86, 92, 99,
    105, 111, 117, 123, 130, 136, 142, 148, 154, 161, 167, 173, 178, 179, 184, 190, 195, 200, 205, 211,
    216, 221, 226, 232, 237, 242, 247, 253, 258, 259, 265, 271, 277, 282, 288, 294, 300, 306, 312, 318,
    323, 329, 335, 341, 347, 352, 353, 359, 365, 372, 378, 384, 390, 396, 403, 409, 415, 421, 428, 434,
    440, 446, 452, 453, 459, 465, 472, 478, 484, 490, 496, 503, 509, 515, 521, 527, 533, 540, 546, 552,
]


def setup_runtime():
    """Apply import-time compatibility shims. Safe to call more than once."""
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # ImageReward imports `wandb` at module level; provide an empty stub when it is not installed.
    if "wandb" not in sys.modules:
        try:
            import wandb  # noqa: F401
        except ImportError:
            stub = types.ModuleType("wandb")
            stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
            stub.__file__ = "wandb"
            sys.modules["wandb"] = stub

    # Older protobuf releases lack `runtime_version`, which recent generated code references.
    try:
        import google.protobuf
        if not hasattr(google.protobuf, "runtime_version"):
            class _RuntimeVersion:
                DOMAIN = "protobuf"
            google.protobuf.runtime_version = _RuntimeVersion
    except ImportError:
        pass

    # Some peft/diffusers releases import `EncoderDecoderCache`, absent from older transformers.
    try:
        import transformers
        if not hasattr(transformers, "EncoderDecoderCache"):
            class EncoderDecoderCache:
                pass
            transformers.EncoderDecoderCache = EncoderDecoderCache
    except ImportError:
        pass

    # Restore transformers utilities that the ImageReward BLIP code still imports.
    try:
        from fkd_diffusers.patch_transformers import patch_transformers_for_imagereward
    except ImportError:
        return
    patch_transformers_for_imagereward()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_prompt_indices(indices_arg):
    """Parse `--prompt_indices`.

    Accepts `None`/`"all"`/`"none"` (no filtering), `"stratified_100"`, a path to a JSON list,
    a range `"a-b"` (inclusive) or a comma-separated list `"1,5,9"`.
    """
    if not indices_arg or str(indices_arg).strip().lower() in ["none", "all", ""]:
        return None
    if isinstance(indices_arg, (list, tuple)):
        return list(indices_arg)
    indices_str = str(indices_arg).strip()
    if indices_str.lower() in ["stratified_100", "100_representative"]:
        repr_path = "prompt_files/geneval_100_representative.json"
        if os.path.exists(repr_path):
            with open(repr_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return list(STRATIFIED_100)
    if os.path.exists(indices_str):
        with open(indices_str, "r", encoding="utf-8") as f:
            return json.load(f)
    if "-" in indices_str and "," not in indices_str:
        parts = indices_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]


def load_geneval_metadata(prompt_path, max_prompts=None, prompt_indices=None):
    """Load GenEval prompts from `.json` / `.jsonl`, tagging each with its `original_idx`."""
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    assert isinstance(data, list)
    if "prompt" not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"
        for item in data:
            item["prompt"] = item["text"]
    for idx, item in enumerate(data):
        if "original_idx" not in item:
            item["original_idx"] = idx
    if prompt_indices is not None and len(prompt_indices) > 0:
        data = [data[i] for i in prompt_indices if i < len(data)]
    elif max_prompts is not None:
        data = data[:max_prompts]
    return data
