#!/usr/bin/env python3
"""Run the official GenEval benchmark (Mask2Former Swin-S detector) on Phase 2 outputs.

Usage:
    python run_geneval_eval.py --target_dir Target_samples/SD15_LiDAR_Table2_Replication
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenEval on generated LiDAR images")
    parser.add_argument("--target_dir", type=str, default="Target_samples/SD15_LiDAR_Table2_Replication", help="Path to Target_samples directory containing prompt folders")
    parser.add_argument("--geneval_dir", type=str, default="geneval", help="Path to clone/run GenEval")
    parser.add_argument("--work_dir", type=str, default=".", help="Working directory for GenEval inputs and results")
    parser.add_argument("--output_jsonl", type=str, default="geneval_results.jsonl", help="Output results file")
    return parser.parse_args()


def prepare_geneval_input(target_dir, geneval_input_dir):
    """Link `<target_dir>/<prompt_idx>/samples/*.png` to `<geneval_input_dir>/<prompt_idx:05d>/*.png`."""
    os.makedirs(geneval_input_dir, exist_ok=True)
    prompt_folders = sorted(glob.glob(os.path.join(target_dir, "[0-9]*")))
    if not prompt_folders:
        prompt_folders = sorted(glob.glob(os.path.join(target_dir, "**", "[0-9]*"), recursive=True))

    print(f"Found {len(prompt_folders)} prompt folders in: {target_dir}")
    total_images = 0

    for p_dir in prompt_folders:
        p_name = os.path.basename(p_dir)
        try:
            p_idx = int(p_name)
        except ValueError:
            continue

        dest_dir = os.path.join(geneval_input_dir, f"{p_idx:05d}")
        os.makedirs(dest_dir, exist_ok=True)

        imgs = sorted(glob.glob(os.path.join(p_dir, "samples", "*.png")))
        if not imgs:
            imgs = sorted(glob.glob(os.path.join(p_dir, "*.png")))
            imgs = [img for img in imgs if not img.endswith("grid.png")]

        for img_idx, img_path in enumerate(imgs):
            dest_path = os.path.join(dest_dir, f"{img_idx:05d}.png")
            if not os.path.exists(dest_path):
                try:
                    os.symlink(os.path.abspath(img_path), dest_path)
                except OSError:  # symlinks unavailable (e.g. Windows without privileges)
                    shutil.copyfile(img_path, dest_path)
            total_images += 1

    print(f"Prepared {total_images} images in: {geneval_input_dir}")
    return total_images


def setup_geneval_env(geneval_dir, work_dir):
    """Clone GenEval and download the Mask2Former checkpoint if needed."""
    print("\nChecking GenEval sources and the Mask2Former detector...")

    if not os.path.exists(geneval_dir):
        print(f"Cloning GenEval into {geneval_dir}...")
        subprocess.run(["git", "clone", "-q", "https://github.com/djghosh13/geneval.git", geneval_dir], check=True)

    models_dir = os.path.join(geneval_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_name = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    ckpt_path = os.path.join(models_dir, ckpt_name)

    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000000:
        url = "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
        print(f"Downloading Mask2Former checkpoint ({url})...")
        subprocess.run(["wget", "-q", "-c", url, "-O", ckpt_path], check=True)
        print(f"Checkpoint downloaded: {ckpt_path}")
    else:
        print(f"Using existing Mask2Former checkpoint: {ckpt_path}")

    return models_dir


def run_evaluation(geneval_dir, geneval_input_dir, models_dir, output_jsonl):
    """Run GenEval's evaluate_images.py followed by summary_scores.py."""
    eval_script = os.path.join(geneval_dir, "evaluation", "evaluate_images.py")
    summary_script = os.path.join(geneval_dir, "evaluation", "summary_scores.py")

    print("\nRunning GenEval evaluation with Mask2Former...")
    cmd_eval = [
        sys.executable, eval_script,
        geneval_input_dir,
        "--outfile", output_jsonl,
        "--model-path", models_dir
    ]
    subprocess.run(cmd_eval, check=True)

    print("\n" + "=" * 80)
    print("GenEval summary")
    print("=" * 80)
    cmd_summary = [sys.executable, summary_script, output_jsonl]
    subprocess.run(cmd_summary, check=True)
    print("=" * 80)
    print(f"Per-image results saved to: {output_jsonl}")


def main():
    args = parse_args()
    geneval_input_dir = os.path.join(args.work_dir, "geneval_input")
    output_jsonl_path = os.path.join(args.work_dir, args.output_jsonl)

    total_imgs = prepare_geneval_input(args.target_dir, geneval_input_dir)
    if total_imgs == 0:
        print(f"No images found in: {args.target_dir}. Please check the path.")
        sys.exit(1)

    models_dir = setup_geneval_env(args.geneval_dir, args.work_dir)
    run_evaluation(args.geneval_dir, geneval_input_dir, models_dir, output_jsonl_path)


if __name__ == "__main__":
    main()
