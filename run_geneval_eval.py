#!/usr/bin/env python3
"""
Automated GenEval Benchmark Evaluation for LiDAR Sampling
Usage:
    python run_geneval_eval.py --target_dir Target_samples/SD15_LiDAR_Table2_Replication
"""

import os
import sys
import glob
import json
import shutil
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenEval on generated LiDAR images")
    parser.add_argument("--target_dir", type=str, default="Target_samples/SD15_LiDAR_Table2_Replication", help="Path to Target_samples directory containing prompt folders")
    parser.add_argument("--geneval_dir", type=str, default="/kaggle/working/geneval", help="Path to clone/run GenEval")
    parser.add_argument("--work_dir", type=str, default="/kaggle/working", help="Working directory on Kaggle/Colab")
    parser.add_argument("--output_jsonl", type=str, default="geneval_results.jsonl", help="Output results file")
    return parser.parse_args()


def prepare_geneval_input(target_dir, geneval_input_dir):
    """Chuẩn hóa đường dẫn ảnh Target_samples/{prompt_idx}/samples/*.png thành geneval_input/{prompt_idx:05d}/*.png"""
    os.makedirs(geneval_input_dir, exist_ok=True)
    prompt_folders = sorted(glob.glob(os.path.join(target_dir, "[0-9]*")))
    if not prompt_folders:
        prompt_folders = sorted(glob.glob(os.path.join(target_dir, "**", "[0-9]*"), recursive=True))

    print(f"🔍 Tìm thấy {len(prompt_folders)} thư mục prompt đã sinh từ: {target_dir}")
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
                except Exception:
                    shutil.copyfile(img_path, dest_path)
            total_images += 1

    print(f"✅ Đã chuẩn bị {total_images} ảnh sẵn sàng tại: {geneval_input_dir}")
    return total_images


def setup_geneval_env(geneval_dir, work_dir):
    """Cài đặt và tải các trọng số Mask2Former nếu chưa có"""
    print("\n📦 Đang kiểm tra mã nguồn GenEval & mô hình phát hiện Mask2Former...")

    if not os.path.exists(geneval_dir):
        print(f"📥 Đang clone repository GenEval vào {geneval_dir}...")
        subprocess.run(["git", "clone", "-q", "https://github.com/djghosh13/geneval.git", geneval_dir], check=True)

    models_dir = os.path.join(geneval_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_name = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    ckpt_path = os.path.join(models_dir, ckpt_name)

    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000000:
        url = "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
        print(f"⬇️ Đang tải checkpoint Mask2Former ({url})...")
        subprocess.run(["wget", "-q", "-c", url, "-O", ckpt_path], check=True)
        print(f"✅ Đã tải xong checkpoint: {ckpt_path}")
    else:
        print(f"✅ Đã có sẵn checkpoint Mask2Former tại: {ckpt_path}")

    return models_dir


def run_evaluation(geneval_dir, geneval_input_dir, models_dir, output_jsonl):
    """Chạy evaluate_images.py và summary_scores.py"""
    eval_script = os.path.join(geneval_dir, "evaluation", "evaluate_images.py")
    summary_script = os.path.join(geneval_dir, "evaluation", "summary_scores.py")

    print("\n🚀 BẮT ĐẦU CHẠY ĐÁNH GIÁ GENEVAL BẰNG MASK2FORMER...")
    cmd_eval = [
        sys.executable, eval_script,
        geneval_input_dir,
        "--outfile", output_jsonl,
        "--model-path", models_dir
    ]
    subprocess.run(cmd_eval, check=True)

    print("\n" + "="*80)
    print("📊 BẢNG TỔNG KẾT ĐIỂM SỐ GENEVAL BENCHMARK")
    print("="*80)
    cmd_summary = [sys.executable, summary_script, output_jsonl]
    subprocess.run(cmd_summary, check=True)
    print("="*80)
    print(f"💾 File chi tiết từng ảnh đã được lưu tại: {output_jsonl}")


def main():
    args = parse_args()
    geneval_input_dir = os.path.join(args.work_dir, "geneval_input")
    output_jsonl_path = os.path.join(args.work_dir, args.output_jsonl)

    total_imgs = prepare_geneval_input(args.target_dir, geneval_input_dir)
    if total_imgs == 0:
        print(f"❌ Không tìm thấy ảnh nào trong: {args.target_dir}. Vui lòng kiểm tra lại đường dẫn!")
        sys.exit(1)

    models_dir = setup_geneval_env(args.geneval_dir, args.work_dir)
    run_evaluation(args.geneval_dir, geneval_input_dir, models_dir, output_jsonl_path)


if __name__ == "__main__":
    main()
