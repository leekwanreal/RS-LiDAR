"""CLI compatibility: every flag passed by the experiment notebooks must still be accepted.

Each entrypoint is invoked with `--help` from the repository root and its help text must list
all flags the notebooks use (Table 2, Table 8, FLUX, Phase-2 steps, benchmark, Lipschitz,
landscape, adversarial/threshold and weakness notebooks).
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOK_FLAGS = {
    "lookahead_sampling.py": [
        "--seed", "--num_particles", "--num_inference_steps", "--model_name", "--guidance_reward_fn",
        "--metrics_to_compute", "--prompt_path", "--max_prompt", "--run_name", "--resume", "--prompt_indices",
        "--reuse_latents_from", "--use_smoothing", "--sigma", "--num_mc_samples", "--M", "--use_crn",
        "--overwrite", "--save_individual_images", "--noise_mode", "--gpu_id", "--num_shards", "--shard_id",
        "--beta_risk", "--use_pareto", "--pareto_rho", "--loop_smoothing",
    ],
    "LiDAR_sampling.py": [
        "--seed", "--model_name", "--num_particles", "--num_inference_steps", "--eta", "--use_rag",
        "--lookahead_path", "--top_k", "--scale", "--lmbda", "--resample_t_end", "--prompt_path", "--max_prompt",
        "--guidance_reward_fn", "--metrics_to_compute", "--save_individual_images", "--run_name", "--resume",
        "--prompt_indices", "--overwrite", "--method", "--s_flow", "--guidance_start", "--guidance_end",
        "--flux_debug", "--output_dir", "--use_smc", "--use_smc_smoothing", "--smc_sigma",
        "--smc_num_mc_samples", "--resample_frequency", "--FK_lmbda", "--FK_resample_t_start",
        "--FK_resample_t_end", "--smc_terminal_resample", "--smc_smoothing_mode", "--smc_use_crn",
        "--smc_anneal_sigma", "--smc_anneal_schedule", "--smc_sigma_min", "--gpu_id", "--num_shards",
        "--shard_id",
    ],
    "experiments.target_steps_ablation": [
        "--seed", "--model_name", "--num_particles", "--num_inference_steps", "--eval_checkpoint_steps", "--eta",
        "--use_rag", "--lookahead_path", "--top_k", "--scale", "--lmbda", "--resample_t_end", "--prompt_path",
        "--max_prompt", "--prompt_indices", "--guidance_reward_fn", "--metrics_to_compute",
        "--save_individual_images", "--run_name", "--resume", "--overwrite",
    ],
    "experiments.efficiency_benchmark": [
        "--setting", "--method", "--prompts_per_category", "--num_prompts", "--lookahead_steps", "--sweep_steps",
        "--sigma", "--num_mc_samples", "--reward_batch_size", "--warmup", "--no_warmup", "--output_dir",
        "--drive_backup_dir",
    ],
    "experiments.lipschitz_empirical": [
        "--prompts_file", "--num_prompts", "--num_particles", "--sigma1", "--perturbation_type", "--sigma2",
        "--sigma2_list", "--num_mc_samples", "--reward_models", "--output_dir", "--num_shards", "--shard_id",
        "--gpu_id", "--use_crn", "--no_crn", "--overwrite", "--merge_shards",
    ],
    "experiments.landscape_slicing": [
        "--prompts_file", "--prompt_indices", "--curated_selection", "--categories", "--prompts_per_category",
        "--anchor_indices", "--random_sample", "--no_random_sample", "--seed", "--force_rerun_indices", "--lmbda",
        "--top_k", "--num_lookahead", "--lookahead_steps", "--guidance_scale", "--particle_scale", "--target_steps",
        "--sigma", "--mc_samples", "--slicing_points", "--slicing_mode", "--slicing_margin", "--slicing_range",
        "--slicing_mc_samples", "--output_dir", "--num_shards", "--shard_id", "--gpu_id", "--use_crn", "--no_crn",
        "--overwrite", "--resume", "--merge_shards",
    ],
    "experiments.adversarial_reward": [
        "--num_particles", "--sigma1", "--sigma1_list", "--perturbation_type", "--sigma2", "--sigma2_list",
        "--m_list", "--primary_m", "--reward_mode", "--output_dir", "--num_shards", "--shard_id", "--gpu_id",
        "--run_case_studies", "--use_crn", "--no_crn", "--overwrite", "--merge_shards",
    ],
    "experiments.threshold_reward": [
        "--num_particles", "--sigma1", "--sigma1_list", "--perturbation_type", "--sigma2", "--sigma2_list",
        "--m_list", "--primary_m", "--output_dir", "--num_shards", "--shard_id", "--gpu_id", "--use_crn",
        "--no_crn", "--overwrite", "--merge_shards",
    ],
    "experiments.lidar_weaknesses": [
        "--test", "--num_prompts", "--num_particles", "--sigma", "--tune_sigma", "--sigmas", "--M", "--prompt_path",
        "--output_dir", "--num_shards", "--shard_id", "--gpu_id", "--use_hps", "--use_aesthetic", "--overwrite",
    ],
}


def _help_text(entry):
    cmd = [sys.executable, entry, "--help"] if entry.endswith(".py") else [sys.executable, "-m", entry, "--help"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", COLUMNS="400")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


class TestNotebookCliCompat(unittest.TestCase):
    def test_entrypoints_accept_notebook_flags(self):
        for entry, flags in NOTEBOOK_FLAGS.items():
            with self.subTest(entry=entry):
                code, text = _help_text(entry)
                self.assertEqual(code, 0, f"{entry} --help failed:\n{text[-2000:]}")
                missing = [f for f in flags if f not in text.split()
                           and not any(tok.startswith(f + ",") or tok.startswith(f + "=") for tok in text.split())]
                self.assertEqual(missing, [], f"{entry} is missing notebook flags: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
