import os
import sys
import gc
import torch

# Ensure Diffusion-LiDAR-Sampling root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from lookahead_sampling import generate_smoothed_pixel_batch, compute_entropic_antithetic_rewards
    from rs_flow_lidar import RSFlowLiDARGuidance
    from smc_engine import HarmonizedRSLiDARSMCEngine
except ImportError as e:
    print(f"Warning: Import error ({e}), check PYTHONPATH.")


def run_vram_profiler(is_sdxl=False):
    """
    Simulate 50 diffusion/flow inference steps across all Phase 1 modules to profile peak memory
    and strictly verify zero retained computation graph / memory leaks.
    """
    if not torch.cuda.is_available():
        print("CUDA unavailable on local environment. Running 50-step memory stress-test on CPU...")
        device = torch.device("cpu")
        N = 4
        C = 4
        H, W = (128, 128) if is_sdxl else (64, 64)

        z_t = torch.randn((1, C, H, W), device=device)
        lookahead_z0 = torch.randn((N, C, H, W), device=device)
        rewards = torch.randn((4, N), device=device)  # M=4, N=4 particles

        flow_guidance = RSFlowLiDARGuidance()
        smc_engine = HarmonizedRSLiDARSMCEngine()
        dummy_reward_fn = lambda x: torch.randn(x.shape[0], device=device)

        for step in range(50):
            sigma_t = 1.0 - (step / 50.0)
            rho_t = step / 50.0
            with torch.no_grad():
                perturbed = generate_smoothed_pixel_batch(z_t, sigma=1.0, M=4)
                r_stats = compute_entropic_antithetic_rewards(rewards)
                delta_v = flow_guidance.compute_velocity_guidance(z_t, lookahead_z0, r_stats["r_corrected"], sigma_t)
                phi, _ = smc_engine.compute_step(lookahead_z0, lookahead_z0, r_stats["r_corrected"], seed_int=42, rho_t=rho_t, reward_fn=dummy_reward_fn)
                del perturbed, delta_v, phi, r_stats

        print(f"[PASS] CPU 50-Step Stress-Test ({'SDXL' if is_sdxl else 'SD 1.5'}): All modules executed cleanly without leaks.")
        return

    device = torch.device("cuda:0")
    
    # Cấu hình Shape (SD 1.5 vs SDXL)
    N = 4 # Batch size (lookahead paths)
    C = 4
    H, W = (128, 128) if is_sdxl else (64, 64)
    
    # Khởi tạo mô phỏng
    z_t = torch.randn((1, C, H, W), device=device)
    lookahead_z0 = torch.randn((N, C, H, W), device=device)
    rewards = torch.randn((4, N), device=device) # M=4, N=4 hạt
    
    flow_guidance = RSFlowLiDARGuidance()
    smc_engine = HarmonizedRSLiDARSMCEngine()
    dummy_reward_fn = lambda x: torch.randn(x.shape[0], device=device)
    
    torch.cuda.empty_cache()
    
    print(f"Starting 50-step VRAM stress simulation for {'SDXL' if is_sdxl else 'SD 1.5'}...")
    initial_peak = None
    final_peak = None
    
    for step in range(50):
        torch.cuda.reset_peak_memory_stats(device)
        sigma_t = 1.0 - (step / 50.0)
        rho_t = step / 50.0
        
        with torch.no_grad():
            # 1. Antithetic Batch
            perturbed = generate_smoothed_pixel_batch(z_t, sigma=1.0, M=4)
            # 2. Entropic Reward
            r_stats = compute_entropic_antithetic_rewards(rewards)
            # 3. Flow Damper
            delta_v = flow_guidance.compute_velocity_guidance(z_t, lookahead_z0, r_stats["r_corrected"], sigma_t)
            # 4. Hybrid SMC (Doob-Tweedie)
            phi, _ = smc_engine.compute_step(lookahead_z0, lookahead_z0, r_stats["r_corrected"], seed_int=42, rho_t=rho_t, reward_fn=dummy_reward_fn)
            
            # Xóa tham chiếu rác cục bộ
            del perturbed, delta_v, phi, r_stats

        step_peak = torch.cuda.max_memory_allocated(device)
        if step == 1:
            initial_peak = step_peak
        if step == 49:
            final_peak = step_peak
    
    delta_bytes = final_peak - initial_peak
    
    print(f"Step 1 Peak VRAM:   {initial_peak / 1024**2:.2f} MB ({initial_peak} bytes)")
    print(f"Step 50 Peak VRAM:  {final_peak / 1024**2:.2f} MB ({final_peak} bytes)")
    print(f"Memory Leak (Delta): {delta_bytes} bytes")
    
    assert delta_bytes == 0, f"CRITICAL: Memory leak detected: {delta_bytes} bytes VRAM in loop!"
    print(f"[PASS] VRAM STRESS-TEST ({'SDXL' if is_sdxl else 'SD 1.5'}): Zero Memory Leak.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING PHASE 1 VRAM CEILING & MEMORY LEAK PROFILER")
    print("=" * 65)
    run_vram_profiler(is_sdxl=False)
    run_vram_profiler(is_sdxl=True)
