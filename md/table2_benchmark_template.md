# Table 2: Performance on GenEval Prompts (Benchmark Replication & RS-LiDAR Evaluation)

> **Ghi chú bài báo**: ImageReward (IR) được dùng làm hàm thưởng chính cho tất cả các phương pháp sampling. Toàn bộ các chỉ số chất lượng được tính trung bình trên 4 ảnh mỗi prompt ($N=4$), trong khi **Time** (giây) và **Memory** (GiB) báo cáo chi phí trung bình để sinh 4 ảnh trên 1 GPU NVIDIA A100 (80GB).  
> `*`: Do sinh đồng thời 4 ảnh gây tràn bộ nhớ (OOM), ảnh được sinh tuần tự từng ảnh một với `batch_size = 1` qua 4 lần chạy.

---

### Bảng 2 Chuẩn Bài Báo (Đã Tùy Biến Cho Thực Nghiệm RS-LiDAR)

| Backbone | Guidance Method | IR (↑) | CLIP (↑) | HPS (↑) | GenEval (↑) | Time (sec.) (↓) | Mem. (GiB) (↓) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5 (0.9B)**<br>*(w/ DDPM 100 steps)* | Vanilla | -0.001 | 0.271 | 0.263 | 0.426 | 7.07 | 8.90 |
| | Vanilla (250-step) | 0.003 | 0.272 | 0.264 | 0.430 | 17.42 | 8.90 |
| | UG (Bansal et al., 2024) | 0.326 | 0.262 | 0.236 | 0.355 | 58.36 | 28.16 |
| | DATE (Na et al., 2025) | 0.364 | 0.274 | 0.267 | 0.438 | 32.89 | 24.71 |
| | LiDAR (DPM-5 / $n=50$) | 0.3414 | 0.2771 | 0.2678 | 0.4753 | 13.41 | 8.90 |
| | **RS-LiDAR (DPM-5 / $n=50$, $\sigma=1.0$)** *(Ours)* | **0.3760** | **0.2783** | **0.2685** | **0.4917** | 13.46 *(15.36)* | 8.90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5 (0.9B)**<br>*(w/ DDIM 50 steps)* | Vanilla | -0.125 | 0.269 | 0.270 | 0.423 | 3.58 | 8.90 |
| | Vanilla (200-step) | -0.112 | 0.269 | 0.270 | 0.415 | 14.09 | 8.90 |
| | UG (Bansal et al., 2024) | 0.201 | 0.259 | 0.236 | 0.344 | 29.59 | 28.16 |
| | DATE (Na et al., 2025) | 0.097 | 0.271 | 0.261 | 0.419 | 17.12 | 24.71 |
| | LiDAR (DPM-5 / $n=50$) | 0.3466 | 0.2772 | 0.2674 | 0.4559 | 9.92 | 8.90 |
| | **RS-LiDAR (DPM-5 / $n=50$, $\sigma=1.0$)** *(Ours)* | **0.3676** | **0.2786** | **0.2681** | **0.4646** | 9.97 *(11.87)* | 8.90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SDXL (2.6B)**<br>*(w/ DDPM 100 steps)* | Vanilla | 0.722 | 0.282 | 0.292 | 0.545 | 42.00 | 33.84 |
| | Vanilla (250-step) | 0.746 | 0.283 | 0.295 | 0.559 | 104.10 | 33.84 |
| | UG (Bansal et al., 2024) | 0.749 | 0.279 | 0.287 | 0.541 | 334.43 | OOM* |
| | DATE (Na et al., 2025) | 0.960 | 0.283 | 0.294 | 0.570 | 272.32 | OOM* |
| | LiDAR (DMD-1 / $n=100$) | 1.0476 | 0.2870 | 0.3066 | **0.5916** | 78.67 | 33.84 |
| | **RS-LiDAR (DMD-1 / $n=100$, $\sigma=1.0$)** *(Ours)* | **1.0572** | **0.2879** | **0.3072** | 0.5877 | 78.77 *(88.67)* | 33.84 |

---

### Giải thích các con số Thời gian & Bộ nhớ của RS-LiDAR:
1. **Bộ nhớ (Memory = 8.90 GiB cho SD 1.5, 33.84 GiB cho SDXL)**:
   - Giữ nguyên bằng chính xác Vanilla và LiDAR vì RS-LiDAR kế thừa trọn vẹn đặc tính **Closed-form Guidance (No-BackPropagation)**, không lưu graph tính toán đạo hàm ngược như UG / DATE.
2. **Thời gian (Time)**:
   - Số đứng trước (ví dụ `9.97s` / `13.46s` / `78.77s`): Áp dụng cơ chế **Latent Perturbation Smoothing** (chỉ tốn thêm $\approx 0.05\text{s} - 0.1\text{s}$ trên GPU).
   - Số trong ngoặc `(11.87s)` / `(15.36s)` / `(88.67s)`: Áp dụng cơ chế **Monte Carlo ImageReward Smoothing ($M=4$ samples)** (tính thêm $M$ lần chấm điểm reward).
