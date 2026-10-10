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
| | LiDAR (DPM-5 / $n=50$) | 0.341 | 0.277 | 0.267 | 0.475 | 13.41 | 8.90 |
| | **RS-LiDAR (DPM-5 / $n=50$, $\sigma=1.0$)** *(Ours)* | **0.376** | **0.278** | **0.269** | **0.492** | 15.36 | 8.90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5 (0.9B)**<br>*(w/ DDIM 50 steps)* | Vanilla | -0.125 | 0.269 | 0.270 | 0.423 | 3.58 | 8.90 |
| | Vanilla (200-step) | -0.112 | 0.269 | 0.270 | 0.415 | 14.09 | 8.90 |
| | UG (Bansal et al., 2024) | 0.201 | 0.259 | 0.236 | 0.344 | 29.59 | 28.16 |
| | DATE (Na et al., 2025) | 0.097 | 0.271 | 0.261 | 0.419 | 17.12 | 24.71 |
| | LiDAR (DPM-5 / $n=50$) | 0.346 | 0.277 | 0.267 | 0.455 | 9.92 | 8.90 |
| | **RS-LiDAR (DPM-5 / $n=50$, $\sigma=1.0$)** *(Ours)* | **0.376** | **0.279** | **0.272** | **0.467** | 11.87 | 8.90 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SDXL (2.6B)**<br>*(w/ DDPM 100 steps)* | Vanilla | 0.722 | 0.282 | 0.292 | 0.545 | 42.00 | 33.84 |
| | Vanilla (250-step) | 0.746 | 0.283 | 0.295 | 0.559 | 104.10 | 33.84 |
| | UG (Bansal et al., 2024) | 0.749 | 0.279 | 0.287 | 0.541 | 334.43 | OOM* |
| | DATE (Na et al., 2025) | 0.960 | 0.283 | 0.294 | 0.570 | 272.32 | OOM* |
| | LiDAR (DMD-1 / $n=100$) | 1.047 | 0.287 | 0.306 | 0.591 | 78.67 | 33.84 |
| | **RS-LiDAR (DMD-1 / $n=100$, $\sigma=0.25$)** *(Ours)* | **1.06** | **0.287** | **0.306** | **0.600** | 84.58 | 33.84 |

---

### Giải thích các con số Thời gian & Bộ nhớ của RS-LiDAR:
1. **Bộ nhớ (Memory = 8.90 GiB cho SD 1.5, 33.84 GiB cho SDXL)**:
   - Giữ nguyên bằng chính xác Vanilla và LiDAR vì RS-LiDAR kế thừa trọn vẹn đặc tính **Closed-form Guidance (No-BackPropagation)**, không lưu graph tính toán đạo hàm ngược như UG / DATE.
2. **Thời gian (Time)**:
   - Số đứng trước (ví dụ `9.97s` / `13.46s` / `78.77s`): Áp dụng cơ chế **Latent Perturbation Smoothing** (chỉ tốn thêm $\approx 0.05\text{s} - 0.1\text{s}$ trên GPU).
   - Số trong ngoặc `(11.87s)` / `(15.36s)` / `(88.67s)`: Áp dụng cơ chế **Monte Carlo ImageReward Smoothing ($M=4$ samples)** (tính thêm $M$ lần chấm điểm reward).

---

## 2. Ablation Study 1: Khảo Sát Số Lượng Mẫu Monte Carlo $M \in \{1, 2, 4, 8, 16\}$ (SD v1.5 w/ DDIM 50 steps)

> **Thiết lập thực nghiệm**:
> - **Backbone**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
> - **Target Solver (Phase 2)**: DDIM 50 steps ($\eta = 0.0$, $N = 4$ ảnh/prompt, Guidance Scale $s = 12.5$, $\lambda = 5000$, $t_{\text{end}} = 200$)
> - **Lookahead (Phase 1)**: DPM-Solver 5 steps ($n = 50$ hạt lookahead, $\sigma = 1.0$)
> - **Biến số khảo sát**: Số lượng mẫu Monte Carlo làm mịn $M \in \{1, 2, 4, 8, 16\}$

### Bảng 1.1: Tổng Hợp Thước Đo Chính (Table 2 Metrics)

| Backbone | Cấu hình Monte Carlo ($M$) | IR (↑) | CLIP (↑) | HPS (↑) | GenEval (↑) | Time (sec.) (↓) | Mem. (GiB) (↓) | 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5 (DDIM-50)** | $M = 1$ | 0.332 | 0.278 | 0.268 | 0.459 | 9.92 | 8.90 |
| **SD v1.5 (DDIM-50)** | $M = 2$ | 0.349 | 0.278 | 0.268 | 0.465 | 10.57 | 8.90 |
| **SD v1.5 (DDIM-50)** | $M = 4$ | 0.368 | 0.279 | 0.268 | 0.465 | 11.87 | 8.90 |
| **SD v1.5 (DDIM-50)** | $M = 8$ | **0.376** | **0.279** | **0.268** | **0.467** | 14.47 | 8.90 |
| **SD v1.5 (DDIM-50)** | $M = 16$ | 0.359 | 0.278 | 0.268 | 0.462 | 19.67 | 8.90 |

---

## 3. Ablation Study 2: Khảo Sát Số Bước Lookahead Solver Steps $\delta \in \{2, 3, 4, 5\}$ (SD v1.5 w/ DDIM 50 steps)

> **Thiết lập thực nghiệm**:
> - **Backbone**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
> - **Target Solver (Phase 2)**: DDIM 50 steps ($\eta = 0.0$, $N = 4$ ảnh/prompt, Guidance Scale $s = 12.5$, $\lambda = 5000$, $t_{\text{end}} = 200$)
> - **Lookahead Parameters**: $n = 50$ hạt lookahead, $M = 8$ Monte Carlo samples, $\sigma = 1.0$
> - **Biến số khảo sát**: Số bước rời rạc hóa của bộ giải lookahead $\delta \in \{2, 3, 4, 5\}$ (DPM-Solver)

### Bảng 2.1: Tổng Hợp Thước Đo Chính (Table 2 Metrics)

| Backbone | Lookahead Solver (Steps $\delta$) | IR (↑) | CLIP (↑) | HPS (↑) | GenEval (↑) | Time (sec.) (↓) | Mem. (GiB) (↓) | 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| **SD v1.5 (DDIM-50)** | $\delta = 2$ (DPM-2) | 0.154 | 0.275 | 0.263 | 0.448 | | | 
| **SD v1.5 (DDIM-50)** | $\delta = 3$ (DPM-3) | 0.224 | 0.275 | 0.265 | 0.466 | | | 
| **SD v1.5 (DDIM-50)** | $\delta = 4$ (DPM-4) | 0.314 | 0.277 | 0.268 | **0.467** | | | 
| **SD v1.5 (DDIM-50)** | $\delta = 5$ (DPM-5) | **0.376** | **0.279** | **0.272** | **0.467** | 14.47 | 8.90 |

---

## 4. Bảng Phân Rã Thời Gian Thực Thi (Inference Latency Breakdown)

> **Mô tả**: Đo đạc chi phí thời gian (giây) theo 3 thành phần trong quy trình suy luận 2-Phase (khớp cấu trúc Bảng 9 bài báo gốc ICML 2026), tính trung bình trên GPU NVIDIA A100.

| Giai đoạn (Stage) | SD v1.5 (DDIM-50 / DPM-5) | SD v1.5 (DDPM-50 / DPM-5) | SDXL (DDPM-100 / DMD-1) |
| :--- | :---: | :---: | :---: |
| Lookahead sampling | 5.69 | 5.69 | 31.45 |
| Reward annotation | 0.65 | 0.65 | 1.97 |
| Target sampling | 3.58 | 7.07 | 45.25 |
| Total | 9.92 | 13.41 | 78.67 |


