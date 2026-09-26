# 📑 BẢN ĐỀ XUẤT THỰC NGHIỆM KHOA HỌC (EXPERIMENT PROPOSALS)
## Kiểm Chứng Trực Tiếp Hai Điểm Yếu Cốt Lõi (Test 2 & Test 3) Trên Quá Trình Sinh Ảnh Thật Của RS-LiDAR

> **Tác giả / Nhóm nghiên cứu**: RS-LiDAR Research Team  
> **Mục tiêu**: Chuyển hóa các phát hiện chẩn đoán vi mô từ **Bài Test 2 (Best-of-1 Trap / Entropy Collapse)** và **Bài Test 3 (Lipschitz Gradient Instability)** thành hai thực nghiệm vĩ mô then chốt trên ảnh thực tế và các thước đo chuẩn mực (ImageReward, GenEval, HPS v2.1).  
> **Backbone thực nghiệm**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`), bộ giải DDIM 50 bước.

---

```
                       [BẰNG CHỨNG NỘI SOI VI MÔ]
        ┌────────────────────────────────────────────────────────┐
        │ Test 2: LiDAR sụp đổ Softmax về 1 hạt (H ≈ 0, Neff ≈ 1) │
        │ Test 3: Gradient LiDAR rung giật, Lipschitz bùng nổ   │
        └───────────────────────────┬────────────────────────────┘
                                    │
                  Chuyển hóa sang thực nghiệm vĩ mô
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│       EXPERIMENT PROPOSAL 1     │   │       EXPERIMENT PROPOSAL 2     │
│   Guidance Scale Stress-Test    │   │     Particle Scaling (N=3->100) │
│   (s ∈ {12.5, 15, 17.5, 20})    │   │  (Pareto Frontier & Multi-Part) │
│                                 │   │                                 │
│  Hệ quả trực tiếp của Test 3:   │   │  Hệ quả trực tiếp của Test 2:   │
│  LiDAR gãy ở s=17.5 do rung giật│   │  LiDAR bão hòa ở N=50 do bẫy 1  │
│  RS-LiDAR mở rộng Stability     │   │  hạt; RS-LiDAR bứt phá ở N=100  │
│  Margin nhờ chặn Lipschitz.     │   │  nhờ sức mạnh Ensemble đa hạt.  │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 🔬 EXPERIMENT PROPOSAL 1: GUIDANCE SCALE STRESS-TEST
### Khảo Sát Ngưỡng Chịu Lực (Stability Margin) & Kháng Vỡ Đa Tạp Của RS-LiDAR

### 1.1. Động Cơ Khoa Học & Cầu Nối Nhân Quả Từ Test 3
* **Phát hiện từ Test 3**:
  Vector dẫn đường đóng của LiDAR gốc:
  $$\mathbf{g}_t(x_t) = \nabla_{x_t} \log \sum_{i=1}^N \exp(\lambda R_i + \text{pot}_i(x_t))$$
  chịu hằng số Lipschitz vô hạn ($L_0 \to \infty$) do bề mặt mô hình phần thưởng (ImageReward) chứa nhiều gợn sóng tần số cao. Khi trạng thái latent $x_t$ bị xê dịch vi mô $\delta = 10^{-3}$, hướng của vector dẫn đường bị rung giật dữ dội ($\text{CosSim}$ suy giảm).
* **Hệ quả trên quá trình khử nhiễu Phase 2**:
  Tại mỗi bước $t$, vector nhiễu dự đoán bởi UNet được lái theo công thức:
  $$\hat{\boldsymbol{\epsilon}}_t = \boldsymbol{\epsilon}_{\theta}(x_t, t, c) - \sqrt{1 - \bar{\alpha}_t} \cdot s \cdot \mathbf{g}_t(x_t)$$
  với $s$ là hệ số khuếch đại (Guidance Scale).
  * **Với LiDAR gốc**: Khi $s$ tăng cao ($s \ge 17.5$), **sự rung giật vi mô của $\mathbf{g}_t$ bị nhân lên gấp bội**. Lực kéo bạo lực và bất ổn định này giật các hạt latent văng ra khỏi đa tạp dữ liệu thực (manifold drift), làm vỡ nát cấu trúc ảnh, bão hòa màu sắc (oversaturation), tạo ra các đốm kỳ dị (artifacts). Thực nghiệm trước đây đã ghi nhận: tại $s = 17.5$, ImageReward của LiDAR sụt từ $0.3466 \to 0.3020$, GenEval rớt từ $0.4331 \to 0.4185$ (dưới cả baseline SD 1.5 gốc $0.4230$).
  * **Với RS-LiDAR**: Nhờ Định lý 3.1 & 3.3 (**Dimension-Free Lipschitz Bound**):
    $$L_\sigma \le \frac{\lambda}{\sigma \sqrt{2\pi}} < \infty$$
    trường vector gradient $\mathbf{g}_t^\sigma$ được làm mịn hoàn hảo, có độ dốc hữu hạn toàn cục. Do đó, RS-LiDAR sở hữu **Ngưỡng Chịu Lực (Stability Margin)** rộng hơn hẳn, cho phép mô hình chịu được scale lớn ($s = 17.5 \sim 20.0$) mà không bị vỡ ảnh, tối đa hóa tiềm năng căn chỉnh phần thưởng.

### 1.2. Giả Thuyết Nghiên Cứu (Research Hypotheses)
1. **Hypothesis 1.1 (Điểm gãy của LiDAR)**: Đường cong hiệu năng của LiDAR gốc đạt đỉnh tại $s \approx 12.5 - 15.0$ và rơi tự do khi $s \ge 17.5$.
2. **Hypothesis 1.2 (Sức bền của RS-LiDAR)**: RS-LiDAR duy trì chất lượng ảnh nguyên vẹn ở $s = 17.5$ và $s = 20.0$, đạt đỉnh ImageReward mới cao hơn đáng kể so với mức đỉnh của LiDAR gốc.
3. **Hypothesis 1.3 (Dung sai GenEval)**: Điểm GenEval của RS-LiDAR tại $s = 17.5$ không bị sụp đổ như LiDAR mà vẫn giữ vững phong độ $\ge 0.46$.

### 1.3. Thiết Lập Thực Nghiệm
* **Backbone**: Stable Diffusion v1.5.
* **Tập Prompt**: 50 prompts GenEval ngẫu nhiên (hoặc toàn bộ 553 prompts nếu đủ quota GPU).
* **Phase 1 (Lookahead)**: Tái sử dụng trực tiếp các hạt lookahead $n=50$, DPM-5 seed 100 đã có sẵn (tiết kiệm 100% chi phí Phase 1).
* **Phase 2 (Target Sampling Sweep)**:
  * Thuật toán: **Vanilla LiDAR** ($\sigma = 0$) vs. **RS-LiDAR** ($\sigma = 1.0, M = 4$).
  * Biến số khảo sát: Guidance Scale $s \in \{7.5, 12.5, 15.0, 17.5, 20.0\}$.
* **Chỉ số đo lường**:
  1. *Độ căn chỉnh*: ImageReward, HPS v2.1, CLIP-Score.
  2. *Độ bám sát cấu trúc đối tượng*: GenEval Overall Score.
  3. *Chất lượng hình ảnh*: Aesthetic Score (đo mức độ cháy màu / artifacts).

### 1.4. Kỳ Vọng Kết Quả (Bảng Mẫu Đưa Vào Bài Báo)

| Guidance Scale ($s$) | ImageReward (LiDAR) | ImageReward (RS-LiDAR) | GenEval (LiDAR) | GenEval (RS-LiDAR) | Đánh Giá Độ Ổn Định Manifold |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$s = 7.5$** | ~0.150 | ~0.180 | 0.435 | 0.442 | Cả hai lái nhẹ, an toàn |
| **$s = 12.5$ (Mặc định)** | 0.346 | **0.368** | 0.455 | **0.465** | Vùng tối ưu của LiDAR |
| **$s = 15.0$** | 0.340 | **0.375** | 0.458 | **0.472** | LiDAR bắt đầu bão hòa |
| **$s = 17.5$ (Vùng quá tải)** | 0.302 *(Gãy -0.044)* | **0.382** *(Tăng)* | 0.418 *(Sụp đổ)* | **0.470** *(Vững)* | **LiDAR vỡ ảnh, RS-LiDAR đạt đỉnh** |
| **$s = 20.0$ (Cực đại)** | < 0.250 *(Hỏng ảnh)* | **0.378** *(Chịu lực tốt)* | < 0.400 | **0.463** | **Minh chứng thép cho Lipschitz Bound** |

---

## 🔬 EXPERIMENT PROPOSAL 2: LOOKAHEAD PARTICLE SCALING ($N \in \{3, 9, 20, 50, 100\}$)
### Khảo Sát Sức Mạnh Hợp Lực Đa Hạt (Consensus Guidance) vs. Bão Hòa Đơn Hạt (Best-of-1 Saturation)

### 2.1. Động Cơ Khoa Học & Cầu Nối Nhân Quả Từ Test 2
* **Phát hiện từ Test 2**:
  Trong Bài Test 2, ta đã chứng minh với $\lambda = 5000$, phân phối trọng số Softmax của LiDAR gốc bị dồn $99.9\%$ vào đúng hạt có điểm cao nhất ($w_{\max} \approx 1.0, H \to 0$ bits).
  * Điều này dẫn tới hệ quả tất yếu: **LiDAR gốc bị bão hòa năng lực tính toán khi tăng số hạt $N$**. Khi tăng từ $N=50$ lên $N=100$, về bản chất LiDAR vẫn chỉ nhặt ra đúng 1 hạt duy nhất. Thậm chí khi $N$ càng lớn, xác suất bốc phải các "hạt nhiễu ngoại lai" (outlier particles với ImageReward ảo do reward model bị overfit) càng cao, dẫn đến **Reward Hacking**.
  * Ngược lại, **RS-LiDAR** duy trì phân bổ xác suất mượt mà trên một cụm các hạt có triển vọng trong lân cận. Khi cấp thêm hạt ($N=100$), RS-LiDAR thực sự **tổng hợp được thông tin đa chiều từ nhiều hạt** (Consensus Guidance: hạt này đóng góp bố cục, hạt kia đóng góp màu sắc và ánh sáng).

### 2.2. Giả Thuyết Nghiên Cứu (Research Hypotheses)
1. **Hypothesis 2.1 (Hiện tượng bão hòa của LiDAR)**: Đường cong ImageReward và GenEval của LiDAR theo số hạt $N$ sẽ chững lại (plateau) hoặc suy giảm nhẹ khi tiến từ $N=50 \to N=100$.
2. **Hypothesis 2.2 (Sức bật quy mô của RS-LiDAR)**: RS-LiDAR tiếp tục leo dốc khi tăng từ $N=50 \to N=100$, tạo ra khoảng cách vượt trội lớn nhất so với LiDAR ở mốc $N=100$.
3. **Hypothesis 2.3 (Pareto Frontier Vượt Trội)**: Trên đồ thị đánh đổi **Thời gian tính toán (Wall-clock time) vs. Điểm GenEval/ImageReward**, đường cong của RS-LiDAR hoàn toàn bao bọc (strictly dominates) đường cong của LiDAR gốc ở mọi ngân sách tính toán.

### 2.3. Thiết Lập Thực Nghiệm
* **Backbone**: Stable Diffusion v1.5, DDIM 50 bước.
* **Tập Prompt**: 20 prompts GenEval (hoặc 50 prompts).
* **Phase 1 (Lookahead Sweep)**:
  * Sinh các bộ lookahead với số lượng hạt: $N \in \{3, 9, 20, 50, 100\}$.
  * Bộ giải: DPM-Solver 5 bước ($S=5$).
* **Phase 2 (Target Sampling)**:
  * Chạy song song:
    * LiDAR gốc ($\sigma = 0, \lambda = 5000$).
    * RS-LiDAR ($\sigma = 1.0, M = 4, \lambda = 5000$).
* **Chỉ số đo lường**:
  1. *Hiệu năng*: ImageReward, GenEval, HPS v2.1.
  2. *Chi phí*: Thời gian Phase 1 + Phase 2 (giây), Bộ nhớ VRAM (GiB).
  3. *Hiệu quả hạt*: Số lượng hạt hiệu dụng trung bình $N_{eff} = 2^H$ từ Test 2.

### 2.4. Kỳ Vọng Kết Quả (Biểu Đồ & Bảng Số Liệu)

```
ImageReward
   ▲                                              ● RS-LiDAR (N=100) [Đỉnh cao]
   │                                  ● RS-LiDAR (N=50)
   │                      ● RS-LiDAR (N=20)       ▲
   │          ● RS-LiDAR (N=9)                    │ Gap lớn nhất
   │  ● RS-LiDAR (N=3)                ■ LiDAR (N=50) ─── ■ LiDAR (N=100) [Bão hòa]
   │                      ■ LiDAR (N=20)
   │          ■ LiDAR (N=9)
   │  ■ LiDAR (N=3)
   └────────────────────────────────────────────────────────► Số hạt lookahead (N)
     3        9           20          50                  100
```

| Số Hạt Lookahead ($N$) | Chi Phí Phase 1 (s) | ImageReward (LiDAR) | ImageReward (RS-LiDAR) | GenEval (LiDAR) | GenEval (RS-LiDAR) | Nhận Xét Khoa Học |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$N = 3$** | ~0.8s | 0.172 | **0.198** | 0.449 | **0.455** | Ngân sách siêu rẻ, RS lọc nhiễu tốt |
| **$N = 9$** | ~1.5s | 0.211 | **0.245** | 0.453 | **0.462** | RS bắt đầu bứt tốc |
| **$N = 20$** | ~2.6s | 0.285 | **0.315** | 0.460 | **0.470** | Cân bằng hiệu năng / tốc độ |
| **$N = 50$ (Chuẩn)** | ~5.7s | 0.347 | **0.368** | 0.456 | **0.465** | Chuẩn Table 2 bài báo |
| **$N = 100$ (Scaling)** | ~11.0s | 0.350 *(Bão hòa)* | **0.388** *(Bứt phá)* | 0.457 *(Ngang)* | **0.478** *(Vượt trội)* | **Chứng minh sức mạnh Multi-particle** |

---

## 🛠️ KẾ HOẠCH TRIỂN KHAI NHANH TRÊN KAGGLE (ACTION PLAN)

| Thứ Tự | Thực Nghiệm | Tài Nguyên Cần | Thời Gian Dự Kiến | Mức Độ Ưu Tiên |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Proposal 1: Guidance Scale Sweep** ($s \in \{12.5, 15.0, 17.5, 20.0\}$) | Rất nhẹ (Tái sử dụng Phase 1 $N=50$ đã có sẵn) | ~15–20 phút trên 2x GPU T4 (20 prompts) | ⭐⭐⭐⭐⭐ (Khuyên chạy trước) |
| **2** | **Proposal 2: Particle Scaling** ($N \in \{9, 20, 50, 100\}$) | Cần sinh Phase 1 cho $N=9, 20, 100$ | ~40–50 phút trên 2x GPU T4 (20 prompts) | ⭐⭐⭐⭐ |

---

### Kết Luận:
Hai đề xuất trên là **vũ khí phản công học thuật hoàn hảo**: 
* Đề xuất 1 chứng minh RS-LiDAR không bị vỡ ảnh ở scale cao (xác thực tính ổn định của Test 3).
* Đề xuất 2 chứng minh RS-LiDAR thực sự tận dụng được $N=100$ hạt chứ không bị bẫy 1 hạt như LiDAR (xác thực tính đa hạt của Test 2).
