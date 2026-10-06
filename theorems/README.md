# Tổng Quan Hệ Thống Định Lý Toán Học RS-LiDAR (Theorems Master Guide)

> **Dự án**: RS-LiDAR: Randomized Smoothing Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models  
> **Cấu trúc tài liệu**: Thư mục `theorems/` tổng hợp các phát biểu định lý chính quy, ngắn gọn, súc tích và các trực giác vật lý/hình học. Toàn bộ chứng minh giải tích chi tiết từng bước được lưu trữ tương ứng tại thư mục [`../proofs/`](../proofs/README.md).

---

## 1. BẢN ĐỒ LOGIC HỆ THỐNG 4 ĐỊNH LÝ

Hệ thống lý thuyết của RS-LiDAR được xây dựng theo cấu trúc phân tầng 4 cấp độ, giải quyết từ tính khả vi của các reward gián đoạn cho đến sự suy giảm độ dốc cục bộ và quy luật tỷ lệ tối ưu theo số chiều:

```
[THIẾT LẬP NỀN TẢNG]
  Mọi hàm Reward bị chặn (R ∈ L^∞, ΔR < ∞)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ĐỊNH LÝ 1: UNIVERSAL GRADIENT BOUNDEDNESS                             │
│  Tạo độ trơn C^∞ từ con số 0; Gradient bị chặn cứng ||∇R_σ|| ≤ ΔR/(σ√(2π))│
│  (Áp dụng cho cả reward gián đoạn, logic nhị phân, 0/1 như GenEval)    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Khi R là mạng nơ-ron Lipschitz (R ∈ W^{1,∞})
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ĐỊNH LÝ 2: LIPSCHITZ CONTRACTION & CRITICAL THRESHOLD σ*             │
│  Tồn tại ngưỡng chuyển pha vĩ mô: σ* = ΔR / (L_before √(2π))           │
│  - σ ≤ σ*: Vùng bảo toàn (L_after ≤ L_before, tính không giãn nở)      │
│  - σ > σ*: Vùng co rút chủ động (L_after giảm tỷ lệ nghịch 1/σ)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Giải mã nghịch lý σ* quá lớn (~103)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ĐỊNH LÝ 3: VARIANCE-AWARE REDUCTION & BISTABLE POTENTIAL (σ, M)       │
│  Trong vùng nhiễu vi mô (σ < σ*), triệt tiêu phương sai hướng gradient:│
│  ||∇R_σ|| ≤ √(L_before² - V(σ, x)) ≈ L₀ - c·σ²                         │
│  Giải mã đáy chữ U tại σ* ≈ 0.25 và hiệu ứng số mẫu Monte Carlo M      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Mở rộng không gian ảnh đa chiều D
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ĐỊNH LÝ 4: DIMENSIONALITY OPTIMAL SCALING LAW (σ_opt ∝ D^{-1/3})      │
│  Cân bằng Minimax giữa Độ trơn (1/σ) và Sai số méo mó ảnh Bias (D·σ²):  │
│  σ_opt = (ΔR / (√(2π) H D))^{1/3}; Chuẩn năng lượng ||σε||₂ ≈ σ√D      │
│  Giải mã vì sao SD 1.5 chọn σ=1.0 còn SDXL chọn σ=0.25 - 0.5           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DANH MỤC CÁC ĐỊNH LÝ CHI TIẾT

| Định Lý | Tiêu Đề Khoa Học | Đối Tượng Không Gian Hàm | Kết Quả Trọng Tâm | File Chi Tiết | File Chứng Minh |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Theorem 1** | **Universal Gradient Boundedness** | Mọi $R \in L^\infty(\mathbb{R}^D)$ (Bị chặn, kể cả gián đoạn) | $\|\nabla R_\sigma(\mathbf{x})\|_2 \le \dfrac{\Delta R}{\sigma \sqrt{2\pi}}$ | [`theorem_1_gradient_bound.md`](theorem_1_gradient_bound.md) | [`../proofs/proof_theorem_1.md`](../proofs/proof_theorem_1.md) |
| **Theorem 2** | **Lipschitz Contraction & Regimes** | $R \in W^{1,\infty}(\mathbb{R}^D)$ (Hàm $L$-Lipschitz toàn cục) | $L_{\text{after}} \le \min\left( L_{\text{before}}, \frac{\Delta R}{\sigma \sqrt{2\pi}} \right)$ | [`theorem_2_phase_transition.md`](theorem_2_phase_transition.md) | [`../proofs/proof_theorem_2.md`](../proofs/proof_theorem_2.md) |
| **Theorem 3** | **Variance-Aware Reduction & Bistable Formula** | $R \in W^{1,\infty} \cap C^2$, $M$ mẫu Monte Carlo | $\|\nabla R_\sigma\|_2 \le \sqrt{L_{\text{before}}^2 - \mathcal{V}(\sigma, \mathbf{x})}$<br>$\mathbb{E}[\widehat{L}_M^2] \approx L_0^2 - (1-\frac{1}{M})\sigma^2 A + \frac{KD}{M}\sigma^4$ | [`theorem_3_variance_reduction.md`](theorem_3_variance_reduction.md) | [`../proofs/proof_theorem_3.md`](../proofs/proof_theorem_3.md) |
| **Theorem 4** | **Dimensionality Optimal Scaling** | $R \in C^2(\mathbb{R}^D)$, $\|\nabla^2 R\|_2 \le H < \infty$ | $\sigma_{\text{opt}} \propto \mathcal{O}(D^{-1/3})$<br>$\|\sigma \mathbf{u}\|_2 \approx \sigma \sqrt{D} \in [443, 886]$ | [`theorem_4_optimal_scaling.md`](theorem_4_optimal_scaling.md) | [`../proofs/proof_theorem_4.md`](../proofs/proof_theorem_4.md) |

---

## 3. QUY ƯỚC KÝ HIỆU TOÁN HỌC CHUẨN

* $\mathbf{x} \in \mathbb{R}^D$: Vector ảnh trong miền giá trị pixel chuẩn hóa $[-1.0, 1.0]^D$.
* $\sigma > 0$: Bán kính làm mịn Gaussian trong không gian ảnh (Smoothing Radius).
* $\phi_\sigma(\mathbf{z}) = \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left(-\frac{\|\mathbf{z}\|_2^2}{2\sigma^2}\right)$: Nhân làm trơn Gauss (Gaussian mollifier).
* $R_\sigma(\mathbf{x}) = \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u})]$: Hàm phần thưởng làm trơn kỳ vọng.
* $\Delta R \triangleq \operatorname{ess\,sup} R - \operatorname{ess\,inf} R < \infty$: Biên độ dao động hữu hạn của hàm phần thưởng.
* $M$: Số lượng mẫu ngẫu nhiên Monte Carlo dùng để xấp xỉ kỳ vọng trong thực nghiệm.
