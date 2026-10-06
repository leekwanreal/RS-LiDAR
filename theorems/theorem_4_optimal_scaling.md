# Định Lý 4: Quy Luật Phụ Thuộc Số Chiều Của Bán Kính Tối Ưu (Dimensionality Optimal Scaling Law)

> **Mã số**: Theorem 4 (Tương ứng với Theorem 3 trong tài liệu giải tích chứng minh)  
> **Chứng minh chi tiết**: Xem tại [`../proofs/proof_theorem_4.md`](../proofs/proof_theorem_4.md)

---

## 1. KHÔNG GIAN HÀM & ĐIỀU KIỆN TIÊN QUYẾT

* Xét không gian ảnh $D$ chiều: $D = 3 \times H \times W$ (ví dụ: $D = 786,432$ trên SD 1.5 và $D = 3,145,728$ trên SDXL).
* Hàm phần thưởng $R: \mathbb{R}^D \to \mathbb{R}$ khả vi liên tục hai lần ($C^2$), có biên độ $\Delta R < \infty$.
* Ma trận Hessian bị chặn đều theo chuẩn phổ (Spectral Norm):
  $$\|\nabla^2 R(\mathbf{x})\|_2 \triangleq \sup_{\|\mathbf{v}\|_2=1} |\mathbf{v}^\top \nabla^2 R(\mathbf{x}) \mathbf{v}| \le H < \infty, \quad \forall \mathbf{x} \in \mathbb{R}^D$$

---

## 2. PHÁT BIỂU ĐỊNH LÝ (FORMAL STATEMENT)

1. **Chặn Trên Sai Số Làm Méo Mó Ảnh (Bias Bound)**:
   Sai số xấp xỉ cục bộ giữa hàm làm trơn $R_\sigma$ và hàm gốc $R$ bị chặn trên bởi:
   $$\operatorname{Bias}(\sigma, \mathbf{x}) \triangleq |R_\sigma(\mathbf{x}) - R(\mathbf{x})| \le \frac{D \cdot H}{2} \sigma^2$$
2. **Cân Bằng Đánh Đổi Minimax & Nghiệm Giải Tích**:
   Hàm mục tiêu chi phí tổng hòa cân bằng giữa độ trơn Lipschitz ($\frac{\Delta R}{\sigma \sqrt{2\pi}}$) và sai số méo mó ngữ nghĩa ($\frac{D \cdot H}{2} \sigma^2$):
   $$\mathcal{J}(\sigma) \triangleq \frac{\Delta R}{\sigma \sqrt{2\pi}} + \frac{D \cdot H}{2} \sigma^2$$
   đạt cực tiểu toàn cục duy nhất tại nghiệm giải tích:
   $$\boxed{\sigma_{\text{opt}} = \left( \frac{\Delta R}{\sqrt{2\pi} \cdot H \cdot D} \right)^{1/3} \propto \mathcal{O}\left( D^{-1/3} \right)}$$

---

## 3. Ý NGHĨA VẬT LÝ & ĐỊNH LÝ TẬP TRUNG ĐỘ ĐO (CONCENTRATION OF MEASURE)

Trong không gian ảnh số $D$ chiều, vector nhiễu ngẫu nhiên Gauss $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_D)$ tập trung ngặt nghèo trên vỏ cầu mỏng với bán kính chuẩn Euclidean:
$$\mathbb{E}[\|\boldsymbol{\epsilon}\|_2] \approx \sigma \sqrt{D}$$

### Đối Chiếu Hai Nền Tảng Mô Hình:
1. **Stable Diffusion 1.5 ($512 \times 512$)**:
   * Số chiều $D_{\text{SD1.5}} = 3 \times 512^2 = \mathbf{786,432} \implies \sqrt{D} \approx \mathbf{886.8}$.
   * Tại $\sigma = 1.0$: Chuẩn độ dịch chuyển vật lý là $\|\boldsymbol{\epsilon}\|_2 = 1.0 \times 886.8 = \mathbf{886.8}$.
2. **Stable Diffusion XL ($1024 \times 1024$)**:
   * Số chiều $D_{\text{SDXL}} = 3 \times 1024^2 = \mathbf{3,145,728}$ (**Gấp đúng 4 lần SD 1.5!**) $\implies \sqrt{D} \approx \mathbf{1,773.6}$ (**Gấp đôi SD 1.5!**).
   * Tại $\sigma = 0.5$: Chuẩn độ dịch chuyển là $\|\boldsymbol{\epsilon}\|_2 = 0.5 \times 1,773.6 = \mathbf{886.8}$.
   $$\implies \|\boldsymbol{\epsilon}\|_{2, \text{SDXL}}^{(\sigma = 0.5)} \equiv \|\boldsymbol{\epsilon}\|_{2, \text{SD1.5}}^{(\sigma = 1.0)}$$

### Tỷ Lệ Co Bán Kính Theo Định Lý 4:
$$\frac{\sigma_{\text{opt}}(\text{SDXL})}{\sigma_{\text{opt}}(\text{SD1.5})} = \left( \frac{D_{\text{SDXL}}}{D_{\text{SD1.5}}} \right)^{-1/3} = (4)^{-1/3} \approx \mathbf{0.63}$$
Nhân tỷ lệ $0.63 \times [0.5, 1.0] = \mathbf{[0.31, 0.63]}$, hoàn toàn giải thích vì sao Sweet Spot trên SDXL co lại về dải **$\sigma = 0.25 - 0.5$**!
