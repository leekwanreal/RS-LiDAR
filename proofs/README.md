# Tuyển Tập Chứng Minh Toán Học Giải Tích Chuẩn Mực RS-LiDAR (Proofs Master Guide)

> **Dự án**: RS-LiDAR: Randomized Smoothing Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models  
> **Mục tiêu**: Thiết lập chứng minh từng bước (Step-by-Step Deductive Proofs) bảo đảm tính chính xác 100% theo chuẩn mực xuất bản các hội nghị khoa học hàng đầu (ICML, NeurIPS). Mọi giả định về không gian hàm Sobolev, tính khả vi yếu, và các định lý giải tích xác suất đều được xác lập tường minh.

---

## 1. KHUNG TOÁN HỌC NỀN TẢNG (MATHEMATICAL FOUNDATION)

### 1.1. Không Gian Xác Suất & Nhân Làm Trơn Gauss
Xét không gian xác suất Euclid $(\mathbb{R}^D, \mathcal{B}(\mathbb{R}^D), \gamma_\sigma)$ được trang bị độ đo Gauss đẳng hướng $\gamma_\sigma \triangleq \mathcal{N}(0, \sigma^2 \mathbf{I}_D)$ với tham số làm mịn $\sigma > 0$. Đạo hàm Radon-Nikodym của $\gamma_\sigma$ đối với độ đo Lebesgue $d\mathbf{z}$ là nhân Gauss (Gaussian mollifier):
$$\phi_\sigma(\mathbf{z}) \triangleq \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left( -\frac{\|\mathbf{z}\|_2^2}{2\sigma^2} \right)$$
Do $\phi_\sigma \in \mathcal{S}(\mathbb{R}^D)$ (không gian hàm Schwartz giảm nhanh), phép tích chập hàm suy rộng với $\phi_\sigma$ biến đổi bất kỳ phân phối hoặc hàm đo được nào thành hàm khả vi vô hạn lần $C^\infty(\mathbb{R}^D)$.

### 1.2. Định Lý Rademacher & Đạo Hàm Yếu
Cho hàm phần thưởng $R: \mathbb{R}^D \to \mathbb{R}$.
* Nếu $R \in W^{1,\infty}(\mathbb{R}^D)$ (hàm Lipschitz toàn cục với hằng số $L_{\text{before}} < \infty$): Theo **Định lý Rademacher**, $R$ khả vi theo nghĩa cổ điển tại hầu khắp nơi (Lebesgue-almost everywhere - a.e.) trên $\mathbb{R}^D$.
* Đạo hàm yếu (weak gradient) $\nabla R = (\partial_1 R, \dots, \partial_D R)^\top \in L^\infty(\mathbb{R}^D; \mathbb{R}^D)$ trùng với đạo hàm cổ điển hầu khắp nơi và thỏa mãn:
  $$\|\nabla R\|_{L^\infty(\mathbb{R}^D, \ell_2)} \triangleq \operatorname{ess\,sup}_{\mathbf{y} \in \mathbb{R}^D} \|\nabla R(\mathbf{y})\|_2 = L_{\text{before}} < \infty$$

### 1.3. Định Lý Biểu Diễn Vi Phân Của Tích Chập
Với hàm suy rộng $R$ và nhân trơn $\phi_\sigma$:
$$\nabla (R * \phi_\sigma)(\mathbf{x}) = (R * \nabla \phi_\sigma)(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{y}) \nabla_\mathbf{x} \phi_\sigma(\mathbf{x} - \mathbf{y}) \, d\mathbf{y}$$
Nếu $R \in W^{1,\infty}$, phép lấy đạo hàm cũng có thể chuyển sang đạo hàm yếu của $R$:
$$\nabla (R * \phi_\sigma)(\mathbf{x}) = ((\nabla R) * \phi_\sigma)(\mathbf{x}) = \int_{\mathbb{R}^D} \nabla R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$

---

## 2. DANH MỤC CÁC TỆP CHỨNG MINH CHI TIẾT

| Tệp Chứng Minh | Định Lý Tương Ứng | Công Cụ Giải Tích Sử Dụng |
| :--- | :--- | :--- |
| [`proof_theorem_1.md`](proof_theorem_1.md) | **Theorem 1: Universal Gradient Boundedness** | Stein's Identity, Đối xứng cầu, Mô-men tuyệt đối phân phối chuẩn |
| [`proof_theorem_2.md`](proof_theorem_2.md) | **Theorem 2: Lipschitz Contraction & Regimes** | Sobolev Mollification, Định lý Rademacher, Bất đẳng thức Minkowski |
| [`proof_theorem_3.md`](proof_theorem_3.md) | **Theorem 3: Variance-Aware Reduction & (σ, M)** | Phân rã phương sai vector, Khai triển Hessian Peano, Bổ đề Remark 1 |
| [`proof_theorem_4.md`](proof_theorem_4.md) | **Theorem 4: Dimensionality Optimal Scaling** | Taylor-Lagrange phần dư tích phân, Phân phối $\chi^2(D)$, Cực trị lồi Minimax |
