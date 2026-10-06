# Định Lý 3: Giảm Lipschitz Do Phân Rã Phương Sai Gradient Trong Vùng Bảo Toàn (Variance-Aware Lipschitz Reduction)

> **Mã số**: Theorem 3 (Tương ứng với Theorem 2 trong tài liệu giải tích chứng minh)  
> **Chứng minh chi tiết**: Xem tại [`../proofs/proof_theorem_3.md`](../proofs/proof_theorem_3.md)

---

## 1. KHÔNG GIAN HÀM & ĐIỀU KIỆN TIÊN QUYẾT

* Hàm phần thưởng $R: \mathbb{R}^D \to \mathbb{R}$ là hàm $L_{\text{before}}$-Lipschitz trên $\mathbb{R}^D$ ($R \in W^{1,\infty}(\mathbb{R}^D)$).
* Tại lân cận điểm ảnh $\mathbf{x}$, $R$ khả vi hai lần liên tục ($C^2$) với ma trận Hessian $\nabla^2 R(\mathbf{x}) \in \mathbb{R}^{D \times D}$.
* Xét vùng làm mịn vi mô ($\sigma < \sigma^*$, nơi lý thuyết cận biên độ toàn cục của Định lý 2 chưa kích hoạt).

---

## 2. PHÁT BIỂU ĐỊNH LÝ (FORMAL STATEMENT)

Với mọi tham số làm mịn $\sigma > 0$ và tại mọi điểm $\mathbf{x} \in \mathbb{R}^D$, chuẩn Euclidean của gradient hàm làm trơn $R_\sigma$ thỏa mãn bất đẳng thức phân rã phương sai chính xác:

$$\boxed{\|\nabla R_\sigma(\mathbf{x})\|_2 \le \sqrt{L_{\text{before}}^2 - \mathcal{V}(\sigma, \mathbf{x})}}$$

trong đó $\mathcal{V}(\sigma, \mathbf{x})$ là độ đo phương sai định hướng của vector gradient trong lân cận Gauss:
$$\mathcal{V}(\sigma, \mathbf{x}) \triangleq \operatorname{tr}\big(\operatorname{Cov}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)}(\nabla R(\mathbf{x} + \sigma \mathbf{u}))\big) \ge 0$$

Khi khai triển cục bộ quanh $\sigma \to 0$:
$$\boxed{\mathcal{V}(\sigma, \mathbf{x}) = \sigma^2 \|\nabla^2 R(\mathbf{x})\|_F^2 + \mathcal{O}(\sigma^3)}$$
kéo theo quy luật suy giảm bậc hai:
$$\boxed{L(\sigma) \approx L_{\text{before}} - \frac{\|\nabla^2 R(\mathbf{x})\|_F^2}{2 L_{\text{before}}} \sigma^2 + \mathcal{O}(\sigma^3)}$$
*(với $\|\nabla^2 R(\mathbf{x})\|_F^2 = \sum_{i,j=1}^D (\frac{\partial^2 R}{\partial x_i \partial x_j})^2 > 0$ là chuẩn Frobenius đo tổng độ cong gồ ghề của mạng)*.

---

## 3. CÔNG THỨC TỔNG QUÁT LƯỠNG ỔN ĐỊNH $(\sigma, M)$ & BỔ ĐỀ MONTE CARLO

Trên thực tế máy tính, kỳ vọng được xấp xỉ bởi $M$ mẫu ngẫu nhiên: $\widehat{\nabla} R_{\sigma, M}(\mathbf{x}) \triangleq \frac{1}{M} \sum_{m=1}^M \nabla R(\mathbf{x} + \sigma \mathbf{u}_m)$.

### Bổ Đề Remark 1 (Độ Chệch Dương Monte Carlo)
$$\mathbb{E}\left[ \|\widehat{\nabla} R_{\sigma, M}(\mathbf{x})\|_2^2 \right] = \|\nabla R_\sigma(\mathbf{x})\|_2^2 + \frac{\mathcal{V}(\sigma, \mathbf{x})}{M}$$

### Phương Trình Tổng Quát Lưỡng Ổn Định (Bistable Potential Formula)
Kết hợp Định lý 3 và Bổ đề Remark 1 cùng thành phần sai số kẹp biên ảnh $[-1, 1]$ ở vùng nhiễu lớn:

$$\boxed{\mathbb{E}\left[ \widehat{L}_M^2(\sigma, \mathbf{x}) \right] \approx L_{\text{before}}^2 - \underbrace{\left(1 - \frac{1}{M}\right) \sigma^2 \|\nabla^2 R(\mathbf{x})\|_F^2}_{\textbf{Lực lượng 1: Triệt tiêu vi nhiễu } (\propto -\sigma^2)} + \underbrace{\frac{K \cdot D}{M} \sigma^4}_{\textbf{Lực lượng 2: Sai số Monte Carlo hữu hạn } (\propto +\frac{\sigma^4}{M})}}$$

---

## 4. GIẢI MÃ BẢN CHẤT: VÌ SAO KHI $M=4$ THÌ TẠI $\sigma = 1.0$ ĐỘ DỐC $L$ LẠI NHÍCH TĂNG?

Hiện tượng đường cong chữ U (chạm đáy tại $\sigma = 0.25$ và nhích nhẹ tại $\sigma = 1.0$) trên 5,530 cặp mẫu GenEval được giải mã tường minh qua sự cạnh tranh giữa 2 lực lượng:

1. **Ở vùng $\sigma$ nhỏ ($\sigma = 0.1 - 0.25$)**:
   * Số hạng bậc bốn $\sigma^4 = 0.1^4 = 10^{-4}$ quá nhỏ so với $\sigma^2 = 10^{-2}$. Lực lượng 2 hoàn toàn bị đè bẹp.
   * Lực lượng 1 chiếm ưu thế tuyệt đối: Hệ số $\left(1 - \frac{1}{M}\right) = 1 - \frac{1}{4} = 75\%$ sức mạnh triệt tiêu độ dốc được giải phóng, kéo $L$ giảm dốc từ $2.35\times$ đến $4.09\times$.
2. **Ở vùng $\sigma$ lớn ($\sigma = 1.0$)**:
   * Chuẩn dịch chuyển lớn ($\|\sigma \mathbf{u}\|_2 \approx 886.8$) khiến hơn **35% pixel bị kẹp bão hòa vào biên $[-1.0, 1.0]$**, ảnh rơi vào vùng Out-of-Distribution (OOD), làm phương sai giữa các mẫu ngẫu nhiên $\operatorname{Var}_{\mathbf{u}}(R)$ bùng nổ dữ dội.
   * Với $M = 4$ hữu hạn, khoản phạt phương sai $\frac{1}{M} = \frac{1}{4} = 25\%$ cộng dồn trực tiếp vào giá trị đo đạc.
   * Số hạng $\sigma^4 = 1.0^4 = 1.0$ bùng nổ gấp $10,000$ lần, khiến Lực lượng 2 vượt qua Lực lượng 1, kéo giá trị đo đạc $L_{\max}$ nhích tăng tạo thành hình chữ U.
3. **Điểm cực tiểu toàn cục thực nghiệm $\sigma^*$**:
   $$\frac{\partial}{\partial \sigma} \mathbb{E}[\widehat{L}_M^2] = 0 \implies \boxed{\sigma^* = \sqrt{\frac{(M - 1) \|\nabla^2 R(\mathbf{x})\|_F^2}{2 K \cdot D}} \approx \mathbf{0.25}}$$
   Khẳng định con số $\sigma = 0.25$ đo được trên cả 4 mô hình reward là **hệ quả giải tích tất yếu của sự tương tác giữa $\sigma$ và $M=4$**.
