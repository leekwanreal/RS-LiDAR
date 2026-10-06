# Chứng Minh Toán Học: Định Lý 4 (Dimensionality Optimal Scaling Law)

> **Định lý tương ứng**: [`../theorems/theorem_4_optimal_scaling.md`](../theorems/theorem_4_optimal_scaling.md)  
> **Mục tiêu**: Chứng minh cận trên sai số méo mó ảnh $\operatorname{Bias}(\sigma, \mathbf{x}) \le \frac{D \cdot H}{2} \sigma^2$ bằng công thức Taylor với phần dư tích phân, và giải bài toán tối ưu hóa Minimax để tìm nghiệm giải tích $\sigma_{\text{opt}} \propto \mathcal{O}(D^{-1/3})$.

---

### PHẦN 1: CHỨNG MINH CẬN TRÊN SAI SỐ BIAS QUA PHẦN DƯ TÍCH PHÂN

Cố định $\mathbf{x} \in \mathbb{R}^D$ và $\sigma > 0$. Với mỗi vector nhiễu $\mathbf{u} \in \mathbb{R}^D$, xét hàm một biến số:
$$h(t) \triangleq R(\mathbf{x} + t \sigma \mathbf{u}), \quad t \in [0, 1]$$

Do $R \in C^2(\mathbb{R}^D)$, $h(t)$ khả vi liên tục hai lần trên $[0, 1]$. Đạo hàm cấp một và cấp hai của $h(t)$:
$$h'(t) = \sigma \nabla R(\mathbf{x} + t \sigma \mathbf{u})^\top \mathbf{u}$$
$$h''(t) = \sigma^2 \mathbf{u}^\top \nabla^2 R(\mathbf{x} + t \sigma \mathbf{u}) \mathbf{u}$$

Áp dụng công thức Taylor với phần dư dạng tích phân (Taylor's theorem with integral remainder):
$$R(\mathbf{x} + \sigma \mathbf{u}) = h(1) = h(0) + h'(0) \cdot 1 + \int_0^1 (1-t) h''(t) \, dt$$
$$= R(\mathbf{x}) + \sigma \nabla R(\mathbf{x})^\top \mathbf{u} + \sigma^2 \int_0^1 (1-t) \mathbf{u}^\top \nabla^2 R(\mathbf{x} + t \sigma \mathbf{u}) \mathbf{u} \, dt \tag{4.1}$$

Lấy kỳ vọng hai vế của (4.1) theo phân phối chuẩn tắc $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$:
$$R_\sigma(\mathbf{x}) = \mathbb{E}_{\mathbf{u}}[R(\mathbf{x} + \sigma \mathbf{u})] = R(\mathbf{x}) + \sigma \nabla R(\mathbf{x})^\top \underbrace{\mathbb{E}_{\mathbf{u}}[\mathbf{u}]}_{=\mathbf{0}} + \sigma^2 \mathbb{E}_{\mathbf{u}} \left[ \int_0^1 (1-t) \mathbf{u}^\top \nabla^2 R(\mathbf{x} + t \sigma \mathbf{u}) \mathbf{u} \, dt \right]$$

Số hạng tuyến tính bậc nhất triệt tiêu hoàn toàn vì $\mathbb{E}[\mathbf{u}] = \mathbf{0}$. Chuyển vế $R(\mathbf{x})$:
$$R_\sigma(\mathbf{x}) - R(\mathbf{x}) = \sigma^2 \mathbb{E}_{\mathbf{u}} \left[ \int_0^1 (1-t) \mathbf{u}^\top \nabla^2 R(\mathbf{x} + t \sigma \mathbf{u}) \mathbf{u} \, dt \right]$$

Áp dụng giả định ma trận Hessian bị chặn phổ $\|\nabla^2 R(\mathbf{y})\|_2 \le H$ với mọi $\mathbf{y} \in \mathbb{R}^D$:
$$|\mathbf{u}^\top \nabla^2 R(\mathbf{y}) \mathbf{u}| \le \|\nabla^2 R(\mathbf{y})\|_2 \|\mathbf{u}\|_2^2 \le H \|\mathbf{u}\|_2^2$$

Lấy trị tuyệt đối sai số và áp dụng bất đẳng thức tam giác tích phân:
$$|R_\sigma(\mathbf{x}) - R(\mathbf{x})| \le \sigma^2 \int_0^1 (1-t) \mathbb{E}_{\mathbf{u}}\left[ H \|\mathbf{u}\|_2^2 \right] dt = \sigma^2 H \mathbb{E}_{\mathbf{u}}[\|\mathbf{u}\|_2^2] \int_0^1 (1-t) \, dt$$

Tính toán hai đại lượng độc lập:
1. $\int_0^1 (1-t) \, dt = \left[ t - \frac{t^2}{2} \right]_0^1 = 1 - \frac{1}{2} = \frac{1}{2}$.
2. Chuẩn bình phương của vector Gauss $\|\mathbf{u}\|_2^2 = \sum_{i=1}^D u_i^2$ tuân theo phân phối $\chi^2(D)$, do đó:
   $$\mathbb{E}_{\mathbf{u}}[\|\mathbf{u}\|_2^2] = \sum_{i=1}^D \mathbb{E}[u_i^2] = \sum_{i=1}^D 1 = D$$

Thay vào bất đẳng thức, ta thu được chặn trên sai số Bias:
$$\boxed{\operatorname{Bias}(\sigma, \mathbf{x}) \triangleq |R_\sigma(\mathbf{x}) - R(\mathbf{x})| \le \frac{D \cdot H}{2} \sigma^2}$$
Chứng minh phần sai số hoàn tất. $\blacksquare$

---

### PHẦN 2: THIẾT LẬP BÀI TOÁN TỐI ƯU HÓA MINIMAX & TÌM NGHIỆM GIẢI TÍCH

Để tìm bán kính làm mịn tối ưu $\sigma$, ta cân bằng giữa 2 lực lượng cạnh tranh:
* **Mục tiêu 1**: Giảm độ dốc Lipschitz theo Định lý 1: $\mathcal{L}_{\text{bound}}(\sigma) = \frac{\Delta R}{\sigma \sqrt{2\pi}}$.
* **Mục tiêu 2**: Hạn chế sai số làm méo mó hàm phần thưởng: $\mathcal{B}_{\text{bound}}(\sigma) = \frac{D \cdot H}{2} \sigma^2$.

Hàm chi phí tổng thể trên khoảng mở $\sigma \in (0, \infty)$ là:
$$\mathcal{J}(\sigma) \triangleq \frac{C_1}{\sigma} + C_2 D \sigma^2$$
với các hằng số dương độc lập với $\sigma$ và $D$:
$$C_1 \triangleq \frac{\Delta R}{\sqrt{2\pi}} > 0, \quad C_2 \triangleq \frac{H}{2} > 0$$

#### Bước 1: Tìm điểm dừng (Critical Point)
Hàm $\mathcal{J}(\sigma)$ khả vi liên tục trên $(0, \infty)$. Đạo hàm cấp một theo $\sigma$:
$$\frac{d\mathcal{J}}{d\sigma} = -\frac{C_1}{\sigma^2} + 2 C_2 D \sigma$$

Điểm dừng thỏa mãn phương trình đạo hàm bằng 0:
$$\frac{d\mathcal{J}}{d\sigma} = 0 \iff 2 C_2 D \sigma = \frac{C_1}{\sigma^2} \iff \sigma^3 = \frac{C_1}{2 C_2 D}$$

Thay lại giá trị của $C_1$ và $2 C_2 = H$:
$$\sigma^3 = \frac{\Delta R}{\sqrt{2\pi} \cdot H \cdot D}$$

Lấy căn bậc ba hai vế (vì $\sigma > 0$):
$$\boxed{\sigma_{\text{opt}} = \left( \frac{\Delta R}{\sqrt{2\pi} \cdot H \cdot D} \right)^{1/3} = \left( \frac{\Delta R}{\sqrt{2\pi} H} \right)^{1/3} \cdot D^{-1/3} \propto \mathcal{O}\left( D^{-1/3} \right)}$$

#### Bước 2: Kiểm tra tính lồi và cực tiểu toàn cục duy nhất
Tính đạo hàm cấp hai:
$$\frac{d^2\mathcal{J}}{d\sigma^2} = \frac{2 C_1}{\sigma^3} + 2 C_2 D$$

Vì $C_1 > 0$, $C_2 > 0$, $D \ge 1$ và $\sigma > 0$, ta có:
$$\frac{d^2\mathcal{J}}{d\sigma^2} > 0, \quad \forall \sigma \in (0, \infty)$$
Hàm $\mathcal{J}(\sigma)$ là **hàm lồi nghiêm ngặt (strictly convex)** trên toàn miền xác định. Hơn nữa:
$$\lim_{\sigma \to 0^+} \mathcal{J}(\sigma) = +\infty, \quad \lim_{\sigma \to +\infty} \mathcal{J}(\sigma) = +\infty$$
Do đó điểm dừng $\sigma_{\text{opt}}$ là **cực tiểu toàn cục duy nhất (unique global minimum)** của bài toán Minimax.

Chứng minh hoàn tất 100%. $\blacksquare$
