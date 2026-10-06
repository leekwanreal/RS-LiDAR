# Chứng Minh Toán Học: Định Lý 2 (Lipschitz Contraction & Critical Threshold)

> **Định lý tương ứng**: [`../theorems/theorem_2_phase_transition.md`](../theorems/theorem_2_phase_transition.md)  
> **Mục tiêu**: Chứng minh cận kép $L_{\text{after}} \le \min\left( L_{\text{before}}, \frac{\Delta R}{\sigma \sqrt{2\pi}} \right)$ và thiết lập ngưỡng chuyển pha $\sigma^* = \frac{\Delta R}{L_{\text{before}} \sqrt{2\pi}}$ khi hàm $R$ là hàm $L_{\text{before}}$-Lipschitz.

---

### BƯỚC 1: CHỨNG MINH CẬN KHÔNG GIÃN NỞ ($L_{\text{after}} \le L_{\text{before}}$) QUA ĐẠO HÀM YẾU SOBOLEV

Giả sử $R \in W^{1,\infty}(\mathbb{R}^D)$ với hằng số Lipschitz toàn cục $L_{\text{before}} < \infty$.

1. Theo **Định lý Rademacher**, $R$ khả vi theo nghĩa cổ điển tại hầu khắp nơi trên $\mathbb{R}^D$ theo độ đo Lebesgue. Gradient yếu $\nabla R \in L^\infty(\mathbb{R}^D; \mathbb{R}^D)$ thỏa mãn:
   $$\|\nabla R(\mathbf{y})\|_2 \le L_{\text{before}} \quad \text{cho hầu hết mọi } \mathbf{y} \in \mathbb{R}^D$$

2. Xét tích chập $R_\sigma = R * \phi_\sigma$. Theo tính chất phân phối vi phân của hàm Schwartz $\phi_\sigma \in \mathcal{S}(\mathbb{R}^D)$, đạo hàm của hàm tích chập bằng tích chập của đạo hàm yếu:
   $$\nabla R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} \nabla R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$

3. Lấy chuẩn Euclidean $\|\cdot\|_2$ và áp dụng bất đẳng thức tích phân Minkowski (hoặc bất đẳng thức Jensen đối với độ đo xác suất $\phi_\sigma(\mathbf{z}) d\mathbf{z}$):
   $$\|\nabla R_\sigma(\mathbf{x})\|_2 = \left\| \int_{\mathbb{R}^D} \nabla R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z} \right\|_2 \le \int_{\mathbb{R}^D} \|\nabla R(\mathbf{x} - \mathbf{z})\|_2 \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$

4. Do $\|\nabla R(\mathbf{x} - \mathbf{z})\|_2 \le L_{\text{before}}$ a.e., và nhân Gauss có tổng diện tích bằng 1 ($\int_{\mathbb{R}^D} \phi_\sigma(\mathbf{z}) d\mathbf{z} = 1$):
   $$\|\nabla R_\sigma(\mathbf{x})\|_2 \le L_{\text{before}} \int_{\mathbb{R}^D} \phi_\sigma(\mathbf{z}) \, d\mathbf{z} = L_{\text{before}} \cdot 1 = L_{\text{before}}$$

Lấy supremum theo mọi $\mathbf{x} \in \mathbb{R}^D$:
$$L_{\text{after}} \le L_{\text{before}} \tag{2.1}$$

---

### BƯỚC 2: CHỨNG MINH CẬN CO RÚT TIỆM CẬN ($L_{\text{after}} \le \frac{\Delta R}{\sigma \sqrt{2\pi}}$)

Do $R$ có biên độ hữu hạn $\Delta R < \infty$, áp dụng trực tiếp kết quả của **Định lý 1** ([`proof_theorem_1.md`](proof_theorem_1.md)):
$$\|\nabla R_\sigma(\mathbf{x})\|_2 \le \frac{\Delta R}{\sigma \sqrt{2\pi}}, \quad \forall \mathbf{x} \in \mathbb{R}^D$$

Lấy supremum theo $\mathbf{x} \in \mathbb{R}^D$:
$$L_{\text{after}} \le \frac{\Delta R}{\sigma \sqrt{2\pi}} \tag{2.2}$$

---

### BƯỚC 3: HỢP NHẤT CẬN KÉP VÀ XÁC LẬP NGƯỠNG CHUYỂN PHA $\sigma^*$

Kết hợp đồng thời hai bất đẳng thức (2.1) và (2.2):
$$L_{\text{after}} \le \min\left( L_{\text{before}}, \, \frac{\Delta R}{\sigma \sqrt{2\pi}} \right)$$

Xét tỷ số co rút độ dốc giữa trước và sau khi làm mịn:
$$\frac{L_{\text{before}}}{L_{\text{after}}} \ge \frac{L_{\text{before}}}{\min\left( L_{\text{before}}, \, \frac{\Delta R}{\sigma \sqrt{2\pi}} \right)} = \max\left( 1, \, \frac{L_{\text{before}} \sigma \sqrt{2\pi}}{\Delta R} \right)$$

Định nghĩa điểm chuyển pha giải tích $\sigma^*$ là giá trị tại đó hai đại lượng trong hàm $\min$ bằng nhau:
$$L_{\text{before}} = \frac{\Delta R}{\sigma^* \sqrt{2\pi}} \iff \boxed{\sigma^* \triangleq \frac{\Delta R}{L_{\text{before}} \sqrt{2\pi}}}$$

Thay $\sigma^*$ vào tỷ số co rút:
$$\frac{L_{\text{before}}}{L_{\text{after}}} \ge \max\left( 1, \, \frac{\sigma}{\sigma^*} \right) = \begin{cases} 
1 & \text{khi } \sigma \le \sigma^* \quad \textbf{(Vùng Bảo Toàn)} \\[10pt]
\dfrac{\sigma}{\sigma^*} & \text{khi } \sigma > \sigma^* \quad \textbf{(Vùng Co Rút Chủ Động)}
\end{cases}$$

Chứng minh hoàn tất 100%. $\blacksquare$
