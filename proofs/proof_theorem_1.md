# Chứng Minh Toán Học: Định Lý 1 (Universal Gradient Boundedness)

> **Định lý tương ứng**: [`../theorems/theorem_1_gradient_bound.md`](../theorems/theorem_1_gradient_bound.md)  
> **Mục tiêu**: Chứng minh rằng với MỌI hàm phần thưởng bị chặn $R \in L^\infty(\mathbb{R}^D)$ (kể cả gián đoạn, không Lipschitz), gradient của hàm làm mịn $R_\sigma$ luôn bị chặn bởi $\|\nabla R_\sigma(\mathbf{x})\|_2 \le \frac{\Delta R}{\sigma \sqrt{2\pi}}$.

---

### BƯỚC 1: BIỂU DIỄN GRADIENT QUA ĐẠO HÀM NHÂN GAUSS (STEIN IDENTITY)

Hàm phần thưởng làm mịn Gaussian được định nghĩa bởi tích chập:
$$R_\sigma(\mathbf{x}) \triangleq \int_{\mathbb{R}^D} R(\mathbf{y}) \phi_\sigma(\mathbf{x} - \mathbf{y}) \, d\mathbf{y}$$
trong đó $\phi_\sigma(\mathbf{z}) = \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left(-\frac{\|\mathbf{z}\|_2^2}{2\sigma^2}\right)$.

Do $\phi_\sigma \in \mathcal{S}(\mathbb{R}^D)$ và $R \in L^\infty(\mathbb{R}^D)$, theo quy tắc đạo hàm dưới dấu tích phân Leibniz:
$$\nabla R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{y}) \nabla_\mathbf{x} \phi_\sigma(\mathbf{x} - \mathbf{y}) \, d\mathbf{y}$$

Tính đạo hàm riêng của nhân Gauss theo biến $\mathbf{x}$:
$$\nabla_\mathbf{x} \phi_\sigma(\mathbf{x} - \mathbf{y}) = -\frac{\mathbf{x} - \mathbf{y}}{\sigma^2} \phi_\sigma(\mathbf{x} - \mathbf{y})$$

Thực hiện phép đổi biến số $\mathbf{y} = \mathbf{x} + \sigma \mathbf{u}$ với $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$ (chuẩn tắc):
$$\nabla R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{x} + \sigma \mathbf{u}) \left(-\frac{-\sigma \mathbf{u}}{\sigma^2}\right) \frac{1}{(2\pi)^{D/2}} e^{-\|\mathbf{u}\|_2^2/2} \, d\mathbf{u} = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u}) \, \mathbf{u}]$$

---

### BƯỚC 2: KHỬ TRUNG VỊ (ZERO-MEAN CENTERING)

Đặt $c \triangleq \frac{1}{2} (\operatorname{ess\,sup} R + \operatorname{ess\,inf} R)$ là điểm chính giữa của miền giá trị hữu hạn của $R$.  
Do tính đối xứng tâm của phân phối chuẩn Gauss $\mathcal{N}(0, \mathbf{I}_D)$, kỳ vọng của vector Gauss bằng vector không:
$$\mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [\mathbf{u}] = \mathbf{0} \implies \mathbb{E}_{\mathbf{u}}[c \, \mathbf{u}] = c \, \mathbb{E}_{\mathbf{u}}[\mathbf{u}] = \mathbf{0}$$

Ta có thể trừ $c$ vào biểu thức kỳ vọng mà không làm thay đổi giá trị của tích phân:
$$\nabla R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} \big[ (R(\mathbf{x} + \sigma \mathbf{u}) - c) \, \mathbf{u} \big]$$

Theo định nghĩa của $c$ và biên độ $\Delta R \triangleq \operatorname{ess\,sup} R - \operatorname{ess\,inf} R$:
$$|R(\mathbf{x} + \sigma \mathbf{u}) - c| \le \frac{\Delta R}{2} \quad \text{hầu chắc chắn với mọi } \mathbf{u}$$

---

### BƯỚC 3: CHIẾU ĐẲNG HƯỚNG LÊN MẶT CẦU ĐƠN VỊ (DUAL VARIATIONAL PROJECTION)

Theo biểu diễn đối ngẫu của chuẩn Euclidean thông qua tích vô hướng trên mặt cầu đơn vị $\mathbb{S}^{D-1} \triangleq \{ \mathbf{v} \in \mathbb{R}^D : \|\mathbf{v}\|_2 = 1 \}$:
$$\|\nabla R_\sigma(\mathbf{x})\|_2 = \sup_{\mathbf{v} \in \mathbb{S}^{D-1}} |\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})|$$

Cố định một vector đơn vị tùy ý $\mathbf{v} \in \mathbb{S}^{D-1}$. Tính tích vô hướng:
$$\mathbf{v}^\top \nabla R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u}} \big[ (R(\mathbf{x} + \sigma \mathbf{u}) - c) (\mathbf{v}^\top \mathbf{u}) \big]$$

Áp dụng bất đẳng thức tam giác cho kỳ vọng:
$$|\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})| \le \frac{1}{\sigma} \mathbb{E}_{\mathbf{u}} \left[ |R(\mathbf{x} + \sigma \mathbf{u}) - c| \cdot |\mathbf{v}^\top \mathbf{u}| \right] \le \frac{\Delta R}{2\sigma} \mathbb{E}_{\mathbf{u}}[|\mathbf{v}^\top \mathbf{u}|]$$

---

### BƯỚC 4: TÍNH MÔ-MEN BẬC NHẤT TUYỆT ĐỐI CỦA BIẾN CHUẨN ĐƠN BIẾN

Xét biến ngẫu nhiên $Z \triangleq \mathbf{v}^\top \mathbf{u}$. Do $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$ có tính bất biến quay và $\|\mathbf{v}\|_2 = 1$, phép chiếu tuyến tính $Z$ là biến ngẫu nhiên Gauss chuẩn tắc đơn biến:
$$Z \sim \mathcal{N}(0, \|\mathbf{v}\|_2^2) = \mathcal{N}(0, 1)$$

Tính kỳ vọng trị tuyệt đối của biến chuẩn tắc đơn biến:
$$\mathbb{E}[|Z|] = \int_{-\infty}^\infty |z| \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz = \frac{2}{\sqrt{2\pi}} \int_0^\infty z e^{-z^2/2} \, dz$$

Đặt $t = z^2 / 2 \implies dt = z dz$:
$$\int_0^\infty z e^{-z^2/2} \, dz = \int_0^\infty e^{-t} \, dt = 1$$

Do đó:
$$\mathbb{E}[|Z|] = \frac{2}{\sqrt{2\pi}} \cdot 1 = \sqrt{\frac{2}{\pi}}$$

---

### BƯỚC 5: KẾT LUẬN

Thay kết quả $\mathbb{E}[|Z|] = \sqrt{\frac{2}{\pi}}$ vào bất đẳng thức ở Bước 3:
$$|\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})| \le \frac{\Delta R}{2\sigma} \sqrt{\frac{2}{\pi}} = \frac{\Delta R}{\sigma \sqrt{2\pi}}$$

Vì bất đẳng thức này đúng với mọi vector đơn vị $\mathbf{v} \in \mathbb{S}^{D-1}$, lấy supremum theo $\mathbf{v}$:
$$\|\nabla R_\sigma(\mathbf{x})\|_2 = \sup_{\mathbf{v} \in \mathbb{S}^{D-1}} |\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})| \le \frac{\Delta R}{\sigma \sqrt{2\pi}}$$

Bất đẳng thức đúng với mọi $\mathbf{x} \in \mathbb{R}^D$ mà không cần bất kỳ giả định nào về tính khả vi hay tính Lipschitz của $R$.  
Chứng minh hoàn tất. $\blacksquare$
