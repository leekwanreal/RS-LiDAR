### Theorem 2: Variance-Aware Lipschitz Reduction in the Conservative Regime

**Định lý:**
Trong vùng nhiễu bảo toàn ($\sigma < \sigma^*$), mặc dù cận vĩ mô dựa trên biên độ chưa bị bẻ gãy, hệ số Lipschitz của hàm phần thưởng sau khi làm trơn ($L_{\text{after}}$) vẫn được chứng minh là giảm tuyệt đối so với hàm gốc ($L_{\text{before}}$) dựa trên phương sai cục bộ của các vector gradient.

**Suy luận toán học:**

1. **Định nghĩa gradient làm trơn:**
Theo tính chất của tích chập và kỳ vọng, gradient của hàm sau khi thêm nhiễu bằng kỳ vọng của các gradient tại các điểm lân cận:

$$\nabla R_\sigma(\mathbf{x}) = \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [\nabla R(\mathbf{x} + \sigma \mathbf{u})]$$


2. **Phân rã Phương sai - Kỳ vọng (Bias-Variance Decomposition):**
Xét biến ngẫu nhiên $\mathbf{g}(\mathbf{u}) \triangleq \nabla R(\mathbf{x} + \sigma \mathbf{u})$. Áp dụng hằng đẳng thức phân rã cho chuẩn L2 của vector ngẫu nhiên, ta thu được mối liên hệ trực tiếp giữa chuẩn của kỳ vọng và kỳ vọng của chuẩn:

$$\Vert{}\mathbb{E}[\mathbf{g}(\mathbf{u})]\Vert{}_2 = \sqrt{\mathbb{E}[\Vert{}\mathbf{g}(\mathbf{u})\Vert{}_2^2] - \operatorname{tr}\big(\operatorname{Var}(\mathbf{g}(\mathbf{u}))\big)}$$


3. **Thiết lập cận trên:**
Bởi vì hàm gốc $R$ là $L_{\text{before}}$-Lipschitz, chuẩn gradient của nó bị chặn: $\Vert{}\nabla R\Vert{}_2 \le L_{\text{before}}$, suy ra $\mathbb{E}[\Vert{}\nabla R\Vert{}_2^2] \le L_{\text{before}}^2$. Thay vào hằng đẳng thức trên, ta có kết quả cuối cùng:

$$\boxed{L_{\text{after}} \le \sqrt{L_{\text{before}}^2 - \mathcal{V}(\sigma, \mathbf{x})}}$$



Trong đó, $\mathcal{V}(\sigma, \mathbf{x}) \triangleq \operatorname{tr}\big(\operatorname{Var}(\nabla R(\mathbf{x} + \sigma \mathbf{u}))\big) \ge 0$ đại diện cho sự nhiễu loạn hướng (misalignment) của các vector gradient trong bán kính nhiễu $\sigma$.

**Tác dụng thực tiễn:**
Định lý này bảo vệ mặt lý thuyết cho hiện tượng suy giảm $L_{\max}$ được quan sát trong thực nghiệm ở các mức nhiễu rất nhỏ ($\sigma \in [0.1, 0.25]$). Trong các mạng nơ-ron sâu (như ImageReward), bề mặt hàm số có nhiều nhiễu cục bộ (adversarial spikes), khiến các vector gradient biến thiên liên tục về hướng. Sự bất đồng nhất này tạo ra một phương sai $\mathcal{V} > 0$, làm các gradient ngược hướng tự triệt tiêu lẫn nhau. Qua đó, hệ số Lipschitz thực tế bị ép giảm xuống dưới $L_{\text{before}}$ ngay cả khi $\sigma$ chưa đạt đến ngưỡng vĩ mô $\sigma^*$.

---

### Theorem 3: Dimensionality Dependence of Optimal Smoothing ($\sigma_{\text{opt}}$)

**Định lý:**
Mức độ nhiễu tối ưu $\sigma_{\text{opt}}$ để làm trơn bề mặt mạng nơ-ron tỷ lệ nghịch với căn bậc ba của số chiều không gian $D$. Sự cân bằng Minimax giữa độ trơn (Lipschitz Bound) và sai số xấp xỉ (Bias Error) phụ thuộc chặt chẽ vào độ phân giải dữ liệu.

**Suy luận toán học:**

1. **Thiết lập Sai số Xấp xỉ (Taylor Bias Bound):**
Giả sử hàm $R$ có ma trận Hessian bị chặn $\Vert{}\nabla^2 R(\mathbf{x})\Vert{}_2 \le H$. Khai triển Taylor bậc hai của $R(\mathbf{x} + \sigma \mathbf{u})$ và lấy kỳ vọng, các số hạng bậc nhất triệt tiêu do $\mathbb{E}[\mathbf{u}] = \mathbf{0}$. Sai số chỉ còn lại biểu thức phụ thuộc vào Trace của Hessian:

$$\text{Bias}(\mathbf{x}) \approx \frac{\sigma^2}{2} \operatorname{tr}(\nabla^2 R(\mathbf{x})) \le \frac{D \cdot H}{2} \sigma^2$$


2. **Bài toán Tối ưu hóa Minimax:**
Thiết lập hàm mục tiêu $\mathcal{J}(\sigma)$ nhằm cực tiểu hóa tổng chi phí giữa Cận Lipschitz (giảm theo $1/\sigma$) và Sai số (tăng theo $\sigma^2 D$):

$$\mathcal{J}(\sigma) = \frac{\Delta R}{\sigma \sqrt{2\pi}} + \frac{D \cdot H}{2} \sigma^2$$


3. **Nghiệm cực trị:**
Lấy đạo hàm bậc nhất của $\mathcal{J}(\sigma)$ theo $\sigma$ và cho bằng $0$:

$$\frac{d\mathcal{J}}{d\sigma} = 0 \implies \sigma^3 = \frac{\Delta R}{\sqrt{2\pi} \cdot H \cdot D}$$


$$\boxed{\sigma_{\text{opt}} \propto \mathcal{O}\left( D^{-1/3} \right)}$$



**Tác dụng thực tiễn:**
Định lý này giải thích hiện tượng "Lời nguyền số chiều" (Curse of Dimensionality) giữa các mô hình sinh ảnh khác nhau. Nó cung cấp cơ sở toán học để chứng minh tại sao SDXL (ảnh $1024 \times 1024$, không gian $D \approx 3.14 \times 10^6$ chiều) bắt buộc phải sử dụng mức nhiễu rất nhỏ ($\sigma = 0.25$). Nếu tăng $\sigma$, thành phần $D$ khổng lồ sẽ làm sai số bùng nổ, đẩy ảnh văng khỏi data manifold và làm hệ số Lipschitz tăng vọt trở lại. Ngược lại, mô hình SDv1.5 (ảnh $512 \times 512$) có số chiều nhỏ hơn 4 lần, nên tích lũy sai số ít hơn, cho phép sử dụng lượng nhiễu lớn hơn ($\sigma = 1.0$) để mượn động lượng thoát khỏi các hố cục bộ (local minima).

---

### Remark 1: Phân tích Toán học cho Đồ thị Hình chữ U và Sai số Ước lượng Monte Carlo ($M$)

**Phát biểu:**
Mặc dù hệ số Lipschitz giảm mạnh ở vùng nhiễu nhỏ ($\sigma \in [0.1, 0.25]$), thực nghiệm cho thấy hệ số này thường dội ngược (tăng vọt) tạo thành đồ thị hình chữ U khi $\sigma$ tiến tới các giá trị lớn ($\ge 0.5$). Hiện tượng này được giải thích chặt chẽ bằng sự kết hợp giữa **Tính rời rạc khỏi đa tạp (Manifold Departure)** và **Độ chệch dương của ước lượng Monte Carlo (Positive Bias of Monte Carlo Estimation)**.

**Suy luận toán học:**

Trong thực hành, thay vì có thể tính được kỳ vọng giải tích tuyệt đối, ta phải xấp xỉ gradient làm trơn thông qua trung bình của $M$ mẫu nhiễu Monte Carlo độc lập:


$$\widehat{\nabla} R_\sigma(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M \nabla R(\mathbf{x} + \sigma \mathbf{u}_m)$$

Theo lý thuyết học thống kê, chuẩn bình phương của một ước lượng trung bình mẫu luôn bị chệch dương (positively biased) so với chuẩn bình phương thực tế. Cụ thể, kỳ vọng của hệ số Lipschitz đo đạc được thông qua $M$ mẫu tuân theo hằng đẳng thức:


$$\mathbb{E} \left[ \Vert{}\widehat{\nabla} R_\sigma(\mathbf{x})\Vert{}_2^2 \right] = \Vert{}\nabla R_\sigma(\mathbf{x})\Vert{}_2^2 + \frac{1}{M} \operatorname{tr}\big(\operatorname{Var}(\nabla R(\mathbf{x} + \sigma \mathbf{u}))\big)$$

Sử dụng định nghĩa phương sai gradient $\mathcal{V}(\sigma, \mathbf{x})$ từ Theorem 2, ta có:


$$\boxed{\mathbb{E} \left[ \Vert{}\widehat{\nabla} R_\sigma(\mathbf{x})\Vert{}_2^2 \right] = \text{True Lipschitz}^2 + \frac{\mathcal{V}(\sigma, \mathbf{x})}{M}}$$

**Phân tích tác động thực tiễn (Sự hình thành chữ U):**

1. **Hiệu ứng trượt Đa tạp (Manifold Departure):** Khi $\sigma$ lớn ($\ge 0.5$), các mẫu nhiễu $\mathbf{x} + \sigma \mathbf{u}_m$ đẩy bức ảnh ra quá xa khỏi không gian dữ liệu tự nhiên (Out-of-Distribution). Đồng thời, việc ép (clamp) giá trị điểm ảnh về dải $[-1, 1]$ gây ra hiện tượng bão hòa (saturation). Tại vùng OOD này, mô hình mạng nơ-ron mất phương hướng, đạo hàm trở nên hỗn loạn cực độ khiến phương sai gradient $\mathcal{V}(\sigma, \mathbf{x})$ phình to khổng lồ.
2. **Hiệu ứng khuếch đại do $M$ nhỏ:** Vì số lượng mẫu Monte Carlo trong quá trình sinh ảnh thực tế thường rất nhỏ nhằm tiết kiệm tính toán (ví dụ $M=4$), đại lượng $\frac{1}{M}$ đóng vai trò là một hệ số phạt (penalty) rất nặng. Một tỷ lệ lớn (như $25\%$ nếu $M=4$) của phương sai bùng nổ $\mathcal{V}(\sigma, \mathbf{x})$ sẽ bị cộng dồn trực tiếp vào giá trị Lipschitz đo đạc được.

Chính sự bùng nổ phương sai do OOD kết hợp với độ chệch từ $M$ mẫu hữu hạn này đã áp đảo sự suy giảm của "True Lipschitz", tạo nên sự dội ngược của $L_{\max}$ ở đuôi đồ thị, minh chứng lý do toán học tại sao việc sử dụng $\sigma$ nhỏ ($0.1-0.25$) lại tối ưu cho sự ổn định của hệ thống sinh ảnh.