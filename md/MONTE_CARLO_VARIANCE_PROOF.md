# 📐 CHỨNG MINH TOÁN HỌC: PHƯƠNG SAI BỘ ƯỚC LƯỢNG MONTE CARLO TRONG RS-LiDAR

> **Tài liệu tham khảo chuyên sâu cho buổi báo cáo & phụ lục nghiên cứu**  
> **Chủ đề**: Chứng minh công thức phương sai $\text{Var}\left(\tilde{r}_{\sigma, M}\right) = \frac{1}{M} \text{Var}_{\boldsymbol{\xi}}\left(r(x + \boldsymbol{\xi})\right) \propto \frac{1}{M}$, tốc độ hội tụ $\mathcal{O}(1/\sqrt{M})$, và tác động động lực học đến trường dẫn đường RS-LiDAR.

---

## 1. ĐẶT BÀI TOÁN & CÁC ĐỊNH NGHĨA

### 1.1. Hàm thưởng làm mịn lý thuyết (True Smoothed Reward)
Xét không gian ảnh sinh ra $x \in \mathbb{R}^d$ và prompt văn bản điều kiện $c \in \mathcal{C}$. Cho $r: \mathbb{R}^d \times \mathcal{C} \to \mathbb{R}$ là một mô hình phần thưởng thị giác (ví dụ: ImageReward, CLIP-Score, HPS v2.1).

Do $r(x, c)$ có bề mặt nhạy cảm với nhiễu vi mô của pixel (hằng số Lipschitz cục bộ bùng nổ vô hạn $L \to \infty$), phương pháp **Randomized Smoothing** định nghĩa hàm thưởng làm mịn lý thuyết $r_\sigma(x, c)$ thông qua kỳ vọng lấy mẫu nhiễu Gaussian đa chiều $\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_d)$:

$$r_\sigma(x, c) \triangleq \mathbb{E}_{\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_d)} \left[ r(x + \boldsymbol{\xi}, c) \right] = \int_{\mathbb{R}^d} r(x + \boldsymbol{\xi}, c) \, p(\boldsymbol{\xi}) \, d\boldsymbol{\xi}$$

trong đó hàm mật độ xác suất Gaussian chuẩn tắc là:
$$p(\boldsymbol{\xi}) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp\left( - \frac{\|\boldsymbol{\xi}\|_2^2}{2\sigma^2} \right)$$

### 1.2. Bộ ước lượng Monte Carlo thực nghiệm với $M$ mẫu (Empirical Estimator)
Trong quá trình suy luận test-time, tích phân trên không gian $d$ chiều ($d = 512 \times 512 \times 3$ hoặc $1024 \times 1024 \times 3$) không thể tính dạng giải tích đóng. Do đó, ta xấp xỉ bằng phương pháp Monte Carlo với $M$ vector nhiễu độc lập cùng phân phối (i.i.d.):

$$\boldsymbol{\xi}_1, \boldsymbol{\xi}_2, \dots, \boldsymbol{\xi}_M \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2 \mathbf{I}_d)$$

Bộ ước lượng thực nghiệm $\tilde{r}_{\sigma, M}(x, c)$ được định nghĩa là trung bình mẫu:

$$\tilde{r}_{\sigma, M}(x, c) \triangleq \frac{1}{M} \sum_{m=1}^M r\left(x + \boldsymbol{\xi}_m, c\right)$$

Để đơn giản hóa ký hiệu, đặt biến ngẫu nhiên vô hướng:
$$Y_m \triangleq r(x + \boldsymbol{\xi}_m, c) \quad (m = 1, 2, \dots, M)$$

Khi đó:
$$\tilde{r}_{\sigma, M}(x, c) = \frac{1}{M} \sum_{m=1}^M Y_m$$

---

## 2. PHÁT BIỂU ĐỊNH LÝ (THEOREM STATEMENT)

### Định Lý 1 (Tính Không Chệch & Phương Sai Của Bộ Ước Lượng Monte Carlo)
*Giả sử hàm phần thưởng $r(x, c)$ khả tích bình phương đối với độ đo Gauss $\mathcal{N}(0, \sigma^2 \mathbf{I}_d)$, tức là phương sai hữu hạn:*
$$\sigma_r^2(x) \triangleq \text{Var}_{\boldsymbol{\xi}}\left( r(x + \boldsymbol{\xi}, c) \right) = \mathbb{E}_{\boldsymbol{\xi}}\left[ \left( r(x + \boldsymbol{\xi}, c) - r_\sigma(x, c) \right)^2 \right] < \infty$$

*Khi đó, bộ ước lượng Monte Carlo $\tilde{r}_{\sigma, M}(x, c)$ thỏa mãn 2 tính chất cơ bản sau:*
1. **Tính không chệch (Unbiasedness)**:
   $$\mathbb{E}\left[ \tilde{r}_{\sigma, M}(x, c) \right] = r_\sigma(x, c)$$
2. **Phương sai tỷ lệ nghịch bậc 1 với số mẫu $M$ (Variance Decay)**:
   $$\text{Var}\left( \tilde{r}_{\sigma, M}(x, c) \right) = \frac{\sigma_r^2(x)}{M} = \frac{1}{M} \text{Var}_{\boldsymbol{\xi}}\left( r(x + \boldsymbol{\xi}, c) \right)$$
3. **Độ lệch chuẩn sai số hội tụ theo bậc $\mathcal{O}(1/\sqrt{M})$**:
   $$\text{SE}\left( \tilde{r}_{\sigma, M}(x, c) \right) \triangleq \sqrt{\text{Var}\left(\tilde{r}_{\sigma, M}(x, c)\right)} = \frac{\sigma_r(x)}{\sqrt{M}} = \mathcal{O}\left( \frac{1}{\sqrt{M}} \right)$$

---

## 3. CHỨNG MINH TOÁN HỌC CHI TIẾT TỪNG BƯỚC

### Bước 1: Xác định tính chất phân phối của biến ngẫu nhiên $Y_m$
Vì $\boldsymbol{\xi}_1, \boldsymbol{\xi}_2, \dots, \boldsymbol{\xi}_M$ là các vector ngẫu nhiên độc lập và cùng phân phối:
$$\boldsymbol{\xi}_m \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2 \mathbf{I}_d)$$
và $Y_m = r(x + \boldsymbol{\xi}_m, c)$ là hàm tất định của biến ngẫu nhiên $\boldsymbol{\xi}_m$, nên theo lý thuyết xác suất, các biến ngẫu nhiên vô hướng $\{Y_m\}_{m=1}^M$ cũng **độc lập và cùng phân phối (i.i.d.)**.

Do đó:
- **Kỳ vọng của mỗi $Y_m$**:
  $$\mathbb{E}[Y_m] = \mathbb{E}_{\boldsymbol{\xi}_m}\left[ r(x + \boldsymbol{\xi}_m, c) \right] = r_\sigma(x, c), \quad \forall m \in \{1, \dots, M\}$$
- **Phương sai của mỗi $Y_m$**:
  $$\text{Var}(Y_m) = \mathbb{E}\left[ (Y_m - \mathbb{E}[Y_m])^2 \right] = \sigma_r^2(x), \quad \forall m \in \{1, \dots, M\}$$

---

### Bước 2: Chứng minh Tính Không Chệch (Unbiasedness)
Áp dụng tính chất tuyến tính của kỳ vọng toán học $\mathbb{E}\left[\sum_{i} a_i X_i\right] = \sum_{i} a_i \mathbb{E}[X_i]$:

$$\begin{aligned}
\mathbb{E}\left[ \tilde{r}_{\sigma, M}(x, c) \right] &= \mathbb{E}\left[ \frac{1}{M} \sum_{m=1}^M Y_m \right] \\
&= \frac{1}{M} \sum_{m=1}^M \mathbb{E}[Y_m] \\
&= \frac{1}{M} \sum_{m=1}^M r_\sigma(x, c) \\
&= \frac{1}{M} \cdot \left( M \cdot r_\sigma(x, c) \right) \\
&= r_\sigma(x, c) \quad \blacksquare
\end{aligned}$$

$\implies$ Ước lượng Monte Carlo không bị lệch (unbiased) đối với bất kỳ giá trị $M \ge 1$ nào.

---

### Bước 3: Chứng minh Công thức Phương sai $\text{Var}\left( \tilde{r}_{\sigma, M} \right) = \frac{\sigma_r^2(x)}{M}$
Áp dụng định nghĩa và tính chất cơ bản của toán tử phương sai đối với hằng số nhân $a = \frac{1}{M}$:
$$\text{Var}(a X) = a^2 \text{Var}(X)$$

Ta có:
$$\text{Var}\left( \tilde{r}_{\sigma, M}(x, c) \right) = \text{Var}\left( \frac{1}{M} \sum_{m=1}^M Y_m \right) = \frac{1}{M^2} \text{Var}\left( \sum_{m=1}^M Y_m \right)$$

Khai triển phương sai của một tổng biến ngẫu nhiên theo công thức tổng quát:
$$\text{Var}\left( \sum_{m=1}^M Y_m \right) = \sum_{m=1}^M \text{Var}(Y_m) + 2 \sum_{1 \le i < j \le M} \text{Cov}(Y_i, Y_j)$$

trong đó $\text{Cov}(Y_i, Y_j)$ là hiệp phương sai giữa $Y_i$ và $Y_j$:
$$\text{Cov}(Y_i, Y_j) = \mathbb{E}\left[ (Y_i - \mathbb{E}[Y_i])(Y_j - \mathbb{E}[Y_j]) \right]$$

Vì $Y_i$ và $Y_j$ (với $i \neq j$) được tính từ hai vector nhiễu hoàn toàn độc lập $\boldsymbol{\xi}_i \perp \boldsymbol{\xi}_j$, nên hai biến ngẫu nhiên này độc lập thống kê:
$$\text{Cov}(Y_i, Y_j) = 0, \quad \forall i \neq j$$

Do toàn bộ các số hạng hiệp phương sai triệt tiêu về 0, ta thu được:
$$\text{Var}\left( \sum_{m=1}^M Y_m \right) = \sum_{m=1}^M \text{Var}(Y_m) = \sum_{m=1}^M \sigma_r^2(x) = M \cdot \sigma_r^2(x)$$

Thay kết quả này trở lại vào biểu thức phương sai của $\tilde{r}_{\sigma, M}$:
$$\begin{aligned}
\text{Var}\left( \tilde{r}_{\sigma, M}(x, c) \right) &= \frac{1}{M^2} \cdot \left( M \cdot \sigma_r^2(x) \right) \\
&= \frac{\sigma_r^2(x)}{M} \\
&= \frac{1}{M} \text{Var}_{\boldsymbol{\xi}}\left( r(x + \boldsymbol{\xi}, c) \right) \quad \blacksquare
\end{aligned}$$

---

### Bước 4: Tốc độ hội tụ theo Định lý Giới hạn Trung tâm (CLT)
Theo **Định lý Giới hạn Trung tâm (Central Limit Theorem - CLT)**, khi $M$ tăng lên, phân phối xác suất của sai số chuẩn hóa hội tụ theo phân phối chuẩn tắc:

$$\sqrt{M} \left( \frac{\tilde{r}_{\sigma, M}(x, c) - r_\sigma(x, c)}{\sigma_r(x)} \right) \xrightarrow{d} \mathcal{N}(0, 1)$$

hay viết dưới dạng sai số tuyệt đối:
$$\left| \tilde{r}_{\sigma, M}(x, c) - r_\sigma(x, c) \right| = \mathcal{O}_p\left( \frac{1}{\sqrt{M}} \right)$$

Độ lệch chuẩn của sai số ước lượng (Standard Error):
$$\text{SE}\left( \tilde{r}_{\sigma, M} \right) = \sqrt{\text{Var}(\tilde{r}_{\sigma, M})} = \frac{\sigma_r(x)}{\sqrt{M}}$$

---

## 4. TÁC ĐỘNG CỦA PHƯƠNG SAI $M$ ĐẾN TRƯỜNG DẪN ĐƯỜNG RS-LiDAR

Tại sao sự suy giảm phương sai theo bậc $\frac{1}{M}$ lại có ý nghĩa quyết định đến chất lượng sinh ảnh trong thuật toán RS-LiDAR?

### 4.1. Hiệu ứng khuếch đại qua hàm Softmax và hệ số nhân $\lambda = 5000$
Trong thuật toán LiDAR / RS-LiDAR, trọng số dẫn đường của hạt lookahead $\hat{x}_0^i$ được tính bằng:
$$w_i^{r, \sigma} = \frac{\exp\left( \lambda \cdot \tilde{r}_{\sigma, M}(\hat{x}_0^i, c) - \frac{\|x_t - \hat{x}_0^i\|^2}{2\sigma_t^2} \right)}{\sum_{j=1}^n \exp\left( \lambda \cdot \tilde{r}_{\sigma, M}(\hat{x}_0^j, c) - \frac{\|x_t - \hat{x}_0^j\|^2}{2\sigma_t^2} \right)}$$

Giả sử sai số ước lượng của hạt thứ $i$ là $\varepsilon_i = \tilde{r}_{\sigma, M}(\hat{x}_0^i, c) - r_\sigma(\hat{x}_0^i, c)$, với $\mathbb{E}[\varepsilon_i] = 0$ và $\text{Var}(\varepsilon_i) = \frac{\sigma_r^2}{M}$.

Trong số mũ của Softmax, số hạng phần thưởng là:
$$\lambda \cdot \tilde{r}_{\sigma, M} = \lambda \cdot r_\sigma + \lambda \cdot \varepsilon_i$$

Phương sai của số hạng nhiễu trong hàm mũ:
$$\text{Var}(\lambda \cdot \varepsilon_i) = \lambda^2 \cdot \text{Var}(\varepsilon_i) = \frac{\lambda^2 \sigma_r^2}{M}$$

Với giá trị mặc định của bài báo gốc $\lambda = 5000$, ta thấy:
$$\lambda^2 = (5000)^2 = 2.5 \times 10^7$$

### 4.2. Phân tích định lượng qua các mốc $M$ (Giải thích cơ sở Slide 8):
1. **Tại $M = 1$ (Mốc bản lề lý thuyết)**:
   - Phương sai sai số đạt cực đại: $\text{Var}(\varepsilon_i) = \sigma_r^2$.
   - Lượng nhiễu này khi nhân với $\lambda^2 = 2.5 \times 10^7$ sẽ tạo ra sự biến thiên ngẫu nhiên khổng lồ giữa các hạt.
   - Hạt nào tình cờ bốc phải vector nhiễu $\boldsymbol{\xi}_m$ làm tăng vọt điểm $r$ sẽ chiếm trọn $100\%$ trọng số $w_i^r$, khiến phân phối bị sụp đổ do nhiễu Monte Carlo chứ không phải do chất lượng hạt thực sự.
   - $\implies$ **Dự đoán kết quả tại $M=1$ sẽ rất kém và bất ổn định, minh chứng thực nghiệm cho thấy việc lấy kỳ vọng ($M > 1$) là bắt buộc**.
2. **Tại $M = 4$ (Mốc mặc định hiện tại)**:
   - Phương sai giảm xuống $4$ lần: $\text{Var} = \frac{\sigma_r^2}{4}$.
   - Độ lệch chuẩn sai số giảm $50\%$: $\text{SE} = \frac{\sigma_r}{2}$.
   - Đủ ổn định để lọc bỏ các gai nhọn, cân bằng giữa tốc độ và độ mượt của trường dẫn đường.
3. **Tại $M = 8$ và $M = 16$ (Khảo sát ngưỡng bão hòa)**:
   - Tại $M = 16$: Phương sai giảm $16$ lần, sai số $\text{SE} = \frac{\sigma_r}{4}$ (giảm $75\%$).
   - Tuy nhiên, thời gian tính toán và lượng VRAM để decode/chấm điểm tăng gấp $4$ lần so với $M=4$.
   - Do quy luật tốc độ hội tụ $\mathcal{O}(1/\sqrt{M})$ có đạo hàm giảm dần $\frac{d}{dM}(M^{-1/2}) = -\frac{1}{2} M^{-3/2} \to 0$, hiệu quả gia tăng cận biên (marginal gain) từ $M=8 \to 16$ sẽ bão hòa rõ rệt.

---

## 5. TỔNG KẾT
| Mức $M$ | Phương sai $\text{Var}(\tilde{r})$ | Độ lệch chuẩn sai số $\text{SE}$ | Chi phí tính toán | Đánh giá kỳ vọng khoa học |
| :---: | :---: | :---: | :---: | :--- |
| **$M = 1$** | $1.00 \cdot \sigma_r^2$ | $1.00 \cdot \sigma_r$ | $1\times$ *(Rất nhanh)* | Bất ổn định nghiêm trọng, khuếch đại nhiễu qua $\lambda=5000$ (Mốc bản lề). |
| **$M = 2$** | $0.50 \cdot \sigma_r^2$ | $0.71 \cdot \sigma_r$ | $2\times$ | Giảm $29\%$ sai số, cải thiện đáng kể so với $M=1$. |
| **$M = 4$** | $0.25 \cdot \sigma_r^2$ | $0.50 \cdot \sigma_r$ | $4\times$ *(Mặc định)* | Sweet Spot cân bằng tối ưu giữa chất lượng và thời gian suy luận. |
| **$M = 8$** | $0.125 \cdot \sigma_r^2$ | $0.35 \cdot \sigma_r$ | $8\times$ | Tăng độ mịn ước lượng, bắt đầu tiệm cận ngưỡng bão hòa. |
| **$M = 16$**| $0.0625 \cdot \sigma_r^2$| $0.25 \cdot \sigma_r$ | $16\times$ *(Nặng VRAM)* | Giảm $75\%$ sai số, bão hòa chất lượng, cần kích hoạt VAE chunking để chống OOM. |
