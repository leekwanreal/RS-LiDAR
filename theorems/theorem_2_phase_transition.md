# Định Lý 2: Co Rút Lipschitz & Ngưỡng Chuyển Pha (Lipschitz Contraction & Critical Threshold)

> **Mã số**: Theorem 2 (Tương ứng với Proposition 2 trong tài liệu giải tích Lipschitz)  
> **Chứng minh chi tiết**: Xem tại [`../proofs/proof_theorem_2.md`](../proofs/proof_theorem_2.md)

---

## 1. KHÔNG GIAN HÀM & ĐIỀU KIỆN TIÊN QUYẾT

* Hàm phần thưởng $R: \mathbb{R}^D \to \mathbb{R}$ thỏa mãn 2 điều kiện chính quy:
  1. **Tính liên tục Lipschitz toàn cục ban đầu**: $R \in W^{1,\infty}(\mathbb{R}^D)$ với hằng số $L_{\text{before}} < \infty$:
     $$|R(\mathbf{x}_1) - R(\mathbf{x}_2)| \le L_{\text{before}} \|\mathbf{x}_1 - \mathbf{x}_2\|_2, \quad \forall \mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^D$$
  2. **Biên độ dao động hữu hạn**: $\Delta R \triangleq \sup R - \inf R < \infty$.
* **Phạm vi áp dụng**: Các mạng nơ-ron đánh giá sở thích liên tục như ImageReward, CLIP-Score, HPS v2.1, Aesthetic Score.

---

## 2. PHÁT BIỂU ĐỊNH LÝ (FORMAL STATEMENT)

Với bất kỳ hàm phần thưởng $L_{\text{before}}$-Lipschitz và có biên độ $\Delta R < \infty$, hàm làm trơn $R_\sigma$ là hàm $L_{\text{after}}$-Lipschitz toàn cục thỏa mãn **Bất đẳng thức Cận Kép (Dual Non-Asymptotic Bound)**:

$$\boxed{L_{\text{after}} \le \min\left( L_{\text{before}}, \, \frac{\Delta R}{\sigma \sqrt{2\pi}} \right)}$$

Bất đẳng thức này xác lập **Ngưỡng chuyển pha vĩ mô duy nhất (Critical Threshold)**:
$$\boxed{\sigma^* \triangleq \frac{\Delta R}{L_{\text{before}} \sqrt{2\pi}}}$$

Tỷ số co rút Lipschitz thỏa mãn:
$$\boxed{\frac{L_{\text{before}}}{L_{\text{after}}} \ge \max\left( 1, \, \frac{\sigma}{\sigma^*} \right) = \begin{cases} 
1 & \text{khi } \sigma \le \sigma^* \quad \textbf{(Vùng Bảo Toàn - Conservative Regime)} \\[10pt]
\dfrac{\sigma}{\sigma^*} & \text{khi } \sigma > \sigma^* \quad \textbf{(Vùng Co Rút Chủ Động - Active Smoothing Regime)}
\end{cases}}$$

---

## 3. PHÂN TÍCH HAI VÙNG HOẠT ĐỘNG (TWO OPERATING REGIMES)

1. **Vùng Bảo Toàn ($\sigma \le \sigma^*$ - Conservative Regime)**:
   - Khi bán kính làm mịn nhỏ hơn ngưỡng $\sigma^*$, cận trên bị chi phối bởi $L_{\text{before}}$.
   - Phép tích chập Gauss bảo đảm tính **không giãn nở (Non-expansion)**: $L_{\text{after}} \le L_{\text{before}}$, tuyệt đối không làm khuếch đại gradient hay phát sinh bất ổn định số học.
2. **Vùng Co Rút Chủ Động ($\sigma > \sigma^*$ - Active Smoothing Regime)**:
   - Khi $\sigma$ vượt ngưỡng $\sigma^*$, cận thứ hai trở nên kích hoạt độc quyền.
   - Hệ số Lipschitz co rút đơn điệu với tốc độ $\mathcal{O}(1/\sigma)$, làm phẳng mọi sườn dốc hiểm trở và dập tắt các cực đại giả tạo (spurious local maxima) trên bề mặt phần thưởng.

---

## 4. LIÊN HỆ VỚI NGHỊCH LÝ $\sigma^*$ TRONG THỰC TIỄN

* **Nghịch lý giá trị**: Với ImageReward ($\Delta R \approx 4.0, L_{\text{before}} \approx 0.01548$), công thức tính ra $\sigma^* \approx 103.1$ (quá lớn trên thang ảnh $[-1, 1]$).
* **Vai trò**: Định lý 2 cung cấp **chặn an toàn toàn cục trong trường hợp xấu nhất (worst-case guarantee)**. Để giải thích vì sao trong thực tế chỉ cần $\sigma = 0.1 - 0.25$ mà $L$ đã giảm mạnh, ta cần bước chuyển tiếp sang **Định lý 3 (Variance-Aware Reduction)**.
