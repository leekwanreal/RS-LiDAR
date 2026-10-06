# Định Lý 1: Chặn Gradient Tuyệt Đối Toàn Cục (Universal Gradient Boundedness)

> **Mã số**: Theorem 1 (Tương ứng với Theorem 3.1 trong bài báo RS-LiDAR)  
> **Chứng minh chi tiết**: Xem tại [`../proofs/proof_theorem_1.md`](../proofs/proof_theorem_1.md)

---

## 1. KHÔNG GIAN HÀM & ĐIỀU KIỆN TIÊN QUYẾT (FOUNDATIONAL SETTING)

* Xét không gian ảnh Euclid $\mathbb{R}^D$ trang bị độ đo Gauss đẳng hướng $\mathcal{N}(0, \sigma^2 \mathbf{I}_D)$ với bán kính làm mịn $\sigma > 0$.
* Hàm phần thưởng $R: \mathbb{R}^D \to \mathbb{R}$ chỉ cần thỏa mãn duy nhất **Giả định Biên độ Hữu hạn (Bounded Essential Range)**:
  $$\Delta R \triangleq \operatorname{ess\,sup}_{\mathbf{y} \in \mathbb{R}^D} R(\mathbf{y}) - \operatorname{ess\,inf}_{\mathbf{y} \in \mathbb{R}^D} R(\mathbf{y}) < \infty$$
* **Lưu ý đặc biệt**: Hàm gốc $R$ **KHÔNG CẦN KHẢ VI**, **KHÔNG CẦN LIÊN TỤC**, và **KHÔNG CẦN LIPSCHITZ** ($L_{\text{before}}$ có thể bằng $+\infty$, ví dụ như các hàm bước nhảy, reward nhị phân 0/1 của GenEval).

---

## 2. PHÁT BIỂU ĐỊNH LÝ (FORMAL STATEMENT)

Với bất kỳ hàm phần thưởng bị chặn $R \in L^\infty(\mathbb{R}^D)$ và mọi bán kính làm mịn Gaussian $\sigma > 0$, hàm phần thưởng làm mịn Gaussian:
$$R_\sigma(\mathbf{x}) \triangleq \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u})] = \int_{\mathbb{R}^D} R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$
là một hàm khả vi vô hạn lần ($R_\sigma \in C^\infty(\mathbb{R}^D)$), và chuẩn Euclidean của vector gradient tại mọi điểm $\mathbf{x} \in \mathbb{R}^D$ được chặn cứng toàn cục bởi:

$$\boxed{\|\nabla R_\sigma(\mathbf{x})\|_2 \le \frac{\Delta R}{\sigma \sqrt{2\pi}}}$$

---

## 3. Ý NGHĨA VẬT LÝ & HÌNH HỌC TRỰC QUAN

1. **Khởi tạo tính trơn từ con số 0 (Smoothness from Non-smoothness)**:
   - Các hàm đánh giá dựa trên logic Boolean (như GenEval: đếm đủ 3 con mèo thì cho 1, thiếu thì cho 0) có gradient tại điểm biên bằng vô cùng hoặc không xác định.
   - Định lý 1 chứng minh rằng việc áp dụng Randomized Smoothing trong không gian ảnh biến đổi hàm gián đoạn thành một bề mặt cong trơn láng $C^\infty$.
2. **Kỹ thuật chuyển đạo hàm sang nhân Gauss (Stein's Identity)**:
   - Thay vì lấy đạo hàm trên hàm reward $R$, toán tử đạo hàm được đẩy hoàn toàn sang hàm mật độ Gaussian $\phi_\sigma$. Do $\phi_\sigma$ trơn nhẵn vô hạn và suy giảm theo hàm mũ ở vô cực, gradient $\nabla R_\sigma$ luôn luôn xác định và không bao giờ bị nổ số học.
3. **Quy luật suy giảm $\mathcal{O}(1/\sigma)$**:
   - Khi tăng bán kính làm mịn $\sigma$, chặn trên của gradient suy giảm tỷ lệ nghịch tuyệt đối theo tốc độ $\mathcal{O}(1/\sigma)$, bảo đảm rằng không gian phần thưởng luôn được dập tắt dao động khi $\sigma$ tăng.

---

## 4. LIÊN HỆ VỚI THỰC NGHIỆM RS-LiDAR

* **Áp dụng cho GenEval Benchmark**: Giải thích vì sao RS-LiDAR có thể điều hướng hạt khuếch tán bằng Closed-form Guidance ngay cả khi sử dụng mô hình phát hiện vật thể Mask2Former và logic kiểm tra điều kiện nhị phân.
* **Chống nổ gradient ở Phase 2**: Ngăn chặn hoàn toàn hiện tượng rung lắc quỹ đạo sinh ảnh khi guidance scale được đẩy lên cao ($s \ge 15.0$).
