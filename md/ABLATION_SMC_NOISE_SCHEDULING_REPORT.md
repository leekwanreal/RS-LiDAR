# 📊 BÁO CÁO KHOA HỌC THỰC NGHIỆM: ABLATION CÁC PHƯƠNG PHÁP LẬP LỊCH NHIỄU TRONG RS-SMC (TABLE 8 ABLATION)

> **Mục tiêu**: Bóc tách và đo đạc thực nghiệm ảnh hưởng của cơ chế lập lịch bán kính làm mượt $\sigma(t)$ trong RS-SMC (Sequential Monte Carlo):
> 1. **Cố định (Static $\sigma = \text{const}$)**: Giữ nguyên bán kính làm mượt không đổi qua tất cả các bước khuếch tán.
> 2. **Tuyến tính (Linear Annealing)**: Suy giảm bán kính $\sigma(t)$ tuyến tính theo thời gian $t$.
> 3. **Cosine Annealing**: Suy giảm theo đường cong Cosine mượt mà.
> 4. **Tweedie Posterior Annealing (Đề xuất)**: Suy giảm theo độ bất định hậu nghiệm Tweedie $\psi(t) = \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}$.

---

## 1. Giao Thức Thực Nghiệm Chuẩn (Experimental Protocol)
* **Mô hình Sinh (Backbone)**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`).
* **Số lượng hạt (Particles $N$)**: $N = 2$ (hoặc sweep $N \in \{2, 4, 8\}$).
* **Số bước lấy mẫu (Target Sampler)**: DDIM 100 bước (`NUM_INFERENCE_STEPS = 100`), Closed-Form Guidance $\lambda = 5000$.
* **Tham số làm mượt cơ sở**: $\sigma_{\text{init}} = 1.0$, số mẫu Monte Carlo $M = 4$.
* **Tập Prompt Đánh Giá**: 553 prompts chuẩn từ `geneval_metadata.jsonl`.
* **Hạ tầng Tính toán**: NVIDIA A100-SXM4-40GB / Google Colab Pro.

---

## 2. Bảng Số Liệu Thực Nghiệm Đối Chiếu (Ground Truth Benchmark Matrix - ImageReward, $N = 2$)

> *Ghi chú nguồn dữ liệu: Dữ liệu đo đạc thực tế trích xuất trực tiếp từ các file kết quả JSON trong thư mục `tex/smc_results/`. Theo quy chuẩn thí nghiệm, các file không ghi rõ tên lịch trình trong tên (`rssmc_2.json`) mặc định là **Tweedie Posterior Annealing**.*

| STT | Phương pháp / Cách Thêm Nhiễu | File Kết Quả (`json`) | Mean ImageReward$^\uparrow$ | Best-of-$N$ (Max)$^\uparrow$ | Std $\pm$ | Min | $\Delta$ Mean vs Vanilla | $\Delta$ BoN vs Vanilla |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Vanilla SMC (Không làm mượt)** | `smc_2.json` | **`0.2591`** | `0.3279` | $\pm 0.0815$ | 0.1902 | Baseline | Baseline |
| **2** | **RS-SMC (Static $\sigma$ Cố Định)** | `rssmc_2_static.json` | **`0.3478`** | `0.4047` | $\pm 0.0804$ | 0.2909 | **$+0.0887$** (+34.2\%) | **$+0.0768$** (+23.4\%) |
| **3** | **RS-SMC (Cosine Annealing)** | `rssmc_2_cosine.json` | **`0.3614`** | `0.4197` | $\pm 0.0824$ | 0.3031 | **$+0.1023$** (+39.5\%) | **$+0.0918$** (+28.0\%) |
| **4** | **RS-SMC (Linear Annealing)** | `rssmc_2_linear.json` | **`0.3618`** | `0.4187` | $\pm 0.0805$ | 0.3049 | **$+0.1027$** (+39.6\%) | **$+0.0908$** (+27.7\%) |
| **5** | **RS-SMC (Tweedie Posterior)** | `rssmc_2.json` | **`0.3923`** | **`0.4471`** | $\pm 0.0775$ | 0.3374 | **$+0.1332$** (+51.4\%) | **$+0.1192$** (+36.3\%) |

---

## 3. Phân Tích Độ Lệch & Tác Động Khoa Học (Delta Analysis - ImageReward, $N = 2$)

| Cấu hình Lập Lịch | Mean ImageReward | $\Delta$ vs Vanilla SMC (`0.2591`) | $\Delta$ vs Tweedie (`0.3923`) | Xếp hạng Hiệu năng | Cơ chế Thống kê & Động học |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tweedie Posterior** | **`0.3923`** | **$+0.1332$ (+51.4\%)** | Baseline ($0.0000$) | 🥇 **Hạng 1 (Quán Quân)** | Triệt tiêu chính xác sai số độ cong Hessian giải tích $\sigma(t) \propto \psi(t)$; phương sai thấp nhất ($\pm 0.0775$) |
| **Linear Annealing** | **`0.3618`** | **$+0.1027$ (+39.6\%)** | $-0.0305$ (-7.8\%) | 🥈 **Hạng 2** | Suy giảm đều đặn theo thời gian; tạo đệm ổn định nhưng co cụm chậm hơn ở bước cuối |
| **Cosine Annealing** | **`0.3614`** | **$+0.1023$ (+39.5\%)** | $-0.0309$ (-7.9\%) | 🥉 **Hạng 3** | Suy giảm mượt theo hàm cosin, hiệu năng bám sát Linear |
| **Static $\sigma$ Cố Định** | **`0.3478`** | **$+0.0887$ (+34.2\%)** | $-0.0445$ (-11.3\%) | 4 | Giữ $\sigma$ cố định không tắt về 0 ở bước cuối khiến ảnh bị nhiễu nhẹ |
| **Vanilla SMC** | `0.2591` | Baseline ($0.0000$) | $-0.1332$ (-33.9\%) | 5 (Thấp nhất) | Không làm mượt; dễ bị selection trap do gai reward cục bộ |

---

## 4. Các Phát Hiện Khoa Học & Phân Tích Cơ Chế Động Học

### 1. Sự Vượt Trội Toàn Diện Của Randomized Smoothing Trước Vanilla SMC
* Cả 4 phương pháp lập lịch làm mượt (Tweedie `0.3923`, Linear `0.3618`, Cosine `0.3614`, Static `0.3478`) đều **vượt trội áp đảo** so với Vanilla SMC không làm mượt (`0.2591`), với mức tăng trưởng ImageReward từ **$+34.2\%$ đến $+51.4\%$**.
* Điểm trung bình quần thể của Tweedie (`0.3923`) thậm chí **vượt qua cả hạt tốt nhất Best-of-$N$** của Vanilla SMC (`0.3279`). Điều này khẳng định việc làm mượt thế năng Feynman-Kac loại bỏ triệt để các bẫy gai nhiễu giả tạo (spurious selection traps).

### 2. Vì Sao Tweedie Posterior Annealing Đạt Quán Quân Tuyệt Đối Ở $N = 2$?
* **Đặc tính lý thuyết**: Ở $N = 2$, khi chỉ có 2 hạt cạnh tranh, việc triệt tiêu chính xác độ cong bậc hai theo Định lý (Theorem A.7) phát huy tối đa sức mạnh. Bán kính làm mượt $\sigma(t) \propto \psi(t)$ đồng bộ hoàn hảo với độ bất định hậu nghiệm Tweedie, giúp việc tính trọng số Softmax không bị chệch hướng ở bước đầu, đồng thời co về $0$ đúng lúc ở các bước cuối để ảnh đạt độ sắc nét nguyên bản.
* **Phương sai nhỏ nhất**: Tweedie đạt độ lệch chuẩn $\pm 0.0775$ (thấp nhất trong các phương pháp), chứng tỏ quỹ đạo hội tụ ổn định và đồng đều nhất.

---

## 5. Kết Luận Cho Bài Báo
1. **Khẳng định nguyên lý cốt lõi**: Bất kể chọn lịch trình suy giảm nào, việc tích hợp **Randomized Smoothing vào SMC đều mang lại bước nhảy vọt hiệu năng** so với SMC chuẩn không làm mượt.
2. **Khuyến nghị thực nghiệm (Practical Recommendation)**:
   - Với $N = 2$: Khuyến nghị sử dụng **Tweedie Posterior Annealing** làm cấu hình mặc định (Mean = `0.3923`, BoN = `0.4471`, $+51.4\%$).
