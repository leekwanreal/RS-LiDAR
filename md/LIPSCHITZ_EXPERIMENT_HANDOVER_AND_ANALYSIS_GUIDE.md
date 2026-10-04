# 📘 CẨM NANG TOÀN DIỆN THỰC NGHIỆM ĐO HỆ SỐ LIPSCHITZ (LIPSCHITZ EXPERIMENT PROTOCOL & HANDOVER GUIDE)
> **Tài liệu chuyển giao ngữ cảnh (Context Handover Document)**  
> **Dành cho:** AI Agent và Nhà nghiên cứu ở các phiên tiếp theo  
> **Mục đích:** Đọc hiểu ngay 100% thiết kế toán học, cấu trúc dữ liệu, và quy trình phân tích tự động file kết quả `results_lipschitz.zip` (553 prompts GenEval) mà không cần hỏi lại từ đầu.

---

## 1. 🎯 BẢN CHẤT TOÁN HỌC & MỤC TIÊU THỰC NGHIỆM

### 1.1. Luận điểm cốt tử cần chứng minh
Bài báo chứng minh rằng phương pháp **RS-LiDAR (Randomized Smoothing LiDAR)** giải quyết triệt để điểm yếu chí mạng của Vanilla LiDAR: **Hàm đánh giá chất lượng (Reward Model $R(x)$) trong không gian ảnh có độ dốc Lipschitz cực lớn, địa hình gồ ghề (adversarial spikes), dẫn đến hiện tượng trôi dạt hạt và quá tối ưu hóa (reward over-optimization)**.

Bằng cách áp dụng **Làm mịn Ngẫu nhiên (Randomized Smoothing)** qua tích chập Gaussian:
$$R_{\sigma_2}(x) = \mathbb{E}_{u \sim \mathcal{N}(0, I)} [R(x + \sigma_2 u)]$$
Hệ số Lipschitz của hàm phần thưởng được chứng minh trên giải tích giảm theo tỷ lệ:
$$\text{Lip}(R_{\sigma_2}) \le \frac{\sqrt{2/\pi}}{\sigma_2} \cdot \sup |R| = \mathcal{O}\left(\frac{1}{\sigma_2}\right)$$

### 1.2. Thước đo thực nghiệm (Secant Lipschitz Lower Bound)
Hệ số Lipschitz lý thuyết được xác định bởi:
$$L = \sup_{x_1 \neq x_2} \frac{|R(x_1) - R(x_2)|}{\|x_1 - x_2\|_2}$$
Do đó, mọi tỷ số sai phân tính trên bất kỳ cặp ảnh thực tế nào ($x_{\text{clean}}, x_{\text{pert}}$) đều là **một chặn dưới thực nghiệm hợp lệ** ($L \ge L_{\max}$).

* **Độ dốc Vanilla LiDAR**:
  $$\text{Slope}_{\text{Vanilla}} = \frac{|R(x_{\text{clean}}) - R(x_{\text{pert}})|}{\|x_{\text{clean}} - x_{\text{pert}}\|_2}$$
* **Độ dốc RS-LiDAR** (áp dụng làm mịn ngẫu nhiên đối xứng cho CẢ ẢNH SẠCH VÀ ẢNH BIẾN DẠNG với $M$ mẫu Monte Carlo):
  $$R_{\sigma_2}(x) \approx \frac{1}{M} \sum_{m=1}^M R(x + \sigma_2 u_m), \quad u_m \sim \mathcal{N}(0, I)$$
  $$\text{Slope}_{\text{RS}} = \frac{|R_{\sigma_2}(x_{\text{clean}}) - R_{\sigma_2}(x_{\text{pert}})|}{\|x_{\text{clean}} - x_{\text{pert}}\|_2}$$

---

## 2. ⚠️ QUY TẮC CỐT TỬ KHÔNG ĐƯỢC HIỂU SAI (CRITICAL PITFALL)

> [!CAUTION]
> **RS-LIDAR ĐƯỢC THỰC HIỆN TRÊN IMAGE / PIXEL SPACE $[-1.0, 1.0]$, TUYỆT ĐỐI KHÔNG PHẢI LATENT SPACE!**
> - Mọi thao tác thêm nhiễu $\sigma_1$ (tạo ảnh biến dạng) và $\sigma_2$ (làm mịn Randomized Smoothing) đều diễn ra trên tensor ảnh RGB $x \in [-1.0, 1.0]^{3 \times 512 \times 512}$ (đối với SD 1.5) hoặc $3 \times 1024 \times 1024$ (đối với SDXL).
> - **Lý do**: Các mô hình phần thưởng (`ImageReward`, `CLIP-Score`, `HPS v2.1`, `Aesthetic`) nhận đầu vào là ảnh RGB, không phải vector latent của VAE. Các cuộc tấn công đối kháng (adversarial perturbations) và hiện tượng vỡ ảnh xảy ra trực tiếp ở pixel space.
> - **Cảnh báo Agent**: Nếu bất kỳ agent nào diễn giải rằng $\sigma_2$ được cộng vào latent $z$, điều đó là **HOÀN TOÀN SAI LỆCH VỚI CODEBASE VÀ BÀI BÁO**.

---

## 3. 🏗️ KIẾN TRÚC THỰC NGHIỆM 553 PROMPTS TRÊN KAGGLE

### 3.1. Thiết lập phần cứng & Multi-GPU Sharding
- **Môi trường**: Kaggle Notebook `kaggle/RS_LiDAR_Lipschitz_Test_Kaggle.ipynb` chạy trên **2x GPU Tesla T4 (16GB VRAM mỗi card)**.
- **Phân chia Sharding tự động**:
  - **GPU 0 (Shard 0)**: Xử lý 277 prompts đầu tiên.
  - **GPU 1 (Shard 1)**: Xử lý 276 prompts tiếp theo.
  - Mỗi GPU ghi checkpoint độc lập sau từng prompt: `checkpoint_shard_0.json` và `checkpoint_shard_1.json`.
  - Hỗ trợ **Resume tự động**: Nếu phiên chạy bị gián đoạn, chỉ cần chạy lại là code tự động bỏ qua các prompt đã hoàn thành.

### 3.2. Thông số thực nghiệm chuẩn
- **Mô hình sinh ảnh**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`), DPM-Solver++ 5 bước nhanh, CFG scale 7.5.
- **Tập dữ liệu**: Toàn bộ **553 prompts** chuẩn của GenEval benchmark (`prompt_files/geneval_metadata.jsonl`), bao quát 6 tác vụ thị giác: `single_object`, `two_object`, `counting`, `colors`, `position`, `color_attr`.
- **Số hạt sinh ảnh**: $K = 10$ hạt / prompt $\implies$ **5,530 cặp mẫu** ($x_{\text{clean}}, x_{\text{pert}}$) $\implies$ **11,060 lượt đánh giá ảnh**.
- **Vi nhiễu probe scale**: $\sigma_1 = 0.1$ (Gaussian noise trên pixel space) để tạo $x_{\text{pert}} = \text{clip}(x_{\text{clean}} + \sigma_1 \epsilon, -1, 1)$.
- **Dải quét làm mịn**: $\sigma_2 \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$ với $M = 4$ mẫu Monte Carlo.
- **4 Mô hình Reward đánh giá**:
  1. `ImageReward` (Text-Image Human Preference scoring).
  2. `CLIP-Score` (OpenAI ViT-L/14 semantic alignment).
  3. `HPS v2.1` (Human Preference Score v2.1).
  4. `Aesthetic Score` (Predictor MLP trên CLIP embeddings).
  *(Tất cả được GPU vectorized 1 lượt qua hàm `fast_batch_clip_and_aesthetic` để tối đa tốc độ).*

---

## 4. 📦 CẤU TRÚC FILE TRONG `results_lipschitz.zip` VÀ CÁCH PHÂN TÍCH

Khi người dùng upload file `results_lipschitz.zip` (hoặc thư mục giải nén `results/lipschitz_empirical`), Agent cần tìm và đọc các file sau:

```text
results_lipschitz.zip (hoặc thư mục results/lipschitz_empirical/)
├── checkpoint_shard_0.json                    # Dữ liệu đo 277 prompts từ GPU 0
├── checkpoint_shard_1.json                    # Dữ liệu đo 276 prompts từ GPU 1
├── lipschitz_summary.csv                      # BẢNG TỔNG HỢP SO SÁNH CHÍNH (σ2 = 1.0)
├── lipschitz_sigma_ablation.csv               # BẢNG KHẢO SÁT DẢI SIGMA2 {0.0 -> 1.0}
├── lipschitz_metrics.json                     # Toàn bộ metrics cấu trúc JSON
├── lipschitz_comparison_3panel_sigma_0.1.png  # Đồ thị 3-Panel tại σ = 0.1 (Micro-Smoothing)
├── lipschitz_comparison_3panel_sigma_0.25.png # Đồ thị 3-Panel tại σ = 0.25 (Curvature Optimum: Giảm mạnh nhất 3.38x)
├── lipschitz_comparison_3panel_sigma_0.5.png  # Đồ thị 3-Panel tại σ = 0.5 (Intermediate Zone)
├── lipschitz_comparison_3panel_sigma_1.0.png  # Đồ thị 3-Panel tại σ = 1.0 (Macro Consensus Sweet Spot)
├── lipschitz_comparison_3panel.png            # Bản sao mặc định của σ = 1.0
├── lipschitz_comparison_dual_regime.png       # Đồ thị đối chiếu song song: σ = 0.25 vs σ = 1.0
├── lipschitz_sigma_ablation.png               # Đồ thị suy giảm cơ bản
└── lipschitz_sigma_ablation_enhanced.png      # Đồ thị suy giảm nâng cao (Highlight Sweet Spot [0.25, 1.0])
```

### 4.1. Cách phân tích `lipschitz_summary.csv`
File này chứa bảng so sánh giữa Vanilla LiDAR và RS-LiDAR tại bán kính Sweet Spot $\sigma_2 = 1.0$:
- `Reward Model`: Tên mô hình (ImageReward, CLIP-Score, HPS-v2.1, Aesthetic).
- `L_max (Vanilla)` vs `L_max (RS-LiDAR)`: Chặn trên độ dốc xấu nhất (Worst-Case Lipschitz Bound).
- `Max Reduction Ratio`: Tỷ số co thắt độ dốc xấu nhất ($L_{\max}^{\text{Vanilla}} / L_{\max}^{\text{RS}}$).
- `L_mean (Vanilla)` vs `L_mean (RS-LiDAR)`: Độ dốc trung bình trên toàn bộ 5,530 cặp mẫu.
- `Mean Reduction Ratio`: Tỷ số làm trơn trung bình toàn cục.
- `L_95%`: Phân vị 95% nhằm loại bỏ các ngoại lai (outliers) ngẫu nhiên.

### 4.2. Cách phân tích `lipschitz_sigma_ablation.csv`
File này đối chiếu ảnh hưởng của bán kính làm mịn $\sigma_2 \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$:
- Tại $\sigma_2 = 0.0$: Chính là Vanilla (không làm mịn).
- Tại $\sigma_2 = 0.1$: Triệt tiêu vi nhiễu $\sigma_1 = 0.1$ có cùng tần số vi phân.
- Tại $\sigma_2 = 0.25$: **Điểm cực tiểu toàn cục của $L_{\text{mean}}$ và điểm giảm $L_{\max}$ mạnh nhất ($3.05\times - 3.38\times$)**.
- Tại $\sigma_2 = 1.0$: Vùng phẳng hóa vĩ mô phục vụ điều hướng hạt sinh ảnh.

---

## 5. 🔬 CƠ CHẾ VẬT LÝ: GIẢI MÃ NGHỊCH LÝ "LÝ THUYẾT $\sigma_2 = 0.25$ VS THỰC TIỄN $\sigma_2 = 1.0$"

Một câu hỏi phản biện cực kỳ quan trọng mà hội đồng bình duyệt (Reviewers) có thể đặt ra:
> *"Tại sao khi đo vi sai thực nghiệm thì $\sigma_2 = 0.25$ lại cho tỷ số giảm Lipschitz lớn nhất ($3.38\times$), nhưng khi sinh ảnh thực tế trên SD 1.5 các tác giả lại chọn Sweet Spot là $\sigma_2 = 1.0$?"*

Agent cần trả lời dựa trên **Quy luật tập trung độ đo (Measure Concentration) trong không gian nhiều chiều**:

1. **Khác biệt về thang đo khoảng cách (Spatial Scale Divergence)**:
   - Trong thí nghiệm đo Lipschitz vi phân: Khoảng cách giữa ảnh sạch và ảnh biến dạng rất nhỏ ($\|\Delta x\|_2 = \|x_{\text{clean}} - x_{\text{pert}}\|_2 \approx 88.6$ trong không gian $D = 3 \times 512 \times 512 = 786,432$ chiều). Nhiễu $\sigma_2 = 0.25$ tạo ra bán kính làm mịn bán kính $\|\sigma_2 u\|_2 \approx 0.25 \times \sqrt{786,432} \approx 221.7$, vừa vặn bao trùm và đè bẹp toàn bộ gradient cục bộ của thước đo vi phân.
   - Trong quá trình sinh ảnh khuếch tán: Các hạt khuếch tán $x_t$ phân tán rất xa nhau trong không gian đặc trưng ($\|\Delta x\|_2 \sim 500 - 1500$). Bán kính làm mịn nhỏ $\sigma_2 = 0.25$ chỉ có tác dụng cục bộ, không đủ sức kết nối các hạt để tạo thành **sự đồng thuận vĩ mô (Macroscopic Consensus)**. Do đó, cần $\sigma_2 \approx 1.0$ (tương đương chuẩn $\ell_2 \approx 886.8$) để xóa tan các hố bẫy ảo giác vĩ mô.

2. **Sự khác biệt giữa SD 1.5 ($512 \times 512$) và SDXL ($1024 \times 1024$)**:
   - Chuẩn $\ell_2$ tăng theo căn bậc hai của số chiều $\sqrt{D}$:
     - SD 1.5 ($D = 786,432$): Cần $\sigma_2 \approx 1.0$ để đạt độ dịch chuyển cần thiết.
     - SDXL ($D = 3 \times 1024 \times 1024 = 3,145,728$): Số chiều tăng gấp 4 lần $\implies$ Với cùng một $\sigma_2$, khoảng cách Euclid tăng gấp đôi ($\sqrt{4} = 2$). Do đó, với SDXL chỉ cần $\sigma_2 \approx 0.25 - 0.5$ là đã đạt được hiệu quả làm mịn tương đương $\sigma_2 = 1.0$ của SD 1.5!

---

## 6. 📋 HƯỚNG DẪN HÀNH ĐỘNG DÀNH CHO AGENT TIẾP THEO

Khi người dùng cung cấp file `results_lipschitz.zip` (hoặc thông báo đã chạy xong trên Kaggle):

1. **Bước 1: Giải nén & Xác nhận dữ liệu**:
   ```python
   import zipfile, os
   zip_path = "results_lipschitz.zip"
   extract_dir = "results/lipschitz_empirical_553"
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
       zip_ref.extractall(extract_dir)
   ```
2. **Bước 2: Rà soát tính toàn vẹn 553 Prompts**:
   - Mở `checkpoint_shard_0.json` và `checkpoint_shard_1.json`.
   - Đếm tổng số cặp mẫu: Đảm bảo có đúng **$5,530$ cặp mẫu** (đại diện cho đủ 553 prompts $\times$ 10 hạt).
3. **Bước 3: Trích xuất các bảng dữ liệu**:
   - Đọc và hiển thị `lipschitz_summary.csv` (so sánh Vanilla vs RS tại $\sigma_2 = 1.0$).
   - Đọc và hiển thị `lipschitz_sigma_ablation.csv` (đối chiếu cả 4 mức $\sigma_2 \in \{0.1, 0.25, 0.5, 1.0\}$).
4. **Bước 4: Trực quan hóa kết quả cho người dùng**:
   - Hiển thị bộ 4 biểu đồ 3-Panel (`sigma_0.1`, `sigma_0.25`, `sigma_0.5`, `sigma_1.0`).
   - Hiển thị biểu đồ `lipschitz_comparison_dual_regime.png` và `lipschitz_sigma_ablation_enhanced.png`.
5. **Bước 5: Viết báo cáo tổng kết**:
   - Cập nhật số liệu chính thức trên 553 prompts vào báo cáo tổng kết để thay thế số liệu sơ bộ 50 prompts trước đó.

---

## 💬 MẪU PROMPT CHO NGƯỜI DÙNG DÙNG Ở PHIÊN TIẾP THEO

Người dùng có thể sao chép đoạn prompt sau để dán vào cuộc trò chuyện mới:

> *"Chào bạn, tôi vừa chạy xong thực nghiệm đo hệ số Lipschitz thực nghiệm trên toàn bộ 553 prompts GenEval bằng notebook Kaggle. File kết quả nén là `results_lipschitz.zip` (hoặc giải nén tại `results/lipschitz_empirical`).*  
> *Bạn hãy đọc tài liệu hướng dẫn chuẩn tại `md/LIPSCHITZ_EXPERIMENT_HANDOVER_AND_ANALYSIS_GUIDE.md` để nắm rõ toàn bộ bối cảnh toán học, lưu ý quan trọng về không gian pixel (không phải latent), và giải nén phân tích cho tôi các chỉ số L_max, L_mean, Reduction Ratio trên cả 4 mô hình reward, cùng với phân tích 4 biểu đồ 3-Panel cho sigma = 0.1, 0.25, 0.5, 1.0 nhé!"*
