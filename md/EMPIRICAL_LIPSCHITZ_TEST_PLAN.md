# Kế hoạch Thực nghiệm: Đo Đạc & So Sánh Hệ Số Lipschitz Thực Nghiệm (Vanilla LiDAR vs. RS-LiDAR)

> **Mục tiêu nghiên cứu**: Thiết kế một khung thực nghiệm độc lập chuẩn khoa học để đo đạc và so sánh trực tiếp hệ số Lipschitz thực nghiệm (Empirical Lipschitz Constant) giữa **Vanilla LiDAR** và **RS-LiDAR (Randomized Smoothing)** trên các mô hình Reward thực tế (ImageReward, CLIP-Score, HPSv2, GenEval). Chứng minh rằng phép làm mịn Monte Carlo của RS-LiDAR giúp làm phẳng cảnh quan hàm thưởng (landscape flattening) và giảm mạnh chặn dưới Lipschitz thực nghiệm.

---

## 1. Cơ sở Lý thuyết & Toán học Phép Đo

### 1.1. Định nghĩa Hệ số Lipschitz
Hệ số Lipschitz của một hàm thưởng $R: \mathcal{X} \to \mathbb{R}$ trên không gian ảnh $\mathcal{X} \subset \mathbb{R}^D$ được định nghĩa nghiêm ngặt là:
$$L = \sup_{\mathbf{x}_1 \ne \mathbf{x}_2} \frac{|R(\mathbf{x}_1) - R(\mathbf{x}_2)|}{\|\mathbf{x}_1 - \mathbf{x}_2\|_2}$$

* **Tính chất chặn dưới thực nghiệm (Empirical Lower Bound)**:
  Do không thể duyệt qua vô hạn điểm trong không gian ảnh $D = 512 \times 512 \times 3$, bất kỳ một cặp ảnh cụ thể $(\mathbf{x}_1, \mathbf{x}_2)$ nào có thương số sai phân (Secant Slope):
  $$\text{Slope}(\mathbf{x}_1, \mathbf{x}_2) = \frac{|R(\mathbf{x}_1) - R(\mathbf{x}_2)|}{\|\mathbf{x}_1 - \mathbf{x}_2\|_2}$$
  đều là một **chặn dưới hợp lệ (valid lower bound)** của hệ số Lipschitz:
  $$L \ge \max_{i} \text{Slope}(\mathbf{x}_1^{(i)}, \mathbf{x}_2^{(i)})$$
* **Ưu điểm vượt trội**: Không yêu cầu tính đạo hàm (`requires_grad=True`), tiết kiệm tối đa VRAM, không bị lỗi backward qua các mô hình reward phức tạp hoặc phi khả vi (như mô hình phân loại / detection).

---

### 1.2. Phép đo Vanilla LiDAR vs. RS-LiDAR (Đặc biệt Lưu ý tính cho cả ảnh biến dạng)

Với mỗi cặp ảnh gồm **ảnh sạch** $\mathbf{x}_{\text{clean}}$ và **ảnh bị biến dạng** $\mathbf{x}_{\text{pert}}$:
$$\mathbf{x}_{\text{pert}} = \text{clip}(\mathbf{x}_{\text{clean}} + \sigma_1 \boldsymbol{\epsilon}, -1, 1), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
Khoảng cách không gian pixel giữa hai ảnh là:
$$\Delta x = \|\mathbf{x}_{\text{clean}} - \mathbf{x}_{\text{pert}}\|_2 = \sqrt{\sum_{c,h,w} (x_{\text{clean}} - x_{\text{pert}})^2}$$

#### A. Đối với Vanilla LiDAR (Hàm thưởng gốc $R$):
Đo trực tiếp sai phân giá trị hàm thưởng gốc:
$$\text{Slope}_{\text{vanilla}} = \frac{|R(\mathbf{x}_{\text{clean}}) - R(\mathbf{x}_{\text{pert}})|}{\Delta x}$$

#### B. Đối với RS-LiDAR (Toán tử làm mịn $R_{\sigma_2}$):
Hàm thưởng làm mịn $R_{\sigma_2}(\mathbf{x}) = \mathbb{E}_{\mathbf{u}}[R(\mathbf{x} + \sigma_2 \mathbf{u})]$ được xấp xỉ bằng $M$ mẫu Gaussian Monte Carlo.
> [!IMPORTANT]
> **Yêu cầu cốt lõi**: Để đo đúng hệ số Lipschitz của toán tử làm mịn $R_{\sigma_2}$, phép tính kỳ vọng Monte Carlo $M$ mẫu **BẮT BUỘC PHẢI THỰC HIỆN ĐỘC LẬP CHO CẢ ẢNH SẠCH VÀ ẢNH BIẾN DẠNG**:
> 
> 1. **Giá trị RS trên ảnh sạch $\mathbf{x}_{\text{clean}}$**:
>    $$R_{\sigma_2}(\mathbf{x}_{\text{clean}}) = \frac{1}{M} \sum_{m=1}^M R\left(\text{clip}(\mathbf{x}_{\text{clean}} + \sigma_2 \mathbf{u}_m, -1, 1)\right), \quad \mathbf{u}_m \sim \mathcal{N}(0, \mathbf{I})$$
> 2. **Giá trị RS trên ảnh biến dạng $\mathbf{x}_{\text{pert}}$**:
>    $$R_{\sigma_2}(\mathbf{x}_{\text{pert}}) = \frac{1}{M} \sum_{m=1}^M R\left(\text{clip}(\mathbf{x}_{\text{pert}} + \sigma_2 \mathbf{u}'_m, -1, 1)\right), \quad \mathbf{u}'_m \sim \mathcal{N}(0, \mathbf{I})$$
> 3. **Độ dốc thực nghiệm RS-LiDAR**:
>    $$\text{Slope}_{\text{RS}} = \frac{|R_{\sigma_2}(\mathbf{x}_{\text{clean}}) - R_{\sigma_2}(\mathbf{x}_{\text{pert}})|}{\Delta x}$$

---

## 2. Thiết kế Thực nghiệm Chi tiết

### 2.1. Tập Prompts Phân Tầng (50 Stratified Prompts)
Để kết quả mang tính khái quát cao và đại diện cho toàn bộ benchmark GenEval, 50 prompts được bốc ngẫu nhiên phân tầng (stratified sampling) chia đều cho 6 task:
* **Single Object** (8 prompts): Ví dụ `"a photo of a cat"`, `"a picture of an airplane"`.
* **Two Objects** (9 prompts): Ví dụ `"a dog and a frisbee"`, `"a bear and a backpack"`.
* **Counting** (8 prompts): Ví dụ `"three apples on a wooden table"`, `"two cars parked"`.
* **Colors** (9 prompts): Ví dụ `"a red sports car"`, `"a green cup and a yellow plate"`.
* **Position** (8 prompts): Ví dụ `"a laptop on top of a desk"`, `"a cat under a chair"`.
* **Color Attribution** (8 prompts): Ví dụ `"a white dog and a black cat"`, `"a blue book and a red pen"`.

File lưu trữ: `prompt_files/geneval_50_stratified.jsonl`.

### 2.2. Quy mô Mẫu và Số lượng Đánh giá
* **Số prompts ($N$)**: 50 prompts (mặc định, có thể tùy chỉnh).
* **Số ảnh mỗi prompt ($K$)**: 10 ảnh thường ($x_{\text{clean}}$).
* **Ảnh biến dạng**: Mỗi ảnh thường cộng vi nhiễu $\sigma_1$ $\to$ 10 ảnh biến dạng ($x_{\text{pert}}$).
* **Tổng số cặp mẫu đo độ dốc**: $50 \times 10 = 500$ cặp $(\mathbf{x}_{\text{clean}}, \mathbf{x}_{\text{pert}})$.
* **Tổng số lần đánh giá hàm thưởng**:
  - Vanilla: $500 \times 2 = 1,000$ lần.
  - RS-LiDAR: $500 \times 2 \times M = 1,000 \times M$ lần (với $M=4 \implies 4,000$ lần).
  - Được thực hiện theo batching vector hóa trên GPU để tối ưu tốc độ.

---

## 3. Danh mục Tham số Dòng lệnh CLI (`--...`)

Mọi con số trong thực nghiệm đều được tham số hóa 100% qua CLI flags:

| Tham số CLI | Kiểu dữ liệu | Giá trị Mặc định | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--model_name` | `str` | `runwayml/stable-diffusion-v1-5` | Mô hình sinh ảnh (hỗ trợ SD 1.5, SDXL). |
| `--prompts_file` | `str` | `prompt_files/geneval_50_stratified.jsonl` | Đường dẫn file chứa các prompts kiểm thử. |
| `--num_prompts` | `int` | `50` | Số lượng prompt chạy thực nghiệm. |
| `--num_particles` | `int` | `10` | Số lượng ảnh sinh ra cho mỗi prompt ($K$). |
| `--sigma1` | `float` | `0.1` | Độ lệch chuẩn vi nhiễu tạo ảnh biến dạng $\mathbf{x}_{\text{pert}}$. |
| `--sigma2` | `float` | `1.0` | Bán kính làm mịn của RS-LiDAR Monte Carlo. |
| `--sigma2_list` | `str` | `None` | Cho phép quét danh sách $\sigma_2$ (ví dụ: `"0.25,0.5,1.0"`). |
| `--num_mc_samples` / `-M` | `int` | `4` | Số mẫu Monte Carlo của RS-LiDAR. |
| `--reward_models` | `str` | `"ImageReward,CLIP-Score,HPS"` | Danh sách các reward model đánh giá (hỗ trợ cả GenEval). |
| `--guidance_steps` | `int` | `5` | Số bước DPM-Solver sinh ảnh (tiết kiệm thời gian chạy). |
| `--guidance_scale` | `float` | `7.5` | Classifier-free guidance scale. |
| `--batch_size` | `int` | `8` | Batch size suy luận tính reward trên GPU. |
| `--num_shards` | `int` | `1` | Tổng số worker phân mảnh (dùng khi chạy multi-GPU). |
| `--shard_id` | `int` | `0` | ID phân mảnh của tiến trình hiện tại. |
| `--gpu_id` | `int` | `0` | Chỉ số GPU vật lý gán cho tiến trình. |
| `--output_dir` | `str` | `results/lipschitz_empirical` | Thư mục xuất file CSV, JSON và hình vẽ đồ thị. |
| `--merge_shards` | `action` | `False` | Cờ tự động gộp kết quả từ các shard `shard_0`, `shard_1`. |

---

## 4. Hệ thống Chỉ số (Metrics) & Đồ thị Trực quan hóa

### 4.1. Các chỉ số thống kê trích xuất
Với mỗi reward model, hệ thống tự động tổng hợp:
1. **$L_{\text{max}} = \max_i (\text{Slope}^{(i)})$**: Chặn dưới thực nghiệm nghiêm ngặt của hệ số Lipschitz ($L \ge L_{\text{max}}$).
2. **$L_{\text{mean}} = \frac{1}{N \cdot K} \sum_i \text{Slope}^{(i)}$**: Độ dốc trung bình của cảnh quan (phản ánh độ phẳng tổng thể).
3. **$L_{95\%}$**: Phân vị thứ 95 của phân phối độ dốc (loại bỏ ngoại lai cực đoan).
4. **$L_{\text{median}}$ & $\sigma_{\text{slope}}$**: Trung vị và độ lệch chuẩn của độ dốc.
5. **Lipschitz Reduction Ratio**:
   $$\text{Ratio}_{\text{max}} = \frac{L_{\text{max}}^{\text{vanilla}}}{L_{\text{max}}^{\text{RS}}}, \qquad \text{Ratio}_{\text{mean}} = \frac{L_{\text{mean}}^{\text{vanilla}}}{L_{\text{mean}}^{\text{RS}}}$$
   *(Nếu $\text{Ratio} > 1$, thực nghiệm khẳng định RS-LiDAR làm giảm hệ số Lipschitz)*.

### 4.2. Bộ Đồ thị Trực quan hóa 3 Panel (Publication-Ready)
Hệ thống sử dụng `matplotlib` và `seaborn` xuất ra hình ảnh phân giải cao (`lipschitz_comparison.png`):
* **Panel A (Bar Chart)**: So sánh trực quan $L_{\text{max}}$ và $L_{\text{mean}}$ giữa Vanilla LiDAR và RS-LiDAR (chia theo từng Reward Model).
* **Panel B (KDE Density / Distribution Plot)**: Đường cong mật độ phân phối của $\text{Slope}_{\text{vanilla}}$ vs. $\text{Slope}_{\text{RS}}$, minh họa trực quan sự co cụm độ dốc về phía 0 của RS-LiDAR.
* **Panel C (Scatter Plot)**: Tọa độ từng cặp điểm $(\text{Slope}_{\text{vanilla}}^{(i)}, \text{Slope}_{\text{RS}}^{(i)})$ so với đường chéo tham chiếu $y = x$. (Các điểm nằm dưới đường $y=x$ minh chứng $\text{Slope}_{\text{RS}} < \text{Slope}_{\text{vanilla}}$).

---

## 5. Kiến trúc Triển khai Mã Nguồn

```
RS-LiDAR/Diffusion-LiDAR-Sampling/
├── prompt_files/
│   └── geneval_50_stratified.jsonl            # 50 prompt phân tầng 6 tasks GenEval
├── test_empirical_lipschitz.py                # Script CLI độc lập chạy kiểm thử Lipschitz
├── kaggle/
│   └── RS_LiDAR_Lipschitz_Test_Kaggle.ipynb   # Notebook chạy song song 2 GPU T4 trên Kaggle
└── results/lipschitz_empirical/               # Thư mục lưu kết quả
    ├── lipschitz_raw_pairs.csv                # Bảng chi tiết 500 cặp ảnh (delta_x, delta_R, slope)
    ├── lipschitz_summary.csv                  # Bảng tổng hợp L_max, L_mean, Reduction Ratio
    ├── lipschitz_metrics.json                 # JSON lưu chỉ số định lượng
    └── lipschitz_comparison.png               # Đồ thị 3 panel chuẩn công bố
```

### 5.1. Thiết kế Script CLI (`test_empirical_lipschitz.py`)
* Kế thừa cấu trúc xử lý nhẹ từ `test_lidar_weaknesses.py`.
* Tối ưu hóa bộ nhớ: Vòng lặp giải phóng latent sau khi chuyển đổi sang PIL/Tensor ảnh, xóa bộ đệm CUDA định kỳ (`torch.cuda.empty_cache()`).
* Đánh giá song song batch: Đưa toàn bộ $M$ mẫu nhiễu Monte Carlo vào tensor kích thước `[M, 3, H, W]` để tính reward trong 1 lượt forward duy nhất.

### 5.2. Thiết kế Notebook Kaggle 2x GPU T4 (`RS_LiDAR_Lipschitz_Test_Kaggle.ipynb`)
* **Kiểm tra phần cứng**: Tự động nhận diện 2 GPU Tesla T4 (16GB mỗi card).
* **Phân chia Shard (Multi-GPU Sharding)**:
  - GPU 0 xử lý Shard 0 (25 prompts đầu).
  - GPU 1 xử lý Shard 1 (25 prompts sau).
* **Khởi chạy nền song song**: Sử dụng 2 luồng `subprocess.Popen` kết hợp `threading` để stream đồng thời log `[GPU 0]` và `[GPU 1]` theo thời gian thực trong output cell của Jupyter Notebook.
* **Tự động gộp & Render**: Sau khi 2 tiến trình hoàn tất, cell cuối tự động kích hoạt `--merge_shards`, in bảng Markdown tóm tắt và hiển thị đồ thị 3-panel.

---

## 6. Kế hoạch Thực hiện Từng Bước (Roadmap)

1. [x] **Bước 1**: Viết tài liệu Kế hoạch chi tiết vào file [`Diffusion-LiDAR-Sampling/md/EMPIRICAL_LIPSCHITZ_TEST_PLAN.md`](file:///c:/Users/Admin/Desktop/Deep%20Learning%20Research/RS-LiDAR/Diffusion-LiDAR-Sampling/md/EMPIRICAL_LIPSCHITZ_TEST_PLAN.md).
2. [ ] **Bước 2**: Tạo file 50 prompts phân tầng chuẩn GenEval: `prompt_files/geneval_50_stratified.jsonl`.
3. [ ] **Bước 3**: Cài đặt Script Python CLI `test_empirical_lipschitz.py` hoàn chỉnh mọi cờ tham số, logic tính toán đối xứng cho cả ảnh sạch và ảnh biến dạng, xuất file CSV/JSON và đồ thị Matplotlib.
4. [ ] **Bước 4**: Xây dựng Notebook Kaggle `kaggle/RS_LiDAR_Lipschitz_Test_Kaggle.ipynb` chạy 2 GPU song song.
5. [ ] **Bước 5**: Kiểm tra tính toàn vẹn (Verification):
   - Chạy `python -m py_compile test_empirical_lipschitz.py`.
   - Chạy `python check_notebook_syntax.py kaggle/RS_LiDAR_Lipschitz_Test_Kaggle.ipynb`.
   - Chạy dry-run test nhanh trên CPU/GPU với 1 prompt, 2 particles để xác nhận không có bug runtime.
6. [ ] **Bước 6**: Git commit và push toàn bộ lên repo bằng 1 dòng lệnh theo quy định.
