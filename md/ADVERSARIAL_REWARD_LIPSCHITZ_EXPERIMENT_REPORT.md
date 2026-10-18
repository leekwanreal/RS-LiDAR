# [DRAFT TEMPLATE] Báo Cáo Thực Nghiệm: Khảo Sát Hệ Số Lipschitz, Ức Chế Bẫy Đối Nghịch & Bảo Toàn Ngữ Nghĩa Trên Hàm Adversarial Reward (ResNet-50)

> ⚠️ **LƯU Ý QUAN TRỌNG (DRAFT TEMPLATE)**:  
> Báo cáo này hiện tại là **Khung mẫu cấu trúc (Report Template)** được xây dựng theo kế hoạch khoa học [Exp 2 Plan](NEW_REWARD_EXPERIMENT_PLAN_EXP2_ADVERSARIAL_REWARD_DETAILED_PLAN.md).  
> Toàn bộ các bảng số liệu bên dưới đang để ở trạng thái **`[Chờ số liệu thực nghiệm / TBD]`**. Các con số chính thức sẽ được trích xuất trực tiếp từ các file kết quả thực tế (`adversarial_lipschitz_benchmark_table.md`, `summary_adversarial_metrics.json`, `adversarial_cohen_certificates.md`) sau khi thí nghiệm chạy xong trên Google Colab.

---

> **Dự án**: RS-LiDAR — Randomized Smoothing Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models  
> **Căn cứ lý thuyết**: [Theorem 1 (Lipschitz Smoothing)](../theorems/theorem_1_lipschitz_smoothing.md), [Theorem 3 (Variance Reduction)](../theorems/theorem_3_variance_reduction.md), Cohen et al. (ICML 2019 - Certified Adversarial Robustness via Randomized Smoothing), [Kế hoạch thử nghiệm chi tiết](NEW_REWARD_EXPERIMENT_PLAN_EXP2_ADVERSARIAL_REWARD_DETAILED_PLAN.md)  
> **Quy mô dữ liệu**: 50 prompt đại diện 10 ImageNet Super-Categories, $K=10$ hạt/prompt = **500 cặp mẫu ảnh**, khảo sát qua $M \in \{1, 2, 4, 8\}$, $\sigma_2 \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$, $\sigma_1 \in \{0.001 \dots 0.05\}$ với Common Random Numbers (CRN).  
> **Mã nguồn thực thi**: [`test_adversarial_reward.py`](../test_adversarial_reward.py), [`colab/RS_LiDAR_Adversarial_Reward_Experiment_Colab.ipynb`](../colab/RS_LiDAR_Adversarial_Reward_Experiment_Colab.ipynb)  

---

## 1. TÓM TẮT KẾT QUẢ CỐT LÕI (EXECUTIVE SUMMARY)

Trong khi Thí nghiệm 1 (Threshold Reward - OWL-ViT) xử lý bệnh lý **gián đoạn hoàn toàn** ($L = \infty$, 89.2% mẫu bị mù gradient), Thí nghiệm 2 (Adversarial Reward - ResNet-50) nhắm vào bệnh lý: **Hàm liên tục khả vi nhưng có hằng số Lipschitz cục bộ khổng lồ và chứa đầy gai nhọn đối nghịch ($L_{\text{local}} \gg 1$)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             ĐỐI CHIẾU HAI BỆNH LÝ TRỌNG YẾU GIỮA HAI THÍ NGHIỆM                      │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ THÍ NGHIỆM 1: THRESHOLD REWARD         │ THÍ NGHIỆM 2: ADVERSARIAL REWARD              │
│ • Bệnh lý: L = ∞ (Hàm gián đoạn nhị phân)│ • Bệnh lý: L_local >> 1 (Liên tục, siêu gồ ghề) │
│ • Vấn đề: Gradient Starvation (89.2% = 0)│ • Vấn đề: Adversarial Trap & Misdirection     │
│ • Cơ chế RS: "Ngọn hải đăng" tạo gradient│ • Cơ chế RS: "San phẳng gai nhọn", nén ||∇R|| │
│ • Bảo đảm: Chuyển Rời rạc → Liên tục   │ • Bảo đảm: Cohen Certified Robustness Radius  │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```

### Các Luận Điểm Khoa Học Chờ Xác Nhận Bằng Số Liệu Thực Nghiệm:

1. **Ức Chế Độ Dốc Cục Bộ & Co Rút Hằng Số Lipschitz**:
   - Khi $\sigma_1 \to 0$, khoảng cách vi phân $\|\Delta x\|_2 \to 0$, khảo sát hiện tượng bùng nổ đạo hàm của Vanilla ResNet-50 do va chạm gai nhọn đối kháng tần số cao.
   - RS-LiDAR ($\sigma_2 = 0.5, M = 4$) co rút hằng số Lipschitz $L_{\text{mean}}$ và $L_{\max}$, kiểm chứng chặn trên theo **Theorem 1**.

2. **Dập Tắt Gradient Spikes & Bó Hẹp Chuẩn $\|\nabla_x R(x)\|_2$**:
   - So sánh chuẩn gradient $\|\nabla_x R(x)\|_2$ giữa Autograd raw của Vanilla và Stein's score function estimator của RS-LiDAR.

3. **Bán Kính Bền Vững Được Chứng Thực Toán Học (Cohen Certified $\ell_2$ Radius)**:
   - Kiểm chứng định lý Cohen et al. (ICML 2019): Đo đạc xác suất $p_A$ và tính bán kính an toàn được chứng nhận toán học $R_{\text{cert}} = \sigma_2 \Phi^{-1}(\underline{p_A})$ (Clopper-Pearson 99.9%).

4. **Trực Quan Hóa Saliency & Tiêu Điểm Ngữ Nghĩa (Saliency Focus Ratio)**:
   - Đo tỷ lệ năng lượng gradient tập trung tại vùng đối tượng chính (Saliency Focus Ratio) giữa Vanilla (phân tán nhiễu nền) và RS-LiDAR (tập trung cấu trúc ngữ nghĩa).

5. **Độ Tương Quan Ngữ Nghĩa Thực Chất (Logit Gain vs. CLIP Score Alignment)**:
   - Đo tương quan Pearson $r$ và Spearman $\rho$ giữa reward logit và CLIP Score để kiểm chứng: Vanilla có bị bẫy đối kháng (logit cao ảo nhưng CLIP score thấp) hay không, và RS-LiDAR có cải thiện tương quan ngữ nghĩa không.

---

## 2. CƠ SỞ LÝ THUYẾT & KẾT NỐI TOÁN HỌC

### 2.1. Bản Chất Bẫy Đối Nghịch (Adversarial Fragility of Deep Classifiers)

Hàm reward dựa trên logit của ResNet-50 cho lớp mục tiêu $c^*$:
$$R(x) = f_{c^*}(x)$$

Dù khả vi ở mọi nơi, hàm $R(x)$ tồn tại các hướng đạo hàm cực dốc $\nabla_x R(x)$ xuất phát từ các trọng số tích chập tần số cao:
$$L_{\text{local}}(x) = \lim_{\|\Delta x\|_2 \to 0} \frac{|R(x + \Delta x) - R(x)|}{\|\Delta x\|_2} = \|\nabla_x R(x)\|_2$$

Khi áp dụng guidance thông thường (Vanilla LiDAR / P2P):
$$x_{t-1} \leftarrow x_{t-1} + \eta \nabla_{x_t} R(x_t)$$
Quá trình tối ưu có thể bị hút vào các cực đại địa phương giả (Adversarial Traps): logit tăng vọt nhưng hình ảnh bị suy giảm chất lượng thị giác.

### 2.2. Cơ Chế Làm Mịn Gaussian & Theorem 1

RS-LiDAR định nghĩa hàm reward làm mịn kỳ vọng:
$$\tilde{R}_{\sigma_2}(x) = \mathbb{E}_{u \sim \mathcal{N}(0, I)} \left[ R(x + \sigma_2 u) \right]$$

Theo **Theorem 1 (Lipschitz Smoothing Bound)**:
$$\|\nabla \tilde{R}_{\sigma_2}(x)\|_2 \le \frac{B \sqrt{d}}{\sigma_2}$$

Nhờ tích chập Gaussian, các gai nhọn tần số cao bị tích phân triệt tiêu, khống chế độ dốc cục bộ trong dải an toàn.

### 2.3. Cầu Nối Định Lý Bền Vững Cohen et al. (2019)

Cohen et al. chứng minh: Nếu bộ phân loại làm mịn $g(x) = \arg\max_c \mathbb{E}_{\epsilon}[f_c(x + \epsilon)]$ dự đoán lớp mục tiêu $c^*$ với xác suất $p_A > 0.5$ dưới nhiễu $\epsilon \sim \mathcal{N}(0, \sigma_2^2 I)$, thì dự đoán này được chứng nhận bất biến trong quả cầu bán kính $\ell_2$:
$$R_{\text{cert}} = \sigma_2 \cdot \Phi^{-1}(\underline{p_A})$$
trong đó $\Phi^{-1}$ là hàm phân vị chuẩn tắc và $\underline{p_A}$ là cận dưới khoảng tin cậy Clopper-Pearson 99.9%.

---

## 3. THIẾT KẾ THỰC NGHIỆM CHI TIẾT

| Thành phần | Cấu hình | Mục đích khoa học |
|:---|:---:|:---|
| **Diffusion Backbone** | Stable Diffusion v1.5 | Tạo ảnh $512 \times 512$, đồng nhất với Exp 1 |
| **Reward Model** | ResNet-50 (ImageNet Pretrained) | Đại diện cho classifier chuẩn, nhạy cảm đối nghịch |
| **Tập Prompts** | 50 prompts (5 prompts × 10 classes) | Bao phủ 10 super-categories WordNet, triệt tiêu cherry-picking |
| **Số lượng mẫu** | $K = 10$ hạt $\to$ **500 cặp mẫu** | Đảm bảo ý nghĩa thống kê cao ($p$-value, phân phối KDE) |
| **Dải vi nhiễu $\sigma_1$** | Quét $\{0.001, 0.005, 0.01, 0.02, 0.05\}$ | Khảo sát bùng nổ đạo hàm vi phân khi $\sigma_1 \to 0$ |
| **Bán kính làm mịn $\sigma_2$** | Quét $\{0.0, 0.1, 0.25, 0.5, 1.0\}$ | Khảo sát tỷ lệ co rút Lipschitz và độ mượt landscape |
| **Số mẫu Monte Carlo $M$** | Quét $\{1, 2, 4, 8\}$ | Khảo sát hội tụ kỳ vọng theo Theorem 3 |
| **Khử phương sai CRN** | Bắt buộc (`True`) | Dùng chung chuỗi nhiễu Monte Carlo giữa $x_{\text{clean}}$ và $x_{\text{pert}}$ |
| **Semantic Validator** | OpenAI CLIP ViT-B/32 | Đánh giá độc lập mức độ tương thích ngữ nghĩa văn bản - ảnh |

---

## 4. KẾT QUẢ ĐỊNH LƯỢNG (CHỜ ĐIỀN TỪ OUTPUT COLAB)

### 4.1. Bảng 1: Benchmark Lipschitz, Gradient Norm & Semantic Correlation ($\sigma_2 = 0.5$)
*(Nguồn trích xuất: `adversarial_lipschitz_benchmark_table.md` và `summary_adversarial_metrics.json`)*

| Phương Pháp / Cấu Hình | $L_{\text{mean}}$ | $L_{\max}$ | $L_{95\%}$ | $\|\nabla R\|_{\text{mean}}$ | Tỷ Lệ Co Rút | CLIP Score | Logit-CLIP Corr ($r$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vanilla ResNet-50** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | 1.00× (Baseline) | `[TBD]` | `[TBD]` |
| **RS-LiDAR ($M=1$)** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **RS-LiDAR ($M=2$)** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **RS-LiDAR ($M=4$)** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **RS-LiDAR ($M=8$)** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 4.2. Bảng 2: Phân Tích Độ Bền Vững Trên 10 Super-Categories ImageNet
*(Nguồn trích xuất: `adversarial_class_breakdown_table.md`)*

| Nhóm Danh Mục (Super-Category) | Lớp Mục Tiêu Đại Diện | Vanilla $L_{\text{mean}}$ | RS-LiDAR ($M=4$) | Tỷ Lệ Co Rút |
|:---|:---:|:---:|:---:|:---:|
| **Canine (Chó)** | Golden Retriever (207) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Feline (Mèo)** | Tabby Cat (281) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Ursidae (Thú lớn hoang dã)** | Giant Panda (388) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Avian (Loài chim)** | Flamingo (130) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Land Vehicle (Xe cơ giới)** | Sports Car (817) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Aviation (Hàng không)** | Airliner (404) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Flora (Thực vật)** | Daisy (985) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Fruit / Food (Thực phẩm)** | Strawberry (949) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Electronics (Công nghệ)** | Laptop (620) | `[TBD]` | `[TBD]` | `[TBD]` |
| **Landscape (Địa chất tự nhiên)** | Volcano (980) | `[TBD]` | `[TBD]` | `[TBD]` |

### 4.3. Bảng 3: Khảo Sát Bán Kính Vi Nhiễu $\sigma_1 \to 0$ (Spike Explosion vs Bounded)
*(Nguồn trích xuất: `adversarial_table_sigma1_sweep.png`)*

| Bán Kính Probe $\sigma_1$ | Khoảng Cách $\|\Delta x\|_2$ | Vanilla $L_{\max} \to$ RS | Tỷ Lệ Co Rút Cực Đại | Vanilla $L_{\text{mean}} \to$ RS | Tỷ Lệ Co Rút Trung Bình |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.050 (Primary)** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **0.020** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **0.010** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **0.005** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **0.001** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 5. PHÂN TÍCH HÌNH ẢNH & CASE STUDIES (CHỜ ĐIỀN TỪ OUTPUT COLAB)

### 5.1. Saliency Heatmaps: Nhiễu Đối Kháng vs Đặc Trưng Ngữ Nghĩa Vững Chắc
*(Ảnh tương ứng: `adversarial_visual_saliency_grid.png`)*
- `[Chờ phân tích Focus Ratio từ thực nghiệm thực tế trên Colab]`

### 5.2. 1D Landscape Slicing: Dập Tắt Hoàn Toàn Gai Nhọn Cực Trị
*(Ảnh tương ứng: `adversarial_1d_landscape_slicing.png`)*
- `[Chờ đồ thị thực nghiệm 200 điểm nội suy từ Colab]`

### 5.3. Bán Kính Chứng Nhận Cohen (Certified Robustness Radius)
*(Nguồn: `adversarial_cohen_certificates.md` & `adversarial_cohen_certified_radius.png`)*
- `[Chờ số liệu p_A và R_cert từ case studies]`

### 5.4. Tương Quan Ngữ Nghĩa (Logit Gain vs. CLIP Score)
*(Ảnh tương ứng: `adversarial_clip_logit_correlation.png`)*
- `[Chờ giá trị Pearson r, Spearman rho và p-value thực tế từ 500 mẫu]`

---

## 6. KẾT LUẬN & ĐÓNG GÓP CHO BÀI BÁO

Cùng với Thí nghiệm 1 (Threshold Reward), Thí nghiệm 2 hoàn thiện **"Bộ Tam Giác Vàng"** chứng minh sức mạnh của RS-LiDAR:

1. **Về mặt Lý Thuyết**: Theorem 1 chứng minh RS biến đổi mọi hàm reward thành $C^\infty$-Lipschitz bị chặn bởi $\frac{B\sqrt{d}}{\sigma_2}$.
2. **Về Hàm Gián Đoạn (Exp 1 - Threshold)**: Chứng minh RS giải quyết triệt để bệnh lý **Gradient Starvation**, "khai sinh" gradient từ con số 0 và giảm $L_{\text{boundary}}$ tới 19.72×.
3. **Về Hàm Đối Nghịch (Exp 2 - Adversarial)**: Chứng minh RS giải quyết bệnh lý **Adversarial Traps**, san phẳng gai nhọn cục bộ, bảo vệ đặc trưng ngữ nghĩa bền vững và mang lại bán kính an toàn được chứng thực theo Cohen et al. (2019).
