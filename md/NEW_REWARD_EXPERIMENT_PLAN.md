# Kế hoạch thử nghiệm: Các Hàm Reward Bất Lợi (Non-Lipschitz & Spiky) - Chi Tiết Triển Khai

Để làm nổi bật tác dụng của **Randomized Smoothing (RS)** trong RS-LiDAR, việc thử nghiệm trên những hàm reward có tính "độc hại" (cực kỳ gồ ghề, hằng số Lipschitz cực lớn, hoặc hàm bước nhảy gián đoạn) là một hướng đi mang tính chất định hình (defining contribution) cho bài báo.

Dưới đây là chi tiết cụ thể (từ logic hàm, công cụ, đến phương pháp đánh giá) cho 3 hướng thử nghiệm đột phá, kết hợp giữa tính trực quan và tính chặt chẽ về toán học:

---

## 1. Hàm "Bước nhảy" (Binary / Threshold Reward) - *Không Lipschitz (Lipschitz = ∞)*

Hàm bước nhảy là minh chứng kinh điển nhất cho sức mạnh của RS. Đạo hàm của chúng bằng 0 ở mọi nơi và bằng vô cực ở điểm giao, khiến các phương pháp Vanilla bị "mù" gradient. RS sẽ biến hàm bậc thang thành hàm mượt (như Sigmoid), tạo ra sườn dốc để tối ưu. Thay vì chỉ kiểm tra sinh ảnh bằng mắt, chúng ta sẽ áp dụng **pipeline đo Lipschitz thực nghiệm** để chứng minh toán học sự thay đổi này.

### Chi tiết triển khai với OWL-ViT (Zero-shot Object Counting):
*   **Mục đích cốt lõi:** 
    1. Đo đạc trực tiếp hệ số Lipschitz thực nghiệm (Empirical Lipschitz Constant) của hàm Threshold Reward để chứng minh: Vanilla Reward không có tính Lipschitz (độ dốc cực trị tiến tới $\infty$), trong khi RS-LiDAR biến đổi nó thành hàm Lipschitz-continuous (độ dốc bị chặn).
    2. Vẽ biểu đồ đối chiếu phân phối phần thưởng (Reward Distribution) và tỷ lệ thành công (Success Rate) sinh đúng số lượng vật thể.
*   **Cài đặt Model:** 
    * Sử dụng `google/owlvit-base-patch32` (`transformers`) làm Zero-shot Detector để chấm điểm.
*   **Logic Hàm Reward (Pseudocode):**
    ```python
    def binary_counting_reward(image, text_prompt="apple", target_count=3):
        # 1. Truyền ảnh và text qua OWL-ViT
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # 2. Đếm số lượng bounding box
        probs = outputs.logits[0].sigmoid().max(dim=-1).values
        count = (probs > 0.1).sum().item()
        
        # 3. Hàm bước nhảy (Binary)
        return 1.0 if count == target_count else 0.0
    ```
*   **Phương pháp đo đạc & Tiêu chí đánh giá (Dựa theo pipeline `test_empirical_lipschitz.py`):**
    *   **Thực nghiệm Đo Lipschitz:** 
        * Tạo cặp ảnh sạch ($x$) và ảnh vi nhiễu ($x_{pert} = x + \sigma_1$). 
        * **Vanilla LiDAR**: Tính sai phân $\frac{|R(x) - R(x_{pert})|}{\Delta x}$. Ở hầu hết không gian, độ dốc $= 0$. Nhưng tại ranh giới quyết định (chuyển từ đếm sai sang đếm đúng), độ dốc bắn vọt lên cực đại ($L \to \infty$).
        * **RS-LiDAR**: Tính toán tử làm mịn Monte Carlo $\tilde{R}_{\sigma_2}(x)$. Kỳ vọng $\mathbb{E}[R(x+\epsilon)]$ sẽ san phẳng bước nhảy, độ dốc thực nghiệm bị chặn ở mức an toàn.
    *   **Thực nghiệm Tỷ lệ thành công & Phân phối Reward:** Chạy thuật toán lấy mẫu trên tập 50 prompts GenEval (Counting tasks).
    *   **Output Metrics & Plots cần có:**
        1. **Bảng tóm tắt Lipschitz**: Cung cấp $L_{max}$ và $L_{mean}$ cho Vanilla và RS-LiDAR.
        2. **KDE Density Plot**: Biểu đồ phân phối độ dốc (Slope Distribution) chứng minh RS-LiDAR ép độ dốc về dải hẹp, triệt tiêu các "spikes" vô cực.
        3. **Reward Landscape / Distribution Plot**: Biểu đồ Histogram so sánh Reward trả về. Vanilla chỉ có giá trị `0` và `1` (Rời rạc). RS-LiDAR trải đều liên tục từ `0.0` đến `1.0` (Mượt).
        4. **Success Rate Plot**: Biểu đồ cột so sánh tỷ lệ đếm đúng (Exact Match) giữa Vanilla và RS-LiDAR.

---

## 2. Hàm Dễ Bị Tấn Công Đối Nghịch (Adversarially Vulnerable Models) - *Hằng số Lipschitz cục bộ khổng lồ*

Sử dụng Randomized Smoothing để phòng thủ tấn công đối nghịch đã rất nổi tiếng (Cohen et al., 2019). Kết nối lý thuyết này vào RS-LiDAR sẽ là một "killer experiment".

### Chi tiết triển khai với Vanilla ResNet50:
*   **Mục đích:** Chứng minh Vanilla LiDAR hoạt động như một thuật toán tấn công (Adversarial Attacker) tạo ra nhiễu rác, trong khi RS-LiDAR kế thừa tính "Robustness" (bền vững) từ thuật toán gốc của Cohen để sinh ra ảnh đúng Semantic.
*   **Cài đặt Model:**
    * Sử dụng `torchvision.models.resnet50(pretrained=True)`. Cần nhớ hàm chuẩn hóa (Normalize) của ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
*   **Logic Hàm Reward (Pseudocode):**
    ```python
    def resnet_adversarial_reward(image, target_class_id=207): # 207 = Golden Retriever
        image_norm = imagenet_normalize(image)
        logits = resnet50(image_norm)
        # Lấy logit của class mục tiêu làm Reward để kích hoạt tối đa
        return logits[0, target_class_id]
    ```
*   **Kỳ vọng & Tiêu chí đánh giá:**
    *   **Vanilla LiDAR:** Tối ưu hóa "mù quáng" vào các điểm mù của mạng (spikes). Ảnh sinh ra sẽ đầy nhiễu hột, dải màu kỳ lạ (adversarial artifacts), có thể không ra hình con chó nhưng logit trả về lại cao chót vót.
    *   **RS-LiDAR:** Bộ lọc Gaussian của RS sẽ "san phẳng" các điểm mù hẹp của ResNet50. Để đạt logit cao trong môi trường bị nhiễu, ảnh bắt buộc phải có các "Robust Features" (lông, tai, mõm chó). Ảnh sinh ra sẽ sắc nét và chuẩn xác.
    *   **Metric:** So sánh bằng mắt thường (Visual Comparison - rất ấn tượng để đưa vào bài). Đo đạc hằng số Lipschitz cục bộ (Norm của Gradient Reward) trong quá trình sampling - Vanilla sẽ cao vọt, RS sẽ rất thấp.

---

## 3. Hàm Nhiễu Nhân Tạo và Toán Học - *Minh chứng tính đúng đắn Giải Tích*

Nhóm thử nghiệm này giúp củng cố sức mạnh của bài báo dưới góc nhìn toán học chặt chẽ, loại bỏ hoàn toàn sự nghi ngờ về yếu tố "ăn may" của mô hình.

### Thí nghiệm 3A: Nhiễu Tần Số Cao (Sinusoidal Perturbed Reward)
*   **Mục đích:** Chứng minh bằng định lý giải tích: Tích chập (convolution) của phân bố Gaussian (RS) và sóng Sine sẽ dập tắt hoàn toàn nhiễu sóng Sine, chỉ để lại hàm mục tiêu nguyên bản.
*   **Cơ sở Toán học (LaTeX):**
    Giả sử hàm Reward bị nhiễu có dạng: 
    $$ R_{spiky}(x) = R_{base}(x) + \gamma \sin(\omega x) $$
    Trong đó $R_{base}$ là hàm mục tiêu thật sự (VD: CLIP Score), $\gamma$ là biên độ nhiễu, và $\omega$ là tần số cực lớn (gây ra nhiễu gai).
    Khi áp dụng Randomized Smoothing với nhiễu Gaussian $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, hàm Reward được làm mượt trở thành kỳ vọng:
    $$ \tilde{R}(x) = \mathbb{E}_{\epsilon}[R_{spiky}(x + \epsilon)] = \mathbb{E}[R_{base}(x + \epsilon)] + \gamma \mathbb{E}[\sin(\omega (x + \epsilon))] $$
    Theo tính chất của biến đổi Fourier đối với phân bố Gaussian, số hạng nhiễu thứ hai sẽ bị triệt tiêu theo hàm mũ:
    $$ \mathbb{E}[\sin(\omega (x + \epsilon))] = \sin(\omega x) \cdot e^{-\frac{\sigma^2 \omega^2}{2}} $$
    Do đó:
    $$ \tilde{R}(x) \approx \tilde{R}_{base}(x) + \gamma \sin(\omega x) e^{-\frac{\sigma^2 \omega^2}{2}} $$
    Vì $\omega$ lớn, thành phần $e^{-\frac{\sigma^2 \omega^2}{2}}$ tiến rất nhanh về 0. Điều này chứng minh RS-LiDAR sẽ "nhìn xuyên" qua nhiễu Sine và tối ưu hóa chính xác theo $R_{base}$.

*   **Logic Hàm Reward:**
    ```python
    def sinusoidal_reward(image, text_prompt):
        # Tính Base Reward (ví dụ: CLIP Score gốc vốn khá mượt)
        base_r = clip_score_fn(image, text_prompt) 
        
        # Thêm sóng Sine tần số cực cao vào Base Reward
        gamma = 0.5   # Biên độ nhiễu
        omega = 50.0  # Tần số (rất gồ ghề)
        feature_norm = torch.norm(extract_features(image))
        
        spiky_r = base_r + gamma * torch.sin(omega * feature_norm)
        return spiky_r
    ```
*   **Kỳ vọng & Đánh giá:** 
    *   **Vanilla LiDAR:** Bị đánh lừa bởi hàm Sine, gradient bị kẹt tại các đỉnh (local maxima) của sóng.
    *   **RS-LiDAR:** Tự động dập tắt nhiễu Sine nhờ bộ lọc Gaussian, đi thẳng lên đỉnh của $R_{base}$.
    *   **Metric:** Biểu đồ 2D về quỹ đạo tối ưu (Optimization Trajectory) so sánh giữa Vanilla và RS, cho thấy rõ đường đi zig-zag của Vanilla và đường thẳng mượt của RS.

### Thí nghiệm 3B: Cảnh Quan "Lỗ Golf" (Narrow Funnel)
*   **Mục đích:** Khảo sát khả năng tìm kiếm "mò kim đáy biển" bằng cách sử dụng nhiễu Gaussian để mở rộng phạm vi hấp dẫn (basin of attraction).
*   **Cơ sở Toán học (LaTeX):**
    Cho hàm $R(x)$ là hàm chỉ báo (indicator function) hình trụ với bán kính $r$ rất nhỏ quanh mục tiêu $x^*$:
    $$ R(x) = \begin{cases} C, & \text{nếu } \|x - x^*\| \leq r \\ 0, & \text{nếu } \|x - x^*\| > r \end{cases} $$
    Với các thuật toán tìm kiếm cục bộ (như Vanilla LiDAR), khi $x$ nằm ngoài bán kính $r$, gradient là $0$, hệ thống không có hướng đi.
    Với RS, hàm mục tiêu được làm mượt sẽ có dạng xấp xỉ tỷ lệ với hàm mật độ xác suất (PDF) của phân phối Gaussian $\mathcal{N}(x^*, \sigma^2 I)$:
    $$ \tilde{R}(x) = \mathbb{E}_{\epsilon}[R(x+\epsilon)] \approx C \cdot V(r) \cdot \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{\|x - x^*\|^2}{2\sigma^2}\right) $$
    *(với $V(r)$ là thể tích của khối cầu bán kính $r$).*
    Kết quả là hàm $\tilde{R}(x)$ cung cấp một gradient mềm và liên tục $\nabla \tilde{R}(x) \propto -(x - x^*)$, trực tiếp chỉ hướng về phía mục tiêu $x^*$ từ mọi vị trí trong không gian!

*   **Logic Hàm Reward:**
    ```python
    def golf_hole_reward(image, text_prompt, epsilon=0.05):
        img_embed = get_image_embedding(image)
        text_embed = get_text_embedding(text_prompt)
        
        distance = MSE(img_embed, text_embed)
        if distance < epsilon:
            return 100.0  # Lỗ golf
        else:
            return 0.0    # Vùng phẳng lì
    ```
*   **Kỳ vọng & Đánh giá:** 
    *   Vanilla LiDAR vĩnh viễn kẹt ở $R=0$. 
    *   RS cộng nhiễu $\sigma$ lớn tương đương với việc trải rộng xác suất, giúp mẫu lookahead phát hiện được hướng đi tới mục tiêu.

---
**💡 Lời khuyên thứ tự code thực tế:**
1. Code **Thí nghiệm 2 (ResNet50)** trước. Lý do: Hàm cực kỳ dễ viết (chỉ cần load ResNet50), không cần GPU mạnh, và kết quả visual sinh ra (ảnh nhiễu vs ảnh đẹp) sẽ cực kỳ thuyết phục Mentor.
2. Code **Thí nghiệm 3A (Sinusoidal)** để lấy dữ liệu vẽ Chart (biểu đồ) chứng minh toán học.
3. Code **Thí nghiệm 1 (OWL-ViT)** sau cùng nếu có thời gian, do cần setup mô hình detection và viết hàm parse output.
