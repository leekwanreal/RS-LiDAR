# Kế Hoạch Chi Tiết: Thí Nghiệm 2 — Hàm Dễ Bị Tấn Công Đối Nghịch (Adversarially Vulnerable Models)

> **Dự án**: RS-LiDAR  
> **Thí nghiệm**: #2 trong [NEW_REWARD_EXPERIMENT_PLAN.md](file:///c:/HUST/Projects/Lab/RS-LiDAR/md/NEW_REWARD_EXPERIMENT_PLAN.md)  
> **Tiền đề**: Thí nghiệm 1 (Threshold Reward) đã hoàn thành — xem [báo cáo](file:///c:/HUST/Projects/Lab/RS-LiDAR/md/THRESHOLD_REWARD_LIPSCHITZ_EXPERIMENT_REPORT.md)  
> **Lý thuyết nền**: [Theorem 1 (Gradient Bound)](file:///c:/HUST/Projects/Lab/RS-LiDAR/theorems/theorem_1_gradient_bound.md), Cohen et al. 2019 (Randomized Smoothing for Certified Adversarial Robustness)

---

## 0. ĐỘNG LỰC KHOA HỌC — TẠI SAO THÍ NGHIỆM NÀY LÀ "KILLER EXPERIMENT"

### 0.1. Bối cảnh: Thí nghiệm 1 đã chứng minh gì?

Thí nghiệm 1 xử lý hàm reward **gián đoạn nhị phân** ($L = \infty$ tại ranh giới):
- RS-LiDAR giảm $L_{\text{boundary}}$ tới **19.72×** ($M=8$), phục hồi gradient từ **10.8% → 28.2%**
- Đây là trường hợp **"bệnh lý nhất"** — hàm không liên tục

### 0.2. Thí nghiệm 2 nhắm vào bệnh lý khác: Hằng số Lipschitz cục bộ khổng lồ

Khác với Threshold (gián đoạn hoàn toàn), ResNet50 là hàm **liên tục nhưng có Lipschitz cục bộ cực lớn**:
$$L_{\text{local}}(x) = \|\nabla_x f_{\theta}(x)\|_2 \quad \text{có thể lên tới } 10^3 \sim 10^5$$

Điều này tạo ra hiện tượng:
- **Adversarial Vulnerability**: Nhiễu $\delta$ cực nhỏ ($\|\delta\|_2 < 0.5$) khiến ResNet50 thay đổi dự đoán hoàn toàn
- **Gradient-based optimization bị lừa**: Vanilla LiDAR đi theo gradient dốc đứng cục bộ → ảnh bị đẩy vào vùng **adversarial** (logit cao nhưng không có semantic content) 
- **RS-LiDAR**: Gaussian smoothing san phẳng các spikes cục bộ → ảnh phải chứa **robust features** (đặc trưng ngữ nghĩa thực sự) để đạt reward cao trung bình

### 0.3. Kết nối lý thuyết Robustness (Cohen et al., 2019)

Cohen et al. chứng minh: Nếu classifier $f$ đúng trên phần lớn quả cầu Gaussian quanh $x$:
$$\Pr_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}[f(x + \epsilon) = c_A] \geq p_A > 0.5$$

thì $g(x) = c_A$ (smoothed classifier) **được chứng nhận bền vững** trong bán kính $\ell_2$:
$$\|x - x'\|_2 < \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)$$

> [!IMPORTANT]
> **RS-LiDAR kế thừa trực tiếp tính chất này**: Khi dùng RS để đánh giá reward trên ResNet50, quá trình tối ưu tự động bị ép vào vùng **certified robust** — không thể tấn công adversarial → ảnh phải chứa semantic thực.

---

## 1. MỤC TIÊU KHOA HỌC CỤ THỂ

| # | Mục tiêu | Metric đo | Kỳ vọng |
|:---:|:---|:---|:---|
| 1 | Chứng minh Vanilla LiDAR hoạt động như **Adversarial Attacker** | Visual comparison + Saliency maps | Ảnh Vanilla: nhiễu adversarial rõ ràng |
| 2 | Chứng minh RS-LiDAR kế thừa **Certified Robustness** | Cohen certificate radius | RS output nằm trong vùng certified |
| 3 | Đo hằng số Lipschitz cục bộ bị san phẳng | $L_{\text{local}}, L_{\text{mean}}, L_{\text{max}}$ | RS giảm $L$ hàng chục lần |
| 4 | So sánh gradient norm trong quá trình sampling | $\|\nabla_x R\|_2$ distribution | Vanilla: spikes; RS: smooth |
| 5 | Trực quan hóa đặc trưng ngữ nghĩa vững chắc | Grad-CAM Saliency Focus Ratio | RS tập trung vào vật thể thay vì nhiễu |

---

## 2. CÀI ĐẶT THỰC NGHIỆM CHI TIẾT

### 2.1. Mô Hình Reward: ResNet50 ImageNet

```python
import torch
import torchvision.models as models
import torchvision.transforms as T

# ==========================================
# RESNET50 REWARD FUNCTION
# ==========================================
class ResNet50AdversarialReward:
    """
    Hàm reward dựa trên logit (hoặc softmax probability) của ResNet50 
    cho target class. Đây là reward liên tục nhưng cực kỳ gồ ghề 
    (local Lipschitz >> 1).
    """
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Các target class thú vị cho thí nghiệm
    TARGET_CLASSES = {
        207: "golden retriever",
        281: "tabby cat", 
        417: "balloon",
        949: "strawberry",
        985: "daisy",
        388: "giant panda",
    }
    
    def __init__(self, target_class_id=207, reward_mode="logit", device="cuda"):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        self.model.eval()
        self.target_class_id = target_class_id
        self.reward_mode = reward_mode  # "logit", "softmax", "margin"
        self.device = device
        
        self.normalize = T.Normalize(
            mean=self.IMAGENET_MEAN, 
            std=self.IMAGENET_STD
        )
        # Resize về 224x224 (input chuẩn ResNet50)
        self.resize = T.Resize((224, 224), antialias=True)
    
    @torch.inference_mode()
    def __call__(self, images):
        """
        Args:
            images: Tensor [N, 3, H, W] trong dải [-1, 1] (output từ SD pipeline)
        Returns:
            rewards: Tensor [N] - giá trị reward cho từng ảnh
        """
        # Chuyển từ [-1, 1] → [0, 1]
        imgs_01 = (images + 1.0) / 2.0
        imgs_01 = imgs_01.clamp(0.0, 1.0)
        
        # Resize về 224x224
        imgs_resized = self.resize(imgs_01)
        
        # Chuẩn hóa ImageNet
        imgs_norm = self.normalize(imgs_resized)
        
        # Forward pass
        logits = self.model(imgs_norm)  # [N, 1000]
        
        if self.reward_mode == "logit":
            # Raw logit của target class — gradient cực kỳ nhạy
            rewards = logits[:, self.target_class_id]
        elif self.reward_mode == "softmax":
            # Softmax probability — bounded [0, 1] nhưng vẫn rất gồ ghề
            probs = torch.softmax(logits, dim=-1)
            rewards = probs[:, self.target_class_id]
        elif self.reward_mode == "margin":
            # Target logit - max logit of other classes
            # Ép model phải tự tin VÀ vượt trội
            target_logit = logits[:, self.target_class_id]
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
            mask[self.target_class_id] = False
            max_other = logits[:, mask].max(dim=-1).values
            rewards = target_logit - max_other
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        
        return rewards
```

### 2.2. Cấu Hình Thực Nghiệm Đề Xuất

| Tham Số | Giá Trị | Lý Do |
|:---|:---:|:---|
| **Diffusion Model** | `stable-diffusion-v1-5` | Giống Exp 1 để so sánh công bằng |
| **Reward Model** | `torchvision.resnet50(pretrained=True)` | Classifier ImageNet chuẩn, dễ tái tạo |
| **Target Classes** | 10 classes đại diện 10 super-categories | Tránh cherry-pick, bao phủ toàn diện |
| **Reward Mode** | `logit` (chính) + `softmax` (phụ) | Logit cho gradient mạnh nhất |
| **Prompts** | **50 prompts** (5 prompts/class × 10 classes) | Quy mô ngang bằng 50 prompts của Exp 1 |
| **Seeds / Particles** | $K = 10$ hạt/prompt | **Tổng 500 cặp mẫu** (đạt ý nghĩa thống kê cao) |
| **$\sigma_1$ (vi nhiễu)** | $0.05$ (chính); sweep $\{0.001, 0.005, 0.01, 0.02, 0.05, 0.1\}$ | So sánh với Exp 1 |
| **$\sigma_2$ (smoothing)** | sweep $\{0.0, 0.1, 0.25, 0.5, 1.0\}$ | Giống Exp 1 |
| **$M$ (MC samples)** | sweep $\{1, 2, 4, 8\}$ | Giống Exp 1 |
| **CRN** | `True` | Bắt buộc — khử phương sai |
| **Resolution** | $512 \times 512$ (SD) → resize $224 \times 224$ (ResNet) | Chuẩn ImageNet |

### 2.3. Cơ Sở Khoa Học & Luận Điểm Biện Minh (Scientific Justification) Cho Việc Chọn 10 Classes

Việc chọn lựa 10 classes ImageNet cho Thí nghiệm 2 được xây dựng dựa trên 4 luận điểm học thuật chặt chẽ nhằm đáp ứng các tiêu chuẩn phản biện khắt khe nhất của các hội nghị hàng đầu (NeurIPS, ICML, CVPR):

1. **Tiền lệ học thuật trong Adversarial Machine Learning**:
   - **Class 207 (Golden Retriever)**: Là class kinh điển trong các nghiên cứu tấn công đối nghịch trên ImageNet.
   - **Class 388 (Giant Panda)**: Là ví dụ lịch sử mang tính định hình của toàn bộ ngành học tập đối nghịch — công trình nền tảng của *Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples"* đã dùng hình ảnh chú gấu trúc (Panda) bị thêm nhiễu vi mô không nhìn thấy bằng mắt thường để đánh lừa mạng phân loại tin chắc 99.3% là vượn Gibbon. Việc đưa Giant Panda vào thí nghiệm kết nối trực tiếp RS-LiDAR với cội nguồn của lý thuyết adversarial robustness.

2. **Bao phủ toàn diện 10 Super-Categories (Triệt tiêu nghi ngờ Cherry-picking)**:
   - Thay vì chỉ thử nghiệm hẹp trên 2-3 class dễ, 10 class này được phân bổ trải rộng theo cây phân cấp **WordNet Hierarchy** của ImageNet, đại diện cho 10 miền hoàn toàn tách biệt:
     - *Canine (Chó)*: Golden Retriever (207)
     - *Feline (Mèo)*: Tabby Cat (281)
     - *Ursidae / Wild Mammal (Thú lớn hoang dã)*: Giant Panda (388)
     - *Avian (Loài chim)*: Flamingo (130)
     - *Land Vehicle (Phương tiện cơ giới)*: Sports Car (817)
     - *Aviation (Hàng không)*: Airliner (404)
     - *Flora (Thực vật)*: Daisy (985)
     - *Fruit / Food (Thực phẩm)*: Strawberry (949)
     - *Electronics (Thiết bị công nghệ)*: Laptop (620)
     - *Natural Landscape (Cảnh quan địa chất)*: Volcano (980)

3. **Ma trận kiểm thử đa dạng về tần số thị giác & kết cấu (Spatial Frequency & Texture Matrix)**:
   - Các class đại diện cho đầy đủ các đặc tính không gian khác nhau:
     - *Mảng màu mượt, tần số thấp/trung bình*: Golden Retriever, Giant Panda, Flamingo.
     - *Hoa văn sọc vằn tần số cao*: Tabby Cat.
     - *Cấu trúc hình học đối xứng tỏa tròn*: Daisy.
     - *Bề mặt lấm tấm hạt vi mô*: Strawberry.
     - *Cạnh sắc nhân tạo, vật liệu phản quang kim loại*: Sports Car, Airliner, Laptop.
     - *Bề mặt phi hình học, khói bụi tự nhiên*: Volcano.
   - Thử nghiệm trên ma trận này chứng minh khả năng san phẳng Lipschitz của RS là phổ quát trên mọi loại đặc trưng ảnh.

4. **Đồng nhất quy mô với Thí nghiệm 1 (Scale Parity with Exp 1)**:
   - 50 prompts $\times$ 10 particles = **500 mẫu ảnh**, hoàn toàn tương đương với 50 prompts / 500 mẫu của Thí nghiệm 1 (Threshold Reward).
   - Quy mô này mang lại ý nghĩa thống kê vượt trội (đủ lớn để tính $p$-value, khoảng tin cậy chặt, phân phối KDE chuẩn xác), loại bỏ hoàn toàn nhận xét "kết quả ăn may trên tập mẫu nhỏ".

### 2.4. Danh Sách Đầy Đủ 50 Prompts (ADVERSARIAL_REWARD_PROMPTS)

```python
ADVERSARIAL_REWARD_PROMPTS = [
    # === 1. Canine: Golden Retriever (target_class_id = 207) ===
    {"prompt": "a golden retriever sitting in a park", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a cute golden retriever puppy looking at camera", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a golden retriever running through autumn leaves", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a golden retriever lying on green grass", "target_class": 207, "class_name": "golden retriever"},
    {"prompt": "a happy golden retriever with tongue out in sunlight", "target_class": 207, "class_name": "golden retriever"},
    
    # === 2. Feline: Tabby Cat (target_class_id = 281) ===
    {"prompt": "a tabby cat sleeping on a windowsill", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "a tabby cat sitting on a sofa", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "a striped tabby cat on a wooden floor", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "a cute tabby kitten playing with yarn", "target_class": 281, "class_name": "tabby cat"},
    {"prompt": "close-up portrait of a striped tabby cat", "target_class": 281, "class_name": "tabby cat"},
    
    # === 3. Ursidae: Giant Panda (target_class_id = 388 - Goodfellow et al. 2014) ===
    {"prompt": "a giant panda sitting and eating bamboo", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "a cute giant panda bear in a bamboo forest", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "close-up portrait of a giant panda looking forward", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "a giant panda resting on a wooden log", "target_class": 388, "class_name": "giant panda"},
    {"prompt": "a baby giant panda playing on green grass", "target_class": 388, "class_name": "giant panda"},
    
    # === 4. Avian: Flamingo (target_class_id = 130) ===
    {"prompt": "a vibrant pink flamingo standing in shallow water", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "a flock of pink flamingos in a tropical lagoon", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "close-up portrait of an elegant pink flamingo", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "a pink flamingo resting on one leg in the sun", "target_class": 130, "class_name": "flamingo"},
    {"prompt": "a beautiful flamingo reflected in calm water", "target_class": 130, "class_name": "flamingo"},
    
    # === 5. Vehicle: Sports Car (target_class_id = 817) ===
    {"prompt": "a sleek red sports car parked on a city street", "target_class": 817, "class_name": "sports car"},
    {"prompt": "a modern luxury sports car driving on highway", "target_class": 817, "class_name": "sports car"},
    {"prompt": "front view of a shiny black sports car", "target_class": 817, "class_name": "sports car"},
    {"prompt": "a yellow sports car parked on a mountain road", "target_class": 817, "class_name": "sports car"},
    {"prompt": "a red sports car under bright studio lights", "target_class": 817, "class_name": "sports car"},
    
    # === 6. Aviation: Airliner (target_class_id = 404) ===
    {"prompt": "a commercial airliner flying through blue sky with clouds", "target_class": 404, "class_name": "airliner"},
    {"prompt": "a passenger airliner approaching runway for landing", "target_class": 404, "class_name": "airliner"},
    {"prompt": "a large passenger jet airliner parked at an airport", "target_class": 404, "class_name": "airliner"},
    {"prompt": "side view of a white passenger airliner in flight", "target_class": 404, "class_name": "airliner"},
    {"prompt": "an airliner taking off into a golden sunset", "target_class": 404, "class_name": "airliner"},
    
    # === 7. Flora: Daisy (target_class_id = 985) ===
    {"prompt": "a field of white daisies in bright sunlight", "target_class": 985, "class_name": "daisy"},
    {"prompt": "a single white daisy flower close-up in morning dew", "target_class": 985, "class_name": "daisy"},
    {"prompt": "white daisy flowers blooming in a lush garden", "target_class": 985, "class_name": "daisy"},
    {"prompt": "macro photography of a daisy with yellow center", "target_class": 985, "class_name": "daisy"},
    {"prompt": "wild white daisies growing in green grass", "target_class": 985, "class_name": "daisy"},
    
    # === 8. Food: Strawberry (target_class_id = 949) ===
    {"prompt": "a bowl of fresh ripe red strawberries", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "close-up of a single juicy red strawberry with green leaves", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "freshly picked red strawberries on a wooden table", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "a pile of sweet red strawberries in sunlight", "target_class": 949, "class_name": "strawberry"},
    {"prompt": "ripe red strawberries growing in a garden patch", "target_class": 949, "class_name": "strawberry"},
    
    # === 9. Electronics: Laptop (target_class_id = 620) ===
    {"prompt": "a modern silver laptop open on a wooden desk", "target_class": 620, "class_name": "laptop"},
    {"prompt": "a sleek laptop with glowing screen in a modern cafe", "target_class": 620, "class_name": "laptop"},
    {"prompt": "a laptop computer next to a cup of coffee on desk", "target_class": 620, "class_name": "laptop"},
    {"prompt": "a slim laptop on an office desk beside a window", "target_class": 620, "class_name": "laptop"},
    {"prompt": "close-up view of a modern laptop keyboard and screen", "target_class": 620, "class_name": "laptop"},
    
    # === 10. Landscape: Volcano (target_class_id = 980) ===
    {"prompt": "a majestic snow-capped volcano under blue sky", "target_class": 980, "class_name": "volcano"},
    {"prompt": "an active volcano emitting smoke and ash into the sky", "target_class": 980, "class_name": "volcano"},
    {"prompt": "a dramatic volcano mountain landscape at sunrise", "target_class": 980, "class_name": "volcano"},
    {"prompt": "a towering volcano rising above a peaceful lake", "target_class": 980, "class_name": "volcano"},
    {"prompt": "aerial view of a volcanic mountain crater with clouds", "target_class": 980, "class_name": "volcano"},
]
```

---

## 3. PIPELINE ĐO ĐẠC LIPSCHITZ (Theo mẫu Exp 1)

### 3.1. Cấu Trúc Dữ Liệu Thu Thập

Theo đúng schema đã dùng trong [test_threshold_reward.py](file:///c:/HUST/Projects/Lab/RS-LiDAR/test_threshold_reward.py), mỗi cặp mẫu lưu:

```json
{
    "prompt_idx": 0,
    "particle_idx": 3,
    "prompt": "a golden retriever sitting in a park",
    "target_class": 207,
    "delta_x": 43.56,
    "models": {
        "ResNet50-Adversarial": {
            "r_clean_vanilla": 8.234,
            "r_pert_vanilla": 7.891,
            "delta_r_vanilla": 0.343,
            "slope_vanilla": 0.00788,
            "rs_by_m": {
                "1": { "0.5": { "r_clean": 5.12, "r_pert": 5.08, "delta_r": 0.04, "slope": 0.00092 } },
                "2": { ... },
                "4": { ... },
                "8": { ... }
            },
            "sigma1_sweep": { ... },
            "gradient_norm_vanilla": 1247.3,
            "gradient_norm_rs": 85.2
        }
    }
}
```

### 3.2. Thực Nghiệm Đo Lipschitz — Pseudocode Pipeline

```python
def run_adversarial_lipschitz_test(pipe, vae, resnet_reward, prompt_list, args):
    """
    Pipeline đo đạc Lipschitz cho ResNet50 Adversarial Reward.
    Theo cấu trúc y hệt run_empirical_lipschitz_test() trong test_threshold_reward.py
    """
    for prompt_data in prompt_list:
        prompt = prompt_data["prompt"]
        target_class = prompt_data["target_class"]
        
        # 1. Sinh ảnh sạch x_clean và ảnh vi nhiễu x_pert
        clean_tensors, pert_tensors, delta_x = generate_clean_and_perturbed(
            pipe, vae, prompt, sigma1=args.sigma1
        )
        
        # 2. Đánh giá Vanilla Reward
        r_clean_vanilla = resnet_reward(clean_tensors)   # logit của target class
        r_pert_vanilla = resnet_reward(pert_tensors)
        slope_vanilla = |r_clean - r_pert| / delta_x     # Lipschitz cục bộ
        
        # 3. [MỚI] Đo Gradient Norm trực tiếp (khác với Exp 1!)
        # Vì ResNet50 khả vi, ta có thể tính ∇R(x) trực tiếp
        clean_tensors.requires_grad_(True)
        logit = resnet_reward.model(normalize(resize(clean_tensors)))[:, target_class]
        logit.sum().backward()
        grad_norm_vanilla = clean_tensors.grad.view(N, -1).norm(2, dim=1)
        
        # 4. RS-LiDAR Monte Carlo Smoothing (prefix-mean trick từ Exp 1)
        for sigma2 in sigma2_vals:
            for m in range(max_M):
                u_m = torch.randn_like(clean_tensors) * sigma2
                noisy_clean = (clean_tensors + u_m).clamp(-1, 1)
                r_m_clean = resnet_reward(noisy_clean)
                # CRN: Dùng cùng u_m cho pert
                noisy_pert = (pert_tensors + u_m).clamp(-1, 1)
                r_m_pert = resnet_reward(noisy_pert)
            # Prefix-mean: RS(M=k) = mean(r_1, ..., r_k) cho k in m_vals
        
        # 5. [MỚI] Đo Gradient Norm cho RS-LiDAR (Score Function Estimator)
        # ∇R_σ(x) ≈ (1/M) Σ [R(x + σu_m) · u_m / σ]  (Stein's identity)
        grad_estimates = []
        for m in range(M):
            u_m = torch.randn_like(clean_tensors)
            r_m = resnet_reward(clean_tensors + sigma2 * u_m)
            grad_estimates.append(r_m.unsqueeze(-1) * u_m.view(N, -1) / sigma2)
        grad_rs = torch.stack(grad_estimates).mean(dim=0)
        grad_norm_rs = grad_rs.norm(2, dim=1)
```

---

## 4. CÁC THỰC NGHIỆM ĐẶC THÙ CHO EXP 2 (ĐỘC LẬP 100% TRONG `test_adversarial_reward.py`)

> [!IMPORTANT]
> Toàn bộ các thực nghiệm dưới đây được thiết kế để chạy **hoàn toàn độc lập** trong module `test_adversarial_reward.py` (tương tự như `test_threshold_reward.py`).  
> **Tuyệt đối không can thiệp hay phụ thuộc** vào `fkd_class.py`, `LiDAR_sampling.py`, hay bất kỳ thành phần nào của nhánh Ablation 2 (SMC Resampling) của collaborator.

### 4.1. Thực Nghiệm A: Visual Comparison & Saliency / Adversarial Sensitivity Map

**Mục đích:** Chứng minh trực quan bằng hình ảnh: Gradient của Vanilla ResNet50 tập trung vào các đốm nhiễu tần số cao (adversarial noise), trong khi RS-LiDAR ép gradient tập trung vào cấu trúc ngữ nghĩa thực tế (robust features).

**Protocol độc lập trong `test_adversarial_reward.py`:**
1. Lấy 3 prompts đại diện (1 prompt / class: Golden Retriever, Tabby Cat, Daisy).
2. Với mỗi prompt, sinh các mẫu ảnh bằng SD-v1.5 (`StableDiffusionPipeline`).
3. So sánh 3 góc nhìn trực quan:
   - **Ảnh gốc (Clean Sample - No Guidance)**: Ảnh SD sinh ra tự nhiên.
   - **Vanilla Adversarial Sensitivity (Grad-CAM Layer 4)**:
     - Trích xuất activation map $A$ tại tầng tích chập sâu nhất `resnet50.layer4` ($7 \times 7$, 2048 channels) và tính trọng số gradient $\alpha_k = \frac{1}{Z} \sum_{i,j} \frac{\partial R}{\partial A_{i,j}^k}$.
     - Bản đồ Grad-CAM của Vanilla bị phân tán rải rác trên nền cỏ, góc ảnh và các hoa văn giả mạo (do các đòn bẩy đối kháng kích hoạt sai lệch).
   - **RS-LiDAR Robust Features (Smoothed Grad-CAM Layer 4)**:
     - Tính Smoothed Grad-CAM qua kỳ vọng Monte Carlo: $\text{CAM}_{\sigma_2}(x) = \mathbb{E}_{\epsilon}[\text{Grad-CAM}(x + \epsilon)]$ với $M$ mẫu nhiễu Gaussian.
     - Lọc bỏ các xung kích hoạt nhiễu tần số cao, hội tụ sắc nét vào đúng cấu trúc thực thể (khuôn mặt gấu trúc, mõm chó, thân xe thể thao).

**Kỳ vọng visual:**

```
┌────────────────────────────────────────────────────────────────────────┐
│ Case Study: "portrait of a giant panda" (ImageNet Class 388)           │
├────────────────────┬──────────────────────────┬────────────────────────┤
│ Clean Image (SD)   │ Vanilla Grad-CAM (L4)    │ RS-LiDAR Smoothed CAM  │
│                    │ (Spurious Background)    │ (Robust Object Focus)  │
│                    │                          │                        │
│ 🖼️ Ảnh gấu trúc      │ 🗺️ Kích hoạt phân tán     │ 🗺️ Tập trung sắc nét:   │
│ chuẩn (SD 1.5)     │ rải rác ngoài viền nền   │ khuôn mặt gấu trúc,    │
│                    │ (Adversarial artifacts)  │ mắt, tai, mũi (Robust) │
└────────────────────┴──────────────────────────┴────────────────────────┘
```

**Metrics định lượng khách quan:**

| Tiêu chí | Clean Image (No Guidance) | Vanilla Grad-CAM (Layer 4) | RS-LiDAR Smoothed Grad-CAM |
|:---|:---:|:---:|:---:|
| **Mean Target Logit** | ~2 – 5 | **15 – 30** (cao bất thường/ảo) | 8 – 15 (hợp lý, vững chắc) |
| **Central Focus Ratio** | — | Phân tán (~25 – 40%) | **Tập trung cao (~60 – 85%)** |
| **Top-20% Concentration** | — | Thấp (~30 – 45%) | **Tập trung cao (~65 – 85%)** |

---

### 4.2. Thực Nghiệm B: Gradient Norm Distribution & 1D Landscape Slicing

Khác với Exp 1 (Threshold) không khả vi ($L=\infty$), ResNet50 có gradient xác định ở mọi điểm nhưng bị bệnh **bùng nổ gradient cục bộ** (Gradient Spikes). Ta đo đạc đặc tính này bằng 2 phương pháp toán học độc lập:

#### 1. Phân phối chuẩn Gradient Norm (KDE & Boxplot):
- Đo trực tiếp $\|\nabla_x R(x)\|_2$ cho toàn bộ batch mẫu bằng `torch.autograd.grad`.
- **Vanilla**: $\|\nabla_x R(x)\|_2$ vọt lên cực đại ($10^2 \sim 10^4$), độ lệch chuẩn khổng lồ.
- **RS-LiDAR**: $\|\nabla_x \tilde{R}_{\sigma_2}(x)\|_2$ bị chặn theo đúng Theorem 1:
  $$\|\nabla \tilde{R}_{\sigma_2}(x)\|_2 \le \frac{B \sqrt{d}}{\sigma_2}$$
  Độ dốc bị nén về dải an toàn, dập tắt hoàn toàn các bẫy cực trị hẹp.

#### 2. Cắt lát cảnh quan 1D (1D Landscape Slicing):
- Chọn 2 điểm ảnh $x_A$ và $x_B$ trong không gian biểu diễn, nội suy 1000 điểm trên đoạn thẳng $x(\alpha) = x_A + \alpha (x_B - x_A)$ với $\alpha \in [0, 1]$ (tái sử dụng pattern từ Exp 1).
- Vẽ đường đồ thị $R(x(\alpha))$ (Vanilla) vs $\tilde{R}_{\sigma_2}(x(\alpha))$ (RS):
  - **Vanilla curve**: Gồ ghề chằng chịt các gai nhọn li ti (adversarial spikes).
  - **RS curve**: Đường cong lượn sóng mượt mà, dập tắt toàn bộ sóng tần số cao.

---

### 4.3. Thực Nghiệm C: Certified Robustness Radius (Cohen Certificate)

**Protocol:** Dựa trên định lý Cohen et al. (2019) về Randomized Smoothing cho bộ phân loại:
1. Cho mỗi mẫu ảnh $x$, ước lượng xác suất phân lớp đúng của class mục tiêu dưới nhiễu Gaussian $\epsilon \sim \mathcal{N}(0, \sigma_2^2 I)$:
   $$p_A = \Pr_{\epsilon}[\arg\max f(x + \epsilon) = \text{target}]$$
2. Tính bán kính bền vững được chứng thực toán học (Certified $\ell_2$ Radius):
   $$R_{\text{cert}} = \sigma_2 \cdot \Phi^{-1}(\underline{p_A})$$
   (với $\underline{p_A}$ là cận dưới khoảng tin cậy Clopper-Pearson 99.9%).
3. **So sánh**:
   - Vùng ảnh chịu ảnh hưởng của Vanilla gradient: $p_A \approx 0.5$ (dễ bị lật nhãn khi rung nhẹ) $\to R_{\text{cert}} \approx 0$.
   - Vùng ảnh qua RS smoothing: $p_A \ge 0.85 \sim 0.95 \to R_{\text{cert}} > 0.4 \sim 0.8$ (được bảo vệ toán học).


---

## 5. OUTPUT METRICS & PLOTS CẦN CÓ

### 5.1. Bảng Benchmark Lipschitz (Tương tự Bảng 1A/1B Exp 1)

| Metric | Vanilla | RS (M=1) | RS (M=2) | RS (M=4) | RS (M=8) |
|:---|:---:|:---:|:---:|:---:|:---:|
| $L_{\text{mean}}$ (All samples) | ? | ? | ? | ? | ? |
| $L_{\text{max}}$ (All samples) | ? | ? | ? | ? | ? |
| $\|\nabla R\|_{\text{mean}}$ | ? | ? | ? | ? | ? |
| $\|\nabla R\|_{\text{max}}$ | ? | ? | ? | ? | ? |

> [!NOTE]
> **Khác biệt quan trọng với Exp 1**: Ở đây không có "boundary samples" vì reward liên tục. Thay vào đó, **mọi mẫu đều có gradient khác 0** — đây chính là điểm mạnh: $L$ lớn ở mọi nơi, không chỉ ở ranh giới.

### 5.2. Biểu Đồ Dashboard 2×3 (6 panels)

```
┌───────────────────────┬───────────────────────┬───────────────────────┐
│ A. Bảng Lipschitz     │ B. L_mean vs σ₂       │ C. Gradient Norm      │
│    Benchmark Table    │    (Contraction Line) │    Distribution (KDE) │
├───────────────────────┼───────────────────────┼───────────────────────┤
│ D. Reward Dist        │ E. 1D Landscape Slice │ F. Saliency Grid      │
│    (Histogram)        │    (Spike Suppression)│    (Vanilla vs RS)    │
└───────────────────────┴───────────────────────┴───────────────────────┘
```

### 5.3. Figure Chủ Đạo Cho Paper

1. **Figure chính**: Grad-CAM Saliency Grid (Layer 4) — trực quan hóa robust semantic attribution vs adversarial spurious activation
2. **Figure phụ**: 1D Landscape Slicing — chứng minh dập tắt các gai nhọn (spikes) cục bộ
3. **Table**: Lipschitz Benchmark Table + Contraction Ratio ($L_{\text{RS}} / L_{\text{Vanilla}}$)

---

## 6. SO SÁNH VỚI EXP 1: TẠI SAO CẦN CẢ HAI?

| Tiêu chí | Exp 1 (Threshold) | Exp 2 (Adversarial) |
|:---|:---|:---|
| **Loại bệnh lý** | $L = \infty$ (gián đoạn) | $L_{\text{local}} \gg 1$ (liên tục nhưng gồ ghề) |
| **Gradient Starvation** | ✅ 89.2% mẫu bị mù | ❌ Gradient luôn tồn tại — nhưng **SAI HƯỚNG** |
| **Vấn đề chính** | Không có gradient | Gradient dẫn vào adversarial trap |
| **RS giải quyết bằng** | Phục hồi gradient (beacon) | San phẳng landscape → tránh trap |
| **Hình ảnh minh họa** | Phân phối đếm bounding box | Saliency map & Robust semantic features |
| **Kết nối lý thuyết** | Theorem 1 (Smoothness) | Theorem 1 + Cohen 2019 (Certified Robustness) |

> [!TIP]
> Hai thí nghiệm bổ trợ nhau hoàn hảo: Exp 1 chứng minh RS **tạo ra** gradient từ con số 0; Exp 2 chứng minh RS **sửa hướng** gradient sai lệch.

---

## 7. CẤU TRÚC FILE CODE ĐỀ XUẤT (ĐỘC LẬP 100%)

> [!NOTE]
> Module này hoàn toàn khép kín, hoạt động độc lập tương tự `test_threshold_reward.py`. Tuyệt đối không import hoặc sửa đổi code trong `fkd_diffusers/`, `LiDAR_sampling.py`, đảm bảo không ảnh hưởng đến nhánh Ablation 2 của collaborator.

```
RS-LiDAR/
├── test_adversarial_reward.py          # Script chính ĐỘC LẬP (tương tự test_threshold_reward.py)
│   ├── ResNet50AdversarialReward       # Class reward (ResNet-50 ImageNet target logit)
│   ├── ADVERSARIAL_REWARD_PROMPTS      # Dataset 50 prompts (5 prompts x 10 classes)
│   ├── generate_clean_and_perturbed()  # Sinh ảnh probe qua StableDiffusionPipeline
│   ├── run_empirical_lipschitz_test()  # Pipeline đo Lipschitz thực nghiệm (tái sử dụng từ Exp 1)
│   ├── compute_gradient_and_saliency() # Đo Gradient Norm & vẽ Saliency Heatmap
│   ├── slice_1d_landscape()            # Cắt lát 1D landscape chứng minh dập tắt spikes
│   └── merge_and_plot_results()        # Xuất Bảng Lipschitz & Dashboard 6 panels
│
├── colab/RS_LiDAR_Adversarial_Reward_Experiment_Colab.ipynb
│   └── Notebook Colab 1-click tự chạy độc lập (theo mẫu Threshold Colab)
│
├── md/ADVERSARIAL_REWARD_LIPSCHITZ_EXPERIMENT_REPORT.md
│   └── Báo cáo kết quả phân tích toán học & hình ảnh
│
└── figures/
    ├── adversarial_lipschitz_combined_*.png
    ├── adversarial_gradient_landscape_*.png
    ├── adversarial_visual_saliency_*.png
    ├── adversarial_reward_dist_*.png
    ├── adversarial_slope_scatter_*.png
    └── adversarial_table_*.png
```

---

## 8. THỨ TỰ TRIỂN KHAI TỪNG BƯỚC

### Phase 1: Viết Script Độc Lập `test_adversarial_reward.py` (Ưu tiên cao)
- [ ] Cài đặt class `ResNet50AdversarialReward` với ImageNet normalization chuẩn
- [ ] Xây dựng bộ dataset `ADVERSARIAL_REWARD_PROMPTS` (50 prompts cho 10 classes)
- [ ] Triển khai hàm `generate_clean_and_perturbed()` và pipeline đo Lipschitz `run_empirical_lipschitz_test()`
- [ ] Thêm module tính Gradient Norm trực tiếp và Saliency Heatmap
- [ ] Thêm module 1D Landscape Slicing (1000 điểm) để vẽ đồ thị dập tắt spike
- [ ] Thêm module xuất Bảng Lipschitz và Dashboard 6 panels

### Phase 2: Xây Dựng Notebook Colab Độc Lập
- [ ] Tạo `colab/RS_LiDAR_Adversarial_Reward_Experiment_Colab.ipynb` theo pattern chuẩn của `RS_LiDAR_Threshold_Reward_Experiment_Colab.ipynb`
- [ ] Đảm bảo 1-click Run All trên GPU T4/A100 (Google Colab hoặc Kaggle)
- [ ] Kiểm tra cơ chế checkpointing resume an toàn

### Phase 3: Thực Thi Thí Nghiệm & Thu Thập Kết Quả
- [ ] Chạy sweep $\sigma_1 \in \{0.001, 0.005, 0.01, 0.02, 0.05, 0.1\}$ và $\sigma_2 \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$ với $M \in \{1, 2, 4, 8\}$
- [ ] Xuất số liệu Lipschitz contraction ratio, phân phối Gradient Norm, đồ thị 1D Slicing, và Saliency Maps

### Phase 4: Tổng Hợp Báo Cáo
- [ ] Soạn thảo `md/ADVERSARIAL_REWARD_LIPSCHITZ_EXPERIMENT_REPORT.md`
- [ ] Phân tích đối chiếu giữa Exp 1 (Threshold) và Exp 2 (Adversarial) để hoàn thiện luận điểm cho bài báo

---

## 9. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Xác suất | Giải pháp |
|:---|:---:|:---|
| Logit ResNet50 có biên độ quá rộng giữa các ảnh | Trung bình | Dùng Margin Reward: $R(x) = f_{target}(x) - \max_{j \neq target} f_j(x)$ |
| Gradient norm quá lớn gây tràn số float16 | Thấp | Tính toán gradient trong `float32` cho phần ResNet50 |
| VRAM đầy khi chạy SD1.5 + ResNet50 | Thấp | Đặt `eval()` và `torch.inference_mode()` cho generation, chỉ bật grad cho ảnh đơn khi đo |
| Ảnh probe từ SD không ra đúng class | Rất thấp | Bộ prompt đã được tinh tuyển kỹ lưỡng với từ khóa rõ ràng (golden retriever, tabby cat, daisy) |

---

## 10. KẾT LUẬN KỲ VỌNG

Sau khi hoàn thành Exp 2, bài báo sẽ có **bộ tam giác hoàn hảo**:

1. **Theorem 1** (Lý thuyết): RS biến mọi hàm bị chặn thành $C^\infty$ Lipschitz với độ dốc bị chặn bởi $\frac{B\sqrt{d}}{\sigma}$.
2. **Exp 1** (Threshold): Chứng minh trên hàm gián đoạn — RS tạo ra gradient từ con số 0 (chữa bệnh mù gradient).
3. **Exp 2** (Adversarial): Chứng minh trên hàm liên tục có $L$ khổng lồ — RS san phẳng các gai nhọn cục bộ, triệt tiêu bẫy đối nghịch và bảo vệ các đặc trưng ngữ nghĩa bền vững.
