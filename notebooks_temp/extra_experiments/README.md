# Kế hoạch thí nghiệm bổ sung

> Thư mục tạm cho cộng tác. Sẽ bị loại khỏi bản code nộp kèm bài.

**Hạn chót (giờ AoE):** đăng ký abstract **14/1/2027**, nộp bài **22/1/2027**.

**Mục tiêu:** cả ba reviewer chấm 7/10. Phần viết và lý thuyết đã xong. Điểm bây giờ phụ thuộc vào 6 thí nghiệm dưới đây, mỗi thí nghiệm trả lời một câu hỏi mà reviewer chắc chắn sẽ đặt ra.

Ý tưởng: thay vì chấm mỗi ứng viên bằng **một lần** gọi reward model, ta chấm bằng **trung bình reward trên M bản nhiễu Gauss** của ảnh (điểm làm trơn R̄σ). Áp dụng cho LiDAR (RS-LiDAR) và SMC (RS-SMC).

---

## Mục lục

1. [Chuẩn bị trước khi chạy](#1-chuẩn-bị-trước-khi-chạy)
2. [Bảng tổng quan 6 thí nghiệm](#2-bảng-tổng-quan)
3. [Chi tiết từng thí nghiệm](#3-chi-tiết-từng-thí-nghiệm): [E1](#e1--smoothing-có-chỉ-là-hạ-nhiệt-độ-không) · [E2](#e2--smoothing-đổi-ứng-viên-thắng-ở-đâu-và-có-đổi-đúng-không) · [E3](#e3--so-sánh-cùng-số-lần-gọi-reward) · [E4](#e4--reward-rời-rạc-dẫn-hướng-bằng-bộ-chấm-geneval) · [E5](#e5--nhiều-reward-cùng-lúc-ir--hps) · [E6](#e6--flow-matching-flux1-dev)
4. [Lịch và phân công](#4-lịch-và-phân-công)
5. [Quy tắc chung khi chạy](#5-quy-tắc-chung-khi-chạy)

---

## 1. Chuẩn bị trước khi chạy

### 1.1 Dùng chung Google Drive (bắt buộc cho người thứ hai)

Mọi notebook đọc và ghi vào `MyDrive/RS-LiDAR/`. Các thí nghiệm E1–E5 **dùng lại pool lookahead của Table 1**, và pool này đang nằm trên Drive của chủ dự án.

- **Chủ dự án:** chia sẻ thư mục `RS-LiDAR` trên Google Drive cho người kia với quyền **Editor**.
- **Người thứ hai:** trong mục "Được chia sẻ với tôi", chuột phải vào `RS-LiDAR` → **Sắp xếp → Thêm lối tắt** → chọn **Drive của tôi** (gốc). Tên lối tắt phải đúng là `RS-LiDAR`. Không cần sửa gì trong notebook.
- Dữ liệu mới sinh ra sẽ tính vào dung lượng Drive của chủ dự án.

### 1.2 Hugging Face token (chỉ cho E6)

FLUX.1-dev là mô hình gated: vào https://huggingface.co/black-forest-labs/FLUX.1-dev, bấm chấp nhận điều khoản, rồi tạo token đọc. Trên Colab: biểu tượng 🔑 (Secrets) → thêm `HF_TOKEN` và bật quyền truy cập cho notebook.

### 1.3 GPU trên Colab

| GPU | VRAM | Dùng cho |
|---|---|---|
| **A100** (40 GB) | 40 GB | Mọi thứ. **Bắt buộc** cho E6 (FLUX) và cho các bước Phase 1 (sinh hoặc chấm lại pool lookahead: decode 50 ảnh cùng lúc) |
| **L4** | 24 GB | Phase 2 SD 1.5, chấm GenEval, phân tích. Chậm hơn A100 khoảng 2–3 lần (ước lượng, chưa đo) |
| T4 | 16 GB | Không khuyến nghị |

Trong bảng ở mục 2 và 3: **"L4 được"** nghĩa là chạy được trên L4 nếu không phải sinh lại pool; **"A100"** nghĩa là phải dùng A100.

---

## 2. Bảng tổng quan

| | Tên thí nghiệm | Câu hỏi reviewer mà nó trả lời | Vị trí trong bài | GPU | Thời gian A100 (553 prompt) | Ưu tiên |
|---|---|---|---|---|---|---|
| **E1** | Nhiệt độ λ và cắt reward | "Smoothing có chỉ là hạ λ hoặc cắt reward ngoại lai không?" | Section 6, bảng/hình mới "Is smoothing a lower temperature?" | **L4 được** | ~17 h (7 run); ~3 h bản thử 100 prompt | 1 |
| **E2** | Flip rate và chất lượng ứng viên thắng | "Smoothing có thật sự đổi ứng viên được chọn không, và đổi có đúng không?" | Section 6.3 "Smoothing changes which candidate wins" (thay ví dụ đơn lẻ hiện có) | **L4 được** | ~0.5–1 h | 1 |
| **E3** | Cùng ngân sách reward | "RS gọi reward nhiều hơn 4 lần, thắng là đương nhiên?" | Section 6, gộp vào bảng E1 hoặc đoạn "Cost" | **A100** (Phase 1) + L4 được (Phase 2) | ~6 h | 2 |
| **E4** | Reward rời rạc (GenEval 0/1) | "Phương pháp có làm được gì mà cách khác không làm được?" | Section 6, mục riêng + contribution mới | **L4 được** | ~1.5 h (100 prompt); ~8 h (553) | 2 |
| **E5** | Nhiều reward cùng lúc (IR + HPS) | "Chỉ một reward thì hẹp" | Appendix | **A100** (Phase 1) + L4 được (Phase 2) | ~6 h | 3 |
| **E6** | Flow matching (FLUX.1-dev) | "Có dùng được cho mô hình rectified flow không?" | Section 6, hàng FLUX trong Table 1 hoặc bảng nhỏ; gỡ Limitation (vi) | **A100 bắt buộc** | ~25 h | 3 |

**Thứ tự ưu tiên khi thiếu GPU:** E2 > E1 > E3 > E4 > E6 > E5.

---

## 3. Chi tiết từng thí nghiệm

Tất cả notebook E1–E5 có cùng cấu trúc: 7 cell đầu là setup (kiểm tra GPU, mount Drive, clone repo, cài thư viện), mục 4 là cấu hình chung (giống Table 1), các mục sau là thí nghiệm. **Cấu hình chuẩn đã đặt sẵn**; chỉ đổi những gì ghi trong mục "Chỉ được đổi" của từng thí nghiệm.

Cấu hình chung của E1–E5 (khớp Table 1, đã khoá trong notebook):

| Tham số | Giá trị |
|---|---|
| Backbone | SD 1.5 (`runwayml/stable-diffusion-v1-5`) |
| Lookahead (Phase 1) | DPM-Solver 5 bước, K = 50 ứng viên, seed 100 |
| Target (Phase 2) | DDPM 100 bước (η = 1), 4 ảnh/prompt, seed 100, guidance scale s = 12.5, guidance khi t > 200 |
| LiDAR | λ = 5000 (mốc) |
| RS-LiDAR | σ = 1.0, M = 4, nhiễu **I.I.D.** (không CRN) |
| Prompt | 553 prompt GenEval (`PROMPT_MODE = "ALL"`) hoặc 100 prompt đại diện (`"100_REPRESENTATIVE"`) |
| Đánh giá | ImageReward, CLIP, HPSv2.1, Aesthetic (tự động) + GenEval Mask2Former Swin-S (cell GenEval trong notebook) |

---

### E1 — Smoothing có chỉ là hạ nhiệt độ không?

📓 **Notebook:** [E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb](E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Đây là phản biện nguy hiểm nhất. Làm trơn kéo các điểm reward lại gần nhau, trông giống như hạ nhiệt độ λ của softmax LiDAR, hoặc như cắt bớt các điểm reward ngoại lai. Nếu LiDAR ở λ thích hợp, hoặc LiDAR với reward bị cắt, đạt được bằng RS-LiDAR, thì đóng góp của bài chỉ còn là "chỉnh siêu tham số". Lý thuyết (Prop 2, mục 4.3) nói hai thứ này khác nhau: hạ λ không đổi thứ tự ứng viên, còn làm trơn thì có. E1 kiểm chứng điều đó bằng thực nghiệm.

**Thiết kế.** Cùng pool 50 ứng viên của Table 1, chạy Phase 2 với:
- LiDAR ở λ ∈ {10, 30, 100, 300} (vùng chuyển từ "chọn gần như argmax" sang "gần như không dẫn hướng"; λ ≥ ~1000 thì trọng số đã dồn hết vào một ứng viên nên giống hệt λ = 5000);
- RS-LiDAR ở λ ∈ {30, 300}, để so từng cặp ở cùng λ;
- LiDAR-Clip: reward mỗi prompt bị cắt ở phân vị 90% (5 ứng viên đứng đầu hoà nhau), λ = 5000;
- mốc LiDAR và RS-LiDAR λ = 5000 của Table 1 (notebook tự tìm trên Drive, không thấy thì tự chạy).

**Vị trí trong bài.** Section 6, mục mới "Is smoothing a lower temperature?": một bảng hoặc hình IR/GenEval theo λ, có đường ngang RS-LiDAR λ = 5000 và điểm LiDAR-Clip. Claim nếu đạt: *"No temperature of LiDAR, and no clipping of its scores, reaches RS-LiDAR."*

**GPU.** L4 được (Phase 2 SD 1.5 dùng ~9 GB). Nếu mục 5 báo phải chấm lại pool RS (xem dưới) thì bước đó cần A100.

**Cách chạy.**
1. Lần đầu: đặt `PROMPT_MODE = "100_REPRESENTATIVE"` ở mục 4 (~3 h A100).
2. Chạy mục 5, xem bảng ESS (số ứng viên thực sự được dùng theo λ). Dải λ đúng là dải phủ từ ESS ≈ 1 tới ESS ≈ 5–15. Nếu không, sửa `LAMBDAS_LIDAR` ở mục 6.
3. Chạy hết. Khi kết quả bản thử tốt: đổi `PROMPT_MODE = "ALL"`, chạy lại (run đã có tự bỏ qua).

**Chỉ được đổi:** `PROMPT_MODE`, `LAMBDAS_LIDAR`, `LAMBDAS_RS`.

**Kết quả gửi lại:** `MyDrive/RS-LiDAR/extra_experiments/E1_temperature_clipping/E1_summary.csv`.

**Đạt khi:** RS-LiDAR λ = 5000 cao hơn LiDAR ở **mọi** λ và cao hơn LiDAR-Clip, trên cả IR và GenEval (cột `Δ… [95% CI]` có dấu `*`).

**Nếu không đạt:**
- LiDAR ở λ nhỏ bắt kịp RS-LiDAR → bỏ claim "beyond temperature"; dồn trọng tâm sang E4 (với reward 0/1, hạ λ không thể thay thế làm trơn).
- Chỉ LiDAR-Clip bắt kịp → viết rằng làm trơn là một cách loại ngoại lai có nguyên lý, không cần chọn ngưỡng; chạy thêm Clip ở q ∈ {0.8, 0.95}.

---

### E2 — Smoothing đổi ứng viên thắng ở đâu, và có đổi đúng không?

📓 **Notebook:** [E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb](E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb)

**Mục đích.** Kiểm chứng trực tiếp cơ chế của bài. Lý thuyết nói reward model có những "spike" (điểm cao giả do mẫu pixel nhỏ), và làm trơn loại chúng khỏi thứ hạng. Với λ = 5000, LiDAR gần như chọn đúng một ứng viên điểm cao nhất, nên câu hỏi đo được là: **làm trơn đổi ứng viên thắng ở bao nhiêu prompt, và ứng viên mới có tốt hơn trên các metric không dùng để tối ưu không?** Hạ λ không bao giờ đổi ứng viên thắng, nên đây là đối chứng sạch nhất.

**Thiết kế.** Dùng hai pool của Table 1 (cùng 50 latents, một bên chấm bằng R, một bên bằng R̄σ):
1. **Flip rate:** tỷ lệ prompt mà argmax R ≠ argmax R̄σ (chỉ CPU).
2. **Độ mong manh:** ứng viên thắng theo R có hiệu R − R̄σ lớn bất thường không, so với mức nền hoán vị (để trừ hiệu ứng "winner's curse").
3. **Chất lượng:** trên các prompt bị flip, decode hai ứng viên thắng, chấm CLIP, HPS, Aesthetic, GenEval.
4. **Lợi ích tập trung ở đâu:** chênh lệch RS-LiDAR − LiDAR của Table 1 trên nhóm prompt bị flip và nhóm không flip.

**Vị trí trong bài.** Section 6.3, thay đoạn "Smoothing can change which candidate wins" (hiện chỉ có một ví dụ); đưa tiêu đề về dạng khẳng định. Đồng thời xác nhận dự đoán ở 4.6. Claim nếu đạt: *"Smoothing changes the selected candidate on X% of prompts; on these prompts the new winner is better on every held-out metric."*

**GPU.** L4 được. Nếu mục 5 báo phải chấm lại pool thì bước đó cần A100 (~0.6 h).

**Cách chạy.** Chạy từ trên xuống, `PROMPT_MODE = "ALL"`. Ở mục 5, đọc dòng in ra:
- "✅ Pool RS dùng chung latents với pool LiDAR" → tốt.
- "⚠️ … KHÔNG dùng chung latents" → notebook tự chấm lại pool (thư mục `..._rescore`). Báo lại cho chủ dự án, vì khi đó câu "LiDAR and RS-LiDAR share the same candidates" ở appendix C.3 cần xem lại.

**Chỉ được đổi:** không đổi gì.

**Kết quả gửi lại:** `E2_per_prompt.csv`, `E2_flipped_winner_quality.csv`, `E2_gain_by_flip.csv` trong `MyDrive/RS-LiDAR/extra_experiments/E2_flip_rate/`.

**Đạt khi:** flip rate ≥ 15%; trên prompt bị flip, ứng viên thắng theo R̄σ tốt hơn trên HPS **và** GenEval (CI 95% không chứa 0); chênh lệch Table 1 lớn hơn ở nhóm flip.

**Nếu không đạt:** bỏ câu chuyện "spike" khỏi Intro, giữ Prop 2 như một điều kiện đủ; giải thích lợi ích bằng Prop 3 (giảm nhiễu khi xếp hạng).

---

### E3 — So sánh cùng số lần gọi reward

📓 **Notebook:** [E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb)

**Mục đích.** RS-LiDAR gọi reward K × M = 50 × 4 = 200 lần, LiDAR chỉ 50 lần. Reviewer sẽ hỏi: nếu cho LiDAR cùng 200 lần gọi reward, tức 200 ứng viên, thì LiDAR có bắt kịp không?

**Thiết kế.** Sinh thêm 3 pool 50 ứng viên (seed 200, 300, 400), ghép với pool seed 100 của Table 1 thành K = 100 và K = 200 (pool nhỏ nằm trong pool lớn). Chạy Phase 2 LiDAR λ = 5000 với `top_k` = K. So với RS-LiDAR K = 50, M = 4.

**Vị trí trong bài.** Gộp vào bảng E1 hoặc đoạn "Smoothing costs runtime, not memory" ở 6.1. Claim nếu đạt: *"At equal reward evaluations, RS-LiDAR (K = 50, M = 4) outperforms LiDAR with K = 200, which also needs 4× more lookahead sampling."*

**GPU.** Mục 5 (sinh 3 pool mới) cần **A100**: Phase 1 decode 50 ảnh cùng lúc. Mục 7 (Phase 2) L4 được. Có thể chạy mục 5 trên A100, ngắt, rồi mở lại trên L4 chạy từ mục 6.

**Chỉ được đổi:** `PROMPT_MODE`.

**Kết quả gửi lại:** `E3_summary.csv`.

**Nếu không đạt** (LiDAR K = 200 ≥ RS-LiDAR): ghi trung thực vào bài; claim còn lại là "RS-LiDAR đạt chất lượng tương đương với Phase 1 rẻ hơn 4× và không tăng VRAM".

---

### E4 — Reward rời rạc: dẫn hướng bằng bộ chấm GenEval

📓 **Notebook:** [E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Cho thấy một năng lực mà phương pháp khác không có. Reward ở đây là **0 hoặc 1**: "ảnh có đúng prompt theo GenEval không" (detector Mask2Former + phân loại màu CLIP, giống hệt bộ chấm).
- Gradient guidance không dùng được vì reward không khả vi.
- LiDAR với reward 0/1 chỉ lọc đúng/sai; nếu không ứng viên nào đúng thì không có tín hiệu; mọi λ đều cho các ứng viên đúng hoà nhau.
- RS biến 0/1 thành R̄σ = "xác suất ảnh vẫn đúng khi bị nhiễu": một tín hiệu liên tục đo "đúng chắc chắn tới đâu".

Đây là **đòn bẩy lớn nhất để lên 7**: nó đổi câu chuyện của bài từ "cải thiện nhỏ cho LiDAR" thành "mở ra một lớp reward mới".

**Thiết kế.** Dùng lại 50 latents seed 100 của Table 1. Chấm 0/1 cho từng ứng viên và R̄σ với M = 8 nhiễu, tạo hai pool dẫn xuất, chạy Phase 2 LiDAR λ = 5000 trên mỗi pool. So thêm với LiDAR / RS-LiDAR dùng ImageReward của Table 1.

**Vị trí trong bài.** Section 6, mục riêng "Discrete verifiers" và một contribution mới; gỡ Limitation (iv) "single reward". **Bắt buộc ghi rõ:** trong thí nghiệm này GenEval là metric được tối ưu, không còn held-out; held-out là IR, HPS, CLIP, Aesthetic.

**GPU.** L4 được (VAE decode theo lô 10, Mask2Former theo lô 16).

**Cách chạy.**
1. Mặc định `PROMPT_MODE = "100_REPRESENTATIVE"` (~1.5 h A100).
2. **Mục 6 (hiệu chỉnh σ) bắt buộc chạy trước.** Mask2Former xem ảnh ở 800 px, không thu nhỏ như ImageReward, nên σ = 1 sẽ phá hỏng ảnh. Chọn σ lớn nhất mà cột "R̄σ trên ứng viên đúng" còn ≥ 0.5 và cột "số mức R̄σ khác nhau" ≥ 4. Sửa `SIGMA_D` ở mục 7 (mặc định 0.1).
3. Khi bản 100 prompt đạt: `PROMPT_MODE = "ALL"` (~8 h A100).

**Chỉ được đổi:** `PROMPT_MODE`, `SIGMA_D` (theo bảng hiệu chỉnh). Giữ `M_D = 8`.

**Kết quả gửi lại:** bảng hiệu chỉnh σ (chụp màn hình), `E4_summary_sig{σ}_M8.csv`.

**Đạt khi:** RS-LiDAR (R̄σ) tăng GenEval ≥ +0.03 so với LiDAR (0/1), CI không chứa 0, và IR/HPS không giảm có ý nghĩa.

**Nếu không đạt:** thử σ ∈ {0.03, 0.05} với M = 16; nếu vẫn không, đưa vào appendix hoặc bỏ.

---

### E5 — Nhiều reward cùng lúc (IR + HPS)

📓 **Notebook:** [E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb](E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Khi dẫn hướng bằng tổng hai reward, ứng viên có spike ở **một trong hai** reward cũng có thể thắng, nên rủi ro spike tăng. Kiểm tra làm trơn có giúp nhiều hơn trường hợp một reward không.

**Thiết kế.** Chấm thêm HPS cho 50 latents seed 100 (pool thô và pool làm trơn σ = 1, M = 4, cùng nhiễu cho cả hai reward). Reward kết hợp: r = s_IR·(½·IR/s_IR + ½·HPS/s_HPS), giữ thang đo của IR để λ = 5000 vẫn có nghĩa. Phase 2 LiDAR λ = 5000.

**Vị trí trong bài.** Appendix (mô tả, không thành contribution). Chỉ đưa vào main text nếu kết quả rất rõ.

**GPU.** Mục 5 (chấm lại pool với IR + HPS) cần **A100**; Phase 2 L4 được.

**Chỉ được đổi:** `PROMPT_MODE`, `W_IR` (mặc định 0.5).

**Kết quả gửi lại:** `E5_summary_w0.5.csv`.

**Chỉ chạy khi:** đã xong E1–E4 và E6 mà vẫn còn GPU.

---

### E6 — Flow matching (FLUX.1-dev)

📓 **Notebook** (thư mục [E6_flow_matching/](E6_flow_matching/)):

| Thứ tự | Notebook | Vai trò | Colab |
|---|---|---|---|
| 1 | [FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) | LiDAR: sinh pool 50 ứng viên + Phase 2 | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) |
| 2 | [FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) | **RS-FlowLiDAR chính**: chấm lại pool của bước 1 bằng R̄σ + Phase 2 | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) |
| 3 | [FLUX_Vanilla_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) | Mốc không dẫn hướng. **Bỏ qua nếu Drive đã có run Vanilla 553 prompt** | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) |
| 4 (tuỳ chọn) | [FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) | Ablation σ = 1.0 trên cùng pool | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) |

**Mục đích.** Bài hiện chỉ có SD 1.5 và SDXL (diffusion), và Limitation (vi) tự nhận chưa thử rectified flow. E6 cho thấy phương pháp chuyển sang FLUX được.

**Cấu hình chuẩn (đã đặt sẵn, không đổi):** FLUX.1-dev; lookahead Hyper-SD 8 bước, n = 50 ứng viên sinh theo lô 5 (`LOOKAHEAD_BATCH = 5`, vừa A100 40 GB); s_flow = 3.0; cửa sổ dẫn hướng t ∈ [1.0, 0.95]; λ = 5000; 28 bước; 1 ảnh/prompt; seed theo prompt (`PER_PROMPT_SEED = True`) ở cả hai phase; RS chấm lại đúng pool LiDAR (`REUSE_LIDAR_POOL = True`). Cấu hình RS chính σ = 0.25, M = 4 chuyển nguyên từ SDXL, không chỉnh trên GenEval.

**Vị trí trong bài.** Hàng FLUX trong Table 1 (hoặc một bảng nhỏ ở Section 6); gỡ Limitation (vi). Claim nếu đạt: *"RS-TTA transfers to rectified-flow models without re-tuning."*

**GPU.** **A100 bắt buộc** (trọng số FLUX + T5 đã ~24 GB). Cần `HF_TOKEN` (mục 1.2).

**Cách chạy.** Chạy notebook 1 tới hết, **rồi** mới chạy notebook 2 (notebook 2 dừng nếu pool của notebook 1 chưa đủ). Phase 1 của notebook 1 là phần nặng nhất (~10–14 h). Nhờ seed theo prompt và hậu tố `_pps` ở tên thư mục, bị ngắt thì chạy lại là tiếp tục đúng chỗ.

**Chỉ được đổi:** `PROMPT_PRESET` (`"stratified_100"` để thử trước, `"all"` để lấy số).

**Kết quả gửi lại:** đường dẫn hai thư mục kết quả `..._pps` trong `MyDrive/RS-LiDAR/results/`.

**Nếu RS ≈ LiDAR:** đã dự kiến (σ = 0.25 ở 1024² là nhiễu yếu, λ = 5000 chọn gần argmax). Viết trung thực "transfers without degradation"; dùng ablation σ = 1.0 nếu có.

---

## 4. Lịch và phân công

Gợi ý: **Người A** là chủ dự án (đã có pool Table 1 trên Drive). **Người B** là người cộng tác.

| Tuần | Người A | Người B | Cổng quyết định |
|---|---|---|---|
| 1 (16–22/11) | **E2** (L4/A100, ~1 h) → **E1 bản 100 prompt** (~3 h A100) | **E6 notebook 1** (A100) với `PROMPT_PRESET = "stratified_100"` để kiểm tra chạy được | **G1:** E1 và E2 đạt? |
| 2–4 (23/11–13/12) | **E1 đủ 553** (L4 được, ~17 h A100 / ~40 h L4) | **E3** (A100 mục 5, L4 mục 7) → **E4 bản 100 prompt** (L4) | **G2:** E4 đạt? |
| 5–7 (14/12–3/1) | **E4 đủ 553** (nếu G2 đạt) | **E6 đủ 553**: notebook 1 → 2 (→ 3 nếu cần) | |
| 8–10 (4/1–22/1) | Viết Section 6, cắt 8 trang, soát; đăng ký abstract trước **14/1** | E5 nếu còn GPU; kiểm tra lại số liệu | |

**Ước lượng tổng:** ~65 A100-giờ. FLUX chiếm khoảng 40%.

**Sau mỗi cổng:** gửi file CSV cho chủ dự án để đưa số vào bài.

---

## 5. Quy tắc chung khi chạy

1. **Không chạy cùng một run ở hai máy cùng lúc.** Hai người chia nhau theo bảng phân công; mỗi run ghi vào thư mục riêng trên Drive chung.
2. **E1–E5: để mỗi run chạy một mạch.** Các run này so từng cặp với run Table 1 (seed toàn cục). Nếu Colab ngắt giữa chừng một run Phase 2, xoá thư mục `Target_samples/<tên run>` đó và chạy lại từ đầu, không dùng `--resume`.
3. **E6:** bị ngắt thì cứ chạy lại; seed theo prompt giữ đúng nhiễu.
4. **Không sửa cấu hình ngoài mục "Chỉ được đổi".** Muốn thử cấu hình khác thì báo trước, vì số trong bài phải khớp một cấu hình duy nhất.
5. **Lỗi khi chạy:** gửi traceback đầy đủ kèm tên notebook và số cell.
6. **Kết quả xấu cũng phải báo.** Mỗi thí nghiệm có phương án "nếu không đạt"; số xấu không được bỏ đi âm thầm.
