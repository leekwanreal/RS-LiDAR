# Kế hoạch thí nghiệm bổ sung: phân công và hướng dẫn chạy

> Thư mục tạm cho cộng tác. Sẽ bị loại khỏi bản code nộp kèm bài.

**Hạn chót (giờ AoE):** đăng ký abstract **14/1/2027**, nộp bài **22/1/2027**.

**Mục tiêu:** cả ba reviewer chấm 7/10. Phần viết và lý thuyết đã xong; điểm bây giờ phụ thuộc vào 6 thí nghiệm E1–E6, mỗi thí nghiệm trả lời một câu hỏi reviewer chắc chắn sẽ hỏi.

**Ý tưởng của bài trong một câu:** thay vì chấm mỗi ứng viên bằng **một lần** gọi reward model, ta chấm bằng **trung bình reward trên M bản nhiễu Gauss** của ảnh (điểm làm trơn R̄σ). Áp dụng cho LiDAR (RS-LiDAR) và SMC (RS-SMC).

**Phân công:** **Hiếu** làm T1, T3, T5, T7, T9 ([mục 4A](#4a-việc-của-hiếu)); **Quân** làm T2, T4, T6, T8 ([mục 4B](#4b-việc-của-quân)). Mỗi người chỉ cần đọc phần của mình, làm theo thứ tự.

---

## Mục lục

0. [Đọc file này thế nào](#0-đọc-file-này-thế-nào)
1. [Bảng phân công](#1-bảng-phân-công)
2. [Chuẩn bị một lần](#2-chuẩn-bị-một-lần)
3. [Cách chạy một notebook](#3-cách-chạy-một-notebook)
4. Từng việc: [4A. Hiếu](#4a-việc-của-hiếu) · [4B. Quân](#4b-việc-của-quân)
5. [Cổng quyết định sau E4](#5-cổng-quyết-định)
6. [Quy tắc chung](#6-quy-tắc-chung)
7. [Tham khảo: chi tiết thí nghiệm E1–E6](#7-tham-khảo-chi-tiết-thí-nghiệm)

---

## 0. Đọc file này thế nào

1. Làm [mục 2](#2-chuẩn-bị-một-lần) **một lần** trước khi chạy bất cứ thứ gì.
2. Đọc phần việc của mình: Hiếu ở [mục 4A](#4a-việc-của-hiếu), Quân ở [mục 4B](#4b-việc-của-quân).
3. Mỗi việc viết theo **cùng một khuôn**:

| Phần | Nội dung |
|---|---|
| Thẻ việc | 👤 người chạy · 🧪 thí nghiệm · 📓 notebook (có nút **Mở trên Colab**) · 🖥️ GPU · ⏱️ thời gian · 🔗 việc phải xong trước · 📤 thứ phải nộp |
| ▶ Các bước | Làm lần lượt từng bước. Bước có **⏸** là chỗ phải **dừng lại đọc kết quả** (và thường chụp màn hình) trước khi chạy tiếp |
| ⚠️ Lưu ý | Lỗi hay gặp, cách xử lý, và cách **chạy thử 100 prompt** nếu muốn xem xu hướng trước (tuỳ chọn) |
| 📊 Đọc kết quả và làm gì tiếp | Bảng 3 cột. Dòng **✅** = kết quả tốt, **⚠️** = lẫn lộn, **❌** = xấu; mỗi dòng nói nghĩa là gì và việc tiếp theo |

---

## 1. Bảng phân công

Thời gian là cho bản đủ trên A100, ước lượng; L4 chậm hơn khoảng 2–3 lần (chưa đo).

### 👤 Hiếu: T1 → T3 → T5 → T7 → T9

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T1**](#t1) | E2 · flip rate | L4 | ~1 h | — |
| [**T3**](#t3) | E1 · nhiệt độ λ và cắt reward | L4/A100 | ~17 h | nên sau T1 |
| [**T5**](#t5) | E3 · cùng số lần gọi reward | A100 → L4 | ~6 h | — |
| [**T7**](#t7) | E4 dự phòng **hoặc** E5 | L4 | ~6–8 h | T6 + [cổng E4](#5-cổng-quyết-định) |
| [**T9**](#t9) | GenEval cho FLUX (⛔ chưa có notebook) | — | — | T2, T4 |

### 👤 Quân: T2 → T4 → T6 → T8

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T2**](#t2) | E6 · FLUX notebook 1 (LiDAR) | **A100** | ~10–14 h | `HF_TOKEN` |
| [**T4**](#t4) | E6 · FLUX notebook 2 (RS σ = 0.25) | **A100** | ~8–10 h | T2 |
| [**T6**](#t6) | E4 · reward rời rạc | L4 | ~5 phút + 4–6 h | — |
| [**T8**](#t8) | E6 · FLUX notebook 3 (Vanilla), 4 (σ = 1.0) | **A100** | ~5–18 h | T4 |

Việc của hai người chỉ phụ thuộc nhau ở hai chỗ: **T7** (Hiếu) chờ **T6** (Quân) xong và qua [cổng E4](#5-cổng-quyết-định); **T9** (Hiếu) chờ **T2, T4** (Quân).

**Khi thiếu GPU, ưu tiên:** E2 > E1 > E3 > E4 > E6 > E5.

**Ngân sách Colab (ước lượng, kiểm tra lại bảng giá trên trang Colab):** A100 tốn khoảng 11–15 compute unit mỗi giờ, L4 khoảng 5. Phần của Hiếu (chủ yếu L4) cần khoảng 350–400 unit; phần của Quân (FLUX bắt buộc A100) khoảng 500–650 unit. Colab Pro có 100 unit/tháng, nên mỗi người cần **Colab Pro+ (500 unit/tháng) hoặc mua thêm unit** cho khoảng 2 tháng tới hạn nộp.

---

## 2. Chuẩn bị một lần

### 2.1 Dùng chung Google Drive (bắt buộc)

Mọi notebook đọc và ghi vào `MyDrive/RS-LiDAR/`. E1–E5 **dùng lại pool lookahead của Table 1**, nằm trong thư mục `RS-LiDAR` của chủ dự án.

1. **Người giữ thư mục gốc:** chia sẻ thư mục `RS-LiDAR` trên Google Drive cho người còn lại, quyền **Editor**.
2. **Người được chia sẻ:** vào **Được chia sẻ với tôi** → chuột phải `RS-LiDAR` → **Sắp xếp → Thêm lối tắt** → chọn **Drive của tôi** (gốc). Tên lối tắt phải đúng là `RS-LiDAR`.
3. **Làm bước 2 TRƯỚC khi mở bất kỳ notebook nào.** Nếu chạy notebook trước, nó tự tạo một thư mục `RS-LiDAR` rỗng; khi đó lối tắt bị trùng tên và notebook ghi vào thư mục rỗng.
4. Kiểm tra sau khi notebook nối Drive (mục 2 của notebook): `os.listdir("/content/drive/MyDrive/RS-LiDAR/Lookahead_samples")` phải thấy các pool của Table 1.

**Dung lượng:** file do ai tạo thì **tính vào Drive của người đó**. Ước lượng:

| Người | Tạo ra | Dung lượng | Gói Drive cần |
|---|---|---|---|
| Quân | 3 pool FLUX (notebook 1, 2, 4; pool RS lưu một bản latents riêng, ~14 GB mỗi pool) + ảnh 1024² của các run | ~45 GB | 100 GB |
| Hiếu | Ảnh của ~10 run SD 1.5 (E1, E3, E5), 3 pool mới của E3, 2 pool chấm lại của E5 | ~15–20 GB | 100 GB (gói miễn phí 15 GB sát giới hạn) |

Đầy Drive thì notebook lỗi khi ghi file giữa chừng; kiểm tra dung lượng trống trước mỗi việc lớn.

### 2.2 Hugging Face token (chỉ người chạy E6)

1. Vào https://huggingface.co/black-forest-labs/FLUX.1-dev, bấm chấp nhận điều khoản.
2. Tạo token đọc (Settings → Access Tokens).
3. Trên Colab: biểu tượng 🔑 (**Secrets**) → thêm `HF_TOKEN` → bật quyền truy cập cho notebook.

### 2.3 Chọn GPU

| GPU | Dùng cho |
|---|---|
| **A100** (40 GB, bật **High-RAM**) | Mọi thứ. **Bắt buộc** cho E6 (FLUX) |
| **L4** (24 GB) | Mọi thí nghiệm SD 1.5 (E1–E5), kể cả Phase 1. Chậm hơn A100 khoảng 2–3 lần |
| Tesla T4 (16 GB) | Không dùng |

### 2.4 Môi trường

- Cell cài đặt của E1–E5 chép nguyên từ notebook Table 1 (đã chạy được trên Colab). Thông báo xung đột phụ thuộc của pip về TensorFlow/google-cloud là bình thường.
- Ngay sau cài đặt có **cell kiểm tra môi trường**: phải in ✅ cho `hpsv2`, `clip`, `open_clip`, `ImageReward`, `timm`, `google.protobuf`. Có ❌ thì: **Runtime → Restart session**, chạy lại từ đầu.
- Code chấm điểm **báo lỗi** khi HPS/CLIP không chạy được (không âm thầm ghi 0). Cell GenEval dừng nếu `open_clip` không nạp được.
- FLUX (E6): xem lưu ý ở [T2](#t2).

---

## 3. Cách chạy một notebook

Áp dụng cho mọi việc; phần riêng của từng việc nằm ở mục 4.

1. Bấm **Mở trên Colab** ở thẻ việc. Notebook luôn lấy code mới nhất từ GitHub khi chạy, nên không cần tải gì về.
2. **Runtime → Change runtime type**: chọn GPU theo thẻ việc. Với A100, bật **High-RAM**.
3. Sửa đúng các biến được ghi trong **▶ Các bước** (không sửa gì khác).
4. Chạy theo đúng cách ghi trong bước:
   - **Run all**: chạy hết.
   - **Run before** (chọn một cell → Runtime → Run before): chạy mọi cell phía trên cell đó, rồi dừng để đọc kết quả.
   - **Run after** (chọn một cell → Runtime → Run after): chạy từ cell đó trở xuống.
5. Khi Colab hỏi quyền truy cập Google Drive: cho phép, bằng tài khoản có lối tắt `RS-LiDAR`.
6. **Bị ngắt kết nối:** mở lại notebook, chạy lại theo bước 4. Phần đã xong tự bỏ qua. Ngoại lệ: run Phase 2 dùng seed toàn cục (E1, E3) bị ngắt giữa chừng thì notebook in đường dẫn và dừng; **xoá thư mục đó trên Drive** rồi chạy lại.
7. **Lỗi:** gửi nhóm **traceback đầy đủ** + tên notebook + số mục.
8. **Nộp kết quả:** file nằm ở `MyDrive/RS-LiDAR/extra_experiments/<tên thí nghiệm>/` (E6: `MyDrive/RS-LiDAR/results/`). Gửi tên file hoặc ảnh chụp vào nhóm chat.

---

<a id="4-từng-việc"></a>
## 4. Từng việc

<a id="4a-việc-của-hiếu"></a>
## 4A. Việc của Hiếu (T1, T3, T5, T7, T9)

<a id="t1"></a>
### T1 · Hiếu · E2: smoothing đổi ứng viên thắng ở bao nhiêu prompt

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E2 (xem [tham khảo](#ref-e2)) |
| 📓 Notebook | [E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb](E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb) |
| 🖥️ GPU | **L4** (A100 cũng được) |
| ⏱️ Thời gian | ~0.5–1 h; thêm ~0.6 h (A100) / ~1.5 h (L4) nếu phải chấm lại pool RS |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp output mục 5 và mục 6; `E2_per_prompt.csv`, `E2_flipped_winner_quality.csv`, `E2_gain_by_flip.csv` |

**▶ Các bước**

1. Mở notebook, chọn GPU (xem [cách chạy một notebook](#3-cách-chạy-một-notebook)).
2. Không sửa biến nào. **Runtime → Run all**.
3. ⏸ **Mục 5:** chụp dòng in ra: `✅ Pool RS dùng chung latents …` hoặc `⚠️ … KHÔNG dùng chung latents`.
4. ⏸ **Mục 6:** chụp các dòng `Flip rate`, `Spearman`, bảng phân vị gap và dòng `Wilcoxon`.
5. Chờ mục 7–8 chạy xong (giải mã ảnh, chấm điểm).
6. Nộp ảnh chụp và 3 file CSV trong `MyDrive/RS-LiDAR/extra_experiments/E2_flip_rate/`.

**⚠️ Lưu ý**

- Nếu mục 5 báo ⚠️, notebook tự chấm lại pool (thư mục `..._rescore`). Việc này chỉ tốn một lần: E1 (T3) dùng lại. **Báo nhóm ngay**, vì khi đó câu "LiDAR and RS-LiDAR share the same candidates" ở appendix C.3 phải xem lại.
- Mục 8 cần hai run Table 1 trên Drive. Nếu nó in `⚠️ Chưa có đủ hai run Table 1` thì bỏ qua, chạy lại riêng mục 8 sau T3.
- Nên chạy T1 **trước** T3 (cùng người), để hai notebook không chấm lại pool RS cùng lúc.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Flip rate ≥ 15%, và ở mục 7 ứng viên thắng theo R̄σ tốt hơn trên HPS **và** GenEval (CI 95% không chứa 0) | Có bằng chứng trực tiếp cho cơ chế của bài | Gửi kết quả; số này vào Section 6.3 |
| ⚠️ Flip rate ≥ 15% nhưng chất lượng ở mục 7 không khác (CI chứa 0) | Smoothing có đổi lựa chọn, nhưng chưa chứng minh được lựa chọn mới tốt hơn | Trong bài chỉ giữ flip rate, bỏ chữ "better" |
| ❌ Flip rate < 15% | Spike hiếm; câu chuyện "spike" yếu | Báo nhóm. Bỏ câu chuyện spike khỏi Intro, giải thích lợi ích bằng Prop 3. E1 và E4 trở nên quan trọng hơn |

---

<a id="t3"></a>
### T3 · Hiếu · E1: smoothing có chỉ là hạ nhiệt độ λ hoặc cắt reward ngoại lai không

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E1 (xem [tham khảo](#ref-e1)) |
| 📓 Notebook | [E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb](E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** hoặc A100 |
| ⏱️ Thời gian | ~17 h A100 (~40 h L4), chia nhiều phiên Colab |
| 🔗 Cần xong trước | Nên sau T1 |
| 📤 Nộp lại | Ảnh chụp bảng ESS (mục 5); `E1_summary.csv` |

**▶ Các bước**

1. Mở notebook, chọn GPU. Mục 4 giữ `PROMPT_MODE = "ALL"`.
2. Chọn cell đầu tiên của mục 7 → menu **Runtime → Run before** (chạy tới hết mục 6).
3. ⏸ **Mục 5:** chụp bảng ESS. Kiểm tra: ở λ = 1, ESS phải lớn (≥ 10); ở λ = 30, ESS gần 1.
4. Nếu ESS ở λ = 1 đã < 10: sửa `LAMBDAS_LIDAR` ở mục 6 sang dải nhỏ hơn (ví dụ `[0.3, 1, 3, 10]`). Nếu ESS ở λ = 30 vẫn > 2: dời dải lên (ví dụ `[3, 10, 30, 100]`). Sửa `LAMBDAS_RS` theo cùng hướng, chạy lại cell mục 6, báo nhóm.
5. Chọn cell đầu tiên của mục 7 → **Runtime → Run after** (7 run Phase 2, rồi GenEval và bảng).
6. Phiên Colab hết giờ: mở lại, **Run all**. Run đã xong tự bỏ qua.
7. Nộp ảnh bảng ESS và `E1_summary.csv`.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** muốn xem xu hướng trước thì đặt `PROMPT_MODE = "100_REPRESENTATIVE"` (100 prompt, ~3 h A100). Bản thử nằm ở thư mục riêng nên không được dùng lại cho bản đủ; nhớ đổi lại giá trị mặc định khi chạy đủ.
- Run của E1 dùng seed toàn cục để ghép cặp với Table 1. Bị ngắt giữa một run thì notebook **từ chối chạy tiếp** run đó và in đường dẫn: xoá thư mục đó trên Drive rồi chạy lại.
- Mỗi run ~2.4 h A100 (~5–6 h L4). Trên L4, nên bắt đầu một phiên khi còn đủ thời gian cho trọn một run.
- Đọc cột `Δ… [95% CI]`: dấu `*` nghĩa là khác biệt có ý nghĩa.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ RS-LiDAR λ = 5000 cao hơn mọi LiDAR λ và LiDAR-Clip trên IR và GenEval, có `*` | Smoothing không phải hạ nhiệt độ hay cắt ngoại lai | Gửi CSV; viết vào Section 6 |
| ⚠️ Hơn nhưng không có `*` ở GenEval | Chỉ chắc chắn trên IR | Viết claim cho IR; GenEval ghi là xu hướng |
| ⚠️ Chỉ LiDAR-Clip bắt kịp | Smoothing giống một cách loại ngoại lai, nhưng không cần chọn ngưỡng | Báo nhóm để thêm Clip ở ngưỡng 0.8 và 0.95 (cần sửa code) |
| ❌ LiDAR ở một λ nhỏ bắt kịp hoặc vượt | Có thể chỉ là hiệu ứng nhiệt độ | Bỏ claim "beyond temperature"; E4 (T6) lên ưu tiên số 1 |

---

<a id="t5"></a>
### T5 · Hiếu · E3: so sánh cùng số lần gọi reward

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E3 (xem [tham khảo](#ref-e3)) |
| 📓 Notebook | [E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **A100** cho mục 5 (sinh 3 pool mới); L4 được cho phần sau |
| ⏱️ Thời gian | ~6 h A100 |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | `E3_summary.csv` |

**▶ Các bước**

1. Mở notebook, chọn A100. Không sửa gì (`PROMPT_MODE = "ALL"`).
2. **Runtime → Run all**.
3. Mục 5 sinh 3 pool 50 ứng viên (seed 200, 300, 400) vào `Lookahead_samples/`; pool nào đã có thì bỏ qua.
4. Muốn đổi sang L4 sau mục 5: ở máy L4 chạy lại **từ đầu tới hết mục 5** (mục 5 tự bỏ qua vì pool đã có), rồi chạy tiếp mục 6–8.
5. Nộp `E3_summary.csv`.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** muốn xem xu hướng trước thì đặt `PROMPT_MODE = "100_REPRESENTATIVE"` (100 prompt, khoảng 1/5 thời gian). Bản thử nằm ở thư mục riêng nên không được dùng lại cho bản đủ; nhớ đổi lại giá trị mặc định khi chạy đủ.
- Pool ghép K = 100, K = 200 (mục 6) nằm trên đĩa Colab, không phải Drive, nên đổi máy thì phải tạo lại (vài phút).
- Run LiDAR K = 100/200 dùng seed toàn cục: bị ngắt giữa run thì xoá thư mục được in ra rồi chạy lại.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ RS-LiDAR (K = 50, M = 4) ≥ LiDAR K = 200 trên IR và GenEval | Thắng ngay cả khi cùng số lần gọi reward | Claim: "At equal reward evaluations, RS-LiDAR outperforms LiDAR with K = 200" |
| ⚠️ Ngang nhau | Cùng chất lượng nhưng Phase 1 rẻ hơn 4 lần | Claim về chi phí: chất lượng tương đương, Phase 1 rẻ hơn 4×, không tăng VRAM |
| ❌ LiDAR K = 200 hơn rõ | Một phần lợi ích đến từ việc gọi reward nhiều hơn | Ghi trung thực; chỉ giữ claim về chi phí |

---

<a id="t7"></a>
### T7 · Hiếu · E4 dự phòng hoặc E5 (theo cổng E4)

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E5 (xem [tham khảo](#ref-e5)) |
| 📓 Notebook | [E4](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb)<br>[E5](E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | ~6–8 h |
| 🔗 Cần xong trước | [T6](#t6) xong, đã qua [cổng E4](#5-cổng-quyết-định) |
| 📤 Nộp lại | CSV của nhánh đã chạy |

**▶ Các bước**

1. **Nếu cổng E4 chọn "dự phòng":** mở E4, giữ nguyên `E4_TAGS`, ở mục 9 đặt `SIGMA_D = 0.05` (không đạt thì lần sau 0.03) và `M_D = 16`, **Run all** (kết quả chạy thử đã lưu nên không chạy lại).
2. **Nếu cổng E4 chọn "E5":** mở E5, không sửa gì, **Run all**.
3. E5 — ⏸ **Mục 6:** đọc dòng `tỷ trọng hiệu dụng`: ở pool thô phải gần IR 0.50 | HPS 0.50.
4. Nộp CSV.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** muốn xem xu hướng trước thì đặt `PROMPT_MODE = "100_REPRESENTATIVE"` (100 prompt, chỉ cho E5). Bản thử nằm ở thư mục riêng nên không được dùng lại cho bản đủ; nhớ đổi lại giá trị mặc định khi chạy đủ.
- E5 mục 5 chấm HPS rất chậm (thư viện nạp lại checkpoint mỗi lần gọi). Bị ngắt thì chạy lại, phần đã chấm được giữ.
- E5 không dùng chế độ `--use_pareto` có sẵn trong code (xem tham khảo E5).
- Nếu cổng E4 chọn "dự phòng" và vẫn còn GPU sau đó, chạy thêm E5.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ E4 dự phòng: RS hơn LiDAR (0/1), có `*` | σ nhỏ hơn mới hợp với Mask2Former | E4 vào Section 6 |
| ❌ E4 dự phòng vẫn không hơn |  | E4 vào appendix hoặc bỏ |
| ✅ E5: RS-LiDAR (IR + HPS) hơn LiDAR (IR + HPS) trên cả IR và HPS | Phương pháp dùng được với nhiều reward | Một đoạn trong appendix |
| ⚠️/❌ E5 không hơn |  | Ghi trung thực trong appendix hoặc bỏ |

---

<a id="t9"></a>
### T9 · Hiếu · Chấm GenEval cho các run FLUX

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | ⛔ **Chưa có notebook** |
| 🖥️ GPU | A100 hoặc L4 |
| ⏱️ Thời gian | ? |
| 🔗 Cần xong trước | [T2](#t2), [T4](#t4) xong |
| 📤 Nộp lại | — |

**▶ Các bước**

1. Notebook FLUX chỉ chấm IR và CLIP, không chấm GenEval. Khi T2 và T4 xong, báo nhóm để viết notebook chấm GenEval cho các thư mục `..._pps` (cùng bộ chấm với E1–E5).

**⚠️ Lưu ý**

- Không tự chấm bằng script khác: số phải dùng đúng bộ chấm của Table 1.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| — | — | — |

---

<a id="4b-việc-của-quân"></a>
## 4B. Việc của Quân (T2, T4, T6, T8)

<a id="t2"></a>
### T2 · Quân · E6 notebook 1: LiDAR trên FLUX (sinh pool 50 ứng viên + Phase 2)

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [FLUX LiDAR (notebook 1)](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** (bắt buộc) |
| ⏱️ Thời gian | ~10–14 h Phase 1 + Phase 2 (ước lượng), chia nhiều phiên Colab |
| 🔗 Cần xong trước | Có `HF_TOKEN` ([mục 2.2](#22-hugging-face-token-chỉ-người-chạy-e6)) |
| 📤 Nộp lại | Đường dẫn thư mục kết quả `..._pps` trong `MyDrive/RS-LiDAR/results/`; ảnh chụp `pip list`; ảnh chụp bảng IR mục 8 và lưới ảnh mục 9 |

**▶ Các bước**

1. Mở notebook, chọn **A100** và bật **High-RAM**.
2. Chạy mục 1–3 (kiểm tra GPU, nối Drive, cài thư viện).
3. Thêm một cell ngay dưới mục 3, chạy `!pip list | grep -i -E "transformers|diffusers|tokenizers|peft|torchao|accelerate"`, chụp kết quả.
4. Chạy mục 4 (đăng nhập Hugging Face bằng `HF_TOKEN`).
5. Mục 5: giữ `PROMPT_PRESET = "all"`. Không đổi gì khác.
6. Chọn cell đầu mục 5 → **Runtime → Run after**.
7. ⏸ **Mục 6.1:** phải báo latents có kích thước 16 × 128 × 128.
8. Bị ngắt: mở lại, chạy lại từ đầu; notebook tiếp đúng chỗ (seed theo prompt).
9. Khi xong mục 7–9: xem lưới ảnh mục 9, nộp đường dẫn thư mục, ảnh `pip list`, bảng IR.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** muốn xem xu hướng trước thì đặt `PROMPT_PRESET = "stratified_100"` (100 prompt, ~3–4 h). Bản thử nằm ở thư mục riêng nên không được dùng lại cho bản đủ; nhớ đổi lại giá trị mặc định khi chạy đủ.
- pip có thể báo lỗi `tokenizers` (Python 3.13 của Colab không có bản dựng sẵn). Nếu notebook vẫn chạy tiếp được thì bỏ qua, nhưng **phải** chụp `pip list` lần chạy đầu và kiểm tra các lần sau ra đúng phiên bản đó. Gặp lỗi import lạ: **Runtime → Change runtime type → Runtime version 2026.07** (Python 3.12), chạy lại từ đầu.
- Dung lượng: khoảng 14 GB cho pool này, **tính vào Drive của người chạy** (xem [mục 2.1](#21-dùng-chung-google-drive-bắt-buộc)).

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Chạy hết 553 prompt, ảnh mục 9 rõ | Có pool cho notebook 2 | Làm T4 |
| ⚠️ Chạy được nhưng chậm hơn ước lượng khoảng 2 lần | E6 sẽ tốn nhiều GPU hơn dự kiến | Báo nhóm để cân lại việc |
| ❌ Lỗi môi trường hoặc hết bộ nhớ | Chưa chạy được FLUX | Gửi traceback và `pip list`; thử runtime 2026.07. Vẫn lỗi thì tạm gác E6, làm T6 trước |

---

<a id="t4"></a>
### T4 · Quân · E6 notebook 2: RS-FlowLiDAR trên cùng pool

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [FLUX RS σ = 0.25 (notebook 2)](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** |
| ⏱️ Thời gian | ~8–10 h (ước lượng) |
| 🔗 Cần xong trước | [T2](#t2) xong (cùng `PROMPT_PRESET`) |
| 📤 Nộp lại | Đường dẫn thư mục `..._pps`; ảnh chụp bảng IR mục 8 |

**▶ Các bước**

1. Mở notebook, chọn A100 + High-RAM.
2. Chạy mục 1–4 như T2. Cài đặt phải ra đúng các phiên bản trong ảnh `pip list` của T2; khác thì dừng và báo nhóm.
3. Mục 5: giữ `PROMPT_PRESET` giống T2 (`"all"`) và `REUSE_LIDAR_POOL = True`.
4. Chọn cell đầu mục 5 → **Runtime → Run after**.
5. ⏸ **Mục 6:** phải thấy notebook nạp latents từ pool của notebook 1 rồi chấm lại, **không** sinh ứng viên mới. Nếu thấy thanh tiến trình sinh ảnh Hyper-SD: dừng và báo nhóm.
6. Chờ xong; so bảng IR với T2 và nộp.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** muốn xem xu hướng trước thì đặt `PROMPT_PRESET = "stratified_100"` (100 prompt, ~1–2 h; chỉ chạy được khi T2 cũng đã chạy thử). Bản thử nằm ở thư mục riêng nên không được dùng lại cho bản đủ; nhớ đổi lại giá trị mặc định khi chạy đủ.
- Notebook 2 dừng nếu pool của notebook 1 chưa đủ prompt.
- Không đổi σ hay M: cấu hình chính σ = 0.25, M = 4 chuyển nguyên từ SDXL.
- Phase 1 RS tự tính lại những prompt mà cache cũ không dùng đúng pool LiDAR; việc này là bình thường.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ IR của RS ≥ LiDAR (T2) | Phương pháp chuyển sang FLUX được | Hàng FLUX cho Table 1 hoặc bảng nhỏ ở Section 6; gỡ Limitation (vi) |
| ⚠️ RS ≈ LiDAR (chênh < 0.01) | Đã dự kiến: σ = 0.25 ở ảnh 1024² là nhiễu yếu | Claim "transfers without degradation"; chạy notebook 4 (σ = 1.0) ở T8 |
| ❌ RS thấp hơn rõ (> 0.03 IR) | Cấu hình chưa hợp với FLUX | Báo nhóm; ghi trung thực, cân nhắc notebook 4 |

---

<a id="t6"></a>
### T6 · Quân · E4: reward rời rạc, dẫn hướng bằng verifier kiểu GenEval

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E4 (xem [tham khảo](#ref-e4)) |
| 📓 Notebook | [E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | chạy thử theo task ~5 phút; phần còn lại ~4–6 h A100 (L4 chậm hơn khoảng 2 lần) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | `E4_pilot.csv`, `E4_pilot_grid.png`, ảnh chụp bảng hiệu chỉnh σ (mục 8), `E4_summary_*.csv`, `E4_geneval_by_task_*.csv` |

**▶ Các bước**

1. Mở notebook, chọn L4. Không sửa gì ở mục 4 (`PROMPT_MODE = "ALL"`).
2. Chọn cell đầu tiên của mục 7 → menu **Runtime → Run before** (chạy tới hết mục 6: chạy thử theo task, có sẵn trong notebook).
3. ⏸ **Mục 6:** gửi nhóm bảng chạy thử, lưới ảnh và dòng `Gợi ý E4_TAGS`.
4. Mục 7: sửa `E4_TAGS` theo gợi ý (cả nhóm thống nhất). **Chốt ở đây, không đổi sau khi xem kết quả.**
5. Chọn cell đầu tiên của mục 9 → menu **Runtime → Run before** (chạy mục 7–8: chọn task, hiệu chỉnh σ).
6. ⏸ **Mục 8:** chọn σ lớn nhất mà cột `R̄σ trên ứng viên đúng` ≥ 0.5 **và** cột `số mức R̄σ khác nhau` ≥ 4. Sửa `SIGMA_D` ở mục 9.
7. Chọn cell đầu tiên của mục 9 → **Runtime → Run after** (chấm toàn bộ, Phase 2, bảng).
8. Nộp các file trong `MyDrive/RS-LiDAR/extra_experiments/E4_discrete_reward/`.

**⚠️ Lưu ý**

- Quy tắc giữ task: ít nhất 50% prompt có ít nhất một ứng viên đúng. single_object luôn bị loại (đã chạm trần).
- Mục 9 lưu điểm từng prompt lên Drive; bị ngắt thì chạy lại là tiếp đúng chỗ. Phase 2 dùng seed theo prompt (`_pps`) nên cũng tiếp được.
- Chỉ được đổi `E4_TAGS` và `SIGMA_D`. Giữ `M_D = 8`.
- Trong bài phải ghi: GenEval ở thí nghiệm này là metric được tối ưu; held-out là IR, HPS, CLIP, Aesthetic.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Chạy thử: ≥ 2 task qua. Kết quả cuối: GenEval của RS-LiDAR (R̄σ) cao hơn LiDAR (0/1) ≥ +0.03, có `*`; IR/HPS không giảm có ý nghĩa | Có năng lực mới: dẫn hướng bằng verifier rời rạc | Contribution mới cho Section 6. Xem [cổng E4](#5-cổng-quyết-định): T7 = E5 |
| ⚠️ Hơn nhưng không có `*` | Có xu hướng, chưa chắc | Cổng E4: T7 = E4 dự phòng |
| ❌ Chạy thử: chỉ 0–1 task qua | Ứng viên lookahead quá mờ, verifier không có tín hiệu | **Dừng ngay sau mục 6**, không tốn GPU. Báo nhóm: cần reward "mềm" (độ tin cậy detector), phải viết thêm code |
| ❌ Kết quả cuối: RS không hơn | σ chưa hợp | Cổng E4: T7 = E4 dự phòng |

---

<a id="t8"></a>
### T8 · Quân · E6 notebook 3 (Vanilla) và notebook 4 (σ = 1.0)

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [FLUX Vanilla (notebook 3)](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb)<br>[FLUX RS σ = 1.0 (notebook 4)](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) · [**Mở trên Colab**](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** |
| ⏱️ Thời gian | notebook 3: ~5–8 h; notebook 4: ~8–10 h (ước lượng) |
| 🔗 Cần xong trước | [T4](#t4) xong |
| 📤 Nộp lại | Đường dẫn thư mục kết quả |

**▶ Các bước**

1. **Notebook 3:** kiểm tra trước trong `MyDrive/RS-LiDAR/results/` đã có run Vanilla 553 prompt chưa. **Có thì bỏ qua.** Chưa có: giữ `PROMPT_PRESET = "all"`, **Run all**.
2. **Notebook 4:** chạy khi T4 cho RS ≈ LiDAR hoặc RS thấp hơn, hoặc khi còn GPU. Làm như T4 nhưng mở notebook 4.

**⚠️ Lưu ý**

- Vanilla không dùng seed theo prompt, nên chỉ là mốc tham khảo, không ghép cặp với LiDAR/RS.
- pip có thể báo lỗi `tokenizers` (Python 3.13 của Colab không có bản dựng sẵn). Nếu notebook vẫn chạy tiếp được thì bỏ qua, nhưng **phải** chụp `pip list` lần chạy đầu và kiểm tra các lần sau ra đúng phiên bản đó. Gặp lỗi import lạ: **Runtime → Change runtime type → Runtime version 2026.07** (Python 3.12), chạy lại từ đầu.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Có Vanilla và σ = 1.0 |  | Đủ hàng cho bảng FLUX; σ = 1.0 là ablation |
| ❌ Lỗi |  | Gửi traceback |

---

<a id="5-cổng-quyết-định"></a>
## 5. Cổng quyết định

### 🔀 Cổng E4 · sau T6 · cả hai cùng xem

| Tình huống | Quyết định |
|---|---|
| E4 ✅ | E4 thành contribution mới ở Section 6. T7 = **E5** |
| E4 ⚠️ hoặc kết quả cuối ❌ | T7 = **E4 dự phòng** (σ nhỏ hơn, M = 16) |
| Chạy thử E4 ❌ (0–1 task có tín hiệu) | Dừng E4; cần viết code reward "mềm" trước khi chạy lại. T7 = E5 |

Các kết quả khác (E1, E2, E3, E6) không chặn việc của ai; hướng xử lý nằm trong bảng 📊 của từng việc.

---

## 6. Quy tắc chung

1. **Không chạy cùng một run ở hai máy cùng lúc.** Mỗi người chỉ chạy việc của mình.
2. **Không sửa cấu hình ngoài những gì ghi trong ▶ Các bước.** Muốn thử cấu hình khác thì báo trước, vì số trong bài phải khớp một cấu hình duy nhất.
3. **E1, E3: để mỗi run Phase 2 chạy một mạch** (seed toàn cục, để ghép cặp với Table 1). E4, E5, E6 dùng seed theo prompt nên chạy tiếp được sau khi bị ngắt.
4. **Kết quả xấu cũng phải báo.** Mỗi việc có dòng ❌ và hướng xử lý; số xấu không được bỏ đi âm thầm.
5. **Lỗi:** traceback đầy đủ + tên notebook + số mục.

---

<a id="7-tham-khảo-chi-tiết-thí-nghiệm"></a>
## 7. Tham khảo: chi tiết thí nghiệm

Phần này giải thích **vì sao** chạy từng thí nghiệm, nó vào đâu trong bài, và notebook làm gì ở từng mục. Không cần đọc để chạy.

| | Tên | Câu hỏi reviewer mà nó trả lời | Vị trí trong bài |
|---|---|---|---|
| [E1](#ref-e1) | Nhiệt độ λ và cắt reward | "Smoothing có chỉ là hạ λ hoặc cắt reward ngoại lai không?" | Section 6, bảng/hình mới |
| [E2](#ref-e2) | Flip rate | "Smoothing có thật sự đổi ứng viên được chọn không, và đổi có đúng không?" | Section 6.3 |
| [E3](#ref-e3) | Cùng ngân sách reward | "RS gọi reward nhiều hơn 4 lần, thắng là đương nhiên?" | Section 6, đoạn "Cost" |
| [E4](#ref-e4) | Reward rời rạc | "Phương pháp làm được gì mà cách khác không làm được?" | Section 6, mục riêng + contribution mới |
| [E5](#ref-e5) | Nhiều reward (IR + HPS) | "Chỉ một reward thì hẹp" | Appendix |
| [E6](#ref-e6) | Flow matching (FLUX.1-dev) | "Có dùng được cho rectified flow không?" | Hàng FLUX hoặc bảng nhỏ ở Section 6; gỡ Limitation (vi) |

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

<a id="ref-e1"></a>
### E1 — Smoothing có chỉ là hạ nhiệt độ không?

📓 **Notebook:** [E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb](E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Đây là phản biện nguy hiểm nhất. Làm trơn kéo các điểm reward lại gần nhau, trông giống như hạ nhiệt độ λ của softmax LiDAR, hoặc như cắt bớt các điểm reward ngoại lai. Nếu LiDAR ở λ thích hợp, hoặc LiDAR với reward bị cắt, đạt được bằng RS-LiDAR, thì đóng góp của bài chỉ còn là "chỉnh siêu tham số". Lý thuyết (Prop 2, mục 4.3) nói hai thứ này khác nhau: hạ λ không đổi thứ tự ứng viên, còn làm trơn thì có. E1 kiểm chứng điều đó bằng thực nghiệm.

**Thiết kế.** Cùng pool 50 ứng viên của Table 1, chạy Phase 2 với:
- LiDAR ở λ ∈ {1, 3, 10, 30} (vùng chuyển từ "gần như không dẫn hướng" sang "chọn gần như argmax"). Độ lệch chuẩn ImageReward giữa các ứng viên của một prompt khoảng 0.8, nên từ λ ≈ 30 trở lên trọng số đã dồn gần hết vào một ứng viên, giống hệt λ = 5000. Bảng ESS ở mục 5 xác nhận lại trên dữ liệu thật;
- RS-LiDAR ở λ ∈ {3, 10}, để so từng cặp ở cùng λ;
- LiDAR-Clip: reward mỗi prompt bị cắt ở phân vị 90% (5 ứng viên đứng đầu hoà nhau), λ = 5000;
- mốc LiDAR và RS-LiDAR λ = 5000 của Table 1 (notebook tự tìm trên Drive, không thấy thì tự chạy).

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung (bảng cấu hình chung ở trên) | Đặt `PROMPT_MODE` |
| 5 | Tìm hai pool ứng viên của Table 1 trên Drive (dừng nếu không thấy). Nếu pool RS không cùng 50 ứng viên với pool LiDAR, chấm lại chúng bằng reward làm trơn. In bảng ESS: với mỗi λ, LiDAR thực sự dùng trung bình bao nhiêu ứng viên | Đọc bảng ESS, chụp màn hình |
| 6 | Tạo pool LiDAR-Clip: cùng ứng viên, nhưng điểm của mỗi prompt bị cắt ở phân vị 90% (chỉ ghi lại file điểm, vài giây). In danh sách 7 run sẽ chạy | Sửa `LAMBDAS_LIDAR` nếu bảng ESS cho thấy dải λ chưa đúng |
| 7 | Sinh ảnh cuối (Phase 2) cho 7 run, lần lượt. Mỗi run ghi vào `MyDrive/RS-LiDAR/Target_samples/E1_...`; run đã xong tự bỏ qua | Chờ. ~2.4 h mỗi run với 553 prompt trên A100 |
| 8 | Chấm GenEval (Mask2Former) cho mọi run, kể cả hai run Table 1 | Chờ |
| 9 | Lập bảng: mỗi run so với LiDAR λ = 5000 trên IR, CLIP, HPS, Aesthetic, GenEval, kèm khoảng tin cậy 95% | Gửi `E1_summary.csv` |

**Vị trí trong bài.** Section 6, mục mới "Is smoothing a lower temperature?": một bảng hoặc hình IR/GenEval theo λ, có đường ngang RS-LiDAR λ = 5000 và điểm LiDAR-Clip. Claim nếu đạt: *"No temperature of LiDAR, and no clipping of its scores, reaches RS-LiDAR."*

<a id="ref-e2"></a>
### E2 — Smoothing đổi ứng viên thắng ở đâu, và có đổi đúng không?

📓 **Notebook:** [E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb](E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb)

**Mục đích.** Kiểm chứng trực tiếp cơ chế của bài. Lý thuyết nói reward model có những "spike" (điểm cao giả do mẫu pixel nhỏ), và làm trơn loại chúng khỏi thứ hạng. Với λ = 5000, LiDAR gần như chọn đúng một ứng viên điểm cao nhất, nên câu hỏi đo được là: **làm trơn đổi ứng viên thắng ở bao nhiêu prompt, và ứng viên mới có tốt hơn trên các metric không dùng để tối ưu không?** Hạ λ không bao giờ đổi ứng viên thắng, nên đây là đối chứng sạch nhất.

**Thiết kế.** Dùng hai pool của Table 1 (cùng 50 latents, một bên chấm bằng R, một bên bằng R̄σ):
1. **Flip rate:** tỷ lệ prompt mà argmax R ≠ argmax R̄σ (chỉ CPU).
2. **Độ mong manh:** ứng viên thắng theo R có hiệu R − R̄σ lớn bất thường không, so với mức nền hoán vị (để trừ hiệu ứng "winner's curse").
3. **Chất lượng:** trên các prompt bị flip, decode hai ứng viên thắng, chấm CLIP, HPS, Aesthetic, GenEval.
4. **Lợi ích tập trung ở đâu:** chênh lệch RS-LiDAR − LiDAR của Table 1 trên nhóm prompt bị flip và nhóm không flip.

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung | Không đổi gì |
| 5 | Tìm hai pool của Table 1 và kiểm tra chúng có cùng 50 ứng viên không. Nếu không, chấm lại 50 ứng viên của pool LiDAR bằng reward làm trơn (~0.6 h A100) | Đọc dòng ✅ hoặc ⚠️ |
| 6 (chỉ CPU) | Với từng prompt: ghép 50 ứng viên của hai pool; so ứng viên thắng theo R với ứng viên thắng theo R̄σ; tính tỷ lệ prompt bị đổi (flip rate), hạng của ứng viên thắng cũ khi xếp theo R̄σ, và mức "mong manh" so với nền ngẫu nhiên | Ghi lại flip rate |
| 7 (GPU) | Ở các prompt bị đổi: giải mã hai ứng viên thắng thành ảnh 512², lưu vào `winners/`, chấm IR, CLIP, HPS, Aesthetic, GenEval; lập bảng chênh lệch (thắng theo R̄σ − thắng theo R) | Chờ |
| 8 | Chia prompt thành nhóm bị đổi và không bị đổi; trong mỗi nhóm tính chênh lệch RS-LiDAR − LiDAR của hai run Table 1 | Gửi 3 file CSV |

**Vị trí trong bài.** Section 6.3, thay đoạn "Smoothing can change which candidate wins" (hiện chỉ có một ví dụ); đưa tiêu đề về dạng khẳng định. Đồng thời xác nhận dự đoán ở 4.6. Claim nếu đạt: *"Smoothing changes the selected candidate on X% of prompts; on these prompts the new winner is better on every held-out metric."*

<a id="ref-e3"></a>
### E3 — So sánh cùng số lần gọi reward

📓 **Notebook:** [E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb)

**Mục đích.** RS-LiDAR gọi reward K × M = 50 × 4 = 200 lần, LiDAR chỉ 50 lần. Reviewer sẽ hỏi: nếu cho LiDAR cùng 200 lần gọi reward, tức 200 ứng viên, thì LiDAR có bắt kịp không?

**Thiết kế.** Sinh thêm 3 pool 50 ứng viên (seed 200, 300, 400), ghép với pool seed 100 của Table 1 thành K = 100 và K = 200 (pool nhỏ nằm trong pool lớn). Chạy Phase 2 LiDAR λ = 5000 với `top_k` = K. So với RS-LiDAR K = 50, M = 4.

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung | Không đổi gì |
| 5 | Phase 1: sinh 3 bộ 50 ứng viên mới (seed 200, 300, 400), cùng cấu hình pool Table 1, ghi vào `MyDrive/RS-LiDAR/Lookahead_samples/`. Bộ nào đã có thì bỏ qua | Chờ (phần nặng nhất, nên dùng A100) |
| 6 | Ghép: bộ 100 = seed 100 + 200; bộ 200 = cả bốn. Xếp theo điểm IR. Lưu trên **đĩa Colab**, không phải Drive, nên đổi máy thì phải chạy lại mục 5–6 | Chờ (vài phút) |
| 7 | Phase 2 LiDAR λ = 5000 trên bộ 100 và bộ 200 ứng viên. Tìm hai run Table 1 làm mốc (không thấy thì tự chạy) | Chờ |
| 8 | Chấm GenEval; bảng so mọi run với RS-LiDAR 50 ứng viên, M = 4 | Gửi `E3_summary.csv` |

**Vị trí trong bài.** Gộp vào bảng E1 hoặc đoạn "Smoothing costs runtime, not memory" ở 6.1. Claim nếu đạt: *"At equal reward evaluations, RS-LiDAR (K = 50, M = 4) outperforms LiDAR with K = 200, which also needs 4× more lookahead sampling."*

<a id="ref-e4"></a>
### E4 — Reward rời rạc: dẫn hướng bằng bộ chấm GenEval

📓 **Notebook:** [E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Cho thấy một năng lực mà phương pháp khác không có. Reward ở đây là **0 hoặc 1**: "ảnh có đúng prompt theo GenEval không" (detector Mask2Former + phân loại màu CLIP, giống hệt bộ chấm).
- Gradient guidance không dùng được vì reward không khả vi.
- LiDAR với reward 0/1 chỉ lọc đúng/sai; nếu không ứng viên nào đúng thì không có tín hiệu; mọi λ đều cho các ứng viên đúng hoà nhau.
- RS biến 0/1 thành R̄σ = "xác suất ảnh vẫn đúng khi bị nhiễu": một tín hiệu liên tục đo "đúng chắc chắn tới đâu".

Đây là **đòn bẩy lớn nhất để lên 7**: nó đổi câu chuyện của bài từ "cải thiện nhỏ cho LiDAR" thành "mở ra một lớp reward mới".

**Rủi ro chính và cách xử lý.** Ứng viên lookahead là ảnh DPM-Solver 5 bước, vật thường mờ. Với task khó (position, color_attr; SD 1.5 chạy đủ bước cũng chỉ đạt khoảng 0.04–0.06), gần như mọi ứng viên bị chấm 0, khi đó cả hai phương pháp đều mất tín hiệu. Vì vậy notebook **chạy thử theo task trước**, rồi chỉ giữ các task mà ứng viên lookahead còn được chấm đúng ở ít nhất 50% prompt. single_object luôn bị loại vì SD 1.5 đã đạt khoảng 0.97, không còn chỗ để cải thiện. Dự kiến giữ **two_object, colors, counting** (273 prompt); kết quả chạy thử quyết định.

**Thiết kế.** Dùng lại 50 latents seed 100 của Table 1, không sinh ứng viên mới. Reward là luật GenEval của từng task (two_object: phát hiện đủ hai vật; colors: phát hiện vật và CLIP chấm đúng màu; counting: đúng số lượng ở ngưỡng tin cậy 0.9), dùng cùng bộ chấm với cột GenEval của Table 1. Với mỗi prompt của các task đã chọn: chấm 0/1 cho 50 ứng viên, và R̄σ với M = 8 nhiễu; tạo hai pool dẫn xuất; chạy Phase 2 LiDAR λ = 5000 trên mỗi pool. So thêm với LiDAR / RS-LiDAR dùng ImageReward của Table 1 trên cùng các prompt.

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung (`PROMPT_MODE = "ALL"`) | Không đổi gì |
| 5 | Tìm pool Table 1; nạp VAE (để giải mã ứng viên thành ảnh) và bộ chấm GenEval (Mask2Former + CLIP) | Chờ |
| 6 | **Chạy thử** (~5 phút A100): giải mã 50 ứng viên của 10 prompt mỗi task (6 task), chấm đúng/sai. In bảng theo task: tỷ lệ ứng viên đúng, % prompt toàn 0 / lẫn đúng-sai / toàn 1, % prompt có ít nhất một ứng viên đúng; in "Gợi ý E4_TAGS"; vẽ lưới 8 ứng viên mỗi task (viền xanh = đúng, đỏ = sai). Lưu `E4_pilot.csv`, `E4_pilot_grid.png` | **Dừng ở đây.** Gửi bảng và lưới ảnh cho A |
| 7 | Lọc prompt theo `E4_TAGS`; ghi danh sách prompt lên Drive để Phase 2 dùng đúng tập này; cảnh báo nếu chọn task không qua chạy thử | Sửa `E4_TAGS` theo gợi ý (A duyệt). **Chốt ở đây, không đổi sau khi đã xem kết quả** |
| 8 | Hiệu chỉnh σ trên 4 prompt mỗi task đã chọn: thêm nhiễu σ = 0.05/0.1/0.25/0.5, chấm lại 4 lần; in bảng | Chọn σ theo quy tắc ở [T6](#t6) bước 6, sửa `SIGMA_D` ở mục 9 |
| 9 | Với mọi prompt đã chọn: giải mã 50 ứng viên, chấm 0/1, chấm thêm 8 bản nhiễu để được R̄σ. Lưu từng prompt lên Drive (`scores_sig…`), bị ngắt thì chạy tiếp được. In lại bảng toàn 0 / lẫn / toàn 1 theo task trên toàn bộ prompt | Chờ (phần nặng nhất) |
| 10 | Giải phóng VRAM; tạo 2 pool dẫn xuất (cùng ứng viên, điểm = 0/1 hoặc R̄σ); 2 run Phase 2 LiDAR λ = 5000, tên thư mục có hậu tố task (ví dụ `_to-cnt-col`) | Chờ |
| 11 | Chấm GenEval; bảng tổng so với LiDAR dùng điểm 0/1 (kèm hai run IR của Table 1), và bảng GenEval theo từng task | Gửi các file CSV |

**Vị trí trong bài.** Section 6, mục riêng "Discrete verifiers" và một contribution mới; gỡ Limitation (iv) "single reward". **Bắt buộc ghi rõ:** trong thí nghiệm này GenEval là metric được tối ưu, không còn held-out; held-out là IR, HPS, CLIP, Aesthetic.

<a id="ref-e5"></a>
### E5 — Nhiều reward cùng lúc (IR + HPS)

📓 **Notebook:** [E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb](E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Cho thấy RS-TTA dùng được nguyên vẹn cho mục tiêu gồm nhiều reward. Về lý thuyết: với tổng có trọng số và nhiễu dùng chung, làm trơn tổng bằng tổng các reward đã làm trơn, và Prop 2 vẫn đúng với J = J_IR + J_HPS. Lý thuyết **không** nói làm trơn giúp nhiều hơn khi có nhiều reward (mỗi spike bị nhân trọng số ½), nên nếu thấy điều đó thì chỉ được viết như một quan sát thực nghiệm.

**Thiết kế.** Chấm thêm HPS cho 50 latents seed 100 (pool thô và pool làm trơn σ = 1, M = 4, cùng nhiễu cho cả hai reward). Reward kết hợp: r = s_IR·(½·IR/s_IR + ½·HPS/s_HPS), với s là std **trong từng prompt** của reward thô (guidance chỉ so ứng viên trong cùng prompt); cùng hằng số dùng cho pool làm trơn để hai bên có cùng trọng số. Notebook in tỷ trọng hiệu dụng IR/HPS và dừng nếu HPS toàn 0. Phase 2 LiDAR λ = 5000, seed theo prompt (`_pps`).

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung | Không đổi gì |
| 5 | Chấm lại 50 ứng viên của Table 1 bằng IR + HPS, hai lần: điểm thường và điểm làm trơn (σ = 1, M = 4, cùng ảnh nhiễu cho hai reward). Không sinh ứng viên mới | Chờ (chậm vì HPS) |
| 6 | Tính std trong từng prompt; cộng IR và HPS thành một reward; in tỷ trọng hiệu dụng IR/HPS; tạo 2 pool dẫn xuất | Xem tỷ trọng in ra (nên gần 0.5/0.5 ở pool thô) |
| 7 | 2 run Phase 2: LiDAR và RS-LiDAR với reward kết hợp | Chờ |
| 8 | Chấm GenEval; bảng so với LiDAR (IR + HPS), kèm hai run chỉ IR của Table 1 | Gửi CSV |

**Không dùng** chế độ `--use_pareto` có sẵn trong code: điểm Pareto phụ thuộc cả quần thể ứng viên, ra đơn vị z-score (λ = 5000 không còn phù hợp), không có phiên bản không làm trơn để so, và nếu dùng cho SMC sẽ phá Prop 4. Phase 2 cũng chưa nhận được nó làm reward.

**Vị trí trong bài.** Appendix (mô tả, không thành contribution). Chỉ đưa vào main text nếu kết quả rất rõ.

<a id="ref-e6"></a>
### E6 — Flow matching (FLUX.1-dev)

📓 **Notebook** (thư mục [E6_flow_matching/](E6_flow_matching/)):

| Thứ tự | Notebook | Vai trò | Colab |
|---|---|---|---|
| 1 | [FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) | LiDAR: sinh pool 50 ứng viên + Phase 2 | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) |
| 2 | [FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) | **RS-FlowLiDAR chính**: chấm lại pool của bước 1 bằng R̄σ + Phase 2 | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) |
| 3 | [FLUX_Vanilla_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) | Mốc không dẫn hướng. **Bỏ qua nếu Drive đã có run Vanilla 553 prompt** | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) |
| 4 (tuỳ chọn) | [FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) | Ablation σ = 1.0 trên cùng pool | [Mở](https://colab.research.google.com/github/leekwanreal/RS-LiDAR/blob/main/notebooks_temp/extra_experiments/E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) |

**Mục đích.** Bài hiện chỉ có SD 1.5 và SDXL (diffusion), và Limitation (vi) tự nhận chưa thử rectified flow. E6 cho thấy phương pháp chuyển sang FLUX được.

**Notebook làm gì, từng mục** (notebook 1 và 2 cùng cấu trúc):

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Drive, cài thư viện, tải code | Sau mục 3 chạy lệnh `pip list` ở mục "Môi trường FLUX" bên dưới, ghi lại |
| 4 | Đăng nhập Hugging Face bằng `HF_TOKEN` | Cần làm [mục 2.2](#22-hugging-face-token-chỉ-người-chạy-e6) trước |
| 5 | Bảng cấu hình | Chỉ đổi `PROMPT_PRESET` |
| 6 | **Notebook 1:** sinh 50 ứng viên mỗi prompt bằng Hyper-SD 8 bước, chấm ImageReward. **Notebook 2:** không sinh mới, nạp đúng 50 ứng viên của notebook 1 và chấm bằng R̄σ (σ = 0.25, M = 4) | Chờ (notebook 1: ~10–14 h với 553 prompt) |
| 6.1 | Kiểm tra kích thước latents (16 × 128 × 128) | — |
| 6.2 | Hiệu chỉnh s_flow: **tắt sẵn**, không bật | — |
| 7 | Phase 2: FLUX 28 bước, dẫn hướng LiDAR bằng điểm của mục 6; chấm IR và CLIP | Chờ |
| 8–9 | Bảng IR, lưới ảnh mẫu | Gửi đường dẫn thư mục kết quả |

**GenEval chưa có trong notebook FLUX** (việc [T9](#t9)).

**Cấu hình chuẩn (đã đặt sẵn, không đổi):** FLUX.1-dev; lookahead Hyper-SD 8 bước, n = 50 ứng viên sinh theo lô 5 (`LOOKAHEAD_BATCH = 5`, vừa A100 40 GB); s_flow = 3.0; cửa sổ dẫn hướng t ∈ [1.0, 0.95]; λ = 5000; 28 bước; 1 ảnh/prompt; seed theo prompt (`PER_PROMPT_SEED = True`) ở cả hai phase; RS chấm lại đúng pool LiDAR (`REUSE_LIDAR_POOL = True`). Cấu hình RS chính σ = 0.25, M = 4 chuyển nguyên từ SDXL, không chỉnh trên GenEval.

**Vị trí trong bài.** Hàng FLUX trong Table 1 (hoặc một bảng nhỏ ở Section 6); gỡ Limitation (vi). Claim nếu đạt: *"RS-TTA transfers to rectified-flow models without re-tuning."*

**Môi trường FLUX (đọc trước khi chạy).** Cell cài đặt ghim `transformers>=4.43.3,<4.45`, nhưng các bản này cần `tokenizers 0.19`, không có bản dựng sẵn cho Python 3.13 (runtime mặc định của Colab). Khi đó pip lỗi mà cell không dừng, và notebook chạy bằng bản transformers/diffusers có sẵn của Colab. Đây có lẽ là cấu hình đã chạy được trong lần hiệu chỉnh s_flow trước, nhưng phiên bản không cố định. Việc cần làm: sau cell cài đặt, chạy `!pip list | grep -i -E "transformers|diffusers|tokenizers|peft|torchao|accelerate"` và **ghi lại** các phiên bản khi chạy thành công; nếu gặp lỗi lạ, đổi runtime sang **2026.07 (Python 3.12)** (Runtime → Change runtime type → Runtime version) để pip cài đúng bản đã ghim.

**Vanilla** không dùng seed theo prompt (giống run Vanilla 553 cũ), nên không ghép cặp nhiễu với LiDAR/RS; nó chỉ là mốc tham khảo.
