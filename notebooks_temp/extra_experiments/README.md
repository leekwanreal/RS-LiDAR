# Kế hoạch thí nghiệm bổ sung: phân công và hướng dẫn chạy

> Thư mục tạm cho cộng tác. Sẽ bị loại khỏi bản code nộp kèm bài.

**Hạn chót (giờ AoE):** đăng ký abstract **14/1/2027**, nộp bài **22/1/2027**.

**Mục tiêu:** cả ba reviewer chấm 7/10. Phần viết và lý thuyết đã xong; điểm bây giờ phụ thuộc vào 7 thí nghiệm E1–E7, mỗi thí nghiệm trả lời một câu hỏi reviewer chắc chắn sẽ hỏi.

**Ý tưởng của bài trong một câu:** thay vì chấm mỗi ứng viên bằng **một lần** gọi reward model, ta chấm bằng **trung bình reward trên M bản nhiễu Gauss** của ảnh (điểm làm trơn R̄σ). Áp dụng cho LiDAR (RS-LiDAR) và SMC (RS-SMC).

**Phân công:** **Hiếu** làm T1, T3, T5, T7 ([mục 4A](#4a-việc-của-hiếu)); **Quân** làm T2, T4, T6, T8, T9 ([mục 4B](#4b-việc-của-quân)); **tác giả** làm T10 ([mục 4C](#4c-việc-của-tác-giả)). Mỗi người chỉ cần đọc phần của mình, làm theo thứ tự.

---

## Mục lục

0. [Đọc file này thế nào](#0-đọc-file-này-thế-nào)
1. [Bảng phân công](#1-bảng-phân-công)
2. [Chuẩn bị một lần](#2-chuẩn-bị-một-lần)
3. [Cách chạy một notebook](#3-cách-chạy-một-notebook)
4. Từng việc: [4A. Hiếu](#4a-việc-của-hiếu) · [4B. Quân](#4b-việc-của-quân) · [4C. Tác giả](#4c-việc-của-tác-giả)
5. [Cổng quyết định sau E4](#5-cổng-quyết-định)
6. [Quy tắc chung](#6-quy-tắc-chung)
7. [Tham khảo: chi tiết thí nghiệm E1–E7](#7-tham-khảo-chi-tiết-thí-nghiệm)

---

## 0. Đọc file này thế nào

1. Làm [mục 2](#2-chuẩn-bị-một-lần) **một lần** trước khi chạy bất cứ thứ gì.
2. Đọc phần việc của mình: Hiếu ở [mục 4A](#4a-việc-của-hiếu), Quân ở [mục 4B](#4b-việc-của-quân).
3. Mỗi việc viết theo **cùng một khuôn**:

| Phần | Nội dung |
|---|---|
| Thẻ việc | 👤 người chạy · 🧪 thí nghiệm · 📓 notebook (tên file, bấm vào để mở file trong repo) · 🖥️ GPU · ⏱️ thời gian · 🔗 việc phải xong trước · 📤 thứ phải nộp |
| 🎯 Mục tiêu | Việc này trả lời câu hỏi gì |
| ▶ Các bước | Làm lần lượt, từng thao tác bấm. Bước có **⏸** là chỗ phải **dừng lại đọc kết quả** (và chụp màn hình) trước khi đi tiếp; bước đó ghi rõ phải thấy dòng nào |
| 🔧 Tham số | Biến nào giữ nguyên, biến nào được đổi và đổi theo quy tắc nào |
| 📦 Đầu ra | Notebook in ra gì, ghi file gì ở đâu, mỗi bảng/cột nghĩa là gì |
| 🌟 Kết quả lý tưởng | Kết quả tốt trông thế nào, kèm **mốc kiểm tra** để phát hiện chạy sai trước khi phân tích |
| ⚠️ Lưu ý | Lỗi hay gặp, cách xử lý, cách **chạy thử 100 prompt** (tuỳ chọn) |
| 📊 Đọc kết quả và làm gì tiếp | Bảng 3 cột. Dòng **✅** = kết quả tốt, **⚠️** = lẫn lộn, **❌** = xấu; mỗi dòng nói nghĩa là gì và việc tiếp theo |

---

## 1. Bảng phân công

Thời gian là cho bản đủ trên A100, ước lượng; L4 chậm hơn khoảng 2–3 lần (chưa đo).

**Bản Run all (17/11/2026).** Mọi notebook đã cấu hình sẵn: chỉ cần chọn GPU rồi **Runtime → Run all**. Notebook tự bỏ qua phần đã xong, tự chọn các tham số trước đây phải chọn tay (E4), và **tự ngắt máy** khi chạy xong (cell cuối, `AUTO_DISCONNECT = True`). Các bước ⏸ trong từng thẻ là để chụp màn hình output sau khi chạy, không phải dừng giữa chừng. Số run sinh ảnh cuối (Phase 2) của từng thí nghiệm:

| Thí nghiệm | Run Phase 2 mới | Ghi chú |
|---|---|---|
| E2 (T1) | 0 | Chỉ phân tích hai pool Table 1 và chấm ứng viên thắng |
| E1 (T3) | 4 | LiDAR λ = 100, 1000; LiDAR-Clip q = 0.8, 0.95 |
| E3 (T5) | 1 | LiDAR K = 200 (thêm 3 pool Phase 1: seed 200, 300, 400) |
| E4 (T6) | 2 | LiDAR (0/1) và RS-LiDAR (R̄σ); tự dừng sau chạy thử nếu dưới 2 task đạt |
| E4 dự phòng (T7) | 0 hoặc 2 | Chỉ khi cổng E4 yêu cầu |
| E5 | 0 | Bỏ (17/11) |
| E6 (T2, T4, T8, T9) | 2 hoặc 3 | FLUX LiDAR (kèm sinh pool), RS σ = 0.25; Vanilla chỉ khi Drive chưa có; σ = 1.0 bỏ; T9 chỉ chấm điểm |
| E7 (T10) | 2 (+ tối đa 1) | SMC N = 4, λ_FK = 3 và 30; thêm 1 run nếu đạt điều kiện ở thẻ T10 |

### 👤 Hiếu: T1 → T3 → T5 → T7

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T1**](#t1) | E2 · flip rate | L4 | ~1 h | — |
| [**T3**](#t3) | E1 · nhiệt độ λ và cắt reward | L4/A100 | ~6–10 h | nên sau T1 |
| [**T5**](#t5) | E3 · cùng số lần gọi reward (chỉ K = 200) | A100 → L4 | ~3–5 h | — |
| [**T7**](#t7) | E4 dự phòng, **chỉ khi** cổng E4 yêu cầu (E5 đã bỏ 17/11) | L4 | 0 hoặc ~4–6 h | T6 + [cổng E4](#5-cổng-quyết-định) |

### 👤 Quân: T2 → T4 → T6 → T8 → T9

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T2**](#t2) | E6 · FLUX notebook 1 (LiDAR) | **A100** | ~10–14 h | `HF_TOKEN` |
| [**T4**](#t4) | E6 · FLUX notebook 2 (RS σ = 0.25) | **A100** | ~8–10 h | T2 |
| [**T6**](#t6) | E4 · reward rời rạc | L4 | ~5 phút + 4–6 h | — |
| [**T8**](#t8) | E6 · FLUX notebook 3 (Vanilla; bỏ qua nếu Drive đã có). Notebook 4 (σ = 1.0) đã bỏ 17/11 | **A100** | 0 hoặc ~5–8 h | T4 |
| [**T9**](#t9) | E6 · chấm GenEval cho FLUX | L4 | chưa đo (ngắn, không sinh ảnh) | T2, T4 |

### 👤 Tác giả: T10

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T10**](#t10) | E7 · SMC ở nhiệt độ λ_FK khác | A100 (hoặc L4) | 2 run bắt buộc (2 notebook cấu hình sẵn) + tối đa 1 run có điều kiện, ước lượng 1–1.5 h/run A100 | — |

Việc của hai người chỉ phụ thuộc nhau ở một chỗ: **T7** (Hiếu) chờ **T6** (Quân) xong và qua [cổng E4](#5-cổng-quyết-định).

**Khi thiếu GPU, ưu tiên:** E2 > E1 > E3 > E7 > E4 > E6 > E5.

**Ngân sách Colab:** tốc độ tiêu thụ đo thực tế (16/11/2026): **L4 1.94 unit/giờ, A100 6.5 unit/giờ**. Phần của Hiếu (khoảng 80 giờ GPU, chủ yếu L4, một phần A100 cho E3) cần khoảng 180–220 unit; phần của Quân (khoảng 45 giờ A100 cho FLUX + khoảng 10 giờ L4 cho E4) khoảng 300–350 unit; phần của tác giả (T10, 2–3 run A100) khoảng 15–30 unit. Số giờ là ước lượng. Colab Pro có 100 unit/tháng; Pro+ có 500–600 unit/tháng (xem trang đăng ký), đủ cho phần của mỗi người trong một tháng. Tốc độ tiêu thụ xem ở Colab → biểu tượng RAM/Disk góc phải → **View resources**.

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

1. **Mở notebook từ file trên máy, không mở thẳng từ GitHub** (kiểu link `colab.research.google.com/github/...`). Mở từ GitHub thì khi chạy xong, phiên Colab thường không tự kết thúc và vẫn tiếp tục tốn compute unit. Cách mở đúng:
   - Lấy file `.ipynb` về máy: `git pull` trong bản clone của repo (thư mục `notebooks_temp/extra_experiments/`), hoặc bấm vào tên notebook ở thẻ việc để mở file trong repo rồi tải file đó về.
   - Trên Colab: **File → Upload notebook** → chọn file vừa tải.
   - Khi nhóm báo notebook có bản mới thì tải lại file. Code `.py` thì không cần: notebook tự lấy bản mới nhất từ GitHub khi chạy.
2. **Runtime → Change runtime type**: chọn GPU theo thẻ việc. Với A100, bật **High-RAM**.
3. Sửa đúng các biến được ghi trong **▶ Các bước** (không sửa gì khác).
4. Chạy theo đúng cách ghi trong bước:
   - **Run all**: chạy hết.
   - **Run before** (chọn một cell → Runtime → Run before): chạy mọi cell phía trên cell đó, rồi dừng để đọc kết quả.
   - **Run after** (chọn một cell → Runtime → Run after): chạy từ cell đó trở xuống.
5. Khi Colab hỏi quyền truy cập Google Drive: cho phép, bằng tài khoản có lối tắt `RS-LiDAR`.
6. **Bị ngắt kết nối:** mở lại notebook, chạy lại theo bước 4. Phần đã xong tự bỏ qua. Ngoại lệ: run Phase 2 dùng seed toàn cục (E1, E3) bị ngắt giữa chừng thì notebook in đường dẫn và dừng; **xoá thư mục đó trên Drive** rồi chạy lại.
7. **Lỗi:** gửi nhóm **traceback đầy đủ** + tên notebook + số mục.
8. **Nộp kết quả:** file nằm ở `MyDrive/RS-LiDAR/extra_experiments/<tên thí nghiệm>/` (E6 notebook 1–4: `MyDrive/RS-LiDAR/results/`; notebook chấm GenEval FLUX: `extra_experiments/E6_flux_geneval/`). Gửi tên file hoặc ảnh chụp vào nhóm chat.

---

<a id="4-từng-việc"></a>
## 4. Từng việc

**Mốc kiểm tra dùng chung cho E1–E5** (Table 1, SD 1.5, DDPM-100, run seed 100, λ = 5000, chấm bằng đúng các hàm trong notebook). Dòng mốc trong bảng kết quả của bạn phải gần các số này; lệch quá khoảng 0.02 thì gần như chắc chắn đang đọc nhầm pool hoặc nhầm run, phải báo nhóm trước khi phân tích tiếp.

| Run | IR | CLIP | HPS | GenEval |
|---|---|---|---|---|
| LiDAR | 0.350 | 0.277 | 0.268 | 0.481 |
| RS-LiDAR (σ = 1, M = 4) | 0.383 | 0.278 | 0.269 | 0.499 |

**Cách đọc bảng kết quả `summarize`** (E1, E3, E4, E5 đều in cùng dạng). Mỗi dòng là một run; cột `IR`, `CLIP`, `HPS`, `AS` (Aesthetic), `GenEval` là trung bình trên các prompt. Cột `ΔIR [95% CI]` là chênh lệch so với **run mốc** (ghi ở dòng đầu ra `Δ = run − <mốc>`), ví dụ `+0.0330 [+0.0250, +0.0410]*`:
- số đầu là chênh lệch trung bình;
- trong ngoặc là khoảng tin cậy 95% (bootstrap theo prompt);
- dấu `*` nghĩa là khoảng tin cậy **không chứa 0**, tức khác biệt có ý nghĩa. Không có `*` thì chưa kết luận được là hơn hay kém.

Dòng của run mốc có cột Δ để trống (`NaN`), là bình thường.

<a id="4a-việc-của-hiếu"></a>
## 4A. Việc của Hiếu (T1, T3, T5, T7)

<a id="t1"></a>
### T1 · Hiếu · E2: smoothing đổi ứng viên thắng ở bao nhiêu prompt

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E2 (xem [tham khảo](#ref-e2)) |
| 📓 Notebook | [`E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb`](E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb) |
| 🖥️ GPU | **L4** (A100 cũng được) |
| ⏱️ Thời gian | ~0.5–1 h; thêm ~0.6 h (A100) / ~1.5 h (L4) nếu phải chấm lại pool RS (ước lượng) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp output mục 5, 6, 7, 8; ba file `E2_per_prompt.csv`, `E2_flipped_winner_quality.csv`, `E2_gain_by_flip.csv` |

**🎯 Mục tiêu.** Đếm xem trên bao nhiêu prompt, ứng viên điểm cao nhất theo reward thô R **khác** ứng viên điểm cao nhất theo reward làm trơn R̄σ ("flip"), và kiểm tra ứng viên mới có thật sự tốt hơn trên các metric không được tối ưu (HPS, GenEval).

**▶ Các bước**

1. **Lấy notebook về máy:** `git pull` trong bản clone của repo rồi dùng file ở link trong thẻ (file trong repo). Không mở bằng link `colab.research.google.com/github/...`.
2. Vào https://colab.research.google.com → **File → Upload notebook** → chọn file vừa tải.
3. **Runtime → Change runtime type** → *Hardware accelerator*: **L4 GPU** → **Save**.
4. **Không sửa biến nào** (xem bảng tham số bên dưới).
5. **Runtime → Run all** (Ctrl + F9). Khi hiện hộp thoại Google Drive: chọn tài khoản có lối tắt `RS-LiDAR`, bấm **Continue/Allow** hết các bước.
6. ⏸ **Mục 3 (cài đặt):** cell kiểm tra môi trường phải in ✅ cho mọi thư viện. Có ❌ → **Runtime → Restart session** → **Run all** lại.
7. ⏸ **Mục 4:** dòng in ra phải là `E2_flip_rate: 553 prompt | kết quả tổng hợp lưu tại …/RS-LiDAR/extra_experiments/E2_flip_rate`.
8. ⏸ **Mục 5:** chụp màn hình. Hai dòng `Pool LiDAR :` và `Pool RS-LiDAR :` phải là đường dẫn (không phải `None`), và có dòng `✅ Pool RS dùng chung latents với pool LiDAR: …`. Nếu là `⚠️ … KHÔNG dùng chung latents …` thì notebook tự chấm lại (thêm ~1.5 h trên L4); vẫn để chạy tiếp nhưng **báo nhóm ngay** (xem Lưu ý).
9. ⏸ **Mục 6:** chụp toàn bộ output (vài phút, chỉ dùng CPU).
10. Chờ mục 7 (giải mã và chấm ảnh của các prompt bị flip) và mục 8 chạy xong. Chụp hai bảng.
11. Mở `MyDrive/RS-LiDAR/extra_experiments/E2_flip_rate/` → gửi nhóm 3 file CSV và các ảnh chụp.

**🔧 Tham số**

| Biến | Ở mục | Giá trị | Có đổi không |
|---|---|---|---|
| `PROMPT_MODE` | 4 | `"ALL"` | Không |
| Mọi biến khác | — | — | Không |

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 6 | `Số prompt phân tích: N` | Số prompt ghép được hai pool. Phải là 553 |
| 6 | `Flip rate (argmax R ≠ argmax R̄σ): x%` | Tỉ lệ prompt có ứng viên thắng bị đổi |
| 6 | `Spearman(R, R̄σ) trung bình trong prompt: s` | Mức giống nhau giữa thứ hạng theo R và theo R̄σ (1 = y hệt) |
| 6 | `Hạng của ứng viên thắng theo R khi xếp theo R̄σ (trung vị, chỉ prompt flip): r / 50` | Trên các prompt flip, ứng viên thắng cũ bị R̄σ xếp tụt xuống hạng mấy |
| 6 | 3 dòng `thắng theo R`, `thắng theo R̄σ`, `nền hoán vị` + dòng `Wilcoxon … p = …` | Độ "mong manh" (gap = R − R̄σ, quy ra phân vị trong pool; 0.5 = điển hình, 1 = mong manh nhất) của ứng viên thắng, so với mức nền do may rủi chọn |
| 6 | Bảng theo task (`flip`, `spearman`, `raw_winner_gap_pct`) và file `E2_per_prompt.csv` | Như trên, tách theo loại prompt |
| 7 | Bảng `metric · thắng theo R · thắng theo R̄σ · Δ (R̄σ − R) [95% CI] · R̄σ thắng · hoà`, file `E2_flipped_winner_quality.csv` | Trên các prompt flip: chất lượng ảnh của ứng viên thắng mới so với cũ |
| 8 | Bảng `nhóm (flip / không flip) · n · ΔIR · ΔHPS · ΔCLIP · ΔGenEval`, file `E2_gain_by_flip.csv` | Lợi ích của RS-LiDAR so với LiDAR (Table 1) nằm ở prompt flip hay không |

**🌟 Kết quả lý tưởng**

- **Mục 6:**
  - `Số prompt phân tích: 553`.
  - **Flip rate ≥ 15%.**
  - Spearman **dương rõ nhưng nhỏ hơn 1**: R̄σ giữ phần lớn thứ hạng, chỉ đổi một phần. Gần 0 nghĩa là R̄σ gần như ngẫu nhiên → báo nhóm.
  - Dòng `thắng theo R` **lớn hơn** dòng `nền hoán vị`, Wilcoxon **p < 0.05**: ứng viên thắng theo R mong manh hơn mức may rủi, đúng câu chuyện "spike".
  - Dòng `thắng theo R̄σ` **nhỏ hơn** dòng `thắng theo R`.
- **Mục 7:**
  - Dòng **HPS** và **GenEval** có Δ dương, **cả hai cận của khoảng tin cậy đều dương**; cột `R̄σ thắng` trên 50%.
  - Dòng `IR (tối ưu, thiên về R)` có Δ **âm**. Đây là **bình thường**: ứng viên thắng theo R luôn có IR cao nhất theo định nghĩa. Không dùng dòng này để kết luận.
  - CLIP và Aesthetic không giảm có ý nghĩa.
- **Mục 8:** ΔIR và ΔGenEval ở nhóm **flip** lớn hơn rõ so với nhóm **không flip**; nhóm không flip có Δ gần 0. Cột `n` của nhóm flip khớp với flip rate × 553.

**⚠️ Lưu ý**

- Nếu mục 5 báo ⚠️, notebook tự chấm lại pool (thư mục `..._rescore`), chỉ tốn một lần vì E1 (T3) dùng lại. **Báo nhóm ngay**, vì khi đó câu "LiDAR and RS-LiDAR share the same candidates" ở appendix C.3 phải xem lại.
- Mục 8 cần hai run Table 1 trên Drive. Nếu in `⚠️ Chưa có đủ hai run Table 1` thì bỏ qua; chạy lại riêng mục 8 sau khi xong T3.
- Chạy T1 **trước** T3 để hai notebook không chấm lại pool RS cùng lúc.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Flip rate ≥ 15%, và ở mục 7 HPS **và** GenEval có Δ dương với CI không chứa 0 | Có bằng chứng trực tiếp cho cơ chế của bài | Gửi kết quả; số này vào Section 6.3 |
| ⚠️ Flip rate ≥ 15% nhưng mục 7 có CI chứa 0 | Smoothing có đổi lựa chọn, nhưng chưa chứng minh được lựa chọn mới tốt hơn | Trong bài chỉ giữ flip rate, bỏ chữ "better" |
| ❌ Flip rate < 15% | Spike hiếm; câu chuyện "spike" yếu | Báo nhóm. Bỏ câu chuyện spike khỏi Intro, giải thích lợi ích bằng Prop 3. E1 và E4 trở nên quan trọng hơn |

---

<a id="t3"></a>
### T3 · Hiếu · E1: smoothing có chỉ là hạ nhiệt độ λ hoặc cắt reward ngoại lai không

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E1 (xem [tham khảo](#ref-e1)) |
| 📓 Notebook | [`E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb`](E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** hoặc A100 |
| ⏱️ Thời gian | 4 run Phase 2 (LiDAR λ = 100, 1000; LiDAR-Clip q = 0.8, 0.95), mỗi run ước lượng 1.5–2.5 h A100, tổng khoảng 6–10 h (chưa đo; L4 chậm hơn khoảng 2–3 lần). Chia nhiều phiên Colab |
| 🔗 Cần xong trước | Nên sau T1 |
| 📤 Nộp lại | Ảnh chụp mục 5 và mục 6; `E1_summary.csv`; thời gian thực của run đầu tiên |

**🎯 Mục tiêu.** Kiểm tra xem LiDAR có thể bắt kịp RS-LiDAR chỉ bằng cách đổi nhiệt độ λ hoặc cắt reward ngoại lai không. Nếu không, đóng góp của bài đứng vững.

**▶ Các bước**

1. **Lấy lại notebook** (bản đã sửa dải λ ngày 16/11): `git pull` rồi dùng file trong repo ở link trong thẻ. Không dùng bản tải trước ngày đó.
2. Colab → **File → Upload notebook** → chọn file.
3. **Runtime → Change runtime type** → **L4 GPU** (hoặc **A100 GPU** + bật **High-RAM**) → **Save**.
4. **Runtime → Run all** (Ctrl + F9). Cho phép Google Drive khi được hỏi. Notebook chạy hết mục 1–9 (4 run Phase 2, GenEval, bảng) rồi tự ngắt máy. Các bước 5–13 là để kiểm tra output và chụp màn hình.
5. ⏸ **Mục 3:** cell kiểm tra môi trường phải in ✅ hết. Có ❌ (notebook dừng ở đó) → Restart session → Run all lại.
6. ⏸ **Mục 4:** phải in `E1_temperature_clipping: 553 prompt | …`.
7. ⏸ **Mục 5** (vài phút CPU). Chụp màn hình. Kiểm tra theo thứ tự:
   - `Pool LiDAR :` và `Pool RS-LiDAR :` là đường dẫn, không phải `None`.
   - Dòng `79 bước có guidance: t = 990 … 210`.
   - Bảng 4 cột × 10 dòng (λ = 1 … 20000).
   - Ba dòng `LiDAR đạt 25% / 50% / 90% độ mạnh của λ = 5000 từ λ ≈ …`.
8. **Kiểm tra dải λ** từ ba dòng ở bước 7 (xem lại sau khi chạy; mặc định đúng trong đa số trường hợp):

   | Ba dòng in ra | Làm gì |
   |---|---|
   | Mốc 50% ở λ trong khoảng 30–300 **và** mốc 90% ở λ trong khoảng 300–3000 | Giữ nguyên `LAMBDAS_LIDAR = [100, 1000]`, sang bước 9 |
   | Một trong hai mốc nằm ngoài khoảng trên | Mục 6: đổi `LAMBDAS_LIDAR` thành **[giá trị trong lưới gần mốc 50% nhất, giá trị gần mốc 90% nhất]** |

   Chỉ chạy 2 giá trị λ: λ thấp hơn gần như không dẫn hướng và Figure 8 của bài LiDAR đã cho thấy IR ở đó thấp hẳn. `LAMBDAS_RS` để trống. Nếu đổi thì chạy lại cell mục 6 (Shift + Enter) và báo nhóm giá trị mới.
9. ⏸ **Mục 6:** chụp màn hình.
   - Hai dòng `Mốc LiDAR λ=5000 :` và `Mốc RS-LiDAR λ=5000 :` nên là đường dẫn. Nếu là `None`, notebook sẽ tự chạy lại mốc (thêm 1–2 run).
   - Hai dòng `📁 LiDAR_winsor0.8: 553 prompt` và `📁 LiDAR_winsor0.95: 553 prompt`.
   - Danh sách run dạng `• LiDAR λ=100 -> Target_samples/E1_LiDAR_SD15_n50_DDPM100_lmbda100_seed100`. Với dải mặc định phải có đúng **4 dòng**: 2 LiDAR (λ = 100, 1000), 2 LiDAR-Clip (q = 0.8, 0.95).
10. (Không cần thao tác: Run all đã chạy tiếp mục 7–9, lần lượt 4 run, rồi GenEval và bảng.)
11. **Ghi lại thời gian** của run đầu tiên: từ lúc in dòng `▶ <tên run đầu>` đến lúc in dòng `▶ <tên run thứ hai>` (gửi nhóm để chỉnh ước lượng).
12. **Phiên Colab hết giờ:** upload lại notebook (nếu cần), chọn GPU, **Run all**. Run đã xong được bỏ qua và in `✅ Phase 2 đã đủ 553 prompt: <tên run>`. Nếu notebook báo `Run dở dang (…)`: vào Drive xoá đúng thư mục được in ra, rồi **Run all** lại.
13. ⏸ **Mục 9:** chụp bảng. Gửi `MyDrive/RS-LiDAR/extra_experiments/E1_temperature_clipping/E1_summary.csv`.

**🔧 Tham số**

| Biến | Ở mục | Mặc định | Có đổi không |
|---|---|---|---|
| `PROMPT_MODE` | 4 | `"ALL"` | Không (chỉ đổi khi chạy thử, xem Lưu ý) |
| `LAMBDAS_LIDAR` | 6 | `[100, 1000]` | **Chỉ** theo bảng ở bước 8 |
| `LAMBDAS_RS` | 6 | `[]` | Không (thêm `[100]` chỉ khi còn dư GPU) |
| `WINSOR_QS` | 6 | `[0.8, 0.95]` | Không |

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 5 | Bảng cột `LiDAR: ‖wʳ−w‖₁` | Độ mạnh dẫn hướng ở mỗi λ: 0 = không dẫn hướng, 2 = chuyển hẳn sang ứng viên khác |
| 5 | Cột `LiDAR: % bước đổi ứng viên` | Tỉ lệ bước mà reward đổi được ứng viên chi phối |
| 5 | Hai cột `RS-LiDAR: …` | Như trên nhưng với R̄σ |
| 5 | Dòng `Độ lệch chuẩn reward trong một prompt: {...}` | Độ trải của R và của R̄σ giữa các ứng viên |
| 9 | Bảng `summarize`, mốc là `LiDAR λ=5000`, file `E1_summary.csv` | Các dòng: `LiDAR λ=5000`, `RS-LiDAR λ=5000`, `LiDAR λ=100`, `LiDAR λ=1000`, `LiDAR-Clip q=0.8`, `LiDAR-Clip q=0.95` |

**🌟 Kết quả lý tưởng**

- **Mục 5:**
  - Cột `LiDAR: ‖wʳ−w‖₁` **tăng dần** theo λ, từ gần 0 ở λ = 1 rồi **chững lại** (không vượt 2) ở λ lớn.
  - Mốc 50% ở khoảng λ = 30–300, mốc 90% ở khoảng λ = 300–3000.
  - Nếu độ lệch chuẩn của RS-LiDAR nhỏ hơn của LiDAR (dòng cuối) thì đường RS-LiDAR lệch sang phải (cần λ lớn hơn); đó là bình thường.
- **Mục 9, kiểm tra trước:** dòng `LiDAR λ=5000` có IR ≈ 0.350, GenEval ≈ 0.481; dòng `RS-LiDAR λ=5000` có IR ≈ 0.383, GenEval ≈ 0.499 (mốc ở đầu mục 4).
- **Mục 9, kết quả mong muốn:**
  - `RS-LiDAR λ=5000`: `ΔIR` và `ΔGenEval` **dương, có `*`**.
  - **Mọi** dòng `LiDAR λ=…` và `LiDAR-Clip` có IR **và** GenEval **thấp hơn** dòng `RS-LiDAR λ=5000`.
  - Các dòng `LiDAR λ=…`: IR tăng theo λ rồi chững lại, không dòng nào vượt `LiDAR λ=5000` rõ rệt.
  - CLIP và HPS của các dòng RS-LiDAR không thấp hơn LiDAR có ý nghĩa.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** muốn xem xu hướng trước, đặt `PROMPT_MODE = "100_REPRESENTATIVE"` ở mục 4 (100 prompt, khoảng 1/5 thời gian). Bản thử ghi vào thư mục riêng (hậu tố `_100p`), không dùng lại được cho bản đủ. Nhớ đổi lại `"ALL"` khi chạy đủ.
- Run của E1 dùng seed toàn cục để ghép cặp với Table 1. Bị ngắt giữa một run thì notebook **từ chối chạy tiếp** run đó và in đường dẫn: xoá thư mục đó trên Drive rồi chạy lại.
- Trên L4, chỉ bắt đầu một phiên khi còn đủ thời gian cho trọn một run.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ RS-LiDAR λ = 5000 cao hơn mọi LiDAR λ và LiDAR-Clip trên IR và GenEval, có `*` | Smoothing không phải hạ nhiệt độ hay cắt ngoại lai | Gửi CSV; viết vào Section 6 |
| ⚠️ Hơn nhưng không có `*` ở GenEval | Chỉ chắc chắn trên IR | Viết claim cho IR; GenEval ghi là xu hướng |
| ⚠️ LiDAR-Clip bắt kịp ở một ngưỡng nhưng không ở ngưỡng kia | Cắt ngoại lai chỉ bằng smoothing khi chọn đúng ngưỡng; smoothing không cần chọn ngưỡng | Báo nhóm; viết claim kèm điều kiện ngưỡng |
| ❌ LiDAR ở một λ nào đó bắt kịp hoặc vượt | Có thể chỉ là hiệu ứng nhiệt độ | Bỏ claim "beyond temperature"; E4 (T6) lên ưu tiên số 1 |

---

<a id="t5"></a>
### T5 · Hiếu · E3: so sánh cùng số lần gọi reward

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E3 (xem [tham khảo](#ref-e3)) |
| 📓 Notebook | [`E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **A100** cho mục 5 (sinh 3 pool mới); L4 được cho phần sau |
| ⏱️ Thời gian | ~3–5 h A100 (ước lượng: 3 pool Phase 1 + 1 run Phase 2) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp mục 5 và mục 8; `E3_summary.csv` |

**🎯 Mục tiêu.** RS-LiDAR (K = 50 ứng viên, M = 4 lần nhiễu) gọi reward 200 lần mỗi prompt. Kiểm tra xem LiDAR với **200 ứng viên** (cũng 200 lần gọi reward) có bắt kịp không.

**▶ Các bước**

1. Lấy notebook từ repo (`git pull`, file ở link trong thẻ), Colab → **File → Upload notebook**.
2. **Runtime → Change runtime type** → **A100 GPU**, bật **High-RAM** → **Save**.
3. Không sửa biến nào. **Runtime → Run all**. Cho phép Google Drive.
4. ⏸ **Mục 3–4:** như T1 (✅ hết; `E3_compute_matched: 553 prompt | …`).
5. ⏸ **Mục 5:** lần lượt in `▶ Pool seed 200`, `▶ Pool seed 300`, `▶ Pool seed 400`, mỗi pool kèm log sinh ảnh của `lookahead_sampling.py` (pool đã có sẵn thì in `✅ Phase 1 đã đủ 553 prompt: …` và bỏ qua). Dòng cuối là danh sách 4 đường dẫn pool (seed 100, 200, 300, 400). Chụp màn hình.
6. **Muốn chuyển sang L4 sau mục 5 (tiết kiệm CU):** bấm **Runtime → Disconnect and delete runtime**, đổi sang **L4 GPU**, rồi **Run all**. Mục 5 tự bỏ qua vì pool đã có; mục 6 tạo lại pool ghép trên đĩa Colab (vài phút).
7. ⏸ **Mục 7:** dòng đầu in một dict có `LiDAR K=50` và `RS-LiDAR K=50 M=4` trỏ tới hai run Table 1 (không phải `None`). Sau đó chạy `▶ LiDAR K=200` (bản gọn 17/11: không còn K = 100).
8. ⏸ **Mục 8:** chụp bảng. Gửi `MyDrive/RS-LiDAR/extra_experiments/E3_compute_matched/E3_summary.csv`.

**🔧 Tham số**

| Biến | Ở mục | Giá trị | Có đổi không |
|---|---|---|---|
| `PROMPT_MODE` | 4 | `"ALL"` | Không |
| `EXTRA_SEEDS` | 5 | `[200, 300, 400]` | Không |

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 5 | 3 pool mới trong `MyDrive/RS-LiDAR/Lookahead_samples/` | 150 ứng viên mới mỗi prompt |
| 6 | `LiDAR_K100`, `LiDAR_K200` trên đĩa Colab | Pool 100 và 200 ứng viên (pool nhỏ nằm trong pool lớn) |
| 8 | Bảng `summarize`, **mốc là `RS-LiDAR K=50 M=4`**, file `E3_summary.csv` | Các dòng: `LiDAR K=50`, `RS-LiDAR K=50 M=4`, `LiDAR K=200` |

**🌟 Kết quả lý tưởng**

- **Kiểm tra trước:** `LiDAR K=50` có IR ≈ 0.350, GenEval ≈ 0.481; `RS-LiDAR K=50 M=4` có IR ≈ 0.383, GenEval ≈ 0.499.
- **Kiểm tra hợp lý:** IR của LiDAR tăng dần K = 50 → 100 → 200 (thêm ứng viên thì dẫn hướng tốt hơn). Nếu IR **giảm** khi K tăng → báo nhóm, có thể ghép pool sai.
- **Mong muốn:** cả ba dòng LiDAR có `ΔIR` và `ΔGenEval` **âm** so với mốc RS-LiDAR. Tốt nhất là dòng `LiDAR K=200` có `ΔIR` âm **có `*`**.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** đặt `PROMPT_MODE = "100_REPRESENTATIVE"` (100 prompt, khoảng 1/5 thời gian). Bản thử ở thư mục riêng; nhớ đổi lại `"ALL"`.
- Run `LiDAR K=200` dùng seed toàn cục: bị ngắt giữa run thì xoá thư mục được in ra rồi chạy lại.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ `LiDAR K=200` có ΔIR ≤ 0 và ΔGenEval ≤ 0 (RS-LiDAR K = 50, M = 4 không thua) | Thắng ngay cả khi cùng số lần gọi reward | Claim: "At equal reward evaluations, RS-LiDAR outperforms LiDAR with K = 200" |
| ⚠️ Ngang nhau (CI chứa 0) | Cùng chất lượng nhưng Phase 1 rẻ hơn 4 lần | Claim về chi phí: chất lượng tương đương, Phase 1 rẻ hơn 4×, không tăng VRAM |
| ❌ `LiDAR K=200` hơn rõ (Δ dương có `*`) | Một phần lợi ích đến từ việc gọi reward nhiều hơn | Ghi trung thực; chỉ giữ claim về chi phí |

---

<a id="t7"></a>
### T7 · Hiếu · E4 dự phòng hoặc E5 (theo cổng E4)

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E4 dự phòng **hoặc** E5 (xem [tham khảo E4](#ref-e4), [E5](#ref-e5)) |
| 📓 Notebook | E4: [`E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb`](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb)<br>E5: [`E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb`](E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | ~6–8 h (ước lượng) |
| 🔗 Cần xong trước | [T6](#t6) xong, đã qua [cổng E4](#5-cổng-quyết-định) |
| 📤 Nộp lại | CSV và ảnh chụp của nhánh đã chạy (liệt kê bên dưới) |

**🎯 Mục tiêu.**
- **Nhánh E4 dự phòng:** chạy lại E4 với σ nhỏ hơn và nhiều lần nhiễu hơn, khi T6 chưa ra kết quả rõ.
- **Nhánh E5:** kiểm tra RS-LiDAR vẫn có ích khi tối ưu **hai** reward cùng lúc (ImageReward + HPS).

**▶ Các bước, nhánh E4 dự phòng** (khi cổng E4 ghi "T7 = E4 dự phòng")

1. Tải notebook E4, upload lên Colab, chọn **L4 GPU**.
2. Mục 7: không sửa gì; notebook tự chọn lại đúng task như T6 (kết quả chạy thử đã lưu trên Drive).
3. Mục 9: sửa dòng `SIGMA_D, M_D = SIGMA_AUTO, 8` thành `SIGMA_D, M_D = 0.05, 16`.
4. **Runtime → Run all**. Mục 6 không chạy thử lại vì kết quả đã lưu trên Drive (`E4_pilot_raw.json`).
5. ⏸ Mục 9: dòng tổng kết từng task giống T6. Mục 11: chụp bảng.
6. Gửi `E4_summary_<hậu tố>_sig0.05_M16.csv` và `E4_geneval_by_task_<hậu tố>_sig0.05_M16.csv`.
7. Nếu vẫn không đạt và nhóm đồng ý: lặp lại với `SIGMA_D = 0.03`.

**▶ Các bước, nhánh E5: ĐÃ BỎ (17/11/2026, giới hạn GPU).** Không chạy. Phần dưới giữ lại để tham khảo.

1. Tải notebook E5, upload lên Colab, chọn **L4 GPU**.
2. Không sửa biến nào. **Runtime → Run all**.
3. ⏸ **Mục 5:** chấm thêm HPS cho hai pool. Đây là bước chậm nhất (thư viện HPS nạp lại checkpoint mỗi lần gọi). Bị ngắt thì **Run all** lại, phần đã chấm được giữ. Xong khi cell mục 5 chạy hết không lỗi; chạy lại cell này thì phải in hai dòng `✅ Phase 1 đã đủ 553 prompt: …_IRHPS`.
4. ⏸ **Mục 6:** chụp 3 dòng:
   - `std trong prompt (thô): IR …, HPS …; tương quan IR–HPS trong prompt = …`
   - `tỷ trọng hiệu dụng trong prompt (thô): IR 0.50 | HPS 0.50`
   - `tỷ trọng hiệu dụng trong prompt (làm trơn): IR … | HPS …`
5. Chờ mục 7 (2 run Phase 2) và mục 8. Chụp bảng mục 8.
6. Gửi `MyDrive/RS-LiDAR/extra_experiments/E5_multi_reward/E5_summary_w0.5.csv`.

**🔧 Tham số**

| Nhánh | Biến | Ở mục | Giá trị | Có đổi không |
|---|---|---|---|---|
| E4 dự phòng | `E4_TAGS` | 7 | như T6 | Không |
| E4 dự phòng | `SIGMA_D, M_D` | 9 | `0.05, 16` (lần sau `0.03, 16`) | **Có** |
| E5 | `W_IR` | 6 | `0.5` | Không |
| E5 | mọi biến khác | — | — | Không |

**📦 Đầu ra (E5)**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 6 | Dòng `tương quan IR–HPS trong prompt` | IR và HPS giống nhau tới đâu khi xếp hạng ứng viên |
| 6 | Hai dòng `tỷ trọng hiệu dụng` | Đóng góp thực của IR và HPS vào reward kết hợp |
| 8 | Bảng `summarize`, mốc là `LiDAR (IR+HPS)`, file `E5_summary_w0.5.csv` | Các dòng: `LiDAR (IR)`, `RS-LiDAR (IR)` (hai run Table 1), `LiDAR (IR+HPS)`, `RS-LiDAR (IR+HPS)` |

**🌟 Kết quả lý tưởng**

- **E4 dự phòng:** như T6 (RS-LiDAR hơn LiDAR 0/1 về GenEval ≥ +0.03 có `*`).
- **E5, mục 6:**
  - Dòng `(thô)` đúng `IR 0.50 | HPS 0.50` (theo cách chuẩn hoá).
  - Dòng `(làm trơn)` không lệch quá xa, mỗi bên trong khoảng 0.3–0.7 (ngưỡng tham khảo). Lệch hơn thì báo nhóm.
  - Tương quan IR–HPS rất cao (gần 1) thì hai reward gần như một, thí nghiệm ít thông tin → báo nhóm.
- **E5, mục 8:**
  - Kiểm tra hợp lý: `LiDAR (IR+HPS)` có HPS **cao hơn** `LiDAR (IR)`, vì giờ HPS cũng được tối ưu.
  - Mong muốn: `RS-LiDAR (IR+HPS)` có `ΔIR` **và** `ΔHPS` dương có `*`; `ΔCLIP`, `ΔAS`, `ΔGenEval` không âm có ý nghĩa.

**⚠️ Lưu ý**

- **Chạy thử E5 (tuỳ chọn):** `PROMPT_MODE = "100_REPRESENTATIVE"`; nhớ đổi lại `"ALL"`.
- Run E5 dùng seed theo prompt (hậu tố `_pps`), còn hai dòng Table 1 dùng seed toàn cục. So sánh chính là `RS-LiDAR (IR+HPS)` với `LiDAR (IR+HPS)`; hai dòng Table 1 chỉ để tham khảo.
- E5 không dùng chế độ `--use_pareto` có sẵn trong code (xem tham khảo E5).
- Nếu nhánh E4 dự phòng xong mà vẫn còn GPU, chạy thêm E5.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ E4 dự phòng: RS hơn LiDAR (0/1), có `*` | σ nhỏ hơn mới hợp với Mask2Former | E4 vào Section 6 |
| ❌ E4 dự phòng vẫn không hơn | Reward 0/1 trên ứng viên DPM-5 quá thô | E4 vào appendix hoặc bỏ |
| ✅ E5: RS-LiDAR (IR + HPS) hơn LiDAR (IR + HPS) trên cả IR và HPS, có `*` | Phương pháp dùng được với nhiều reward | Một đoạn trong appendix |
| ⚠️/❌ E5 không hơn | Không có lợi thêm khi tối ưu hai reward | Ghi trung thực trong appendix hoặc bỏ |

---

<a id="4b-việc-của-quân"></a>
## 4B. Việc của Quân (T2, T4, T6, T8, T9)

**Chung cho 4 notebook sinh ảnh FLUX (T2, T4, T8):**
- **Bắt buộc A100 + High-RAM**, và có `HF_TOKEN` trong Secrets (🔑 ở thanh bên trái Colab) với quyền truy cập notebook **bật** ([mục 2.2](#22-hugging-face-token-chỉ-người-chạy-e6)).
- Kết quả nằm ở `MyDrive/RS-LiDAR/results/<RUN_NAME>/`. Mỗi prompt một thư mục con có ảnh và `results.json`.
- **Mục 10** của notebook có 3 dòng tự ngắt máy khi chạy xong (đang bị comment bằng `#`). Khi để máy chạy qua đêm, xoá dấu `#` ở 3 dòng đó trước khi chạy để không tốn compute unit lúc máy rảnh.
- Mục 8 in sẵn số của bài LiDAR (Table 3 của họ, đã kiểm tra khớp): FLUX không dẫn hướng IR 1.019; LiDAR n = 50 IR 1.114; CLIP 0.282–0.284. Cấu hình của mình khác bài (s_flow = 3, cửa sổ t ∈ [0.95, 1], lookahead Hyper-SD 8 bước), nên **không cần trùng**; dùng để biết đúng cỡ.

<a id="t2"></a>
### T2 · Quân · E6 notebook 1: LiDAR trên FLUX (sinh pool 50 ứng viên + Phase 2)

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** (bắt buộc) |
| ⏱️ Thời gian | ~10–14 h Phase 1 + Phase 2 (ước lượng), chia nhiều phiên Colab |
| 🔗 Cần xong trước | Có `HF_TOKEN` ([mục 2.2](#22-hugging-face-token-chỉ-người-chạy-e6)) |
| 📤 Nộp lại | Ảnh chụp `pip list`, mục 6.1, mục 8, mục 9; tên thư mục kết quả |

**🎯 Mục tiêu.** Tạo pool 50 ứng viên cho mỗi prompt bằng FLUX, rồi sinh ảnh cuối bằng LiDAR. Pool này được notebook 2 và 4 dùng lại.

**▶ Các bước**

1. Lấy notebook từ repo (`git pull`, file ở link trong thẻ), Colab → **File → Upload notebook**.
2. **Runtime → Change runtime type** → **A100 GPU**, bật **High-RAM** → **Save**.
3. Kiểm tra 🔑 Secrets có `HF_TOKEN`, công tắc *Notebook access* đang bật.
4. **Runtime → Run all**. Cho phép Google Drive khi được hỏi. Notebook chạy hết Phase 1, Phase 2, bảng kết quả rồi tự ngắt máy (mục 10). Các bước sau là để kiểm tra output và chụp màn hình.
   - Mục 1 phải in `✅ Tuyệt vời! GPU A100 có đủ VRAM …`.
   - Mục 3 phải in `✅ Môi trường đã sẵn sàng 100%!`; dòng cuối mục 3 in phiên bản `transformers`, `diffusers`, `tokenizers`, `peft`, `torchao`, `accelerate`: **chụp màn hình**.
5. ⏸ **Mục 4:** phải in `✅ Đã đăng nhập Hugging Face thành công!`. Nếu in `⚠️ CẢNH BÁO: Thiếu HF_TOKEN!` (notebook sẽ lỗi khi tải FLUX) thì làm lại bước 3 rồi Run all.
6. **Mục 5:** các giá trị phải như bảng tham số bên dưới (không sửa gì).
7. (Không cần thao tác.)
8. (Không cần thao tác: tự ngắt máy đã bật sẵn ở mục 10.)
9. (Không cần thao tác.)
10. ⏸ **Mục 6** (Phase 1, phần lâu nhất) kết thúc bằng `✅ Hoàn thành Phase 1! Dữ liệu đã lưu tại: …/Lookahead_samples/FLUX_100_50_8`.
11. ⏸ **Mục 6.1:** chụp 3 dòng:
    - `📊 Tìm thấy 553 prompt lookahead đã hoàn thành.`
    - `✅ Phase 1 phủ đủ 553/553 prompt của tập all.`
    - `✅ Kiểm tra toàn vẹn thành công: Latents khớp chuẩn 100% không gian FLUX.1 (16x128x128)!`
12. Mục 6.2 tự bỏ qua (`RUN_SFLOW_SWEEP = False`).
13. ⏸ **Mục 7** (Phase 2) kết thúc bằng `✅ Hoàn thành Phase 2! Kết quả lưu tại: …/results/FLUX_LiDAR_N1_n50_sflow3.0_w1.0-0.95_all553_pps`.
14. ⏸ **Mục 8:** chụp bảng. **Mục 9:** chụp lưới ảnh.
15. **Bị ngắt giữa chừng:** upload lại notebook, chọn A100, chạy lại từ mục 1 theo các bước trên. Notebook tiếp đúng chỗ (seed theo prompt, `--resume`).
16. Gửi nhóm ảnh chụp và tên thư mục kết quả.

**🔧 Tham số (mục 5; tất cả giữ nguyên)**

| Biến | Giá trị |
|---|---|
| `METHOD` | `"LiDAR"` |
| `PROMPT_PRESET` | `"all"` |
| `NUM_LOOKAHEAD` | `50` |
| `LOOKAHEAD_BATCH` | `5` |
| `PER_PROMPT_SEED` | `True` |
| `S_FLOW_LIDAR` | `3.0` |
| `PHASE2_SEED` | `42` |
| `EVAL_METRICS` | `"ImageReward#Clip-Score"` |

**📦 Đầu ra**

| Ở đâu | Là gì |
|---|---|
| `MyDrive/RS-LiDAR/Lookahead_samples/FLUX_100_50_8/` | Pool 50 ứng viên × 553 prompt (~14 GB) |
| `MyDrive/RS-LiDAR/results/FLUX_LiDAR_N1_n50_sflow3.0_w1.0-0.95_all553_pps/` | Ảnh cuối và `results.json` từng prompt |
| Mục 8 | Dòng `KẾT QUẢ ĐO ĐƯỢC (LiDAR, N=… samples)` với IR trung bình, và dòng `CLIP-Score (cùng run)` |

**🌟 Kết quả lý tưởng**

- Mục 6.1 đủ 3 dòng ✅ như bước 11.
- Mục 8: `N=553 samples`. IR **cao hơn rõ** so với Vanilla (so khi có notebook 3; bài LiDAR báo Vanilla 1.019). CLIP khoảng 0.28.
- Mục 9: ảnh rõ, đúng nội dung prompt, không có ảnh đen, ảnh nhiễu hay ảnh vỡ.

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** `PROMPT_PRESET = "stratified_100"` (100 prompt, ~3–4 h). Bản thử nằm ở thư mục riêng; nhớ đổi lại `"all"`.
- pip có thể báo lỗi `tokenizers` (Python 3.13 của Colab không có bản dựng sẵn). Nếu notebook vẫn chạy tiếp thì bỏ qua, nhưng **phải** chụp `pip list` lần đầu và kiểm tra các lần sau ra đúng phiên bản đó. Gặp lỗi import lạ: **Runtime → Change runtime type → Runtime version 2026.07** (Python 3.12), chạy lại từ đầu.
- Dung lượng: khoảng 14 GB cho pool này, **tính vào Drive của người chạy** (xem [mục 2.1](#21-dùng-chung-google-drive-bắt-buộc)).

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Chạy hết 553 prompt, mục 6.1 đủ ✅, ảnh mục 9 rõ | Có pool cho notebook 2 | Làm T4 |
| ⚠️ Chạy được nhưng chậm hơn ước lượng khoảng 2 lần | E6 sẽ tốn nhiều GPU hơn dự kiến | Báo nhóm để cân lại việc |
| ❌ Lỗi môi trường hoặc hết bộ nhớ | Chưa chạy được FLUX | Gửi traceback và `pip list`; thử runtime 2026.07. Vẫn lỗi thì tạm gác E6, làm T6 trước |

---

<a id="t4"></a>
### T4 · Quân · E6 notebook 2: RS-FlowLiDAR trên cùng pool

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** |
| ⏱️ Thời gian | ~8–10 h (ước lượng) |
| 🔗 Cần xong trước | [T2](#t2) xong (cùng `PROMPT_PRESET`) |
| 📤 Nộp lại | Ảnh chụp mục 6 và mục 8; tên thư mục kết quả |

**🎯 Mục tiêu.** Chấm lại **đúng** 50 ứng viên của notebook 1 bằng reward làm trơn (σ = 0.25, M = 4), rồi sinh ảnh cuối bằng RS-FlowLiDAR. So với T2, chỉ khác điểm của ứng viên.

**▶ Các bước**

1. Tải notebook, upload lên Colab, chọn **A100 GPU + High-RAM**.
2. Làm bước 3 như T2, rồi **Runtime → Run all**. Ảnh phiên bản thư viện (cuối mục 3) phải ra **đúng** các phiên bản ở T2; khác thì báo nhóm.
3. **Mục 5:** kiểm tra theo bảng tham số bên dưới, không sửa gì.
4. (Không cần thao tác: Run all chạy hết rồi tự ngắt máy.)
5. ⏸ **Mục 6:** notebook nạp latents từ `Lookahead_samples/FLUX_100_50_8` rồi chấm lại, ghi vào `Lookahead_samples/FLUX_100_50_8_rs_sig0.25_M4_iid`. **Không** được thấy thanh tiến trình sinh ảnh Hyper-SD cho 50 ứng viên; nếu thấy thì dừng và báo nhóm. Chụp màn hình.
6. ⏸ **Mục 6.1:** đủ 3 dòng ✅ như T2.
7. ⏸ **Mục 7** kết thúc bằng `✅ Hoàn thành Phase 2! Kết quả lưu tại: …/results/FLUX_RSLiDAR_N1_n50_sig0.25_M4_sflow3.0_w1.0-0.95_all553_pps`.
8. ⏸ **Mục 8:** chụp bảng, đặt cạnh số của T2.
9. Gửi nhóm ảnh chụp và tên thư mục.

**🔧 Tham số (mục 5; tất cả giữ nguyên)**

| Biến | Giá trị |
|---|---|
| `METHOD` | `"RS-FlowLiDAR"` |
| `PROMPT_PRESET` | `"all"` (phải giống T2) |
| `SMOOTH_SIGMA` | `0.25` |
| `SMOOTH_M` | `4` |
| `REUSE_LIDAR_POOL` | `True` |
| `PER_PROMPT_SEED` | `True` |
| `S_FLOW_LIDAR` | `3.0` |
| `PHASE2_SEED` | `42` |

**📦 Đầu ra**

| Ở đâu | Là gì |
|---|---|
| `Lookahead_samples/FLUX_100_50_8_rs_sig0.25_M4_iid/` | Cùng 50 ứng viên, điểm R̄σ (lưu một bản latents riêng, ~14 GB) |
| `results/FLUX_RSLiDAR_N1_n50_sig0.25_M4_sflow3.0_w1.0-0.95_all553_pps/` | Ảnh cuối và `results.json` |
| Mục 8 | IR trung bình và CLIP của run RS |

**🌟 Kết quả lý tưởng**

- Mục 8: `N=553 samples`.
- IR của RS **≥** IR của LiDAR (T2), CLIP không thấp hơn.
- Vì cùng ứng viên và cùng nhiễu theo prompt, chênh lệch IR giữa T4 và T2 là so sánh ghép cặp; khoảng tin cậy được tính ở [T9](#t9).

**⚠️ Lưu ý**

- **Chạy thử (tuỳ chọn):** `PROMPT_PRESET = "stratified_100"` (~1–2 h; chỉ được khi T2 cũng đã chạy thử). Nhớ đổi lại `"all"`.
- Notebook dừng nếu pool của notebook 1 chưa đủ prompt.
- Không đổi σ hay M: cấu hình chính σ = 0.25, M = 4 chuyển nguyên từ SDXL.
- Phase 1 RS tự tính lại những prompt mà cache cũ không dùng đúng pool LiDAR; việc này là bình thường.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ IR của RS ≥ LiDAR (T2) | Phương pháp chuyển sang FLUX được | Hàng FLUX cho Table 1 hoặc bảng nhỏ ở Section 6; gỡ Limitation (vi). Chờ T9 để có CI |
| ⚠️ RS ≈ LiDAR (chênh < 0.01) | Đã dự kiến: σ = 0.25 ở ảnh 1024² là nhiễu yếu | Claim "transfers without degradation"; chạy notebook 4 (σ = 1.0) ở T8 |
| ❌ RS thấp hơn rõ (> 0.03 IR) | Cấu hình chưa hợp với FLUX | Báo nhóm; ghi trung thực, cân nhắc notebook 4 |

---

<a id="t6"></a>
### T6 · Quân · E4: reward rời rạc, dẫn hướng bằng verifier kiểu GenEval

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E4 (xem [tham khảo](#ref-e4)) |
| 📓 Notebook | [`E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb`](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | Chạy thử theo task ~5 phút; phần còn lại ~4–6 h A100, L4 chậm hơn khoảng 2 lần (ước lượng) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp mục 6, 8, 9, 11; `E4_pilot.csv`, `E4_pilot_grid.png`, `E4_summary_*.csv`, `E4_geneval_by_task_*.csv` |

**🎯 Mục tiêu.** Dùng bộ chấm đúng/sai của GenEval (0 hoặc 1) làm reward. Kiểm tra xem RS-LiDAR (lấy trung bình 0/1 trên nhiều bản nhiễu, thành điểm từ 0 đến 1) có dẫn hướng tốt hơn LiDAR dùng thẳng 0/1 không.

**▶ Các bước**

1. Tải notebook, upload lên Colab, chọn **L4 GPU**.
2. Không sửa gì ở mục 4 (`PROMPT_MODE = "ALL"`).
3. **Runtime → Run all**. Notebook chạy thử theo task (mục 6), **tự chọn** `E4_TAGS` (mục 7) và σ (mục 8) theo quy tắc đã định trước, chấm toàn bộ (mục 9), chạy 2 run Phase 2 (mục 10), in bảng (mục 11) rồi tự ngắt máy. Nếu dưới 2 task qua chạy thử, notebook **tự dừng ở mục 7** (không tốn GPU cho Phase 2).
4. ⏸ **Mục 3–4:** như T1 (✅ hết; `E4_discrete_reward: 553 prompt | …`).
5. ⏸ **Mục 5:** `Pool:` là đường dẫn (không phải `None`); có dòng `✅ GenEval sẵn sàng.`
6. ⏸ **Mục 6:** chụp bảng 6 dòng (6 task), lưới ảnh `E4_pilot_grid.png` và dòng `Gợi ý E4_TAGS = […]`.
7. ⏸ **Mục 7:** dòng `✅ E4_TAGS = […]` và `E4: N prompt {…} | hậu tố tên run: _…` (ví dụ `_to-cnt-col`). Ghi lại hậu tố. Nếu notebook dừng ở đây với `❌ Chỉ … task qua chạy thử` thì gửi nhóm ảnh mục 6 và dừng E4.
8. ⏸ **Mục 8:** bảng 4 dòng (σ = 0.05, 0.1, 0.25, 0.5) và dòng `✅ σ chọn tự động: …` (quy tắc: σ lớn nhất có `R̄σ trên ứng viên đúng` ≥ 0.5 **và** `số mức R̄σ khác nhau` ≥ 4). Chụp bảng.
9. (Không cần thao tác: Run all chạy tiếp mục 9–11.)
10. Bị ngắt giữa chừng: **Run all** lại. Kết quả chạy thử và điểm từng prompt đã lưu trên Drive nên được dùng lại; task và σ được chọn lại đúng như lần đầu.
11. (Không cần thao tác.)
12. ⏸ **Mục 9:** mỗi task một dòng dạng `two_object  99 prompt | ứng viên đúng … | prompt toàn 0 …% | lẫn đúng/sai …% | toàn 1 …%`. Chụp.
13. ⏸ **Mục 11:** chụp bảng `summarize` và bảng `GenEval theo task`.
14. Gửi các file trong `MyDrive/RS-LiDAR/extra_experiments/E4_discrete_reward/`.

**🔧 Tham số**

| Biến | Ở mục | Mặc định | Có đổi không |
|---|---|---|---|
| `PROMPT_MODE` | 4 | `"ALL"` | Không |
| `E4_TAGS` | 7 | `["two_object", "colors", "counting"]` | **Có**, theo kết quả chạy thử (bước 7) |
| `SIGMA_D` | 9 | `0.1` | **Có**, theo bảng hiệu chỉnh (bước 10) |
| `M_D` | 9 | `8` | Không |

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 6 | Bảng `task · số prompt · tỷ lệ ứng viên đúng · % prompt toàn 0 · % prompt lẫn đúng/sai · % prompt toàn 1 · % prompt có ≥1 đúng`, file `E4_pilot.csv` | Trên ứng viên lookahead (ảnh DPM-5 còn thô), verifier chấm được tới đâu |
| 6 | `E4_pilot_grid.png` | 8 ứng viên mỗi task, viền xanh = đúng, đỏ = sai |
| 8 | Bảng `σ · tỷ lệ đúng (sạch) · R̄σ trung bình · R̄σ trên ứng viên đúng · R̄σ trên ứng viên sai · số mức R̄σ khác nhau` | Nhiễu σ ảnh hưởng tới điểm của ứng viên đúng và sai thế nào |
| 11 | Bảng `summarize`, **mốc là `LiDAR (GenEval 0/1)`**, file `E4_summary_<hậu tố>_sig<σ>_M8.csv` | Các dòng: `LiDAR (IR, Table 1)`, `RS-LiDAR (IR, Table 1)`, `LiDAR (GenEval 0/1)`, `RS-LiDAR (GenEval R̄σ)` |
| 11 | Bảng `GenEval theo task`, file `E4_geneval_by_task_<hậu tố>_sig<σ>_M8.csv` | GenEval của 4 run, tách theo task |

**🌟 Kết quả lý tưởng**

- **Mục 6:**
  - Ít nhất **2 task** (ngoài single_object) có `% prompt có ≥1 đúng` ≥ 0.5.
  - Cột `% prompt lẫn đúng/sai` càng cao càng tốt: đó là những prompt verifier phân biệt được ứng viên.
  - Lưới ảnh: viền xanh và đỏ khớp với mắt thường.
- **Mục 8:**
  - `R̄σ trên ứng viên đúng` giảm dần khi σ tăng.
  - `R̄σ trên ứng viên sai` luôn thấp hơn `R̄σ trên ứng viên đúng`.
- **Mục 11:**
  - Kiểm tra hợp lý: `LiDAR (GenEval 0/1)` có GenEval **cao hơn** `LiDAR (IR, Table 1)` trên các task này, vì giờ GenEval được tối ưu trực tiếp. Nếu thấp hơn thì dẫn hướng bằng verifier chưa hoạt động → báo nhóm.
  - Mong muốn: `RS-LiDAR (GenEval R̄σ)` có `ΔGenEval` **≥ +0.03, có `*`**; `ΔIR` và `ΔHPS` không âm có ý nghĩa (CI chứa 0 hoặc dương).
  - Bảng theo task: RS-LiDAR ≥ LiDAR (0/1) ở đa số task.

**⚠️ Lưu ý**

- Quy tắc giữ task: ít nhất 50% prompt có ít nhất một ứng viên đúng. single_object luôn bị loại (đã chạm trần).
- Mục 9 lưu điểm từng prompt lên Drive; bị ngắt thì chạy lại là tiếp đúng chỗ. Phase 2 dùng seed theo prompt (`_pps`) nên cũng tiếp được.
- Không sửa biến nào; `E4_TAGS` và `SIGMA_D` được chọn tự động theo quy tắc ở mục 6 và 8.
- Trong bài phải ghi: GenEval ở thí nghiệm này là metric được tối ưu; held-out là IR, HPS, CLIP, Aesthetic.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Chạy thử: ≥ 2 task qua. Kết quả cuối: GenEval của RS-LiDAR (R̄σ) cao hơn LiDAR (0/1) ≥ +0.03, có `*`; IR/HPS không giảm có ý nghĩa | Có năng lực mới: dẫn hướng bằng verifier rời rạc | Contribution mới cho Section 6. Xem [cổng E4](#5-cổng-quyết-định): T7 = E5 |
| ⚠️ Hơn nhưng không có `*` | Có xu hướng, chưa chắc | Cổng E4: T7 = E4 dự phòng |
| ❌ Chạy thử: chỉ 0–1 task qua | Ứng viên lookahead quá mờ, verifier không có tín hiệu | Notebook **tự dừng ở mục 7**, không tốn GPU. Báo nhóm: cần reward "mềm" (độ tin cậy detector), phải viết thêm code |
| ❌ Kết quả cuối: RS không hơn | σ chưa hợp | Cổng E4: T7 = E4 dự phòng |

---

<a id="t8"></a>
### T8 · Quân · E6 notebook 3 (Vanilla) và notebook 4 (σ = 1.0)

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | Notebook 3: [`E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb)<br>Notebook 4: [`E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** |
| ⏱️ Thời gian | Notebook 3: ~5–8 h; notebook 4: ~8–10 h (ước lượng) |
| 🔗 Cần xong trước | [T4](#t4) xong |
| 📤 Nộp lại | Ảnh chụp mục 8 của mỗi notebook; tên thư mục kết quả |

**🎯 Mục tiêu.**
- **Notebook 3:** mốc FLUX không dẫn hướng.
- **Notebook 4:** ablation với σ lớn hơn (1.0), cùng ứng viên và cùng M với notebook 2.

**▶ Các bước, notebook 3 (Vanilla)**

1. Mở `MyDrive/RS-LiDAR/results/`. Nếu đã có thư mục `FLUX_Vanilla_N1_steps28_all553` với 553 thư mục con thì **bỏ qua notebook 3**.
2. Nếu chưa có: tải notebook 3, upload, chọn **A100 + High-RAM**, làm bước 3–6 như T2.
3. Mục 5: `METHOD = "Vanilla"`, `PROMPT_PRESET = "all"`, không sửa gì. **Runtime → Run all** (tự ngắt máy khi xong).
4. Notebook 3 **không có Phase 1** (không sinh ứng viên). ⏸ Mục 7 kết thúc bằng `✅ Hoàn thành Phase 2! Kết quả lưu tại: …/results/FLUX_Vanilla_N1_steps28_all553`.
5. ⏸ Mục 8: chụp bảng.

**▶ Các bước, notebook 4 (σ = 1.0): ĐÃ BỎ (17/11/2026, giới hạn GPU).** Không chạy. Phần dưới giữ lại để tham khảo.

1. Chỉ chạy khi T4 cho RS ≈ LiDAR hoặc RS thấp hơn, **hoặc** khi còn GPU.
2. Làm y như T4 nhưng mở notebook 4. Mục 5 phải có `SMOOTH_SIGMA = 1.0`, `SMOOTH_M = 4`, `REUSE_LIDAR_POOL = True`.
3. ⏸ Mục 6 ghi vào `Lookahead_samples/FLUX_100_50_8_rs_sig1.0_M4_iid` và không sinh ứng viên mới.
4. ⏸ Mục 7 kết thúc với thư mục `…/results/FLUX_RSLiDAR_N1_n50_sig1.0_M4_sflow3.0_w1.0-0.95_all553_pps`. Mục 8: chụp bảng.

**🔧 Tham số:** không đổi gì ở cả hai notebook.

**🌟 Kết quả lý tưởng**

- **Notebook 3:** IR gần 1.019 và CLIP gần 0.282 (số của bài LiDAR cho FLUX không dẫn hướng). Không cần trùng, nhưng lệch lớn (quá 0.05 IR) thì báo nhóm. IR của LiDAR (T2) phải **cao hơn rõ** con số này.
- **Notebook 4:** không có "kết quả đúng"; đây là ablation. Đặt cạnh T4 (σ = 0.25) để xem σ lớn hơn giúp hay hại trên ảnh 1024².

**⚠️ Lưu ý**

- Vanilla không dùng seed theo prompt, nên chỉ là mốc tham khảo, không ghép cặp với LiDAR/RS.
- Lỗi `tokenizers` và cách xử lý: như T2.
- Notebook 4 tạo thêm một pool ~14 GB trên Drive của người chạy.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Có Vanilla, và LiDAR (T2) > Vanilla rõ | Guidance trên FLUX hoạt động | Đủ hàng cho bảng FLUX |
| ⚠️ LiDAR (T2) ≈ Vanilla | Guidance gần như không có tác dụng ở cấu hình này | Báo nhóm trước khi viết bảng FLUX |
| ✅ σ = 1.0 hơn σ = 0.25 | σ cho ảnh 1024² nên lớn hơn | Báo nhóm; ghi vào ablation, không đổi cấu hình chính sau khi đã thấy kết quả |
| ❌ Lỗi | — | Gửi traceback |

---

<a id="t9"></a>
### T9 · Quân · E6: chấm GenEval cho các run FLUX

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_GenEval_Scoring_Colab.ipynb`](E6_flow_matching/FLUX_GenEval_Scoring_Colab.ipynb) |
| 🖥️ GPU | **L4** (A100 cũng được) |
| ⏱️ Thời gian | Chưa đo. Chỉ chấm 1 ảnh/prompt cho mỗi run, không sinh ảnh, nên ngắn hơn nhiều so với T2/T4 |
| 🔗 Cần xong trước | [T2](#t2) và [T4](#t4). Có thêm T8 thì bảng có thêm dòng Vanilla và σ = 1.0 |
| 📤 Nộp lại | Ảnh chụp mục 5, 6, 7; `E6_FLUX_summary.csv`, `E6_FLUX_geneval_by_task.csv` |

**🎯 Mục tiêu.** Notebook FLUX chỉ chấm IR và CLIP. Notebook này chấm thêm **GenEval** cho ảnh FLUX bằng **đúng** bộ chấm của Table 1 và E1–E5, rồi lập bảng so sánh RS-FlowLiDAR với LiDAR **có khoảng tin cậy 95%** (ghép cặp theo prompt). Đây là bảng FLUX sẽ đưa vào bài.

**▶ Các bước**

1. Lấy notebook từ repo (`git pull`, file ở link trong thẻ), Colab → **File → Upload notebook**.
2. **Runtime → Change runtime type** → **L4 GPU** → **Save**. Không cần `HF_TOKEN`.
3. **Runtime → Run all**. Cho phép Google Drive bằng tài khoản có lối tắt `RS-LiDAR` (cùng tài khoản đã chạy T2/T4).
4. ⏸ **Mục 3:** cell kiểm tra môi trường in ✅ hết. Có ❌ → **Runtime → Restart session** → **Run all** lại.
5. ⏸ **Mục 4:** phải in `E6_flux_geneval: 553 prompt | kết quả tổng hợp lưu tại …/RS-LiDAR/extra_experiments/E6_flux_geneval`.
6. ⏸ **Mục 5:** chụp bảng 4 dòng (`LiDAR`, `RS-FlowLiDAR σ=0.25`, `RS-FlowLiDAR σ=1.0`, `Vanilla`) với hai cột `prompt có ảnh` và `prompt có results.json`.
   - `LiDAR` và `RS-FlowLiDAR σ=0.25` phải là **553 / 553**.
   - Run chưa chạy hiện 0 và được bỏ qua; như vậy là bình thường.
   - Nếu một run hiện **số lẻ** (ví dụ 300) thì run đó chưa xong, và bảng ở mục 7 chỉ dùng số prompt chung. **Cách xử lý:** chạy xong run đó trước, **hoặc** ở cell mục 5 thêm `#` vào đầu dòng của run đó trong `FLUX_RUNS`, rồi **Run all** lại.
7. ⏸ **Mục 6:** sau dòng `✅ GenEval sẵn sàng.` là mỗi run một dòng `<run>  GenEval (trung bình 6 task) = …  (553 prompt)`. Chụp.
8. ⏸ **Mục 7:** chụp bảng `summarize` (mốc là `LiDAR`) và bảng `GenEval theo task`.
9. Gửi nhóm hai file trong `MyDrive/RS-LiDAR/extra_experiments/E6_flux_geneval/`.
10. **Khi có thêm run** (ví dụ T8 xong sau): chỉ cần **Run all** lại. Phần đã chấm được lưu trong `geneval_cache/` nên không chấm lại.

**🔧 Tham số**

| Biến | Ở mục | Mặc định | Có đổi không |
|---|---|---|---|
| `FLUX_RUNS` | 5 | 4 tên thư mục do notebook FLUX 1–4 đặt | **Chỉ** khi tên thư mục trên Drive khác (ví dụ chạy thử `stratified100`), hoặc khi cần bỏ một run dở dang (bước 6) |
| `REF_RUN` | 5 | `"LiDAR"` | Không |
| `PROMPT_MODE` | 4 | `"ALL"` | Không |

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 5 | Bảng run × số prompt | Run nào có trên Drive, đủ prompt chưa |
| 6 | Dòng `GenEval (trung bình 6 task)` cho mỗi run | GenEval tổng, tính như bài báo (trung bình 6 điểm theo task) |
| 7 | Bảng `run · n_prompts · IR · CLIP · GenEval · ΔIR [95% CI] · ΔCLIP [95% CI] · ΔGenEval [95% CI]`, file `E6_FLUX_summary.csv` | So từng run với LiDAR, ghép cặp theo prompt |
| 7 | Bảng `GenEval theo task` (6 task + dòng `overall (TB 6 task)`), file `E6_FLUX_geneval_by_task.csv` | GenEval tách theo loại prompt |

**🌟 Kết quả lý tưởng**

- **Mục 5:** `LiDAR` và `RS-FlowLiDAR σ=0.25` đều 553 / 553.
- **Mục 6:** GenEval của LiDAR **cao hơn** Vanilla (nếu có). Bài LiDAR báo 0.645 (FLUX không dẫn hướng) và 0.668 (LiDAR, n = 50); cấu hình của mình khác bài nên không cần trùng, nhưng nên cùng cỡ.
- **Mục 7:**
  - Cột `n_prompts` = 553. Nhỏ hơn thì có run dở dang, xử lý theo bước 6.
  - Cột IR và CLIP của mỗi run khớp với số ở mục 8 của notebook FLUX tương ứng.
  - Mong muốn: dòng `RS-FlowLiDAR σ=0.25` có `ΔIR` **dương, có `*`**; `ΔGenEval` ≥ 0; `ΔCLIP` không âm có ý nghĩa.
  - Dòng `Vanilla` (nếu có) có `ΔIR` và `ΔGenEval` âm rõ, tức guidance có giúp. Dòng này không ghép cặp, chỉ để tham khảo.
- **Bảng theo task:** RS-FlowLiDAR σ = 0.25 ≥ LiDAR ở đa số task.

**⚠️ Lưu ý**

- Không chấm GenEval bằng script khác: số phải dùng đúng bộ chấm của Table 1 (notebook này đã chép nguyên cell GenEval đó).
- Notebook chỉ đọc ảnh trên Drive, không sửa thư mục kết quả của T2/T4/T8. Kết quả chấm ghi vào `extra_experiments/E6_flux_geneval/`.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ `RS-FlowLiDAR σ=0.25`: ΔIR > 0 có `*`, ΔGenEval ≥ 0 | Phương pháp chuyển sang FLUX được, có ý nghĩa thống kê | Gửi CSV; đưa bảng FLUX vào bài, gỡ Limitation (vi) |
| ⚠️ ΔIR > 0 nhưng không có `*` | Không làm hại nhưng chưa chắc có lợi | Claim "transfers without degradation"; xem dòng σ = 1.0 (T8) |
| ❌ ΔIR < 0 có `*` | RS với σ = 0.25 làm FLUX kém đi | Báo nhóm; ghi trung thực; xem dòng σ = 1.0 |

---

<a id="4c-việc-của-tác-giả"></a>
## 4C. Việc của tác giả (T10)

<a id="t10"></a>
### T10 · Tác giả · E7: SMC ở nhiệt độ λ_FK khác có đuổi kịp RS-SMC không

| | |
|---|---|
| 👤 Người chạy | **Tác giả** (người đã chạy Table 2) |
| 🧪 Thí nghiệm | E7 (xem [tham khảo](#ref-e7)) |
| 📓 Notebook | [`notebooks_temp/colab/RS_SMC_Table8_Replication_Colab.ipynb`](../colab/RS_SMC_Table8_Replication_Colab.ipynb), đúng notebook đã chạy Table 2 |
| 🖥️ GPU | **A100** (L4 cũng được, chậm hơn) |
| ⏱️ Thời gian | 2 run bắt buộc + tối đa 1 run có điều kiện. Chưa đo; ước lượng 1–1.5 h A100 mỗi run (suy từ tốc độ sinh 4 ảnh DDPM-100 ở Appendix E) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp mục 4 và mục 6 (số phút chạy) của từng run; file `final_metrics.json` của từng run |

**🎯 Mục tiêu.** Section 6.3 nói smoothing nén độ chênh reward gần như đều tay, giống việc hạ nhiệt độ. Trong SMC, nhiệt độ là λ_FK. Câu reviewer sẽ hỏi: *SMC ở λ_FK khác 10 có đạt được RS-SMC không?* E1 trả lời câu này cho LiDAR; T10 trả lời cho SMC, ở N = 4 (nơi gain lớn nhất, 0.563 → 0.744).

**▶ Các bước**

1. Lấy notebook từ repo (`git pull`, file ở link trong thẻ), Colab → **File → Upload notebook**.
2. **Runtime → Change runtime type** → **A100 GPU** → **Save**.
3. Dùng notebook **đã cấu hình sẵn** ở thư mục gốc `RS-LiDAR/` (ngoài repo): `RS_SMC_E7_N4_VanillaSMC_FK3_Colab.ipynb` (`METHOD = "Vanilla-SMC"`, `NUM_PARTICLES = 4`, `FK_LMBDA = 3.0`, mọi biến khác như Table 2; tự ngắt máy khi xong). Không sửa gì.
4. **Runtime → Run all**. Cho phép Google Drive.
5. ⏸ **Mục 4:** kiểm tra các dòng `Phương pháp: Vanilla-SMC`, `Số hạt mục tiêu N: 4`, `Feynman-Kac λ: 3.0`, `Tập prompt: ALL (553 prompts)`, và dòng `Thư mục kết quả` có `_N4_` và `_lmbda3.0_`. Thư mục phải **khác** run λ = 10 của Table 2 (tên có `_lmbda10.0_`); dòng `Tiến độ sẵn có` phải là `0/553`. Chụp.
6. ⏸ **Mục 6:** chờ đến dòng `✅ Hoàn thành thực nghiệm đơn lẻ trong … phút!`. Chụp (đây cũng là số đo thời gian SMC đầu tiên, dùng cho ngân sách).
7. Mở `Target_samples/<RUN_NAME>/final_metrics.json` trên Drive: ghi `ImageReward.mean` (cột Mean của Table 2) và `ImageReward.max` (cột Best-of-N).
8. Lặp bước 3–7 với `RS_SMC_E7_N4_VanillaSMC_FK30_Colab.ipynb` (`FK_LMBDA = 30.0`). **Dừng ở đây** nếu cả hai run có `mean` dưới 0.744 và run λ = 30 không cao hơn rõ run λ = 10 (0.563). Nếu run λ = 30 vẫn tăng rõ so với 0.563 (xu hướng chưa dừng), chạy thêm **đúng một** run `FK_LMBDA = 100.0` rồi dừng.
9. **Chỉ khi** một trong hai run có `mean` ≥ **0.744**: chạy thêm `METHOD = "RS-SMC"`, `NUM_PARTICLES = 4`, `FK_LMBDA` = đúng giá trị đó; các biến smoothing giữ nguyên (`SMC_SIGMA = 0.25`, `SMC_ANNEAL_SIGMA = True`, `SMC_ANNEAL_SCHEDULE = "tweedie_analytical"`, `SMC_NUM_MC_SAMPLES = 4`, `SMC_USE_CRN = False`).

**🔧 Tham số**

| Biến (mục 4) | Giá trị | Có đổi không |
|---|---|---|
| `METHOD` | `"Vanilla-SMC"` (run 1, 2); `"RS-SMC"` (run 3, có điều kiện) | Theo bước 3 và 9 |
| `NUM_PARTICLES` | 4 | Không |
| `FK_LMBDA` | 3.0, rồi 30.0 | Theo bước 3 và 8 |
| `PROMPT_MODE` | `"ALL"` | Không |
| `NUM_INFERENCE_STEPS`, `ETA` | 100, 1.0 | Không |
| `RESAMPLE_FREQUENCY`, `RESAMPLE_T_START`, `RESAMPLE_T_END` | 20, 20, 80 | Không (lịch resampling của Table 2) |
| `SEED` | 100 | Không |
| `OVERWRITE` | False | Không |

**📦 Đầu ra**

| Ở đâu | Nội dung |
|---|---|
| Mục 4 | Bảng cấu hình và tên thư mục kết quả |
| Mục 6 | Số phút chạy |
| `Target_samples/<RUN_NAME>/final_metrics.json` | `ImageReward.mean` (trung bình mọi hạt) và `ImageReward.max` (trung bình hạt tốt nhất mỗi prompt), cùng dạng với các file Table 2 |

**🌟 Kết quả lý tưởng**

- Mốc (Table 2, N = 4, λ_FK = 10): SMC mean **0.563** / Best-of-N **0.684**; RS-SMC mean **0.744** / Best-of-N **0.816**.
- Lý tưởng: SMC ở cả λ_FK = 3 và λ_FK = 30 có mean **dưới 0.744**, tốt nhất là thấp rõ.
- Mốc kiểm tra: mỗi run đủ 553 prompt; thời gian mỗi run cùng cỡ với run λ = 10 cũ. Nếu một run xong nhanh bất thường (vài phút) thì nó đang đọc lại thư mục cũ: xem lại bước 5.

**⚠️ Lưu ý**

- Tên thư mục chứa giá trị λ, nên run mới **không** ghi đè run λ = 10 của Table 2.
- Bị ngắt giữa chừng: **Run all** lại với đúng cấu hình; notebook tự chạy tiếp các prompt còn thiếu (`--resume`).
- Không đổi lịch resampling, số bước hay seed: số mới phải so được trực tiếp với Table 2.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ SMC mean < 0.744 ở cả λ_FK = 3 và 30 | Đổi nhiệt độ không thay được smoothing trong SMC | Gửi hai file JSON; thêm một câu vào Section 6.2 và một dòng ở appendix |
| ⚠️ SMC mean ≥ 0.744 ở một λ | Ở λ đó SMC đuổi kịp mốc RS-SMC λ = 10 | Chạy bước 9. Nếu RS-SMC ở cùng λ vẫn hơn SMC ở λ đó: vẫn là ✅ (so cùng nhiệt độ), báo cả ba số |
| ❌ RS-SMC không hơn SMC ở cùng λ tốt nhất | Lợi ích của RS-SMC ở N = 4 có thể chỉ là do λ_FK = 10 chưa tối ưu cho SMC | Báo nhóm ngay; phải xem lại cách trình bày RS-SMC trong bài |

---

<a id="5-cổng-quyết-định"></a>
## 5. Cổng quyết định

### 🔀 Cổng E4 · sau T6 · cả hai cùng xem

| Tình huống | Quyết định |
|---|---|
| E4 ✅ | E4 thành contribution mới ở Section 6. T7: không chạy gì (E5 đã bỏ) |
| E4 ⚠️ hoặc kết quả cuối ❌ | T7 = **E4 dự phòng** (σ nhỏ hơn, M = 16) |
| Chạy thử E4 ❌ (0–1 task có tín hiệu) | Dừng E4; cần viết code reward "mềm" trước khi chạy lại. T7: không chạy gì |

Các kết quả khác (E1, E2, E3, E6) không chặn việc của ai; hướng xử lý nằm trong bảng 📊 của từng việc.

---

## 6. Quy tắc chung

0. **Mở notebook từ file trên máy** (Colab → **File → Upload notebook**), **không mở từ GitHub**, vì mở từ GitHub thì chạy xong phiên thường không tự kết thúc. Xem [mục 3](#3-cách-chạy-một-notebook).
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
| [E7](#ref-e7) | SMC ở λ_FK khác | "SMC chỉ cần chỉnh λ_FK là bằng RS-SMC?" | Một câu ở Section 6.2 + một dòng appendix |

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

📓 **Notebook:** [E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb](E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb)

**Mục đích.** Đây là phản biện nguy hiểm nhất. Làm trơn kéo các điểm reward lại gần nhau, trông giống như hạ nhiệt độ λ của softmax LiDAR, hoặc như cắt bớt các điểm reward ngoại lai. Nếu LiDAR ở λ thích hợp, hoặc LiDAR với reward bị cắt, đạt được bằng RS-LiDAR, thì đóng góp của bài chỉ còn là "chỉnh siêu tham số". Lý thuyết (Prop 2, mục 4.3) nói hai thứ này khác nhau: hạ λ không đổi thứ tự ứng viên, còn làm trơn thì có. E1 kiểm chứng điều đó bằng thực nghiệm.

**Thiết kế.** Cùng pool 50 ứng viên của Table 1, chạy Phase 2 với:
- LiDAR ở λ ∈ {100, 1000} (bản gọn 17/11 vì giới hạn GPU; λ thấp hơn gần như không dẫn hướng, trích Figure 8 của bài LiDAR thay vì chạy lại). Trong trọng số LiDAR, λ·r không đứng một mình mà cạnh tranh với số hạng khoảng cách ‖xₜ − √ᾱₜ·zₖ‖²/2(1−ᾱₜ), cộng trên 4×64×64 chiều latent (`fkd_pipeline_sd.py`, hàm `get_sample_guide`). Vì vậy vùng chuyển từ "gần như không dẫn hướng" sang "bão hoà" nằm ở λ cỡ hàng chục đến hàng trăm; mô phỏng với latent ngẫu nhiên cho ‖wʳ − w‖₁ ≈ 0.5 ở λ = 10, ≈ 1.9 ở λ = 300 và gần bão hoà từ λ ≈ 1000. Mục 5 đo lại trên pool thật;
- LiDAR-Clip: reward mỗi prompt bị cắt ở phân vị q ∈ {0.8, 0.95} (khoảng 10 và khoảng 2–3 ứng viên đứng đầu hoà nhau), λ = 5000; hai mức kẹp hai bên để kết luận không phụ thuộc ngưỡng;
- mốc LiDAR và RS-LiDAR λ = 5000 của Table 1 (notebook tự tìm trên Drive, không thấy thì tự chạy).

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung (bảng cấu hình chung ở trên) | Đặt `PROMPT_MODE` |
| 5 | Tìm hai pool ứng viên của Table 1 trên Drive (dừng nếu không thấy). Nếu pool RS không cùng 50 ứng viên với pool LiDAR, chấm lại chúng bằng reward làm trơn. In bảng độ mạnh dẫn hướng ‖wʳ − w‖₁ theo λ, đo trên pool thật có tính số hạng khoảng cách (vài phút CPU) | Đọc bảng, chụp màn hình |
| 6 | Tạo 2 pool LiDAR-Clip: cùng ứng viên, nhưng điểm của mỗi prompt bị cắt ở phân vị 0.8 và 0.95 (chỉ ghi lại file điểm, vài giây). In danh sách 4 run sẽ chạy | Sửa `LAMBDAS_LIDAR` nếu bảng ở mục 5 cho thấy dải λ chưa đúng |
| 7 | Sinh ảnh cuối (Phase 2) cho 4 run, lần lượt. Mỗi run ghi vào `MyDrive/RS-LiDAR/Target_samples/E1_...`; run đã xong tự bỏ qua | Chờ. ~2.4 h mỗi run với 553 prompt trên A100 |
| 8 | Chấm GenEval (Mask2Former) cho mọi run, kể cả hai run Table 1 | Chờ |
| 9 | Lập bảng: mỗi run so với LiDAR λ = 5000 trên IR, CLIP, HPS, Aesthetic, GenEval, kèm khoảng tin cậy 95% | Gửi `E1_summary.csv` |

**Vị trí trong bài.** Section 6, mục mới "Is smoothing a lower temperature?": một bảng hoặc hình IR/GenEval theo λ, có đường ngang RS-LiDAR λ = 5000 và điểm LiDAR-Clip. Claim nếu đạt: *"No temperature of LiDAR, and no clipping of its scores, reaches RS-LiDAR."*

<a id="ref-e2"></a>
### E2 — Smoothing đổi ứng viên thắng ở đâu, và có đổi đúng không?

📓 **Notebook:** [E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb](E2_flip_rate/E2_FlipRate_SpikeEvidence_SD15_Colab.ipynb)

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

📓 **Notebook:** [E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb)

**Mục đích.** RS-LiDAR gọi reward K × M = 50 × 4 = 200 lần, LiDAR chỉ 50 lần. Reviewer sẽ hỏi: nếu cho LiDAR cùng 200 lần gọi reward, tức 200 ứng viên, thì LiDAR có bắt kịp không?

**Thiết kế.** Sinh thêm 3 pool 50 ứng viên (seed 200, 300, 400), ghép với pool seed 100 của Table 1 thành K = 200 (bản gọn 17/11 bỏ K = 100). Chạy Phase 2 LiDAR λ = 5000 với `top_k` = 200. So với RS-LiDAR K = 50, M = 4.

**Notebook làm gì, từng mục.**

| Mục | Máy làm gì | Bạn cần làm |
|---|---|---|
| 1–3 | Kiểm tra GPU, nối Google Drive, tải code mới nhất từ GitHub, cài thư viện, kiểm tra môi trường | Cell kiểm tra môi trường dừng thì: Runtime → Restart session, chạy lại từ đầu |
| 4 | Nạp cấu hình chung | Không đổi gì |
| 5 | Phase 1: sinh 3 bộ 50 ứng viên mới (seed 200, 300, 400), cùng cấu hình pool Table 1, ghi vào `MyDrive/RS-LiDAR/Lookahead_samples/`. Bộ nào đã có thì bỏ qua | Chờ (phần nặng nhất, nên dùng A100) |
| 6 | Ghép: bộ 200 = cả bốn pool. Xếp theo điểm IR. Lưu trên **đĩa Colab**, không phải Drive, nên đổi máy thì phải chạy lại mục 5–6 | Chờ (vài phút) |
| 7 | Phase 2 LiDAR λ = 5000 trên bộ 200 ứng viên. Tìm hai run Table 1 làm mốc (không thấy thì tự chạy) | Chờ |
| 8 | Chấm GenEval; bảng so mọi run với RS-LiDAR 50 ứng viên, M = 4 | Gửi `E3_summary.csv` |

**Vị trí trong bài.** Gộp vào bảng E1 hoặc đoạn "Smoothing costs runtime, not memory" ở 6.1. Claim nếu đạt: *"At equal reward evaluations, RS-LiDAR (K = 50, M = 4) outperforms LiDAR with K = 200, which also needs 4× more lookahead sampling."*

<a id="ref-e4"></a>
### E4 — Reward rời rạc: dẫn hướng bằng bộ chấm GenEval

📓 **Notebook:** [E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb)

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

📓 **Notebook:** [E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb](E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb)

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

| Thứ tự | Notebook | Vai trò |
|---|---|---|
| 1 | [FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) | LiDAR: sinh pool 50 ứng viên + Phase 2 |
| 2 | [FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) | **RS-FlowLiDAR chính**: chấm lại pool của bước 1 bằng R̄σ + Phase 2 |
| 3 | [FLUX_Vanilla_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) | Mốc không dẫn hướng. **Bỏ qua nếu Drive đã có run Vanilla 553 prompt** |
| 4 (tuỳ chọn) | [FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) | Ablation σ = 1.0 trên cùng pool |
| 5 | [FLUX_GenEval_Scoring_Colab.ipynb](E6_flow_matching/FLUX_GenEval_Scoring_Colab.ipynb) | Chấm GenEval cho ảnh của notebook 1–4 bằng bộ chấm của Table 1; bảng so với LiDAR có CI 95% ghép cặp ([T9](#t9)) |

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

**GenEval:** notebook 1–4 chỉ chấm IR và CLIP; GenEval chấm bằng notebook 5 (việc [T9](#t9)), cùng bộ chấm với Table 1 và E1–E5.

**Cấu hình chuẩn (đã đặt sẵn, không đổi):** FLUX.1-dev; lookahead Hyper-SD 8 bước, n = 50 ứng viên sinh theo lô 5 (`LOOKAHEAD_BATCH = 5`, vừa A100 40 GB); s_flow = 3.0; cửa sổ dẫn hướng t ∈ [1.0, 0.95]; λ = 5000; 28 bước; 1 ảnh/prompt; seed theo prompt (`PER_PROMPT_SEED = True`) ở cả hai phase; RS chấm lại đúng pool LiDAR (`REUSE_LIDAR_POOL = True`). Cấu hình RS chính σ = 0.25, M = 4 chuyển nguyên từ SDXL, không chỉnh trên GenEval.

**Vị trí trong bài.** Hàng FLUX trong Table 1 (hoặc một bảng nhỏ ở Section 6); gỡ Limitation (vi). Claim nếu đạt: *"RS-TTA transfers to rectified-flow models without re-tuning."*

**Môi trường FLUX (đọc trước khi chạy).** Cell cài đặt ghim `transformers>=4.43.3,<4.45`, nhưng các bản này cần `tokenizers 0.19`, không có bản dựng sẵn cho Python 3.13 (runtime mặc định của Colab). Khi đó pip lỗi mà cell không dừng, và notebook chạy bằng bản transformers/diffusers có sẵn của Colab. Đây có lẽ là cấu hình đã chạy được trong lần hiệu chỉnh s_flow trước, nhưng phiên bản không cố định. Việc cần làm: sau cell cài đặt, chạy `!pip list | grep -i -E "transformers|diffusers|tokenizers|peft|torchao|accelerate"` và **ghi lại** các phiên bản khi chạy thành công; nếu gặp lỗi lạ, đổi runtime sang **2026.07 (Python 3.12)** (Runtime → Change runtime type → Runtime version) để pip cài đúng bản đã ghim.

**Vanilla** không dùng seed theo prompt (giống run Vanilla 553 cũ), nên không ghép cặp nhiễu với LiDAR/RS; nó chỉ là mốc tham khảo.

<a id="ref-e7"></a>
### E7 — SMC ở nhiệt độ λ_FK khác

📓 **Notebook:** `notebooks_temp/colab/RS_SMC_Table8_Replication_Colab.ipynb` (notebook của Table 2), việc [T10](#t10).

**Mục đích.** Câu hỏi "smoothing có chỉ là hạ nhiệt độ không" của E1, nhưng cho SMC, nơi nhiệt độ là λ_FK. RS-SMC là một trong hai phương pháp của bài, và gain ở N = 4 (+0.181) là gain lớn nhất của Table 2.

**Thiết kế (bản tối thiểu, chọn vì chi phí).** Chỉ chạy SMC, chỉ N = 4, λ_FK ∈ {3, 30} cạnh mốc 10 đã có. λ = 3 là phép so sánh chính (nhiệt độ thấp hơn là thứ smoothing có thể "giả làm"); λ = 30 cho chiều ngược lại. RS-SMC chỉ chạy lại ở đúng λ mà SMC đạt ≥ 0.744. **Không** chạy: thêm seed (gain +0.08 đến +0.18 lớn hơn nhiều độ lệch giữa seed ước tính, cỡ 0.01), N = 32, sweep λ cho RS-SMC, chấm held-out.

**Vị trí trong bài.** Một câu ở Section 6.2 ("SMC with λ_FK ∈ {3, 30} reaches … at N = 4, below RS-SMC") và một dòng ở appendix.
