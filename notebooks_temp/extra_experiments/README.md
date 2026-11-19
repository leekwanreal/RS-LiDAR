# Kế hoạch thí nghiệm bổ sung: phân công và hướng dẫn chạy

> Thư mục tạm cho cộng tác. Sẽ bị loại khỏi bản code nộp kèm bài. Bản cập nhật **17/11/2026**: mọi notebook chạy bằng **Run all** và tự ngắt máy khi xong.

**Hạn chót (giờ AoE):** đăng ký abstract **14/1/2027**, nộp bài **22/1/2027**.

**Mục tiêu:** cả ba reviewer chấm 7/10. Phần viết và lý thuyết đã xong; điểm bây giờ phụ thuộc vào các thí nghiệm E1–E7 (E5 đã bỏ vì giới hạn GPU), mỗi thí nghiệm trả lời một câu hỏi reviewer chắc chắn sẽ hỏi.

**Ý tưởng của bài trong một câu:** thay vì chấm mỗi ứng viên bằng **một lần** gọi reward model, ta chấm bằng **trung bình reward trên M bản nhiễu Gauss** của ảnh (điểm làm trơn R̄σ). Áp dụng cho LiDAR (RS-LiDAR) và SMC (RS-SMC).

**Phân công:** **Hiếu** làm T1, T3, T5, T7 ([mục 4A](#4a-việc-của-hiếu)); **Quân** làm T2, T4, T6, T8, T9 ([mục 4B](#4b-việc-của-quân)); **tác giả** làm T10 ([mục 4C](#4c-việc-của-tác-giả)). Mỗi người chỉ cần đọc phần của mình, làm theo thứ tự.

---

## Mục lục

0. [Đọc file này thế nào](#0-đọc-file-này-thế-nào)
1. [Bảng phân công và số run](#1-bảng-phân-công)
2. [Chuẩn bị một lần](#2-chuẩn-bị-một-lần)
3. [Cách chạy một notebook](#3-cách-chạy-một-notebook)
4. Từng việc: [4A. Hiếu](#4a-việc-của-hiếu) · [4B. Quân](#4b-việc-của-quân) · [4C. Tác giả](#4c-việc-của-tác-giả)
5. [Cổng quyết định sau E4](#5-cổng-quyết-định)
6. [Quy tắc chung](#6-quy-tắc-chung)
7. [Tham khảo: chi tiết thí nghiệm E1–E7](#7-tham-khảo-chi-tiết-thí-nghiệm)

---

## 0. Đọc file này thế nào

1. Làm [mục 2](#2-chuẩn-bị-một-lần) **một lần** trước khi chạy bất cứ thứ gì.
2. Đọc [mục 3](#3-cách-chạy-một-notebook) (cách chạy chung cho mọi notebook), rồi đọc phần việc của mình ở mục 4.
3. Mỗi việc viết theo **cùng một khuôn**:

| Phần | Nội dung |
|---|---|
| Thẻ việc | 👤 người chạy · 🧪 thí nghiệm · 📓 notebook (bấm vào để mở file trong repo) · 🖥️ GPU · ⏱️ thời gian · 🔗 việc phải xong trước · 📤 thứ phải nộp |
| 🎯 Mục tiêu | Việc này trả lời câu hỏi gì, notebook làm gì |
| ▶ Các bước | Thao tác bấm, rồi các chỗ cần **xem lại và chụp màn hình** sau khi Run all chạy xong (đánh dấu ⏸) |
| 📦 Đầu ra | Notebook in ra gì, ghi file gì, mỗi bảng/cột nghĩa là gì |
| 🌟 Kết quả lý tưởng | Kết quả tốt trông thế nào, kèm **mốc kiểm tra** để phát hiện chạy sai |
| ⚠️ Lưu ý | Lỗi hay gặp và cách xử lý |
| 📊 Đọc kết quả và làm gì tiếp | Dòng **✅** = tốt, **⚠️** = lẫn lộn, **❌** = xấu; mỗi dòng nói nghĩa là gì và việc tiếp theo |

---

## 1. Bảng phân công

Thời gian là **ước lượng**, chưa đo trên Colab. Cơ sở: bảng latency ở Appendix E của bài (A100): sinh 50 ứng viên SD 1.5 mất 5.7 s mỗi prompt, chấm IR 50 ứng viên 0.65 s, sinh 4 ảnh DDPM-100 mất 7.1 s. L4 chậm hơn A100 khoảng 2–3 lần.

**Số run sinh ảnh cuối (Phase 2) của từng thí nghiệm:**

| Thí nghiệm | Việc | Run Phase 2 mới | Gồm những gì |
|---|---|---|---|
| E2 | T1 | 0 hoặc 2 | Phân tích hai pool Table 1 và chấm ứng viên thắng; mục 8 tự chạy hai run mốc LiDAR / RS-LiDAR λ = 5000 nếu Drive không còn run Table 1 (lưu tên `E1_…`, E1 dùng lại) |
| E1 | T3 | 4 | LiDAR λ = 100, λ = 1000; LiDAR-Clip q = 0.8, q = 0.95 |
| E3 | T5 | 1 | LiDAR K = 200 (trước đó sinh 3 pool ứng viên mới: seed 200, 300, 400) |
| E4 | T6 | 2 | LiDAR (reward 0/1) và RS-LiDAR (R̄σ); tự dừng trước Phase 2 nếu dưới 2 task qua chạy thử |
| E4 dự phòng | T7 | 0 hoặc 2 | Chỉ khi [cổng E4](#5-cổng-quyết-định) yêu cầu |
| E5 | — | 0 | **Đã bỏ** (17/11) |
| E6 | T2, T4, T8, T9 | 2 hoặc 3 | FLUX LiDAR (kèm sinh pool), RS-FlowLiDAR σ = 0.25; Vanilla chỉ khi Drive chưa có; σ = 1.0 **đã bỏ**; T9 chỉ chấm điểm |
| E7 | T10 | 2 (+ tối đa 1) | SMC N = 4, λ_FK = 3 và λ_FK = 30; thêm 1 run nếu đạt điều kiện ở thẻ T10 |

### 👤 Hiếu: T1 → T3 → T5 → T7

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T1**](#t1) | E2 · flip rate | L4 | ~1–2.5 h | — |
| [**T3**](#t3) | E1 · nhiệt độ λ và cắt reward | L4 (hoặc A100) | ~15–25 h L4 (~6–10 h A100) | nên sau T1 |
| [**T5**](#t5) | E3 · cùng số lần gọi reward | L4 | ~9–14 h | — |
| [**T7**](#t7) | E4 dự phòng, **chỉ khi** cổng E4 yêu cầu | L4 | 0 hoặc ~8–12 h | T6 + [cổng E4](#5-cổng-quyết-định) |

### 👤 Quân: T2 → T4 → T6 → T8 → T9

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T2**](#t2) | E6 · FLUX notebook 1 (LiDAR, sinh pool) | **A100** | ~10–14 h | `HF_TOKEN` |
| [**T4**](#t4) | E6 · FLUX notebook 2 (RS-FlowLiDAR σ = 0.25) | **A100** | ~8–10 h | T2 |
| [**T6**](#t6) | E4 · reward rời rạc | L4 | ~8–12 h | — |
| [**T8**](#t8) | E6 · FLUX notebook 3 (Vanilla), **bỏ qua nếu Drive đã có** | **A100** | 0 hoặc ~5–8 h | T4 |
| [**T9**](#t9) | E6 · chấm GenEval cho FLUX | L4 | ngắn (chỉ chấm điểm, chưa đo) | T2, T4 |

### 👤 Tác giả: T10

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T10**](#t10) | E7 · SMC ở nhiệt độ λ_FK khác | L4 | 2 run (+ tối đa 1), ~2–4 h L4 mỗi run | — |

Việc của hai người chỉ phụ thuộc nhau ở một chỗ: **T7** (Hiếu) chờ **T6** (Quân) xong và qua [cổng E4](#5-cổng-quyết-định).

**Khi thiếu GPU, ưu tiên:** E2 > E1 > E3 > E7 > E4 > E6.

**Ngân sách Colab** (tốc độ tiêu thụ đo thực tế 16/11/2026: **L4 1.94 unit/giờ, A100 6.5 unit/giờ**; số giờ là ước lượng):

| Người | Giờ GPU | Unit |
|---|---|---|
| Hiếu | ~25–40 h L4 (+ 8–12 h nếu có T7) | ~50–80 (+ 15–25) |
| Quân | ~18–24 h A100 (+ 5–8 h nếu chạy Vanilla) và ~10–14 h L4 | ~140–180 (+ 30–50) |
| Tác giả | 2–3 run, ~4–12 h L4 | ~8–23 |

**Chọn GPU để tiết kiệm unit:** dùng **L4 ở mọi việc chạy được trên L4** (mọi thí nghiệm SD 1.5 và chấm điểm). L4 chậm hơn A100 khoảng 2–3 lần nhưng tiêu thụ 1.94 so với 6.5 unit/giờ, nên rẻ hơn khoảng 1.1–1.7 lần. Chỉ FLUX (T2, T4, T8) bắt buộc A100: riêng phần transformer của FLUX.1-dev đã khoảng 24 GB ở bf16, vượt bộ nhớ của L4.

Colab Pro có 100 unit/tháng; Pro+ có 500–600 unit/tháng (xem trang đăng ký). Tốc độ tiêu thụ xem ở Colab → biểu tượng RAM/Disk góc phải → **View resources**.

---

## 2. Chuẩn bị một lần

### 2.1 Dùng chung Google Drive (bắt buộc)

Mọi notebook đọc và ghi vào `MyDrive/RS-LiDAR/`. E1–E4 **dùng lại pool lookahead của Table 1**, nằm trong thư mục `RS-LiDAR` của chủ dự án.

1. **Người giữ thư mục gốc:** chia sẻ thư mục `RS-LiDAR` trên Google Drive cho người còn lại, quyền **Editor**.
2. **Người được chia sẻ:** vào **Được chia sẻ với tôi** → chuột phải `RS-LiDAR` → **Sắp xếp → Thêm lối tắt** → chọn **Drive của tôi** (gốc). Tên lối tắt phải đúng là `RS-LiDAR`.
3. **Làm bước 2 TRƯỚC khi mở bất kỳ notebook nào.** Nếu chạy notebook trước, nó tự tạo một thư mục `RS-LiDAR` rỗng; khi đó lối tắt bị trùng tên và notebook ghi vào thư mục rỗng.
4. Kiểm tra sau khi notebook nối Drive (mục 2 của notebook): `os.listdir("/content/drive/MyDrive/RS-LiDAR/Lookahead_samples")` phải thấy các pool của Table 1.

**Dung lượng:** file do ai tạo thì **tính vào Drive của người đó**. Ước lượng:

| Người | Tạo ra | Dung lượng | Gói Drive cần |
|---|---|---|---|
| Quân | 2 pool FLUX (notebook 1, 2; pool RS lưu một bản latents riêng, ~14 GB mỗi pool) + ảnh 1024² của các run | ~30–35 GB | 100 GB |
| Hiếu | Ảnh của 5 run SD 1.5 (E1, E3), 3 pool mới của E3 | ~10–15 GB | 100 GB (gói miễn phí 15 GB sát giới hạn) |

Đầy Drive thì notebook lỗi khi ghi file giữa chừng; kiểm tra dung lượng trống trước mỗi việc lớn.

### 2.2 Hugging Face token (chỉ người chạy E6)

1. Vào https://huggingface.co/black-forest-labs/FLUX.1-dev, bấm chấp nhận điều khoản.
2. Tạo token đọc (Settings → Access Tokens).
3. Trên Colab: biểu tượng 🔑 (**Secrets**) ở thanh bên trái → thêm `HF_TOKEN` → bật **Notebook access**. Notebook tự đọc token từ đây, không hỏi khi chạy.

### 2.3 Chọn GPU

| GPU | Dùng cho |
|---|---|
| **A100** (40 GB, bật **High-RAM**) | **Chỉ** FLUX (T2, T4, T8): bắt buộc, vì FLUX.1-dev không vừa 24 GB |
| **L4** (24 GB) | Mọi việc còn lại: T1, T3, T5, T6, T7, T9, T10. Đủ VRAM: SD 1.5 với 50 ứng viên cần khoảng 9 GiB. Rẻ hơn A100 tính theo unit |
| Tesla T4 (16 GB) | Không dùng |

### 2.4 Môi trường

- Cell cài đặt của E1–E4 chép nguyên từ notebook Table 1 (đã chạy được trên Colab). Thông báo xung đột phụ thuộc của pip (TensorFlow, google-cloud, protobuf, sentence-transformers, gradio, open-clip/timm) là bình thường, không cần xử lý.
- **Colab bản mới (từ 11/2026: Python 3.13):** bản `peft` và `wandb` cài sẵn không tương thích với `transformers==4.38.2` và `protobuf` cũ mà notebook cài, nên `import ImageReward` lỗi (`cannot import name 'EncoderDecoderCache'` hoặc `cannot import name 'Imports' from 'wandb.proto…'`). Mọi notebook (E1–E6, E7, Table 2 SMC) đã gọi `setup_runtime()` của repo trước khi import ImageReward để xử lý hai lỗi này; script chạy thực nghiệm cũng gọi hàm này. **Không** tự `pip install` thêm hay đổi bản `peft`/`wandb`/`protobuf`. Nếu vẫn gặp lỗi: kiểm tra đang dùng bản notebook mới nhất trong repo, rồi **Runtime → Restart session** và **Run all**.
- Ngay sau cài đặt có **cell kiểm tra môi trường**: phải in ✅ cho `hpsv2`, `clip`, `open_clip`, `ImageReward`, `timm`, `google.protobuf`. Có ❌ thì notebook dừng ở đó: **Runtime → Restart session** rồi **Run all** lại.
- Code chấm điểm **báo lỗi** khi HPS/CLIP không chạy được (không âm thầm ghi 0). Cell GenEval dừng nếu `open_clip` không nạp được.
- FLUX (E6): xem lưu ý ở [T2](#t2).

---

## 3. Cách chạy một notebook

Áp dụng cho mọi việc; phần riêng của từng việc nằm ở mục 4.

1. **Lấy notebook từ repo, không mở thẳng từ GitHub** (kiểu link `colab.research.google.com/github/...`). Mở từ GitHub thì phiên Colab thường không tự kết thúc. Cách đúng:
   - `git pull` trong bản clone của repo (thư mục `notebooks_temp/extra_experiments/`), hoặc bấm vào tên notebook ở thẻ việc để mở file trong repo rồi tải file đó về.
   - Trên Colab: **File → Upload notebook** → chọn file.
   - Khi nhóm báo notebook có bản mới thì lấy lại file. Code `.py` thì không cần: notebook tự tải bản mới nhất từ GitHub khi chạy.
2. **Runtime → Change runtime type**: chọn GPU theo thẻ việc. Với A100, bật **High-RAM**.
3. **Không sửa biến nào**, trừ khi thẻ việc ghi rõ.
4. **Runtime → Run all** (Ctrl + F9). Khi Colab hỏi quyền truy cập Google Drive: cho phép, bằng tài khoản có lối tắt `RS-LiDAR`.
5. Notebook chạy hết rồi **tự ngắt máy** ở cell cuối (`AUTO_DISCONNECT = True`): đẩy hết dữ liệu lên Drive rồi tắt runtime, nên không tốn compute unit lúc rảnh. Output vẫn hiện trong tab trình duyệt để xem lại và chụp màn hình (các bước ⏸ trong thẻ việc). Muốn giữ máy sau khi chạy thì đặt `AUTO_DISCONNECT = False` ở cell cuối trước khi Run all.
6. **Notebook dừng vì lỗi** thì máy **không** tự ngắt: đọc lỗi, gửi nhóm, rồi tự ngắt (**Runtime → Disconnect and delete runtime**).
7. **Bị ngắt kết nối giữa chừng:** upload lại notebook (nếu cần), chọn GPU, **Run all** lại. Phần đã xong tự bỏ qua. Ngoại lệ: run Phase 2 dùng seed toàn cục (E1, E3) bị ngắt giữa run thì notebook in đường dẫn và dừng; **xoá thư mục đó trên Drive** rồi Run all lại. Vì vậy với E1, E3: chỉ bắt đầu một phiên khi còn đủ thời gian cho trọn một run Phase 2.
8. **Lỗi:** gửi nhóm **traceback đầy đủ** + tên notebook + số mục.
9. **Nộp kết quả:** file nằm ở `MyDrive/RS-LiDAR/extra_experiments/<tên thí nghiệm>/` (FLUX notebook 1–3: `MyDrive/RS-LiDAR/results/`; chấm GenEval FLUX: `extra_experiments/E6_flux_geneval/`). Gửi tên file hoặc ảnh chụp vào nhóm chat.

---

<a id="4-từng-việc"></a>
## 4. Từng việc

**Mốc kiểm tra dùng chung cho E1–E4** (Table 1, SD 1.5, DDPM-100, run seed 100, λ = 5000, chấm bằng đúng các hàm trong notebook). Dòng mốc trong bảng kết quả của bạn phải gần các số này; lệch quá khoảng 0.02 thì gần như chắc chắn đang đọc nhầm pool hoặc nhầm run, phải báo nhóm trước khi phân tích tiếp.

| Run | IR | CLIP | HPS | GenEval |
|---|---|---|---|---|
| LiDAR | 0.350 | 0.277 | 0.268 | 0.481 |
| RS-LiDAR (σ = 1, M = 4) | 0.383 | 0.278 | 0.269 | 0.499 |

**Cách đọc bảng kết quả `summarize`** (E1, E3, E4 và T9 đều in cùng dạng). Mỗi dòng là một run; cột `IR`, `CLIP`, `HPS`, `AS` (Aesthetic), `GenEval` là trung bình trên các prompt. Cột `ΔIR [95% CI]` là chênh lệch so với **run mốc** (ghi ở dòng đầu ra `Δ = run − <mốc>`), ví dụ `+0.0330 [+0.0250, +0.0410]*`:
- số đầu là chênh lệch trung bình;
- trong ngoặc là khoảng tin cậy 95% (bootstrap theo prompt, ghép cặp);
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
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | ~1 h; thêm ~1.5 h nếu phải chấm lại pool RS (xem mục 5 của notebook) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp output mục 5, 6, 7, 8; ba file `E2_per_prompt.csv`, `E2_flipped_winner_quality.csv`, `E2_gain_by_flip.csv` |

**🎯 Mục tiêu.** Với LiDAR ở λ = 5000, ứng viên điểm cao nhất gần như quyết định ảnh cuối. Notebook đếm xem trên bao nhiêu prompt, ứng viên điểm cao nhất theo reward thô R **khác** ứng viên điểm cao nhất theo reward làm trơn R̄σ ("flip"), và kiểm tra ứng viên mới có thật sự tốt hơn trên các metric không được tối ưu (HPS, GenEval). Không sinh ảnh cuối mới: chỉ dùng hai pool của Table 1 (cùng 50 ứng viên, một pool chấm bằng R, một pool chấm bằng R̄σ).

**▶ Các bước**

1. Lấy notebook, upload lên Colab, chọn **L4 GPU** ([mục 3](#3-cách-chạy-một-notebook)).
2. **Runtime → Run all**, không sửa gì.
3. Khi xong, xem lại và chụp:
   - ⏸ **Mục 3:** cell kiểm tra môi trường in ✅ cho mọi thư viện.
   - ⏸ **Mục 4:** dòng `E2_flip_rate: 553 prompt | kết quả tổng hợp lưu tại …/RS-LiDAR/extra_experiments/E2_flip_rate`.
   - ⏸ **Mục 5:** `Pool LiDAR :` và `Pool RS-LiDAR :` là đường dẫn (không phải `None`), có dòng `✅ Pool RS dùng chung latents với pool LiDAR: …`. Nếu là `⚠️ … KHÔNG dùng chung latents …` thì notebook đã tự chấm lại (thêm ~1.5 h); **báo nhóm ngay** (xem Lưu ý).
   - ⏸ **Mục 6:** toàn bộ output.
   - ⏸ **Mục 7, 8:** hai bảng.
4. Gửi nhóm 3 file CSV trong `MyDrive/RS-LiDAR/extra_experiments/E2_flip_rate/` và các ảnh chụp.

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 6 | `Số prompt phân tích: N` | Số prompt ghép được hai pool. Phải là 553 |
| 6 | `Flip rate (argmax R ≠ argmax R̄σ): x%` | Tỉ lệ prompt có ứng viên thắng bị đổi |
| 6 | `Spearman(R, R̄σ) trung bình trong prompt: s` | Mức giống nhau giữa thứ hạng theo R và theo R̄σ (1 = y hệt) |
| 6 | `Hạng của ứng viên thắng theo R khi xếp theo R̄σ (trung vị, chỉ prompt flip): r / 50` | Trên các prompt flip, ứng viên thắng cũ bị R̄σ xếp tụt xuống hạng mấy |
| 6 | 3 dòng `thắng theo R`, `thắng theo R̄σ`, `nền hoán vị` + dòng `Wilcoxon … p = …` | Độ "mong manh" (gap = R − R̄σ, quy ra phân vị trong pool; 0.5 = điển hình, 1 = mong manh nhất) của ứng viên thắng, so với mức nền do may rủi |
| 6 | Bảng theo task và file `E2_per_prompt.csv` | Như trên, tách theo loại prompt |
| 7 | Dòng `Kiểm tra decode: |IR chấm lại − IR đã lưu| trung vị …` | IR của ảnh decode phải khớp IR trong pool (trung vị < 0.05, nếu không notebook dừng): xác nhận đúng thứ tự ứng viên và đúng decode |
| 7 | Bảng `metric · thắng theo R · thắng theo R̄σ · Δ (R̄σ − R) [95% CI] · R̄σ thắng · hoà`, file `E2_flipped_winner_quality.csv` | Trên các prompt flip: chất lượng ảnh của ứng viên thắng mới so với cũ |
| 8 | Bảng `nhóm (flip / không flip) · n · ΔIR · ΔHPS · ΔCLIP · ΔGenEval`, file `E2_gain_by_flip.csv` | Lợi ích của RS-LiDAR so với LiDAR (Table 1) nằm ở prompt flip hay không |

**🌟 Kết quả lý tưởng**

- **Mục 6:** `Số prompt phân tích: 553`; **flip rate ≥ 15%**; Spearman **dương rõ nhưng nhỏ hơn 1** (gần 0 nghĩa là R̄σ gần như ngẫu nhiên → báo nhóm); ở prompt flip, ứng viên thắng theo R tụt sâu khi xếp theo R̄σ trong khi ứng viên thắng theo R̄σ vẫn ở gần đầu theo R. Các dòng phân vị gap, `nền hoán vị` và Wilcoxon chỉ để mô tả, **không** dùng làm bằng chứng spike: `thắng theo R̄σ` < `thắng theo R` luôn đúng theo đại số ở prompt flip, và nền hoán vị chỉ đo gap có gắn với R̄σ hay không.
- **Mục 7:** HPS và GenEval có Δ dương, **cả hai cận của khoảng tin cậy đều dương**; cột `R̄σ thắng` trên 50%. Dòng `IR (tối ưu, thiên về R)` có Δ **âm** là **bình thường** (ứng viên thắng theo R luôn có IR cao nhất theo định nghĩa), không dùng để kết luận. CLIP và Aesthetic không giảm có ý nghĩa.
- **Mục 8:** ΔIR và ΔGenEval ở nhóm **flip** lớn hơn rõ so với nhóm **không flip**; nhóm không flip có Δ gần 0. Cột `n` của nhóm flip khớp với flip rate × 553.

**⚠️ Lưu ý**

- Nếu mục 5 báo ⚠️, notebook tự chấm lại pool (thư mục `..._rescore`), chỉ tốn một lần vì E1 (T3) dùng lại. **Báo nhóm ngay**, vì khi đó câu "LiDAR and RS-LiDAR share the same candidates" ở appendix phải xem lại.
- Mục 8 dùng hai run Table 1 nếu còn trên Drive và khớp cấu hình. Nếu không, nó tự chạy hai run mốc λ = 5000 (khoảng 2 giờ A100, lâu hơn trên L4) với tên `E1_LiDAR_…` và `E1_RSLiDAR_…`; T3 (E1) chạy sau sẽ thấy "✅ Phase 2 đã đủ" và dùng lại. Hai run này dùng seed toàn cục: nếu bị ngắt giữa chừng, xoá thư mục dở dang trong `Target_samples/` rồi chạy lại mục 8.
- Chỉ thiếu mục 8 (đã có `E2_per_prompt.csv` và `E2_flipped_winner_quality.csv`): chạy các cell từ đầu đến hết mục 6, ở mục 7 chỉ chạy cell `# GenEval chính thức`, rồi chạy mục 8.
- Chạy T1 **trước** T3 để hai notebook không chấm lại pool RS cùng lúc.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Flip rate ≥ 15%, và ở mục 7 HPS **và** GenEval có Δ dương với CI không chứa 0 | Có bằng chứng trực tiếp cho cơ chế của bài | Gửi kết quả; số này vào Section 6.3 |
| ⚠️ Flip rate ≥ 15% nhưng mục 7 có CI chứa 0 | Smoothing có đổi lựa chọn, nhưng chưa chứng minh được lựa chọn mới tốt hơn | Trong bài chỉ giữ flip rate, bỏ chữ "better" |
| ❌ Flip rate < 15% | Spike hiếm; câu chuyện "spike" yếu | Báo nhóm. E1 và E4 trở nên quan trọng hơn |

---

<a id="t3"></a>
### T3 · Hiếu · E1: smoothing có chỉ là hạ nhiệt độ λ hoặc cắt reward ngoại lai không

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E1 (xem [tham khảo](#ref-e1)) |
| 📓 Notebook | [`E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb`](E1_temperature_clipping/E1_Temperature_Clipping_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** (A100 nhanh hơn khoảng 2–3 lần) |
| ⏱️ Thời gian | 4 run Phase 2, mỗi run ~1.5–2.5 h A100 (~4–6 h L4); tổng ~15–25 h L4. Chia nhiều phiên Colab |
| 🔗 Cần xong trước | Nên sau T1 |
| 📤 Nộp lại | Ảnh chụp mục 5, 6, 9; `E1_summary.csv`, `E1_summary_vs_RSLiDAR.csv`; thời gian thực của run đầu tiên |

**🎯 Mục tiêu.** Làm trơn kéo các điểm reward lại gần nhau, trông giống như hạ nhiệt độ λ của LiDAR hoặc cắt bớt điểm ngoại lai. Notebook kiểm tra xem LiDAR có bắt kịp RS-LiDAR chỉ bằng cách đó không. 4 run, cùng 50 ứng viên của Table 1:
- **LiDAR λ = 100** (giữa vùng chuyển tiếp) và **LiDAR λ = 1000** (sát vùng bão hoà). λ thấp hơn gần như không dẫn hướng; Figure 8 của bài LiDAR đã cho thấy IR ở đó thấp hẳn, nên không chạy.
- **LiDAR-Clip q = 0.8 và q = 0.95**: điểm của mỗi prompt bị cắt ở phân vị q (`min(r, quantile(r, q))`), khoảng 10 hoặc 2–3 ứng viên đầu hoà nhau, λ = 5000. Hai mức kẹp hai bên để kết luận không phụ thuộc ngưỡng.
- Mốc LiDAR và RS-LiDAR λ = 5000 lấy từ run Table 1 trên Drive (không chạy lại).

**▶ Các bước**

1. Lấy notebook (bản 17/11), upload lên Colab, chọn **L4 GPU** ([mục 3](#3-cách-chạy-một-notebook)).
2. **Runtime → Run all**, không sửa gì. Notebook chạy mục 1–9 (4 run Phase 2, chấm GenEval, bảng) rồi tự ngắt máy.
3. **Ghi lại thời gian** của run đầu tiên: từ dòng `▶ <tên run đầu>` đến dòng `▶ <tên run thứ hai>` (gửi nhóm để chỉnh ước lượng).
4. Khi xong, xem lại và chụp:
   - ⏸ **Mục 3–4:** ✅ hết; dòng `E1_temperature_clipping: 553 prompt | …`.
   - ⏸ **Mục 5** (vài phút CPU): `Pool LiDAR :` và `Pool RS-LiDAR :` là đường dẫn; dòng `79 bước có guidance: t = 990 … 210`; bảng độ mạnh dẫn hướng (λ = 1 … 20000); ba dòng `LiDAR đạt 25% / 50% / 90% độ mạnh của λ = 5000 từ λ ≈ …`.
   - ⏸ **Mục 6:** `Mốc LiDAR λ=5000 :` và `Mốc RS-LiDAR λ=5000 :` là đường dẫn (nếu là `None`, notebook tự chạy lại mốc, thêm 1–2 run); hai dòng `📁 LiDAR_winsor0.8: 553 prompt`, `📁 LiDAR_winsor0.95: 553 prompt`; danh sách đúng **4 run**.
   - ⏸ **Mục 9:** bảng kết quả.
5. **Kiểm tra dải λ** từ ba dòng ở mục 5:

   | Ba dòng in ra | Làm gì |
   |---|---|
   | Mốc 50% ở λ trong khoảng 30–300 **và** mốc 90% ở λ trong khoảng 300–3000 | Đúng như dự kiến, không làm gì |
   | Một trong hai mốc nằm ngoài khoảng trên | Báo nhóm trước khi gửi kết quả. Nhóm có thể đổi `LAMBDAS_LIDAR` (mục 6) thành [giá trị gần mốc 50%, gần mốc 90%] rồi Run all lại; run đã xong được giữ |

6. Gửi `MyDrive/RS-LiDAR/extra_experiments/E1_temperature_clipping/E1_summary.csv` và các ảnh chụp.

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 5 | Bảng cột `LiDAR: ‖wʳ−w‖₁` | Độ mạnh dẫn hướng ở mỗi λ: 0 = không dẫn hướng, 2 = chuyển hẳn sang ứng viên khác |
| 5 | Cột `LiDAR: % bước đổi ứng viên` | Tỉ lệ bước mà reward đổi được ứng viên chi phối |
| 5 | Hai cột `RS-LiDAR: …` | Như trên nhưng với R̄σ |
| 9 | Bảng `summarize`, mốc là `LiDAR λ=5000`, file `E1_summary.csv` | Các dòng: `LiDAR λ=5000`, `RS-LiDAR λ=5000`, `LiDAR λ=100`, `LiDAR λ=1000`, `LiDAR-Clip q=0.8`, `LiDAR-Clip q=0.95` |
| 9 | Bảng thứ hai, mốc là `RS-LiDAR λ=5000`, file `E1_summary_vs_RSLiDAR.csv` | Cùng các dòng, nhưng Δ = run − RS-LiDAR. **Đây là bảng để kết luận E1**: mọi dòng LiDAR và LiDAR-Clip cần Δ âm |
| 5, 6 | Các dòng `✅ Run mốc khớp cấu hình: …` hoặc `⚠️ … không khớp cấu hình …` | Notebook đọc `args.json` của run Table 1 (pool, seed, λ, K) và cấu hình smoothing của pool RS (i.i.d., σ, M). Không khớp thì tự chạy lại mốc (thêm 1–2 run): **báo nhóm** |

**🌟 Kết quả lý tưởng**

- **Mục 5:** cột `LiDAR: ‖wʳ−w‖₁` tăng dần theo λ rồi chững lại (không vượt 2).
- **Mục 9, kiểm tra trước:** `LiDAR λ=5000` có IR ≈ 0.350, GenEval ≈ 0.481; `RS-LiDAR λ=5000` có IR ≈ 0.383, GenEval ≈ 0.499.
- **Mục 9, mong muốn:** `RS-LiDAR λ=5000` có `ΔIR` và `ΔGenEval` **dương, có `*`**; **mọi** dòng `LiDAR λ=…` và `LiDAR-Clip` có IR **và** GenEval **thấp hơn** `RS-LiDAR λ=5000`; các dòng `LiDAR λ=…` không vượt `LiDAR λ=5000` rõ rệt; CLIP và HPS của RS-LiDAR không thấp hơn LiDAR có ý nghĩa.

**⚠️ Lưu ý**

- Run của E1 dùng seed toàn cục để ghép cặp với Table 1. Bị ngắt giữa một run thì notebook **từ chối chạy tiếp** run đó và in đường dẫn: xoá thư mục đó trên Drive rồi Run all lại. Trên L4 một run mất ~4–6 h, nên chỉ bắt đầu phiên mới khi còn đủ thời gian.
- **Chạy thử (tuỳ chọn):** đặt `PROMPT_MODE = "100_REPRESENTATIVE"` ở mục 4 (khoảng 1/5 thời gian). Bản thử ghi vào thư mục riêng (hậu tố `_100p`). Nhớ đổi lại `"ALL"` khi chạy đủ.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ RS-LiDAR λ = 5000 cao hơn mọi LiDAR λ và cả hai LiDAR-Clip trên IR và GenEval, có `*` | Smoothing không phải hạ nhiệt độ hay cắt ngoại lai | Gửi CSV; viết vào Section 6 |
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
| 📓 Notebook | [`E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) (tên file giữ chữ K100 cho link không đổi; notebook chỉ chạy K = 200) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | ~9–14 h L4 (~4–5 h A100). Khoảng 18–27 unit |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp mục 5, 7, 8; `E3_summary.csv` |

**🎯 Mục tiêu.** RS-LiDAR chấm mỗi ứng viên 4 lần (50 ứng viên × 4 = **200 lần gọi reward** mỗi prompt), còn LiDAR chỉ chấm 50 lần. Câu reviewer sẽ hỏi: *cho LiDAR cũng 200 lần gọi reward, tức 200 ứng viên, thì LiDAR có bắt kịp không?* Notebook tự làm 3 việc:

| Việc | Ở mục | Nội dung | Thời gian (L4, ước lượng) |
|---|---|---|---|
| 1. Sinh ứng viên | 5 | 3 lần, mỗi lần 50 ứng viên mỗi prompt (seed 200, 300, 400), lưu lên Drive. Dùng lại 50 ứng viên seed 100 của Table 1 nên chỉ phải sinh mới 150 | ~2–3 h mỗi lần |
| 2. Gộp | 6 | Gộp 4 bộ thành 200 ứng viên mỗi prompt, lưu trên đĩa Colab | vài phút |
| 3. Sinh ảnh cuối | 7 | LiDAR dùng 200 ứng viên (λ = 5000, DDPM-100, 4 ảnh/prompt). Hai mốc (LiDAR và RS-LiDAR 50 ứng viên) lấy từ run Table 1 trên Drive | ~2.5–4 h |

Sau đó mục 8 in bảng so sánh và cell cuối tự ngắt máy.

**▶ Các bước**

1. Lấy notebook, upload lên Colab, chọn **L4 GPU** ([mục 3](#3-cách-chạy-một-notebook)).
2. **Runtime → Run all**, không sửa gì.
3. Khi xong, xem lại và chụp:
   - ⏸ **Mục 3–4:** ✅ hết; dòng `E3_compute_matched: 553 prompt | …`.
   - ⏸ **Mục 5:** ba dòng `▶ Pool seed 200`, `▶ Pool seed 300`, `▶ Pool seed 400`; dòng cuối là danh sách 4 thư mục (seed 100, 200, 300, 400).
   - ⏸ **Mục 7:** dòng đầu có `LiDAR K=50` và `RS-LiDAR K=50 M=4` trỏ tới thư mục (không phải `None`), sau đó là `▶ LiDAR K=200`.
   - ⏸ **Mục 8:** bảng kết quả.
4. Gửi nhóm ảnh chụp và `MyDrive/RS-LiDAR/extra_experiments/E3_compute_matched/E3_summary.csv`.

**📦 Bảng kết quả (mục 8, `E3_summary.csv`)**, mốc là `RS-LiDAR K=50 M=4`:

| Dòng | Là gì |
|---|---|
| `RS-LiDAR K=50 M=4` | Mốc: RS-LiDAR của Table 1 (200 lần gọi reward) |
| `LiDAR K=50` | LiDAR của Table 1 (50 lần gọi reward) |
| `LiDAR K=200` | LiDAR với 200 ứng viên (200 lần gọi reward), run mới |

**🌟 Kết quả lý tưởng**

- **Kiểm tra mốc:** `LiDAR K=50` có IR ≈ 0.350, GenEval ≈ 0.481; `RS-LiDAR K=50 M=4` có IR ≈ 0.383, GenEval ≈ 0.499.
- **Kiểm tra hợp lý:** IR của `LiDAR K=200` **cao hơn** `LiDAR K=50` (nhiều ứng viên hơn thì dẫn hướng tốt hơn). Nếu thấp hơn, có thể gộp sai: báo nhóm.
- **Mong muốn:** dòng `LiDAR K=200` có `ΔIR` và `ΔGenEval` **âm hoặc bằng 0**.

**⚠️ Lưu ý**

- Tổng thời gian dài nên có thể cần 2 phiên Colab. Phần sinh ứng viên bị ngắt thì Run all lại là chạy tiếp. Run `LiDAR K=200` dùng seed toàn cục: bị ngắt giữa run thì notebook in đường dẫn và dừng; xoá thư mục đó trên Drive rồi Run all lại. Chỉ bắt đầu phiên chạy Phase 2 khi còn ít nhất ~4–5 h.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả của dòng `LiDAR K=200` | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ ΔIR ≤ 0 và ΔGenEval ≤ 0 | RS-LiDAR thắng ngay cả khi cùng số lần gọi reward | Claim: "At equal reward evaluations, RS-LiDAR outperforms LiDAR with K = 200" |
| ⚠️ Ngang nhau (CI chứa 0) | Chất lượng như nhau, nhưng RS-LiDAR chỉ cần sinh 50 ứng viên thay vì 200 | Claim về chi phí: cùng chất lượng, sinh ứng viên rẻ hơn 4 lần, không tăng VRAM |
| ❌ Dương, có `*` | Một phần lợi ích đến từ việc gọi reward nhiều hơn | Ghi trung thực; chỉ giữ claim về chi phí |

---

<a id="t7"></a>
### T7 · Hiếu · E4 dự phòng (chỉ khi cổng E4 yêu cầu)

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E4 dự phòng (xem [tham khảo E4](#ref-e4)) |
| 📓 Notebook | [`E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb`](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) (cùng notebook với T6) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | ~8–12 h L4 |
| 🔗 Cần xong trước | [T6](#t6) xong, và [cổng E4](#5-cổng-quyết-định) ghi "T7 = E4 dự phòng". Nếu không, **T7 không chạy gì** |
| 📤 Nộp lại | Ảnh chụp mục 9, 11; `E4_summary_<hậu tố>_sig0.05_M16.csv`, `E4_geneval_by_task_<hậu tố>_sig0.05_M16.csv` |

**🎯 Mục tiêu.** Chạy lại E4 với σ nhỏ hơn và nhiều lần nhiễu hơn, khi T6 chưa ra kết quả rõ. (Nhánh E5 trước đây của T7 **đã bỏ** ngày 17/11.)

**▶ Các bước**

1. Lấy notebook E4, upload lên Colab, chọn **L4 GPU**.
2. **Chỉ sửa một dòng** ở mục 9: `SIGMA_D, M_D = SIGMA_AUTO, 8` thành `SIGMA_D, M_D = 0.05, 16`. Không sửa gì khác: mục 7 tự chọn lại đúng task như T6 (kết quả chạy thử đã lưu trên Drive).
3. **Runtime → Run all**.
4. ⏸ Mục 9: dòng tổng kết từng task. Mục 11: chụp bảng.
5. Gửi hai file CSV ở thẻ.
6. Nếu vẫn không đạt và nhóm đồng ý: lặp lại với `SIGMA_D, M_D = 0.03, 16`.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ RS hơn LiDAR (0/1) về GenEval, có `*` | σ nhỏ hơn mới hợp với Mask2Former | E4 vào Section 6 |
| ❌ Vẫn không hơn | Reward 0/1 trên ứng viên DPM-5 quá thô | E4 vào appendix hoặc bỏ |

---

<a id="4b-việc-của-quân"></a>
## 4B. Việc của Quân (T2, T4, T6, T8, T9)

**Chung cho các notebook sinh ảnh FLUX (T2, T4, T8):**
- **Bắt buộc A100 + High-RAM**, và có `HF_TOKEN` trong Secrets với *Notebook access* **bật** ([mục 2.2](#22-hugging-face-token-chỉ-người-chạy-e6)).
- Kết quả nằm ở `MyDrive/RS-LiDAR/results/<RUN_NAME>/`. Mỗi prompt một thư mục con có ảnh và `results.json`.
- **Mục 10** (cell cuối) tự ngắt máy khi chạy xong, đã bật sẵn.
- Cuối mục 3 in phiên bản `transformers`, `diffusers`, `tokenizers`, `peft`, `torchao`, `accelerate`: chụp màn hình ở T2, và kiểm tra T4, T8 ra **đúng** các phiên bản đó.
- Mục 8 in sẵn số của bài LiDAR (Table 3 của họ): FLUX không dẫn hướng IR 1.019; LiDAR n = 50 IR 1.114; CLIP 0.282–0.284. Cấu hình của mình khác bài (s_flow = 3, cửa sổ t ∈ [0.95, 1], lookahead Hyper-SD 8 bước), nên **không cần trùng**; dùng để biết đúng cỡ.

<a id="t2"></a>
### T2 · Quân · E6 notebook 1: LiDAR trên FLUX (sinh pool 50 ứng viên + Phase 2)

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** (bắt buộc) |
| ⏱️ Thời gian | ~10–14 h (Phase 1 + Phase 2), chia nhiều phiên Colab |
| 🔗 Cần xong trước | Có `HF_TOKEN` ([mục 2.2](#22-hugging-face-token-chỉ-người-chạy-e6)) |
| 📤 Nộp lại | Ảnh chụp phiên bản thư viện (cuối mục 3), mục 6.1, mục 8, mục 9; tên thư mục kết quả |

**🎯 Mục tiêu.** Tạo pool 50 ứng viên cho mỗi prompt bằng FLUX, rồi sinh ảnh cuối bằng LiDAR. Pool này được notebook 2 (T4) dùng lại.

**▶ Các bước**

1. Lấy notebook, upload lên Colab, chọn **A100 GPU + High-RAM**.
2. Kiểm tra 🔑 Secrets có `HF_TOKEN`, *Notebook access* đang bật.
3. **Runtime → Run all**, không sửa gì.
4. Khi xong, xem lại và chụp:
   - Mục 1 in `✅ Tuyệt vời! GPU A100 có đủ VRAM …`; mục 3 in `✅ Môi trường đã sẵn sàng 100%!` và phiên bản thư viện.
   - ⏸ **Mục 4:** `✅ Đã đăng nhập Hugging Face thành công!`. Nếu in `⚠️ CẢNH BÁO: Thiếu HF_TOKEN!` thì notebook sẽ lỗi khi tải FLUX: làm lại bước 2 rồi Run all.
   - ⏸ **Mục 6** (Phase 1, lâu nhất) kết thúc bằng `✅ Hoàn thành Phase 1! Dữ liệu đã lưu tại: …/Lookahead_samples/FLUX_100_50_8`.
   - ⏸ **Mục 6.1:** `📊 Tìm thấy 553 prompt lookahead đã hoàn thành.`, `✅ Phase 1 phủ đủ 553/553 prompt của tập all.`, `✅ Kiểm tra toàn vẹn thành công: Latents khớp chuẩn 100% không gian FLUX.1 (16x128x128)!`. Mục 6.2 tự bỏ qua (`RUN_SFLOW_SWEEP = False`).
   - ⏸ **Mục 7** (Phase 2) kết thúc bằng `✅ Hoàn thành Phase 2! Kết quả lưu tại: …/results/FLUX_LiDAR_N1_n50_sflow3.0_w1.0-0.95_all553_pps`.
   - ⏸ **Mục 8:** bảng. **Mục 9:** lưới ảnh.
5. **Bị ngắt giữa chừng:** Run all lại; notebook tiếp đúng chỗ (seed theo prompt, `--resume`).
6. Gửi nhóm ảnh chụp và tên thư mục kết quả.

**🔧 Cấu hình (mục 5, đã đặt sẵn, không sửa):** `METHOD = "LiDAR"`, `PROMPT_PRESET = "all"`, `NUM_LOOKAHEAD = 50`, `LOOKAHEAD_BATCH = 5`, `PER_PROMPT_SEED = True`, `S_FLOW_LIDAR = 3.0`, `PHASE2_SEED = 42`, `EVAL_METRICS = "ImageReward#Clip-Score"`.

**📦 Đầu ra**

| Ở đâu | Là gì |
|---|---|
| `MyDrive/RS-LiDAR/Lookahead_samples/FLUX_100_50_8/` | Pool 50 ứng viên × 553 prompt (~14 GB) |
| `MyDrive/RS-LiDAR/results/FLUX_LiDAR_N1_n50_sflow3.0_w1.0-0.95_all553_pps/` | Ảnh cuối và `results.json` từng prompt |
| Mục 8 | Dòng `KẾT QUẢ ĐO ĐƯỢC (LiDAR, N=… samples)` với IR trung bình, và dòng `CLIP-Score (cùng run)` |

**🌟 Kết quả lý tưởng:** mục 6.1 đủ 3 dòng ✅; mục 8 `N=553 samples`, IR **cao hơn rõ** Vanilla (bài LiDAR báo Vanilla 1.019), CLIP khoảng 0.28; mục 9 ảnh rõ, đúng nội dung prompt, không có ảnh đen hay vỡ.

**⚠️ Lưu ý**

- pip có thể báo lỗi `tokenizers` (Python 3.13 của Colab không có bản dựng sẵn). Nếu notebook vẫn chạy tiếp thì bỏ qua, nhưng **phải** chụp phiên bản thư viện lần đầu. Gặp lỗi import lạ: **Runtime → Change runtime type → Runtime version 2026.07** (Python 3.12), Run all lại.
- **Chạy thử (tuỳ chọn):** `PROMPT_PRESET = "stratified_100"` (100 prompt, ~3–4 h). Bản thử nằm ở thư mục riêng; nhớ đổi lại `"all"`.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ Chạy hết 553 prompt, mục 6.1 đủ ✅, ảnh mục 9 rõ | Có pool cho notebook 2 | Làm T4 |
| ⚠️ Chạy được nhưng chậm hơn ước lượng khoảng 2 lần | E6 tốn nhiều GPU hơn dự kiến | Báo nhóm để cân lại việc |
| ❌ Lỗi môi trường hoặc hết bộ nhớ | Chưa chạy được FLUX | Gửi traceback và phiên bản thư viện; thử runtime 2026.07. Vẫn lỗi thì tạm gác E6, làm T6 trước |

---

<a id="t4"></a>
### T4 · Quân · E6 notebook 2: RS-FlowLiDAR trên cùng pool

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** |
| ⏱️ Thời gian | ~8–10 h |
| 🔗 Cần xong trước | [T2](#t2) xong (cùng `PROMPT_PRESET`) |
| 📤 Nộp lại | Ảnh chụp mục 6 và mục 8; tên thư mục kết quả |

**🎯 Mục tiêu.** Chấm lại **đúng** 50 ứng viên của notebook 1 bằng reward làm trơn (σ = 0.25, M = 4), rồi sinh ảnh cuối bằng RS-FlowLiDAR. So với T2, chỉ khác điểm của ứng viên.

**▶ Các bước**

1. Lấy notebook, upload lên Colab, chọn **A100 GPU + High-RAM**; kiểm tra `HF_TOKEN` như T2.
2. **Runtime → Run all**, không sửa gì.
3. Khi xong, xem lại và chụp:
   - Phiên bản thư viện (cuối mục 3) phải **đúng** như T2; khác thì báo nhóm.
   - ⏸ **Mục 6:** notebook nạp latents từ `Lookahead_samples/FLUX_100_50_8` rồi chấm lại, ghi vào `Lookahead_samples/FLUX_100_50_8_rs_sig0.25_M4_iid`. **Không** được thấy thanh tiến trình sinh ảnh Hyper-SD cho 50 ứng viên; nếu thấy thì báo nhóm.
   - ⏸ **Mục 6.1:** đủ 3 dòng ✅ như T2.
   - ⏸ **Mục 7** kết thúc bằng `✅ Hoàn thành Phase 2! Kết quả lưu tại: …/results/FLUX_RSLiDAR_N1_n50_sig0.25_M4_sflow3.0_w1.0-0.95_all553_pps`.
   - ⏸ **Mục 8:** bảng, đặt cạnh số của T2.
4. Gửi nhóm ảnh chụp và tên thư mục.

**🔧 Cấu hình (mục 5, không sửa):** `METHOD = "RS-FlowLiDAR"`, `PROMPT_PRESET = "all"` (phải giống T2), `SMOOTH_SIGMA = 0.25`, `SMOOTH_M = 4`, `REUSE_LIDAR_POOL = True`, `PER_PROMPT_SEED = True`, `S_FLOW_LIDAR = 3.0`, `PHASE2_SEED = 42`.

**📦 Đầu ra**

| Ở đâu | Là gì |
|---|---|
| `Lookahead_samples/FLUX_100_50_8_rs_sig0.25_M4_iid/` | Cùng 50 ứng viên, điểm R̄σ (lưu một bản latents riêng, ~14 GB) |
| `results/FLUX_RSLiDAR_N1_n50_sig0.25_M4_sflow3.0_w1.0-0.95_all553_pps/` | Ảnh cuối và `results.json` |
| Mục 8 | IR trung bình và CLIP của run RS |

**🌟 Kết quả lý tưởng:** mục 8 `N=553 samples`; IR của RS **≥** LiDAR (T2), CLIP không thấp hơn. Vì cùng ứng viên và cùng nhiễu theo prompt, chênh lệch giữa T4 và T2 là so sánh ghép cặp; khoảng tin cậy được tính ở [T9](#t9).

**⚠️ Lưu ý**

- Notebook dừng nếu pool của notebook 1 chưa đủ prompt.
- Không đổi σ hay M: cấu hình σ = 0.25, M = 4 chuyển nguyên từ SDXL.
- Phase 1 RS tự tính lại những prompt mà cache cũ không dùng đúng pool LiDAR; việc này là bình thường.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ IR của RS ≥ LiDAR (T2) | Phương pháp chuyển sang FLUX được | Chờ T9 để có CI; hàng FLUX cho bài |
| ⚠️ RS ≈ LiDAR (chênh < 0.01) | σ = 0.25 ở ảnh 1024² là nhiễu yếu | Claim "transfers without degradation" |
| ❌ RS thấp hơn rõ (> 0.03 IR) | Cấu hình chưa hợp với FLUX | Báo nhóm; ghi trung thực |

---

<a id="t6"></a>
### T6 · Quân · E4: reward rời rạc, dẫn hướng bằng verifier kiểu GenEval

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E4 (xem [tham khảo](#ref-e4)) |
| 📓 Notebook | **Khuyên dùng: 3 phần nối tiếp** (mỗi phần Run all, tự ngắt máy): [`E4_part1_pilot_select_Colab.ipynb`](E4_discrete_reward/E4_part1_pilot_select_Colab.ipynb) → [`E4_part2_score_Colab.ipynb`](E4_discrete_reward/E4_part2_score_Colab.ipynb) → [`E4_part3_phase2_tables_Colab.ipynb`](E4_discrete_reward/E4_part3_phase2_tables_Colab.ipynb). Bản một notebook vẫn giữ: [`E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb`](E4_discrete_reward/E4_DiscreteGenEvalReward_SD15_DDPM100_Colab.ipynb) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | ~8–12 h L4 (~4–6 h A100) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp mục 6, 7, 8, 9, 11; `E4_pilot.csv`, `E4_pilot_grid.png`, `E4_summary_*.csv`, `E4_geneval_by_task_*.csv` |

**🎯 Mục tiêu.** Dùng bộ chấm đúng/sai của GenEval (0 hoặc 1) làm reward. Kiểm tra xem RS-LiDAR (lấy trung bình 0/1 trên nhiều bản nhiễu, thành điểm từ 0 đến 1) có dẫn hướng tốt hơn LiDAR dùng thẳng 0/1 không. Notebook tự làm:
1. **Chạy thử** (mục 6): chấm đúng/sai 50 ứng viên của 10 prompt mỗi task.
2. **Tự chọn task** (mục 7): giữ task có ≥ 50% prompt có ít nhất một ứng viên đúng, luôn bỏ single_object. **Dưới 2 task đạt thì notebook dừng ở đây**, không tốn GPU cho phần sau.
3. **Tự chọn σ** (mục 8): σ lớn nhất trong {0.05, 0.1, 0.25, 0.5} có `R̄σ trên ứng viên đúng` ≥ 0.5 **và** `số mức R̄σ khác nhau` ≥ 4; không σ nào đạt thì dùng 0.1.
4. Chấm toàn bộ prompt đã chọn (mục 9), 2 run Phase 2 (mục 10), bảng (mục 11).

Hai quy tắc chọn đã định trước khi chạy, nên không bị coi là chọn sau khi xem kết quả.

**▶ Chạy theo 3 phần (khuyên dùng)**

Mỗi phần có cùng mục 1–4 như bản gốc (GPU, Drive và repo, cài đặt, kiểm tra môi trường có `setup_runtime()`, cấu hình Table 1). Các phần nối với nhau qua file `E4_config.json` trên Drive: Phần 1 ghi lựa chọn (task, danh sách prompt, σ), Phần 2 và 3 chỉ đọc, nên không bao giờ lệch nhau. Mỗi phần tự kiểm tra phần trước đã xong; nếu chưa, in rõ phải chạy phần nào rồi tự ngắt máy.

**Cấu hình đã chốt (18/11/2026):** Phần 1 giữ 4 task, σ = 0.25. Phần 2 và 3 chỉ chạy **two_object** (99 prompt) với **M = 4** (biến `RUN_TAGS`, `M_RUN` ở mục 5 của cả hai notebook, phải giống nhau). Lý do: chạy thử cho thấy colors gần trần (42/50 ứng viên đúng), color_attr gần như không có ứng viên đúng; two_object có 12/50 ứng viên đúng (trung vị) và bộ chấm ngưỡng 0.3 ổn định với nhiễu hơn counting (ngưỡng 0.9). M = 4 mô phỏng từ dữ liệu hiệu chỉnh giữ nguyên nhóm ứng viên dẫn đầu của M = 8 (Jaccard 0.93–0.98 trên two_object và counting). Phần 2 khoảng 6.5 giờ, Phần 3 khoảng 1–1.5 giờ trên L4.

| Phần | Làm gì | GPU | Thời gian | Đọc kết quả trên Drive |
|---|---|---|---|---|
| 1 | Chạy thử 10 prompt mỗi task, chọn task, chọn σ | L4 (T4 được) | ~1–1.5 h | `E4_pilot.csv`, `E4_pilot_grid.png`, `E4_sigma_calibration_*.csv`, `E4_config.json` |
| 2 | Chấm 0/1 và R̄σ cho mọi prompt đã chọn | L4 (hoặc T4 nếu đo thấy rẻ hơn) | vài giờ | `scores_sig<σ>_M8/`, `E4_scores_summary_*.csv` |
| 3 | Pool dẫn xuất, 2 run Phase 2, bảng | L4 | vài giờ | `E4_summary_*.csv`, `E4_geneval_by_task_*.csv` |

- Phần 1 kết luận dưới 2 task qua chạy thử: config ghi `status: stop`, Phần 2 và 3 sẽ báo dừng ngay. Gửi nhóm `E4_pilot.csv` và lưới ảnh.
- Bị ngắt giữa chừng ở phần nào thì Run all lại đúng phần đó; điểm và ảnh đã lưu trên Drive được dùng lại.
- Chạy lại Phần 1 sau khi đã có config thì dùng lại σ đã chọn, không hiệu chỉnh lại.

**▶ Các bước (bản một notebook)**

1. Lấy notebook, upload lên Colab, chọn **L4 GPU**.
2. **Runtime → Run all**, không sửa gì.
3. Khi xong, xem lại và chụp:
   - ⏸ **Mục 3–4:** ✅ hết; `E4_discrete_reward: 553 prompt | …`.
   - ⏸ **Mục 5:** `Pool:` là đường dẫn; có dòng `✅ GenEval sẵn sàng.`
   - ⏸ **Mục 6:** bảng 6 dòng (6 task), lưới ảnh `E4_pilot_grid.png`, dòng `Gợi ý E4_TAGS = […]`.
   - ⏸ **Mục 7:** `✅ E4_TAGS = […]` và `E4: N prompt {…} | hậu tố tên run: _…` (ví dụ `_to-cnt-col`). Nếu notebook dừng ở đây với `❌ Chỉ … task qua chạy thử` thì gửi nhóm ảnh mục 6 và dừng E4.
   - ⏸ **Mục 8:** bảng 4 dòng (σ) và dòng `✅ σ chọn tự động: …`.
   - ⏸ **Mục 9:** mỗi task một dòng dạng `two_object  99 prompt | ứng viên đúng … | prompt toàn 0 …% | lẫn đúng/sai …% | toàn 1 …%`.
   - ⏸ **Mục 11:** bảng `summarize` và bảng `GenEval theo task`.
4. **Bị ngắt giữa chừng:** Run all lại. Kết quả chạy thử và điểm từng prompt đã lưu trên Drive nên được dùng lại; task và σ được chọn lại đúng như lần đầu. Phase 2 dùng seed theo prompt (`_pps`) nên cũng chạy tiếp được.
5. Gửi các file trong `MyDrive/RS-LiDAR/extra_experiments/E4_discrete_reward/`.

**📦 Đầu ra**

| Mục | In ra / ghi ra | Ý nghĩa |
|---|---|---|
| 6 | Bảng theo task, file `E4_pilot.csv` | Trên ứng viên lookahead (ảnh DPM-5 còn thô), verifier chấm được tới đâu |
| 6 | `E4_pilot_grid.png` | 8 ứng viên mỗi task, viền xanh = đúng, đỏ = sai |
| 8 | Bảng `σ · tỷ lệ đúng (sạch) · R̄σ trung bình · R̄σ trên ứng viên đúng · R̄σ trên ứng viên sai · số mức R̄σ khác nhau` | Nhiễu σ ảnh hưởng tới điểm của ứng viên đúng và sai thế nào |
| 11 | Bảng `summarize`, **mốc là `LiDAR (GenEval 0/1)`**, file `E4_summary_<hậu tố>_sig<σ>_M8.csv` | Các dòng: `LiDAR (IR, Table 1)`, `RS-LiDAR (IR, Table 1)`, `LiDAR (GenEval 0/1)`, `RS-LiDAR (GenEval R̄σ)` |
| 11 | Bảng `GenEval theo task`, file `E4_geneval_by_task_<hậu tố>_sig<σ>_M8.csv` | GenEval của 4 run, tách theo task |

**🌟 Kết quả lý tưởng**

- **Mục 6:** ít nhất 2 task (ngoài single_object) có `% prompt có ≥1 đúng` ≥ 0.5; cột `% prompt lẫn đúng/sai` càng cao càng tốt; lưới ảnh: viền xanh và đỏ khớp với mắt thường.
- **Mục 8:** `R̄σ trên ứng viên đúng` giảm dần khi σ tăng và luôn cao hơn `R̄σ trên ứng viên sai`.
- **Mục 11:** kiểm tra hợp lý: `LiDAR (GenEval 0/1)` có GenEval **cao hơn** `LiDAR (IR, Table 1)` trên các task này (nếu thấp hơn thì dẫn hướng bằng verifier chưa hoạt động → báo nhóm). Mong muốn: `RS-LiDAR (GenEval R̄σ)` có `ΔGenEval` **≥ +0.03, có `*`**; `ΔIR`, `ΔHPS` không âm có ý nghĩa; RS-LiDAR ≥ LiDAR (0/1) ở đa số task.

**⚠️ Lưu ý:** trong bài phải ghi rõ GenEval ở thí nghiệm này là metric được tối ưu; held-out là IR, HPS, CLIP, Aesthetic.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ ≥ 2 task qua chạy thử; RS-LiDAR (R̄σ) hơn LiDAR (0/1) về GenEval ≥ +0.03, có `*`; IR/HPS không giảm có ý nghĩa | Có năng lực mới: dẫn hướng bằng verifier rời rạc | Contribution mới cho Section 6. [Cổng E4](#5-cổng-quyết-định): T7 không chạy |
| ⚠️ Hơn nhưng không có `*` | Có xu hướng, chưa chắc | Cổng E4: T7 = E4 dự phòng |
| ❌ Notebook dừng ở mục 7 (dưới 2 task qua) | Ứng viên lookahead quá mờ, verifier không có tín hiệu | Báo nhóm: cần reward "mềm" (độ tin cậy detector), phải viết thêm code |
| ❌ Kết quả cuối: RS không hơn | σ chưa hợp | Cổng E4: T7 = E4 dự phòng |

---

<a id="t8"></a>
### T8 · Quân · E6 notebook 3: FLUX không dẫn hướng (Vanilla)

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb`](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) |
| 🖥️ GPU | **A100 + High-RAM** |
| ⏱️ Thời gian | 0 (nếu Drive đã có) hoặc ~5–8 h |
| 🔗 Cần xong trước | [T4](#t4) xong |
| 📤 Nộp lại | Ảnh chụp mục 8; tên thư mục kết quả |

**🎯 Mục tiêu.** Mốc FLUX không dẫn hướng, để thấy guidance có giúp không. (Notebook 4, ablation σ = 1.0, **đã bỏ** ngày 17/11.)

**▶ Các bước**

1. Mở `MyDrive/RS-LiDAR/results/`. Nếu đã có thư mục `FLUX_Vanilla_N1_steps28_all553` với 553 thư mục con thì **bỏ qua T8**.
2. Nếu chưa có: lấy notebook 3, upload, chọn **A100 + High-RAM**, kiểm tra `HF_TOKEN` như T2, **Runtime → Run all**.
3. Notebook 3 **không có Phase 1**. ⏸ Mục 7 kết thúc bằng `✅ Hoàn thành Phase 2! Kết quả lưu tại: …/results/FLUX_Vanilla_N1_steps28_all553`. ⏸ Mục 8: chụp bảng.

**🌟 Kết quả lý tưởng:** IR gần 1.019 và CLIP gần 0.282 (số của bài LiDAR). Không cần trùng, nhưng lệch quá 0.05 IR thì báo nhóm. IR của LiDAR (T2) phải **cao hơn rõ** con số này.

**⚠️ Lưu ý:** Vanilla không dùng seed theo prompt, nên chỉ là mốc tham khảo, không ghép cặp với LiDAR/RS. Lỗi `tokenizers`: như T2.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ LiDAR (T2) > Vanilla rõ | Guidance trên FLUX hoạt động | Đủ hàng cho bảng FLUX |
| ⚠️ LiDAR (T2) ≈ Vanilla | Guidance gần như không có tác dụng ở cấu hình này | Báo nhóm trước khi viết bảng FLUX |
| ❌ Lỗi | — | Gửi traceback |

---

<a id="t9"></a>
### T9 · Quân · E6: chấm GenEval cho các run FLUX

| | |
|---|---|
| 👤 Người chạy | **Quân** |
| 🧪 Thí nghiệm | E6 (xem [tham khảo](#ref-e6)) |
| 📓 Notebook | [`E6_flow_matching/FLUX_GenEval_Scoring_Colab.ipynb`](E6_flow_matching/FLUX_GenEval_Scoring_Colab.ipynb) |
| 🖥️ GPU | **L4** |
| ⏱️ Thời gian | Ngắn (chỉ chấm 1 ảnh/prompt cho mỗi run, không sinh ảnh), chưa đo |
| 🔗 Cần xong trước | [T2](#t2) và [T4](#t4). Có thêm T8 thì bảng có thêm dòng Vanilla |
| 📤 Nộp lại | Ảnh chụp mục 5, 6, 7; `E6_FLUX_summary.csv`, `E6_FLUX_geneval_by_task.csv` |

**🎯 Mục tiêu.** Notebook FLUX chỉ chấm IR và CLIP. Notebook này chấm thêm **GenEval** cho ảnh FLUX bằng **đúng** bộ chấm của Table 1 và E1–E4, rồi lập bảng so sánh RS-FlowLiDAR với LiDAR **có khoảng tin cậy 95%** (ghép cặp theo prompt). Đây là bảng FLUX sẽ đưa vào bài.

**▶ Các bước**

1. Lấy notebook, upload lên Colab, chọn **L4 GPU**. Không cần `HF_TOKEN`.
2. **Runtime → Run all**, cho phép Google Drive bằng tài khoản có lối tắt `RS-LiDAR`.
3. Khi xong, xem lại và chụp:
   - ⏸ **Mục 3–4:** ✅ hết; `E6_flux_geneval: 553 prompt | …`.
   - ⏸ **Mục 5:** bảng run × số prompt. `LiDAR` và `RS-FlowLiDAR σ=0.25` phải là **553 / 553**. Run không chạy (σ = 1.0, và Vanilla nếu bỏ qua T8) hiện 0 và được bỏ qua; như vậy là bình thường. Nếu một run hiện **số lẻ** (ví dụ 300) thì run đó chưa xong: chạy xong run đó trước, **hoặc** thêm `#` vào đầu dòng của run đó trong `FLUX_RUNS` (mục 5) rồi Run all lại.
   - ⏸ **Mục 6:** mỗi run một dòng `<run>  GenEval (trung bình 6 task) = …  (553 prompt)`.
   - ⏸ **Mục 7:** bảng `summarize` (mốc là `LiDAR`) và bảng `GenEval theo task`.
4. Gửi hai file trong `MyDrive/RS-LiDAR/extra_experiments/E6_flux_geneval/`.
5. **Khi có thêm run** (ví dụ T8 xong sau): Run all lại. Phần đã chấm được lưu trong `geneval_cache/` nên không chấm lại.

**🌟 Kết quả lý tưởng**

- **Mục 6:** GenEval của LiDAR **cao hơn** Vanilla (nếu có). Bài LiDAR báo 0.645 (FLUX không dẫn hướng) và 0.668 (LiDAR, n = 50); không cần trùng nhưng nên cùng cỡ.
- **Mục 7:** cột `n_prompts` = 553; IR và CLIP của mỗi run khớp với mục 8 của notebook FLUX tương ứng. Mong muốn: dòng `RS-FlowLiDAR σ=0.25` có `ΔIR` **dương, có `*`**; `ΔGenEval` ≥ 0; `ΔCLIP` không âm có ý nghĩa. Dòng `Vanilla` (nếu có) có `ΔIR` và `ΔGenEval` âm rõ.

**⚠️ Lưu ý:** không chấm GenEval bằng script khác (số phải dùng đúng bộ chấm của Table 1). Notebook chỉ đọc ảnh, không sửa thư mục kết quả của T2/T4/T8.

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ `RS-FlowLiDAR σ=0.25`: ΔIR > 0 có `*`, ΔGenEval ≥ 0 | Phương pháp chuyển sang FLUX được, có ý nghĩa thống kê | Gửi CSV; đưa bảng FLUX vào bài |
| ⚠️ ΔIR > 0 nhưng không có `*` | Không làm hại nhưng chưa chắc có lợi | Claim "transfers without degradation" |
| ❌ ΔIR < 0 có `*` | RS với σ = 0.25 làm FLUX kém đi | Báo nhóm; ghi trung thực |

---

<a id="4c-việc-của-tác-giả"></a>
## 4C. Việc của tác giả (T10)

<a id="t10"></a>
### T10 · Tác giả · E7: SMC ở nhiệt độ λ_FK khác có đuổi kịp RS-SMC không

| | |
|---|---|
| 👤 Người chạy | **Tác giả** (người đã chạy Table 2) |
| 🧪 Thí nghiệm | E7 (xem [tham khảo](#ref-e7)) |
| 📓 Notebook | Hai bản cấu hình sẵn ở thư mục gốc `RS-LiDAR/` (ngoài repo): `RS_SMC_E7_N4_VanillaSMC_FK3_Colab.ipynb` và `RS_SMC_E7_N4_VanillaSMC_FK30_Colab.ipynb`, sao từ notebook Table 2 ([`../colab/RS_SMC_Table8_Replication_Colab.ipynb`](../colab/RS_SMC_Table8_Replication_Colab.ipynb)) |
| 🖥️ GPU | **L4** (SMC SD 1.5, N = 4 cần khoảng 9 GiB) |
| ⏱️ Thời gian | 2 run bắt buộc + tối đa 1 run có điều kiện; ~2–4 h L4 mỗi run (~1–1.5 h A100) |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Ảnh chụp mục 4 và mục 6 của từng run; `final_metrics.json` của từng run |

**🎯 Mục tiêu.** Câu reviewer sẽ hỏi: *SMC ở λ_FK khác 10 có đạt được RS-SMC không?* E1 trả lời câu này cho LiDAR; T10 trả lời cho SMC, ở N = 4 (nơi gain lớn nhất, 0.563 → 0.744). Hai notebook chạy **SMC thường** (`METHOD = "Vanilla-SMC"`, không smoothing, không lookahead), N = 4, λ_FK = 3 và 30; mọi thứ khác như Table 2 (SD 1.5, DDPM-100, seed 100, 553 prompt, resampling ở bước 20/40/60/80 và bước cuối khi ESS < N/2).

Hai notebook bật thêm `--fk_stable_weights`. Code SMC mặc định kẹp trọng số exp(λ_FK·Δr) ở 1e10; với λ_FK = 30, khi reward tăng hơn 0.77 giữa hai lần resampling, mọi hạt vượt ngưỡng nhận trọng số bằng nhau, nên trọng số không còn đúng là exp(λ_FK·Δr). Cờ này tính trọng số bằng exp(λ_FK·(Δr − max Δr)): giống hệt cách cũ khi không chạm ngưỡng (λ_FK = 3 không bao giờ chạm; λ_FK = 10 của Table 2 chỉ chạm khi Δr > 2.3), và đúng khi chạm. Trong log, dòng `⚠️ FK weight clamp …` chỉ xuất hiện khi chạy **không** có cờ này.

**▶ Các bước**

1. Upload `RS_SMC_E7_N4_VanillaSMC_FK3_Colab.ipynb` lên Colab, chọn **L4 GPU**, **Runtime → Run all**, không sửa gì.
2. ⏸ **Mục 4:** các dòng `Phương pháp: Vanilla-SMC`, `Số hạt mục tiêu N: 4`, `Feynman-Kac λ: 3.0`, `Tập prompt: ALL (553 prompts)`; `Thư mục kết quả` có `_N4_` và `_lmbda3.0_` (khác run λ = 10 của Table 2); `Tiến độ sẵn có` là `0/553`. Chụp.
3. ⏸ **Mục 6:** dòng `✅ Hoàn thành thực nghiệm đơn lẻ trong … phút!`. Chụp.
4. Mở `Target_samples/<RUN_NAME>/final_metrics.json` trên Drive: ghi `ImageReward.mean` (cột Mean của Table 2) và `ImageReward.max` (cột Best-of-N).
5. Lặp bước 1–4 với `RS_SMC_E7_N4_VanillaSMC_FK30_Colab.ipynb` (`FK_LMBDA = 30.0`).
6. Run thứ 3 (chưa có notebook cấu hình sẵn, tạo khi cần):
   - Cả hai run có `mean` dưới 0.744 và run λ = 30 không cao hơn rõ 0.563: **xong E7**.
   - Run λ = 30 vẫn tăng rõ so với 0.563 (xu hướng chưa dừng): chạy thêm **đúng một** run SMC `FK_LMBDA = 100.0`.
   - Một run có `mean` ≥ **0.744**: chạy thêm `METHOD = "RS-SMC"` ở đúng λ đó, các biến smoothing như Table 2 (`SMC_SIGMA = 0.25`, `SMC_ANNEAL_SIGMA = True`, `SMC_ANNEAL_SCHEDULE = "tweedie_analytical"`, `SMC_NUM_MC_SAMPLES = 4`, `SMC_USE_CRN = False`).

**🌟 Kết quả lý tưởng**

- Mốc (Table 2, N = 4, λ_FK = 10): SMC mean **0.563** / Best-of-N **0.684**; RS-SMC mean **0.744** / Best-of-N **0.816**.
- Lý tưởng: SMC ở cả λ_FK = 3 và 30 có mean **dưới 0.744**, tốt nhất là thấp rõ.
- Mỗi run đủ 553 prompt; thời gian cùng cỡ với run λ = 10 cũ. Xong nhanh bất thường (vài phút) nghĩa là đang đọc lại thư mục cũ: xem lại bước 2.

**⚠️ Lưu ý:** tên thư mục chứa giá trị λ nên không ghi đè run λ = 10 của Table 2. Bị ngắt giữa chừng: Run all lại; notebook chạy tiếp các prompt còn thiếu (`--resume`).

**📊 Đọc kết quả và làm gì tiếp**

| Kết quả | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ SMC mean < 0.744 ở cả λ_FK = 3 và 30 | Đổi nhiệt độ không thay được smoothing trong SMC | Một câu ở Section 6.2 và một dòng ở appendix |
| ⚠️ SMC mean ≥ 0.744 ở một λ | Ở λ đó SMC đuổi kịp mốc RS-SMC λ = 10 | Chạy RS-SMC ở λ đó (bước 6). Nếu RS-SMC vẫn hơn SMC ở cùng λ: vẫn là ✅, báo cả ba số |
| ❌ RS-SMC không hơn SMC ở cùng λ tốt nhất | Lợi ích ở N = 4 có thể chỉ do λ_FK = 10 chưa tối ưu cho SMC | Xem lại cách trình bày RS-SMC trong bài |

---

<a id="5-cổng-quyết-định"></a>
## 5. Cổng quyết định

### 🔀 Cổng E4 · sau T6 · cả hai cùng xem

| Tình huống | Quyết định |
|---|---|
| E4 ✅ | E4 thành contribution mới ở Section 6. T7 không chạy |
| E4 ⚠️ hoặc kết quả cuối ❌ | T7 = **E4 dự phòng** (σ = 0.05, M = 16) |
| Notebook E4 dừng ở mục 7 (dưới 2 task có tín hiệu) | Dừng E4; cần viết code reward "mềm" trước khi chạy lại. T7 không chạy |

Các kết quả khác (E1, E2, E3, E6, E7) không chặn việc của ai; hướng xử lý nằm trong bảng 📊 của từng việc.

---

## 6. Quy tắc chung

1. **Lấy notebook từ repo, mở bằng File → Upload notebook**, không mở từ GitHub ([mục 3](#3-cách-chạy-một-notebook)).
2. **Không chạy cùng một run ở hai máy cùng lúc.** Mỗi người chỉ chạy việc của mình.
3. **Không sửa cấu hình ngoài những gì thẻ việc ghi.** Muốn thử cấu hình khác thì báo trước, vì số trong bài phải khớp một cấu hình duy nhất.
4. **E1, E3: để mỗi run Phase 2 chạy một mạch** (seed toàn cục, để ghép cặp với Table 1). E4, E6 dùng seed theo prompt nên chạy tiếp được sau khi bị ngắt.
5. **Kết quả xấu cũng phải báo.** Mỗi việc có dòng ❌ và hướng xử lý; số xấu không được bỏ đi âm thầm.
6. **Lỗi:** traceback đầy đủ + tên notebook + số mục.

---

<a id="7-tham-khảo-chi-tiết-thí-nghiệm"></a>
## 7. Tham khảo: chi tiết thí nghiệm

Phần này giải thích **vì sao** chạy từng thí nghiệm và nó vào đâu trong bài. Không cần đọc để chạy.

| | Tên | Câu hỏi reviewer mà nó trả lời | Vị trí trong bài |
|---|---|---|---|
| [E1](#ref-e1) | Nhiệt độ λ và cắt reward | "Smoothing có chỉ là hạ λ hoặc cắt reward ngoại lai không?" | Section 6, bảng/hình mới |
| [E2](#ref-e2) | Flip rate | "Smoothing có thật sự đổi ứng viên được chọn không, và đổi có đúng không?" | Section 6.3 |
| [E3](#ref-e3) | Cùng ngân sách reward | "RS gọi reward nhiều hơn 4 lần, thắng là đương nhiên?" | Section 6.1, đoạn chi phí |
| [E4](#ref-e4) | Reward rời rạc | "Phương pháp làm được gì mà cách khác không làm được?" | Section 6, mục riêng + contribution mới |
| [E5](#ref-e5) | Nhiều reward (IR + HPS) | "Chỉ một reward thì hẹp" | **Đã bỏ** (17/11) |
| [E6](#ref-e6) | Flow matching (FLUX.1-dev) | "Có dùng được cho rectified flow không?" | Hàng FLUX hoặc bảng nhỏ ở Section 6 |
| [E7](#ref-e7) | SMC ở λ_FK khác | "SMC chỉ cần chỉnh λ_FK là bằng RS-SMC?" | Một câu ở Section 6.2 + một dòng appendix |

Cấu hình chung của E1–E4 (khớp Table 1, đã khoá trong notebook):

| Tham số | Giá trị |
|---|---|
| Backbone | SD 1.5 (`runwayml/stable-diffusion-v1-5`) |
| Lookahead (Phase 1) | DPM-Solver 5 bước, K = 50 ứng viên, seed 100 |
| Target (Phase 2) | DDPM 100 bước (η = 1), 4 ảnh/prompt, seed 100, guidance scale s = 12.5, guidance khi t > 200 |
| LiDAR | λ = 5000 (mốc) |
| RS-LiDAR | σ = 1.0, M = 4, nhiễu **I.I.D.** (không CRN) |
| Prompt | 553 prompt GenEval (`PROMPT_MODE = "ALL"`) |
| Đánh giá | ImageReward, CLIP, HPSv2.1, Aesthetic (tự động) + GenEval Mask2Former Swin-S (cell GenEval trong notebook) |

<a id="ref-e1"></a>
### E1 — Smoothing có chỉ là hạ nhiệt độ không?

**Mục đích.** Đây là phản biện nguy hiểm nhất. Làm trơn kéo các điểm reward lại gần nhau, trông giống như hạ nhiệt độ λ của softmax LiDAR, hoặc như cắt bớt điểm ngoại lai. Nếu LiDAR ở λ thích hợp, hoặc LiDAR với reward bị cắt, đạt được bằng RS-LiDAR, thì đóng góp của bài chỉ còn là "chỉnh siêu tham số". Lý thuyết (§4.3) nói hai thứ này khác nhau: hạ λ không đổi thứ tự ứng viên, còn làm trơn thì có.

**Thiết kế.** Cùng pool 50 ứng viên của Table 1:
- **LiDAR λ ∈ {100, 1000}.** Trong trọng số LiDAR, λ·r cạnh tranh với số hạng khoảng cách ‖xₜ − √ᾱₜ·zₖ‖²/2(1−ᾱₜ), cộng trên 4×64×64 chiều latent (`fkd_pipeline_sd.py`, `get_sample_guide`), nên vùng chuyển từ "gần như không dẫn hướng" sang "bão hoà" nằm ở λ cỡ hàng chục đến hàng nghìn. Figure 8 của bài LiDAR (trục λ 10⁰–10⁴) cho thấy IR tăng ở khoảng 10¹–10³ rồi phẳng; họ chọn 5000 trên đoạn phẳng. λ = 100 nằm giữa vùng chuyển tiếp, λ = 1000 sát bão hoà; λ thấp hơn trích Figure 8 thay vì chạy lại. Mục 5 của notebook đo lại độ mạnh dẫn hướng trên pool thật.
- **LiDAR-Clip q ∈ {0.8, 0.95}:** winsorize điểm theo prompt, `min(r, quantile(r, q))`, λ = 5000. Các ứng viên đầu hoà nhau (khoảng 10 ở q = 0.8, 2–3 ở q = 0.95), nên một ứng viên có điểm bị thổi phồng không còn chiếm trọn guidance. Đây là cách chống ngoại lai đơn giản nhất, không cần gọi thêm reward. Hai mức kẹp hai bên để kết luận không phụ thuộc ngưỡng.
- Mốc LiDAR và RS-LiDAR λ = 5000 của Table 1.

**Vị trí trong bài.** Section 6, mục "Is smoothing a lower temperature?": bảng hoặc hình IR/GenEval theo λ, có đường ngang RS-LiDAR λ = 5000 và điểm LiDAR-Clip. Claim nếu đạt: *"No temperature of LiDAR, and no clipping of its scores, reaches RS-LiDAR."*

<a id="ref-e2"></a>
### E2 — Smoothing đổi ứng viên thắng ở đâu, và có đổi đúng không?

**Mục đích.** Kiểm chứng trực tiếp cơ chế của bài. Lý thuyết nói reward model có những "spike" (điểm cao giả do mẫu pixel nhỏ), và làm trơn loại chúng khỏi thứ hạng. Với λ = 5000, LiDAR gần như chọn đúng một ứng viên điểm cao nhất, nên câu hỏi đo được là: **làm trơn đổi ứng viên thắng ở bao nhiêu prompt, và ứng viên mới có tốt hơn trên các metric không dùng để tối ưu không?** Hạ λ không bao giờ đổi ứng viên thắng, nên đây là đối chứng sạch nhất. Figure 1 của bài chỉ minh hoạ một prompt; E2 là bằng chứng trên toàn bộ 553 prompt.

**Thiết kế.** Hai pool của Table 1 (cùng 50 latents, một bên chấm bằng R, một bên bằng R̄σ): (1) flip rate, tỷ lệ prompt có argmax R ≠ argmax R̄σ; (2) độ mong manh: ứng viên thắng theo R có hiệu R − R̄σ lớn bất thường không, so với nền hoán vị (để trừ hiệu ứng "winner's curse"); (3) chất lượng: trên các prompt bị flip, decode hai ứng viên thắng và chấm CLIP, HPS, Aesthetic, GenEval; (4) lợi ích RS-LiDAR − LiDAR của Table 1 trên nhóm prompt flip và không flip.

**Vị trí trong bài.** Section 6.3, cạnh Figure 1. Claim nếu đạt: *"Smoothing changes the selected candidate on X% of prompts; on these prompts the new winner is better on every held-out metric."*

<a id="ref-e3"></a>
### E3 — So sánh cùng số lần gọi reward

**Mục đích.** RS-LiDAR gọi reward K × M = 50 × 4 = 200 lần, LiDAR chỉ 50 lần. Nếu cho LiDAR cùng 200 lần gọi reward, tức 200 ứng viên, thì LiDAR có bắt kịp không?

**Thiết kế.** Sinh thêm 3 pool 50 ứng viên (seed 200, 300, 400), ghép với pool seed 100 của Table 1 thành 200 ứng viên mỗi prompt; Phase 2 LiDAR λ = 5000 với `top_k` = 200, cùng seed 100 với Table 1 (nhiễu khởi đầu của ảnh cuối giống hệt ở mọi run). So với RS-LiDAR K = 50, M = 4. Dùng lại pool Table 1 nên chỉ sinh mới 150 ứng viên mỗi prompt; sinh thẳng 200 ứng viên với một seed sẽ tốn hơn và không lấy lại được 50 ứng viên cũ (nhiễu được lấy nối tiếp qua các prompt). E3 chỉ cân bằng **số lần gọi reward**; LiDAR K = 200 vẫn tốn hơn ở phần sinh ứng viên.

**Vị trí trong bài.** Đoạn chi phí ở Section 6.1 hoặc gộp vào bảng E1. Claim nếu đạt: *"At equal reward evaluations, RS-LiDAR (K = 50, M = 4) outperforms LiDAR with K = 200, which also needs 4× more lookahead sampling."*

<a id="ref-e4"></a>
### E4 — Reward rời rạc: dẫn hướng bằng bộ chấm GenEval

**Mục đích.** Cho thấy một năng lực mà phương pháp khác không có. Reward ở đây là **0 hoặc 1**: "ảnh có đúng prompt theo GenEval không" (detector Mask2Former + phân loại màu CLIP, giống hệt bộ chấm).
- Gradient guidance không dùng được vì reward không khả vi.
- LiDAR với reward 0/1 chỉ lọc đúng/sai; nếu không ứng viên nào đúng thì không có tín hiệu; mọi λ đều cho các ứng viên đúng hoà nhau.
- RS biến 0/1 thành R̄σ = "xác suất ảnh vẫn đúng khi bị nhiễu": một tín hiệu liên tục đo "đúng chắc chắn tới đâu".

**Rủi ro và cách xử lý.** Ứng viên lookahead là ảnh DPM-Solver 5 bước, vật thường mờ. Với task khó (position, color_attr), gần như mọi ứng viên bị chấm 0, khi đó cả hai phương pháp đều mất tín hiệu. Vì vậy notebook chạy thử theo task và tự giữ task mà ứng viên còn được chấm đúng ở ít nhất 50% prompt; single_object luôn bị loại vì SD 1.5 đã đạt khoảng 0.97. Dự kiến giữ two_object, colors, counting (273 prompt). σ được chọn tự động trên 4 prompt mỗi task theo quy tắc định trước.

**Thiết kế.** Dùng lại 50 latents seed 100 của Table 1. Với mỗi prompt của các task đã chọn: chấm 0/1 cho 50 ứng viên, và R̄σ với M = 8 nhiễu; tạo hai pool dẫn xuất; Phase 2 LiDAR λ = 5000 trên mỗi pool (seed theo prompt). So thêm với LiDAR / RS-LiDAR dùng ImageReward của Table 1 trên cùng các prompt.

**Vị trí trong bài.** Section 6, mục riêng "Discrete verifiers" và một contribution mới. **Bắt buộc ghi rõ:** GenEval ở thí nghiệm này là metric được tối ưu, không còn held-out; held-out là IR, HPS, CLIP, Aesthetic.

<a id="ref-e5"></a>
### E5 — Nhiều reward cùng lúc (IR + HPS): đã bỏ

Bỏ ngày 17/11/2026 vì giới hạn GPU: thí nghiệm chỉ để vào appendix, và lý thuyết không dự đoán gì riêng cho trường hợp nhiều reward. Notebook [`E5_multi_reward/`](E5_multi_reward/E5_MultiReward_IR_HPS_SD15_DDPM100_Colab.ipynb) vẫn giữ trong repo nếu sau này cần.

<a id="ref-e6"></a>
### E6 — Flow matching (FLUX.1-dev)

| Thứ tự | Notebook | Vai trò |
|---|---|---|
| 1 | [FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_LiDAR_n50_sflow3_all553_seed42_Colab.ipynb) | LiDAR: sinh pool 50 ứng viên + Phase 2 ([T2](#t2)) |
| 2 | [FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig0.25_M4_sflow3_all553_seed42_Colab.ipynb) | **RS-FlowLiDAR chính**: chấm lại pool của notebook 1 bằng R̄σ + Phase 2 ([T4](#t4)) |
| 3 | [FLUX_Vanilla_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_Vanilla_all553_seed42_Colab.ipynb) | Mốc không dẫn hướng; bỏ qua nếu Drive đã có ([T8](#t8)) |
| 4 | [FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb](E6_flow_matching/FLUX_RSFlowLiDAR_n50_sig1.0_M4_sflow3_all553_seed42_Colab.ipynb) | Ablation σ = 1.0: **đã bỏ** (17/11), không chạy |
| 5 | [FLUX_GenEval_Scoring_Colab.ipynb](E6_flow_matching/FLUX_GenEval_Scoring_Colab.ipynb) | Chấm GenEval bằng bộ chấm của Table 1; bảng so với LiDAR có CI 95% ghép cặp ([T9](#t9)) |

**Mục đích.** Bài hiện chỉ có SD 1.5 và SDXL (diffusion). E6 cho thấy phương pháp chuyển sang rectified flow (FLUX) được.

**Cấu hình (đã đặt sẵn, không đổi):** FLUX.1-dev; lookahead Hyper-SD 8 bước, n = 50 ứng viên sinh theo lô 5 (vừa A100 40 GB); s_flow = 3.0; cửa sổ dẫn hướng t ∈ [1.0, 0.95]; λ = 5000; 28 bước; 1 ảnh/prompt; seed theo prompt ở cả hai phase; RS chấm lại đúng pool LiDAR. σ = 0.25, M = 4 chuyển nguyên từ SDXL, không chỉnh trên GenEval. Notebook 1–3 chỉ chấm IR và CLIP; GenEval chấm ở notebook 5.

**Môi trường.** Cell cài đặt ghim `transformers>=4.43.3,<4.45`, cần `tokenizers 0.19`, không có bản dựng sẵn cho Python 3.13 (runtime mặc định của Colab). Khi đó pip lỗi mà cell không dừng, và notebook chạy bằng bản transformers/diffusers có sẵn của Colab. Vì vậy cuối mục 3 in phiên bản thư viện để ghi lại; gặp lỗi lạ thì đổi runtime sang **2026.07 (Python 3.12)**.

**Vị trí trong bài.** Hàng FLUX trong Table 1 hoặc một bảng nhỏ ở Section 6. Claim nếu đạt: *"RS-TTA transfers to rectified-flow models without re-tuning."* Vanilla không dùng seed theo prompt nên chỉ là mốc tham khảo.

<a id="ref-e7"></a>
### E7 — SMC ở nhiệt độ λ_FK khác

**Mục đích.** Câu hỏi "smoothing có chỉ là hạ nhiệt độ không" của E1, nhưng cho SMC, nơi nhiệt độ là λ_FK. RS-SMC là một trong hai phương pháp của bài, và gain ở N = 4 (+0.181) là gain lớn nhất của Table 2.

**Thiết kế (bản tối thiểu, chọn vì chi phí).** Chỉ chạy SMC, chỉ N = 4, λ_FK ∈ {3, 30} cạnh mốc 10 đã có. λ = 3 là phép so sánh chính (nhiệt độ thấp hơn là thứ smoothing có thể "giả làm"); λ = 30 cho chiều ngược lại. RS-SMC chỉ chạy lại ở đúng λ mà SMC đạt ≥ 0.744. **Không** chạy: thêm seed (gain +0.08 đến +0.18 lớn hơn nhiều độ lệch giữa seed, cỡ 0.01), N = 32, sweep λ cho RS-SMC, chấm held-out.

**Vị trí trong bài.** Một câu ở Section 6.2 ("SMC with λ_FK ∈ {3, 30} reaches … at N = 4, below RS-SMC") và một dòng ở appendix.
