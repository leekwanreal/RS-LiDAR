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
| E3 | T5 | 1 ✅ + 2–3 | Giai đoạn 1 ✅ 22/11: LiDAR K = 200 (sinh 3 pool ứng viên mới: seed 200, 300, 400). Giai đoạn 2, mỗi run một notebook E3b (xem bảng 📋 ở thẻ T5): LiDAR K = 66, RS-LiDAR K = 200, tuỳ chọn LiDAR K = 100; dùng lại các pool của giai đoạn 1 |
| E4 | T6 | 2 | LiDAR (reward 0/1) và RS-LiDAR (R̄σ); tự dừng trước Phase 2 nếu dưới 2 task qua chạy thử |
| E4 dự phòng | T7 | 0 hoặc 2 | Chỉ khi [cổng E4](#5-cổng-quyết-định) yêu cầu |
| E5 | — | 0 | **Đã bỏ** (17/11) |
| E6 | T2, T4, T8, T9 | 2 hoặc 3 | FLUX LiDAR (kèm sinh pool), RS-FlowLiDAR σ = 0.25; Vanilla chỉ khi Drive chưa có; σ = 1.0 **đã bỏ**; T9 chỉ chấm điểm |
| E7 | T10 | 2 (+ tối đa 1) | SMC N = 4, λ_FK = 3 và λ_FK = 30; thêm 1 run nếu đạt điều kiện ở thẻ T10 |
| F1 | [T11](#t11) | 3 | LiDAR-Aug và LiDAR-LatentSmooth (σ_z = 0.6, kiểm thêm 0.3), cùng M = 4 lần gọi reward mỗi ứng viên, trên pool seed 100 của Table 1 |
| F8 | [T13](#t13) | 2 | Đối chứng thứ hạng: LiDAR dẫn theo ứng viên hạng 2 theo R (F8a) và theo một ứng viên ngẫu nhiên trong top-5 theo R (F8b), trên pool seed 100 của Table 1, không gọi reward lại |
| F3 | [T12](#t12) | 2 mỗi seed | DrawBench (200 prompt), LiDAR và RS-LiDAR với cấu hình SD 1.5 DDPM-100 của Table 1 (σ = 1, M = 4 cố định), seed 100 rồi 50, 150 |

### 👤 Hiếu: T1 → T3 → T5 → T7 → T13

| Việc | Thí nghiệm | GPU | Thời gian | Cần xong trước |
|---|---|---|---|---|
| [**T1**](#t1) | E2 · flip rate | L4 | ~1–2.5 h | — |
| [**T3**](#t3) | E1 · nhiệt độ λ và cắt reward | L4 (hoặc A100) | ~15–25 h L4 (~6–10 h A100) | nên sau T1 |
| [**T5**](#t5) | E3 · cùng số lần gọi reward, cùng thời gian | L4 | Giai đoạn 1 ✅ xong 22/11. Giai đoạn 2: E3b-1 ~4–5 h → E3b-2 ~7–9 h → (tuỳ chọn) E3b-3 ~4–5 h; E3c là **phụ** (bước cứu, chỉ khi S4/S5 và nhóm quyết; ~10–12 h A100) | — |
| [**T7**](#t7) | E4 dự phòng, **chỉ khi** cổng E4 yêu cầu | L4 | 0 hoặc ~8–12 h | T6 + [cổng E4](#5-cổng-quyết-định) |
| [**T13**](#t13) | F8 · đối chứng thứ hạng (hạng 2, ngẫu nhiên top-5), **mới 24/11, làm ngay** | L4 (hoặc A100) | 2 notebook chạy song song, mỗi cái ~4–5.5 h L4 (~1.5 h A100) | — (dùng pool và run mốc seed 100 đã có) |

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
### T5 · Hiếu · E3: so sánh cùng số lần gọi reward và cùng thời gian

| | |
|---|---|
| 👤 Người chạy | **Hiếu** |
| 🧪 Thí nghiệm | E3 (xem [tham khảo](#ref-e3)) |
| 📓 Notebook | Mỗi run có một notebook đã cấu hình sẵn, xem bảng 📋 ngay dưới. Giai đoạn 1 (✅ xong): [`E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) (tên file giữ chữ K100 cho link không đổi; notebook chỉ chạy K = 200) |
| 🖥️ GPU | **L4 + High-RAM** (SD 1.5); **A100** chỉ cho E3c (phụ) |
| ⏱️ Thời gian | Giai đoạn 1: đã xong. Giai đoạn 2: E3b-1 ~4–5 h, E3b-2 ~7–9 h, E3b-3 (tuỳ chọn) ~4–5 h trên L4. E3c (phụ, mặc định không chạy): ~10–12 h A100 |
| 🔗 Cần xong trước | — |
| 📤 Nộp lại | Giai đoạn 1: ảnh chụp mục 5, 7, 8; `E3_summary.csv`. Giai đoạn 2: `E3b_metrics.md` (bảng gọn, không CI), `E3b_verdict.md`, `E3b_equal_time.csv`, `E3b_added_value.csv`, `E3b_quality_vs_time.csv`. Bước cứu: `E3c_metrics.md`, `E3c_verdict.md`, `E3c_equal_time.csv` |

> ✅ **Giai đoạn 1 xong (22/11/2026).** Seed 100, IR / GenEval / CLIP / HPS:
> - LiDAR K = 50: 0.338 / 0.476 / 0.277 / 0.268;
> - RS-LiDAR K = 50: 0.374 / **0.488** / 0.279 / 0.269;
> - LiDAR K = 200: **0.449** / 0.485 / 0.279 / 0.268.
>
> LiDAR K = 200 chỉ hơn ở IR (+0.075 [+0.025, +0.125]*), là reward dùng để dẫn hướng. Ở GenEval (−0.003), CLIP và HPS, nó không hơn RS-LiDAR (CI chứa 0), dù tốn 2.1× thời gian. Theo bảng đọc kết quả bên dưới, đây là dòng ❌ (đúng như dự đoán). **Việc tiếp theo: chạy giai đoạn 2 (E3b)**, bắt đầu với LiDAR K = 66. Lợi ích của việc tăng K là +0.111 IR và +0.009 GenEval, dưới ngưỡng dự đoán (0.18 / 0.061), nên RS-LiDAR nhiều khả năng thắng ở cùng thời gian.

**📋 Kế hoạch chạy tiếp (Hiếu).** Mỗi bước là một notebook đã cấu hình sẵn: upload, chọn GPU, **Runtime → Run all**, không sửa gì. Chạy lần lượt từ trên xuống; mỗi notebook là một phiên Colab.

| Bước | Notebook (bấm để mở) | Chạy gì | GPU | Thời gian | Khi nào chạy | Gửi lại |
|---|---|---|---|---|---|---|
| ✅ 1 | [`E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3_ComputeMatched_K100_K200_SD15_DDPM100_Colab.ipynb) | LiDAR K = 200 | L4 | đã xong 22/11 | — | ✅ đã có `E3_summary.csv` |
| **2** | [`E3b1_EqualTime_LiDAR_K66_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3b1_EqualTime_LiDAR_K66_SD15_DDPM100_Colab.ipynb) | LiDAR K = 66 (tốn nhỉnh hơn RS-LiDAR K = 50: 15.44 s so với 15.36 s) | L4 + High-RAM | ~4–5 h | **Ngay bây giờ** | `E3b_metrics.md`, `E3b_verdict.md` |
| **3** | [`E3b2_AddedValue_RSLiDAR_K200_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3b2_AddedValue_RSLiDAR_K200_SD15_DDPM100_Colab.ipynb) | RS-LiDAR K = 200 (so với LiDAR K = 200) | L4 + High-RAM | ~7–9 h | Sau bước 2 | `E3b_metrics.md`, `E3b_verdict.md`, 3 file csv `E3b_*` |
| 4 (tuỳ chọn) | [`E3b3_Optional_LiDAR_K100_SD15_DDPM100_Colab.ipynb`](E3_compute_matched/E3b3_Optional_LiDAR_K100_SD15_DDPM100_Colab.ipynb) | LiDAR K = 100 (thêm điểm cho hình) | L4 + High-RAM | ~4–5 h | Sau bước 3, nếu còn GPU | `E3b_metrics.md` |
| 5 (**phụ**) | [`E3c_SDXL_EqualTime_K119_vs_RSK100_Colab.ipynb`](E3_compute_matched/E3c_SDXL_EqualTime_K119_vs_RSK100_Colab.ipynb) | **Phụ, không nằm trong kế hoạch chính.** SDXL: LiDAR K = 119 vs RS-LiDAR K = 100 (cùng thời gian) | **A100** | ~10–12 h | Mặc định **không chạy**. Chỉ chạy khi `E3b_verdict.md` sau bước 3 ghi **S4** hoặc **S5** và nhóm đã bàn và quyết định chạy | `E3c_metrics.md`, `E3c_verdict.md` |

- Mỗi notebook E3b tự dùng lại run E3b đã xong trên Drive, nên bảng `E3b_metrics.md` của notebook chạy sau luôn có đủ mọi run đã có. Kết luận S1–S5 chỉ chốt sau bước 3; sau bước 2, file ghi "S1/S2 (chưa có RS K=200)" hoặc tương tự.
- Phase 2 phải chạy một mạch (seed toàn cục 100). Chỉ bắt đầu một notebook khi phiên Colab còn đủ thời gian ở cột "Thời gian". Bị ngắt giữa Phase 2: xoá thư mục `E3b_…` (hoặc `E3c_…`) đang dở trên Drive rồi Run all lại.
- Sau bước 3, nhóm đọc `E3b_verdict.md` theo [bảng quyết định cuối](#t5) bên dưới: S1/S2/S3 thì dừng, không chạy bước 5.

**⏸ Khi chạy mỗi notebook E3b, kiểm tra trong 10–15 phút đầu** (sai ở đây thì dừng và báo nhóm, chưa tốn nhiều GPU):
- **Mục 3–4:** ✅ hết; dòng `E3_compute_matched: 553 prompt | …`.
- **Mục 5:** đúng cấu hình của notebook:
  - E3b-1: `RUN_EQUAL_TIME = True`, `K_EQUAL_TIME = 66`;
  - E3b-2: `RUN_RS_K200 = True`;
  - E3b-3: `RUN_K100 = True`.
- **Mục 6:** dòng `✅ 4 pool: {100: …seed100, 200: …seed200, 300: …seed300, 400: …seed400}`, rồi 3 dòng `✅ Run mốc khớp cấu hình:` cho `E1_LiDAR_…`, `E1_RSLiDAR_…` và `E3_LiDAR_SD15_K200_…`.
- **Mục 7 (E3b-1):** thanh tiến trình `LiDAR_K66` chạy hết 553 prompt, rồi `▶ LiDAR K=66` và Phase 2 bắt đầu chạy.
- **Mục 8 (E3b-2):** `✅ Pool RS dùng chung latents với pool LiDAR: …` (hoặc dòng chấm lại pool seed 100), 3 dòng `▶ Chấm R̄σ cho pool seed 200/300/400`, `✅ 4 pool R̄σ cùng latents với 4 pool LiDAR`, rồi `▶ RS-LiDAR K=200 M=4`.
- **Mục 8 (E3b-2, E3b-3):** dòng `↺ Dùng run đã có trên Drive: LiDAR K=66 …` (run của notebook trước được dùng lại).
- **Mục 9:** bảng có các dòng `RS-LiDAR K=50 M=4` (IR ≈ 0.374), `LiDAR K=50` (≈ 0.338), `LiDAR K=66`, và sau E3b-2 thêm `LiDAR K=200` (≈ 0.449) và `RS-LiDAR K=200 M=4`. Mốc lệch xa các số này thì báo nhóm.

**🎯 Mục tiêu.** RS-LiDAR chấm mỗi ứng viên 4 lần (50 ứng viên × 4 = **200 lần gọi reward** mỗi prompt), còn LiDAR chỉ chấm 50 lần. Reviewer sẽ hỏi hai câu:

1. *Cho LiDAR cũng 200 lần gọi reward, tức 200 ứng viên, thì LiDAR có bắt kịp không?* Giai đoạn 1 trả lời câu này.
2. *Nếu dùng phần thời gian thêm của RS-LiDAR để sinh thêm ứng viên cho LiDAR thì sao?* Đây là phép so sánh công bằng hơn, vì cùng số lần gọi reward **không** có nghĩa là cùng thời gian. Giai đoạn 2 trả lời câu này.

**⏱️ Chi phí của từng cấu hình** (SD 1.5, DDPM-100, 4 ảnh mỗi prompt, A100; tính từ bảng thời gian ở Appendix E.2):
- Sinh một ứng viên mất 5.69 / 50 = **0.114 s**; một lần gọi reward mất 0.65 / 50 = **0.013 s**; sinh 4 ảnh cuối mất **7.07 s** và gần như không đổi theo K.
- LiDAR(K) ≈ 0.127·K + 7.07 s. RS-LiDAR(K, M = 4) ≈ (0.114 + 4 × 0.013)·K + 7.07 = 0.166·K + 7.07 s.

**Thời gian gồm những gì, và vì sao chọn K = 66** (tính lại 22/11):

| Phần | Gồm | Theo K? |
|---|---|---|
| Sinh ứng viên (5.69 s với K = 50) | Mã hoá prompt bằng text encoder CLIP (một lần mỗi prompt, cỡ 0.01 s); khử nhiễu δ = 5 bước DPM-Solver với CFG (2 lượt UNet mỗi bước) cho K ứng viên; VAE decode K ứng viên để chấm reward | Gần như toàn bộ tỉ lệ với K; chỉ phần mã hoá prompt là cố định |
| Chấm reward (0.65 s với K = 50) | ImageReward trên K ảnh đã decode. RS-LiDAR không decode lại, chỉ thêm 3 lượt chấm trên ảnh nhiễu cho mỗi ứng viên (phần thêm 1.95 s đo trực tiếp: 15.36 − 13.41) | Tỉ lệ với K (và với M ở RS-LiDAR) |
| Sinh ảnh cuối (7.07 s) | Mã hoá prompt; 100 bước DDPM cho 4 ảnh với CFG; dẫn hướng LiDAR (tính khoảng cách tới K latent ứng viên ở mỗi bước t > 200, rất nhỏ); VAE decode 4 ảnh | Không đổi (phần dẫn hướng tăng rất nhẹ theo K, làm LiDAR K lớn tốn thêm chút ít) |

- **Điểm hoà thời gian:** 0.127·K + 7.07 = 15.36 → K = 65.4. Với K = 65, LiDAR tốn 15.31 s, **ít hơn** RS-LiDAR 0.05 s, nên không công bằng với LiDAR. Notebook dùng **K = 66**: 15.44 s, **nhiều hơn RS-LiDAR 0.08 s**.
- **K = 66 vẫn đủ nếu một phần thời gian sinh ứng viên là cố định:** nếu phần cố định f (mã hoá prompt và chi phí khởi tạo mỗi prompt) không tỉ lệ với K, mỗi ứng viên rẻ hơn và điểm hoà dịch lên K = (2.60 + 50c) / (c + 0.013) với c = (5.69 − f) / 50. Điểm hoà vẫn ≤ 66 khi f ≤ 0.2 s (3.5% của 5.69 s). Mã hoá prompt chỉ khoảng 0.01 s, nên K = 66 vẫn cho LiDAR nhiều thời gian hơn.
- **SDXL (E3c):** mỗi ứng viên 0.334 s, phần thêm của RS-LiDAR 5.91 s → điểm hoà K = 117.7. Notebook dùng **K = 119**: 85.02 s so với 84.58 s của RS-LiDAR K = 100 (nhiều hơn 0.44 s).

| Cấu hình | Ứng viên phải sinh | Lần gọi reward | Thời gian (s) | So với RS-LiDAR K = 50 |
|---|---|---|---|---|
| LiDAR K = 50 (Table 1) | 50 | 50 | 13.41 | 0.87× |
| **RS-LiDAR K = 50, M = 4** (Table 1) | 50 | 200 | **15.36** | 1× |
| LiDAR K = 66 (giai đoạn 2) | 66 | 66 | ≈ 15.44 | 1.005× (**LiDAR nhỉnh hơn 0.08 s**) |
| LiDAR K = 100 (giai đoạn 2, tuỳ chọn) | 100 | 100 | ≈ 19.75 | 1.29× |
| LiDAR K = 200 (giai đoạn 1) | 200 | 200 | ≈ 32.43 | 2.11× |
| RS-LiDAR K = 200, M = 4 (giai đoạn 2) | 200 | 800 | ≈ 40.23 | 2.62× |

Thời gian trong bài sẽ tính theo công thức này, không lấy thời gian đo trên Colab, vì các run dùng lại pool có sẵn nên thời gian chạy thật không phản ánh chi phí.

**🔮 Dự đoán để đọc kết quả** (chỉ dùng nội bộ, không đưa vào bài). Ở seed 100, RS-LiDAR hơn LiDAR K = 50 khoảng +0.036 IR và +0.012 GenEval. Nếu chất lượng của LiDAR tăng theo log K, thì K = 66 chỉ lấy được khoảng 20% phần lợi ích từ K = 50 lên K = 200 (log 1.32 / log 4). Khi đó RS-LiDAR vẫn thắng ở cùng thời gian, trừ khi LiDAR K = 200 hơn LiDAR K = 50 quá **0.18 IR** hoặc **0.061 GenEval**. Giai đoạn 2 chạy K = 66 thật để không phải dựa vào giả định này.

**Giai đoạn 1** (notebook hiện có). Notebook tự làm 3 việc:

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
   - ⏸ **Mục 7:** dòng đầu in `LiDAR K=50` và `RS-LiDAR K=50 M=4`, sau đó là `▶ LiDAR K=200`. Notebook tìm run mốc theo thứ tự: run Table 1 mang tên gốc, rồi đến run λ = 5000 mà E2/E1 đã chạy (`E1_LiDAR_SD15_n50_DDPM100_lmbda5000_seed100` và `E1_RSLiDAR_SD15_n50_sig1.0_M4_DDPM100_lmbda5000_seed100`). Run nào cũng phải qua bước kiểm tra cấu hình (dòng `✅ Run mốc khớp cấu hình: …`). Bình thường mục 7 sẽ dùng hai run `E1_…`. Chỉ khi cả hai loại đều không có hoặc không khớp, mục 7 mới in `None` và notebook tự chạy lại hai mốc (`E3_LiDAR_…_K50`, `E3_RSLiDAR_…_K50_M4`). Trường hợp đó vẫn hợp lệ nhưng tốn thêm ~5–8 h L4.
   - ⏸ **Mục 8:** bảng kết quả.
4. Gửi nhóm ảnh chụp và `MyDrive/RS-LiDAR/extra_experiments/E3_compute_matched/E3_summary.csv`.

**📦 Bảng kết quả (mục 8, `E3_summary.csv`)**, mốc là `RS-LiDAR K=50 M=4`:

| Dòng | Là gì |
|---|---|
| `RS-LiDAR K=50 M=4` | Mốc: RS-LiDAR của Table 1 (200 lần gọi reward) |
| `LiDAR K=50` | LiDAR của Table 1 (50 lần gọi reward) |
| `LiDAR K=200` | LiDAR với 200 ứng viên (200 lần gọi reward), run mới |

**🌟 Kết quả lý tưởng**

- **Kiểm tra mốc:** nếu mốc là hai run `E1_…` (hoặc notebook tự chạy lại với tên `E3_…`), kết quả phải bằng hoặc gần các run seed 100 của E1/E2: `LiDAR K=50` có IR ≈ 0.338, GenEval ≈ 0.476; `RS-LiDAR K=50 M=4` có IR ≈ 0.374, GenEval ≈ 0.488. Nếu notebook tìm được run Table 1 mang tên gốc, các giá trị là ≈ 0.350 / 0.481 và ≈ 0.383 / 0.499. Lệch xa cả hai bộ số thì báo nhóm trước khi đọc tiếp.
- **Kiểm tra hợp lý:** IR của `LiDAR K=200` **cao hơn** `LiDAR K=50` (nhiều ứng viên hơn thì dẫn hướng tốt hơn). Nếu thấp hơn, có thể gộp sai: báo nhóm.
- **Ghi lại thêm một con số:** IR(`LiDAR K=200`) − IR(`LiDAR K=50`), tức lợi ích của việc tăng K. Con số này dùng cho phần dự đoán ở trên.
- **Kết quả tốt nhất:** dòng `LiDAR K=200` có `ΔIR` và `ΔGenEval` **âm hoặc bằng 0**. Nếu dương thì **chưa phải thất bại**, vì LiDAR K = 200 tốn gấp 2.1 lần thời gian; khi đó chạy giai đoạn 2.

**⚠️ Lưu ý**

- Tổng thời gian dài nên có thể cần 2 phiên Colab. Phần sinh ứng viên bị ngắt thì Run all lại là chạy tiếp. Run `LiDAR K=200` dùng seed toàn cục: bị ngắt giữa run thì notebook in đường dẫn và dừng; xoá thư mục đó trên Drive rồi Run all lại. Chỉ bắt đầu phiên chạy Phase 2 khi còn ít nhất ~4–5 h.

**📊 Đọc kết quả giai đoạn 1 và làm gì tiếp**

| Kết quả của dòng `LiDAR K=200` (Δ = LiDAR K=200 − RS-LiDAR K=50) | Nghĩa là | Làm gì tiếp |
|---|---|---|
| ✅ ΔIR ≤ 0 và ΔGenEval ≤ 0 | RS-LiDAR thắng cả khi LiDAR được cùng số lần gọi reward và tốn gấp 2.1 lần thời gian | Claim mạnh nhất. Giai đoạn 2 chỉ chạy RS-LiDAR K = 200 (tuỳ chọn, để có thông điệp "cộng thêm"); bỏ LiDAR K = 66 vì RS-LiDAR đã thắng một LiDAR tốn hơn |
| ⚠️ Ngang nhau (CI chứa 0) | Cùng chất lượng với chi phí bằng 47% | Claim về chi phí. Chạy giai đoạn 2 đầy đủ để có phép so cùng thời gian |
| ❌ Dương, có `*` | Đúng dự đoán: nhiều ứng viên hơn thì tốt hơn, nhưng tốn gấp 2.1 lần thời gian | **Chưa kết luận.** Chạy giai đoạn 2 đầy đủ; quyết định theo bảng cuối |

**Giai đoạn 2 (E3b): cùng thời gian và cộng thêm.** Ba notebook E3b ở bảng 📋 phía trên dùng lại 4 pool của giai đoạn 1, không sinh ứng viên mới. Mục 6 của mỗi notebook dừng và báo nếu thiếu pool, run mốc hoặc run `E3_LiDAR_…_K200`. Mục 9 in hai bảng có CI, bảng chất lượng theo thời gian và kết luận S1–S5, rồi lưu `E3b_metrics.md` (bảng gọn, không CI) và `E3b_verdict.md`.

| Notebook | Run | Cách tạo | So với |
|---|---|---|---|
| [E3b-1](E3_compute_matched/E3b1_EqualTime_LiDAR_K66_SD15_DDPM100_Colab.ipynb) | **LiDAR K = 66** | Pool seed 100 (50 ứng viên) + 16 ứng viên **chọn ngẫu nhiên** (rng cố định) từ pool seed 200, rồi Phase 2 như Table 1 | RS-LiDAR K = 50 (cùng thời gian) |
| [E3b-2](E3_compute_matched/E3b2_AddedValue_RSLiDAR_K200_SD15_DDPM100_Colab.ipynb) | **RS-LiDAR K = 200, M = 4** | Chấm R̄σ (σ = 1, M = 4, i.i.d.) cho 150 ứng viên của pool seed 200, 300, 400 (pool seed 100 đã có điểm R̄σ), ghép theo R̄σ, rồi Phase 2 | LiDAR K = 200 (cùng 200 ứng viên, ghép cặp) |
| [E3b-3](E3_compute_matched/E3b3_Optional_LiDAR_K100_SD15_DDPM100_Colab.ipynb) (tuỳ chọn) | LiDAR K = 100 | Pool seed 100 + 200, chỉ Phase 2 | Thêm một điểm cho hình chất lượng theo thời gian |

Lưu ý khi tạo LiDAR K = 66: **không** dùng `top_k = 66` trên pool 200. Làm vậy sẽ lấy 66 ứng viên tốt nhất trong 200, tức đã trả chi phí sinh 200 ứng viên. Phải lấy 16 ứng viên ngẫu nhiên để đúng là "sinh 66 ứng viên".

**📊 Bảng quyết định cuối** (sau giai đoạn 2; CI 95% bootstrap theo prompt; notebook E3b tự áp dụng và ghi vào `E3b_verdict.md`):
- **Thua** = LiDAR hơn, có `*`, ở IR hoặc GenEval.
- **Thắng** = RS-LiDAR ≥ LiDAR theo điểm ước lượng ở cả IR và GenEval, và LiDAR không hơn có `*` ở chỉ số nào (IR, CLIP, HPS, GenEval).
- **Hoà** = các trường hợp còn lại (ví dụ IR và GenEval ngược chiều nhau, hoặc LiDAR hơn có `*` ở CLIP/HPS).

Vì sao không đòi CI của RS-LiDAR không chứa 0 (sửa 22/11): mọi run E3 chỉ có một seed. Ở seed 100, ngay cả phép so chính RS-LiDAR với LiDAR K = 50 cũng có CI chứa 0 (ΔIR −0.036 [−0.082, +0.008]), dù Table 1 thắng ở cả 3 seed. Đòi CI không chứa 0 thì gần như mọi kết quả sẽ thành "Hoà". Tiêu chí trên đọc một seed theo cùng cách bài đang đọc E1 và D.6.

| # | Cùng thời gian: RS K=50 vs LiDAR K=66 | Cộng thêm: RS K=200 vs LiDAR K=200 | Kết luận | Làm gì |
|---|---|---|---|---|
| S1 | Thắng | Thắng | Smoothing đáng giá hơn việc sinh thêm ứng viên, và vẫn có lợi khi đã có nhiều ứng viên | **Nộp ICML.** Một câu ở §6.1 + hình chất lượng theo thời gian ở appendix; rebuttal D1 dùng cả hai |
| S2 | Thắng | Hoà hoặc Thua | Ở cùng thời gian RS-LiDAR thắng, nhưng lợi ích giảm khi K lớn | **Nộp ICML**, claim giới hạn ở "at equal wall-clock time"; ghi trung thực vào Limitations rằng lợi ích nhỏ dần khi K lớn |
| S3 | Hoà | Thắng | Cùng thời gian thì ngang; nhưng trên cùng một pool, smoothing luôn cộng thêm | Nộp được, nhưng đổi cách viết: smoothing **bổ sung** cho việc tăng K chứ không thay thế. Cả nhóm bàn trước khi viết |
| S4 | Hoà | Hoà hoặc Thua | Lợi ích của RS-LiDAR tương đương việc chi thêm thời gian cho ứng viên | Chưa nộp ICML với câu chuyện này → **bước cứu** bên dưới |
| S5 | Thua | bất kỳ | Chi thời gian để sinh thêm ứng viên tốt hơn smoothing | Thất bại theo cổng của checklist → **bước cứu** bên dưới |

**Bước cứu** (chỉ cho S4/S5; chỉ thử **một** lần, có giới hạn chi phí):
1. **SDXL cùng thời gian** (F4 trong checklist). Đây là thí nghiệm **phụ**: không nằm trong kế hoạch chính và mặc định không chạy. Chỉ chạy khi nhóm đã bàn và quyết định. Notebook: [`E3_compute_matched/E3c_SDXL_EqualTime_K119_vs_RSK100_Colab.ipynb`](E3_compute_matched/E3c_SDXL_EqualTime_K119_vs_RSK100_Colab.ipynb). Chọn **A100**, Run all, không sửa gì.
   - Notebook sinh 19 ứng viên mỗi prompt (seed 200), ghép với pool K = 100 của Table 1, chạy LiDAR K = 119, rồi so với RS-LiDAR K = 100 và in kết luận vào `E3c_verdict.md`.
   - Notebook tự tìm hai run mốc SDXL seed 100 của Table 1 và kiểm tra cấu hình. Nếu không có hoặc không khớp, nó tự chạy lại hai mốc (thêm ~15 h A100); dòng `▶ Không có run mốc …` báo trường hợp này. Trên SDXL, phần thêm của RS-LiDAR chỉ là +7.5% thời gian (78.67 → 84.58 s), so với +14.5% trên SD 1.5. Vì vậy điểm hoà thời gian là K ≈ 117.7; notebook dùng K = 119 (≈ 85.02 s, nhỉnh hơn 84.58 s của RS-LiDAR K = 100). Chi phí khoảng 8 h A100 (sinh thêm 19 ứng viên mỗi prompt + Phase 2 SDXL). Phải báo cả kết quả SD 1.5, không được giấu.
2. Nếu SDXL cũng không thắng: theo bảng quyết định của checklist, **không nộp ICML với câu chuyện hiện tại**. Chuyển sang TMLR hoặc workshop, viết thành bài phân tích về reward spike; E1, E2 và E4 vẫn đứng được độc lập với chi phí.

Không nên thử thêm các phương án sau, vì số đã có cho thấy chúng không cứu được:
- **Giảm M xuống 2 để rẻ hơn:** Appendix D.8 cho IR 0.349, gần như bằng LiDAR (0.346).
- **Giảm δ để sinh ứng viên rẻ hơn:** RS-LiDAR ở δ = 2 thua xa LiDAR ở δ = 5.

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

## 4D. Việc mới sau E3 (T11)

<a id="t11"></a>
### T11 · (nhóm phân công) · F1: làm reward ổn định bằng cách khác, cùng số lần gọi reward

| | |
|---|---|
| 👤 Người chạy | Quân hoặc Hiếu, nhóm phân công. **Ba notebook chạy song song được** (mỗi notebook một tab Colab, cùng một tài khoản cũng được) |
| 🧪 Thí nghiệm | F1 (checklist F1) |
| 📓 Notebook | 3 notebook đã cấu hình sẵn, xem bảng 📋 dưới |
| 🖥️ GPU | **L4 + High-RAM** |
| ⏱️ Thời gian | F1a ~5–7 h, F1b ~6–9 h, F1c ~6–9 h trên L4 |
| 🔗 Cần xong trước | Không cần gì mới: chỉ dùng pool seed 100 của Table 1 và hai run mốc `E1_…` đã có trên Drive |
| 📤 Nộp lại | Mỗi notebook hai file: `F1_metrics_Aug.md` + `F1_summary_Aug.csv` (F1a), `F1_metrics_Latent_sz0.6.md` + `F1_summary_Latent_sz0.6.csv` (F1b) và `F1_metrics_Latent_sz0.3.md` + `F1_summary_Latent_sz0.3.csv` (F1c). File `.md` là bảng gọn không CI, file `.csv` có CI |

**🎯 Mục tiêu.** Reviewer sẽ hỏi: *"Sao phải làm trơn bằng nhiễu Gaussian trên pixel? Lấy trung bình reward theo cách khác có được không?"* F1 so RS-LiDAR với hai cách khác (cách thứ hai ở hai mức nhiễu). Mọi biến thể đều gọi ImageReward **4 lần cho mỗi ứng viên**, như RS-LiDAR (M = 4), và đều chấm lại **đúng 50 ứng viên** của pool seed 100. Phase 2 cũng chạy với seed 100, nên so được ghép cặp với LiDAR và RS-LiDAR seed 100 đã có. F1 không sinh ứng viên mới và không sửa code trong repo.

**📋 Kế hoạch chạy.** Mỗi notebook một phiên Colab: upload, chọn GPU, **Runtime → Run all**, không sửa gì.

| Bước | Notebook (bấm để mở) | Chạy gì | Thời gian (L4) | Gửi lại |
|---|---|---|---|---|
| 1 | [`F1a_Aug_SD15_DDPM100_Colab.ipynb`](F1_augmentation/F1a_Aug_SD15_DDPM100_Colab.ipynb) | **LiDAR-Aug:** trung bình IR qua 4 biến đổi nhẹ, cố định (lật ngang; crop 90% rồi resize lại; JPEG q = 75; Gaussian blur r = 1) | ~5–7 h | `F1_metrics_Aug.md`, `F1_summary_Aug.csv` |
| 2 | [`F1b_LatentSmooth_sz0.6_SD15_DDPM100_Colab.ipynb`](F1_augmentation/F1b_LatentSmooth_sz0.6_SD15_DDPM100_Colab.ipynb) | **LiDAR-LatentSmooth:** cộng nhiễu N(0, 0.6²) vào latent, decode, chấm IR, lấy trung bình 4 lần | ~6–9 h | `F1_metrics_Latent_sz0.6.md`, `F1_summary_Latent_sz0.6.csv` |
| 3 | [`F1c_LatentSmooth_sz0.3_SD15_DDPM100_Colab.ipynb`](F1_augmentation/F1c_LatentSmooth_sz0.3_SD15_DDPM100_Colab.ipynb) | **LiDAR-LatentSmooth:** cộng nhiễu N(0, 0.3²) vào latent, decode, chấm IR, lấy trung bình 4 lần | ~6–9 h | `F1_metrics_Latent_sz0.3.md`, `F1_summary_Latent_sz0.3.csv` |

- Ba notebook độc lập, **chạy song song được**: mỗi notebook ghi file điểm, run Phase 2, cache GenEval và bảng kết quả **riêng** (đuôi `_Aug`, `_Latent_sz0.6` hoặc `_Latent_sz0.3`), không ghi đè nhau. Bảng của mỗi notebook có 3 dòng: RS-LiDAR, LiDAR và biến thể của notebook đó; gửi cả ba file `.md` để ghép.
- **σ_z được đặt trước khi chạy, bằng đo đạc, không dùng reward** (23/11): sau khi decode và resize 224 như ImageReward, nhiễu pixel σ = 1 của RS-LiDAR làm mỗi pixel lệch 0.302 (Appendix A.1 báo 0.29); latent **σ_z = 0.6** làm lệch 0.307, nên là mức chính (F1b). F1c chạy mức nhẹ hơn **σ_z = 0.3** (0.120) để kiểm độ vững. Số đo: `experiment_outputs/results/F1_augmentation/sigma_z_measurement_prompt98.json`. Nếu không chạy được cả ba cùng lúc: F1a và F1b trước, F1c sau.
- Phần chấm lại (mục 7) chạy tiếp được sau khi bị ngắt. Phase 2 (mục 8) phải chạy **một mạch**. Nếu bị ngắt giữa Phase 2: xoá thư mục `F1_…` đang dở trên Drive rồi Run all lại.

**⏸ Kiểm tra trong 10–15 phút đầu:**
- **Mục 5:** `VARIANT = "aug"` (F1a) hoặc `VARIANT = "latent"` với `SIGMA_Z = 0.6` (F1b) hoặc `SIGMA_Z = 0.3` (F1c).
- **Mục 6:** `✅ Pool: Lookahead_SD15_…_seed100`, rồi 2 dòng `✅ Run mốc khớp cấu hình:` cho `E1_RSLiDAR_…` và `E1_LiDAR_…`.
- **Mục 7:** thanh tiến trình `Chấm Aug`, `Chấm Latent_sz0.6` hoặc `Chấm Latent_sz0.3` chạy tới 553 prompt.
- **Mục 9:** bảng có `RS-LiDAR K=50 M=4` (IR ≈ 0.374) và `LiDAR K=50` (IR ≈ 0.338). Lệch xa thì báo nhóm.

**📊 Đọc kết quả** (mỗi `F1_metrics_<biến thể>.md` ghi sẵn kết luận cho biến thể của nó, dùng cùng tiêu chí một seed như E3b):

| Kết luận "RS-LiDAR so với …" | Nghĩa là | Làm gì |
|---|---|---|
| ✅ Thắng ở cả hai biến thể | Lợi ích đến từ làm trơn Gaussian trên pixel, không phải từ việc lấy trung bình reward nói chung | Một câu ở §6.4 (Ablations) + bảng nhỏ ở appendix; rebuttal trả lời "why Gaussian pixel" |
| ⚠️ Hoà ở một biến thể | Biến thể đó làm được gần như RS-LiDAR | Ghi trung thực ở appendix. Lập luận còn lại của bài: Prop 2 chỉ áp dụng cho nhiễu Gaussian, và biến thể đó không có bảo đảm lý thuyết |
| ❌ Thua (biến thể kia hơn có `*`) | Cách làm ổn định khác tốt hơn | Cả nhóm bàn: báo trung thực, cân nhắc đổi cách trình bày |

<a id="t12"></a>
### T12 · Quân · F3: bộ prompt thứ hai (DrawBench), σ cố định từ trước

| | |
|---|---|
| 👤 Người chạy | Quân |
| 🧪 Thí nghiệm | F3 (checklist F3); chốt trước trong `experiment_outputs/results/F3_drawbench/F3_preregistration.json` (24/11/2026) |
| 📓 Notebook | [`F3a_DrawBench_Generate_SD15_DDPM100_Colab.ipynb`](F3_drawbench/F3a_DrawBench_Generate_SD15_DDPM100_Colab.ipynb) (sinh ảnh, một seed mỗi lần), [`F3c_DrawBench_Score_VQAScore_XXL_A100_Colab.ipynb`](F3_drawbench/F3c_DrawBench_Score_VQAScore_XXL_A100_Colab.ipynb) (VQAScore-XXL và bảng tổng hợp, chấm sau) |
| 🖥️ GPU | F3a: **L4 + High-RAM**. F3c: **A100 + High-RAM** (mô hình 11B, nạp cần ~45 GB RAM) |
| ⏱️ Thời gian | F3a khoảng 2.5–3 h mỗi seed; F3c khoảng 15–30 phút mỗi seed, cộng 10–15 phút tải mô hình mỗi lần mở runtime |
| 🔗 Cần xong trước | Không cần gì: pool ứng viên mới cho từng seed. Nếu bản clone chưa có `prompt_files/drawbench.jsonl`, F3a tự tạo từ bảng DrawBench chính thức và kiểm checksum |
| 📤 Nộp lại | `F3_metrics.md` và `F3_summary.csv` (bản cuối do F3c ghi), cùng `F3_gen_seed<SEED>.json` (F3a) |

**🎯 Mục tiêu.** Cả ba reviewer mô phỏng (24/11) xếp yêu cầu này lên đầu: σ được chọn trên GenEval, vậy cấu hình đó có còn giúp trên một bộ prompt khác không. F3 chạy đúng dòng SD 1.5 DDPM-100 của Table 1 trên DrawBench, không chỉnh tham số nào.

**Chỉ số (đã chốt trước).**
- IR: reward dẫn hướng, chỉ báo, không dùng làm bằng chứng.
- Held-out:
  - CLIPScore và HPSv2.1, như Table 1;
  - VQAScore với CLIP-FlanT5-XXL (mặc định của `t2v-metrics`), dùng thay GenEval vì DrawBench không có metadata đối tượng.
- Báo cáo:
  - mean ± std qua các seed;
  - Δ ghép cặp theo từng seed;
  - % prompt mà RS-LiDAR thắng.
- Không có CI. Báo đủ mọi chỉ số, dù kết quả theo chiều nào.
- Chốt trước:
  - chỉ số chính là VQAScore;
  - quy tắc diễn đạt: "cao hơn" hoặc "thấp hơn" chỉ khi đúng ở mọi seed, còn lại là "ngang nhau";
  - bảng theo 11 nhóm chỉ để mô tả.

**📋 Chạy.**
0. **Chạy thử cả chuỗi trước (nên làm):** F3a với `SMOKE_TEST = True` (2 prompt, khoảng 10 phút), rồi F3c với `INCLUDE_SMOKE = True`. Cách này kiểm môi trường `t2v-metrics` trên A100 trước khi chạy thật.
1. F3a với `SEED = 100`, `SMOKE_TEST = False`.
2. Cuối F3a in bảng nhanh IR/CLIP/HPS của seed đó.
3. Lặp lại với `SEED = 50`, rồi `SEED = 150`.
4. Lúc nào có A100 thì chạy F3c trên runtime **A100 + High-RAM** mới. F3c chấm VQAScore-XXL cho mọi seed đã xong và ghi bảng đầy đủ. Không đổi sang bản XL hay lượng tử hoá.

**⏸ Kiểm tra trong 10–15 phút đầu.**
- **Mục 4** in `F3 DrawBench | seed 100 | 200 prompt` và không báo lỗi checksum.
- **Mục 5** chạy Phase 1, sau đó in `✅ Hai pool dùng chung 50 ứng viên`.
- Phase 2 (mục 6) phải chạy một mạch. Nếu bị ngắt, xoá thư mục `F3DB_…` đang dở trên Drive rồi Run all lại.

<a id="t13"></a>
### T13 · Hiếu · F8: đối chứng thứ hạng (dẫn theo ứng viên hạng 2, hoặc ngẫu nhiên trong top-5)

| | |
|---|---|
| 👤 Người chạy | Hiếu. **Hai notebook chạy song song được** (mỗi notebook một tab Colab) |
| 🧪 Thí nghiệm | F8 (checklist "Việc còn lại" 24/11, mục (2)) |
| 📓 Notebook | 2 notebook đã cấu hình sẵn, xem bảng 📋 dưới. Bản sao ở thư mục gốc `RS-LiDAR/` |
| 🖥️ GPU | **L4 + High-RAM** (A100 cũng được, nhanh hơn khoảng 3 lần) |
| ⏱️ Thời gian | Mỗi notebook ~4–5.5 h trên L4 (Phase 2 ~3–5 h + GenEval ~30 phút); ~1.5 h trên A100 |
| 🔗 Cần xong trước | Không cần gì mới: chỉ dùng pool seed 100 của Table 1 và hai run mốc `E1_…` đã có trên Drive (như F1) |
| 📤 Nộp lại | `F8_metrics_Rank2.md` (F8a) và `F8_metrics_RandTop5.md` (F8b), trong `MyDrive/RS-LiDAR/extra_experiments/F8_rank_control/`. Kèm `.csv` nếu được |

**🎯 Mục tiêu.** Reviewer (mock review 24/11, R1 và R3) hỏi: *RS-LiDAR hơn LiDAR vì chọn **đúng** ứng viên khác, hay chỉ vì **tránh** ứng viên hạng 1 theo R?* Hai run đối chứng giữ nguyên mọi thứ của LiDAR, chỉ đổi ứng viên đứng đầu:
- **F8a · LiDAR-Rank2:** ở mọi prompt, ứng viên hạng 2 theo R đổi điểm với hạng 1, nên LiDAR dẫn theo hạng 2;
- **F8b · LiDAR-RandTop5:** ở mỗi prompt, chọn ngẫu nhiên đều một ứng viên hạng 1–5 theo R (rng cố định theo prompt), đổi điểm với hạng 1.

Chỉ **hoán đổi điểm của hai ứng viên**: cùng 50 latent, cùng tập điểm R (độ nhọn của softmax λR giữ nguyên), λ = 5000, K = 50, DDPM-100, seed 100. Không gọi reward lại, không sinh ứng viên mới, không sửa code trong repo. So ghép cặp với run mốc LiDAR (IR ≈ 0.338) và RS-LiDAR (IR ≈ 0.374) seed 100.

**📋 Kế hoạch chạy.** Mỗi notebook một phiên Colab: upload, chọn GPU, **Runtime → Run all**, không sửa gì.

| Bước | Notebook (bấm để mở) | Chạy gì | Thời gian (L4) | Gửi lại |
|---|---|---|---|---|
| 1 | [`F8a_Rank2_SD15_DDPM100_Colab.ipynb`](F8_rank_control/F8a_Rank2_SD15_DDPM100_Colab.ipynb) | **LiDAR-Rank2:** dẫn theo ứng viên hạng 2 theo R ở mọi prompt | ~4–5.5 h | `F8_metrics_Rank2.md` |
| 2 | [`F8b_RandTop5_SD15_DDPM100_Colab.ipynb`](F8_rank_control/F8b_RandTop5_SD15_DDPM100_Colab.ipynb) | **LiDAR-RandTop5:** dẫn theo một ứng viên ngẫu nhiên trong top-5 theo R | ~4–5.5 h | `F8_metrics_RandTop5.md` |

- Hai notebook ghi file, run Phase 2 và cache GenEval **riêng** (đuôi `Rank2`, `RandTop5`), không ghi đè nhau.
- Phase 2 (mục 8) phải chạy **một mạch**. Nếu bị ngắt giữa Phase 2: xoá thư mục `F8_…` đang dở trong `MyDrive/RS-LiDAR/Target_samples/` rồi Run all lại. Pool dẫn xuất nằm trên đĩa Colab và được tạo lại trong vài giây.

**⏸ Kiểm tra trong 10–15 phút đầu:**
- **Mục 5:** `VARIANT = "rank2"` (F8a) hoặc `VARIANT = "top5"` (F8b).
- **Mục 6:** `✅ Pool: Lookahead_SD15_…_seed100`, rồi 2 dòng `✅ Run mốc khớp cấu hình:` cho `E1_RSLiDAR_…` và `E1_LiDAR_…`.
- **Mục 7:** in `✅ Hàm hoán đổi sẵn sàng`.
- **Mục 8:** in `✅ Pool dẫn xuất đúng. Hạng (theo R) của ứng viên được đưa lên đầu: {2: 553}` (F8a) hoặc khoảng 110 prompt cho mỗi hạng 1–5 (F8b), rồi Phase 2 chạy.
- **Mục 9:** bảng có `RS-LiDAR K=50 M=4` (IR ≈ 0.374) và `LiDAR K=50` (IR ≈ 0.338). Lệch xa thì báo nhóm.

**📊 Đọc kết quả** (mỗi `F8_metrics_<…>.md` ghi sẵn kết luận "RS-LiDAR so với …" theo tiêu chí một seed như E3b):

| Kết quả | Nghĩa là | Làm gì |
|---|---|---|
| ✅ Hai run đối chứng không hơn LiDAR, RS-LiDAR thắng cả hai | Chỉ tránh ứng viên hạng 1 thì không đủ; lợi ích đến từ việc R̄σ chọn đúng ứng viên | Một câu ở §6.3 (cạnh đoạn clipping / M = 1) + một dòng ở D.13; rebuttal B4 |
| ⚠️ Đối chứng hơn LiDAR nhưng RS-LiDAR vẫn thắng | Tránh hạng 1 có giúp một phần; R̄σ chọn tốt hơn chọn mù | Báo trung thực ở appendix, diễn đạt phần "chọn đúng" theo đúng mức chênh |
| ❌ Đối chứng hoà hoặc hơn RS-LiDAR | Lợi ích chủ yếu do tránh ứng viên hạng 1 | Cả nhóm bàn: báo trung thực, viết lại lập luận cơ chế ở §6.3 |

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
| [E3](#ref-e3) | Cùng ngân sách reward và cùng thời gian | "RS gọi reward nhiều hơn 4 lần, thắng là đương nhiên? Sao không dùng phần thời gian đó để sinh thêm ứng viên?" | Section 6.1, đoạn chi phí + appendix |
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

**Vì sao cần thêm giai đoạn 2 (21/11).** Cùng số lần gọi reward chưa phải cùng chi phí. Thời gian chủ yếu nằm ở việc sinh ứng viên (0.114 s mỗi ứng viên), còn mỗi lần gọi reward chỉ tốn 0.013 s. Vì vậy LiDAR K = 200 tốn khoảng 32.4 s cho 4 ảnh, **gấp 2.1 lần** RS-LiDAR K = 50 (15.36 s). LiDAR K = 200 thắng là điều dễ xảy ra và chưa bác bỏ được gì. Phép thử công bằng là so **cùng thời gian**: LiDAR K = 66 (≈ 15.44 s, nhỉnh hơn) với RS-LiDAR K = 50 (15.36 s). Thêm RS-LiDAR K = 200 để xem smoothing có cộng thêm vào lợi ích của việc tăng K hay không. Chi tiết và cách quyết định ở [thẻ T5](#t5).

**Vị trí trong bài.** Một câu ở đoạn chi phí của Section 6.1, kèm bảng hoặc hình chất lượng theo thời gian ở appendix. Claim tuỳ kết quả, xem bảng quyết định cuối của [thẻ T5](#t5). Ví dụ khi thắng cả hai: *"At equal wall-clock time, RS-LiDAR (K = 50, M = 4) outperforms LiDAR with K = 66, and smoothing still improves LiDAR when both use K = 200 candidates."*

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
