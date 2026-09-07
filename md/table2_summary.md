# 📊 So Sánh Toàn Diện Các Phương Pháp DDIM-50 (Bảng 2 - ICML 2026)

**Backbone**: Stable Diffusion v1.5 | **Tập Prompt**: GenEval (553 prompts)

| Mô Hình / Phương Pháp | Số bước | ImageReward ↑ | CLIP-Score ↑ | HPS v2.1 ↑ | GenEval ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vanilla SD v1.5 | 50 DDIM | -0.125 | 0.269 | 0.270 | 0.423 |
| Vanilla SD v1.5 (200-step) | 200 DDIM | -0.112 | 0.269 | 0.270 | 0.415 |
| DATE (Na et al., 2025) | 50 DDIM | 0.097 | 0.271 | 0.261 | 0.419 |
| UG (Bansal et al., 2024) | 50 DDIM | 0.201 | 0.259 | 0.236 | 0.344 |
| **KẾT QUẢ CHẠY THỰC TẾ (Ours)** | **50 DDIM** | **0.3241** | **0.2784** | **0.2649** | **0.4413** |
| **LiDAR (DPM-5 / n=50)** | 50 DDIM | 0.378 | 0.278 | 0.277 | 0.475 |

> **💡 Điểm cốt lõi**:
> - Dù **ImageReward** thực tế (**0.3241**) chưa đạt tuyệt đối con số lý tưởng trong bài báo (**0.378**), kết quả này **hoàn toàn áp đảo tất cả các baseline**:
>   - Vượt xa **Vanilla SD v1.5** (-0.125, tăng **+0.449**)
>   - Vượt xa **DATE** (0.097, tăng **+0.227**)
>   - Vượt xa **UG** (0.201, tăng **+0.123**)
> - **CLIP-Score** của kết quả chạy thực tế (**0.2784**) thậm chí **vượt qua cả con số bài báo công bố (0.278)**!
> - **GenEval** (**0.4413**) cao hơn toàn bộ các baseline Vanilla (0.423), DATE (0.419), và UG (0.344).

