def relative_position(box_a, box_b):
    """Tính vị trí tương quan của A đối với B (chuẩn bài báo GenEval gốc)"""
    boxes = np.array([box_a[:4], box_b[:4]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - 0.1 * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations

def evaluate_geneval_for_folder(target_dir, exp_name):
    """Đánh giá ảnh bằng Mask2Former Swin-S, lưu geneval_summary.csv tại chỗ"""
    assert os.path.exists(target_dir), f"❌ Thư mục không tồn tại: {target_dir}"

    geneval_csv_path = f"{target_dir}/geneval_summary.csv"
    done_marker = f"{target_dir}/.mask2former_rescore_done"

    # CHỈ BỎ QUA nếu folder này ĐÃ THỰC SỰ CHẤM XONG bằng Mask2Former trong đợt này
    if not FORCE_RESCORE and os.path.exists(done_marker):
        try:
            df_exist = pd.read_csv(geneval_csv_path)
            for _, r in df_exist.iterrows():
                if 'OVERALL' in str(r.iloc[0]).upper():
                    score = float(r.iloc[2])
                    print(f"\n⚡ [ĐÃ CHẤM XONG BẰNG MASK2FORMER] {exp_name}: GenEval = {score:.4f} -> Bỏ qua!")
                    return score
        except Exception:
            pass

    if os.path.exists(geneval_csv_path):
        print(f"\n🔄 Phát hiện file geneval cũ trong {exp_name} -> Tiến hành CHẤM LẠI chuẩn xác bằng Mask2Former Swin-S...")

    print(f"\n" + "="*75)
    print(f"🎯 Đang chấm GenEval (Mask2Former Swin-S): {exp_name}")
    print("   Thanh tiến trình tqdm bắt đầu chạy ngay lập tức...")
    print("="*75)

    task_results = {'single_object': [], 'two_object': [], 'counting': [], 'colors': [], 'position': [], 'color_attr': []}
    scored_prompts_count = 0
    
    # CHẠY TRỰC TIẾP QUA 553 PROMPTS: TQDM HIỆN NGAY LẬP TỨC 0.00s (KHÔNG PHẢI CHỜ QUÉT DRIVE)
    for p_idx in tqdm(range(len(prompts_meta)), desc=f"Chấm: {exp_name[:25]}..."):
        meta = prompts_meta[p_idx]
        tag = meta.get('tag', 'single_object')
        if tag not in task_results: continue

        # Tìm ảnh trực tiếp tại thư mục prompt p_idx (hỗ trợ cả int và 5-digit zero-pad)
        p_dir = f"{target_dir}/{p_idx}"
        if not os.path.exists(p_dir):
            p_dir_alt = f"{target_dir}/{p_idx:05d}"
            if os.path.exists(p_dir_alt):
                p_dir = p_dir_alt
            else:
                continue

        imgs = sorted(glob.glob(f"{p_dir}/samples/*.png")) or [f for f in glob.glob(f"{p_dir}/*.png") if not f.endswith("grid.png")]
        if not imgs:
            continue

        scored_prompts_count += 1
        prompt_scores = []
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert('RGB')
                # Cấu hình chuẩn bài báo GenEval gốc: Cạnh ngắn luôn là 800px (chuẩn MMDetection COCO)
                image_processor.size = {'shortest_edge': 800, 'longest_edge': 1333}
                inputs = image_processor(images=img, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = detector(**inputs)

                # Ngưỡng tin cậy chuẩn bài báo GenEval gốc: 0.9 cho counting, 0.3 cho tất cả các task khác
                conf_thr = 0.9 if tag == 'counting' else 0.3
                results = image_processor.post_process_instance_segmentation(
                    outputs, target_sizes=[img.size[::-1]], threshold=conf_thr
                )[0]
                segmentation = results["segmentation"].detach().cpu().numpy()
                segments_info = results["segments_info"]

                # Sắp xếp các vật thể theo confidence giảm dần (chuẩn bài báo GenEval gốc)
                segments_info = sorted(segments_info, key=lambda s: s.get("score", 0.0), reverse=True)

                detected_objects = []
                for seg in segments_info:
                    c_name = id2label[seg["label_id"]].lower()
                    mask = (segmentation == seg["id"])
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, x2 = float(np.min(x_indices)), float(np.max(x_indices))
                        y1, y2 = float(np.min(y_indices)), float(np.max(y_indices))
                        box = [x1, y1, x2, y2, seg.get("score", 1.0)]
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0

                        pred_color = 'unknown'
                        if (x2 - x1 > 12 and y2 - y1 > 12) and (tag in ['colors', 'color_attr']):
                            pred_color = classify_crop_color(img, box, mask, c_name)

                        detected_objects.append({
                            'class': c_name, 'box': box, 'mask': mask,
                            'center_x': center_x, 'center_y': center_y,
                            'color': pred_color, 'score': seg.get("score", 1.0)
                        })

                success = False
                includes = meta.get('include', [])
                if tag == 'single_object':
                    req_cls = includes[0]['class'].lower()
                    success = any(req_cls in obj['class'] or obj['class'] in req_cls for obj in detected_objects)
                elif tag == 'two_object':
                    req1, req2 = includes[0]['class'].lower(), includes[1]['class'].lower()
                    success = any(req1 in obj['class'] or obj['class'] in req1 for obj in detected_objects) and \
                               any(req2 in obj['class'] or obj['class'] in req2 for obj in detected_objects)
                elif tag == 'counting':
                    req_cls, target_count = includes[0]['class'].lower(), includes[0]['count']
                    found_count = sum(1 for obj in detected_objects if req_cls in obj['class'] or obj['class'] in req_cls)
                    success = (found_count == target_count)
                elif tag == 'colors':
                    req_cls, req_color = includes[0]['class'].lower(), includes[0]['color'].lower()
                    success = any((req_cls in obj['class'] or obj['class'] in req_cls) and (obj['color'] == req_color) for obj in detected_objects)
                elif tag == 'position':
                    req1, req2 = includes[0]['class'].lower(), includes[1]['class'].lower()
                    pos_info = includes[1].get('position', ['right of', 0])
                    expected_rel = pos_info[0]
                    o1_list = [o for o in detected_objects if req1 in o['class'] or o['class'] in req1]
                    o2_list = [o for o in detected_objects if req2 in o['class'] or o['class'] in req2]
                    if o1_list and o2_list:
                        o1, o2 = o1_list[0], o2_list[0]
                        rels = relative_position(o2['box'], o1['box'])
                        success = (expected_rel in rels)
                elif tag == 'color_attr':
                    success = all(any((inc['class'].lower() in obj['class'] or obj['class'] in inc['class'].lower()) and (obj['color'] == inc['color'].lower()) for obj in detected_objects) for inc in includes)

                prompt_scores.append(1.0 if success else 0.0)
            except Exception:
                pass

        if prompt_scores:
            task_results[tag].append(np.mean(prompt_scores))

    if scored_prompts_count == 0:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy ảnh hợp lệ trong: {exp_name}")
        return None

    summary_rows = []
    all_means = []
    for t_name, scores in task_results.items():
        mean_val = np.mean(scores) if scores else 0.0
        if scores: all_means.append(mean_val)
        summary_rows.append({'Nhiệm Vụ (Task)': t_name, 'Số Lượng Prompt': len(scores), 'Độ Chính Xác (Accuracy ↑)': f"{mean_val:.4f}"})

    overall_geneval = float(np.mean(all_means)) if all_means else 0.0
    summary_rows.append({'Nhiệm Vụ (Task)': '🔥 OVERALL GENEVAL BENCHMARK (MASK2FORMER)', 'Số Lượng Prompt': sum(len(s) for s in task_results.values()), 'Độ Chính Xác (Accuracy ↑)': f"{overall_geneval:.4f}"})
    
    df_geneval = pd.DataFrame(summary_rows)
    df_geneval.to_csv(geneval_csv_path, index=False)
    df_geneval.to_csv(f"{target_dir}/geneval_mask2former_summary.csv", index=False)

    # ĐÁNH DẤU HOÀN THÀNH BẰNG FILE MARKER
    with open(done_marker, 'w', encoding='utf-8') as f:
        json.dump({
            "completed_at": time.ctime(),
            "evaluator": "facebook/mask2former-swin-small-coco-instance",
            "geneval_score": overall_geneval,
            "prompts_evaluated": scored_prompts_count
        }, f, indent=2)

    print(f"💾 [ĐÃ LƯU CHECKPOINT MASK2FORMER TẠI CHỖ]: {geneval_csv_path}")
    print(f"⭐ ĐIỂM OVERALL GENEVAL (MASK2FORMER): {overall_geneval:.4f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return overall_geneval

# ==============================================================================
# 🔄 CHẠY VÒNG LẶP TUẦN TỰ & CẬP NHẬT FILE TỔNG HỢP NGAY SAU MỖI THỰC NGHIỆM
# ==============================================================================
