# results_logger.py
import os
import csv
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def append_result_row(csv_path, row_dict):
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def compute_perturbation_metrics(original_image, adv_image):
    diff = adv_image - original_image
    l2 = float(np.linalg.norm(diff.ravel(), ord=2))
    linf = float(np.max(np.abs(diff)))
    return l2, linf

def build_result_row(
    image_name,
    method,
    attack_type,
    clean_num_boxes,
    adv_num_boxes,
    clean_score_sum,
    adv_score_sum,
    runtime_sec,
    original_image,
    adv_image,
    top_k_boxes=None,
    grid_size=None
):
    l2, linf = compute_perturbation_metrics(original_image, adv_image)

    score_reduction = clean_score_sum - adv_score_sum
    reduction_ratio = score_reduction / clean_score_sum if clean_score_sum > 1e-8 else 0.0

    attack_success = int(
        (adv_num_boxes < clean_num_boxes) or
        (reduction_ratio >= 0.5)
    )

    return {
        "image_name": image_name,
        "method": method,
        "attack_type": attack_type,
        "clean_num_boxes": clean_num_boxes,
        "adv_num_boxes": adv_num_boxes,
        "clean_score_sum": round(clean_score_sum, 6),
        "adv_score_sum": round(adv_score_sum, 6),
        "score_reduction": round(score_reduction, 6),
        "reduction_ratio": round(reduction_ratio, 6),
        "attack_success": attack_success,
        "runtime_sec": round(runtime_sec, 4),
        "l2_perturb": round(l2, 6),
        "linf_perturb": round(linf, 6),
        "top_k_boxes": top_k_boxes if top_k_boxes is not None else "",
        "grid_size": grid_size if grid_size is not None else ""
    }