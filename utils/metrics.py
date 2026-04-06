import numpy as np


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return inter / (union + 1e-6)


def success_auc(pred_boxes, gt_boxes):
    thresholds = np.linspace(0, 1, 21)
    ious = [
        compute_iou(np.array(p), np.array(g))
        for p, g in zip(pred_boxes, gt_boxes)
        if g[2] > 0 and g[3] > 0
    ]
    if not ious:
        return 0.0
    ious = np.array(ious)
    success = [(ious >= t).mean() for t in thresholds]
    return float(np.mean(success))


def normalized_precision(pred_boxes, gt_boxes, threshold=0.1):
    scores = []
    for p, g in zip(pred_boxes, gt_boxes):
        if g[2] <= 0 or g[3] <= 0:
            continue
        p_cx = p[0] + p[2] / 2
        p_cy = p[1] + p[3] / 2
        g_cx = g[0] + g[2] / 2
        g_cy = g[1] + g[3] / 2
        dist = ((p_cx - g_cx)**2 + (p_cy - g_cy)**2) ** 0.5
        norm = (g[2] * g[3]) ** 0.5 + 1e-6
        scores.append(dist / norm < threshold)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_final_score(auc, norm_prec,
                        flops_g=0.0, params_m=0.0,
                        latency_ms=0.0, size_mb=0.0):
    s_acc    = 0.6 * auc + 0.4 * norm_prec
    e_flops  = min(1.0, flops_g    / 30.0)
    e_params = min(1.0, params_m   / 50.0)
    e_lat    = min(1.0, latency_ms / 30.0)
    e_size   = min(1.0, size_mb    / 500.0)
    s_eff    = 0.25*e_flops + 0.15*e_params + 0.35*e_lat + 0.25*e_size
    final    = s_acc - 0.2 * s_eff
    return {
        "final_score":      round(final, 4),
        "accuracy_score":   round(s_acc, 4),
        "efficiency_score": round(s_eff, 4),
        "auc":              round(auc, 4),
        "norm_precision":   round(norm_prec, 4),
    }