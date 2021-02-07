import numpy as np

def roc_auc_score(labels, pres):
    label_pctrs = zip(labels.reshape(-1), pres.reshape(-1))
    sorted_label_pctrs = sorted(
        label_pctrs, key=lambda lc: lc[1], reverse=True)
    last_pctr = -4374.2542434
    total_negative = 0
    total_positive = 0
    positive = 0
    negative = 0
    total_area = 0.0
    for (label, pctr) in sorted_label_pctrs:
        positive += label
        negative += 1 - label
        if pctr != last_pctr:
            last_pctr = pctr
            area = 0.5 * (
                    total_positive + total_positive + positive) * negative
            total_area += area
            total_positive += positive
            total_negative += negative
            positive = 0
            negative = 0
    area = 0.5 * (total_positive + total_positive + positive) * negative
    total_area += area
    total_positive += positive
    total_negative += negative
    return total_area / (total_positive * total_negative)
