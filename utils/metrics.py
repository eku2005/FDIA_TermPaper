"""
metrics.py
──────────
Evaluation metrics and threshold selection utilities.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
) -> Dict:
    """
    Compute detection metrics.

    Parameters
    ----------
    y_true  : (N,) true labels {0, 1}
    y_pred  : (N,) predicted labels {0, 1}
    y_score : (N,) continuous scores (for AUC)

    Returns
    -------
    dict of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    dr  = tp / (tp + fn + 1e-9)
    far = fp / (fp + tn + 1e-9)
    pre = tp / (tp + fp + 1e-9)
    f1  = 2 * pre * dr / (pre + dr + 1e-9)
    acc = (tp + tn) / len(y_true)

    metrics = {
        "detection_rate":   dr,
        "false_alarm_rate": far,
        "precision":        pre,
        "f1":               f1,
        "accuracy":         acc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            metrics["pr_auc"]  = average_precision_score(y_true, y_score)
        except Exception:
            pass

    return metrics


def find_threshold_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.03,
) -> float:
    """
    Find the score threshold that achieves ≤ target_fpr on normal samples.
    """
    normal_scores = y_score[y_true == 0]
    if len(normal_scores) == 0:
        return 0.5
    return float(np.quantile(normal_scores, 1.0 - target_fpr))


def print_metrics(metrics: Dict, prefix: str = ""):
    line = "─" * 50
    print(f"\n{line}")
    if prefix:
        print(f"  {prefix}")
    print(f"  Detection Rate   : {metrics['detection_rate']:.4f} "
          f"({metrics['detection_rate']*100:.1f}%)")
    print(f"  False Alarm Rate : {metrics['false_alarm_rate']:.4f} "
          f"({metrics['false_alarm_rate']*100:.1f}%)")
    print(f"  F1-Score         : {metrics['f1']:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f} "
          f"({metrics['accuracy']*100:.1f}%)")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")
    print(line)
