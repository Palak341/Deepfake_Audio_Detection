"""
evaluation.py
-------------
Accuracy, EER, ROC-AUC, F1, confusion matrices, and comparison reporting
for the general vs. personalized detectors.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from src.utils import config


def eer_threshold(y_true, probs_spoof):
    """Equal Error Rate: the threshold at which false-positive rate equals
    false-negative rate. Returns (threshold, eer)."""
    fpr, tpr, thresholds = roc_curve(y_true, probs_spoof)
    fnr = 1 - tpr
    finite = np.isfinite(thresholds)
    thr, fpr_, fnr_ = thresholds[finite], fpr[finite], fnr[finite]
    srt = np.argsort(thr)
    thr, fpr_, fnr_ = thr[srt], fpr_[srt], fnr_[srt]
    f = interp1d(thr, fpr_ - fnr_)
    t = brentq(f, thr[0], thr[-1])
    eer = float(interp1d(thr, fpr_)(t))
    return t, eer


def best_f1_threshold(y_true, probs_spoof):
    """Threshold that maximizes F1 on the precision-recall curve."""
    prec, rec, thresh = precision_recall_curve(y_true, probs_spoof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_i = np.argmax(f1[:-1])
    return thresh[best_i]


def evaluate_probs(y_true, probs_spoof, threshold=0.5, target_names=None):
    """Full report (classification report + AUC + EER) for a set of
    spoof-class probabilities at a given decision threshold."""
    target_names = target_names or list(config.INVERSE_LABEL_MAP.values())
    preds = (probs_spoof >= threshold).astype(int)
    report = classification_report(y_true, preds, target_names=target_names)
    auc_val = roc_auc_score(y_true, probs_spoof)
    _, eer = eer_threshold(y_true, probs_spoof)
    return {
        "report": report,
        "auc": auc_val,
        "eer": eer,
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "preds": preds,
    }


def compare_general_vs_personalized(true_labels, general_preds, personalized_preds):
    """Side-by-side accuracy/F1/precision/recall table for the general vs.
    personalized detectors, with a delta column."""
    comparison_df = pd.DataFrame({
        "metric": ["accuracy", "f1", "precision", "recall"],
        "general": [
            accuracy_score(true_labels, general_preds),
            f1_score(true_labels, general_preds),
            precision_score(true_labels, general_preds),
            recall_score(true_labels, general_preds),
        ],
        "personalized": [
            accuracy_score(true_labels, personalized_preds),
            f1_score(true_labels, personalized_preds),
            precision_score(true_labels, personalized_preds),
            recall_score(true_labels, personalized_preds),
        ],
    })
    comparison_df["delta"] = comparison_df["personalized"] - comparison_df["general"]
    return comparison_df


def plot_confusion_matrices(true_labels, preds_list, titles, target_names=None, save_path=None):
    import matplotlib.pyplot as plt

    target_names = target_names or list(config.INVERSE_LABEL_MAP.values())
    fig, axes = plt.subplots(1, len(preds_list), figsize=(5.5 * len(preds_list), 4))
    if len(preds_list) == 1:
        axes = [axes]
    for ax, preds, title in zip(axes, preds_list, titles):
        cm = confusion_matrix(true_labels, preds)
        ax.imshow(cm, cmap="Greens")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(target_names, rotation=15)
        ax.set_yticklabels(target_names)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_roc_curve(y_true, probs_spoof, title="ROC Curve", save_path=None):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, probs_spoof)
    auc_val = roc_auc_score(y_true, probs_spoof)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def memory_bank_stats(memory_bank):
    """Per-speaker memory bank summary: entry counts, average confidence,
    average age weight, and total raw samples represented (after prototype
    compression)."""
    rows = memory_bank.conn.execute(
        "SELECT speaker_id, COUNT(*) as n_entries, AVG(confidence) as avg_conf, "
        "AVG(age_weight) as avg_age_weight, SUM(cluster_size) as total_raw_samples_represented "
        "FROM embeddings GROUP BY speaker_id ORDER BY n_entries DESC"
    ).fetchall()
    mem_df = pd.DataFrame(rows, columns=[
        "speaker_id", "n_entries", "avg_confidence", "avg_age_weight", "total_raw_samples_represented"
    ])
    print(f"Speakers with at least one gated memory entry: {len(mem_df)}")
    print(f"Speakers whose memory was prototype-compressed: "
          f"{(mem_df['total_raw_samples_represented'] > mem_df['n_entries']).sum()}")
    return mem_df
