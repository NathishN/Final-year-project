"""
open_set_recognition.py
========================
Open-Set Recognition (OOD Detection) Framework for Fruit Quality Grading.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Implemented Open-Set Detection Strategies:
1. Maximum Softmax Probability (MSP, Hendrycks & Gimpel, ICLR 2017)
2. Energy-Based OOD Detection (Liu et al., NeurIPS 2020)
3. Mahalanobis Distance OOD (Lee et al., NeurIPS 2018)

Evaluation Metrics:
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- FPR95 (False Positive Rate at 95% True Positive Rate)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve


# ─── 1. Maximum Softmax Probability (MSP) Score ──────────────────────────────
def compute_msp_scores(logits: np.ndarray) -> np.ndarray:
    """
    MSP Score: Max softmax probability across classes.
    Higher score => In-Distribution (ID) confidence.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return np.max(probs, axis=1)


# ─── 2. Energy-Based Score ───────────────────────────────────────────────────
def compute_energy_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Energy Score: T * log sum exp(z_i / T).
    Higher score => In-Distribution (ID).
    Free energy formulation (Liu et al., NeurIPS 2020).
    """
    scaled_logits = logits / temperature
    max_l = np.max(scaled_logits, axis=1, keepdims=True)
    energy = temperature * (np.log(np.sum(np.exp(scaled_logits - max_l), axis=1)) + max_l.squeeze())
    return energy


# ─── 3. Mahalanobis Distance Score ──────────────────────────────────────────
class MahalanobisOOD:
    """
    Class-conditional Mahalanobis distance OOD detector (Lee et al., NeurIPS 2018).
    Computes empirical class centroids and shared tied covariance matrix from ID training features.
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.centroids = []
        self.precision_matrix = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        num_samples, feat_dim = features.shape
        self.centroids = np.zeros((self.num_classes, feat_dim))
        tied_covariance = np.zeros((feat_dim, feat_dim))

        for c in range(self.num_classes):
            cls_mask = (labels == c)
            cls_feats = features[cls_mask]
            if len(cls_feats) > 0:
                self.centroids[c] = np.mean(cls_feats, axis=0)
                diff = cls_feats - self.centroids[c]
                tied_covariance += np.dot(diff.T, diff)

        tied_covariance /= num_samples

        # Add small regularization to diagonal to prevent singular matrix
        reg = 1e-5 * np.eye(feat_dim)
        self.precision_matrix = np.linalg.pinv(tied_covariance + reg)

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Returns negative minimum Mahalanobis distance to nearest class centroid.
        Higher score => In-Distribution (ID).
        Vectorized computation across all N samples and C classes.
        """
        # diff shape: (N, C, D)
        diff = features[:, None, :] - self.centroids[None, :, :]
        # Multiply by precision matrix: (N, C, D) @ (D, D) -> (N, C, D)
        diff_prec = np.matmul(diff, self.precision_matrix)
        # Sum along feature dimension: (N, C)
        dists_sq = np.sum(diff_prec * diff, axis=2)
        return -np.min(dists_sq, axis=1)


# ─── 4. OOD Metric Evaluation Functions ──────────────────────────────────────
def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Computes False Positive Rate at 95% True Positive Rate (FPR95)."""
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find FPR at TPR >= 0.95
    target_tpr = 0.95
    idx = np.searchsorted(tpr, target_tpr)
    if idx < len(fpr):
        return float(fpr[idx])
    else:
        return float(fpr[-1])


def evaluate_ood_detection(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """
    Computes AUROC, AUPR, and FPR95 for OOD detection.
    id_scores : Scores for In-Distribution test samples (higher = ID)
    ood_scores: Scores for Out-of-Distribution test samples
    """
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    # AUROC
    auroc = float(roc_auc_score(labels, scores))

    # AUPR (Precision-Recall curve area for ID as positive class)
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = float(auc(recall, precision))

    # FPR95
    fpr95 = compute_fpr95(id_scores, ood_scores)

    return {
        "AUROC": auroc * 100.0,
        "AUPR": aupr * 100.0,
        "FPR95": fpr95 * 100.0,
    }


def run_ood_benchmark(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    id_test_features: np.ndarray,
    id_test_logits: np.ndarray,
    ood_test_features: np.ndarray,
    ood_test_logits: np.ndarray,
) -> dict:
    """
    Run MSP, Energy, and Mahalanobis OOD detection evaluation.
    """
    results = {}

    # 1. MSP Benchmark
    print("Evaluating MSP (Maximum Softmax Probability)...")
    id_msp = compute_msp_scores(id_test_logits)
    ood_msp = compute_msp_scores(ood_test_logits)
    results["MSP"] = evaluate_ood_detection(id_msp, ood_msp)

    # 2. Energy Benchmark (T=1.0)
    print("Evaluating Energy-Based OOD Detection...")
    id_energy = compute_energy_scores(id_test_logits, temperature=1.0)
    ood_energy = compute_energy_scores(ood_test_logits, temperature=1.0)
    results["Energy"] = evaluate_ood_detection(id_energy, ood_energy)

    # 3. Mahalanobis Benchmark
    print("Evaluating Mahalanobis Distance OOD Detection...")
    mahalanobis_detector = MahalanobisOOD(num_classes=5)
    mahalanobis_detector.fit(train_features, train_labels)

    id_mahalanobis = mahalanobis_detector.score(id_test_features)
    ood_mahalanobis = mahalanobis_detector.score(ood_test_features)
    results["Mahalanobis"] = evaluate_ood_detection(id_mahalanobis, ood_mahalanobis)

    return results


if __name__ == "__main__":
    print("[OOD Framework] Open-Set Recognition module loaded successfully.")
