"""
scripts/final_verify_ood.py
===========================
Final (Phase 5.7 lock) independent verification of the real-image OOD benchmark.

What this script does
---------------------
1. Rebuilds the exact 5,000-image OOD sample (seeded `random.sample`, 1,000 per class)
   with the same procedure as `phase50_stage4_real_ood.py` and checks that it matches
   the cached tensor `outputs/casm_v2/phase50_ood_features/ood_real_features.pt`
   path-for-path.
2. Confirms that the in-distribution (ID) feature cache `outputs/features/phase4_dataset.pt`
   is row-aligned with `outputs/casm_v2/phase50_split_v2_manifest.csv`.
3. Recomputes three detector scores from saved embeddings / checkpoint weights:
      Mahalanobis : min_k (x-mu_k)^T Sigma^-1 (x-mu_k), grade-conditional means mu_k and a
                    tied covariance fitted on the split_v2 TRAIN partition only.
      Energy      : E(x) = -T * logsumexp(z(x)/T), T = 1   (higher = more OOD)
      MSP         : -max_k softmax(z(x))_k               (higher = more OOD)
   where z(x) are the 5-way logits of the ConvNeXt-Tiny species classifier
   (`outputs/checkpoints/best_model.pth`), i.e. fc(LayerNorm(avgpool(features))).
4. Reproduces the HISTORICAL (defective) Energy/MSP computation, in which the ID logits
   were computed as fc(avgpool(features)) -- skipping the classifier-head LayerNorm --
   while the OOD logits came from the full forward pass. This reproduces the previously
   published 81.99% / 74.26% AUROC values and documents the defect.
5. Computes the CORRECTED metrics with the identical logit function for ID and OOD.

Metric conventions (authoritative)
----------------------------------
  * y = 1 denotes OOD (5,000 unseen-species images); y = 0 denotes the 12,833 split_v2 ID
    test images. All scores are oriented so that a HIGHER score = MORE OOD.
  * AUROC           : sklearn.metrics.roc_auc_score(y, s).
  * AUPR-OOD        : area under the precision-recall curve with OOD as the positive class,
                      computed by trapezoidal integration `auc(recall, precision)`
                      (the convention used since Phase 5.0; retained, not changed).
  * AP-OOD          : sklearn average_precision_score (step-wise; reported as a cross-check).
  * AUPR-In         : PR area with ID as positive class and the negated score.
  * FPR@95 (primary, unchanged since Phase 5.0): threshold chosen on the TEST scores as the
                      first ROC point at which >= 95% of OOD images are flagged (OOD TPR >= 0.95);
                      reported value = fraction of ID test images flagged as OOD at that threshold.
  * FPR@95-ID (supplementary, Liu et al. 2020 convention): threshold at which >= 95% of ID
                      images are accepted; value = fraction of OOD images accepted as ID.
  No validation-set threshold calibration is used for any reported metric.

Outputs: outputs/phase57_final_submission_lock/verification/ood/
"""
import os
import sys
import json
import random
import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import build_convnext_model  # noqa: E402

OOD_DIR = os.environ.get("FRUIT_OOD_DIR", os.path.join(ROOT, "open-set-dataset"))
OUT_DIR = os.path.join(ROOT, "outputs", "phase57_final_submission_lock", "verification", "ood")
OOD_CACHE = os.path.join(ROOT, "outputs", "casm_v2", "phase50_ood_features", "ood_real_features.pt")
ID_CACHE = os.path.join(ROOT, "outputs", "features", "phase4_dataset.pt")
MANIFEST = os.path.join(ROOT, "outputs", "casm_v2", "phase50_split_v2_manifest.csv")
CKPT = os.path.join(ROOT, "outputs", "checkpoints", "best_model.pth")
CLASSES = ["Cherry", "Kiwi", "Peach", "Pineapple", "Plum"]
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DOMAIN_LABELS = {"fruits360": "Studio (Fruits-360)", "real_quality": "Market bins",
                 "real_grade": "Packing conveyor", "real_severity": "Controlled lab"}

torch.set_grad_enabled(False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    p = str(p)
    for base in (ROOT + os.sep, ROOT.replace("\\", "/") + "/"):
        if p.startswith(base):
            return p[len(base):].replace("\\", "/")
    return p.replace("\\", "/")


def rebuild_ood_sample():
    """Replicates collect_ood_image_paths() from phase50_stage4_real_ood.py."""
    recs = []
    for cls in CLASSES:
        imgs = []
        for r, _, files in os.walk(os.path.join(OOD_DIR, cls)):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    imgs.append(os.path.join(r, f))
        random.seed(42)
        chosen = random.sample(imgs, min(1000, len(imgs)))
        recs += [(p, cls, len(imgs)) for p in chosen]
    return recs


def metrics(in_s, ood_s):
    in_s = np.asarray(in_s, dtype=np.float64)
    ood_s = np.asarray(ood_s, dtype=np.float64)
    y = np.concatenate([np.zeros(len(in_s)), np.ones(len(ood_s))])
    s = np.concatenate([in_s, ood_s])
    auroc = roc_auc_score(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    aupr_trap = auc(rec, prec)
    ap = average_precision_score(y, s)
    prec_i, rec_i, _ = precision_recall_curve(1 - y, -s)
    aupr_in = auc(rec_i, prec_i)
    fpr, tpr, thr = roc_curve(y, s)
    i95 = np.where(tpr >= 0.95)[0][0]
    fpr95 = fpr[i95]
    thr95 = thr[i95]
    # ID-positive convention: accept as ID if s < t; need >=95% ID accepted
    fpr_i, tpr_i, _ = roc_curve(1 - y, -s)
    j95 = np.where(tpr_i >= 0.95)[0][0]
    fpr95_id = fpr_i[j95]
    return {"auroc": auroc * 100, "aupr_ood_trapz": aupr_trap * 100, "ap_ood": ap * 100,
            "aupr_in_trapz": aupr_in * 100, "fpr95_oodtpr": fpr95 * 100, "fpr95_threshold": float(thr95),
            "fpr95_idtpr": fpr95_id * 100, "n_id": len(in_s), "n_ood": len(ood_s),
            "ood_prevalence_pct": 100 * len(ood_s) / (len(in_s) + len(ood_s))}


def bootstrap_auroc(in_s, ood_s, n_boot=500):
    """Identical resampling scheme to phase50_stage4_real_ood.bootstrap_ci (seed 42)."""
    rng = np.random.RandomState(42)
    a = []
    y = np.concatenate([np.zeros(len(in_s)), np.ones(len(ood_s))])
    for _ in range(n_boot):
        b_in = in_s[rng.choice(len(in_s), len(in_s), replace=True)]
        b_ood = ood_s[rng.choice(len(ood_s), len(ood_s), replace=True)]
        a.append(roc_auc_score(y, np.concatenate([b_in, b_ood])))
    return float(np.percentile(a, 2.5) * 100), float(np.percentile(a, 97.5) * 100)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = {}

    # ---------- 1. OOD sample identity ----------
    ood = torch.load(OOD_CACHE, map_location="cpu", weights_only=False)
    cached_paths = [str(p) for p in ood["paths"]]
    cached_cls = [str(c) for c in ood["classes"]]
    assert len(cached_paths) == 5000 and len(set(cached_paths)) == 5000, "OOD cache size/duplicate failure"
    cls_counts = pd.Series(cached_cls).value_counts().to_dict()
    assert all(cls_counts[c] == 1000 for c in CLASSES), cls_counts
    log["ood_cache_sha256"] = sha256(OOD_CACHE)
    log["ood_class_counts"] = cls_counts
    if os.path.isdir(OOD_DIR):
        rebuilt = rebuild_ood_sample()
        same = [os.path.normcase(a) for a, _, _ in rebuilt] == [os.path.normcase(b) for b in cached_paths]
        log["ood_sample_rebuilt_matches_cache"] = bool(same)
        log["ood_pool_sizes"] = {c: n for _, c, n in {(None, c, n) for _, c, n in rebuilt}}
        log["ood_files_exist"] = int(sum(os.path.exists(p) for p in cached_paths))
        assert same, "Rebuilt OOD sample differs from cached sample"
    else:
        log["ood_sample_rebuilt_matches_cache"] = "NOT_CHECKED (open-set-dataset not present)"

    # ---------- 2. ID alignment ----------
    pt = torch.load(ID_CACHE, map_location="cpu", weights_only=False)
    man = pd.read_csv(MANIFEST, low_memory=False)
    assert np.array_equal(man["image_path"].astype(str).values, np.asarray(pt["paths"]).astype(str)), "manifest/feature misalignment"
    assert np.array_equal(man["severity_grade"].values, pt["targets"].numpy())
    split = man["split"].values
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    vis = pt["vis_feats"].float()
    test_dom = man["domain"].values[te]
    log["n_id_train"] = int(len(tr))
    log["n_id_test"] = int(len(te))
    log["id_test_domain_counts"] = pd.Series(test_dom).value_counts().to_dict()
    log["id_test_paths_unique"] = bool(man["image_path"].iloc[te].is_unique)
    ood_set = set(os.path.normcase(p) for p in cached_paths)
    log["ood_id_path_overlap"] = int(sum(os.path.normcase(p) in ood_set for p in man["image_path"].astype(str)))
    log["id_species_in_test"] = sorted(man["fruit"].iloc[te].unique().tolist())

    # ---------- 3. Model head ----------
    model = build_convnext_model(num_classes=5, pretrained=False, device="cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    head = model.model.head
    log["checkpoint_sha256"] = sha256(CKPT)

    def logits_correct(f):
        return head.fc(head.norm(f[:, :, None, None]).flatten(1))

    def logits_defective(f):
        return head.fc(f)

    id_feats = vis[te]
    ood_feats = ood["features"].float()
    ood_logits_stored = ood["logits"].float()

    # Fidelity checks of the head reconstruction against stored full-forward outputs
    msp_stored_id = pt["unc_ctx"][te, 1].float()
    rec_id = torch.softmax(logits_correct(id_feats), 1).max(1).values
    def_id = torch.softmax(logits_defective(id_feats), 1).max(1).values
    log["fidelity"] = {
        "id_msp_correct_vs_stored_fullforward_maxabs": float((rec_id - msp_stored_id).abs().max()),
        "id_msp_defective_vs_stored_fullforward_maxabs": float((def_id - msp_stored_id).abs().max()),
        "id_msp_defective_vs_stored_fullforward_meanabs": float((def_id - msp_stored_id).abs().mean()),
        "ood_logits_correct_vs_stored_fullforward_maxabs": float((logits_correct(ood_feats) - ood_logits_stored).abs().max()),
        "ood_logits_defective_vs_stored_fullforward_maxabs": float((logits_defective(ood_feats) - ood_logits_stored).abs().max()),
    }

    # ---------- 4. Scores ----------
    trf, trt = vis[tr], torch.tensor(man["severity_grade"].values[tr])
    means, cent = [], []
    for k in range(5):
        mk = trf[trt == k].mean(0)
        means.append(mk)
        cent.append(trf[trt == k] - mk)
    means = torch.stack(means)
    cc = torch.cat(cent)
    cov = cc.T @ cc / (cc.shape[0] - 1) + 1e-4 * torch.eye(768)
    P = torch.linalg.inv(cov)

    def mahal(f):
        return torch.stack([((f - means[k]) @ P * (f - means[k])).sum(1) for k in range(5)], 1).min(1).values

    def energy(z):
        return -torch.logsumexp(z, 1)

    def negmsp(z):
        return -torch.softmax(z, 1).max(1).values

    z_id_c, z_ood_c = logits_correct(id_feats), logits_correct(ood_feats)
    z_id_d = logits_defective(id_feats)
    scores = {
        "corrected": {
            "Mahalanobis": (mahal(id_feats).numpy(), mahal(ood_feats).numpy()),
            "Energy": (energy(z_id_c).numpy(), energy(z_ood_c).numpy()),
            "MSP": (negmsp(z_id_c).numpy(), negmsp(z_ood_c).numpy()),
        },
        "historical_defective": {
            "Mahalanobis": (mahal(id_feats).numpy(), mahal(ood_feats).numpy()),
            "Energy": (energy(z_id_d).numpy(), energy(ood_logits_stored).numpy()),
            "MSP": (negmsp(z_id_d).numpy(), negmsp(ood_logits_stored).numpy()),
        },
        "sensitivity_stored_ood_logits": {
            "Energy": (energy(z_id_c).numpy(), energy(ood_logits_stored).numpy()),
            "MSP": (negmsp(z_id_c).numpy(), negmsp(ood_logits_stored).numpy()),
        },
    }

    rows, dom_rows, sp_rows = [], [], []
    for variant, dets in scores.items():
        for det, (si, so) in dets.items():
            m = metrics(si, so)
            if variant == "corrected":
                lo, hi = bootstrap_auroc(si.astype(np.float64), so.astype(np.float64))
                m["auroc_boot_ci_low"], m["auroc_boot_ci_high"] = lo, hi
            m.update({"variant": variant, "detector": det})
            rows.append(m)
            if variant in ("corrected", "historical_defective"):
                for d in ["fruits360", "real_quality", "real_grade", "real_severity"]:
                    msk = test_dom == d
                    md = metrics(si[msk], so)
                    md.update({"variant": variant, "detector": det, "id_domain": d,
                               "id_domain_label": DOMAIN_LABELS[d],
                               "mean_id_score": float(np.mean(si[msk])), "mean_ood_score": float(np.mean(so))})
                    dom_rows.append(md)
                for c in CLASSES:
                    msk = np.array(cached_cls) == c
                    mc = metrics(si, so[msk])
                    mc.update({"variant": variant, "detector": det, "ood_species": c})
                    sp_rows.append(mc)

    df = pd.DataFrame(rows)
    ddf = pd.DataFrame(dom_rows)
    sdf = pd.DataFrame(sp_rows)
    df.to_csv(os.path.join(OUT_DIR, "ood_pooled_metrics_all_variants.csv"), index=False)
    ddf.to_csv(os.path.join(OUT_DIR, "ood_domain_stratified_metrics.csv"), index=False)
    sdf.to_csv(os.path.join(OUT_DIR, "ood_per_species_metrics.csv"), index=False)

    # Per-sample authoritative score file (relative paths only)
    c = scores["corrected"]
    ps = pd.DataFrame({
        "image_path": [rel(p) for p in man["image_path"].iloc[te]] + [rel(p) for p in cached_paths],
        "is_ood": np.r_[np.zeros(len(te), int), np.ones(5000, int)],
        "group": list(test_dom) + cached_cls,
        "mahalanobis_score": np.r_[c["Mahalanobis"][0], c["Mahalanobis"][1]],
        "energy_score": np.r_[c["Energy"][0], c["Energy"][1]],
        "neg_msp_score": np.r_[c["MSP"][0], c["MSP"][1]],
    })
    ps_path = os.path.join(OUT_DIR, "ood_per_sample_scores_corrected.csv.gz")
    ps.to_csv(ps_path, index=False, float_format="%.6f", compression="gzip")
    # Self-check: recompute from the saved file
    re = pd.read_csv(ps_path)
    for col, det in [("mahalanobis_score", "Mahalanobis"), ("energy_score", "Energy"), ("neg_msp_score", "MSP")]:
        a = roc_auc_score(re["is_ood"], re[col]) * 100
        ref = df[(df.variant == "corrected") & (df.detector == det)]["auroc"].iloc[0]
        assert abs(a - ref) < 0.01, (det, a, ref)

    log["n_ood"] = 5000
    log["score_direction"] = "higher = more OOD for all detectors"
    log["positive_class"] = "y=1 is OOD"
    with open(os.path.join(OUT_DIR, "ood_verification_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    pd.set_option("display.width", 250)
    cols = ["variant", "detector", "auroc", "aupr_ood_trapz", "ap_ood", "aupr_in_trapz", "fpr95_oodtpr", "fpr95_idtpr"]
    print(df[cols].round(2).to_string(index=False))
    print()
    print(ddf[["variant", "detector", "id_domain", "n_id", "auroc"]].round(2).to_string(index=False))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
