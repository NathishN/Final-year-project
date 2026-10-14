import os
import json
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict

print("Reconstructing complete label distribution for Phase 4.5 Task 2...")

df = pd.read_csv("master_dataset.csv", low_memory=False)

# Severity mapping
with open("severity_mapping.json") as f:
    sev_map = json.load(f)

def get_severity(row):
    domain = row["domain"]
    if domain == "fruits360":
        return 0
    elif domain == "real_severity":
        lbl = str(row["severity_label_normalized"]).strip()
        return sev_map["mappings_by_domain"]["real_severity"].get(lbl, -1)
    elif domain == "real_quality":
        lbl = str(row["quality_label_normalized"]).strip()
        return sev_map["mappings_by_domain"]["real_quality"].get(lbl, -1)
    elif domain == "real_grade":
        lbl = str(row["grade_label_normalized"]).strip()
        return sev_map["mappings_by_domain"]["real_grade"].get(lbl, -1)
    return -1

df["severity_grade"] = df.apply(get_severity, axis=1)

splits = ["train", "val", "test", "EXCLUDED"]

integrity_data = {
    "overall": {
        "total_records": len(df),
        "split_counts": df["split"].value_counts().to_dict(),
        "excluded_count": int((df["split"] == "EXCLUDED").sum())
    },
    "splits": {}
}

# Audit each split
for sp in ["train", "val", "test"]:
    sub = df[df["split"] == sp].copy()
    
    # 1. Total images
    total_imgs = len(sub)
    
    # 2. Unique image paths
    unique_paths = sub["image_path"].nunique()
    
    # 3. File existence & content hashes (sample check or full if fast)
    # We check existence of all paths
    exists_count = sum(os.path.exists(p) for p in sub["image_path"])
    
    # 4. Samples per ordinal grade
    grade_counts = sub["severity_grade"].value_counts().to_dict()
    # ensure 0..4 all present in keys
    grade_counts_full = {int(k): int(grade_counts.get(k, 0)) for k in range(5)}
    
    # 5. Samples per species
    species_counts = {str(k): int(v) for k, v in sub["fruit"].value_counts().to_dict().items()}
    
    # 6. Samples per acquisition domain
    domain_counts = {str(k): int(v) for k, v in sub["domain"].value_counts().to_dict().items()}
    
    # 7. Joint counts for grade x species
    grade_x_species = {}
    ct_gs = pd.crosstab(sub["severity_grade"], sub["fruit"])
    for g in range(5):
        grade_x_species[f"grade_{g}"] = {}
        for spc in sub["fruit"].unique():
            val = int(ct_gs.loc[g, spc]) if (g in ct_gs.index and spc in ct_gs.columns) else 0
            grade_x_species[f"grade_{g}"][spc] = val
            
    # 8. Joint counts for grade x domain
    grade_x_domain = {}
    ct_gd = pd.crosstab(sub["severity_grade"], sub["domain"])
    for g in range(5):
        grade_x_domain[f"grade_{g}"] = {}
        for dom in sub["domain"].unique():
            val = int(ct_gd.loc[g, dom]) if (g in ct_gd.index and dom in ct_gd.columns) else 0
            grade_x_domain[f"grade_{g}"][dom] = val
            
    # 9. Invalid or unmapped labels
    unmapped_count = int((sub["severity_grade"] == -1).sum())
    
    integrity_data["splits"][sp] = {
        "total_images": total_imgs,
        "unique_paths": unique_paths,
        "files_existing_on_disk": exists_count,
        "grade_counts": grade_counts_full,
        "species_counts": species_counts,
        "domain_counts": domain_counts,
        "grade_x_species": grade_x_species,
        "grade_x_domain": grade_x_domain,
        "unmapped_labels_count": unmapped_count,
    }

# Split path leakage audit
train_paths = set(df[df["split"] == "train"]["image_path"])
val_paths = set(df[df["split"] == "val"]["image_path"])
test_paths = set(df[df["split"] == "test"]["image_path"])

integrity_data["path_leakage"] = {
    "train_val_overlap": len(train_paths & val_paths),
    "train_test_overlap": len(train_paths & test_paths),
    "val_test_overlap": len(val_paths & test_paths),
}

# Image hash duplicate audit across splits
print("Hashing images to check content duplicates across splits...")
# To be fast and thorough, hash 1000 images per split or full check
# Let's compute md5 hashes for real-world images (n ~ 12,000) and sample of fruits360
real_df = df[df["domain"] != "fruits360"].copy()
print(f"Total non-fruits360 images to hash: {len(real_df)}")

md5_map = {} # hash -> list of (split, path)
hash_dups = defaultdict(list)
for idx, r in real_df.iterrows():
    p = r["image_path"]
    sp = r["split"]
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            md5_map[p] = (sp, h)
            hash_dups[h].append((sp, p))
        except Exception as e:
            pass

cross_split_content_dups = []
for h, entries in hash_dups.items():
    sps = set(e[0] for e in entries)
    if len(sps) > 1 and "EXCLUDED" not in sps:
        cross_split_content_dups.append({"hash": h, "splits": list(sps), "paths": [e[1] for e in entries]})

integrity_data["content_hash_audit"] = {
    "non_fruits360_hashed": len(md5_map),
    "unique_hashes": len(hash_dups),
    "cross_split_content_duplicates_count": len(cross_split_content_dups),
    "cross_split_duplicates_details": cross_split_content_dups[:10]
}

# Metadata vs folder name conflict check
conflicts = []
for idx, r in df.iterrows():
    p = str(r["image_path"])
    fruit = str(r["fruit"])
    dom = str(r["domain"])
    # check if fruit name is in parent path
    if fruit not in p:
        conflicts.append({"path": p, "fruit": fruit, "reason": "Fruit name not in path"})

integrity_data["metadata_conflicts"] = {
    "total_conflicts": len(conflicts),
    "details": conflicts[:10]
}

out_json = "outputs/casm_v2/phase45_dataset_integrity.json"
with open(out_json, "w") as f:
    json.dump(integrity_data, f, indent=2)

print(f"Saved {out_json}")
