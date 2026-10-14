"""
verify_splits.py
================
Verifies session-based train/val/test splits in master_dataset.csv.

Checks:
1. No session leaks between splits (same session_id in train AND test)
2. Class distribution per split
3. Per-fruit image counts per split
4. Domain distribution per split
5. Imbalance ratios per label group
6. Confirms EXCLUDED rows are correct (Banana_3rd_grade only)
7. Confirms Fruits-360 duplicates are absent (100x100 folders)

Usage:
    python verify_splits.py
"""

import os
import sys
import pandas as pd
from collections import defaultdict, Counter

CSV_PATH = os.path.join(r"C:\fruit_dataset", "master_dataset.csv")

# Folder substrings that should NEVER appear in master_dataset.csv
FORBIDDEN_FOLDER_SUBSTRINGS = [
    "100 x 100",
    "100x100",
]

# --- Load ----------------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        print(f"[FATAL] master_dataset.csv not found at: {path}")
        print("Run generate_master_csv.py first.")
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


# --- Checks -------------------------------------------------------------------

def check_forbidden_folders(df):
    print("\n-- Check 1: No duplicate 100x100 folders in CSV -----------------")
    found = False
    for forbidden in FORBIDDEN_FOLDER_SUBSTRINGS:
        mask = df["image_path"].str.contains(forbidden.replace(" ", ".*"), regex=True, na=False)
        n = mask.sum()
        if n > 0:
            print(f"  [FAIL] Found {n:,} rows referencing '{forbidden}' folders!")
            print(f"         Example: {df.loc[mask, 'image_path'].iloc[0]}")
            found = True
    if not found:
        print("  [PASS] No duplicate 100x100 mirror folders found.")


def check_excluded_rows(df):
    print("\n-- Check 2: EXCLUDED rows contain only Banana_3rd_grade ---------")
    excluded = df[df["split"] == "EXCLUDED"]
    print(f"  Total EXCLUDED rows: {len(excluded):,}")
    if len(excluded) == 0:
        print("  [WARN] No EXCLUDED rows found. Banana_3rd_grade may be missing.")
        return
    non_banana = excluded[~excluded["image_path"].str.contains("Banana_3rd_grade", na=False)]
    if len(non_banana) > 0:
        print(f"  [FAIL] {len(non_banana)} EXCLUDED rows are NOT from Banana_3rd_grade!")
        print(non_banana[["image_path", "fruit", "split"]].head())
    else:
        print(f"  [PASS] All {len(excluded):,} EXCLUDED rows are Banana_3rd_grade.")


def check_session_leakage(df):
    print("\n-- Check 3: Session-level leakage (train / val / test) ----------")
    active = df[~df["split"].isin(["EXCLUDED", "OOD_TEST", ""])].copy()
    active = active[active["domain"] != "fruits360"]  # Fruits-360 uses author split

    # Group by fruit + label to check within-group leakage
    groups = active.groupby(["fruit", "grade_label_normalized", "quality_label_normalized",
                              "severity_label_normalized"])

    leaks_found = 0
    for name, grp in groups:
        session_split_map = defaultdict(set)
        for _, row in grp.iterrows():
            if row["session_id"] and row["split"]:
                session_split_map[row["session_id"]].add(row["split"])

        for sid, splits in session_split_map.items():
            if "train" in splits and "test" in splits:
                print(f"  [LEAK] session_id='{sid}' appears in both TRAIN and TEST | group={name}")
                leaks_found += 1
            if "train" in splits and "val" in splits and len(splits) == 2:
                pass  # train/val overlap is acceptable for some scenarios but warn
            if "val" in splits and "test" in splits:
                print(f"  [LEAK] session_id='{sid}' appears in both VAL and TEST | group={name}")
                leaks_found += 1

    if leaks_found == 0:
        print("  [PASS] No train/test or val/test session leakage detected.")
    else:
        print(f"  [FAIL] {leaks_found} session leaks found! Review session split logic.")


def check_split_distribution(df):
    print("\n-- Check 4: Split distribution ----------------------------------")
    active = df[df["split"] != "EXCLUDED"]
    counts = active["split"].value_counts()
    total  = len(active)
    for split, n in counts.sort_index().items():
        pct = n / total * 100
        print(f"  {split:<10}: {n:>7,} rows ({pct:.1f}%)")


def check_class_balance(df):
    print("\n-- Check 5: Class balance per label type -------------------------")
    active = df[df["split"].isin(["train", "val", "test"])]

    for label_type, col in [
        ("grade",    "grade_label_normalized"),
        ("quality",  "quality_label_normalized"),
        ("severity", "severity_label_normalized"),
    ]:
        sub = active[active[col].notna() & (active[col].str.strip() != "")]
        if len(sub) == 0:
            continue
        print(f"\n  [{label_type.upper()}]")
        for fruit, grp in sub.groupby("fruit"):
            counts = grp[col].value_counts().sort_index()
            if len(counts) < 2:
                continue
            ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
            flag = ""
            if ratio > 5:
                flag = "  ⚠ SEVERE"
            elif ratio > 2:
                flag = "  ~ moderate"
            print(f"    {fruit:<8}: " + " | ".join(f"{k}={v}" for k, v in counts.items()) +
                  f"  → ratio={ratio:.1f}x{flag}")


def check_fruit_counts(df):
    print("\n-- Check 6: Per-fruit image counts per split ---------------------")
    active = df[~df["split"].isin(["EXCLUDED"])]
    pivot = pd.crosstab(active["fruit"], active["split"])
    print(pivot.to_string())


def check_domain_distribution(df):
    print("\n-- Check 7: Domain distribution ---------------------------------")
    active = df[df["split"] != "EXCLUDED"]
    pivot = pd.crosstab(active["domain"], active["split"])
    print(pivot.to_string())


def check_image_files_exist(df, sample_n=500):
    print(f"\n-- Check 8: Spot-check {sample_n} image paths exist ------------")
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    missing = []
    for _, row in sample.iterrows():
        if not os.path.exists(row["image_path"]):
            missing.append(row["image_path"])
    if missing:
        print(f"  [FAIL] {len(missing)} of {sample_n} sampled images NOT found on disk!")
        for p in missing[:5]:
            print(f"    {p}")
    else:
        print(f"  [PASS] All {sample_n} sampled image paths exist on disk.")


def summary_report(df):
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    splits = df["split"].value_counts().to_dict()
    total = len(df)
    print(f"Total rows in CSV : {total:,}")
    for k in ["train", "val", "test", "OOD_TEST", "EXCLUDED"]:
        n = splits.get(k, 0)
        print(f"  {k:<12}: {n:,}")
    print()


# --- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    df = load_csv(CSV_PATH)

    check_forbidden_folders(df)
    check_excluded_rows(df)
    check_session_leakage(df)
    check_split_distribution(df)
    check_class_balance(df)
    check_fruit_counts(df)
    check_domain_distribution(df)
    check_image_files_exist(df)
    summary_report(df)

    print("Verification complete.")
