"""
generate_master_csv.py
======================
Generates master_dataset.csv for:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading"

Strategy:
- Scans all real-world grade/quality/severity folders
- Scans all Fruits-360 TRAIN360/TEST360 folders (NOT 100x100 duplicates)
- Parses acquisition date from filename for session assignment
- Applies session-stratified GroupShuffleSplit (70/15/15)
- Normalises all label names
- Marks Banana_3rd_grade as EXCLUDED
- Marks is_ood=False for known classes (OOD must be added separately)

Usage:
    python generate_master_csv.py

Output:
    master_dataset.csv
"""

import os
import re
import csv
import random
from collections import defaultdict

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT = r"C:\fruit_dataset"
OUTPUT_CSV = os.path.join(ROOT, "master_dataset.csv")
RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ─── Fruits-360 canonical folders (use ONLY these — not 100x100 mirrors) ───────

FRUITS360_TRAIN = {
    "Apple":  os.path.join(ROOT, "Apple",  "Apple_TRAIN360"),
    "Banana": os.path.join(ROOT, "Banana", "Banana_Train"),
    "Mango":  os.path.join(ROOT, "Mango",  "Mango_Train_360"),
    "Orange": os.path.join(ROOT, "Orange", "Orange_Train360"),
    "Tomato": os.path.join(ROOT, "Tomato", "Tomato_Train360"),
}

FRUITS360_TEST = {
    "Apple":  os.path.join(ROOT, "Apple",  "Apple_TEST360"),
    "Banana": os.path.join(ROOT, "Banana", "Banana_Test"),
    "Mango":  os.path.join(ROOT, "Mango",  "Mango_Test_360"),
    "Orange": os.path.join(ROOT, "Orange", "Orange_Test360"),
    "Tomato": os.path.join(ROOT, "Tomato", "Tomato_Test360"),
}

# ─── Real-world folder definitions ─────────────────────────────────────────────
# Format: (folder_path, label_type, label_original, label_normalized, domain)

REALWORLD_FOLDERS = [
    # ── Apple ──
    ("Apple/Apple_1st_grade",    "grade",    "1st_grade",    "grade_1",  "real_grade"),
    ("Apple/Apple_2nd_grade",    "grade",    "2nd_grade",    "grade_2",  "real_grade"),
    ("Apple/Apple_3rd_grade",    "grade",    "3rd_grade",    "grade_3",  "real_grade"),
    ("Apple/Apple_Good_Quality", "quality",  "Good_Quality", "good",     "real_quality"),
    ("Apple/Apple_Mixed_Quality","quality",  "Mixed_Quality","mixed",    "real_quality"),
    ("Apple/Apple_Bad_Quality",  "quality",  "Bad_Quality",  "bad",      "real_quality"),

    # ── Banana ──
    ("Banana/Banana_1st_grade",    "grade",    "1st_grade",    "grade_1",  "real_grade"),
    ("Banana/Banana_2nd_grade",    "grade",    "2nd_grade",    "grade_2",  "real_grade"),
    ("Banana/Banana_3rd_grade",    "grade",    "3rd_grade",    "grade_3",  "real_grade"),   # will be EXCLUDED
    ("Banana/Banana_Good_Quality", "quality",  "Good_Quality", "good",     "real_quality"),
    ("Banana/Banana_Mixed_Quality","quality",  "Mixed_Quality","mixed",    "real_quality"),
    ("Banana/Banana_Bad_Quality",  "quality",  "Bad_Quality",  "bad",      "real_quality"),
    ("Banana/Banana_Good",         "severity", "Good",         "fresh",    "real_severity"),
    ("Banana/Banana_Mild",         "severity", "Mild",         "mild",     "real_severity"),
    ("Banana/Banana_Rotten",       "severity", "Rotten",       "rotten",   "real_severity"),

    # ── Mango ──
    ("Mango/Mango_1st_grade", "grade", "1st_grade", "grade_1", "real_grade"),
    ("Mango/Mango_2nd_grade", "grade", "2nd_grade", "grade_2", "real_grade"),
    ("Mango/Mango_3rd_grade", "grade", "3rd_grade", "grade_3", "real_grade"),

    # ── Orange ──
    ("Orange/Orange_Good_quality", "quality", "Good_quality", "good",  "real_quality"),
    ("Orange/Orange_Mixed Quality","quality",  "Mixed_Quality","mixed", "real_quality"),
    ("Orange/Orange_Bad_Quality",  "quality",  "Bad_Quality",  "bad",   "real_quality"),

    # ── Tomato ──
    ("Tomato/Tomato_1st_grade", "grade",    "1st_grade", "grade_1", "real_grade"),
    ("Tomato/Tomato_2nd_grade", "grade",    "2nd_grade", "grade_2", "real_grade"),
    ("Tomato/Tomato_3rd_grade", "grade",    "3rd_grade", "grade_3", "real_grade"),
    ("Tomato/Tomato_Good",      "severity", "Good",      "fresh",   "real_severity"),
    ("Tomato/Tomato_Mild",      "severity", "Mild",      "mild",    "real_severity"),
    ("Tomato/Tomato_Rotten",    "severity", "Rotten",    "rotten",  "real_severity"),
]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def parse_session_id(filename):
    """
    Extract a date-based session ID from filename.
    Handles patterns:
      IMG_20230930_*.jpg  -> 20230930
      IMG20200728*.jpg    -> 20200728
      20190809_*.jpg      -> 20190809
      Image*.png          -> sequential_png (no date)
    """
    name = os.path.basename(filename)
    m = re.search(r'(\d{8})', name)
    if m:
        return m.group(1)
    # PNG severity images — treat each as its own session (no date available)
    return f"png_{name[:6]}"


def list_images(folder_path):
    """Recursively list all images in a folder."""
    images = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(root, f))
    return images


def session_stratified_split(records, seed=RANDOM_SEED):
    """
    Session-level split: all images sharing a session_id+fruit+domain stay together.
    
    Key insight: The split is determined at (fruit, domain) level — NOT per label.
    A session like '20230419' that contains Apple grade_2 AND grade_3 images must
    go entirely to ONE split. We assign the session at (fruit, domain) granularity
    so that the same date-session never appears in both train and test.
    """
    rng = random.Random(seed)

    # Step 1: For each (fruit, domain), collect all unique session_ids
    fruit_domain_sessions = defaultdict(set)
    for rec in records:
        key = (rec["fruit"], rec["domain"])
        fruit_domain_sessions[key].add(rec["session_id"])

    # Step 2: Assign each session to a split at (fruit, domain) level
    session_split_map = {}  # session_id -> split
    for key, session_ids in fruit_domain_sessions.items():
        sid_list = sorted(session_ids)
        rng.shuffle(sid_list)
        n = len(sid_list)
        n_train = max(1, int(n * SPLIT_RATIOS[0]))
        n_val   = max(1, int(n * SPLIT_RATIOS[1]))

        if n == 1:
            for s in sid_list:
                session_split_map[s] = "train"
        elif n == 2:
            session_split_map[sid_list[0]] = "train"
            session_split_map[sid_list[1]] = "val"
        else:
            for s in sid_list[:n_train]:
                session_split_map[s] = "train"
            for s in sid_list[n_train:n_train + n_val]:
                session_split_map[s] = "val"
            for s in sid_list[n_train + n_val:]:
                session_split_map[s] = "test"

    # Step 3: Apply session->split to all records
    result = []
    for rec in records:
        rec["split"] = session_split_map.get(rec["session_id"], "train")
        result.append(rec)

    return result


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    records = []

    # ── 1. Real-world folders ──────────────────────────────────────────────────
    print("Scanning real-world folders...")
    for rel_folder, label_type, label_orig, label_norm, domain in REALWORLD_FOLDERS:
        folder_path = os.path.join(ROOT, rel_folder.replace("/", os.sep))
        fruit = rel_folder.split("/")[0]  # e.g. "Apple"
        folder_name = rel_folder.split("/")[1]

        if not os.path.isdir(folder_path):
            print(f"  [WARN] Not found: {folder_path}")
            continue

        images = list_images(folder_path)

        # EXCLUSION: Banana_3rd_grade (only 2 images)
        is_excluded = (fruit == "Banana" and "3rd_grade" in folder_name)

        for img_path in images:
            session_id = parse_session_id(img_path)
            img_format = os.path.splitext(img_path)[1].lower()

            rec = {
                "image_path":              img_path,
                "source_dataset":          domain,
                "fruit":                   fruit,
                "variety":                 "",
                "label_type":              label_type,
                "quality_label_original":  label_orig  if label_type == "quality"  else "",
                "quality_label_normalized":label_norm  if label_type == "quality"  else "",
                "grade_label_original":    label_orig  if label_type == "grade"    else "",
                "grade_label_normalized":  label_norm  if label_type == "grade"    else "",
                "severity_label_original": label_orig  if label_type == "severity" else "",
                "severity_label_normalized":label_norm if label_type == "severity" else "",
                "split":                   "EXCLUDED" if is_excluded else "",
                "is_known":                True,
                "is_ood":                  False,
                "domain":                  domain,
                "image_format":            img_format,
                "session_id":              session_id,
                "label_normalized":        label_norm,
                "notes":                   "Banana_3rd_grade excluded: n=2 insufficient for learning" if is_excluded else "",
            }
            records.append(rec)

        status = f"EXCLUDED ({len(images)} imgs)" if is_excluded else f"{len(images)} images"
        print(f"  {fruit}/{folder_name}: {status}")

    # ── Apply session-stratified split to non-excluded records ─────────────────
    print("\nApplying session-stratified split...")
    non_excluded = [r for r in records if r["split"] != "EXCLUDED"]
    split_records = session_stratified_split(non_excluded)
    # Merge back
    excluded_records = [r for r in records if r["split"] == "EXCLUDED"]
    records = split_records + excluded_records

    # ── 2. Fruits-360 TRAIN folders ────────────────────────────────────────────
    print("\nScanning Fruits-360 TRAIN splits...")
    for fruit, folder_path in FRUITS360_TRAIN.items():
        if not os.path.isdir(folder_path):
            print(f"  [WARN] Not found: {folder_path}")
            continue
        # Each subfolder = a variety
        for variety in sorted(os.listdir(folder_path)):
            variety_path = os.path.join(folder_path, variety)
            if not os.path.isdir(variety_path):
                continue
            images = list_images(variety_path)
            for img_path in images:
                img_format = os.path.splitext(img_path)[1].lower()
                records.append({
                    "image_path":               img_path,
                    "source_dataset":           "fruits360",
                    "fruit":                    fruit,
                    "variety":                  variety,
                    "label_type":               "variety",
                    "quality_label_original":   "",
                    "quality_label_normalized": "",
                    "grade_label_original":     "",
                    "grade_label_normalized":   "",
                    "severity_label_original":  "",
                    "severity_label_normalized":"",
                    "split":                    "train",
                    "is_known":                 True,
                    "is_ood":                   False,
                    "domain":                   "fruits360",
                    "image_format":             img_format,
                    "session_id":               f"f360_train_{fruit}",
                    "label_normalized":         fruit.lower(),
                    "notes":                    "Fruits-360 original author train split",
                })
        print(f"  {fruit} TRAIN: scanned")

    # ── 3. Fruits-360 TEST folders ─────────────────────────────────────────────
    print("Scanning Fruits-360 TEST splits...")
    for fruit, folder_path in FRUITS360_TEST.items():
        if not os.path.isdir(folder_path):
            print(f"  [WARN] Not found: {folder_path}")
            continue
        for variety in sorted(os.listdir(folder_path)):
            variety_path = os.path.join(folder_path, variety)
            if not os.path.isdir(variety_path):
                continue
            images = list_images(variety_path)
            for img_path in images:
                img_format = os.path.splitext(img_path)[1].lower()
                records.append({
                    "image_path":               img_path,
                    "source_dataset":           "fruits360",
                    "fruit":                    fruit,
                    "variety":                  variety,
                    "label_type":               "variety",
                    "quality_label_original":   "",
                    "quality_label_normalized": "",
                    "grade_label_original":     "",
                    "grade_label_normalized":   "",
                    "severity_label_original":  "",
                    "severity_label_normalized":"",
                    "split":                    "test",
                    "is_known":                 True,
                    "is_ood":                   False,
                    "domain":                   "fruits360",
                    "image_format":             img_format,
                    "session_id":               f"f360_test_{fruit}",
                    "label_normalized":         fruit.lower(),
                    "notes":                    "Fruits-360 original author test split",
                })
        print(f"  {fruit} TEST: scanned")

    # ── 4. Write CSV ───────────────────────────────────────────────────────────
    fieldnames = [
        "image_path", "source_dataset", "fruit", "variety",
        "label_type",
        "quality_label_original", "quality_label_normalized",
        "grade_label_original",   "grade_label_normalized",
        "severity_label_original","severity_label_normalized",
        "split", "is_known", "is_ood", "domain",
        "image_format", "session_id", "notes",
    ]

    print(f"\nWriting {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # ── 5. Summary report ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MASTER CSV SUMMARY")
    print("="*60)

    from collections import Counter
    split_counts   = Counter(r["split"]   for r in records)
    domain_counts  = Counter(r["domain"]  for r in records)
    fruit_counts   = Counter(r["fruit"]   for r in records)
    excluded_count = sum(1 for r in records if r["split"] == "EXCLUDED")

    print(f"\nTotal records : {len(records):,}")
    print(f"  EXCLUDED    : {excluded_count:,}  (Banana_3rd_grade)")

    print("\nBy split:")
    for k, v in sorted(split_counts.items()):
        print(f"  {k:<10}: {v:>6,}")

    print("\nBy domain:")
    for k, v in sorted(domain_counts.items()):
        print(f"  {k:<16}: {v:>6,}")

    print("\nBy fruit:")
    for k, v in sorted(fruit_counts.items()):
        print(f"  {k:<10}: {v:>6,}")

    print(f"\nOutput: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
