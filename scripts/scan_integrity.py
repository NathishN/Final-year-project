"""
scripts/scan_integrity.py
Scans outputs/final_submission and paper for:
- local file paths (file:///C:)
- stale OOD metrics (54.06, 61.27)
- unsubstantiated superlative claims
"""

import os

target_dirs = ["outputs/final_submission", "paper"]
patterns = ["file:///C:", "54.06", "61.27", "state-of-the-art", "commercially ready", "turnkey"]

found = []
for d in target_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith((".md", ".csv", ".json", ".txt", ".bib")):
                p = os.path.join(root, f)
                with open(p, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                    for pat in patterns:
                        if pat.lower() in content.lower():
                            found.append((p, pat))

if found:
    print(f"Found {len(found)} matches:")
    for p, pat in found:
        print(f"  {p}: {pat}")
else:
    print("Zero forbidden patterns or stale metrics found in target directories.")
