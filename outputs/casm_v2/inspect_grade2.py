import pandas as pd
import json

df = pd.read_csv("master_dataset.csv")
print("Total rows in master_dataset.csv:", len(df))
print("Split value counts:")
print(df["split"].value_counts(dropna=False))

# Load severity mapping
with open("severity_mapping.json") as f:
    sev_map = json.load(f)

# Inspect how labels map to severity
def get_severity(row):
    domain = row["domain"]
    if domain == "fruits360":
        return 0
    elif domain == "real_severity":
        lbl = row["severity_label_normalized"]
        return sev_map["mappings_by_domain"]["real_severity"].get(lbl, -1)
    elif domain == "real_quality":
        lbl = row["quality_label_normalized"]
        return sev_map["mappings_by_domain"]["real_quality"].get(lbl, -1)
    elif domain == "real_grade":
        lbl = row["grade_label_normalized"]
        return sev_map["mappings_by_domain"]["real_grade"].get(lbl, -1)
    return -1

df["severity_grade"] = df.apply(get_severity, axis=1)

print("\nSeverity Grade counts across entire dataset:")
print(df["severity_grade"].value_counts().sort_index())

print("\nSeverity Grade by Split:")
ct = pd.crosstab(df["severity_grade"], df["split"], margins=True)
print(ct)

# Deep dive into Grade 2
g2 = df[df["severity_grade"] == 2]
print(f"\n--- Grade 2 Deep Dive (n={len(g2)}) ---")
print("Domains:", g2["domain"].value_counts().to_dict())
print("Fruits:", g2["fruit"].value_counts().to_dict())
print("Quality labels:", g2["quality_label_normalized"].value_counts().to_dict())
print("Splits:", g2["split"].value_counts().to_dict())
print("Session IDs in Grade 2:")
print(g2.groupby(["fruit", "session_id", "split"]).size())

# Check all real_quality records
rq = df[df["domain"] == "real_quality"]
print(f"\n--- Real Quality Domain Deep Dive (n={len(rq)}) ---")
print(pd.crosstab([rq["fruit"], rq["quality_label_normalized"]], rq["split"]))
print("\nUnique sessions per fruit in real_quality:")
for fruit in rq["fruit"].unique():
    frq = rq[rq["fruit"] == fruit]
    print(f"Fruit: {fruit}")
    for sid, group in frq.groupby("session_id"):
        print(f"  Session {sid}: count={len(group)}, labels={group['quality_label_normalized'].value_counts().to_dict()}, split={group['split'].iloc[0]}")
