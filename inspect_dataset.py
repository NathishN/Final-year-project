import torch, numpy as np

d = torch.load("outputs/features/phase4_dataset.pt", map_location="cpu", weights_only=False)
splits  = d["splits"]
targets = d["targets"].numpy()
species = d["species_ids"].numpy()

print("=== DATASET PROVENANCE AUDIT ===")
print(f"Total samples: {len(targets)}")
print(f"Keys: {list(d.keys())}")
print()

# Split identity hashes (for reproducibility verification)
for sp in ["train","val","test"]:
    m   = splits == sp
    idx = np.where(m)[0]
    print(f"{sp}: N={int(m.sum())}  idx_range=[{idx.min()},{idx.max()}]  idx_hash={hash(idx.tobytes()) & 0xFFFFFFFF}")

print()
# Grade distribution per split
GRADES = ["Healthy","Mild","Moderate","Severe","Rotten"]
for sp in ["train","val","test"]:
    m = splits == sp
    c = np.bincount(targets[m], minlength=5)
    print(f"{sp} grades: {dict(zip(GRADES,c.tolist()))}")

print()
# Test split: grade 2 (Moderate) check
m_test = splits=="test"
c_test = np.bincount(targets[m_test], minlength=5)
print(f"CRITICAL: test grade2 (Moderate Defect) count = {c_test[2]}")
print(f"  => This means grade=2 is ABSENT from test set.")
print(f"  => QWK/Macro-F1 for grade 2 is undefined in test evaluation.")

print()
# Species per split
SPECIES = {0:"Apple",1:"Banana",2:"Mango",3:"Orange",4:"Tomato"}
for sp in ["train","val","test"]:
    m  = splits == sp
    sc = np.bincount(species[m], minlength=5)
    print(f"{sp} species: { {SPECIES[i]: int(sc[i]) for i in range(5)} }")

print()
# Feature tensor stats
for k in ["vis_feats","grade_ctx","ood_ctx","unc_ctx","species_ids"]:
    v = d[k]
    has_nan = bool(v.isnan().any()) if v.dtype == torch.float32 else "N/A"
    print(f"{k}: shape={tuple(v.shape)}  dtype={v.dtype}  min={float(v.min()):.4f}  max={float(v.max()):.4f}  nan={has_nan}")

print()
# OOD context feature meaning (2D vector)
print("OOD context breakdown (ood_ctx dim=0 and dim=1):")
ood = d["ood_ctx"]
print(f"  dim0 (Mahalanobis dist): mean={float(ood[:,0].mean()):.4f}  std={float(ood[:,0].std()):.4f}")
print(f"  dim1 (OOD prob):         mean={float(ood[:,1].mean()):.4f}  std={float(ood[:,1].std()):.4f}")
# Correlation between OOD-prob and unc-maxconf
unc = d["unc_ctx"]
corr = float(torch.corrcoef(torch.stack([ood[:,1], unc[:,1]]))[0,1])
print(f"  Pearson R(OOD-prob, Unc-MaxConf): {corr:.4f}")

print()
# Grade context (4D): P(G=1), P(G=2), P(G=3), MaxConf
gc = d["grade_ctx"]
print(f"Grade ctx dims (should be 4): {tuple(gc.shape)}")
print(f"  dim0 P(grade=1): mean={float(gc[:,0].mean()):.4f}")
print(f"  dim1 P(grade=2): mean={float(gc[:,1].mean()):.4f}")
print(f"  dim2 P(grade=3): mean={float(gc[:,2].mean()):.4f}")
print(f"  dim3 MaxConf:    mean={float(gc[:,3].mean()):.4f}")
