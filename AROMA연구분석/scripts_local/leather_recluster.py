# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

CSV = "D:/project/AROMA_DATASET/profiling/profilng_leather/morphology_features.csv"
MORPH = ["linearity","solidity","extent","aspect_ratio","eccentricity","circularity"]

df = pd.read_csv(CSV, encoding="utf-8")
print("rows:", len(df), "unique mask paths:", df["defect_mask_path"].nunique())

X = df[MORPH].to_numpy(dtype=float)
# min-max normalize per feature, replicating profiler
Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-6)

best = None
for k in range(1, 6):
    gm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    gm.fit(Xn)
    bic = gm.bic(Xn)
    if best is None or bic < best[1]:
        best = (k, bic, gm)
    print("k=%d BIC=%.4f" % (k, bic))

k, bic, gm = best
labels = gm.predict(Xn)
print("\nSELECTED n_clusters:", k, "BIC=%.4f" % bic)

defect_type = df["defect_type"].to_numpy()
ari = adjusted_rand_score(defect_type, labels)
nmi = normalized_mutual_info_score(defect_type, labels)
print("ARI: %.4f" % ari)
print("NMI: %.4f" % nmi)

ct = pd.crosstab(pd.Series(defect_type, name="defect_type"),
                 pd.Series(labels, name="cluster"))
print("\nCROSSTAB:")
print(ct.to_string())

# per-instance assignment keyed by defect_mask_path
assign = {p: int(l) for p, l in zip(df["defect_mask_path"], labels)}
print("\nassignments:", len(assign))
