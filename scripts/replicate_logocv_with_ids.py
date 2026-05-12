"""LOGO-CV by K-locus (PhageHostLearn's reported evaluation) — but save
host_id + phage_id with each prediction so HR@k can be computed.

LOGO-CV grouping: hosts are grouped by their loci_idx (one group per
unique host loci embedding — equivalent to "leave one K-locus out").
For each fold:
  - Train XGBoost on all pairs NOT in this held-out host group
  - Predict on the held-out pairs
  - Save (host_id, phage_id, score, label)

Then build a (host x phage) score matrix from the pooled predictions
and compute phage-anyhit + host-anyhit HR@k. This is the apples-to-
apples comparison with TropiSEQ / TropiGAT / ciPHer.
"""
import sys, types
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
sys.modules.setdefault("esm", types.ModuleType("esm"))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut

REPO = Path("/Users/leannmlindsey/WORK/CLAUDE_PHAGEHOSTLEARN/claude_copy/PhageHostLearn")
DATA = REPO / "data" / "zenodo_11061100" / "11061100_unpacked"
OUT  = REPO / "data" / "cipher_eval" / "PHL" / "phagehostlearn_logocv"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading embeddings + interaction matrix", flush=True)
rbp = pd.read_csv(DATA / "esm2_embeddings_rbp.csv")
loci = pd.read_csv(DATA / "esm2_embeddings_loci.csv")
inter = pd.read_csv(DATA / "phage_host_interactions.csv", index_col=0)

phage_ids = list(set(rbp["phage_ID"]))
rbp_cols = rbp.columns[2:]
phage_mean = rbp.groupby("phage_ID")[rbp_cols].mean().loc[phage_ids]
phage_id_list = list(phage_mean.index)
phage_emb = phage_mean.values

loci_acc = list(loci["accession"])
loci_emb = loci[loci.columns[1:]].values

interactions = inter.loc[loci_acc, phage_id_list].values
known = ~np.isnan(interactions)
loci_idx, phage_idx = np.where(known)
labels = interactions[loci_idx, phage_idx].astype(int)
features = np.hstack([loci_emb[loci_idx], phage_emb[phage_idx]])
print(f"  features: {features.shape}, positives: {labels.sum()}, negatives: {(labels==0).sum()}")

print("\n[2] Running LOGO-CV by host-K-locus (one host per group)", flush=True)
logo = LeaveOneGroupOut()
n_splits = len(set(loci_idx))
rows = []  # (host_id, phage_id, score, label)
for fi, (tr, te) in enumerate(logo.split(features, labels, loci_idx), 1):
    Xtr, Xte = features[tr], features[te]
    ytr, yte = labels[tr], labels[te]
    pos, neg = int(ytr.sum()), int((ytr==0).sum())
    if pos == 0 or neg == 0:
        for k in te:
            rows.append((loci_acc[loci_idx[k]], phage_id_list[phage_idx[k]], np.nan, int(labels[k])))
        continue
    xgb = XGBClassifier(scale_pos_weight=neg/pos, learning_rate=0.3,
                        n_estimators=250, max_depth=7, n_jobs=4, eval_metric='logloss')
    xgb.fit(Xtr, ytr)
    yhat = xgb.predict_proba(Xte)[:, 1]
    for k_local, k_global in enumerate(te):
        rows.append((loci_acc[loci_idx[k_global]], phage_id_list[phage_idx[k_global]],
                     float(yhat[k_local]), int(labels[k_global])))
    if fi % 20 == 0 or fi == n_splits:
        print(f"  fold {fi}/{n_splits} done", flush=True)

df = pd.DataFrame(rows, columns=["host_id","phage_id","score","label"])
df.to_csv(OUT / "logocv_predictions.csv", index=False)
print(f"\n  -> {OUT / 'logocv_predictions.csv'}  ({len(df)} pairs)")

# Build score matrix (host x phage). Missing pairs (unknown in source matrix) → NaN.
print("\n[3] Building LOGO-CV-derived score matrix and saving", flush=True)
score_mat = df.pivot(index="host_id", columns="phage_id", values="score")
score_mat.to_csv(OUT / "prediction_scores.csv")
print(f"  -> {OUT / 'prediction_scores.csv'}  shape {score_mat.shape}")
print("Done.")
