"""Replicate the LOGO-CV result the PhageHostLearn paper reports
(Fig. 3 / Supplementary Fig. S5 — leave-one-K-locus-out CV with the
ESM-2 + XGBoost configuration).

We skip the preprocessing pipeline entirely and use the pre-computed
ESM-2 embeddings + interaction matrix directly out of Zenodo 11061100.
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPO = Path("/Users/leannmlindsey/WORK/CLAUDE_PHAGEHOSTLEARN/claude_copy/PhageHostLearn")
DATA = REPO / "data" / "zenodo_11061100" / "11061100_unpacked"
sys.path.insert(0, str(REPO / "code"))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve

# Stub out fair-esm before importing phagehostlearn_features — that
# module pulls fair-esm at module load, but we only need its
# construct_feature_matrices function (no embedding extraction).
import types as _types
sys.modules.setdefault("esm", _types.ModuleType("esm"))

# The Zenodo file is just `phage_host_interactions.csv` (no suffix); the
# upstream `construct_feature_matrices` function expects `phage_host_interactions{suffix}.csv`.
# Easiest fix: pass suffix=''. Confirm the file exists.
ICSV = DATA / "phage_host_interactions.csv"
LOCI_EMB = DATA / "esm2_embeddings_loci.csv"
RBP_EMB = DATA / "esm2_embeddings_rbp.csv"
assert ICSV.exists(), f"missing: {ICSV}"
assert LOCI_EMB.exists() and RBP_EMB.exists()

# Vectorized re-implementation of phagehostlearn_features.construct_feature_matrices.
# The published version uses Python double-loop + pd.concat per pair → minutes/hours
# of wall time and ~7 GB RAM. Logic is identical: for each (host, phage) where the
# interaction is known, concatenate the loci embedding with the mean-of-RBPs phage
# embedding. Labels and groups are 0-based indices into the loci_embeddings / phages
# arrays in iteration order, matching the upstream function.
print(f"Loading features from {DATA}", flush=True)
RBP_emb_df = pd.read_csv(RBP_EMB)
loci_emb_df = pd.read_csv(LOCI_EMB)
interactions_df = pd.read_csv(ICSV, index_col=0)

# Mean-pool RBP embeddings per phage_ID (preserve set-order to match upstream).
phage_ids_unique = list(set(RBP_emb_df["phage_ID"]))
rbp_emb_cols = RBP_emb_df.columns[2:]
multiRBP = (
    RBP_emb_df.groupby("phage_ID")[rbp_emb_cols].mean().loc[phage_ids_unique]
)
phage_id_list = list(multiRBP.index)
phage_emb = multiRBP.values  # (n_phages, 1280)

loci_acc = list(loci_emb_df["accession"])
loci_emb_cols = loci_emb_df.columns[1:]
loci_emb = loci_emb_df[loci_emb_cols].values  # (n_loci, 1280)

# Build the (i, j) index pairs where interaction is known.
inter = interactions_df.loc[loci_acc, phage_id_list].values  # (n_loci, n_phages)
known_mask = ~np.isnan(inter)
loci_idx, phage_idx = np.where(known_mask)
labels = inter[loci_idx, phage_idx].astype(int)

# Concatenate loci_emb[i] || phage_emb[j] for each pair — vectorized.
features_esm2 = np.hstack([loci_emb[loci_idx], phage_emb[phage_idx]])
groups_loci = loci_idx
groups_phage = phage_idx
print(f"  vectorized feature build done in <1 sec", flush=True)
print(f"  features:    {features_esm2.shape}")
print(f"  labels:      pos={int(labels.sum())}, neg={int((labels==0).sum())}, total={len(labels)}")
print(f"  unique K-loci groups: {len(set(groups_loci))}")
print(f"  unique phage groups:  {len(set(groups_phage))}")

print("\nRunning LOGO-CV by K-locus...", flush=True)
logo = LeaveOneGroupOut()
scores_all = []
labels_all = []
n_splits = len(set(groups_loci))
for i, (tr, te) in enumerate(logo.split(features_esm2, labels, groups_loci), 1):
    held_out = sorted(set(groups_loci[te]))
    Xtr, Xte = features_esm2[tr], features_esm2[te]
    ytr, yte = labels[tr], labels[te]
    pos = int(ytr.sum()); neg = len(ytr) - pos
    if pos == 0 or neg == 0:
        print(f"  [{i}/{n_splits}] held-out {held_out}  -- skipped (no signal in train)")
        continue
    spw = neg / pos
    xgb = XGBClassifier(
        scale_pos_weight=spw, learning_rate=0.3, n_estimators=250, max_depth=7,
        n_jobs=4, eval_metric='logloss',
    )
    xgb.fit(Xtr, ytr)
    yhat = xgb.predict_proba(Xte)[:, 1]
    scores_all.append(yhat)
    labels_all.append(yte)
    fold_auc = roc_auc_score(yte, yhat) if len(set(yte)) > 1 else float('nan')
    print(f"  [{i}/{n_splits}] held-out {held_out}  n_test={len(te)}  pos_rate={yte.mean():.3f}  fold_AUC={fold_auc:.3f}", flush=True)

scores_all = np.concatenate(scores_all)
labels_all = np.concatenate(labels_all)

# Pooled ROC-AUC (this is the paper's headline metric in Fig. 3)
fpr, tpr, _ = roc_curve(labels_all, scores_all)
pooled_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(labels_all, scores_all)
pr_auc = auc(rec, prec)

print(f"\n=== POOLED LOGO-CV results ===")
print(f"  total predictions: {len(scores_all)}")
print(f"  positive rate:     {labels_all.mean():.4f}")
print(f"  pooled ROC AUC:    {pooled_auc:.4f}")
print(f"  pooled PR  AUC:    {pr_auc:.4f}")

# Save predictions for downstream use
out = pd.DataFrame({
    "score": scores_all,
    "label": labels_all,
})
out_path = REPO / "data" / "zenodo_11061100" / "logocv_predictions.csv"
out.to_csv(out_path, index=False)
print(f"\nPer-pair predictions -> {out_path}")
