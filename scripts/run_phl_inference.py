"""Run PhageHostLearn on cipher's PHL validation matrix.

Uses Boeckaerts' pre-computed Loci + RBP ESM-2 embeddings from their
Zenodo deposit (so we skip Kaptive + ESM-2 entirely). Mirrors
Boeckaerts' own construct_feature_matrices() in feature construction:
mean-pool RBPs per phage, then concat [loci_emb || phage_mean_RBP_emb].

NOTE: PHL is PhageHostLearn's training set, so this is an
in-distribution sanity ceiling, NOT an OOD test. Useful for the
manuscript as a saturated baseline that other methods can be compared
against on the same panel.
"""
import json, pickle, csv
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import PHL_REPO as ROOT, CIPHER_REPO as _CIPHER_REPO, PHL_OUTPUT_ROOT, XGB_MODEL
CIPHER = _CIPHER_REPO / "data" / "validation_data" / "HOST_RANGE" / "PhageHostLearn" / "metadata"
ZEN = ROOT / "data" / "zenodo_11061100" / "11061100_unpacked"

OUT_DIR = PHL_OUTPUT_ROOT / "PHL" / "phagehostlearn_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIM = 1280

print("[1] Loading pre-computed embeddings", flush=True)
loci = pd.read_csv(ZEN / "esm2_embeddings_loci.csv")
rbp  = pd.read_csv(ZEN / "esm2_embeddings_rbp.csv")
print(f"  loci: {loci.shape[0]} hosts x {loci.shape[1]-1} d")
print(f"  RBP:  {rbp.shape[0]} proteins x {rbp.shape[1]-2} d, {rbp['phage_ID'].nunique()} unique phages")

print("\n[2] Mean-pool RBPs per phage", flush=True)
rbp_cols = rbp.columns[2:]
phage_mean = rbp.groupby("phage_ID")[rbp_cols].mean()
print(f"  phage embeddings: {phage_mean.shape}")

print("\n[3] Filter to cipher PHL phages and scoreable hosts", flush=True)
phl_phages = sorted({r["phage_id"] for r in csv.DictReader(open(CIPHER / "interaction_matrix.tsv"), delimiter="\t")})
phl_hosts  = sorted({r["host_id"] for r in csv.DictReader(open(CIPHER / "interaction_matrix.tsv"), delimiter="\t")})

scoreable_phages = [p for p in phl_phages if p in phage_mean.index]
scoreable_hosts  = [h for h in phl_hosts  if h in loci["accession"].values]
print(f"  PHL phages: {len(phl_phages)}, scoreable: {len(scoreable_phages)}")
print(f"  PHL hosts:  {len(phl_hosts)},  scoreable: {len(scoreable_hosts)}")

loci_idx = loci.set_index("accession")
print("\n[4] Building feature matrix and predicting", flush=True)
features = np.zeros((len(scoreable_hosts) * len(scoreable_phages), 2 * EMBED_DIM), dtype=np.float32)
row = 0
for h in scoreable_hosts:
    h_emb = loci_idx.loc[h, loci.columns[1:]].values.astype(np.float32)
    for p in scoreable_phages:
        p_emb = phage_mean.loc[p].values.astype(np.float32)
        features[row] = np.concatenate([h_emb, p_emb])
        row += 1
print(f"  features: {features.shape}")

xgb = XGBClassifier()
xgb.load_model(str(XGB_MODEL))
scores = xgb.predict_proba(features)[:, 1]
print(f"  scores  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}  std={scores.std():.4f}")

print("\n[5] Saving outputs", flush=True)
score_mat = scores.reshape(len(scoreable_hosts), len(scoreable_phages))
df = pd.DataFrame(score_mat, index=scoreable_hosts, columns=scoreable_phages)
df.to_csv(OUT_DIR / "prediction_scores.csv")
print(f"  -> {OUT_DIR / 'prediction_scores.csv'}")

# Ranked phages per host (for the same output format we used on CHEN)
ranked = {h: sorted(zip(scoreable_phages, score_mat[i]), key=lambda x: -x[1])
          for i, h in enumerate(scoreable_hosts)}
with open(OUT_DIR / "ranked_phages_per_host.pkl", "wb") as f:
    pickle.dump(ranked, f)
print(f"  -> {OUT_DIR / 'ranked_phages_per_host.pkl'}")
print("Done.")
