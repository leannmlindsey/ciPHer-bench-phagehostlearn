"""End-to-end PhageHostLearn inference on the Wang validation dataset.

Inputs:
  - Wang/wang_rbps_tail.faa  (194 tail RBP records, 93 phages)
  - Wang/extracted_full/host_kp_genomes/*.fasta  (121 host KP genomes)

Strategy:
  1. Use Boeckaerts' pre-computed ESM-2 650M embeddings from
     Wang/phage_protein_ts_prediction_and_esm_embedding.pkl for phage RBPs
     (already 1280-d, mean-pooled per protein). Filter to is_tail=True
     proteins per phage; then mean-pool per phage.
  2. Run Kaptive on each of the 121 host genomes → K-locus protein sequences.
  3. ESM-2 mean-pool the K-locus proteins per host (1280-d each).
  4. Build (host × phage) features [loci_emb || phage_emb] (2560-d).
  5. Predict with Boeckaerts' bundled XGBoost.
  6. Save score matrix.
"""
import json, os, subprocess, pickle, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier
from tqdm import tqdm

from config import PHL_REPO as ROOT, CIPHER_VAL_GENOMES, PHL_OUTPUT_ROOT, XGB_MODEL
WANG = CIPHER_VAL_GENOMES / "Wang"
HOSTS = WANG/"extracted_full/host_kp_genomes"
ANNOS = WANG/"extracted_full/real_phage_annos.csv"
RBP_PKL = WANG/"phage_protein_ts_prediction_and_esm_embedding.pkl"

OUT_DIR = PHL_OUTPUT_ROOT / "Wang" / "phagehostlearn_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KAPTIVE_DB = ROOT/"data/kaptive_db/Klebsiella_k_locus_primary_reference.gbk"
KAPTIVE_PY = ROOT/"code/kaptive.py"
import os as _os
TROPISEQ_BIN = _os.environ.get("BLAST_BIN_DIR", "")

ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280
MAX_LEN = 1024
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# 1. Phage RBP per-phage mean embeddings (from Wang's pre-computed pickle)
# ---------------------------------------------------------------------------
print("[1] Load phage RBP embeddings from Wang pickle + filter to is_tail=True", flush=True)
rbp_df = pd.read_pickle(RBP_PKL)
print(f"  pickle rows: {len(rbp_df)}; cols: {list(rbp_df.columns)}")

# Wang's `is_tail` annotation lives in real_phage_annos.csv (separate file)
annos = pd.read_csv(ANNOS)
tail_set = set(annos[annos['is_tail'] == True]['ORFname'])
print(f"  is_tail proteins: {len(tail_set)} across {annos[annos['is_tail']==True]['phage_name'].nunique()} phages")

# Filter pickle to tail proteins only
rbp_tail = rbp_df[rbp_df['ORFname'].isin(tail_set)].copy()
print(f"  matched in pickle: {len(rbp_tail)} rows across {rbp_tail['phage_name'].nunique()} phages")

# Convert tensor embeddings to numpy + mean per phage
def to_np(emb):
    return np.array([float(t) for t in emb], dtype=np.float32)

rbp_tail['emb_np'] = rbp_tail['embedding'].apply(to_np)
phage_emb = {}
for phage, group in rbp_tail.groupby('phage_name'):
    phage_emb[phage] = np.mean(np.stack(group['emb_np'].values), axis=0)
print(f"  per-phage mean embeddings: {len(phage_emb)} phages")


# ---------------------------------------------------------------------------
# 2. Kaptive on each host
# ---------------------------------------------------------------------------
KAPTIVE_TMP = OUT_DIR/"kaptive_tmp"
KAPTIVE_TMP.mkdir(exist_ok=True)
LOCIBASE = OUT_DIR/"Locibase.json"

if LOCIBASE.exists():
    print(f"\n[2] Reusing cached {LOCIBASE}")
    locibase = json.load(open(LOCIBASE))
else:
    print(f"\n[2] Run Kaptive on 121 Wang host genomes", flush=True)
    locibase = {}
    serotypes = {}
    env = os.environ.copy()
    env["PATH"] = TROPISEQ_BIN + ":" + env["PATH"]
    host_files = sorted(HOSTS.glob("*.fasta"))
    for host_fa in tqdm(host_files):
        accession = host_fa.stem
        # Delete kaptive_results.json before each run (Kaptive APPENDS otherwise)
        (KAPTIVE_TMP/"kaptive_results.json").unlink(missing_ok=True)
        cmd = ["python", str(KAPTIVE_PY),
               "-a", str(host_fa),
               "-k", str(KAPTIVE_DB),
               "-o", str(KAPTIVE_TMP) + "/",
               "--no_table"]
        result = subprocess.run(cmd, cwd=str(ROOT/"code"), env=env,
                                capture_output=True, text=True)
        if not (KAPTIVE_TMP/"kaptive_results.json").exists():
            print(f"  Kaptive failed for {accession}: {result.stderr[:200]}")
            continue
        kr = json.load(open(KAPTIVE_TMP/"kaptive_results.json"))
        if not kr: continue
        serotypes[accession] = kr[0]["Best match"]["Type"]
        proteins = []
        for gene in kr[0]["Locus genes"]:
            try:
                p = gene["tblastn result"]["Protein sequence"]
            except KeyError:
                p = gene["Reference"]["Protein sequence"]
            p = p.replace("-", "").replace("*", "").rstrip("*")
            if p:
                proteins.append(p)
        locibase[accession] = proteins
    json.dump(locibase, open(LOCIBASE, "w"))
    pd.DataFrame({"accession": list(serotypes.keys()),
                  "kaptive_type": list(serotypes.values())}
                 ).to_csv(OUT_DIR/"kaptive_serotypes.csv", index=False)
    print(f"  -> {LOCIBASE} ({len(locibase)} hosts)")


# ---------------------------------------------------------------------------
# 3. ESM-2 mean embeddings for K-locus proteins (per host)
# ---------------------------------------------------------------------------
print(f"\n[3] ESM-2 650M on K-locus proteins ({len(locibase)} hosts)", flush=True)
tok = AutoTokenizer.from_pretrained(ESM2_NAME)
model = AutoModel.from_pretrained(ESM2_NAME).to(DEVICE)
model.eval()

def embed_seq(seq):
    inputs = tok(seq, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state[0, 1:-1, :]
    return h.mean(dim=0).cpu().numpy()

loci_emb = {}
for accession, proteins in tqdm(locibase.items()):
    if not proteins: continue
    embs = [embed_seq(p) for p in proteins]
    loci_emb[accession] = np.mean(np.stack(embs), axis=0).astype(np.float32)
print(f"  {len(loci_emb)} hosts embedded")


# ---------------------------------------------------------------------------
# 4. Build feature matrix and predict
# ---------------------------------------------------------------------------
print(f"\n[4] Build features + XGBoost predict", flush=True)
phages = sorted(phage_emb.keys())
hosts = sorted(loci_emb.keys())
features = np.zeros((len(hosts) * len(phages), 2 * EMBED_DIM), dtype=np.float32)
row = 0
for h in hosts:
    h_emb = loci_emb[h]
    for p in phages:
        features[row] = np.concatenate([h_emb, phage_emb[p]])
        row += 1
print(f"  features: {features.shape} ({len(hosts)} hosts x {len(phages)} phages)")

xgb = XGBClassifier()
xgb.load_model(str(XGB_MODEL))
scores = xgb.predict_proba(features)[:, 1]
print(f"  scores  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")

# Save matrix
score_mat = scores.reshape(len(hosts), len(phages))
df = pd.DataFrame(score_mat, index=hosts, columns=phages)
df.to_csv(OUT_DIR/"prediction_scores.csv")
print(f"  -> {OUT_DIR/'prediction_scores.csv'}")
print("Done.")
