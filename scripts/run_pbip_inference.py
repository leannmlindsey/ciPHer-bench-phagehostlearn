"""PhageHostLearn on PBIP: cipher's RBPs + Kaptive Locibase for 125 hosts.

Paths come from phagehostlearn.env (source it before running). See config.py.
"""
import json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from xgboost import XGBClassifier
from tqdm import tqdm

from config import PHL_REPO as ROOT, CIPHER_REPO as CIPHER, CIPHER_VAL_GENOMES, PHL_OUTPUT_ROOT, XGB_MODEL
PBIP = CIPHER_VAL_GENOMES / "PBIP"
OUT_DIR = PHL_OUTPUT_ROOT / "PBIP" / "phagehostlearn_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280

print("[1] PBIP RBPs from cipher", flush=True)
phage_to_proteins = defaultdict(list)
with open(CIPHER/"data/validation_data/HOST_RANGE/PBIP/metadata/phage_protein_mapping.csv") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        phage_to_proteins[r["matrix_phage_name"]].append(r["protein_id"])
all_rbp = {r.id: str(r.seq).strip("*") for r in SeqIO.parse(str(CIPHER/"data/validation_data/metadata/validation_rbps_all.faa"), "fasta")}
phage_recs = {p: [(pid, all_rbp[pid]) for pid in pids if pid in all_rbp]
              for p, pids in phage_to_proteins.items()}
n_rbps = sum(len(v) for v in phage_recs.values())
print(f"  {n_rbps} RBPs across {len(phage_recs)} phages")

print("[2] Load Kaptive Locibase", flush=True)
locibase = json.load(open(PBIP/"kaptive_out/Locibase.json"))
print(f"  {len(locibase)} hosts in Locibase")

# Cipher PBIP uses host_id == host_assembly (both like KP4023)
host_alias = {}
with open(CIPHER/"data/validation_data/HOST_RANGE/PBIP/metadata/interaction_matrix.tsv") as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for r in rdr:
        host_alias[r["host_id"]] = r["host_assembly"]
loci_hosts = set(locibase.keys())
cipher_to_loci = {h: asm for h, asm in host_alias.items() if asm in loci_hosts}
print(f"  cipher hosts mapped: {len(cipher_to_loci)} / {len(host_alias)}")

print(f"\n[3] ESM-2 650M", flush=True)
tok = AutoTokenizer.from_pretrained(ESM2_NAME)
model = AutoModel.from_pretrained(ESM2_NAME).to("cpu"); model.eval()

def embed_seq(seq):
    inputs = tok(seq, return_tensors="pt", truncation=True, max_length=1024).to("cpu")
    with torch.no_grad(): out = model(**inputs)
    return out.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()

print("[3a] RBPs", flush=True)
rbp_phage_emb = {}
for phage, recs in tqdm(phage_recs.items()):
    if not recs: continue
    embs = [embed_seq(s) for _, s in recs]
    rbp_phage_emb[phage] = np.mean(embs, axis=0).astype(np.float32)
print(f"  {len(rbp_phage_emb)} phage embeddings")

print("\n[3b] Loci", flush=True)
loci_emb = {}
for acc, prots in tqdm(locibase.items()):
    if not prots: continue
    embs = [embed_seq(p) for p in prots]
    loci_emb[acc] = np.mean(embs, axis=0).astype(np.float32)
print(f"  {len(loci_emb)} loci embeddings")

print("\n[4] Build features + XGBoost predict", flush=True)
phages = sorted(rbp_phage_emb.keys())
hosts = sorted(cipher_to_loci.keys())
features = np.zeros((len(hosts)*len(phages), 2*EMBED_DIM), dtype=np.float32)
row = 0
for h in hosts:
    he = loci_emb.get(cipher_to_loci[h])
    for p in phages:
        features[row] = np.concatenate([he if he is not None else np.zeros(EMBED_DIM, dtype=np.float32), rbp_phage_emb[p]])
        row += 1
print(f"  features {features.shape}")

xgb = XGBClassifier(); xgb.load_model(str(XGB_MODEL))
scores = xgb.predict_proba(features)[:, 1]
print(f"  scores  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")

mat = scores.reshape(len(hosts), len(phages))
df = pd.DataFrame(mat, index=hosts, columns=phages)
df.to_csv(OUT_DIR/"prediction_scores.csv")
print(f"  -> {OUT_DIR/'prediction_scores.csv'}")
