"""PhageHostLearn on GORODNICHIV via the KL23-reference workaround.

GORODNICHIV's 83 hosts have no public genome data (confirmed with author
2026-05-11). All 83 hosts are KL23. We approximate the PhageHostLearn
host-side feature by:

  1. Extract KL23 K-locus protein sequences from Kaptive's bundled
     reference DB (Klebsiella_k_locus_primary_reference.gbk).
  2. ESM-2 mean-embed those proteins → one 1280-d KL23 reference embedding.
  3. Apply that embedding to all 83 hosts.

Phage RBPs come from cipher's validation_rbps_all.faa, filtered to the
proteins listed in cipher's GORODNICHIV phage_protein_mapping.csv.

This is documented in the GORODNICHIV README as the "KL23-reference
workaround."
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

ROOT = Path("/Users/leannmlindsey/WORK/CLAUDE_PHAGEHOSTLEARN/claude_copy/PhageHostLearn")
CIPHER = Path("/Users/leannmlindsey/WORK/PHI_TSP/cipher")
DATASET = "GORODNICHIV"
OUT_DIR = ROOT/"data/cipher_eval/GORODNICHIV/phagehostlearn_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KAPTIVE_DB = ROOT/"data/kaptive_db/Klebsiella_k_locus_primary_reference.gbk"
XGB_MODEL = ROOT/"code/phagehostlearn_esm2_xgb.json"

ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280
MAX_LEN = 1024
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# 1. Phage RBPs from cipher's validation_rbps_all.faa
# ---------------------------------------------------------------------------
print("[1] Load GORODNICHIV phage RBPs from cipher validation set", flush=True)
phage_to_proteins = defaultdict(list)
with open(CIPHER/"data/validation_data/HOST_RANGE"/DATASET/"metadata/phage_protein_mapping.csv") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        phage_to_proteins[r["matrix_phage_name"]].append(r["protein_id"])

all_rbp_seqs = {}
for r in SeqIO.parse(str(CIPHER/"data/validation_data/metadata/validation_rbps_all.faa"), "fasta"):
    all_rbp_seqs[r.id] = str(r.seq).strip("*").rstrip("*")

phage_rbp_records = {}
for phage, prots in phage_to_proteins.items():
    seqs = [(p, all_rbp_seqs[p]) for p in prots if p in all_rbp_seqs]
    phage_rbp_records[phage] = seqs
    print(f"  {phage}: {len(seqs)} RBP sequences")


# ---------------------------------------------------------------------------
# 2. Extract KL23 reference K-locus protein sequences
# ---------------------------------------------------------------------------
print("\n[2] Extract KL23 reference proteins from Kaptive GBK", flush=True)
kl23_proteins = []
kl23_record_id = None
for rec in SeqIO.parse(str(KAPTIVE_DB), "genbank"):
    # Kaptive records use the K-type as the LOCUS / id (e.g. id="K23" or "KL23")
    if rec.id in ("K23", "KL23") or rec.name in ("K23", "KL23"):
        kl23_record_id = rec.id
        for feat in rec.features:
            if feat.type == "CDS" and "translation" in feat.qualifiers:
                prot = feat.qualifiers["translation"][0]
                prot = prot.replace("-","").replace("*","").rstrip("*")
                if prot:
                    kl23_proteins.append(prot)
        break

print(f"  found record: {kl23_record_id}")
print(f"  KL23 reference proteins: {len(kl23_proteins)}")
assert len(kl23_proteins) > 0, "No KL23 record found in Kaptive DB"


# ---------------------------------------------------------------------------
# 3. ESM-2 embeddings: RBPs (per phage) + KL23 reference (one)
# ---------------------------------------------------------------------------
print(f"\n[3] Loading ESM-2 ({ESM2_NAME})", flush=True)
tok = AutoTokenizer.from_pretrained(ESM2_NAME)
model = AutoModel.from_pretrained(ESM2_NAME).to(DEVICE)
model.eval()

def embed_seq(seq):
    inputs = tok(seq, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state[0, 1:-1, :]
    return h.mean(dim=0).cpu().numpy()

print("[3a] RBP per-phage mean embeddings", flush=True)
rbp_phage_emb = {}
for phage, recs in phage_rbp_records.items():
    if not recs: continue
    embs = [embed_seq(s) for _, s in tqdm(recs, desc=phage, leave=False)]
    rbp_phage_emb[phage] = np.mean(embs, axis=0).astype(np.float32)
print(f"  got {len(rbp_phage_emb)} phage embeddings")

print(f"\n[3b] KL23 reference embedding (one — applied to all 83 hosts)", flush=True)
kl23_embs = [embed_seq(p) for p in tqdm(kl23_proteins)]
kl23_ref_emb = np.mean(kl23_embs, axis=0).astype(np.float32)
print(f"  KL23 ref embedding shape: {kl23_ref_emb.shape}")


# ---------------------------------------------------------------------------
# 4. Apply KL23 ref embedding to all 83 hosts and run XGBoost
# ---------------------------------------------------------------------------
print(f"\n[4] Build features (83 hosts × 3 phages) — KL23 ref embedding broadcast", flush=True)
# Get 83 host_ids from cipher's interaction matrix
hosts = []
with open(CIPHER/"data/validation_data/HOST_RANGE"/DATASET/"metadata/interaction_matrix.tsv") as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for r in rdr:
        hosts.append(r["host_id"])
hosts = sorted(set(hosts))
phages = sorted(rbp_phage_emb.keys())
print(f"  hosts: {len(hosts)}, phages: {len(phages)}")

features = np.zeros((len(hosts) * len(phages), 2 * EMBED_DIM), dtype=np.float32)
row = 0
for h in hosts:
    for p in phages:
        features[row] = np.concatenate([kl23_ref_emb, rbp_phage_emb[p]])
        row += 1
print(f"  features: {features.shape}")

xgb = XGBClassifier()
xgb.load_model(str(XGB_MODEL))
scores = xgb.predict_proba(features)[:, 1]
print(f"  scores  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}  std={scores.std():.4f}")

score_mat = scores.reshape(len(hosts), len(phages))
df = pd.DataFrame(score_mat, index=hosts, columns=phages)
df.to_csv(OUT_DIR/"prediction_scores.csv")
print(f"  -> {OUT_DIR/'prediction_scores.csv'}")
print("\nDone.")
