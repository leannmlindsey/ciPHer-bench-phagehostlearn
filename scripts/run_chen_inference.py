"""End-to-end PhageHostLearn inference on cipher's CHEN dataset.

Pipeline:
  1. Pull cipher's CHEN RBPs from validation_rbps_all.faa using
     cipher's phage_protein_mapping.csv (skipping PHANOTATE + PhageRBPdetect
     since cipher has already identified the RBPs).
  2. Run Kaptive on each of the 49 CHEN host genomes -> K-locus proteins
     (Locibase.json).
  3. Compute ESM-2 650M mean embeddings for RBPs and K-locus proteins
     (matches PhageHostLearn's published feature setup; verified
     cosine-1.0 against authors' bundled embeddings in earlier work).
  4. Build (phage, host) feature matrix: concat(loci_mean_emb, rbp_mean_emb).
  5. Predict with the trained XGBoost
     (code/phagehostlearn_esm2_xgb.json from the published deposit).
  6. Save score matrix.
"""
import json, os, subprocess, sys, csv
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
DATASET = "CHEN"
OUT_DIR = ROOT / "data" / "cipher_eval" / DATASET / "phagehostlearn_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KAPTIVE_DB = ROOT / "data" / "kaptive_db" / "Klebsiella_k_locus_primary_reference.gbk"
KAPTIVE_PY = ROOT / "code" / "kaptive.py"
XGB_MODEL = ROOT / "code" / "phagehostlearn_esm2_xgb.json"
# Use TropiSEQ_env's conda BLAST (brew's blast has an mbedtls mismatch).
TROPISEQ_BIN = "/Users/leannmlindsey/miniconda3/envs/TropiSEQ_env/bin"

ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280
MAX_LEN = 1024
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# 1. RBPs for CHEN's phages from cipher
# ---------------------------------------------------------------------------
print("[1/6] Loading CHEN RBPs from cipher's validation_rbps_all.faa", flush=True)
phage_to_proteins = defaultdict(list)
with open(CIPHER / "data/validation_data/HOST_RANGE" / DATASET / "metadata/phage_protein_mapping.csv") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        phage_to_proteins[r["matrix_phage_name"]].append(r["protein_id"])

print(f"  CHEN phages and their RBPs:")
for phage, prots in phage_to_proteins.items():
    print(f"    {phage}: {len(prots)} RBPs")

# Pull each protein's sequence from validation_rbps_all.faa
all_rbp_seqs = {}
for r in SeqIO.parse(str(CIPHER / "data/validation_data/metadata/validation_rbps_all.faa"), "fasta"):
    all_rbp_seqs[r.id] = str(r.seq).strip("*").rstrip("*")

# Build per-phage RBP record list
phage_rbp_records = {}
for phage, prots in phage_to_proteins.items():
    seqs = [(p, all_rbp_seqs[p]) for p in prots if p in all_rbp_seqs]
    phage_rbp_records[phage] = seqs
    print(f"    {phage}: matched {len(seqs)} RBP sequences in FASTA")


# ---------------------------------------------------------------------------
# 2. Kaptive on every CHEN host genome -> Locibase.json
# ---------------------------------------------------------------------------
LOCIBASE = OUT_DIR / "Locibase.json"
KAPTIVE_TMP = OUT_DIR / "kaptive_tmp"
KAPTIVE_TMP.mkdir(exist_ok=True)
locibase = {}

# Skip rerun if cached
if LOCIBASE.exists():
    print(f"\n[2/6] Reusing cached {LOCIBASE}")
    locibase = json.load(open(LOCIBASE))
else:
    print("\n[2/6] Running Kaptive on 49 CHEN host genomes...", flush=True)
    host_dir = ROOT / "data" / "cipher_eval" / DATASET / "bacteria"
    host_files = sorted(host_dir.glob("*.fasta"))
    serotypes = {}
    env = os.environ.copy()
    env["PATH"] = TROPISEQ_BIN + ":" + env["PATH"]
    for host_fa in tqdm(host_files):
        accession = host_fa.stem
        cmd = [
            "python", str(KAPTIVE_PY),
            "-a", str(host_fa),
            "-k", str(KAPTIVE_DB),
            "-o", str(KAPTIVE_TMP) + "/",
            "--no_table",
        ]
        result = subprocess.run(cmd, cwd=str(ROOT / "code"), env=env,
                                capture_output=True, text=True)
        kresults = json.load(open(KAPTIVE_TMP / "kaptive_results.json"))
        serotypes[accession] = kresults[0]["Best match"]["Type"]
        proteins = []
        for gene in kresults[0]["Locus genes"]:
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
                 ).to_csv(OUT_DIR / "kaptive_serotypes.csv", index=False)
    print(f"  Wrote {LOCIBASE} ({len(locibase)} hosts)")


# ---------------------------------------------------------------------------
# 3. ESM-2 embeddings: load model once, embed RBPs and Loci proteins
# ---------------------------------------------------------------------------
print(f"\n[3/6] Loading ESM-2 ({ESM2_NAME}) on {DEVICE}...", flush=True)
tok = AutoTokenizer.from_pretrained(ESM2_NAME)
model = AutoModel.from_pretrained(ESM2_NAME).to(DEVICE)
model.eval()


def embed_seq(seq):
    inputs = tok(seq, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state[0, 1:-1, :]
    return h.mean(dim=0).cpu().numpy()


print("[3a/6] ESM-2 mean embeddings for RBPs (per phage)", flush=True)
rbp_phage_emb = {}  # phage_id -> mean over its RBPs (1280,)
for phage, recs in phage_rbp_records.items():
    if not recs:
        continue
    embs = [embed_seq(s) for _, s in tqdm(recs, desc=phage, leave=False)]
    rbp_phage_emb[phage] = np.mean(embs, axis=0)
print(f"  Got mean RBP embedding for {len(rbp_phage_emb)} phages")

print("\n[3b/6] ESM-2 mean embeddings for K-locus proteins (per host)", flush=True)
loci_host_emb = {}
for accession, proteins in tqdm(locibase.items()):
    if not proteins:
        continue
    embs = [embed_seq(p) for p in proteins]
    loci_host_emb[accession] = np.mean(embs, axis=0)
print(f"  Got mean Loci embedding for {len(loci_host_emb)} hosts")


# ---------------------------------------------------------------------------
# 4. Build (phage x host) feature matrix
# ---------------------------------------------------------------------------
print(f"\n[4/6] Building feature matrix (host_loci_emb || phage_rbp_emb)...", flush=True)
phages = sorted(rbp_phage_emb.keys())
hosts = sorted(loci_host_emb.keys())
features = np.zeros((len(hosts) * len(phages), 2 * EMBED_DIM), dtype=np.float32)
groups_bact = []
row = 0
for i, host in enumerate(hosts):
    h_emb = loci_host_emb[host]
    for j, phage in enumerate(phages):
        p_emb = rbp_phage_emb[phage]
        features[row] = np.concatenate([h_emb, p_emb])
        groups_bact.append(i)
        row += 1
print(f"  features: {features.shape}  pairs: {len(hosts)} hosts x {len(phages)} phages = {features.shape[0]}")


# ---------------------------------------------------------------------------
# 5. Predict with the trained XGBoost
# ---------------------------------------------------------------------------
print("\n[5/6] Loading trained XGBoost and predicting...", flush=True)
xgb = XGBClassifier()
xgb.load_model(str(XGB_MODEL))
scores = xgb.predict_proba(features)[:, 1]
print(f"  predicted scores: min={scores.min():.3f}  max={scores.max():.3f}  mean={scores.mean():.3f}")


# ---------------------------------------------------------------------------
# 6. Save score matrix + ranked_results
# ---------------------------------------------------------------------------
print("\n[6/6] Saving outputs...", flush=True)
score_matrix = scores.reshape(len(hosts), len(phages))
result_df = pd.DataFrame(score_matrix, index=hosts, columns=phages)
out_csv = OUT_DIR / "prediction_scores.csv"
result_df.to_csv(out_csv)
print(f"  score matrix -> {out_csv}")
print(result_df.head())

# Ranked: per host, sort phages descending
ranked = {}
for i, host in enumerate(hosts):
    s = scores[groups_bact == np.array(groups_bact)][i*len(phages):(i+1)*len(phages)]
    ranked[host] = sorted(zip(phages, score_matrix[i]), key=lambda x: -x[1])
import pickle
with open(OUT_DIR / "ranked_phages_per_host.pkl", "wb") as f:
    pickle.dump(ranked, f)
print(f"  ranked phages per host -> {OUT_DIR / 'ranked_phages_per_host.pkl'}")
print("\nDone.")
