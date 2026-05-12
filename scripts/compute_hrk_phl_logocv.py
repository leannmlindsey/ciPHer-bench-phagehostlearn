"""Compute HR@k for PhageHostLearn LOGO-CV-held-out predictions on PHL.

Inputs:
  - logocv_predictions.csv: per-pair (host_id, phage_id, score, label) from
    leave-one-K-locus-out CV (each score is from a fold where that host's
    K-locus was held out)
  - cipher PHL interaction_matrix.tsv: for the host_K labels and the
    canonical candidate-host / candidate-phage sets

Outputs HR@k for: phage-anyhit A/B, host-anyhit A/B, per-paper splits
(Beamud / Ferriol / all). Matches the convention used for TropiSEQ /
TropiGAT in compute_hrk_anyhit.py.
"""
import csv, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PHL_LOGOCV = Path("/Users/leannmlindsey/WORK/CLAUDE_PHAGEHOSTLEARN/claude_copy/PhageHostLearn/data/cipher_eval/PHL/phagehostlearn_logocv")
CIPHER = Path("/Users/leannmlindsey/WORK/PHI_TSP/cipher/data/validation_data/HOST_RANGE/PhageHostLearn/metadata")
DPOT = Path("/Users/leannmlindsey/WORK/CLAUDE_DPOTROPISEARCH/claude_copy/DpoTropiSearch")

# Load LOGO-CV predictions and pivot to (host x phage) score matrix
df = pd.read_csv(PHL_LOGOCV / "logocv_predictions.csv")
scores = df.pivot(index="host_id", columns="phage_id", values="score")
print(f"LOGO-CV score matrix: {scores.shape}")

# Load cipher PHL interactions + phage subsets
interactions = defaultdict(dict)   # phage -> {host: label}
host_to_phages = defaultdict(dict) # host -> {phage: label}
KSTUB = re.compile(r"^K(\d+)$")
host_K = {}
for r in csv.DictReader(open(CIPHER/"interaction_matrix.tsv"), delimiter="\t"):
    try: lbl = int(r.get("label","0").strip() or 0)
    except: lbl = 0
    interactions[r["phage_id"]][r["host_id"]] = lbl
    host_to_phages[r["host_id"]][r["phage_id"]] = lbl
    hk = r.get("host_K","").strip() or None
    if hk is None:
        m = KSTUB.match(r["host_id"])
        if m: hk = "KL" + m.group(1)
    host_K[r["host_id"]] = hk

beamud  = set(open(DPOT/"benchmark_external/phl_phages_beamud.txt").read().split())
ferriol = set(open(DPOT/"benchmark_external/phl_phages_ferriol.txt").read().split())

ks = list(range(1, 21))
def hr(ranks):
    return {str(k): sum(1 for r in ranks if r and r <= k) / len(ranks) for k in ks} if ranks else {str(k):0.0 for k in ks}
def mrr(r): return sum(1.0/x for x in r)/len(r) if r else 0.0
def comp_rank(items):
    items = sorted(items, key=lambda x: -x[1])
    ranks, rp, last = {}, 0, None
    for i, (k, s) in enumerate(items, 1):
        if last is None or s != last:
            rp = i; last = s
        ranks[k] = rp
    return ranks

def evaluate(label, phage_filter=None):
    phages_keep = phage_filter if phage_filter else set(interactions.keys())
    phage_anyhit_A, phage_anyhit_B = [], []
    host_anyhit_A, host_anyhit_B = [], []
    for phage, hl in interactions.items():
        if phage not in phages_keep: continue
        pos = [h for h, l in hl.items() if l == 1]
        if not pos: continue
        cand = list(hl.keys()); n_cand = len(cand)
        if phage in scores.columns:
            host_scores = [(h, float(scores.loc[h,phage]) if (h in scores.index and not pd.isna(scores.loc[h,phage])) else -np.inf) for h in cand]
            ranks = comp_rank(host_scores)
            phage_anyhit_B.append(min(ranks[h] for h in pos))
            scored_pos = [h for h in pos if h in scores.index and not pd.isna(scores.loc[h,phage])]
            if scored_pos: phage_anyhit_A.append(min(ranks[h] for h in scored_pos))
        else:
            phage_anyhit_B.append(n_cand)
    for host, pl in host_to_phages.items():
        pl_kept = {p: l for p, l in pl.items() if p in phages_keep}
        pos = [p for p, l in pl_kept.items() if l == 1]
        if not pos: continue
        cand = list(pl_kept.keys()); n_cand = len(cand)
        if host in scores.index:
            phage_scores = [(p, float(scores.loc[host,p]) if (p in scores.columns and not pd.isna(scores.loc[host,p])) else -np.inf) for p in cand]
            ranks = comp_rank(phage_scores)
            host_anyhit_B.append(min(ranks[p] for p in pos))
            scored_pos = [p for p in pos if p in scores.columns and not pd.isna(scores.loc[host,p])]
            if scored_pos: host_anyhit_A.append(min(ranks[p] for p in scored_pos))
        else:
            host_anyhit_B.append(n_cand)
    return {
        "label": label,
        "ks": ks,
        "phage_anyhit_A": {"n":len(phage_anyhit_A), "hr_at_k": hr(phage_anyhit_A), "mrr": mrr(phage_anyhit_A), "ranks": phage_anyhit_A},
        "phage_anyhit_B": {"n":len(phage_anyhit_B), "hr_at_k": hr(phage_anyhit_B), "mrr": mrr(phage_anyhit_B), "ranks": phage_anyhit_B},
        "host_anyhit_A":  {"n":len(host_anyhit_A),  "hr_at_k": hr(host_anyhit_A),  "mrr": mrr(host_anyhit_A),  "ranks": host_anyhit_A},
        "host_anyhit_B":  {"n":len(host_anyhit_B),  "hr_at_k": hr(host_anyhit_B),  "mrr": mrr(host_anyhit_B),  "ranks": host_anyhit_B},
    }

for label, fil in [("all", None), ("beamud", beamud), ("ferriol", ferriol)]:
    res = evaluate(f"PHL_{label}_PhageHostLearn_LOGO-CV", fil)
    json.dump(res, open(PHL_LOGOCV/f"hrk_{label}.json","w"), indent=2)
    print(f"\n=== PHL/{label} PhageHostLearn LOGO-CV ===")
    for k in ["phage_anyhit_B","host_anyhit_B"]:
        d = res[k]; c = d["hr_at_k"]
        print(f"  {k:18s}  n={d['n']:4d}  HR@1={c['1']:.3f}  HR@3={c['3']:.3f}  HR@5={c['5']:.3f}  HR@10={c['10']:.3f}  MRR={d['mrr']:.3f}")
