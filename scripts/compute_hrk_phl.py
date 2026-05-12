"""HR@k for PhageHostLearn predictions on a cipher validation dataset.

Mirrors cipher.evaluation.ranking.evaluate_rankings() conventions used in
DpoTropiSearch/benchmark_external/tropigat_run/compute_hrk_tropigat.py:

  - Candidate hosts per phage = the hosts that appear in this phage's row of
    the interaction_matrix (typically all hosts in the dataset for CHEN/GORO/UCSD,
    since they are exhaustive panels).

  - Two ranking metrics:
      (1) per-pair hr_at_k         — every (phage, pos_host) contributes one rank
      (2) phage-anyhit hr_at_k     — each phage contributes ONE rank: best across
                                     its true positives. (cipher's headline metric)

  - Two abstention conventions for hosts not in our prediction matrix
    (e.g. NTUH-K2044 missing from CHEN's Globus dump):
      (A) restrict-to-scored: drop phages where no positive host has a score
      (B) all-phages: missing-host scores treated as -inf (so they fall to the
          back of the ranking) — abstaining phages get rank = n_candidates

PhageHostLearn predicts directly per (host, phage) pair, so unlike TropiSEQ
there's no KL-locus indirection — score lookup is just a matrix index.
"""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from config import CIPHER_REPO
CIPHER_DATA = CIPHER_REPO / "data" / "validation_data"
KS = list(range(1, 21))


def load_interactions(dataset, host_key="host_assembly"):
    """Return interactions[phage][host] = 0/1 from cipher's HOST_RANGE matrix.

    PhageHostLearn predicts per host_assembly (we ran Kaptive per GCA), so we
    key on host_assembly by default. Some host_assembly values may be a strain
    name rather than a GCA (CHEN's NTUH-K2044 is keyed as 'NTUH-K2044' in
    interaction_matrix.tsv), which is fine — that just means lookup will miss
    for that host and convention A drops it.
    """
    path = CIPHER_DATA / "HOST_RANGE" / dataset / "metadata" / "interaction_matrix.tsv"
    interactions = defaultdict(dict)
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            try:
                lbl = int(row.get("label", "0").strip() or 0)
            except ValueError:
                lbl = 0
            host = row.get(host_key) or row["host_id"]
            interactions[row["phage_id"]][host] = lbl
    return dict(interactions)


def rank_hosts_for_phage(phage, candidates, scores_df, missing_score=-np.inf):
    """Return host_to_rank using competition tie ranking. Higher score = better rank.

    candidates: list of host_ids from interaction matrix (cipher's host_id, may be
                strain name like NTUH-K2044 OR an assembly accession like GCA_*).
    scores_df:  index = host accession used in our prediction matrix (matches the
                FASTA stem we ran Kaptive on), columns = phage_ids.
    """
    scored = []
    for hid in candidates:
        if hid in scores_df.index and phage in scores_df.columns:
            s = float(scores_df.loc[hid, phage])
        else:
            s = missing_score
        scored.append((hid, s))
    scored.sort(key=lambda x: -x[1])
    ranks = {}
    rank_pos = 0
    last_score = None
    for i, (hid, s) in enumerate(scored, start=1):
        if last_score is None or s != last_score:
            rank_pos = i
            last_score = s
        ranks[hid] = rank_pos
    return ranks


def hr_curve(ranks, ks=KS):
    if not ranks:
        return {str(k): 0.0 for k in ks}
    return {str(k): sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in ks}


def mrr(ranks):
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r and r > 0) / len(ranks)


def rank_phages_for_host(host, candidates, scores_df, missing_score=-np.inf):
    """Inverse of rank_hosts_for_phage: rank phage candidates for a single host."""
    scored = []
    for ph in candidates:
        if host in scores_df.index and ph in scores_df.columns:
            s = float(scores_df.loc[host, ph])
        else:
            s = missing_score
        scored.append((ph, s))
    scored.sort(key=lambda x: -x[1])
    ranks, rank_pos, last = {}, 0, None
    for i, (ph, s) in enumerate(scored, start=1):
        if last is None or s != last:
            rank_pos = i; last = s
        ranks[ph] = rank_pos
    return ranks


def evaluate(dataset, scores_df, host_id_map=None):
    """Compute phage-anyhit AND host-anyhit HR@k.

    Per cipher manuscript convention: only anyhit metrics are reported.
    - phage-anyhit: for each phage, best rank across its positive hosts
    - host-anyhit:  for each host,  best rank across its positive phages
    """
    interactions = load_interactions(dataset)

    # phage-anyhit
    phage_anyhit_A, phage_anyhit_B = [], []
    n_phages_with_pos = 0
    n_phages_with_scored_pos = 0
    # host-anyhit (build host_to_phages by inverting)
    host_to_phages = defaultdict(dict)
    all_phages = set()
    for phage, host_labels in interactions.items():
        all_phages.add(phage)
        for h, lbl in host_labels.items():
            host_to_phages[h][phage] = lbl
    all_phages = sorted(all_phages)

    for phage, host_labels in interactions.items():
        candidates = list(host_labels.keys())
        pos_hosts = [h for h, lbl in host_labels.items() if lbl == 1]
        if not pos_hosts:
            continue
        n_phages_with_pos += 1

        if host_id_map is not None:
            cand_for_rank = [host_id_map.get(h, h) for h in candidates]
            pos_for_rank = [host_id_map.get(h, h) for h in pos_hosts]
        else:
            cand_for_rank = candidates
            pos_for_rank = pos_hosts

        scored_pos_A = [h for h in pos_for_rank if h in scores_df.index and phage in scores_df.columns]
        n_cand = len(cand_for_rank)

        if phage in scores_df.columns:
            ranks_B = rank_hosts_for_phage(phage, cand_for_rank, scores_df, missing_score=-np.inf)
            phage_anyhit_B.append(min(ranks_B[h] for h in pos_for_rank))
            if scored_pos_A:
                n_phages_with_scored_pos += 1
                phage_anyhit_A.append(min(ranks_B[h] for h in scored_pos_A))
        else:
            # Phage abstains entirely (not in prediction matrix): worst rank.
            phage_anyhit_B.append(n_cand)

    host_anyhit_A, host_anyhit_B = [], []
    n_hosts_with_pos = 0
    n_hosts_with_scored_pos = 0
    for host, phage_labels in host_to_phages.items():
        pos_phages = [p for p, lbl in phage_labels.items() if lbl == 1]
        if not pos_phages:
            continue
        n_hosts_with_pos += 1

        host_for_rank = host_id_map.get(host, host) if host_id_map else host
        cand_phages = sorted(phage_labels.keys())
        n_cand_p = len(cand_phages)
        if host_for_rank in scores_df.index:
            ranks_B = rank_phages_for_host(host_for_rank, cand_phages, scores_df, missing_score=-np.inf)
            host_anyhit_B.append(min(ranks_B[p] for p in pos_phages))
            scored_pos_A = [p for p in pos_phages if p in scores_df.columns]
            if scored_pos_A:
                n_hosts_with_scored_pos += 1
                host_anyhit_A.append(min(ranks_B[p] for p in scored_pos_A))
        else:
            # Host abstains entirely (not in prediction matrix): worst rank.
            host_anyhit_B.append(n_cand_p)

    return {
        "dataset": dataset,
        "n_phages_with_pos": n_phages_with_pos,
        "n_phages_scored":   n_phages_with_scored_pos,
        "n_hosts_with_pos":  n_hosts_with_pos,
        "n_hosts_scored":    n_hosts_with_scored_pos,
        "candidate_hosts_avg":  (sum(len(hl) for hl in interactions.values())
                                / len(interactions)) if interactions else 0,
        "candidate_phages_avg": (sum(len(pl) for pl in host_to_phages.values())
                                / len(host_to_phages)) if host_to_phages else 0,
        "phage_anyhit_A": phage_anyhit_A, "phage_anyhit_B": phage_anyhit_B,
        "host_anyhit_A":  host_anyhit_A,  "host_anyhit_B":  host_anyhit_B,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True,
                    help="CSV from run_*_inference.py: rows=hosts, cols=phages, values=score")
    ap.add_argument("--dataset", required=True, choices=["CHEN", "GORODNICHIV", "UCSD", "PBIP", "PhageHostLearn"])
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--host_id_map_tsv", default=None,
                    help="Optional TSV with columns cipher_host_id\\tprediction_index "
                         "to translate cipher matrix host_ids to our prediction matrix index")
    args = ap.parse_args()

    scores_df = pd.read_csv(args.scores_csv, index_col=0)
    print(f"Loaded prediction matrix: {scores_df.shape[0]} hosts x {scores_df.shape[1]} phages")
    print(f"  hosts (head): {list(scores_df.index[:5])}")
    print(f"  phages: {list(scores_df.columns)}")

    host_id_map = None
    if args.host_id_map_tsv:
        host_id_map = {}
        with open(args.host_id_map_tsv) as f:
            for line in f:
                a, b = line.rstrip("\n").split("\t")[:2]
                host_id_map[a] = b

    res = evaluate(args.dataset, scores_df, host_id_map=host_id_map)
    print(f"\n=== {args.dataset} ===")
    print(f"  phages with positives: {res['n_phages_with_pos']}, scored: {res['n_phages_scored']}, "
          f"avg candidate hosts: {res['candidate_hosts_avg']:.1f}")
    print(f"  hosts with positives:  {res['n_hosts_with_pos']}, scored: {res['n_hosts_scored']}, "
          f"avg candidate phages: {res['candidate_phages_avg']:.1f}")
    for label, key in [("phage-anyhit A (scored)", "phage_anyhit_A"),
                        ("phage-anyhit B (all)",    "phage_anyhit_B"),
                        ("host-anyhit  A (scored)", "host_anyhit_A"),
                        ("host-anyhit  B (all)",    "host_anyhit_B")]:
        ranks = res[key]
        c = hr_curve(ranks)
        print(f"  {label:25s} n={len(ranks):4d}  HR@1={c['1']:.3f}  HR@3={c['3']:.3f}  "
              f"HR@5={c['5']:.3f}  HR@10={c['10']:.3f}  MRR={mrr(ranks):.3f}")

    out = {
        "method": "PhageHostLearn (ESM-2 + XGBoost)",
        "dataset": args.dataset,
        "ks": KS,
        "n_phages_with_pos": res["n_phages_with_pos"],
        "n_phages_scored":   res["n_phages_scored"],
        "n_hosts_with_pos":  res["n_hosts_with_pos"],
        "n_hosts_scored":    res["n_hosts_scored"],
        "candidate_hosts_avg":  res["candidate_hosts_avg"],
        "candidate_phages_avg": res["candidate_phages_avg"],
        "phage_anyhit_A_restrict_to_scored": {
            "n_phages": len(res["phage_anyhit_A"]),
            "hr_at_k": hr_curve(res["phage_anyhit_A"]),
            "mrr":     mrr(res["phage_anyhit_A"]),
        },
        "phage_anyhit_B_all_phages": {
            "n_phages": len(res["phage_anyhit_B"]),
            "hr_at_k": hr_curve(res["phage_anyhit_B"]),
            "mrr":     mrr(res["phage_anyhit_B"]),
        },
        "host_anyhit_A_restrict_to_scored": {
            "n_hosts": len(res["host_anyhit_A"]),
            "hr_at_k": hr_curve(res["host_anyhit_A"]),
            "mrr":     mrr(res["host_anyhit_A"]),
        },
        "host_anyhit_B_all_hosts": {
            "n_hosts": len(res["host_anyhit_B"]),
            "hr_at_k": hr_curve(res["host_anyhit_B"]),
            "mrr":     mrr(res["host_anyhit_B"]),
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.output_json}")


if __name__ == "__main__":
    main()
