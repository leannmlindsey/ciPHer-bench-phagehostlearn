# ciPHer-bench-phagehostlearn

Reproducible wrapper around **PhageHostLearn** (Boeckaerts et al. 2024,
*Nature Communications* 15:4226, DOI 10.1038/s41467-024-48675-6) for
evaluation on ciPHer's K. pneumoniae validation panel.

Upstream: https://github.com/dimiboeckaerts/PhageHostLearn

## What this repo contains

- `scripts/run_<dataset>_inference.py` — per-cipher-dataset wrappers that
  load the bundled XGBoost classifier, embed cipher's already-extracted
  RBPs with ESM-2 650M, embed each dataset's host K-locus proteins via
  the Kaptive Locibase, then predict phage-host scores.
- `scripts/compute_hrk_phl.py` + `compute_hrk_phl_logocv.py` — compute
  HR@k metrics from PhageHostLearn prediction matrices using cipher's
  HR@k convention.
- `scripts/replicate_logocv*.py` — honest LOGO-CV evaluation on the PHL
  panel itself (200 folds, leave-one-K-out, pooled ROC AUC).
- `config/` — env-style config templates (laptop / Delta / Biowulf).

## What this repo does NOT contain

- PhageHostLearn source (`git clone` upstream — see [SETUP.md](SETUP.md))
- ESM-2 650M model weights (downloaded by `transformers` at first run)
- The trained XGBoost model `phagehostlearn_esm2_xgb.json` (ships with
  upstream)
- Cipher RBP FASTAs or per-dataset Locibase JSONs (these come from cipher
  itself; path is set via `CIPHER_REPO` / `CIPHER_VAL_GENOMES` in the
  env file)

## Two evaluation modes

| mode | datasets | script | notes |
|---|---|---|---|
| LOGO-CV (honest, leak-aware) | PHL | `replicate_logocv*.py` | PHL is PhageHostLearn's training set, so train-test leakage is inevitable; LOGO-CV at the host-K level removes that. |
| Naive OOD | CHEN, GORODNICHIV, UCSD, PBIP, Wang | `run_<ds>_inference.py` | Bundled XGBoost classifier on cipher's other K. pneumoniae sets — these strains/phages are not in PhageHostLearn's training data. |

GORODNICHIV uses the KL23-reference workaround (no host genomes
available); result is tie-saturated.

## Quick start

```bash
git clone https://github.com/LeAnnMLindsey/ciPHer-bench-phagehostlearn.git
cd ciPHer-bench-phagehostlearn

# Pick the env config for your machine:
cp config/phagehostlearn.env.template phagehostlearn.env  # laptop
# or:
# cp config/phagehostlearn_delta.env phagehostlearn.env
# cp config/phagehostlearn_biowulf.env phagehostlearn.env

pico phagehostlearn.env             # edit the four paths
source phagehostlearn.env

# See SETUP.md for upstream clone + env install
python scripts/run_pbip_inference.py
```

See [SETUP.md](SETUP.md) for full setup.

## Citation

If you use this wrapper, please cite both:
- Boeckaerts D, et al. *Prediction of Klebsiella phage-host specificity
  at the strain level.* Nature Communications 15:4226 (2024).
  https://doi.org/10.1038/s41467-024-48675-6
- (manuscript in prep) ciPHer benchmarking paper, LeAnn M. Lindsey et al.
