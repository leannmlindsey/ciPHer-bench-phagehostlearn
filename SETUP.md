# Setup + reproduce

## 1. Clone upstream PhageHostLearn

```bash
# Wherever PHL_REPO in your env points:
git clone https://github.com/dimiboeckaerts/PhageHostLearn.git "$PHL_REPO"
```

The upstream repo ships the trained XGBoost model at
`code/phagehostlearn_esm2_xgb.json`.

## 2. Build the conda env

```bash
conda create -n phagehostlearn python=3.10 -y
conda activate phagehostlearn
pip install torch transformers xgboost biopython pandas tqdm scikit-learn numpy
```

ESM-2 650M (`facebook/esm2_t33_650M_UR50D`, 2.5 GB) is downloaded
automatically by `transformers` the first time you run inference.

## 3. Configure paths

```bash
# Pick the variant for your machine:
cp config/phagehostlearn.env.template phagehostlearn.env   # laptop
# or:
cp config/phagehostlearn_delta.env    phagehostlearn.env   # NCSA Delta
cp config/phagehostlearn_biowulf.env  phagehostlearn.env   # NIH Biowulf

# Edit + source:
pico phagehostlearn.env
source phagehostlearn.env
```

Verify the four env vars are set:
```bash
echo "PHL_REPO=$PHL_REPO"
echo "CIPHER_REPO=$CIPHER_REPO"
echo "CIPHER_VAL_GENOMES=$CIPHER_VAL_GENOMES"
echo "PHL_OUTPUT_ROOT=$PHL_OUTPUT_ROOT"
ls "$PHL_REPO/code/phagehostlearn_esm2_xgb.json"   # should exist
```

## 4. Run on cipher OOD validation sets

```bash
source phagehostlearn.env
python scripts/run_chen_inference.py
python scripts/run_pbip_inference.py
python scripts/run_ucsd_inference.py
python scripts/run_gorodnichiv_inference.py     # KL23-reference workaround
python scripts/run_wang_inference.py            # in-distribution-ish — flag in manuscript
```

Each script writes to `$PHL_OUTPUT_ROOT/<dataset>/prediction_scores.csv`.

## 5. LOGO-CV on PHL

```bash
source phagehostlearn.env
python scripts/replicate_logocv.py
python scripts/compute_hrk_phl_logocv.py
```

## 6. Compute HR@k

```bash
python scripts/compute_hrk_phl.py <dataset>
```
