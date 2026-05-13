# Setup + reproduce

## Workflow at a glance

```text
1. (on laptop) build data zip      → ciPHer-bench-phagehostlearn-data.zip
2. (on laptop) rsync zip to Delta
3. (on Delta) unzip into data/
4. (on Delta) source phagehostlearn.env, build conda env, run wrappers
```

## 1. Build the data zip on the laptop

```bash
cd /Users/leannmlindsey/Desktop/ciPHer-bench-staging/ciPHer-bench-phagehostlearn
bash build_data_zip.sh
# Output: /Users/leannmlindsey/Desktop/ciPHer-bench-data-zips/ciPHer-bench-phagehostlearn-data.zip
```

Layout mirrors the laptop tree (so env vars work unchanged):
- `data/PhageHostLearn/` — upstream Boeckaerts clone + bundled XGB model
- `data/cipher/data/validation_data/...` — cipher's metadata mirror
- `data/cipher_val_genomes/<DS>/kaptive_out/Locibase.json` — per-dataset
  Kaptive K-locus protein dumps **(only PBIP + UCSD currently present;
  CHEN/GORODNICHIV/PHL/Beamud/Ferriol/Wang need Kaptive run first)**
- `data/cipher_val_genomes/Wang/phage_protein_ts_prediction_and_esm_embedding.pkl`
  — only needed for `run_wang_inference.py`

## 2. Transfer + unzip on Delta

```bash
# On laptop:
ZIP=/Users/leannmlindsey/Desktop/ciPHer-bench-data-zips/ciPHer-bench-phagehostlearn-data.zip
rsync -avz --info=progress2 "${ZIP}" \
    llindsey1@dt-login.delta.ncsa.illinois.edu:/projects/bfzj/llindsey1/PHI_TSP/ciPHer-comparisons/phagehostlearn/data/

# On Delta:
ssh llindsey1@dt-login.delta.ncsa.illinois.edu
cd /projects/bfzj/llindsey1/PHI_TSP/ciPHer-comparisons/phagehostlearn

# First time only:
git clone git@github.com:LeAnnMLindsey/ciPHer-bench-phagehostlearn.git .

cd data
unzip -q ciPHer-bench-phagehostlearn-data.zip
cd ..
```

## 3. Build the conda env (one-time)

```bash
# Make sure conda is installed (Miniforge under project space is fine):
#   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh
#   bash Miniforge3-Linux-$(uname -m).sh -b -p /projects/bfzj/llindsey1/miniforge3
#   source /projects/bfzj/llindsey1/miniforge3/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate phagehostlearn
```

ESM-2 650M (~2.5 GB) is auto-downloaded by `transformers` on first run.

## 4. Configure paths + run

```bash
cp config/phagehostlearn_delta.env phagehostlearn.env
source phagehostlearn.env

# Verify paths:
echo "PHL_REPO=${PHL_REPO}"
echo "CIPHER_REPO=${CIPHER_REPO}"
echo "CIPHER_VAL_GENOMES=${CIPHER_VAL_GENOMES}"
ls "${PHL_REPO}/code/phagehostlearn_esm2_xgb.json"   # should exist
```

### Run on a single dataset

```bash
python scripts/run_pbip_inference.py
python scripts/run_ucsd_inference.py
# CHEN / GORODNICHIV / PHL / Wang need their Locibase.json first — see below.
```

Or wrap in sbatch:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=phl_pbip
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
source $(conda info --base)/etc/profile.d/conda.sh
conda activate phagehostlearn
source phagehostlearn.env
python scripts/run_pbip_inference.py
```

Submit:
```bash
sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
       --gpus-per-node="${GPUS_PER_NODE}" <my_sbatch>.sh
```

## 5. Locibase.json prerequisite for the other datasets

`run_chen_inference.py`, `run_gorodnichiv_inference.py`,
`run_phl_inference.py`, `run_wang_inference.py` all need
`${CIPHER_VAL_GENOMES}/<DS>/kaptive_out/Locibase.json`. Currently
only PBIP + UCSD have this on the laptop.

To generate the missing ones, run Kaptive on each dataset's host
genomes. See cipher's TropiSEQ pipeline notes for the canonical
Kaptive invocation. Once Locibase.json exists for the missing datasets
on the laptop, rebuild + re-transfer the zip.
