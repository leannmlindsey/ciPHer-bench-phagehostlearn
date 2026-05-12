# PhageHostLearn Delta data manifest

What needs to be on Delta before `scripts/run_<dataset>_inference.py`
will succeed.

## Required: upstream PhageHostLearn clone + XGBoost model

```bash
# On Delta:
mkdir -p /projects/bfzj/llindsey1/PHI_TSP/ciPHer-comparisons/phagehostlearn
cd       /projects/bfzj/llindsey1/PHI_TSP/ciPHer-comparisons/phagehostlearn
git clone https://github.com/dimiboeckaerts/PhageHostLearn.git
```

The upstream `code/phagehostlearn_esm2_xgb.json` ships with the clone.

## Required: cipher already on Delta

`CIPHER_REPO=/projects/bfzj/llindsey1/PHI_TSP/ciPHer` — assumed already
present per cipher.env's canonical layout. Contains:
- `data/validation_data/metadata/validation_rbps_all.faa`
- `data/validation_data/HOST_RANGE/<DS>/metadata/{interaction_matrix.tsv,phage_protein_mapping.csv}`

## Required: cipher per-dataset artifacts (NOT in cipher repo)

The PHL wrappers consume `${CIPHER_VAL_GENOMES}/<DS>/kaptive_out/Locibase.json`
for each dataset. These live ONLY on the laptop right now.

| Dataset | Locibase.json status (laptop) | Notes |
|---|---|---|
| CHEN          | **MISSING**     | run Kaptive on CHEN host genomes first |
| GORODNICHIV   | **MISSING**     | (uses KL23-reference workaround — see run_gorodnichiv_inference.py) |
| PBIP          | 924 KB on laptop | needs transfer |
| UCSD          | 788 KB on laptop | needs transfer |
| PhageHostLearn (PHL) | **MISSING** | run Kaptive on PHL host genomes |
| Beamud        | **MISSING**     | not needed for PHL wrappers |
| Ferriol       | **MISSING**     | not needed for PHL wrappers |
| Wang          | **MISSING**     | run Kaptive on Wang host genomes |

### Action: produce missing Locibase.json files, then upload all

```bash
# (on laptop or on Delta — pick where Kaptive is installed)
# For each missing dataset, run Kaptive against its host genomes
# and capture Locibase.json. See cipher's TropiSEQ pipeline for the
# exact Kaptive invocation.
```

## Required: Boeckaerts Zenodo (only for replicate_logocv*.py)

```bash
# On Delta:
mkdir -p /projects/bfzj/llindsey1/PHI_TSP/ciPHer-comparisons/phagehostlearn/PhageHostLearn/data/zenodo_11061100
cd       /projects/bfzj/llindsey1/PHI_TSP/ciPHer-comparisons/phagehostlearn/PhageHostLearn/data/zenodo_11061100
# Download the Boeckaerts Zenodo deposit zip from
#   https://zenodo.org/records/11061100
# Unpack into 11061100_unpacked/. ~2.1 GB.
```

This is only needed if you plan to run the honest LOGO-CV evaluation
on the PHL panel itself (`replicate_logocv*.py`, `run_phl_inference.py`).

## Required for run_wang_inference.py only

Wang's pre-computed phage protein embeddings pickle (117 MB):
```
/Users/leannmlindsey/WORK/cipher_data/validation_genomes/Wang/phage_protein_ts_prediction_and_esm_embedding.pkl
```

Transfer to Delta:
```
/projects/bfzj/llindsey1/PHI_TSP/cipher_data/validation_genomes/Wang/phage_protein_ts_prediction_and_esm_embedding.pkl
```

## SLURM template

These scripts download ESM-2 650M (2.5 GB) on first run and then do
batched embeddings. **GPU** speeds this up dramatically (CPU would take
hours per dataset). Use cipher's GPU partition pattern:

```bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${PHL_CONDA_ENV}
source phagehostlearn.env

python scripts/run_<dataset>_inference.py
```
