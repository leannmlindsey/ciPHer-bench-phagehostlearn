#!/usr/bin/env bash
#SBATCH --job-name=phl_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke.%j.out
#SBATCH --error=logs/smoke.%j.err

# Submit me FIRST. Verifies env, conda, torch+xgboost, ESM-2 download
# pathway, and data layout. 5-10 min.
# Submit:
#   source phagehostlearn.env
#   sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
#          --gpus-per-node="${GPUS_PER_NODE}" delta/00_smoke_test.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/phagehostlearn.env" ]]; then
    echo "FAIL: ${REPO_ROOT}/phagehostlearn.env missing." >&2
    echo "  cp config/phagehostlearn_delta.env phagehostlearn.env first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/phagehostlearn.env"

mkdir -p "${REPO_ROOT}/logs"

echo "============================================================"
echo "PhageHostLearn smoke test"
echo "  hostname    : $(hostname)"
echo "  date        : $(date)"
echo "  cpu arch    : $(uname -m)"
echo "============================================================"

# --- Conda activation
echo
echo "[1/4] Conda activation"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PHL_CONDA_ENV}"
echo "  active env: ${CONDA_DEFAULT_ENV}"
echo "  python:     $(python --version 2>&1)"

# --- Library check
echo
echo "[2/4] Library check"
python <<'PY'
import sys
fails = []
for pkg in ["torch", "transformers", "xgboost", "Bio", "pandas", "numpy", "tqdm", "sklearn"]:
    try:
        m = __import__(pkg if pkg != "Bio" else "Bio")
        ver = getattr(m, "__version__", "n/a")
        print(f"  OK  {pkg:14} {ver}")
    except Exception as e:
        print(f"  FAIL {pkg:14} {e}")
        fails.append(pkg)
if fails:
    sys.exit(2)

# GPU check
import torch
print(f"  CUDA available     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device        : {torch.cuda.get_device_name(0)}")
    print(f"  torch CUDA version : {torch.version.cuda}")
PY

# --- Data layout
echo
echo "[3/4] Data layout"
: "${PHL_DATA_DIR:?PHL_DATA_DIR not set — phagehostlearn.env broken}"
echo "  PHL_DATA_DIR       = ${PHL_DATA_DIR}"
echo "  PHL_REPO           = ${PHL_REPO}"
echo "  CIPHER_REPO        = ${CIPHER_REPO}"
echo "  CIPHER_VAL_GENOMES = ${CIPHER_VAL_GENOMES}"

# Critical files
declare -a checks=(
    "${PHL_REPO}/code/phagehostlearn_esm2_xgb.json"
    "${CIPHER_REPO}/data/validation_data/metadata/validation_rbps_all.faa"
    "${CIPHER_REPO}/data/validation_data/HOST_RANGE/PBIP/metadata/interaction_matrix.tsv"
    "${CIPHER_REPO}/data/validation_data/HOST_RANGE/PBIP/metadata/phage_protein_mapping.csv"
)
for f in "${checks[@]}"; do
    if [[ -e "${f}" ]]; then
        echo "  OK : ${f}"
    else
        echo "  FAIL: missing ${f}" >&2
        exit 3
    fi
done

# Locibase availability per dataset (warn-only)
echo
echo "  per-dataset Locibase.json availability:"
for ds in CHEN GORODNICHIV PBIP UCSD PhageHostLearn Beamud Ferriol Wang; do
    LOCI="${CIPHER_VAL_GENOMES}/${ds}/kaptive_out/Locibase.json"
    if [[ -f "${LOCI}" ]]; then
        echo "    [present]  ${ds}"
    else
        echo "    [missing]  ${ds}  — run_${ds,,}_inference.py will fail without this"
    fi
done

# --- ESM-2 model availability (without downloading)
echo
echo "[4/4] ESM-2 model load (downloads ~2.5 GB on first run — may take a few min)"
python <<'PY'
from transformers import AutoTokenizer, AutoModel
ESM2 = "facebook/esm2_t33_650M_UR50D"
print(f"  loading tokenizer {ESM2}...")
tok = AutoTokenizer.from_pretrained(ESM2)
print(f"  loading model {ESM2}...")
m = AutoModel.from_pretrained(ESM2)
n_params = sum(p.numel() for p in m.parameters())
print(f"  OK — {n_params/1e6:.1f}M params loaded")
PY

echo
echo "All smoke checks passed."
echo "Now you can submit a real inference, e.g.:"
echo "  sbatch ... your_phl_run.sbatch    (calling python scripts/run_pbip_inference.py)"
