#!/bin/bash

set -e

# Resolve repository root relative to this script
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SCRIPT=${ROOT_DIR}/script/MIL_bootstrap_multi_trait_260714.py
GENO_FILE=${ROOT_DIR}/data/clinical_phenotype_analysis/MIL/prepeare/geno_feature.csv
PHENO_FILE=${ROOT_DIR}/data/clinical_phenotype_analysis/MIL/prepeare/nor_pheno_by_Age_Gender_CCI.csv
TRAIT_FILE=${ROOT_DIR}/data/clinical_phenotype_analysis/MIL/traits.txt
OUT_DIR=${ROOT_DIR}/output/multi_traits_results
LOG_DIR=${OUT_DIR}/logs

mkdir -p ${LOG_DIR}

MAX_JOBS=6
GPU_ID=0

N_BOOT=100
N_FULL=10
EPOCHS=150
BASE_SEED=42
R_THRESHOLD=0

# For speed, Phase1 model weights are NOT saved by default; only Phase1 attention and prediction are saved.
# To save Phase1 models, set SAVE_PHASE1_MODEL to 1, but this will significantly increase I/O and runtime.
SAVE_PHASE1_MODEL=1
SAVE_PHASE1_ATTENTION=1
SAVE_PHASE1_PRED=1

SAVE_PHASE2_MODEL=1
SAVE_PHASE2_ATTENTION=1
SAVE_PHASE2_PRED=1

echo "Start running 38 traits with MAX_JOBS=${MAX_JOBS} on GPU ${GPU_ID}"
echo "Output dir: ${OUT_DIR}"

job_count=0

while IFS= read -r trait || [[ -n "$trait" ]]; do
    [[ -z "$trait" ]] && continue

    trait_safe=$(echo "$trait" | sed 's/[^A-Za-z0-9_.%+-]/_/g')
    log_file=${LOG_DIR}/${trait_safe}.log

    echo "[$(date '+%F %T')] Start trait=${trait}, log=${log_file}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} \
        --trait "${trait}" \
        --geno "${GENO_FILE}" \
        --pheno "${PHENO_FILE}" \
        --output_dir "${OUT_DIR}" \
        --n_boot ${N_BOOT} \
        --n_full ${N_FULL} \
        --epochs ${EPOCHS} \
        --base_seed ${BASE_SEED} \
        --r_threshold ${R_THRESHOLD} \
        --save_phase1_model ${SAVE_PHASE1_MODEL} \
        --save_phase1_attention ${SAVE_PHASE1_ATTENTION} \
        --save_phase1_pred ${SAVE_PHASE1_PRED} \
        --save_phase2_model ${SAVE_PHASE2_MODEL} \
        --save_phase2_attention ${SAVE_PHASE2_ATTENTION} \
        --save_phase2_pred ${SAVE_PHASE2_PRED} \
        > "${log_file}" 2>&1 &

    job_count=$((job_count + 1))

    if (( job_count % MAX_JOBS == 0 )); then
        echo "Waiting for current batch of ${MAX_JOBS} jobs..."
        wait
        echo "Batch finished."
    fi

done < "${TRAIT_FILE}"

wait

echo "All trait jobs finished."

# Merge summaries of all traits
python - <<EOF
import os, glob
import pandas as pd

summary_dir = "${OUT_DIR}/summary_each_trait"
out_file = "${OUT_DIR}/Summary_All_Traits.csv"

files = sorted(glob.glob(os.path.join(summary_dir, "Summary_*.csv")))
dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print("Failed to read", f, e)

if len(dfs) > 0:
    pd.concat(dfs, ignore_index=True).to_csv(out_file, index=False)
    print("Merged summary saved to:", out_file)
    print("N summary files:", len(files))
else:
    print("No summary files found.")
EOF

echo "Done."
