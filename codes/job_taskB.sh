#!/bin/bash -l
#$ -S /bin/bash
#$ -N taskB_timepred
#$ -cwd
#$ -l gpu=1
#$ -l h_rt=12:00:00
#$ -l mem=16G
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log

set -eo pipefail

mkdir -p logs

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "PWD: $PWD"
echo "TMPDIR: ${TMPDIR:-}"

module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# conda activate can touch unset vars (MKL). avoid -u during activation.
set +u
conda activate mphy0043-pt
set -u

cd /myriadfs/home/rmaphyo/Scratch/surgery_time_project

python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# -------------------------
# Task B (tool detection) + precomputed timepred
# -------------------------
SCRIPT="train_taskB_timepred.py"   # <- 너가 만든 파일명으로 맞춰줘
EPOCHS=8
OUT_DIR="/myriadfs/home/rmaphyo/Scratch/surgery_time_project/runs_taskB/timepred_feat_u_u2_tnorm"
LAST="${OUT_DIR}/last.pt"

CHOLEC80_DIR="/myriadfs/home/rmaphyo/Scratch/cholec80_data/cholec80"
LABELS_DIR="/myriadfs/home/rmaphyo/Scratch/cholec80_data/labels_1fps"
SPLIT_JSON="/myriadfs/home/rmaphyo/Scratch/cholec80_data/labels_1fps/split_default_60_10_10.json"

# timepred root (expects: {root}/train/{vid}.pt and {root}/val/{vid}.pt)
TIMEPRED_ROOT="/myriadfs/home/rmaphyo/Scratch/surgery_time_project/timepreds_feat_u_u2_tnorm"

# train settings
BATCH=64
LR=1e-3
WD=1e-4
NW=4
STRIDE_TR=2
STRIDE_VA=5
TIME_EMB=64

echo "Target total epochs: ${EPOCHS}"
echo "Out dir: ${OUT_DIR}"
echo "Timepred root: ${TIMEPRED_ROOT}"

COMMON_ARGS=(
  --cholec80_dir "${CHOLEC80_DIR}"
  --labels_dir "${LABELS_DIR}"
  --split_json "${SPLIT_JSON}"
  --out_dir "${OUT_DIR}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH}"
  --lr "${LR}"
  --weight_decay "${WD}"
  --num_workers "${NW}"
  --stride_train "${STRIDE_TR}"
  --stride_val "${STRIDE_VA}"
  --use_timepred
  --timepred_root "${TIMEPRED_ROOT}"
  --time_emb_dim "${TIME_EMB}"
  --freeze_backbone
)

if [ -f "${LAST}" ]; then
  echo "Found checkpoint: ${LAST}"
  echo "=> Resuming training"
  python -u "${SCRIPT}" "${COMMON_ARGS[@]}" --resume "${LAST}"
else
  echo "No checkpoint found at ${LAST}"
  echo "=> Starting fresh training"
  python -u "${SCRIPT}" "${COMMON_ARGS[@]}"
fi

echo "Job finished: $(date)"
