#!/bin/bash -l
#SBATCH --job-name=one_shot_rlvr_pi1_4xa100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/woody/iwi7/iwi7107h/One shot RLVR GIT/One-Shot-RLVR/outputs/one_shot_rlvr_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/One shot RLVR GIT/One-Shot-RLVR/outputs/one_shot_rlvr_%j.err
set -euo pipefail

REPO_DIR="/home/woody/iwi7/iwi7107h/One shot RLVR GIT/One-Shot-RLVR"
TRAIN_SCRIPT_REL="scripts/train/training_1.5b_pi1_r128.sh"
CONDA_ENV_NAME="rlvr"
MODEL_PATH="/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B/"

# Fixed across all job runs so verl resume_mode=auto finds previous checkpoints
CKPT_DIR="${REPO_DIR}/outputs/checkpoints"

# Per-job log directory
LOG_DIR="${REPO_DIR}/outputs/results${SLURM_JOB_ID}/logs"
PATCHED_SCRIPT="${REPO_DIR}/outputs/results${SLURM_JOB_ID}/training_1.5b_pi1_r128_4xa100.sh"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Keep a copy of live stdout/stderr inside per-job log folder
exec > >(tee -a "${LOG_DIR}/train_stdout.log")
exec 2> >(tee -a "${LOG_DIR}/train_stderr.log" >&2)

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="$CKPT_DIR"

# Activate conda - detect base path at runtime
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Could not determine conda base path. Is conda in PATH?"
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_DIR"

if [ ! -f "$TRAIN_SCRIPT_REL" ]; then
    echo "Training script not found: $REPO_DIR/$TRAIN_SCRIPT_REL"
    exit 1
fi

echo "=================================================="
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Repo dir: $REPO_DIR"
echo "Model path: $MODEL_PATH"
echo "Checkpoint dir (fixed): $CKPT_DIR"
echo "Log dir (per-job): $LOG_DIR"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "=================================================="
nvidia-smi

# Show existing checkpoints so it's clear if this is a resume
echo "Existing checkpoints:"
find "$CKPT_DIR" -maxdepth 3 -name "latest_checkpointed_iteration.txt" -exec cat {} \; 2>/dev/null || echo "  none (fresh start)"

# Copy original training script into this job's log folder
cp "$TRAIN_SCRIPT_REL" "$PATCHED_SCRIPT"

# 1) 8 GPUs -> 4 GPUs
sed -i "s/trainer\.n_gpus_per_node=8/trainer.n_gpus_per_node=4/g" "$PATCHED_SCRIPT"

# 2) Distinct experiment name
sed -i "s/trainer\.experiment_name='Qwen2\.5-Math-1\.5B-pi1_r128'/trainer.experiment_name='Qwen2.5-Math-1.5B-pi1_r128_4xa100'/g" "$PATCHED_SCRIPT"

# 3) Use local model path
sed -i "s|actor_rollout_ref\.model\.path='Qwen/Qwen2.5-Math-1.5B'|actor_rollout_ref.model.path='${MODEL_PATH}'|g" "$PATCHED_SCRIPT"

# 4) Replace logger with console only (tensorboard not supported by verl)
sed -i "s/trainer\.logger=\['console','wandb'\]/trainer.logger=['console']/g" "$PATCHED_SCRIPT"
sed -i "s/trainer\.logger=\['wandb'\]/trainer.logger=['console']/g" "$PATCHED_SCRIPT"
sed -i "s/trainer\.logger=\[\"console\",\"wandb\"\]/trainer.logger=['console']/g" "$PATCHED_SCRIPT"
sed -i "s/trainer\.logger=\[\"wandb\"\]/trainer.logger=['console']/g" "$PATCHED_SCRIPT"

# 5) Fix verl_demo.log to write inside LOG_DIR instead of repo root
sed -i "s|tee verl_demo\.log|tee '${LOG_DIR}/verl_demo.log'|g" "$PATCHED_SCRIPT"

# 6) Replace project_name (original has 'verl_few_shot', always replace)
sed -i "s/trainer\.project_name='verl_few_shot'/trainer.project_name='one_shot_rlvr'/g" "$PATCHED_SCRIPT"
sed -i "s/trainer\.project_name=\"verl_few_shot\"/trainer.project_name='one_shot_rlvr'/g" "$PATCHED_SCRIPT"

# 7) Point checkpoints_dir to fixed path so resume_mode=auto works across jobs
sed -i "s|trainer\.checkpoints_dir=\$CHECKPOINTS_DIR|trainer.checkpoints_dir='${CKPT_DIR}'|g" "$PATCHED_SCRIPT"

# 8) Append any still-missing overrides
if ! grep -q "trainer\.logger=" "$PATCHED_SCRIPT"; then
    sed -i '$ s|$| \\\n    trainer.logger=['\''console'\'']|' "$PATCHED_SCRIPT"
fi
if ! grep -q "trainer\.project_name=" "$PATCHED_SCRIPT"; then
    sed -i '$ s|$| \\\n    trainer.project_name='\''one_shot_rlvr'\''|' "$PATCHED_SCRIPT"
fi

echo
echo "Patched config lines:"
grep -E "trainer\.n_gpus_per_node=|trainer\.experiment_name=|actor_rollout_ref\.model\.path=|trainer\.logger=|trainer\.checkpoints_dir=|trainer\.project_name=" "$PATCHED_SCRIPT" || true
echo
echo "Patched script saved to: $PATCHED_SCRIPT"
echo
echo "Launching training..."
bash "$PATCHED_SCRIPT"
echo
echo "Training finished at: $(date)"
echo
echo "Contents of log dir:"
find "$LOG_DIR" -maxdepth 1 | sort
