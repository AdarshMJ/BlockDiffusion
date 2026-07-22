#!/usr/bin/env bash
# Launch the coarse_planar latent-diffusion training (Stage B), resumable.
#
# oarsub example (adjust walltime / GPU resource spec to your site):
#   oarsub -l gpu=1,walltime=12:00:00 "./run_coarse_train.sh"
#   # resume after a walltime kill (auto-discovers latest checkpoint):
#   oarsub -l gpu=1,walltime=12:00:00 "./run_coarse_train.sh experiments/coarse_planar_YYYYMMDD_HHMMSS"
#
# First run coarsens lazily (~45 min, cached) then trains 1000 epochs.
set -euo pipefail

cd "$(dirname "$0")"           # -> BlockDiffusion/

# --- activate your Python env here (EDIT to match your setup) ---------------
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate blockdiff
# OR:  source /path/to/venv/bin/activate

CONFIG=configs/config_coarse_planar.yaml
EXP_DIR="${1:-}"               # optional: pass an experiment dir to resume

if [[ -n "$EXP_DIR" ]]; then
    echo "Resuming from $EXP_DIR"
    python main.py --config "$CONFIG" --mode train --device cuda --experiment_dir "$EXP_DIR"
else
    echo "Fresh run"
    python main.py --config "$CONFIG" --mode train --device cuda
fi
