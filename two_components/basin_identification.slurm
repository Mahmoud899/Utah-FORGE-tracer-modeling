#!/bin/bash
#SBATCH --chdir=/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper
#SBATCH --job-name=tc_basin
#SBATCH --output=/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper/two_components/logs/basin_identification_%j.out
#SBATCH --error=/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper/two_components/logs/basin_identification_%j.err
#SBATCH --partition=nocona
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --export=ALL

set -euo pipefail

# Run from the project root explicitly; SLURM may execute a spool copy of this script.
REPO_ROOT="/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper"
cd "${REPO_ROOT}"

mkdir -p two_components/logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Repo root: ${REPO_ROOT}"
echo "Start: $(date)"

# Activate the project Conda environment in batch mode.
source /home/mabdella/miniforge3/etc/profile.d/conda.sh
conda activate jupyternotebook

echo "Python: $(which python)"
python --version
python -c 'import numpy, pandas, scipy; print("imports ok")'

python -m two_components.basin_identification

echo "End: $(date)"
