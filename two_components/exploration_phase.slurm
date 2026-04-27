#!/bin/bash
#SBATCH --chdir=/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper
#SBATCH --job-name=tc_explore
#SBATCH --output=/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper/two_components/logs/exploration_phase_%A_%a.out
#SBATCH --error=/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper/two_components/logs/exploration_phase_%A_%a.err
#SBATCH --partition=nocona
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --array=0-15%8
#SBATCH --export=ALL

set -euo pipefail

# Run from the project root explicitly; SLURM may execute a spool copy of this script.
REPO_ROOT="/lustre/work/mabdella/Analytical_Modeling/GeothermicsPaper"
cd "${REPO_ROOT}"

mkdir -p two_components/logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Repo root: ${REPO_ROOT}"
echo "Start: $(date)"

# Adjust these together if you change the total exploration size.
MEXP=1024
SHARD_SIZE=64
CPUS_PER_RUN=8
OUTPUT_DIR="two_components/outputs/phase_exploration"
MERGE_JOB_NAME="tc_explore_merge"
MERGE_CPUS=1
MERGE_MEM=8G
MERGE_TIME=00:30:00

# This array layout assumes MEXP=1024 and SHARD_SIZE=64, which gives 16 shards.
# If you change MEXP or SHARD_SIZE, update the --array line accordingly.

# Activate the project Conda environment in batch mode.
source /home/mabdella/miniforge3/etc/profile.d/conda.sh
conda activate jupyternotebook

echo "Python: $(which python)"
python --version
python -c 'import numpy, pandas, scipy, mpmath; print("imports ok")'

# Submit the merge job once. It will wait for the full array to complete successfully.
submit_merge_job() {
  sbatch --parsable --dependency="afterok:${SLURM_ARRAY_JOB_ID}" <<EOF
#!/bin/bash
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --job-name=${MERGE_JOB_NAME}
#SBATCH --output=${REPO_ROOT}/two_components/logs/exploration_phase_merge_%j.out
#SBATCH --error=${REPO_ROOT}/two_components/logs/exploration_phase_merge_%j.err
#SBATCH --partition=${SLURM_JOB_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${MERGE_CPUS}
#SBATCH --time=${MERGE_TIME}
#SBATCH --mem=${MERGE_MEM}
#SBATCH --export=ALL

set -euo pipefail

source /home/mabdella/miniforge3/etc/profile.d/conda.sh
conda activate jupyternotebook

echo "Merge host: \$(hostname)"
echo "Merge start: \$(date)"

python -m two_components.exploration_phase merge-shards \
  --mexp "${MEXP}" \
  --output-dir "${OUTPUT_DIR}"

echo "Merge end: \$(date)"
EOF
}

if [[ "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN:-0}" ]]; then
  MERGE_JOB_ID=$(submit_merge_job)
  echo "Submitted merge job ${MERGE_JOB_ID} with dependency afterok:${SLURM_ARRAY_JOB_ID}"
fi

python -m two_components.exploration_phase run-shard \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --shard-size "${SHARD_SIZE}" \
  --mexp "${MEXP}" \
  --seed-start 101 \
  --cpus-per-run "${CPUS_PER_RUN}" \
  --output-dir "${OUTPUT_DIR}"

echo "End: $(date)"
echo "Shard task ${SLURM_ARRAY_TASK_ID} finished."
