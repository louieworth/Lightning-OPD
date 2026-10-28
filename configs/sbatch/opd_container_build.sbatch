#!/usr/bin/env bash
#SBATCH --job-name=opd-image-build
#SBATCH --account=aip-plbacon
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
cd "${LIGHTNING_OPD_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
test -d "${SLURM_TMPDIR:?Compute-node local temporary directory is required}"
exec /usr/bin/python3 -I -S -B containers/launch.py build --runtime apptainer --base-only
