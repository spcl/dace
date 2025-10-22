#!/bin/bash
#SBATCH --job-name=softhier_job         # Job name
#SBATCH --output=softhier_%j.out         # Standard output file (job ID will be inserted)
#SBATCH --error=softhier_%j.err          # Standard error file (job ID will be inserted)
#SBATCH --ntasks=2                     # Run a single task (job)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task (this value will be read by the Python script)
#SBATCH --time=00:02:00                # Total run time limit (hh:mm:ss)
#SBATCH --mem=50G                     # Total memory required per node
#SBATCH --partition=normal             # Partition name (modify if needed)

# Export the number of processes from SLURM to an environment variable for the Python script
export MAX_PROCS=${SLURM_CPUS_PER_TASK}
export SOFTHIER_TEMP_RUN_PATH=${SCRATCH}/${SLURM_JOB_ID}

mkdir ${SOFTHIER_TEMP_RUN_PATH}

CUR_DIR=$(pwd)

# Change to the directory containing your Python script
source /users/ashen/dace4softhier/.venv/bin/activate 

# Run the Python script
python mat_add.py

rm -rf ${SOFTHIER_TEMP_RUN_PATH}