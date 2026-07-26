#!/bin/bash
#SBATCH --job-name=canon-corpus-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8      # kernels are sliced round-robin across ranks (SLURM_PROCID)
#SBATCH --cpus-per-task=1        # canonicalize is single-threaded Python
#SBATCH --time=04:00:00
#SBATCH --output=canon_corpus_sweep_%j.out
#SBATCH --error=canon_corpus_sweep_%j.err
#
# Canonicalize sweep over the four corpora -- npbench, polybench, tsvc, tsvc_2_5.
#
# Measures, per kernel: residual sequential loops / parallel maps / Reduce / Scan / STATES,
# at three points (untransformed base, LoopToMap-only, full canonicalize), and records any
# kernel whose pipeline raises. The state count is the fusion metric: everything fusable
# fused == minimal states.
#
# Exit status is the number of errored kernels, so a corpus that stops canonicalizing
# fails the job rather than scrolling past in the log.
#
#   sbatch submit_canon_corpus_sweep.sbatch               # canon, all four corpora
#   CONFIG=all ./submit_canon_corpus_sweep.sbatch         # also canon+vec, parallelize+vec
#   CORPUS=np RANKS=4 ./submit_canon_corpus_sweep.sbatch  # one corpus; runs fine without slurm too
#
# CORPUS is one of np / poly / tsvc / tsvc25 / all.
set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)}"
CORPUS="${CORPUS:-all}"
CONFIG="${CONFIG:-canon}"
PEEL="${PEEL:-4}"
# CHECK=1 also compiles + runs each kernel and asserts the result matches the corpus
# reference. Canonicalization is value-preserving, so a mismatch is a miscompile --
# structural counts alone cannot see one (durbin canonicalizes cleanly and answers wrong).
CHECK="${CHECK:-}"
CHECK_FLAG=""
[[ -n "$CHECK" ]] && CHECK_FLAG="--check"
RANKS="${RANKS:-${SLURM_NTASKS:-8}}"
WORK="${WORK:-$HOME/.cache/dace_sweep/canon_corpus}"   # on disk: /tmp is a RAM disk on some hosts
RESULTS="$WORK/results"
mkdir -p "$RESULTS"
rm -f "$RESULTS"/canon.*.csv

# Single-threaded, no MPI probe, no UCX filesystem scan -- the sweep is pure Python
# transformation work and any of those will either flake or serialise the ranks.
export OMP_NUM_THREADS=1 MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n

shard() {  # shard <index>
    local i="$1"
    # Each rank needs its own build folder: the default is shared and the ranks race on it.
    PYTHONPATH="$REPO" DACE_default_build_folder="$WORK/dacecache-$i" \
        python3 -m tests.corpus.measure_parallelization "$CORPUS" \
            --config "$CONFIG" --peel "$PEEL" --shard "$i/$RANKS" --csv "$RESULTS/canon.$i.csv" $CHECK_FLAG \
            > "$RESULTS/canon.$i.log" 2>&1
}

if command -v srun >/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="$RANKS" --cpus-per-task=1 bash -c \
        'PYTHONPATH="'"$REPO"'" DACE_default_build_folder="'"$WORK"'/dacecache-$SLURM_PROCID" \
         python3 -m tests.corpus.measure_parallelization "'"$CORPUS"'" \
             --config "'"$CONFIG"'" --peel "'"$PEEL"'" --shard "$SLURM_PROCID/'"$RANKS"'" \
             --csv "'"$RESULTS"'/canon.$SLURM_PROCID.csv" '"$CHECK_FLAG"' \
             > "'"$RESULTS"'/canon.$SLURM_PROCID.log" 2>&1'
else
    for ((i = 0; i < RANKS; i++)); do shard "$i" & done
    wait
fi

echo "raw results in $RESULTS"
PYTHONPATH="$REPO" python3 -m tests.corpus.measure_parallelization --summarize "$RESULTS"/canon.*.csv
