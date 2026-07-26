#!/bin/bash
#SBATCH --job-name=hpcagent-canon-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8      # kernels are sliced round-robin across ranks (SLURM_PROCID)
#SBATCH --cpus-per-task=1        # canonicalize is single-threaded Python
#SBATCH --time=04:00:00
#SBATCH --output=hpcagent_canon_sweep_%j.out
#SBATCH --error=hpcagent_canon_sweep_%j.err
#
# Canonicalize sweep over the hpcagent_bench (optarena) kernel corpus -- the foundation / hpc /
# ml tracks, EXCLUDING the kernels already swept by submit_canon_corpus_sweep.sbatch's np/poly
# corpora (any kernel tagged 'npbench' or subtrack 'polybench' is the identical kernel, ported;
# see measure_parallelization.py's _hpcagent_names). Same driver, same metrics, same contract --
# see submit_canon_corpus_sweep.sbatch for the npbench/polybench/tsvc/tsvc_2_5 corpora.
#
# Measures, per kernel: residual sequential loops / parallel maps / Reduce / Scan / STATES,
# at three points (untransformed base, LoopToMap-only, full canonicalize), and records any
# kernel whose pipeline raises. The state count is the fusion metric: everything fusable
# fused == minimal states.
#
# Exit status is the number of errored kernels, so a corpus that stops canonicalizing
# fails the job rather than scrolling past in the log.
#
#   sbatch submit_hpcagent_canon_sweep.sbatch                  # canon, hpcagent corpus
#   CONFIG=all ./submit_hpcagent_canon_sweep.sbatch            # also canon+vec, parallelize+vec
#   CHECK=1 RANKS=4 ./submit_hpcagent_canon_sweep.sbatch       # + correctness gate; runs fine without slurm too
#
# HPCAGENT_BENCH_REPO must point at the hpcagent_bench (optarena) checkout -- its editable pip
# install is not guaranteed to exist in every job environment, so PYTHONPATH sets it explicitly,
# the same defensive reason the dace REPO below is on PYTHONPATH rather than relying on its pip
# install.
set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)}"
HPCAGENT_BENCH_REPO="${HPCAGENT_BENCH_REPO:-$HOME/Work/optarena}"
CORPUS=hpcagent
CONFIG="${CONFIG:-canon}"
PEEL="${PEEL:-4}"
# CHECK=1 also compiles + runs each kernel and asserts the result matches the kernel's numpy
# reference (hpcagent_bench's own DaceFramework.verify -- see _hpcagent_case). Canonicalization
# is value-preserving, so a mismatch is a miscompile -- structural counts alone cannot see one.
CHECK="${CHECK:-}"
CHECK_FLAG=""
[[ -n "$CHECK" ]] && CHECK_FLAG="--check"
RANKS="${RANKS:-${SLURM_NTASKS:-8}}"
WORK="${WORK:-$HOME/.cache/dace_sweep/hpcagent_corpus}"   # on disk: /tmp is a RAM disk on some hosts
RESULTS="$WORK/results"
mkdir -p "$RESULTS"
rm -f "$RESULTS"/hpcagent.*.csv

PYPATH="$REPO:$HPCAGENT_BENCH_REPO"

# Single-threaded, no MPI probe, no UCX filesystem scan -- the sweep is pure Python
# transformation work and any of those will either flake or serialise the ranks. Also required
# ad-hoc (not just inside measure_parallelization.py's own os.environ.setdefault guards): a bare
# `python3 -c` probe without these hung waiting on an MPI/UCX socket peer during development.
export OMP_NUM_THREADS=1 MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n

shard() {  # shard <index>
    local i="$1"
    # Each rank needs its own build folder: the default is shared and the ranks race on it.
    PYTHONPATH="$PYPATH" DACE_default_build_folder="$WORK/dacecache-$i" \
        python3 -m tests.corpus.measure_parallelization "$CORPUS" \
            --config "$CONFIG" --peel "$PEEL" --shard "$i/$RANKS" --csv "$RESULTS/hpcagent.$i.csv" $CHECK_FLAG \
            > "$RESULTS/hpcagent.$i.log" 2>&1
}

if command -v srun >/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="$RANKS" --cpus-per-task=1 bash -c \
        'PYTHONPATH="'"$PYPATH"'" DACE_default_build_folder="'"$WORK"'/dacecache-$SLURM_PROCID" \
         python3 -m tests.corpus.measure_parallelization "'"$CORPUS"'" \
             --config "'"$CONFIG"'" --peel "'"$PEEL"'" --shard "$SLURM_PROCID/'"$RANKS"'" \
             --csv "'"$RESULTS"'/hpcagent.$SLURM_PROCID.csv" '"$CHECK_FLAG"' \
             > "'"$RESULTS"'/hpcagent.$SLURM_PROCID.log" 2>&1'
else
    for ((i = 0; i < RANKS; i++)); do shard "$i" & done
    wait
fi

echo "raw results in $RESULTS"
PYTHONPATH="$PYPATH" python3 -m tests.corpus.measure_parallelization --summarize "$RESULTS"/hpcagent.*.csv
