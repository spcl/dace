#!/bin/bash
#SBATCH --job-name=optarena-canon-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8      # kernels are sliced round-robin across ranks (SLURM_PROCID)
#SBATCH --cpus-per-task=1        # one compiled kernel per process; dace_cpu compiles/runs it single-threaded
#SBATCH --time=04:00:00
#SBATCH --output=optarena_canon_sweep_%j.out
#SBATCH --error=optarena_canon_sweep_%j.err
#
# Canonicalize sweep over the FULL HPCAgent-Bench (optarena) kernel set -- foundation + hpc, ~350+
# kernels -- run through OPTARENA'S OWN dace_cpu framework/runner, not the four in-tree DaCe
# corpora tests/corpus/measure_parallelization.py already covers (poly/np/tsvc/tsvc25).
#
# Integration seam: optarena owns kernel discovery + execution (the BenchSpec registry, S/M/L/XL
# presets, the dace_cpu adapter that builds+compiles+verifies each kernel) via
# `hpcagent_bench.cli run-framework`, which now takes --shard/--csv/--summarize
# (hpcagent_bench/support/collect/sweep.py) mirroring measure_parallelization.py's own
# shard/csv/summarize shape one-for-one. This script only contributes what a DaCe corpus sweep
# job always contributes: sharding kernels across ranks, one CSV per rank, merging them, and the
# batch job's exit status. No parallel measurement machinery was written on this side.
#
# dace_cpu's `autoopt` pipeline IS canonicalize + finalize_for_target('cpu')
# (hpcagent_bench/frameworks/dace_framework.py:_pipeline_auto_opt) and is built, compiled, and
# verified against the NumPy reference on EVERY dace_cpu run -- regardless of which of the 3
# scored variants (strict/parallel/autoopt) ends up fastest. Running a kernel through dace_cpu
# always exercises canonicalize; there is no separate "canon-only" toggle to reach for on this
# side the way there is on the DaCe-corpus side.
#
# Each rank runs in ITS OWN scratch cwd (not the optarena checkout): the legacy Test harness
# writes results into `./hpcagent_bench.db` (relative to cwd) on every kernel, and concurrent
# ranks sharing one sqlite file would race writers against each other AND leave sweep churn in
# the shared, uncommitted optarena tree. A per-rank cwd gives each rank its own db and keeps the
# shared tree untouched.
#
# Exit status is the number of CRASHED + MISCOMPILED (failed-validation) kernels, so a track that
# stops compiling or starts answering wrong fails the job instead of scrolling past in the log.
#
#   sbatch submit_optarena_canon_sweep.sbatch                        # foundation + hpc, preset S
#   TRACKS=foundation RANKS=4 ./submit_optarena_canon_sweep.sbatch    # one track; runs fine without slurm too
#   VALIDATE=0 ./submit_optarena_canon_sweep.sbatch                  # skip the NumPy correctness gate
#
# TRACKS is a space-separated list of optarena selectors (track / dwarf / kernel); default is the
# two tracks the task calls out -- foundation (~245 microkernels) and hpc (the dwarf-organized
# full apps) -- deliberately NOT "all", which would also pull in the separate ml track.
set -euo pipefail

OPTARENA="${OPTARENA:-$HOME/Work/optarena}"
DACE_REPO="${DACE_REPO:-$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)}"
TRACKS="${TRACKS:-foundation hpc}"
FRAMEWORK="${FRAMEWORK:-dace_cpu}"
PRESET="${PRESET:-S}"            # this project's default test preset
REPEAT="${REPEAT:-1}"            # a correctness/canonicalize sweep, not a timing sweep -- 1 rep
TIMEOUT="${TIMEOUT:-200}"
VALIDATE="${VALIDATE:-1}"        # verify each kernel's autoopt(=canonicalize) output against NumPy
VALIDATE_FLAG=""
[[ "$VALIDATE" == "0" ]] && VALIDATE_FLAG="--no-validate"
RANKS="${RANKS:-${SLURM_NTASKS:-8}}"
WORK="${WORK:-$HOME/.cache/dace_sweep/optarena_canon}"   # on disk: /tmp is a RAM disk on some hosts
RESULTS="$WORK/results"
mkdir -p "$RESULTS"
rm -f "$RESULTS"/optarena.*.csv

# Single-threaded, no MPI probe, no UCX filesystem scan -- one compiled kernel per process,
# matching the DaCe-corpus sweep's own env.
export OMP_NUM_THREADS=1 MPI4PY_RC_INITIALIZE=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n

shard() {  # shard <track> <slug> <index>
    local track="$1" slug="$2" i="$3"
    # Each rank gets its own DaCe build folder (races on a shared one) AND its own cwd (races the
    # sqlite results db + would leave churn in the shared optarena checkout -- see header note).
    local rundir="$WORK/run-$slug-$i"
    mkdir -p "$rundir"
    (cd "$rundir" && PYTHONPATH="$OPTARENA:$DACE_REPO" DACE_default_build_folder="$WORK/dacecache-$slug-$i" \
        python3 -m hpcagent_bench.cli run-framework \
            --benchmark "$track" --framework "$FRAMEWORK" --preset "$PRESET" --repeat "$REPEAT" \
            --timeout "$TIMEOUT" $VALIDATE_FLAG --shard "$i/$RANKS" \
            --csv "$RESULTS/optarena.$slug.$i.csv") \
        > "$RESULTS/optarena.$slug.$i.log" 2>&1
}

for track in $TRACKS; do
    slug="${track//\//_}"   # a track may be "hpc/<dwarf>" -- flatten before it becomes a filename
    if command -v srun >/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --ntasks="$RANKS" --cpus-per-task=1 bash -c \
            'rundir="'"$WORK"'/run-'"$slug"'-$SLURM_PROCID"; mkdir -p "$rundir"
             (cd "$rundir" && PYTHONPATH="'"$OPTARENA"':'"$DACE_REPO"'" \
                 DACE_default_build_folder="'"$WORK"'/dacecache-'"$slug"'-$SLURM_PROCID" \
                 python3 -m hpcagent_bench.cli run-framework \
                     --benchmark "'"$track"'" --framework "'"$FRAMEWORK"'" --preset "'"$PRESET"'" \
                     --repeat "'"$REPEAT"'" --timeout "'"$TIMEOUT"'" '"$VALIDATE_FLAG"' \
                     --shard "$SLURM_PROCID/'"$RANKS"'" \
                     --csv "'"$RESULTS"'/optarena.'"$slug"'.$SLURM_PROCID.csv") \
                 > "'"$RESULTS"'/optarena.'"$slug"'.$SLURM_PROCID.log" 2>&1'
    else
        for ((i = 0; i < RANKS; i++)); do shard "$track" "$slug" "$i" & done
        wait
    fi
done

echo "raw results in $RESULTS"
PYTHONPATH="$OPTARENA:$DACE_REPO" python3 -m hpcagent_bench.cli run-framework --summarize "$RESULTS"/optarena.*.csv
