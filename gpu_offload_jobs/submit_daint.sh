#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# THE Daint/Alps submitter for the GPU-offloading sweep. One job per corpus.
#
#   ./submit_daint.sh smoke              # debug partition, one corpus, preset S, ~15 min. RUN FIRST.
#   ./submit_daint.sh cpu all            # the CPU job: offload + validate + emit, all 281 kernels
#   ./submit_daint.sh gpu poly           # the GPU job: build, run against the reference, time both arms
#   ./submit_daint.sh gpu all 2          # all four corpora, 2 nodes each
#
# TWO jobs, because they answer different questions at very different cost:
#
#   cpu  -- offloads every kernel at both settings of optimizer.gpu_taskloop_heuristics, validates
#           the graph and emits it. No compiler, no device, seconds per kernel. This is the
#           correctness sweep AND the screen that decides which kernels the GPU job must time twice.
#   gpu  -- compiles, runs against the corpus reference, and times both arms. Minutes per kernel.
#
# A Grace-Hopper node is four modules, each a 72-core Grace CPU with its OWN H100, so one node is
# four ranks and four GPUs with nothing shared. gpu_offload_job.py binds each rank to the GPU of its
# own module; that binding is what makes the GPU job's numbers mean anything, since four ranks
# timing kernels on one card measure each other.
#
# ⛔ The GPU job's timings are only comparable WITHIN a rank. Arms are built and run minutes apart,
# so anything that moves the clock between them (a power cap, a neighbour job on the same module)
# lands on the ratio. The driver already discards warm-up calls and reports the best of N for this
# reason, and the kernels whose two arms compile the same graph are left in deliberately: their
# ratios are the noise floor every other row has to be read against.
#
# Env: PRESET (default paper), TIMELIMIT, ACCOUNT (g34), PARTITION (normal), OUT_DIR,
#      REPEATS (default 20 timed invocations per arm).

set -euo pipefail

KIND="${1:-}"
WHAT="${2:-all}"
NODES="${3:-1}"
# Whatever follows the three positionals goes straight to the driver, so picking a different pair
# of arms (--arms autoopt,canon) is a flag rather than an edit to any script in this folder.
[ $# -gt 0 ] && shift $(( $# < 3 ? $# : 3 ))
PASSTHROUGH=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
PRESET="${PRESET:-paper}"
# 20 timed invocations per arm. The driver discards warm-up calls and reports the BEST of these,
# so the count buys robustness against the power cap moving mid-arm rather than a tighter average.
REPEATS="${REPEATS:-20}"

case "$KIND" in
    cpu|gpu|smoke) ;;
    *)
        echo "usage: $(basename "$0") <smoke|cpu|gpu> [poly|np|tsvc|tsvc25|all] [NODES] [driver flags]" >&2
        echo "  cpu    -- offload, validate and emit every kernel; no device needed" >&2
        echo "  gpu    -- compile, verify against the reference, time both knob settings" >&2
        echo "  smoke  -- one corpus at preset S on the debug partition; run this first" >&2
        exit 2
        ;;
esac

case "$WHAT" in
    poly|np|tsvc|tsvc25|all) ;;
    *) echo "unknown corpus '$WHAT'" >&2; exit 2 ;;
esac

submit_one() {
    local kind=$1 suite=$2 nodes=$3 stage deftime extra
    if [ "$kind" = cpu ]; then
        stage=codegen
        extra=()
        # Emitting is seconds a kernel, but canonicalizing the paper shapes is not, and tsvc is 151
        # of them.
        case "$suite" in
            tsvc) deftime=06:00:00 ;;
            *)    deftime=03:00:00 ;;
        esac
    else
        stage=full
        # --no-screen off: the screen is the point. A kernel whose two arms offload to the same
        # graph is built and timed once, which is what makes the sweep finish -- the knob is inert
        # on most of the corpus, and a CUDA build is minutes.
        extra=(--repeats "$REPEATS")
        case "$suite" in
            tsvc)   deftime=12:00:00 ;;
            poly)   deftime=08:00:00 ;;
            *)      deftime=06:00:00 ;;
        esac
    fi
    local out="${OUT_DIR:-$HERE/results_${kind}_${suite}_${PRESET}}"
    mkdir -p "$out"
    sbatch --job-name="dace-offload-${kind}-${suite}" \
        --account="${ACCOUNT:-g34}" \
        --partition="${PARTITION:-normal}" \
        --exclusive \
        --time="${TIMELIMIT:-$deftime}" \
        --nodes="$nodes" \
        --ntasks-per-node=4 \
        --gpus-per-task=1 \
        --cpus-per-task=72 \
        --output="$out/dace-offload-${kind}-${suite}-%j.out" \
        --error="$out/dace-offload-${kind}-${suite}-%j.out" \
        "$HERE/submit_gpu_offload.sh" --out "$out" --stage "$stage" --preset "$PRESET" \
        --suites "$suite" "${extra[@]}" "${PASSTHROUGH[@]}"
}

SUITES=("$WHAT")
[ "$WHAT" = all ] && SUITES=(poly np tsvc tsvc25)

if [ "$KIND" = smoke ]; then
    # One corpus, preset S, debug partition. Proves the rank binds a GPU, finds the corpus and
    # writes a csv -- everything that fails in the first minute of a real job.
    PARTITION=debug TIMELIMIT=00:25:00 OUT_DIR="$HERE/results_smoke" PRESET=S REPEATS=3 \
        bash -c "$(printf '%q ' "$HERE/submit_gpu_offload.sh" --out "$HERE/results_smoke" \
            --stage full --preset S --suites poly --repeats 3 "${PASSTHROUGH[@]}")"
else
    for suite in "${SUITES[@]}"; do
        echo "=== submitting $KIND/$suite (${NODES} node(s) x 4 ranks x 1 GPU)"
        submit_one "$KIND" "$suite" "$NODES"
    done
    echo
    echo "watch:  squeue --me"
fi
