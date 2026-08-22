#!/bin/bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Daint/Alps: all FOUR in-repo corpora as four independent jobs, one node (4 ranks) each.
#
# Four jobs, not one four-node job: they are independent measurements, so they queue separately and
# a failure or a time-limit in one costs nothing to the other three. Same reason each keeps its own
# results directory.
#
# Usage:
#   bash canon_corpus_perf_job/submit_daint_four.sh              # 1 node per corpus
#   bash canon_corpus_perf_job/submit_daint_four.sh 2            # 2 nodes per corpus
#   PRESET=S bash canon_corpus_perf_job/submit_daint_four.sh     # smoke-sized datasets
#
# Env overrides are forwarded as-is; see submit_daint_corpus.sh (which carries the SIGCHLD/cmake
# note that governs how these jobs must launch).

set -euo pipefail

NODES="${1:-1}"
HERE=$(cd "$(dirname "$0")" && pwd)

for suite in poly np tsvc tsvc25; do
    echo "=== submitting ${suite} (${NODES} node(s) x 4 ranks x ${OMP_NUM_THREADS:-72} threads)"
    bash "$HERE/submit_daint_corpus.sh" "$suite" "$NODES"
done

echo
echo "queued. watch with:  squeue --me --name=dace-corpus-poly,dace-corpus-np,dace-corpus-tsvc,dace-corpus-tsvc25"
