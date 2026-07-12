#!/usr/bin/env bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Single-click numerical-verification + speedup table for the canonicalize corpus.
#
# Times auto-opt (baseline) and canon on EVERY polybench+npbench kernel,
# checks each against its numpy reference, and writes a human-readable Markdown table
# of speedups (plus a CSV and one JSON per kernel). Built for a dedicated benchmark
# box: run it once and read perf_results/speedup_table.md.
#
# Resumable: a re-run only measures kernels whose result JSON is missing. Delete a
# kernel's file (or FORCE=1) to re-measure it; ONLY=<substr> to scope the sweep.
#
# Usage (from anywhere -- it cd's to the repo root itself):
#   tests/passes/canonicalize/run_speedup_table.sh                 # full sweep -> table
#   ONLY=gemm tests/passes/canonicalize/run_speedup_table.sh       # only matching kernels
#   FORCE=1   tests/passes/canonicalize/run_speedup_table.sh       # re-measure everything
#   OUT_DIR=/data/perf OMP_NUM_THREADS=16 CANON_PERF_REPS=15 \
#             tests/passes/canonicalize/run_speedup_table.sh       # tuned for the box
#
# Env knobs: OUT_DIR OMP_NUM_THREADS CANON_PERF_REPS CANON_PERF_WARMUP
#            CANON_PERF_TIMEOUT ONLY FORCE MARKDOWN CSV PYTHON
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"     # python harness defaults to 4 if unset
out_dir="${OUT_DIR:-$here/perf_results}"
markdown="${MARKDOWN:-$out_dir/speedup_table.md}"
csv="${CSV:-$out_dir/speedup.csv}"

args=(--dir "$out_dir" --markdown "$markdown" --csv "$csv")
[ -n "${ONLY:-}" ]  && args+=(--only "$ONLY")
[ -n "${FORCE:-}" ] && args+=(--force)

echo "repo:   $repo_root"
echo "out:    $out_dir"
echo "OMP:    $OMP_NUM_THREADS    best-of: ${CANON_PERF_REPS:-7} (warmup ${CANON_PERF_WARMUP:-2})"
echo "table:  $markdown"
echo "csv:    $csv"
echo

exec "$PYTHON" -m tests.passes.canonicalize.canonicalize_perf_corpus_test "${args[@]}"
