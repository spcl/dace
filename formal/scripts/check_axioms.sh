#!/usr/bin/env bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#
# Builds the Lean library and verifies that no theorem depends on `sorryAx`,
# that no source file contains `sorry`, and that Mathlib is only imported
# from `Dace/Prelude.lean`.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== lake build"
lake build

echo "== sorry scan"
if grep -rn --include='*.lean' -E '\bsorry\b' Dace Tools 2>/dev/null; then
  echo "error: 'sorry' found in sources" >&2
  exit 1
fi

echo "== Mathlib import isolation"
if grep -rln --include='*.lean' -E '^import Mathlib' Dace Tools 2>/dev/null | grep -v '^Dace/Prelude.lean$'; then
  echo "error: Mathlib imported outside Dace/Prelude.lean" >&2
  exit 1
fi

echo "== axiom check"
lake env lean Dace/AxiomCheck.lean | tee /dev/stderr | awk '
  /depends on axioms/ { inaxioms = 1 }
  inaxioms && /sorryAx/ { bad = 1 }
  END { if (bad) { print "error: sorryAx in axiom report" > "/dev/stderr"; exit 1 } }
'
echo "OK"
