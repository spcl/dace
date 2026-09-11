# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Print canonicalize vs auto_optimize timing ratios over the npbench/polybench corpora.

Human-readable measurement only -- no pass/fail semantics. A wall-clock ratio between two
differently-shaped lowerings (canon emits a clean, un-tiled form; auto_optimize tiles and
fuses) does not scale together across core count, cache, or bandwidth, so it cannot be a
committed test assertion (see ``feedback_ab_must_hold_runner_config_fixed`` /
``feedback_canonicalization_does_not_tile``). This script replaces the former
``speedup_vs_autoopt_test.py`` pytest gate: run it by hand and read the table.

Run sequentially with a fixed thread count -- concurrent xdist workers or a shared core
corrupt the timings:
    OMP_NUM_THREADS=8 PYTHONPATH=. python tools/bench/canon_vs_autoopt.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from dataclasses import dataclass

import numpy as np

import dace
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench

#: Skip a kernel pair whose auto-opt median is below this (ms) -- too small to time reliably.
MIN_MS = 0.05
REPS = 10
WARMUP = 2


@dataclass(slots=True)
class Row:
    corpus: str
    name: str
    cn_ms: float
    ao_ms: float

    @property
    def ratio(self) -> float:
        return self.cn_ms / self.ao_ms


def median_ms(sdfg: dace.SDFG, call: dict) -> float:
    csdfg = sdfg.compile()
    with dace.profile(repetitions=REPS, warmup=WARMUP, print_results=False) as prof:
        csdfg(**call)  # dace.profile runs it REPS times internally
    _report, times = prof.times[-1]
    return float(np.median(np.asarray(times)))


def npbench_call(c: dict, arrays: dict, params: dict) -> dict:
    # cavity_flow's inputs are not all arrays: one is a plain float, which has no `copy`.
    fresh = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    call = npbench._map_call(c["program"], fresh, params)
    call.update({k: v for k, v in params.items() if k not in call and not isinstance(v, float)})
    return call


def measure_npbench(name: str) -> Row | None:
    c = next(k for k in npbench.collect() if k["name"] == name)
    arrays, params = npbench.make_inputs(c, cap=None)
    ref = npbench.reference_outputs(c, arrays, params)
    try:
        ao = npbench.fresh_sdfg(c)
        auto_optimize(ao, dace.DeviceType.CPU)
    except Exception as e:
        print(f"{name}: auto_optimize failed: {type(e).__name__}: {str(e)[:60]}")
        return None
    if not npbench.outputs_match(ref, npbench.run_outputs(c, ao, arrays, params)):
        print(f"{name}: auto_optimize output disagrees with reference, skipping")
        return None
    ao_ms = median_ms(ao, npbench_call(c, arrays, params))
    if ao_ms < MIN_MS:
        print(f"{name}: too small to time reliably (auto-opt {ao_ms:.4f} ms)")
        return None
    try:
        cn = npbench.fresh_sdfg(c)
        canonicalize(cn, validate=True)
    except Exception as e:
        print(f"{name}: canonicalize failed: {type(e).__name__}: {str(e)[:60]}")
        return None
    if not npbench.outputs_match(ref, npbench.run_outputs(c, cn, arrays, params)):
        print(f"{name}: CANON OUTPUT INCORRECT vs npbench reference, skipping timing")
        return None
    cn_ms = median_ms(cn, npbench_call(c, arrays, params))
    return Row("npbench", name, cn_ms, ao_ms)


def measure_polybench(name: str) -> Row | None:
    k = polybench.collect(name=name)[0]
    arrays, psize = polybench.make_inputs(k, size_index=2, cap=None)
    ref = polybench.reference(k, arrays, psize)
    try:
        ao = polybench.fresh_sdfg(k)
        auto_optimize(ao, dace.DeviceType.CPU)
    except Exception as e:
        print(f"{name}: auto_optimize failed: {type(e).__name__}: {str(e)[:60]}")
        return None
    if not polybench.outputs_match(ref, polybench.run(ao, arrays, psize)):
        print(f"{name}: auto_optimize output disagrees with baseline, skipping")
        return None
    ao_call = {**{n: v.copy() for n, v in arrays.items()}, **psize}
    ao_ms = median_ms(ao, ao_call)
    if ao_ms < MIN_MS:
        print(f"{name}: too small to time reliably (auto-opt {ao_ms:.4f} ms)")
        return None
    try:
        cn = polybench.fresh_sdfg(k)
        canonicalize(cn, validate=True)
    except Exception as e:
        print(f"{name}: canonicalize failed: {type(e).__name__}: {str(e)[:60]}")
        return None
    if not polybench.outputs_match(ref, polybench.run(cn, arrays, psize)):
        print(f"{name}: CANON OUTPUT INCORRECT vs baseline, skipping timing")
        return None
    cn_call = {**{n: v.copy() for n, v in arrays.items()}, **psize}
    cn_ms = median_ms(cn, cn_call)
    return Row("polybench", name, cn_ms, ao_ms)


def print_table(rows: list[Row]) -> None:
    rows = sorted(rows, key=lambda r: r.ratio, reverse=True)
    header = f"{'corpus':10} {'kernel':24} {'canon ms':>10} {'autoopt ms':>10} {'ratio':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row.corpus:10} {row.name:24} {row.cn_ms:10.4f} {row.ao_ms:10.4f} {row.ratio:8.2f}x")


def main() -> None:
    rows = []
    for c in npbench.collect():
        row = measure_npbench(c["name"])
        if row is not None:
            rows.append(row)
    for k in polybench.collect():
        row = measure_polybench(k.name)
        if row is not None:
            rows.append(row)
    print_table(rows)


if __name__ == "__main__":
    main()
