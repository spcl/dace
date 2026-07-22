# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Full-corpus correctness sweep for the experimental (readable) code generator.

Enumerates every kernel of npbench, polybench, tsvc and tsvc_2_5, builds each
through the standard simplify+LoopToMap+MapFusion pipeline, and compiles+runs it
under legacy and experimental codegen on IDENTICAL inputs (generated once per
kernel and deep-copied per run -- crucial, since some kernels have np.empty
scratch outputs that are not fully overwritten and would otherwise differ purely
from garbage). Each run is crash-isolated (fork). Writes a live TSV of results.

Usage:
    python run_full_corpus_sweep.py [--target cpu|gpu] [--out results.tsv] [--only SUBSTR]
"""
import argparse
import copy
import sys
import time

import numpy as np

from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from tests.codegen.readable.conftest import LEGACY, EXPERIMENTAL, run_isolated, use_implementation
from tests.codegen.readable.test_corpus_equivalence import TSVC_2_5_PROGRAMS
from tests.corpus.npbench import npbench
from tests.corpus.polybench import polybench
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc_2_5 import tsvc_2_5

import dace


def pipeline(sdfg, target):
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify()
    if target == "gpu":
        sdfg.apply_gpu_transformations()
        # Steer library nodes (e.g. Cholesky/Solve with implementation=None, whose library default is
        # the CPU OpenBLAS/MKL expansion) to the cuSOLVER/cuBLAS GPU expansions.
        set_fast_implementations(sdfg, dace.DeviceType.GPU)


def make_base(corpus, name):
    """Generate the input arrays + call kwargs ONCE for a kernel (shared by both runs)."""
    if corpus == "poly":
        kernel = polybench.collect(name)[0]
        arrays, psize = polybench.make_inputs(kernel)
        return dict(kernel=kernel, arrays=arrays, extra=psize)
    if corpus == "np":
        descriptor = npbench.collect(name)[0]
        arrays, params = npbench.make_inputs(descriptor)
        return dict(descriptor=descriptor, arrays=arrays, extra=params)
    if corpus == "tsvc":
        kernel = tsvc.collect(name=name)[0]
        arrays, call_kwargs = tsvc.make_inputs(kernel)
        return dict(kernel=kernel, arrays=arrays, extra=call_kwargs)
    program = TSVC_2_5_PROGRAMS[name]
    arrays, scalars = tsvc_2_5.make_inputs(program)
    return dict(program=program, arrays=arrays, extra=scalars)


def run_impl(corpus, name, impl, target, base):
    """Build under ``impl`` and run on a deep copy of the shared inputs; return outputs."""
    tag = f"{impl}_{target}"
    arrays = copy.deepcopy(base["arrays"])
    extra = base["extra"]
    if corpus == "poly":
        sdfg = polybench.fresh_sdfg(base["kernel"])
        pipeline(sdfg, target)
        sdfg.name = f"{sdfg.name}_{tag}"
        sdfg.compile()(**arrays, **extra)
        return arrays
    if corpus == "np":
        sdfg = npbench.fresh_sdfg(base["descriptor"])
        pipeline(sdfg, target)
        sdfg.name = f"{sdfg.name}_{tag}"
        return npbench.run_outputs(base["descriptor"], sdfg, arrays, extra)
    if corpus == "tsvc":
        sdfg = tsvc.to_sdfg(base["kernel"], tag, simplify=True)
        pipeline(sdfg, target)
        sdfg.compile()(**arrays, **extra)
        return arrays
    program = base["program"]
    sdfg = program.to_sdfg(simplify=True)
    pipeline(sdfg, target)
    sdfg.name = f"{sdfg.name}_{tag}"
    free = {str(s) for s in sdfg.free_symbols}
    for s in free:
        if s not in sdfg.symbols:
            sdfg.add_symbol(s, dace.int64)
    symbols = {n: v for n, v in tsvc_2_5.SIZES.items() if n in free}
    sdfg.compile()(**arrays, **extra, **symbols)
    return arrays


def run_case(corpus, name, impl, target, base):
    work = lambda: run_impl(corpus, name, impl, target, base)
    with use_implementation(impl):
        if target == "gpu":
            return work()
        return run_isolated(work)


def compare(a, b):
    """Return (equal, max_abs_diff) over array outputs shared by both dicts."""
    worst = 0.0
    for k in a:
        if isinstance(a[k], np.ndarray) and k in b and isinstance(b[k], np.ndarray):
            d = float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) if a[k].size else 0.0
            worst = max(worst, d)
    return worst == 0.0, worst


def all_cases():
    cases = []
    cases += [("poly", k.name) for k in polybench.collect()]
    cases += [("np", (c["name"] if isinstance(c, dict) else c.name)) for c in npbench.collect()]
    cases += [("tsvc", k.name) for k in tsvc.collect()]
    cases += [("tsvc25", name) for name in TSVC_2_5_PROGRAMS]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--out", default="corpus_sweep_results.tsv")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    cases = all_cases()
    if args.only:
        cases = [(c, n) for (c, n) in cases if args.only in f"{c}/{n}"]

    npass = nfail = 0
    fails = []
    with open(args.out, "w") as f:
        f.write("corpus\tname\ttarget\tstatus\tseconds\tmaxdiff\treason\n")
        f.flush()
        for i, (corpus, name) in enumerate(cases, 1):
            label = f"{corpus}/{name}"
            t0 = time.time()
            maxdiff = ""
            try:
                base = make_base(corpus, name)  # ONCE -> identical inputs for both
                legacy = run_case(corpus, name, LEGACY, args.target, base)
                experimental = run_case(corpus, name, EXPERIMENTAL, args.target, base)
                equal, worst = compare(legacy, experimental)
                maxdiff = f"{worst:.3e}"
                if equal:
                    status, reason = "PASS", ""
                    npass += 1
                else:
                    status, reason = "FAIL", f"max|diff|={worst:.3e}"
                    nfail += 1
                    fails.append(label)
            except Exception as ex:  # noqa: BLE001
                status = "ERROR"
                reason = f"{type(ex).__name__}: {str(ex).splitlines()[0][:160]}" if str(ex) else type(ex).__name__
                nfail += 1
                fails.append(label)
            dt = time.time() - t0
            f.write(f"{corpus}\t{name}\t{args.target}\t{status}\t{dt:.1f}\t{maxdiff}\t{reason}\n")
            f.flush()
            print(f"[{i:3d}/{len(cases)}] {label:44s} {status:5s} ({dt:5.1f}s) diff={maxdiff:9s} "
                  f"pass={npass} fail={nfail}", flush=True)

    print(f"\n==== SWEEP DONE: {npass}/{len(cases)} PASS, {nfail} FAIL/ERROR ({args.target}) ====")
    if fails:
        print("FAILURES:", ", ".join(fails))
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
