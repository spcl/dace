# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reproduce the four optimized CloudSC SDFGs and run them against the un-transformed reference.

Every step saves its SDFG to ``<out>/<step>.sdfgz`` and a re-run reloads it instead of recomputing, so
the minutes-long parse, canonicalization and vectorization run once.

    python -m tests.corpus.cloudsc.reproduce --out ~/.cache/cloudsc_repro canon_cpu vec_cpu
    python -m tests.corpus.cloudsc.reproduce --out ~/.cache/cloudsc_repro --run canon_gpu vec_gpu
"""

import argparse
import contextlib
import os
import pickle
import time
from collections.abc import Callable
from pathlib import Path

import dace
from dace.config import set_temporary
from dace.libraries.tileops.dispatch import detect_host_isa
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
from tests.corpus.cloudsc.generate_data_for_cloudsc import (
    CLOUDSC_CONSTANTS,
    IEEE_CPU_ARGS,
    build_cloudsc_sdfg,
    compare_outputs,
)
from tests.corpus.cloudsc.offload_cloudsc_to_gpu import offload_cloudsc_to_gpu
from tests.corpus.cloudsc.pipelines import build_reference_outputs, run_candidate, strict_fp_device_build

#: Species PARAMETERs and run-time flags baked in at the reference values; klev/klon stay symbolic.
SPECIALIZE = {
    "nclv": 5,
    "ncldql": 1,
    "ncldqi": 2,
    "ncldqr": 3,
    "ncldqs": 4,
    "ncldqv": 5,
    **{name: int(CLOUDSC_CONSTANTS[name]) for name in ("yrecldp_nssopt", "yrecldp_laericesed")},
}

#: Maps run in parallel, so reductions fold in a different order than the sequential reference.
PARALLEL_TOL = 1e-10
CPU_VECTOR_WIDTH = 8
GPU_VECTOR_WIDTH = 2


def quiet(fn: Callable[[], None]) -> None:
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        fn()


def canonicalize_for(target: str) -> Callable[[dace.SDFG], None]:

    def step(sdfg: dace.SDFG) -> None:
        quiet(lambda: canonicalize(sdfg, validate=True, target=target, specialize_constants=SPECIALIZE))
        if target == "gpu":
            offload_cloudsc_to_gpu(sdfg)

    return step


def vectorize_cpu(sdfg: dace.SDFG) -> None:
    config = VectorizeConfig(widths=(CPU_VECTOR_WIDTH,), target_isa=detect_host_isa())
    quiet(lambda: VectorizeCPUMultiDim(config).apply_pass(sdfg, {}))


def vectorize_gpu(sdfg: dace.SDFG) -> None:
    quiet(lambda: VectorizeGPU(VectorizeConfig(widths=(GPU_VECTOR_WIDTH,))).apply_pass(sdfg, {}))


#: step -> (parent step, transformation). ``reference`` is the un-transformed parse.
STEPS: dict[str, tuple[str, Callable[[dace.SDFG], None]]] = {
    "canon_cpu": ("reference", canonicalize_for("cpu")),
    "canon_gpu": ("reference", canonicalize_for("gpu")),
    "vec_cpu": ("canon_cpu", vectorize_cpu),
    "vec_gpu": ("canon_gpu", vectorize_gpu),
}


def load(path: Path) -> dace.SDFG:
    with set_temporary("testing", "deserialize_exception", value=True):
        return dace.SDFG.from_file(str(path))


def produce(step: str, out: Path) -> dace.SDFG:
    """Return the SDFG of ``step``, loading ``<out>/<step>.sdfgz`` or building it from its parent."""
    path = out / f"{step}.sdfgz"
    if path.is_file():
        return load(path)
    t0 = time.perf_counter()
    if step == "reference":
        sdfg = build_cloudsc_sdfg(simplify=False)
    else:
        parent, transform = STEPS[step]
        sdfg = produce(parent, out)
        transform(sdfg)
    sdfg.validate()
    sdfg.save(str(path), compress=True)
    print(f"{step}: built in {time.perf_counter() - t0:.0f}s -> {path}")
    return sdfg


def reference_io(out: Path) -> tuple[dict, dict]:
    """``(inputs, outputs)`` of the reference run sequentially under the IEEE build, cached on disk."""
    path = out / "reference_io.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    bundle = build_reference_outputs(produce("reference", out), regime="ieee", seed=0)
    path.write_bytes(pickle.dumps(bundle))
    return bundle


def simulate(step: str, out: Path) -> None:
    """Compile and run ``step`` once on the reference inputs and compare every output array."""
    inputs, expected = reference_io(out)
    sdfg = load(out / f"{step}.sdfgz")
    t0 = time.perf_counter()
    context = strict_fp_device_build() if step.endswith("gpu") else contextlib.nullcontext()
    with context:
        result = run_candidate(sdfg, inputs, IEEE_CPU_ARGS, sequential=False, tag=step)
    report = compare_outputs(result, expected, rtol=PARALLEL_TOL, atol=PARALLEL_TOL)
    bad = sorted(name for name, (_, _, ok) in report.items() if not ok)
    worst = max((max_abs for max_abs, _, _ in report.values()), default=0.0)
    print(
        f"{step}: build+run {time.perf_counter() - t0:.0f}s, worst |abs| {worst:.2e}, "
        f"{'OK' if not bad else f'MISMATCH {bad}'}"
    )
    if bad:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=list(STEPS))
    parser.add_argument("--out", type=Path, default=Path.home() / ".cache" / "cloudsc_repro")
    parser.add_argument("--run", action="store_true", help="compile, run and check each step")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        produce(step, args.out)
        if args.run:
            simulate(step, args.out)


if __name__ == "__main__":
    main()
