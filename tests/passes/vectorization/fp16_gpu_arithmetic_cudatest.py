# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""fp16 arithmetic through the GPU half2 vectorizer, end to end.

``float16`` is the dtype where the tile path keeps being wrong in ways nothing else catches, and
always for the same reason: ``__half`` is a class with a dozen implicit conversions, so an operand
that reaches C++ with the wrong type either fails to compile (``ITE`` has no
``std::common_type<double, dace::float16>``) or, worse, picks one of those conversions and narrows
silently. The existing fp16 GPU tests cover the shifted load (``half_shifted_load_cudatest``), the
remainder strategy (``branched_tail_remainder_cudatest``), the hop rules
(``lib_nodes/tile_fp16_conversion_ambiguity_cudatest``) and one ternary
(``fp16_ite_literal_arm_cudatest``). What none of them exercise is ordinary fp16 ARITHMETIC over
the whole tile-op set with the numbers actually checked.

This module does that, one kernel per tile op the vectorizer emits -- binop, mixed-dtype binop,
FMA, unary math, select, reduction -- and for each asserts two things:

* the structural half: the emitted CUDA really took the width-2 fp16 path. A numeric check on a
  kernel that quietly stayed scalar proves nothing about the tile ops.
* the numeric half: BIT-EXACT against a numpy fp16 oracle, at an ODD extent as well as an even
  one. The odd extent is what runs the scalar tail, which is a different emitter from the tile
  body and the one that has broken twice.

Every value fed in is exactly representable in fp16 (integers below 2048, halves, perfect squares),
so "bit-exact" is a statement about the lowering rather than about rounding: any reassociation the
tile path performs is visible as a difference, and none is permitted here.

GPU-executing bodies run in a fresh interpreter (see ``_run_isolated``) so a device fault cannot
take the pytest parent with it.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

_HAS_NVCC = shutil.which("nvcc") is not None

#: Every test here is about float16, so the whole module carries the marker the fp16 CI leg
#: selects on.
pytestmark = pytest.mark.fp16

N = dace.symbol("N")

#: Extents every numeric body runs at: one that divides the width, one that does not (the scalar
#: tail), and a small one where a tail is most of the work.
EXTENTS = (1024, 1023, 7)


@dace.program
def _scale_add16(x: dace.float16[N], y: dace.float16[N], out: dace.float16[N]):
    out[:] = x * y + x


@dace.program
def _mixed16(x: dace.float16[N], z: dace.float64[N], out: dace.float16[N]):
    out[:] = x * 2.0 + z


@dace.program
def _select16(x: dace.float16[N], y: dace.float16[N], out: dace.float16[N]):
    out[:] = np.where(x > y, x, y)


@dace.program
def _sqrt16(x: dace.float16[N], out: dace.float16[N]):
    out[:] = np.sqrt(x)


@dace.program
def _sum16(x: dace.float16[N], out: dace.float16[1]):
    out[0] = np.sum(x)


def _vectorized(program, name: str = None, width: int = 2) -> dace.SDFG:
    """``@dace.program`` -> GPU -> the half2 vectorizer with the branched tail (the GPU default)."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    VectorizeGPU(VectorizeConfig(widths=(width, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
    if name:
        sdfg.name = name
    return sdfg


def _device_code(sdfg: dace.SDFG) -> str:
    return "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")


def _fp16_tile_ops(code: str) -> set:
    """Names of the ``dace::tileops::`` calls emitted at width 2 over ``dace::float16``."""
    return set(re.findall(r"dace::tileops::(\w+)<dace::float16, 2", code))


def _run_isolated(body: str) -> int:
    """Run the module-level ``body`` in a FRESH interpreter; 0 == pass."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    return subprocess.run([sys.executable, __file__, body], env=env).returncode


def _inputs(n: int, seed: int):
    """Exactly-representable fp16 draws: integers in [-8, 8) and their halves."""
    rng = np.random.default_rng(seed)
    x = (rng.integers(-16, 16, size=n) / 2.0).astype(np.float16)
    y = (rng.integers(-16, 16, size=n) / 2.0).astype(np.float16)
    return x, y


# ------------------------------------------------------------------------------------------------
# Structural: the width-2 fp16 path is the one that runs (no GPU device needed)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("program,expected", [
    (_scale_add16, {"tile_load", "tile_store"}),
    (_select16, {"tile_load"}),
    (_sqrt16, {"tile_load", "tile_store"}),
])
def test_kernel_takes_the_fp16_tile_path(program, expected):
    """Each kernel must reach the fp16 tile ops. Asserted per kernel rather than once, because a
    kernel that silently stays scalar makes its numeric body a test of nothing.

    ``_select16`` expects the LOAD only: its store is masked, and a masked sub-32-bit store cannot
    be widened without writing the lanes the predicate excluded. A load may round down to the
    aligned window below it and cut the elements out of the register pair; a store has no such
    escape. :func:`test_masked_select_store_stays_per_element` states that as its own claim.
    """
    ops = _fp16_tile_ops(_device_code(_vectorized(program)))
    assert expected <= ops, f"expected {sorted(expected)} among the emitted fp16 tile ops, got {sorted(ops)}"


def test_masked_select_store_stays_per_element():
    """A masked fp16 store stays per-element even when the extent is a known multiple of the width.

    Pinned because the alternative is a silent memory bug rather than a slow kernel: a widened
    32-bit store writes both halves of the pair, and the lane the predicate excluded is one of
    them. The specialization removes the parity question, so what remains is the mask.
    """
    sdfg = _select16.to_sdfg(simplify=True)
    sdfg.specialize({"N": 1024})
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    VectorizeGPU(VectorizeConfig(widths=(2, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
    ops = _fp16_tile_ops(_device_code(sdfg))
    assert "tile_load" in ops, f"the select kernel stopped widening its loads too: {sorted(ops)}"
    assert "tile_store" not in ops, f"a masked fp16 store was widened: {sorted(ops)}"


def test_mixed_dtype_kernel_keeps_one_dtype_per_tile_op():
    """A float64 operand meeting a float16 one may not reach a tile op as a mixed pair: the
    walker-primary contract locks one dtype per lib node, and the promotion is an explicit cast
    tasklet. What must NOT appear is an fp16 tile op fed a bare ``double``."""
    code = _device_code(_vectorized(_mixed16))
    assert _fp16_tile_ops(code), "the mixed-dtype kernel never reached the fp16 tile path"
    assert "dace.float16(" not in code and "dace.float64(" not in code, \
        "a Python cast call was embedded verbatim into device code"


def test_no_python_syntax_reaches_device_code():
    """Every operand a tile expansion embeds is C++ by the time it is written out. A ``dace.<x>(``
    in a ``.cu`` is a Python expression that was never translated."""
    for program in (_scale_add16, _mixed16, _select16, _sqrt16, _sum16):
        code = _device_code(_vectorized(program))
        leak = re.search(r"\bdace\.[A-Za-z_][A-Za-z_0-9]*\s*\(", code)
        assert leak is None, f"{program.name}: Python syntax leaked into device code: {leak.group(0)}"


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc required to compile the generated device code")
@pytest.mark.parametrize("program", [_scale_add16, _mixed16, _select16, _sqrt16, _sum16])
def test_generated_fp16_code_compiles(program):
    """nvcc is where an fp16 conversion ambiguity surfaces; the host compiler never sees it."""
    _vectorized(program, name=f"fp16_compile_{program.name}").compile()


# ------------------------------------------------------------------------------------------------
# GPU: the numbers, at extents that do and do not divide the vector width
# ------------------------------------------------------------------------------------------------
def _body_scale_add():
    csr = _vectorized(_scale_add16, name="fp16_scale_add_numeric").compile()
    for n in EXTENTS:
        x, y = _inputs(n, n)
        expected = (x * y + x).astype(np.float16)
        out = np.zeros(n, dtype=np.float16)
        csr(x=x.copy(), y=y.copy(), out=out, N=n)
        assert np.array_equal(out.view(np.uint16), expected.view(np.uint16)), f"N={n}: fp16 fma not bit-exact"


def _body_mixed():
    """float16 * double + double, stored back to float16. numpy promotes the multiply to float64
    and rounds once on the store; the tile path must do the same, not accumulate in fp16."""
    csr = _vectorized(_mixed16, name="fp16_mixed_numeric").compile()
    for n in EXTENTS:
        x, _ = _inputs(n, n + 1)
        z = (np.arange(n) % 9 - 4).astype(np.float64) / 4.0
        expected = (x.astype(np.float64) * 2.0 + z).astype(np.float16)
        out = np.zeros(n, dtype=np.float16)
        csr(x=x.copy(), z=z.copy(), out=out, N=n)
        assert np.array_equal(out.view(np.uint16), expected.view(np.uint16)), \
            f"N={n}: mixed fp16/fp64 result not bit-exact vs the numpy promotion"


def _body_select():
    """Both arms are per-lane fp16 tiles, so neither is a broadcast the expansion casts on its own.
    Equal operands are included on purpose: the predicate is strict, so a tie must take ``y``."""
    csr = _vectorized(_select16, name="fp16_select_numeric").compile()
    for n in EXTENTS:
        x, y = _inputs(n, n + 2)
        y[:8] = x[:8]  # ties
        expected = np.where(x > y, x, y).astype(np.float16)
        out = np.zeros(n, dtype=np.float16)
        csr(x=x.copy(), y=y.copy(), out=out, N=n)
        assert np.array_equal(out.view(np.uint16), expected.view(np.uint16)), f"N={n}: fp16 select not bit-exact"


def _body_sqrt():
    """Perfect squares only: their fp16 roots are exact, so a device intrinsic that differs from
    numpy by an ulp on a general input cannot hide a lowering bug behind rounding."""
    csr = _vectorized(_sqrt16, name="fp16_sqrt_numeric").compile()
    for n in EXTENTS:
        x = (np.arange(n) % 32).astype(np.float16)**2
        expected = np.sqrt(x.astype(np.float32)).astype(np.float16)
        out = np.zeros(n, dtype=np.float16)
        csr(x=x.copy(), out=out, N=n)
        assert np.array_equal(out.view(np.uint16), expected.view(np.uint16)), f"N={n}: fp16 sqrt not bit-exact"


def _body_sum():
    """An fp16 reduction is order-dependent in general, so the draw is integers whose partial sums
    all stay below 2048 -- every fp16 value there is exact and every association gives one answer.
    That keeps the assertion about the reduction's lowering rather than about its tree shape."""
    csr = _vectorized(_sum16, name="fp16_sum_numeric").compile()
    for n in EXTENTS:
        x = ((np.arange(n) % 3) - 1).astype(np.float16)
        expected = np.float16(x.astype(np.float64).sum())
        out = np.zeros(1, dtype=np.float16)
        csr(x=x.copy(), out=out, N=n)
        assert out[0] == expected, f"N={n}: fp16 sum {out[0]} != {expected}"


@pytest.mark.gpu
@pytest.mark.parametrize("body", [
    "_body_scale_add",
    "_body_mixed",
    "_body_select",
    "_body_sqrt",
    "_body_sum",
])
def test_fp16_kernel_is_bit_exact(body):
    assert _run_isolated(body) == 0


if __name__ == "__main__":
    if len(sys.argv) > 1:  # re-entry from _run_isolated
        globals()[sys.argv[1]]()
    else:
        for _body in ("_body_scale_add", "_body_mixed", "_body_select", "_body_sqrt", "_body_sum"):
            globals()[_body]()
