# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ternary blend whose arms are a literal and an ``fp16`` value, through the GPU vectorizer.

``np.where(x > 0, 0.0, x)`` on a ``float16`` array reaches codegen as ``ITE(c, 0.0, x)``. Only the
arms that are IN-CONNECTORS were unified to one dtype: a literal arm carries no edge, so nothing
cast it, and the two arms reached C++ with different types. The tile expansions hide that -- they
materialise a non-Tile arm as a broadcast buffer of the tile dtype -- but the scalar tail of a
vectorized map emits the ternary itself, and ``ITE`` returns ``std::common_type<TA, TB>``, which
does not exist for (``double``, ``dace::float16``): every conversion ``__half`` offers applies
equally, so the arms have no common type at all. The tail failed to compile, on a path the tile
assertions never look at.

Fixing that in ``ResolveMixedDtypeBinops`` exposed a second defect underneath: the ISA tile
expansions embedded a ``Symbol``-kind arm's expression into C++ as raw PYTHON text, so the cast the
first fix writes arrived as ``dace.float16(0.0)`` in a ``.cu`` file. Every pure expansion already
routes such an expression through ``pyexpr2cpp``; the ISA ones did not.

Covered:
  * every emitted ``ITE`` arm carries an explicit dtype (no un-typed literal reaches the ternary);
  * no Python syntax survives into the device code;
  * the program compiles with nvcc, and is bit-exact against the NumPy fp16 oracle at an ODD
    extent, which is the extent that actually executes the scalar tail.
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

N = dace.symbol("N")


@dace.program
def _where16(x: dace.float16[N], y: dace.float16[N]):
    y[:] = np.where(x > 0, 0.0, x)


def _vectorized(name: str = None) -> dace.SDFG:
    """``@dace.program`` -> GPU -> the half2 vectorizer with the branched tail, the configuration
    that emits the scalar tail this test is about."""
    sdfg = _where16.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    VectorizeGPU(VectorizeConfig(widths=(2, ), remainder_strategy="branched_tail")).apply_pass(sdfg, {})
    if name:
        sdfg.name = name
    return sdfg


def _device_code(sdfg: dace.SDFG) -> str:
    return "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")


def _run_isolated(body: str) -> int:
    """Run the module-level ``body`` in a FRESH interpreter; 0 == pass. A CUDA context inherited
    from an earlier test in the same session is unusable in a fork, and a device fault must not
    reach the pytest parent."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    return subprocess.run([sys.executable, __file__, body], env=env).returncode


def _oracle(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, np.float16(0.0), x).astype(np.float16)


# ------------------------------------------------------------------------------------------------
# Structural: what reaches the C++ ternary (no GPU device and no nvcc needed)
# ------------------------------------------------------------------------------------------------
def test_every_ite_arm_is_typed():
    """The scalar tail's ternary must not mix an untyped literal with an fp16 value.

    The check is on the ARMS, not on whether the file happens to compile: a toolkit that resolves
    the ambiguity by picking one of ``__half``'s conversions would compile a silent narrowing
    instead of failing, and that is worse than the error this started as.
    """
    cu = _device_code(_vectorized())
    ites = re.findall(r"ITE\((.*?)\);", cu)
    assert ites, "no scalar-tail ITE emitted; the test would prove nothing"
    for args in ites:
        assert "dace::float16(" in args, f"an ITE arm reached C++ without a dtype: ITE({args})"


def test_no_python_syntax_in_device_code():
    """The cast is written into a PYTHON tasklet body, so every site embedding such an expression
    into C++ has to translate it. ``dace.float16(0.0)`` in a ``.cu`` is the ISA expansion skipping
    the ``pyexpr2cpp`` step its pure counterpart performs."""
    cu = _device_code(_vectorized())
    assert "dace.float16(" not in cu, "a Python cast call was embedded verbatim into device code"
    leak = re.search(r"\bdace\.[A-Za-z_][A-Za-z_0-9]*\s*\(", cu)
    assert leak is None, f"Python attribute syntax leaked into device code: {leak.group(0) if leak else ''}"


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc required to compile the generated device code")
def test_generated_code_compiles():
    """The original report: the tail's ternary did not compile at all."""
    _vectorized(name="fp16_ite_literal_arm_compile").compile()


# ------------------------------------------------------------------------------------------------
# GPU: the numbers. The tail is what an odd extent executes, so that is where the arm's dtype shows.
# ------------------------------------------------------------------------------------------------
def _body_bitexact():
    """``apply_gpu_transformations`` keeps the ENTRY on the host and copies in/out around the
    kernel, so the arguments are host arrays -- the device round trip is what runs."""
    csr = _vectorized(name="fp16_ite_literal_arm_numeric").compile()
    for n in (1023, 1024):  # odd: the last element goes through the scalar tail
        rng = np.random.default_rng(n)
        x = rng.integers(-8, 8, size=n).astype(np.float16)
        x[0], x[1], x[2] = np.float16(0.0), np.float16(-0.0), np.float16(7.0)
        expected = _oracle(x)
        y = np.zeros(n, dtype=np.float16)
        csr(x=x, y=y, N=n)
        assert np.array_equal(y.view(np.uint16), expected.view(np.uint16)), \
            f"N={n}: not bit-exact vs the numpy fp16 oracle"


@pytest.mark.gpu
def test_bitexact_including_scalar_tail():
    assert _run_isolated("_body_bitexact") == 0


if __name__ == "__main__":
    if len(sys.argv) > 1:  # re-entry from _run_isolated
        globals()[sys.argv[1]]()
    else:
        test_every_ite_arm_is_typed()
        test_no_python_syntax_in_device_code()
        test_generated_code_compiles()
        test_bitexact_including_scalar_tail()
