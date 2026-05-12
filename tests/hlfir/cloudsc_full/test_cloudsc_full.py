"""Full-source CLOUDSC end-to-end test for the HLFIR bridge.

Drives the entire ECMWF CLOUDSC microphysics kernel through the
bridge and compares the SDFG output against a gfortran/f2py
reference compiled from the same Fortran source.  Catches
integration regressions the per-loopnest tests can't see — state
hoisting across the block loop, the 100+ scalar-arg signature,
deeply-nested ELEMENTAL expressions, rank-4 ``PCLV(:, :, :, JKGLO)``
per-block slicing — and serves as the gate for the full-ICON
integration test that comes next.

**Source:** ``cloudscexp2_simplified.F90`` (3541 LoC).  Single
``MODULE PARKIND1`` + ``SUBROUTINE CLOUDSCOUTER`` (block-loop
wrapper) + ``SUBROUTINE CLOUDSC`` (physics core).  Every physical
constant passes as a scalar argument — no
``YDCST``/``YDTHF``/``YDECLDP`` derived-type bundling.  This is the
bridge-friendly variant; the verification_pipeline's struct-args
variant is harder until DT-of-constants lowers.

**Reference:** ``f2py``-compiled Fortran from the same source.  Per
``feedback_e2e_numerical``: SDFG-producing tests compare against a
non-transformed reference.

**Inputs:** seeded random data (``np.random.default_rng(42)``) via
the registries in ``_registries.py``.  No HDF5 dependency.

**Status:** xfail probe initially — surfaces the first bridge gap
cleanly without breaking the sweep.  Each gap closes in a separate
commit (the test stays xfailed throughout).  Final commit removes
the xfail decorator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang
from cloudsc_full._registries import (
    get_inputs,
    get_outputs,
    program_outputs,
)

_HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _sdfg_call_args(sdfg, scalar_values: dict) -> dict:
    """Route each scalar arg in ``scalar_values`` to either a plain
    Python scalar (if the SDFG classified it as a symbol or Scalar)
    or a length-1 numpy array (if classified as a length-1 Array).
    Per `feedback_scalar_io_convention` the bridge can register a
    Fortran scalar dummy as either Scalar (intent(in)) or length-1
    Array (intent(inout)/(out)) — this helper picks the binding the
    SDFG actually expects.

    For LOGICAL scalars the bridge declares the length-1 array as
    ``bool *`` (1-byte element).  Routing them as ``np.int32`` (4
    bytes) only accidentally works for ``{0, 1}`` because the LSB
    happens to match -- any value with bit-0 = 0 (e.g. 256, -2)
    would read as ``false``.  Match the declared dtype explicitly.
    """
    from dace.data import Scalar

    arglist = sdfg.arglist()
    out = {}
    for k, v in scalar_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            # Pick the numpy dtype that matches the bridge's
            # declaration -- bool stays 1-byte, float stays 8-byte,
            # everything else int32.
            decl_dtype = str(desc.dtype) if hasattr(desc, "dtype") else ""
            if "bool" in decl_dtype.lower():
                out[k] = np.array([bool(v)], dtype=np.bool_)
            elif isinstance(v, float):
                out[k] = np.array([v], dtype=np.float64)
            else:
                out[k] = np.array([v], dtype=np.int32)
    return out


def _lower_keys(d: dict) -> dict:
    """Fortran-source identifier names are case-sensitive in flang's
    HLFIR output but the Python wrappers expect lowercase kwargs.
    Normalise the registry keys at call time."""
    return {k.lower(): v for k, v in d.items()}


def _f2py_argnames(fn) -> set:
    """Parse ``cloudsc_ref.cloudscouter.__doc__`` to extract the actual
    argument-name list f2py exposes.  f2py auto-derives shape symbols
    (``klon``, ``klev``, ``nblocks``, …) from array shapes and lists
    them in brackets at the end of the signature.  Return a set of
    accepted kwargs (lowercased)."""
    import re

    doc = fn.__doc__ or ""
    match = re.match(r"\s*\w+\((.*?)\)", doc, re.DOTALL)
    if not match:
        return set()
    arglist = match.group(1)
    # Brackets enclose auto-derived shape symbols.
    optional = set()
    for m in re.finditer(r"\[([^\]]+)\]", arglist):
        optional.update(s.strip() for s in m.group(1).split(","))
    arglist = re.sub(r"\[[^\]]*\]", "", arglist)
    return {s.strip() for s in arglist.split(",") if s.strip()} | optional


@pytest.fixture(scope="module")
def _f2py_ref(tmp_path_factory):
    """Build the f2py reference once per pytest session.

    f2py compile of the 3541-line source takes ~30-90s, so this is
    a session-scoped fixture instead of per-test setup.
    """
    src = (_HERE / "cloudscexp2_simplified.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_ref")
    return f2py_compile(src, ref_dir, "cloudsc_ref")


@pytest.mark.xfail(
    strict=False,
    reason="full CLOUDSC integration probe — first iteration surfaces "
    "the bridge gaps that the per-loopnest tests can't see "
    "(rank-4 PCLV slicing, 100+ scalar-arg signature, deeply-nested "
    "ELEMENTAL chains, block-loop state hoisting).  Each gap "
    "closes in a follow-up commit; the test stays xfailed until "
    "all gaps clear.",
)
def test_cloudsc_full_numerical(tmp_path, _f2py_ref):
    """End-to-end SDFG-vs-f2py equivalence on the full CLOUDSC.

    Same Fortran source through both paths, seeded random inputs,
    per-output ``assert_allclose(rtol=1e-12, atol=1e-12)``.
    """
    src = (_HERE / "cloudscexp2_simplified.F90").read_text()

    # SDFG via HLFIR bridge.  Use the DEFAULT_PIPELINE (the full
    # bridge pipeline including inline-all / flatten-structs /
    # lift-alloc-array-of-records / etc.) — the minimal
    # ``hlfir-propagate-shapes`` pipeline used by the per-loopnest
    # tests isn't enough for the full kernel.
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="cloudsc", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs(rng)
    # Lowercase output keys at construction so the SDFG-side call (which
    # passes lowercase kwargs) writes through the same dict entries the
    # per-output comparison loop below looks up by name.lower().
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    # Fortran-side call: gfortran-compiled CLOUDSCOUTER.  f2py
    # auto-derives shape symbols from array shapes and accepts only
    # the args in its parsed signature — filter our kwarg dict to
    # avoid ``TypeError: takes at most N keyword arguments``.
    accepted = _f2py_argnames(_f2py_ref.cloudscouter)
    all_kwargs = {**_lower_keys(inputs), **_lower_keys(outputs_ref)}
    _f2py_ref.cloudscouter(**{k: v for k, v in all_kwargs.items() if k in accepted})

    # SDFG-side call: same kwargs, plus the scalar-arg routing that
    # handles Scalar-vs-length-1-Array classification for both int
    # and float scalars (per feedback_scalar_io_convention).
    scalar_kwargs = {k.lower(): v for k, v in inputs.items() if isinstance(v, (int, float, np.integer, np.floating))}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, (int, float, np.integer, np.floating))}
    sdfg_kwargs.update(_lower_keys(outputs_sdfg))
    sdfg_kwargs.update(_sdfg_call_args(sdfg, scalar_kwargs))
    sdfg(**sdfg_kwargs)

    # Per-output numerical check.  Start strict (rtol=1e-12,
    # atol=1e-12).  Per-output relaxation lands in a follow-up
    # commit if IEEE-fma reordering forces it.
    for name in program_outputs:
        np.testing.assert_allclose(
            outputs_sdfg[name.lower()],
            outputs_ref[name.lower()],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"mismatch on output {name}",
        )
