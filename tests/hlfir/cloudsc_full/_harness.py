"""Shared call scaffolding for the full-CLOUDSC test family.

Every ``test_cloudsc_*`` module builds the SDFG, runs the same f2py
reference on identical seeded *physical* inputs, and routes the 100+
scalar args through the same Scalar-vs-length-1-Array ABI convention.
Only the post-run mismatch policy differs per module (full-CLOUDSC
has a PCOVPTOT threshold-flip budget; the carve-outs use a plain
element-wise compare), so that part stays in each test.
"""

import re
from pathlib import Path

import numpy as np

from _util import build_sdfg
from cloudsc_full._registries import get_inputs_physical, get_outputs

_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)
_ENTRY = "_QPcloudscouter"


def lower_keys(d: dict) -> dict:
    """Lowercase every key: flang HLFIR identifiers are case-sensitive
    but the Python/f2py wrappers expect lowercase kwargs."""
    return {k.lower(): v for k, v in d.items()}


def f2py_argnames(fn) -> set:
    """The argument names f2py actually exposes, parsed from its
    generated docstring.  Bracketed entries are auto-derived shape
    symbols (``klon``, ``klev``, ...) and are kept as accepted."""
    doc = fn.__doc__ or ""
    m = re.match(r"\s*\w+\((.*?)\)", doc, re.DOTALL)
    if not m:
        return set()
    al = m.group(1)
    opt = set()
    for mm in re.finditer(r"\[([^\]]+)\]", al):
        opt.update(s.strip() for s in mm.group(1).split(","))
    al = re.sub(r"\[[^\]]*\]", "", al)
    return {s.strip() for s in al.split(",") if s.strip()} | opt


def sdfg_call_args(sdfg, scalar_values: dict) -> dict:
    """Route each scalar arg to a plain Python scalar (the SDFG
    classified it Scalar / symbol) or a length-1 numpy array
    (``intent(out)/inout``), matching the dtype the bridge declared --
    bool stays 1-byte, float 8-byte, everything else int32.  See
    ``feedback_scalar_io_convention``."""
    from dace.data import Scalar

    arglist = sdfg.arglist()
    out = {}
    for k, v in scalar_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            decl = str(desc.dtype) if hasattr(desc, "dtype") else ""
            if "bool" in decl.lower():
                out[k] = np.array([bool(v)], dtype=np.bool_)
            elif isinstance(v, float):
                out[k] = np.array([v], dtype=np.float64)
            else:
                out[k] = np.array([v], dtype=np.int32)
    return out


def run_cloudsc(src: str, name: str, f2py_ref, sdfg_dir: Path, *, seed: int = 42):
    """Build the SDFG and run both the f2py reference and the SDFG on
    identical seeded physical inputs.

    :param src: Fortran source text (the per-test ``cloudsc*.F90``).
    :param name: SDFG / dacecache name.
    :param f2py_ref: compiled f2py module exposing ``.cloudscouter``.
    :param sdfg_dir: scratch directory for the SDFG build.
    :param seed: RNG seed for ``get_inputs_physical`` / ``get_outputs``.
    :returns: ``(outputs_sdfg, outputs_ref)`` -- lowercase-keyed dicts
        for the caller to compare under its own mismatch policy.
    """
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=name, entry=_ENTRY).build()

    rng = np.random.default_rng(seed)
    inputs = get_inputs_physical(rng)
    outputs_ref = lower_keys(get_outputs(rng))
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    accepted = f2py_argnames(f2py_ref.cloudscouter)
    all_kw = {**lower_keys(inputs), **lower_keys(outputs_ref)}
    f2py_ref.cloudscouter(**{k: v for k, v in all_kw.items() if k in accepted})

    scalars = {k.lower(): v for k, v in inputs.items() if isinstance(v, _SCALAR_TYPES)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _SCALAR_TYPES)}
    sdfg_kwargs.update(lower_keys(outputs_sdfg))
    sdfg_kwargs.update(sdfg_call_args(sdfg, scalars))
    sdfg(**sdfg_kwargs)

    return outputs_sdfg, outputs_ref
