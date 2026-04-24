"""SDFG descriptor registration + type mapping + synthetic-scalar lazy decl.

``add_descriptors`` is called once from ``SDFGBuilder.build()`` to register
symbols, arrays, and scalars on the fresh SDFG.  ``auto_declare_synth``
runs on-demand from the emit path when the bridge introduces synthetic
scalars (``__sc_N`` / ``__al_N``) that weren't in the original variable
classification.
"""
from __future__ import annotations

from types import SimpleNamespace

import dace
from dace import SDFG

DTYPE = {
    'float64': dace.float64,
    'float32': dace.float32,
    'int32': dace.int32,
    'int64': dace.int64,
    'bool': dace.bool_,
}


def dt(s: str) -> dace.typeclass:
    return DTYPE.get(s, dace.float64)


def sdfg_name(builder) -> str:
    """Derive the SDFG name from the first Flang mangled name we see."""
    for v in builder.arrays.values():
        mn = v.mangled_name
        if '_QF' in mn and 'E' in mn:
            return mn.split('_QF')[1].split('E')[0]
    return "sdfg"


def add_descriptors(builder, sdfg: SDFG):
    """Add symbols, arrays, and scalars to ``sdfg`` from ``builder``'s
    classified variable dicts.

    Scalar rule: locals (``intent=''``) → ``dace.data.Scalar``.  Dummy-arg
    scalars (``intent in/out/inout``) land as length-1 ``dace.data.Array``
    because DaCe doesn't put non-transient Scalars on the SDFG signature.
    """
    # Named Fortran symbols (nproma, nlev, …).
    for v in builder.symbols.values():
        sdfg.add_symbol(v.fortran_name, dt(v.dtype))

    # Synthetic symbols for dims that stayed unresolved after passes.
    # Literal-integer dimensions (e.g. the "3" in ``edge_idx(nc, 3)``) stay
    # as Python ints and do not need a symbol registration.
    known = {v.fortran_name for v in builder.variables}
    for v in builder.arrays.values():
        for s in v.shape_symbols:
            if s.lstrip('-').isdigit():
                continue
            if s not in known and s not in sdfg.symbols:
                sdfg.add_symbol(s, dace.int64)

    def _dim(s: str):
        if s.lstrip('-').isdigit():
            return int(s)
        return dace.symbol(s)

    def _fortran_strides(dims):
        """Column-major strides: stride[i] = product of dims[0..i-1].
        Fortran's declaration ``real :: a(nproma, nlev, nblks_e)`` has
        nproma as the fastest-varying index (stride 1), matching what
        Flang's HLFIR expects — so the SDFG descriptor must advertise
        the same layout or DaCe's C-order default will mis-index when
        called with numpy F-order inputs."""
        strides = []
        acc = 1
        for d in dims:
            strides.append(acc)
            acc = acc * d
        return strides

    for v in builder.arrays.values():
        dims = [_dim(s) for s in v.shape_symbols]
        sdfg.add_array(
            v.fortran_name,
            shape=dims,
            dtype=dt(v.dtype),
            transient=(v.intent == ''),
            strides=_fortran_strides(dims) if len(dims) > 1 else None,
        )

    for v in builder.scalars.values():
        if v.intent == '':
            sdfg.add_scalar(v.fortran_name, dtype=dt(v.dtype), transient=True)
        else:
            sdfg.add_array(v.fortran_name, shape=(1, ), dtype=dt(v.dtype), transient=False)


def auto_declare_synth(builder, name: str, ctx):
    """Lazy-declare a synthetic scalar minted by the bridge's faithful
    scf.while walker.  ``__sc_N`` names materialise ``scf.if -> T``
    results; ``__al_N`` names come from bare ``fir.alloca`` ops that
    lift-cf-to-scf uses as scratch counters.  Both need an SDFG
    descriptor + an entry in ``builder.scalars`` so ``emit_assign``'s
    existing dispatch (scalar pending, or symbol state-change) can
    fire normally.  Treated as transient ints — they only live for
    the loop's lifetime and are read only by downstream generated
    conditions.
    """
    if name in builder.scalars or name in builder.symbols:
        return
    if not (name.startswith("__sc_") or name.startswith("__al_")):
        return
    # Fake a VarInfo-like record so _add_descriptors-consistent paths work.
    # A ``SimpleNamespace`` is enough — scalar dispatch only reads
    # ``.intent`` and ``.dtype``.
    v = SimpleNamespace(fortran_name=name,
                        intent='',
                        dtype='int32',
                        rank=0,
                        is_dynamic=False,
                        role='scalar',
                        shape_symbols=[],
                        lower_bounds=[])
    builder.scalars[name] = v
    if name not in ctx.sdfg.arrays:
        ctx.sdfg.add_scalar(name, dtype=dace.int32, transient=True)
