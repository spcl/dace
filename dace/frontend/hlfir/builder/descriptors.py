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
    'uint8': dace.uint8,
    'complex64': dace.complex64,
    'complex128': dace.complex128,
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

    Scalar rule: locals (``intent=''``) -> ``dace.data.Scalar`` transient.
    Scalar OUTPUTS (``intent in/out/inout`` other than pure ``in``) land as
    length-1 ``dace.data.Array`` because the caller needs a writable buffer.
    Scalar INPUTS (``intent(in)`` or ``REAL(8), VALUE :: x``) register as
    non-transient ``dace.data.Scalar`` so callers pass plain ``int`` /
    ``float`` and the C++ codegen reads ``x`` directly.
    """
    # Named Fortran symbols (nproma, nlev, …).
    for v in builder.symbols.values():
        sdfg.add_symbol(v.fortran_name, dt(v.dtype))

    # Per-dim ``?`` entries (e.g. ``vn_ie(nproma, nlev+1, nblks_e)`` —
    # ``resolveShapeSyms`` returns ``"?"`` for the arith-derived middle
    # extent) become a synthetic ``<arr>_d<dim>`` name so DaCe sees a
    # legal symbol; the caller-side binding emitter still passes the real
    # extent at call time.  Same shape as the whole-list-empty fallback
    # in the bridge, just applied per-dim.  ``v.shape_symbols`` is a
    # fresh-copy nanobind property on every read, so we materialise the
    # rewritten list in a Python-side dict keyed by name and route the
    # downstream loops through it.
    shape_syms = {}
    for v in builder.arrays.values():
        syms = list(v.shape_symbols)
        for dim, s in enumerate(syms):
            if s == "?":
                syms[dim] = f"{v.fortran_name}_d{dim}"
        shape_syms[v.fortran_name] = syms

    # Synthetic symbols for dims that stayed unresolved after passes.
    # Literal-integer dimensions (e.g. the "3" in ``edge_idx(nc, 3)``) stay
    # as Python ints and do not need a symbol registration.
    known = {v.fortran_name for v in builder.variables}
    for v in builder.arrays.values():
        for s in shape_syms[v.fortran_name]:
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

    # Flang emits internal temporaries with dotted names (e.g.
    # ``.tmp.arrayctor`` for the array constructor backing an
    # ``out = [n, m]``-style RHS, and ``.c.<type>`` /
    # ``.dt.<type>`` for type-info records on derived-type uses).
    # DaCe's ``NestedDict`` treats dots as nesting separators and
    # rejects them as keys.  Skip these declares at descriptor time
    # — the bridge's accesses still reference the original
    # dummy/declare names, so dropping the internals is safe.
    def _is_flang_internal(nm: str) -> bool:
        return nm.startswith(".")

    # Per-axis offset symbols for every array.  ``offset_<arr>_d<i>`` is
    # the value subtracted from the Fortran 1-based index in every
    # memlet (see ``access.py::build_memlet_index``).  Default value for
    # a Fortran array is ``1`` (the standard lb); ``dimension(20:24)``
    # picks up ``20`` from the declare's shape_shift; ``dimension(lo:hi)``
    # with caller-supplied ``lo`` falls through to ``None`` and the
    # symbol stays free on the SDFG signature.  Populated here so
    # ``builder.offset_values`` is fully filled before any AST emission
    # references the symbols in memlet subsets.
    def _offset_value(s: str):
        s = s.strip()
        if s == "?" or not s:
            return None
        if s.lstrip('-').isdigit():
            return int(s)
        # Symbolic lb (e.g. caller-supplied ``arrsize``).  If the symbol
        # is already declared on the SDFG (a known dummy / Fortran sym),
        # pass the name through; sdfg.specialize will alias one symbol
        # to the other.  Otherwise leave unknown so the offset stays
        # free.
        return s if s in sdfg.symbols else None

    for v in builder.arrays.values():
        if _is_flang_internal(v.fortran_name):
            continue
        dims = [_dim(s) for s in shape_syms[v.fortran_name]]
        sdfg.add_array(
            v.fortran_name,
            shape=dims,
            dtype=dt(v.dtype),
            transient=(v.intent == ''),
            strides=_fortran_strides(dims) if len(dims) > 1 else None,
        )
        # Declare an offset symbol per dim, sized from the SDFG array's
        # rank (not ``v.lower_bounds`` which may be shorter for some
        # synth shapes).  Unknown lower bounds default to ``1``.
        rank = len(dims)
        for d in range(rank):
            sym_name = f"offset_{v.fortran_name}_d{d}"
            if sym_name not in sdfg.symbols:
                sdfg.add_symbol(sym_name, dace.int64)
            lb = v.lower_bounds[d] if d < len(v.lower_bounds) else "1"
            builder.offset_values[sym_name] = _offset_value(lb)

    for v in builder.scalars.values():
        if _is_flang_internal(v.fortran_name):
            continue
        if v.intent == '':
            # Local transient scalar.
            sdfg.add_scalar(v.fortran_name, dtype=dt(v.dtype), transient=True)
        elif v.intent in ('out', 'inout'):
            # Scalar OUTPUT must remain a length-1 array on the SDFG
            # signature -- the runtime needs a writable buffer the
            # caller hands in (Python ``float`` would be pass-by-value
            # so updates wouldn't surface on the caller side).
            sdfg.add_array(v.fortran_name, shape=(1, ), dtype=dt(v.dtype), transient=False)
        else:
            # Scalar INPUT (``intent(in)`` or ``REAL(8), VALUE :: x``).
            # Register as a true Scalar -- DaCe accepts plain Python
            # ``int`` / ``float`` for these and the C++ codegen reads
            # ``x`` directly instead of ``x[0]``.  Matches Fortran's
            # pass-by-value semantics (the kernel gets its own copy
            # of the constant).
            sdfg.add_scalar(v.fortran_name, dtype=dt(v.dtype), transient=False)


def declare_synth_array(builder, name: str, shape, dtype: str, ctx):
    """Register a bridge-synthesised transient array on the SDFG and in
    ``builder.arrays``.  Used by the ``kind="declare_transient"`` AST
    handler: when the bridge emits a per-element loop that fills a
    scratch mask before a reduction or select library node, this is the
    one-stop helper that creates the array descriptor.

    ``shape`` is a list of strings; literal-integer entries are parsed
    as Python ints, anything else is treated as a symbol name and looked
    up via ``dace.symbol``.  No-op if ``name`` already exists.
    """
    if name in ctx.sdfg.arrays:
        return
    dims = []
    for s in shape:
        if isinstance(s, int):
            dims.append(s)
            continue
        s_str = str(s).strip()
        if s_str.lstrip('-').isdigit():
            dims.append(int(s_str))
        else:
            if s_str not in ctx.sdfg.symbols:
                ctx.sdfg.add_symbol(s_str, dace.int64)
            dims.append(dace.symbol(s_str))
    # Fortran-style transient: rank > 1 → column-major strides so the
    # matmul / transpose / dot_product library nodes (which inherit
    # layout from the source operands' strides) write the result in the
    # same layout the bridge-declared dummy arrays use.  Single-rank
    # transients (or scalars) take DaCe's default contiguous stride.
    strides = None
    if len(dims) > 1:
        acc = 1
        strides = []
        for d in dims:
            strides.append(acc)
            acc = acc * d
    ctx.sdfg.add_array(name, shape=dims, dtype=dt(dtype), transient=True, strides=strides)
    # Mirror the entry into ``builder.arrays`` so subsequent emit_assign
    # / emit_libcall calls find it via the existing arrays-dict lookups.
    from types import SimpleNamespace
    builder.arrays[name] = SimpleNamespace(
        fortran_name=name,
        intent='',
        dtype=dtype,
        rank=len(shape),
        is_dynamic=False,
        role='array',
        shape_symbols=[str(s) for s in shape],
        lower_bounds=['1'] * len(shape),
    )
    # Per-axis offset symbols + values (always 1 for bridge-synthesised
    # transients — they're allocated fresh with Fortran's default lb).
    for d in range(len(shape)):
        sym_name = f"offset_{name}_d{d}"
        if sym_name not in ctx.sdfg.symbols:
            ctx.sdfg.add_symbol(sym_name, dace.int64)
        builder.offset_values[sym_name] = 1


def emit_declare_transient(builder, ctx, n, region):
    """Handler for ASTNode kind=\"declare_transient\".

    Reads ``n.target`` (name), ``n.expr`` (dtype as string), and shape
    from ``n.accesses[0].index_exprs`` (one string per dim).  Calls
    ``declare_synth_array`` to register the SDFG descriptor.
    """
    shape = list(n.accesses[0].index_exprs) if n.accesses else []
    declare_synth_array(builder, n.target, shape, n.expr or "int32", ctx)


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
