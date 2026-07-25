# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Backend-neutral symbolic engine seam.

DaCe must never construct sympy objects directly. Every file that used to
``import sympy`` (as ``sympy`` or ``sp``) instead imports this module::

    from dace import symbolic_engine as sp        # was: import sympy as sp
    from dace import symbolic_engine as sympy      # was: import sympy

All symbolic construction and querying then flows through a single backend
indirection point, so the engine can be swapped without touching call sites.

The backend is selected once, at import, by ``DACE_SYMBOLIC_BACKEND``:

- ``sympy`` (default): every attribute forwards verbatim to :mod:`sympy` — a
  pure identity layer, zero behavior change. This is Phase A: the mechanical
  migration of the ~120 sympy-importing files is gated only by the DaCe sweep.

- ``idxalg``: the pervasive index/shape arithmetic (``Symbol``, ``Add``/``Mul``/
  ``Pow``, ``Min``/``Max``, ``floor``/``ceiling``/``Mod``, relationals,
  ``sympify``, ``simplify``, and the ``Basic``/``Expr``/``Symbol``/``Number``/...
  isinstance heads) resolves to the idxalg backend
  (:mod:`idxalg.sympy_compat`), whose ``Expr`` quacks like ``sympy.Basic`` and
  is typed + rational-leak-free. The low-frequency non-index features
  (``Wild``/``match``, ``solve``, ``Poly``, ``ask``/``Q``, ``Piecewise``/
  ``Sum``/``cse``, ``diff``, singletons like ``oo``/``S``) keep delegating to
  the retained sympy island. Real sympy is bound into the idxalg backend so
  isinstance still accepts genuine sympy objects during the hybrid period.

**idxalg is an optional dependency.** DaCe's only hard symbolic requirement is
sympy. ``idxalg`` is imported nowhere in the tree except inside the opt-in branch
below, so with the default backend it never enters ``sys.modules`` and DaCe works
normally whether or not it is installed (pinned by
``tests/symbolic/optional_idxalg_backend_test.py``). Opting in without it
installed raises with the variable and package named; it deliberately does *not*
fall back to sympy, since an opt-in that silently ran the other engine would make
a backend comparison meaningless.

``symbolic.py`` is the seam's *implementation* (it constructs the backend's
objects), not a consumer, so it is allowed to touch the backend directly.

Keeping this a module-level ``__getattr__`` (PEP 562) means the seam covers the
*entire* surface — every class, singleton and submodule — without enumerating
names that would drift out of date. Only the names idxalg overrides are listed;
everything else forwards to sympy.
"""
import os

import sympy as backend

_BACKEND_NAME = os.environ.get("DACE_SYMBOLIC_BACKEND", "sympy").lower()

# The single point of backend selection.
BACKEND = backend

# Names the idxalg backend serves natively — the pervasive index/shape arithmetic plus the
# isinstance heads. Everything NOT in this set forwards to sympy (the hard, low-frequency tier
# and all singletons/submodules), so the two-world hybrid stays coherent during the migration.
_IDXALG_NAMES = frozenset({
    # constructors
    "Symbol",
    "Integer",
    "Float",
    "Rational",
    "Add",
    "Mul",
    "Pow",
    "Mod",
    "Min",
    "Max",
    "floor",
    "ceiling",
    "Abs",
    "Not",
    "And",
    "Or",
    "Eq",
    "Ne",
    "Equality",
    "Unequality",
    "StrictLessThan",
    "LessThan",
    "StrictGreaterThan",
    "GreaterThan",
    "Function",
    "sympify",
    "simplify",
    "preorder_traversal",
    # isinstance heads (umbrella / kind)
    "Basic",
    "Expr",
    "Atom",
    "AtomicExpr",
    "Number",
    "Boolean",
    "Relational",
})

if _BACKEND_NAME == "idxalg":
    # idxalg is an OPTIONAL dependency: the default backend is sympy, so this import is reached
    # only when a caller explicitly opts in. Do not fall back silently -- an opt-in that quietly
    # ran on the other engine would make a backend A/B meaningless -- but do say what is missing.
    try:
        from idxalg import sympy_compat as _idx
    except ImportError as ex:
        raise ImportError('DACE_SYMBOLIC_BACKEND=idxalg requires the "idxalg" package, which is '
                          'not installed. Install it or unset the variable to use the default '
                          'sympy backend.') from ex

    # Bind real sympy so idxalg isinstance heads also accept genuine sympy objects during the
    # hybrid period, and rare heads fall back to real sympy constructors.
    _idx.bind_sympy(backend)

    def _to_idx(obj):
        """Structurally convert a sympy expression to an idxalg one (the hybrid boundary: while
        symbolic.py still produces sympy objects, an idxalg operation may be handed one). Walks
        ``.func``/``.args``, mapping each head to the matching idxalg constructor; DaCe's custom
        ``int_floor``/``int_ceil`` Functions map to true floor/ceil division; anything unknown
        becomes an opaque function preserving its head name. Registered as idxalg's foreign
        coercer, so it is consulted only on the cold path (never for Expr/int/float/bool)."""
        if isinstance(obj, _idx.Expr):
            return obj
        if isinstance(obj, bool):
            return _idx.Integer(1 if obj else 0)
        if isinstance(obj, int):
            return _idx.Integer(obj)
        if isinstance(obj, float):
            return _idx.Float(obj)
        if not isinstance(obj, backend.Basic):
            return None
        if obj.is_Symbol:
            a = obj.assumptions0
            # Preserve the DaCe symbol's exact width/signedness across the boundary. A DaCe
            # ``symbol`` carries an authoritative ``.dtype`` typeclass; map it to the idxalg dtype
            # string so int16/int32/uint32/float stay themselves rather than collapsing to int64.
            return _idx_symbol(obj.name, _symbol_idxstr(obj), bool(a.get("positive")), bool(a.get("nonnegative")))
        if obj.is_Integer:
            return _idx.Integer(int(obj))
        if obj.is_Float:
            return _idx.Float(float(obj))
        if obj.is_Rational:  # non-integer rational: shouldn't occur in index math; keep the value
            return _idx.Float(float(obj))
        if obj is backend.true:
            return _idx.Integer(1)
        if obj is backend.false:
            return _idx.Integer(0)
        f = obj.func
        # floor/ceiling: recover the integer numerator/denominator BEFORE converting children, so
        # sympy's rationalized ``ceiling((N-1)/8) == ceiling(Mul(N-1, 1/8))`` becomes integer
        # ceildiv(N-1, 8) rather than ceildiv(Float, ...) — the whole point of idxalg over ℚ.
        if f is backend.floor or f is backend.ceiling:
            num, den = obj.args[0].as_numer_denom()
            ni, di = _to_idx(num), _to_idx(den)
            if ni is None or di is None:
                return None
            return _idx.floor(ni, di) if f is backend.floor else _idx.ceiling(ni, di)
        # A rational-coefficient product ``(p/q)*rest`` (sympy's spelling of ``rest/q`` when the
        # numerator is unknown, e.g. a leaked ``N/8`` in a subset) is integer floor division in
        # DaCe's index world — never real ``0.125*N``, which idxalg rightly rejects as a kind
        # crossing. Recover the integer floordiv so the rational leak never reaches the core.
        if f is backend.Mul:
            coeff, rest = obj.as_coeff_Mul()
            if coeff.is_Rational and not coeff.is_Integer:
                ri = _to_idx(rest)
                if ri is None:
                    return None
                num = ri if coeff.p == 1 else _idx.Mul(_idx.Integer(coeff.p), ri)
                return _idx.floor(num, coeff.q)
        kids = [_to_idx(x) for x in obj.args]
        if any(k is None for k in kids):
            return None
        conv = _HEAD_CONV.get(f)
        if conv is not None:
            return conv(kids)
        name = type(obj).__name__
        if name == "int_floor":
            return _idx.floor(kids[0], kids[1])
        if name == "int_ceil":
            return _idx.ceiling(kids[0], kids[1])
        return _idx.Function(name)(*kids)

    # sympy head class -> a builder over already-converted idxalg children. Resolved once.
    _HEAD_CONV = {
        backend.Add: lambda k: _idx.Add(*k),
        backend.Mul: lambda k: _idx.Mul(*k),
        backend.Pow: lambda k: _idx.Pow(k[0], k[1]),
        backend.Min: lambda k: _idx.Min(*k),
        backend.Max: lambda k: _idx.Max(*k),
        backend.Mod: lambda k: _idx.Mod(k[0], k[1]),
        backend.Abs: lambda k: _idx.Abs(k[0]),
        backend.Not: lambda k: _idx.Not(k[0]),
        backend.And: lambda k: _idx.And(*k),
        backend.Or: lambda k: _idx.Or(*k),
        backend.Equality: lambda k: _idx.Eq(k[0], k[1]),
        backend.Unequality: lambda k: _idx.Ne(k[0], k[1]),
        backend.StrictLessThan: lambda k: _idx.StrictLessThan(k[0], k[1]),
        backend.LessThan: lambda k: _idx.LessThan(k[0], k[1]),
        backend.StrictGreaterThan: lambda k: _idx.StrictGreaterThan(k[0], k[1]),
        backend.GreaterThan: lambda k: _idx.GreaterThan(k[0], k[1]),
    }

    _idx.set_foreign_coercer(_to_idx)

    _FROM_CONV = {
        "Add": lambda a: backend.Add(*a),
        "Mul": lambda a: backend.Mul(*a),
        "Pow": lambda a: backend.Pow(a[0], a[1]),
        "Min": lambda a: backend.Min(*a),
        "Max": lambda a: backend.Max(*a),
        "Mod": lambda a: backend.Mod(a[0], a[1]),
        "floor": lambda a: backend.floor(a[0] / a[1]) if len(a) == 2 else backend.floor(a[0]),
        "ceiling": lambda a: backend.ceiling(a[0] / a[1]) if len(a) == 2 else backend.ceiling(a[0]),
        "Not": lambda a: backend.Not(a[0]),
        "And": lambda a: backend.And(*a),
        "Or": lambda a: backend.Or(*a),
        "Eq": lambda a: backend.Eq(a[0], a[1]),
        "Ne": lambda a: backend.Ne(a[0], a[1]),
        "StrictLessThan": lambda a: backend.StrictLessThan(a[0], a[1]),
        "LessThan": lambda a: backend.LessThan(a[0], a[1]),
        "StrictGreaterThan": lambda a: backend.StrictGreaterThan(a[0], a[1]),
        "GreaterThan": lambda a: backend.GreaterThan(a[0], a[1]),
    }

    _IDX_DTYPE_MAP: dict = {}

    def _dtype_tc(dstr: str):
        """Map an idxalg dtype string (``int64``/``float64``/...) to a DaCe ``typeclass``. Built
        once, lazily, to avoid importing ``dace.dtypes`` at seam-import time."""
        if not _IDX_DTYPE_MAP:
            from dace import dtypes as dt
            _IDX_DTYPE_MAP.update({
                "int8": dt.int8,
                "int16": dt.int16,
                "int32": dt.int32,
                "int64": dt.int64,
                "uint8": dt.uint8,
                "uint16": dt.uint16,
                "uint32": dt.uint32,
                "uint64": dt.uint64,
                "float32": dt.float32,
                "float64": dt.float64,
                "bool": dt.bool_,
            })
        return _IDX_DTYPE_MAP.get(dstr, _IDX_DTYPE_MAP["int32"])

    _TC_TO_IDXSTR: dict = {}

    def _idxstr_of_tc(tc) -> str:
        """DaCe ``typeclass`` -> idxalg dtype string (the reverse of ``_dtype_tc``). Unknown or
        compound typeclasses (pointer/vector/...) fall back to the widest safe integer."""
        if not _TC_TO_IDXSTR:
            _dtype_tc("int32")  # ensure the forward map is built
            for k, v in _IDX_DTYPE_MAP.items():
                _TC_TO_IDXSTR[v] = k
        return _TC_TO_IDXSTR.get(tc, "int64")

    def _symbol_idxstr(obj) -> str:
        """The idxalg dtype string for a symbol crossing the boundary. A DaCe ``symbol`` yields its
        exact declared dtype; a bare sympy symbol is classified from its assumptions (the index
        world is integer unless the symbol is explicitly real-but-not-integer)."""
        from dace import symbolic as dsym
        if isinstance(obj, dsym.symbol):
            return _idxstr_of_tc(obj.dtype)
        a = obj.assumptions0
        if a.get("integer"):
            return "int64"
        if a.get("real"):
            return "float64"
        return "int64"

    def _int_bounds(dstr: str, positive: bool, nonnegative: bool):
        """Inclusive ``[lo, hi]`` for an integer dtype string, tightened by the sign assumption."""
        signed = dstr[0] == "i"
        bits = int(dstr[3:]) if signed else int(dstr[4:])
        if signed:
            lo, hi = -(2**(bits - 1)), 2**(bits - 1) - 1
        else:
            lo, hi = 0, 2**bits - 1
        if positive:
            lo = max(lo, 1)
        elif nonnegative:
            lo = max(lo, 0)
        return lo, hi

    def _idx_symbol(name: str, dstr: str, positive: bool, nonnegative: bool):
        """Declare (or reference) an idxalg symbol with the exact width/signedness ``dstr``. Integer
        types carry a sign-tightened range; on a no-shadow conflict (the name is already declared
        with a different range) reference the existing symbol, matching DaCe's one-symbol-per-name
        (``equalize_symbol``) contract."""
        try:
            if dstr[0] == "i" or dstr[0] == "u":
                lo, hi = _int_bounds(dstr, positive, nonnegative)
                return _idx.Expr(_idx._CTX.symbol(name, dstr, lo, hi))
            return _idx.Expr(_idx._CTX.symbol(name, dstr))
        except ValueError:
            return _idx.Expr(_idx._CTX.symbol_lazy(name))

    def _from_idx(e):
        """Convert an idxalg expression back to sympy (the `_sympy_` protocol): a still-sympy code
        path can then `sympify` an idxalg value handed to it. Symbols are rebuilt as DaCe
        ``symbol``s (carrying ``.dtype`` and assumptions), so a round-tripped symbol that lands in a
        descriptor shape or the SDFG symbol table still answers ``.dtype``; unknown opaque heads
        become sympy Functions of the same name. floor/ceiling re-express as sympy floor/ceiling of
        a quotient."""
        if not isinstance(e, _idx.Expr):
            return e
        name = str(e.func)
        if name == "Basic":
            # Umbrella head over Cast/Undef/IfExpr. A width-promotion ``Cast`` (int32->int64, ...) is
            # a value-preserving widening; sympy index math is untyped, so render it transparently as
            # its operand rather than an opaque ``Basic(...)`` head that sympy cannot compare.
            if _idx._CTX.kind(e._e) == "Cast":
                return _from_idx(e.args[0])
        if name == "Mul" and _idx._CTX.kind(e._e) == "Div":
            # A real-division node shares the Mul head marker; render it as sympy division, not a
            # product (`work/depth` in the cost models must not become `work*depth`).
            num, den = (_from_idx(a) for a in e.args)
            return num / den
        if name == "Symbol":
            # Preserve the covering-relevant assumptions (positive/nonnegative/integer) and the
            # dtype across the round trip. Without the assumptions, ``nng()`` — which rebuilds a
            # symbol as ``sp.Symbol(name, nonnegative=True)`` and subs it into a sympy expr — loses
            # positivity when sympy sympifies the idxalg replacement back, so ``N >= N-1`` no longer
            # folds to True. Without the dtype, SDFG validation's ``sym.dtype`` read crashes.
            from dace import symbolic as dsym
            a = e.assumptions0
            kw = {k: True for k in ("positive", "nonnegative", "integer") if a.get(k)}
            return dsym.symbol(e.name, dtype=_dtype_tc(_idx._CTX.dtype(e._e)), **kw)
        if name == "Integer":
            return backend.Integer(int(e))
        if name == "Float":
            return backend.Float(float(repr(e)))
        args = [_from_idx(a) for a in e.args]
        conv = _FROM_CONV.get(name)
        if conv is not None:
            return conv(args)
        return backend.Function(name)(*args)

    _idx.set_sympy_converter(_from_idx)

    def __getattr__(name: str):
        # PEP 562: only invoked for names not bound as module globals.
        if name in _IDXALG_NAMES:
            # sympy's ``Expr`` maps to the idxalg umbrella head kept under ``Expr_`` (the wrapper
            # class itself is ``Expr``, which we must not shadow).
            return _idx.Expr_ if name == "Expr" else getattr(_idx, name)
        return getattr(backend, name)

    def __dir__() -> list:
        return sorted(set(dir(backend)) | _IDXALG_NAMES)

else:

    def __getattr__(name: str):
        # PEP 562: forwards every symbolic name to sympy with no overhead on normal imports.
        return getattr(backend, name)

    def __dir__() -> list:
        return dir(backend)
