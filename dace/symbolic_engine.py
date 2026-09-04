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
import functools
import math
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
})

# Heads sympy keeps in a submodule, not at top level. Real globals: the forwarding `__getattr__`
# raises `AttributeError` on them (an `ImportError` at a `from` site -- the sympy backend had no
# `Boolean` at all), and no consumer should need to know which submodule hides a head.
Boolean = backend.logic.boolalg.Boolean
Relational = backend.core.relational.Relational

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

    # NOT aligning idxalg's default width to DaCe's int32 here, though the wire format argues for it:
    # doing so makes `int32(qm)` an identity cast, which the core correctly folds away and DaCe then
    # loses a typecast it emits deliberately for codegen. The real fix is for a dtype call on a
    # non-constant to stay structural rather than become a foldable Cast (see TASKS T18).

    # Bind real sympy so idxalg isinstance heads also accept genuine sympy objects during the
    # hybrid period, and rare heads fall back to real sympy constructors.
    _idx.bind_sympy(backend)

    # The submodule-resolved heads above, now this backend's own.
    Boolean = _idx.Boolean
    Relational = _idx.Relational

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
        if isinstance(obj, backend.Wild):
            # Wild/.match stays on the retained-sympy island (module docstring): a Wild is a Symbol
            # SUBCLASS, so `obj.is_Symbol` below would otherwise silently coerce it into a plain
            # idxalg symbol and the pattern loses its wildcard semantics entirely.
            return None
        if obj.is_Symbol:
            a = obj.assumptions0
            # Preserve the DaCe symbol's exact width/signedness across the boundary. A DaCe
            # ``symbol`` carries an authoritative ``.dtype`` typeclass; map it to the idxalg dtype
            # string so int16/int32/uint32/float stay themselves rather than collapsing to int64.
            return _idx_symbol(obj.name, _symbol_idxstr(obj), bool(a.get("positive")), bool(a.get("nonnegative")))
        if obj is backend.oo or obj is -backend.oo:
            # ±oo is a sympy singleton, not a `Float`, so `is_Float` below never catches it and it
            # used to arrive as an opaque `Infinity()` call -- which then matched nothing, so
            # `has(oo)` on a converted expression read False.
            return _idx.Float(math.inf if obj is backend.oo else -math.inf)
        if obj.is_Integer:
            return _idx.Integer(int(obj))
        if obj.is_Float:
            return _idx.Float(float(obj))
        if obj.is_Rational:  # non-integer rational: shouldn't occur in index math; keep the value
            from fractions import Fraction
            value = float(obj)
            if Fraction(*value.as_integer_ratio()) != Fraction(int(obj.p), int(obj.q)):
                raise ValueError(f"rational {obj} does not survive an exact float64 round-trip")
            return _idx.Float(value)
        if obj is backend.true:
            return _idx.Integer(1)
        if obj is backend.false:
            return _idx.Integer(0)
        f = obj.func
        if f is backend.Piecewise or f is backend.Sum:
            # Stay on the retained sympy island (module docstring): opaque-swallowing either would
            # silently discard branch selection (Piecewise) or bound-variable/limits semantics (Sum).
            return None
        # floor/ceiling: recover the integer numerator/denominator BEFORE converting children, so
        # sympy's rationalized ``ceiling((N-1)/8) == ceiling(Mul(N-1, 1/8))`` becomes integer
        # ceildiv(N-1, 8) rather than ceildiv(Float, ...) — the whole point of idxalg over ℚ.
        if f is backend.floor or f is backend.ceiling:
            num, den = obj.args[0].as_numer_denom()
            ni, di = _to_idx(num), _to_idx(den)
            if ni is None or di is None:
                return None
            return _idx.floor(ni, di) if f is backend.floor else _idx.ceiling(ni, di)
        # A rational-coefficient product ``(p/q)*rest`` is sympy's spelling of ``rest/q``. Recovering
        # it as integer floordiv is only correct when ``rest`` is PROVABLY an integer -- gated on
        # sympy's own assumption system (``rest.is_integer``), never on the idxalg side's dtype: a
        # bare unassumed Symbol defaults there to "int64", which is a guess, not evidence. On a real
        # operand ``w/8`` is genuine real division (``work/depth`` in the cost models), and floordiv
        # from that guessed default turned 0.9375 into 0.0 — a miscompile of our own making.
        if f is backend.Mul:
            coeff, rest = obj.as_coeff_Mul()
            if coeff.is_Rational and not coeff.is_Integer:
                ri = _to_idx(rest)
                if ri is None:
                    return None
                num = ri if coeff.p == 1 else _idx.Mul(_idx.Integer(coeff.p), ri)
                if rest.is_integer:
                    return _idx.floor(num, coeff.q)
                return num / _idx.Integer(coeff.q)
        is_rel = f in _RELATIONAL_HEADS
        kids = []
        for x in obj.args:
            if is_rel and (x is backend.true or x is backend.false):
                # Do not pre-flatten a literal True/False that is a direct relational child: the
                # facade's `_rel` bool-literal special case (which triggers `X != True -> Not(X)`)
                # only fires on a raw Python bool, never on an already-converted `Integer(1/0)`.
                kids.append(x is backend.true)
            else:
                kids.append(_to_idx(x))
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
    # The relational heads by identity. sympy 1.14 does not export `Relational` at top level, so an
    # `issubclass(f, backend.Relational)` test raises `AttributeError` instead of classifying; these
    # are exactly the heads `_HEAD_CONV` below knows how to convert anyway.
    _RELATIONAL_HEADS = frozenset({
        backend.Equality,
        backend.Unequality,
        backend.StrictLessThan,
        backend.LessThan,
        backend.StrictGreaterThan,
        backend.GreaterThan,
    })

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
        "Abs": lambda a: backend.Abs(a[0]),
        # A one-argument floor/ceil is the real-valued sympy function; the two-argument form is
        # integer division and is spelled through `_SPELLED_CLASS` below. Rendering that as sympy's
        # `floor(a/b)` would reintroduce the rational leak this engine exists to avoid.
        "floor": lambda a: backend.floor(a[0]),
        # `int_ceil` has NO `__` variant, because Python has no ceiling operator to print as -- so
        # unlike `int_floor` it renders the same however it was written. Asking for `__int_ceil`
        # here raised, since the parser table has no such class.
        "ceiling": lambda a: _dace_op("int_ceil", a) if len(a) == 2 else backend.ceiling(a[0]),
        # And/Or are handled ahead of this table (they need their operands' truthiness casts peeled),
        # so only `Not` -- which sympy accepts on a non-Boolean -- is listed.
        "Not": lambda a: backend.Not(a[0]),
        "Eq": lambda a: backend.Eq(a[0], a[1]),
        "Ne": lambda a: backend.Ne(a[0], a[1]),
        "StrictLessThan": lambda a: backend.StrictLessThan(a[0], a[1]),
        "LessThan": lambda a: backend.LessThan(a[0], a[1]),
        "StrictGreaterThan": lambda a: backend.StrictGreaterThan(a[0], a[1]),
        "GreaterThan": lambda a: backend.GreaterThan(a[0], a[1]),
    }

    # The ops DaCe writes two ways: as an operator, and as a named call. DaCe models each spelling as
    # its own Function class (the `__`-prefixed one prints as the operator), and so answers
    # `a // b != int_floor(a, b)` -- two names for one value. idxalg interns both to ONE node, which
    # compares equal as it should, and records the form the source used alongside the handle; the
    # class is therefore chosen here, from that record, instead of being fixed per head.
    #
    # Only the MODELLED op needs a table. The bitwise and shift ops are opaque, so their head name
    # already carries the form (idxalg reports `__bitwise_or` for the operator form, the bare name
    # for the call form) and the generic lookup below resolves each to the right class unaided.
    # `int_ceil` and the logical shifts appear nowhere here: none has a `__` variant, because none
    # has an operator to print as.
    _SPELLED_CLASS = {
        ("floor", 2): ("int_floor", "__int_floor"),
    }

    _IDX_DTYPE_MAP: dict = {}

    def _dtype_tc(dstr: str):
        """Map an idxalg dtype string (``int64``/``float64``/...) to a DaCe ``typeclass``. Built
        once, lazily, to avoid importing ``dace.dtypes`` at seam-import time. Raises on an unknown
        string rather than defaulting to int32 -- a silent default there previously mapped every
        missing width (float16, complex128, ...) to a 32-bit integer typeclass."""
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
                "float16": dt.float16,
                "bfloat16": dt.bfloat16,
                "float8_e4m3fn": dt.float8_e4m3fn,
                "float8_e5m2": dt.float8_e5m2,
                "float32": dt.float32,
                "float64": dt.float64,
                "complex64": dt.complex64,
                "complex128": dt.complex128,
                "bool": dt.bool_,
            })
        if dstr == "opaque":
            # `Opaque` is idxalg's "no type known" marker (a black-box call, a container base), not a
            # missing entry. DaCe has no opaque typeclass, and its own `symbol()` default is the
            # honest rendering of an undeclared width.
            return _IDX_DTYPE_MAP["int32"]
        if dstr not in _IDX_DTYPE_MAP:
            raise ValueError(f"unknown idxalg dtype string {dstr!r}")
        return _IDX_DTYPE_MAP[dstr]

    _TC_TO_IDXSTR: dict = {}

    def _idxstr_of_tc(tc) -> str:
        """DaCe ``typeclass`` -> idxalg dtype string (the reverse of ``_dtype_tc``). Raises on an
        unknown or compound typeclass (pointer/vector/...) rather than silently declaring it a
        64-bit integer -- the same silent-default bug as ``_dtype_tc``, in reverse."""
        if not _TC_TO_IDXSTR:
            _dtype_tc("int32")  # ensure the forward map is built
            for k, v in _IDX_DTYPE_MAP.items():
                _TC_TO_IDXSTR[v] = k
        if tc not in _TC_TO_IDXSTR:
            raise ValueError(f"no idxalg dtype string for DaCe typeclass {tc!r}")
        return _TC_TO_IDXSTR[tc]

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
        """Declare (or reference) an idxalg symbol with the exact width/signedness ``dstr``.

        idxalg's no-shadow rule makes the declared dtype and range part of a symbol's identity. DaCe
        has the opposite rule: a name IS the symbol, which is exactly what ``equalize_symbol`` exists
        to enforce, so the same name legitimately arrives declared several ways (``nng()`` rebuilds
        ``N`` as nonnegative; a map parameter is int32 in one region and int64 in another). Referencing
        the existing symbol is therefore a faithful rendering of DaCe's contract, not a guess -- an
        earlier revision raised here instead and broke six tests that re-declare a live symbol.
        """
        try:
            if dstr[0] == "i" or dstr[0] == "u":
                lo, hi = _int_bounds(dstr, positive, nonnegative)
                return _idx.Expr(_idx._CTX.symbol(name, dstr, lo, hi))
            return _idx.Expr(_idx._CTX.symbol(name, dstr))
        except ValueError:
            return _idx.Expr(_idx._CTX.symbol_lazy(name))

    def is_native_symbol(obj) -> bool:
        """A symbol THIS backend built -- the one leaf carrying an authoritative dtype."""
        return isinstance(obj, _idx.Expr) and _idx._CTX.kind(obj._e) == "Symbol"

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
        if name == "Basic" and _idx._CTX.kind(e._e) == "Cast":
            # A WIDENING cast is value-preserving, so render it transparently as its operand rather
            # than an opaque head sympy cannot compare. A NARROWING one is not: dropping
            # ``uint16(x)`` changes the value, and printed ``x`` no longer round-trips to a cast.
            # Widening is only decidable WITHIN one dtype KIND (int-family / float / complex / bool):
            # a float64->int32 Cast must never look like a widen just because ``outer`` is "wider" in
            # bits alone -- that silently elided a genuine truncating cast (CRITICAL).
            inner, outer = _idx._CTX.dtype(e.args[0]._e), _idx._CTX.dtype(e._e)
            ow, iw = _int_width(outer), _int_width(inner)
            if ow is not None and iw is not None and ow[0] == iw[0] and ow[1] >= iw[1]:
                return _from_idx(e.args[0])
            from dace.symbolic import _CAST_CLASSES
            cast_cls = _CAST_CLASSES.get(outer)
            arg = _from_idx(e.args[0])
            return cast_cls(arg) if cast_cls is not None else backend.Function(outer)(arg)
        if name == "Mul" and _idx._CTX.kind(e._e) == "Div":
            # A real-division node shares the Mul head marker; render it as sympy division, not a
            # product (`work/depth` in the cost models must not become `work*depth`).
            num, den = (_from_idx(_peel_promotion(e, a)) for a in e.args)
            return num / den
        if name == "Symbol":
            # Preserve the covering-relevant assumptions (positive/nonnegative/integer/real) and the
            # dtype across the round trip. Without the assumptions, ``nng()`` — which rebuilds a
            # symbol as ``sp.Symbol(name, nonnegative=True)`` and subs it into a sympy expr — loses
            # positivity when sympy sympifies the idxalg replacement back, so ``N >= N-1`` no longer
            # folds to True. Without the dtype, SDFG validation's ``sym.dtype`` read crashes. Without
            # ``real``, a float symbol round-trips with ``is_real is None``.
            a = e.assumptions0
            flags = tuple(k for k in ("positive", "nonnegative", "integer", "real") if a.get(k))
            return _rebuild_symbol(e.name, _idx._CTX.dtype(e._e), flags)
        if name == "Boolean":
            # Falls through to the generic Function arm otherwise, printing `Boolean()` -- a zero-arg
            # call that has lost the value entirely.
            return backend.true if bool(e) else backend.false
        if name in ("Integer", "Float"):
            from dace.symbolic import TypedConstant
            value = int(e) if name == "Integer" else float(repr(e))
            if name == "Float" and (value == math.inf or value == -math.inf):
                # sympy has no infinite `Float`; ±oo is a distinct singleton, and the retained sympy
                # island is where DaCe's own `inf` spelling already lands.
                return backend.oo if value > 0 else -backend.oo
            dstr = _idx._CTX.dtype(e._e)
            # Only a NON-default width needs a TypedConstant; a default-width literal stays a plain
            # sympy number so ordinary index arithmetic is not littered with typed wrappers.
            if dstr not in _DEFAULT_CONST_DTYPES:
                return TypedConstant(value, _dtype_tc(dstr))
            return backend.Integer(value) if name == "Integer" else backend.Float(value)
        if name == "Number":
            # `Number` marks a Complex leaf here -- Integer/Float have their own dedicated heads
            # above, so by elimination this is the constant the docstring warns falls through to an
            # opaque, value-losing `Number()` otherwise. idxalg exposes no numeric accessor for a
            # Complex leaf, so its real/imaginary parts are read off the printed "(re + imj)" form,
            # whose separator is a fixed literal " + " regardless of either part's sign.
            kind = _idx._CTX.kind(e._e)
            if kind != "Complex":
                raise ValueError(f"unexpected Number-headed idxalg node of kind {kind!r}")
            from dace.symbolic import TypedConstant
            re_part, im_part = str(e)[1:-1].split(" + ", 1)
            value = complex(float(re_part), float(im_part[:-1]))  # im_part ends in the literal "j"
            return TypedConstant(value, _dtype_tc(_idx._CTX.dtype(e._e)))
        if name in ("[]", "Subscript"):
            # Container access: idxalg's own parser always spells it "[]"; a still-sympy `Subscript`
            # hybrid-converted through `_to_idx`'s generic fallback instead carries the class name
            # literally -- both share the (base, *indices) shape. Without this, either spelling comes
            # back as a bare `Function(...)` call, and `Subscript`'s overridden `.free_symbols`
            # (which excludes the container) never runs.
            from dace.symbolic import Subscript
            b = e.args[0]
            base = backend.Symbol(str(b.func)) if b.is_Function and not b.args else _from_idx(b)
            return Subscript(base, *(_from_idx(a) for a in e.args[1:]))
        if name == "Attr" or name.startswith("."):
            # Field access: idxalg's own parser bakes the field name into the head ("." + field, one
            # child); a hybrid-converted sympy `Attr` instead carries it as a second child. Same
            # container-base rule as `Subscript` above.
            from dace.symbolic import Attr
            b = e.args[0]
            base = backend.Symbol(str(b.func)) if b.is_Function and not b.args else _from_idx(b)
            field = _from_idx(e.args[1]) if name == "Attr" else backend.Symbol(name[1:])
            return Attr(base, field)
        if name in ("And", "Or", "Not"):
            return _from_idx_bool(name, e)
        if _idx._CTX.kind(e._e) == "IfExpr":
            # A select is a structural node here, so it carries no head NAME to resolve -- `.func`
            # reports the umbrella `Basic` and the generic arm below rendered `Basic(c, t, f)`, losing
            # the select entirely. Its condition gets the same truthiness peel as `And`/`Or`.
            #
            # DaCe spells a select two ways and lowers them differently: `ITE` is the branchless blend
            # its vectorizer recognizes, `IfExpr` the branching conditional. One node either way, so
            # the recorded FORM chooses the class -- which is also what keeps `isinstance(x, ITE)`
            # answering for a value this backend built.
            cond, then, els = e.args
            head = "ITE" if _idx._CTX.spelling(e._e) == "alternate" else "IfExpr"
            return _dace_op(head, [_from_idx(_peel_truthiness(cond)), _from_idx(then), _from_idx(els)])
        if name == "Pow" and _idx._CTX.spelling(e._e) == "alternate":
            # `ipow` is the integer power, and DaCe lowers it apart from the real one on purpose:
            # `dace::math::pow` returns a double and will not compile where an array size is wanted.
            # One node here, so the recorded FORM is what tells the two apart -- without it a
            # round-trip through this converter downgraded every `ipow` to `pow`.
            return _dace_op("ipow", [_from_idx(a) for a in e.args])
        # `+ - *` convert their operands to the type the operation evaluates in, in Python and in C
        # alike, so a promotion cast to exactly that type spells a conversion both perform anyway.
        peel = _peel_promotion if name in ("Add", "Mul") else lambda _parent, child: child
        args = [_from_idx(peel(e, a)) for a in e.args]
        spelled = _SPELLED_CLASS.get((name, len(args)))
        if spelled is not None:
            return _dace_op(spelled[_idx._CTX.spelling(e._e) == "operator"], args)
        conv = _FROM_CONV.get(name)
        if conv is not None:
            return conv(args)
        # A DEFINED sympy function (``sin``, ``sqrt``, a cast class) must come back as itself, not as
        # ``Function(name)``: an undefined function is an *applied undef*, which every
        # symbol-extraction helper then reports as a user function, so ``sin`` leaks out of
        # ``free_symbols_and_functions``. Resolve through DaCe's own parser table rather than a second
        # list of names here -- two lists would drift, and the drift is silent.
        cls = _sympy_function_class(name)
        return cls(*args) if cls is not None else backend.Function(name)(*args)

    def _rebuild_symbol(name: str, dstr: str, flags: tuple):
        """The DaCe ``symbol`` for a name/dtype/assumption triple.

        Deliberately NOT memoized, though a profile puts sympy's fact-deduction engine at the top of
        the conversion cost. `dace.symbolic.symbol.__new__` bypasses sympy's own symbol cache on
        purpose -- it assigns `self.dtype` on the instance, so two references to one cached object
        would modify each other -- and handing out a shared instance here would reintroduce exactly
        that aliasing. The construction cost is sympy's to fix, or Phase C's to stop paying.
        """
        from dace import symbolic as dsym
        # Not the `symbol` FACTORY: here it builds one of ours, so the converter would answer a
        # sympy request with an idxalg value and recurse through `_sympy_()` forever.
        return dsym.sympy_symbol(name, dtype=_dtype_tc(dstr), **{k: True for k in flags})

    def _from_idx_bool(name: str, e):
        """`And`/`Or`/`Not` rendered the way DaCe spells them.

        Two things the generic path gets wrong here. First, DaCe's own converter builds ITS `AND`
        and `OR` Function classes, not sympy's -- because a DaCe guard is routinely a bare array
        read (``ldcum[i-1, j-1] and x > 0``) and sympy's ``And`` rejects a non-Boolean outright with
        ``expecting bool or Boolean``. That killed deserialization of real CloudSC SDFGs under this
        backend while the sympy backend loaded them.

        Second, idxalg makes C truthiness explicit as ``Cast(x, Bool)``; rendered literally that is
        ``bool(A[i])``, a spelling DaCe never writes. In a boolean context the raw value IS the
        predicate -- codegen applies the truthiness -- so the cast is peeled here. Only here: in
        ARITHMETIC a ``bool`` cast is a real 0/1 narrowing and must survive, or ``bool(x) + 1``
        would silently become ``x + 1``.
        """
        args = [_from_idx(_peel_truthiness(a)) for a in e.args]
        if name == "Not":
            return backend.Not(args[0])
        op = _dace_op_class("And" if name == "And" else "Or")
        return functools.reduce(op, args)

    def _peel_promotion(parent, child):
        """A cast the promotion rules INSERTED, dropped where the enclosing operation converts anyway.

        Only int -> float, and only under `+ - * /`. The in-kind widening case (`int32` -> `int64`)
        is already dropped in the `Cast` arm, value-preservingly and without needing a parent. A KIND
        crossing is not value-preserving -- above 2^53 an int64 rounds, which is exactly why the
        engine records it -- but C performs the identical conversion at this position, so spelling it
        changes nothing about the result and diverges from every untyped producer of the same
        expression.

        Under a CALL it must stay. `Min`/`Max`/`pow` lower to `dace::math::min(x, j)` and friends,
        whose argument types C++ deduces rather than converts, so an operand short of its cast is not
        merely noisy but uncompilable.
        """
        if _idx._CTX.kind(child._e) != "Cast":
            return child
        outer = _idx._CTX.dtype(child._e)
        if outer != _idx._CTX.dtype(parent._e):
            return child
        ow, iw = _int_width(outer), _int_width(_idx._CTX.dtype(child.args[0]._e))
        return child.args[0] if ow is not None and iw is not None and (iw[0], ow[0]) == ("int", "float") else child

    def _peel_truthiness(a):
        """The truthiness of `x` reduced back to `x`; anything else unchanged.

        C truthiness enters as `cast(x, Bool)`, but the engine FOLDS that to `x != 0` -- the same
        value, and the form a bit-vector solver can reason about. Either shape peels here: in a
        boolean position `x != 0` and `x` are interchangeable (C applies the test itself), and DaCe
        writes the bare value. Only in a boolean position: in ARITHMETIC a bool cast is a real 0/1
        narrowing, and peeling there would turn `bool(x) + 1` into `x + 1`.
        """
        kind = _idx._CTX.kind(a._e)
        if kind == "Cast" and _idx._CTX.dtype(a._e) == "bool":
            return a.args[0]
        if kind == "Ne":
            lhs, rhs = a.args
            if _idx._CTX.kind(rhs._e) == "Integer" and int(rhs) == 0:
                return lhs
        return a

    def _dace_op_class(name: str):
        """The DaCe operator class named `name`; raises rather than letting a generic Function through."""
        cls = _sympy_function_class(name)
        if cls is None:
            raise ValueError(f"DaCe operator class {name!r} is not bound in the parser table")
        return cls

    def _dace_op(name: str, args: list):
        """Apply the DaCe operator Function class named ``name``; raise if the table lost it, since
        falling back to a generic Function would silently print the call spelling instead."""
        cls = _sympy_function_class(name)
        if cls is None:
            raise ValueError(f"DaCe operator class {name!r} is not bound in the parser table")
        return cls(*args)

    @functools.lru_cache(maxsize=None, typed=True)
    def _sympy_function_class(name: str):
        """The real sympy ``Function`` subclass DaCe's parser binds to ``name``, else ``None``.

        Falls back to sympy's own top-level namespace, which is exactly what `sympify` resolves
        against under the incumbent backend -- so `sin`/`sqrt`/`gamma` come back as themselves and
        not as applied undefs. A second hand-written name list here would drift silently.
        """
        from dace.symbolic import _PYSTR2SYM_locals
        bound = _PYSTR2SYM_locals.get(name)
        if not isinstance(bound, type):
            # The table also holds plain Symbols (sympy's ``_clash`` names) and `None` sentinels.
            bound = vars(backend).get(name)
        return bound if isinstance(bound, type) and issubclass(bound, backend.Function) else None

    _idx.set_sympy_converter(_from_idx)
    # `Expr.dtype` must read as DaCe's own symbol dtype does -- a `typeclass`, not its name. DaCe
    # feeds it straight into type-keyed tables (`SDFG.add_symbol` -> `dtype_to_typeclass`), where the
    # bare string raised `KeyError: 'int64'`.
    _idx.set_dtype_mapper(_dtype_tc)

    # Widths that need no `TypedConstant` wrapper: idxalg's own fallbacks plus DaCe's default symbol
    # width, since a constant at any of these carries no information a bare sympy number loses.
    _DEFAULT_CONST_DTYPES = frozenset(("int32", "int64", "float64"))

    def _int_width(dstr: str) -> tuple[str, int] | None:
        """`(kind, bit width)` for a recognized idxalg dtype string, else `None`. A Cast is only a
        value-preserving widen WITHIN one kind (int-family / float / complex / bool); returning
        `None` for an unrecognized string, rather than a numeric default, means the call site can
        never mistake "unknown" for "trivially wide enough" (the CRITICAL bug this replaces)."""
        if dstr.startswith("uint"):
            return "int", int(dstr[4:])
        if dstr.startswith("int"):
            return "int", int(dstr[3:])
        float_bits = {
            "float8_e4m3fn": 8,
            "float8_e5m2": 8,
            "float16": 16,
            "bfloat16": 16,
            "float32": 32,
            "float64": 64,
        }
        if dstr in float_bits:
            return "float", float_bits[dstr]
        if dstr in ("complex64", "complex128"):
            return "complex", int(dstr[len("complex"):])
        if dstr == "bool":
            return "bool", 1
        return None

    def to_sympy(obj):
        """An idxalg `Expr` rendered back as sympy, else `None`.

        Lets the still-sympy island (printers, `solve`, `Poly`, `match`) work on a natively-parsed
        expression. Reusing `DaceSympyPrinter` this way is deliberate: a second printer would be two
        implementations of DaCe's spellings, free to drift, and spelling drift is a miscompile.
        """
        return _from_idx(obj) if isinstance(obj, _idx.Expr) else None

    def native_parse(text: str):
        """Parse an expression string with idxalg's own Rust parser, bypassing sympy entirely.

        This is what makes the idxalg backend actually *engage*: without it `pystr_to_symbolic`
        builds sympy objects no matter which backend is selected, so idxalg never runs and an A/B
        measures sympy against sympy. Handles DaCe's typed wire spellings (`$name`, `2i16`,
        `(1.0 + 2.0j)c128`) via the normalizer, so no pre-pass is needed at the call site.
        """
        return _idx.parse_str(text)

    # The concrete expression type this backend builds, so a caller can recognize a value from a
    # NON-sympy backend without importing idxalg (the seam is the only place allowed to) and without
    # a `getattr` probe. `None` under the sympy backend, where no such value can exist.
    NATIVE_EXPR = _idx.Expr

    # Kinds that can never carry a DaCe head class. A DENY list, not an allow list: a missing
    # allow-list entry would silently answer "not an instance" -- the wrong-branch failure this
    # protocol exists to remove -- whereas a missing deny entry only costs a conversion.
    # Keyed on `kind` strings, so head NAMES ("Number", "Boolean", "Cmp") never matched and are gone;
    # their kinds (Complex/Bool/Eq/...) are deliberately left out, since converting them yields the
    # same head name sympy would report and keeps the two backends' answers identical.
    _NO_DACE_HEAD = frozenset({"Symbol", "Integer", "Float", "Undef", "Add", "Mul", "Min", "Max"})

    _DACE_HEAD_MEMO: dict = {}

    def dace_head_name(obj):
        """Name of :func:`dace_head_class`, else ``None``."""
        cls = dace_head_class(obj)
        return None if cls is None else cls.__name__

    def dace_head_class(obj):
        """The DaCe head class this backend's value would convert to, else ``None``.

        Some heads this backend names as DaCe does (an opaque call; the operator spellings, which
        carry their ``__`` prefix). Others it holds STRUCTURALLY -- ``int_floor(a, b)`` is a division
        node named ``floor`` -- and only the converter knows their DaCe spelling. So the answer comes
        FROM the converter rather than from a second name table that would be free to drift.

        Memoized on (head, arity, spelling), which is everything the converter's choice of head
        depends on, so the conversion runs once per shape instead of once per query. Without it an
        `isinstance` MISS converted a whole expression tree just to reject it.
        """
        if not isinstance(obj, _idx.Expr):
            return None
        # Keyed on the ENGINE's own identifiers, not on `.func`/`.args`: those build a head object
        # and wrap every child in a fresh handle, which cost more than the conversion they were
        # meant to avoid (2.5us per query, on hits as well as misses). `kind` already encodes both
        # the structure and an opaque head's name, `dtype` distinguishes the cast targets, and
        # `spelling` picks between an operator head and its function twin.
        e = obj._e
        kind = _idx._CTX.kind(e)
        if kind in _NO_DACE_HEAD:
            # One engine read settles the common query. These kinds are plain algebra and leaves:
            # sympy owns their heads and DaCe defines no class for any of them, so no conversion can
            # produce one and the remaining reads would be pure cost on the path taken most.
            return None
        # `dtype` only ever names a class for a cast, and that is keyed on the kind being exactly
        # `Cast`, so skipping it elsewhere cannot mislead. The SPELLING is always read: restricting it
        # to the kinds known to have a second form put `b if c else d` and `ITE(c, t, f)` under one key,
        # and the conditional then inherited the blend's cached answer. A memo key that omits
        # something the answer depends on is the same silent-wrong-branch bug this protocol removes.
        dtype = _idx._CTX.dtype(e) if kind == "Cast" else None
        key = (kind, dtype, _idx._CTX.spelling(e))
        if key in _DACE_HEAD_MEMO:
            return _DACE_HEAD_MEMO[key]
        converted = _from_idx(obj)
        # `type()`, not `.func`: only a Function subclass stringifies to its own name; a singleton head
        # (`true`, `Zero`) gives `<class '...'>`, so `Boolean` leaked a class repr as a head name.
        head = None if converted is None else type(converted)
        _DACE_HEAD_MEMO[key] = head
        return head

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
    # Defined in both branches so a caller can test them without `getattr`.
    native_parse = None
    to_sympy = None
    NATIVE_EXPR = None
    dace_head_name = None
    dace_head_class = None

    def is_native_symbol(_obj) -> bool:
        """Callable rather than `None` so call sites need no capability guard."""
        return False

    def __getattr__(name: str):
        # PEP 562: forwards every symbolic name to sympy with no overhead on normal imports.
        return getattr(backend, name)

    def __dir__() -> list:
        return dir(backend)
