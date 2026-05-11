.. _symbolic:

Working with Symbolic Expressions
=================================

DaCe relies on `SymPy <https://docs.sympy.org/>`_ to reason about array
shapes, memlet ranges, schedules, and any other quantity that may depend
on a runtime parameter. Almost every numeric attribute on the SDFG IR -
shape entries, ranges, parameter bounds, transient sizes - is either a
Python integer or a SymPy expression. This page is a tour of the helpers
in :mod:`dace.symbolic` that the rest of DaCe (and most extensions) use
when they need to manipulate those expressions.

Symbols
-------

DaCe symbols are SymPy symbols with an optional fixed type
and attached integer assumption (based on the type):

.. code-block:: python

    import dace
    N = dace.symbol('N', positive=True)
    M = dace.symbol('M', dtype=dace.int64)

The class :class:`dace.symbolic.symbol` extends :class:`sympy.Symbol` so
that any DaCe symbol can be used as-is in SymPy expressions, while still
carrying the metadata DaCe needs (a name, a data type, and a set of SymPy
assumptions). The convenience type alias
:data:`~dace.symbolic.SymbolicType` stands for ``Union[sympy.Basic,
SymExpr]`` and is the type hint to use whenever a function accepts a
symbol or a symbolic expression.

The richer container :class:`~dace.symbolic.SymExpr` carries both an
*exact* expression and an *approximate* expression. Most code only needs
the exact one; over-approximations come into play when the IR has to
guarantee an upper or lower bound (see below).

Indeterminate comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing symbolic expressions returns one of three answers:

* ``(N > 0) == True`` - the inequality is implied by the assumptions on
  the symbols.
* ``(N > 0) == False`` - the inequality is provably false.
* ``(N > 0)`` is an unevaluated SymPy expression - SymPy could neither
  prove nor refute the inequality. This is a frequent source of subtle
  bugs in transformations: never use such an expression in a Python
  ``if``. Use :func:`~dace.symbolic.simplify` (or supply more
  assumptions) before branching on it.

Integer arithmetic caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~

SymPy's default rational arithmetic does not match the C/C++ semantics
that the code generators eventually emit. DaCe ships two SymPy functions,
:class:`~dace.symbolic.int_floor` and ``int_ceil``, that correspond to
``a // b`` and ``ceil(a / b)`` for positive integers respectively. Use
them whenever you need to keep the symbolic result in ``int`` arithmetic;
the simplifier and the code generator both know about them.

Analysis
--------

* :func:`~dace.symbolic.issymbolic` checks whether a value is a SymPy
  expression that depends on at least one symbol (treating literal
  ``Integer``/``Float`` as non-symbolic).
* :func:`~dace.symbolic.free_symbols_and_functions` returns the names of
  every free symbol *and* every undefined function appearing in the
  expression. This is the right helper to use when computing the symbol
  set that must be present in an SDFG before an expression can be
  evaluated.
* :func:`~dace.symbolic.swalk` is a small visitor that yields every
  sub-expression in pre-order traversal, optionally descending into
  function arguments. Use it to look for specific patterns or to gather
  all occurrences of a kind of node.
* For sub-expression matching, SymPy's ``expr.match(pattern)`` and
  ``expr.find(pattern)`` are usually sufficient; the SymPy documentation
  has examples.

Conversion
----------

* :func:`~dace.symbolic.pystr_to_symbolic` parses a Python-style string
  (``"N + 2*M"``) into a SymPy expression while honoring DaCe's
  conventions (e.g., ``int_floor`` for ``//``).
* :func:`~dace.symbolic.symstr` renders a SymPy expression back to a
  Python-style string. This is the inverse used when serializing the
  IR.
* :func:`dace.codegen.common.sym2cpp` emits a C/C++-friendly string from
  the same expression. Code generators should use ``sym2cpp`` instead of
  ``str(expr)`` so that integer division, ``min``/``max``, and the DaCe
  helper functions produce valid C++.

Mutation and simplification
---------------------------

* :func:`~dace.symbolic.simplify` is the recommended simplifier
  throughout the DaCe codebase. Unlike ``sympy.simplify`` it preserves
  integer semantics and runs efficiently on the kinds of expressions
  that show up in memlet ranges.
* :func:`~dace.symbolic.safe_replace` performs a substitution that is
  safe under aliasing - replacing ``a -> b`` and ``b -> c`` simultaneously
  produces ``a -> b, b -> c`` rather than ``a -> c``. Use it whenever
  you build a substitution dictionary from a mapping that could overlap.
* :func:`~dace.symbolic.overapproximate` returns a syntactic
  over-approximation of an expression (for instance, replacing a
  data-dependent ``Min`` with one of its arguments). Together with the
  ``approx`` field of :class:`~dace.symbolic.SymExpr` this is what allows
  memlet propagation to compute conservative ranges when the exact range
  is data-dependent.

.. _symbolic-when:

Symbolic types vs. scalars
--------------------------

When extending DaCe, a recurring decision is whether a quantity should be
modeled as a *symbol* or as a *scalar transient*. The rule of thumb is:

* Use a **symbol** for quantities that are constant over the lifetime of
  the SDFG (typically loop bounds and array shapes provided by the
  caller), or a state (e.g., indices used in memlets). Symbols participate
  in the symbolic propagation system.
* Use a **scalar transient** for quantities that may change within
  states (counters, accumulators, intermediate results). They live in
  :attr:`~dace.sdfg.sdfg.SDFG.arrays` and are written by tasklets like
  any other data.

When in doubt, prefer a symbol if the value is set once or consumed in
ranges or schedules; prefer a scalar otherwise. See the FAQ entry on this
question for a longer discussion.
