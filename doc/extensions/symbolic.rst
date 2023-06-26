.. _symbolic:

How to Work with Symbolic Expressions
=====================================

SymPy is very strong, functions like ``match`` for sub-expression matching, ``solve`` etc.
For more information about SymPy, refer to their `documentation <https://docs.sympy.org/>`_.

Talk about issymbolic, symbolic.py.
DaCe symbols are extended sympy symbols. :class:`~dace.symbolic.symbol` with a name and assumption

``dace.symbolic.SymbolicType`` type hint covers a symbol or a symbolic expression

Talk about indeterminate expressions, there are three kinds of answers: ``(N > 0) == True``

Integer math caveats, int_ceil and int_floor.

Analysis
--------

free_symbols_and_functions

swalk

Conversion
----------

:func:`~dace.symbolic.pystr_to_symbolic`

:func:`~dace.symbolic.symstr`, :func:`~dace.symbolic.sym2cpp` in code generators

Mutation
--------

:func:`~dace.symbolic.simplify`, :func:`~dace.symbolic.safe_replace`

over-approximation, :func:`~dace.symbolic.overapproximate`, customizable via :class:`~dace.symbolic.SymExpr`.


