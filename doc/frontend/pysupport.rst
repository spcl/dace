Core Python Language Support
============================

.. note::

   This section is a work in progress.  It may not reflect the capabilities provided in the latest version of DaCe.


This document describes in detail which features of Python are supported by the Data-Centric Python-Frontend.
The comparison is made against the `Python Language Reference <https://docs.python.org/3/reference>`_.

2 Lexical Analysis
------------------

2.1 Line Structure
^^^^^^^^^^^^^^^^^^
The DaCe Python-Frontend uses the Python AST module to parse code.
Therefore, full support of the line structure section is expected.
However, we explicitly test for the following subsections (see :mod:`~tests.python_frontend.line_structure_test`):

- 2.1.3 Comments
- 2.1.5 Explicit Line Joining
- 2.1.6 Implicit Line Joining
- 2.1.7 Blank Lines

2.3 Identifiers and Keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DaCe Python-Frontend uses exactly the same rules for identifiers as normal Python.
However, for a generated SDFG to compile to a target architecture, the identifiers used
must be valid in the output language (e.g., C/C++). We currently provide no automatic
transformation of identifiers used as variable names. Therefore, it is the responsibility
of the user to ensure that the variable names are compatible.

2.3.1 Keywords:

The following keywords are recognized (for at least a subset of their Python functionality):

- True, False, None
- or, and, not
- if, elif, else
- for, while, break, continue, in
- def, return, lambda, with, pass
- assert, raise (handled by the preprocessor; see 7.3 and 7.8 below)
- class (only when used together with ``@dace.method``; see 8.7)

The following keywords are NOT accepted in the body of a ``@dace.program``:

- global, nonlocal
- try, except, finally
- yield
- import, from, as
- async, await, del

The authoritative list lives in ``DISALLOWED_STMTS`` and ``_DISALLOWED_STMTS`` in
:mod:`dace.frontend.python.newast`.

2.3.2 Reserved classes of identifiers:

Reserved class of Python identifiers are not supported. Furthermore, identifiers
starting with double underscore (``__``) are reserved by the SDFG language.

2.4 Literals
^^^^^^^^^^^^

The DaCe Python-Frontend supports in general the same literals as Python.
However, there is currently limited support for strings and char/byte arrays.
For example, it is not possible to instantiate an (u)int8 array with a string
or byte literal.

2.5 Operators
^^^^^^^^^^^^^

The DaCe Python-Frontend supports all Python operators.
The operators are only supported in the context of arithmetic/logical operations among
scalar values and DaCe (NumPy-compatible) arrays. For example, it is not possible
to concatenate 2 strings with the ``+`` operator.
The ``:=`` operator (Named Expression) is parsed as an assignment statement.

2.6 Delimiters
^^^^^^^^^^^^^^

The DaCe Python-Frontend supports all Python delimiters. However, not all uses of
those delimiters are supported. For example, we do not support lists, sets, and
dictionaries. Therefore, the delimiters ``[, ], {, }`` cannot be used to define
those datatypes.

6 Expressions
-------------

6.1 Arithmetic Conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The arithmetic conversions for most binary operators are implemented with casting:

- If any of the operands is of complex type, but the other operand is a float, int or bool, then it is cast to the same complex type.
- If any of the operands is of float type, but the other operand is int or bool, the it is cast to the same float type.

Some binary operations are handled differently, as described in the sections below.

6.2 Atoms
^^^^^^^^^

All Python atoms are parsed. However, their intended usage may not be supported:

- Identifiers: Supported
- Literals: Only use of numerical literals supported
- Parethesized forms: Some uses of tuples supported
- Displays for lists, sets and dictionaries: Unsupported
- List displays: Supported only for (supported) method arguments that expect a list/tuple
- Set displays: Unsupported
- Dictionary displays: Unsupported
- Generator expressions: Only the built-in range supported
- Yield expressions: Unsupported

6.3 Primaries
^^^^^^^^^^^^^

Similarly to atoms, Python primaries are parsed. However, their intended usage may not be supported:

- Attribute references: Supported for a subset of the DaCe and Numpy modules
- Subscripts: Supported on DaCe/Numpy arrays
- Slicing: Supported on DaCe/Numpy arrays
- Calls: Supported for other DaCe programs, and a subset of methods from the DaCe and NumPy modules

One caveat of subscripts with NumPy arrays is that NumPy allows negative indices to wrap around the array. In DaCe
this is not supported.

6.4 Await expression
^^^^^^^^^^^^^^^^^^^^

Unsupported

6.5 The power (**) operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported. If the base is an integer and the exponent a signed integer, both
operands are cast to float64 and the result is also of type float64.

6.6 Unary arithmetic and bitwise operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported

6.7 Binary arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported. Notable differences compared to the expected Python result:

- Modulo operator always returns a natural number (like in C/C++)

6.8 Shifting operations
^^^^^^^^^^^^^^^^^^^^^^^

Only integral types supported.

6.9 Binary bitwise operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only integral types supported.

6.10 Comparisons
^^^^^^^^^^^^^^^^

Supported

6.11 Boolean operations
^^^^^^^^^^^^^^^^^^^^^^^

Supported

6.12 Assignment expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experimental support

6.13 Conditional expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported

6.14 Lambdas
^^^^^^^^^^^^

Lambdas used as the WCR (write-conflict-resolution) argument of
``dace.reduce`` and similar reductions are supported as-is. Lambdas used
as regular callables are inlined at their call site by the preprocessor.
Returning a lambda from a ``@dace.program`` is not supported.

6.15 Expression lists
^^^^^^^^^^^^^^^^^^^^^

Supported only for (supported) method arguments that expect a list/tuple

6.16 Evaluation order
^^^^^^^^^^^^^^^^^^^^^

Supported

6.17 Operator precedence
^^^^^^^^^^^^^^^^^^^^^^^^

Evaluated exactly as in Python.

7 Simple Statements
-------------------

7.1 Expression statements
^^^^^^^^^^^^^^^^^^^^^^^^^

Partially supported, as described in the previous section. Python interactive mode is not supported.

7.2 Assignment statements
^^^^^^^^^^^^^^^^^^^^^^^^^

Assignment statements with single or multiple targets are supported, both with
and without parentheses. Statements with starred targets are not supported.
Targets may only be identifiers, and subscriptions/slices of Numpy arrays.

7.2.1 Augmented assignment statements:

Supported with the same constraints for targets as in assignment statements.

7.2.2 Annotated assignment statements:

Supported. Annotations on local variables are accepted by ``visit_AnnAssign``
(see :mod:`~dace.frontend.python.newast`); the annotation is honored when
creating the underlying data container.

Note that annotated assignments to *compile-time constants* in the closure are
rejected by ``DisallowedAssignmentChecker`` during preprocessing.

7.3 The assert statement
^^^^^^^^^^^^^^^^^^^^^^^^

The Python preprocessor evaluates assertions statically against the program's
closure (``globals`` plus any captured constants). If the condition is
statically true, the ``assert`` is dropped; if it is statically false, the
DaCe program fails to compile with an ``AssertionError``. ``assert``
statements that depend on runtime values cannot be checked and are skipped
with a warning.

7.4 The pass statement
^^^^^^^^^^^^^^^^^^^^^^

Supported

7.5 The del statement
^^^^^^^^^^^^^^^^^^^^^

Unsupported

7.6 The return statement
^^^^^^^^^^^^^^^^^^^^^^^^

Supported

7.7 The yield statement
^^^^^^^^^^^^^^^^^^^^^^^

Unsupported

7.8 The raise statement
^^^^^^^^^^^^^^^^^^^^^^^

Not supported at runtime. ``raise`` statements found in a ``@dace.program``
are reported with a warning during preprocessing and stripped from the
generated code; control will continue past the would-be raise site.

7.9 The break statement
^^^^^^^^^^^^^^^^^^^^^^^

Supported for for/while loops, as long as the break statement is in the same
SDFG-level as the for/while statement.

7.10 The continue statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported for for/while loops, as long as the continue statement is in the same
SDFG-level as the for/while statement.

7.11 The import statement
^^^^^^^^^^^^^^^^^^^^^^^^^

Unsupported inside the body of a ``@dace.program``. ``from __future__``
imports placed at module scope are tolerated (the preprocessor strips them
before parsing).

7.12 The global statement
^^^^^^^^^^^^^^^^^^^^^^^^^

Unsupported

7.13 The nonlocal statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unsupported

8 Compound Statements
---------------------

8.1 The if statement
^^^^^^^^^^^^^^^^^^^^

Supported. Note that if the type of some variable depends on the branch taken,
then the variable will always have the type of the first branch. E.g., in the
following code, variable b will be of type dace.int always, even if
``a[0] == np.float32(np.pi)``, unless it is explicitly declared as such:

.. code-block:: python

    @dace.program
    def single_target(a: dace.float32[1]):
        if (a[0] < 0):
            b = 0
        elif (a[0] < 1):
            b = 1
        else:
            b = a
        return b


8.2 The while statement
^^^^^^^^^^^^^^^^^^^^^^^

Supported

8.3 The for statement
^^^^^^^^^^^^^^^^^^^^^

Supported, but only with `range`, `parrange`, and `dace.map`.

8.4 The try statement
^^^^^^^^^^^^^^^^^^^^^

Unsupported

8.5 The with statement
^^^^^^^^^^^^^^^^^^^^^^

``with dace.tasklet`` is the canonical form for explicit-dataflow tasklets
(see :ref:`explicit-dataflow-mode`). General ``with`` statements are supported through
the preprocessing-time :class:`~dace.frontend.python.preprocessing.ContextManagerInliner`,
which inlines the ``__enter__`` / ``__exit__`` calls of the context manager
at the appropriate program points. ``with`` blocks that depend on raised
exceptions for control flow are not supported.

8.6 Function definitions
^^^^^^^^^^^^^^^^^^^^^^^^

The top-level callable must be decorated with ``@dace.program`` (or
``@dace.method`` when defined inside a class - see 8.7). Function arguments
must be type-annotated.

Calls from one ``@dace.program`` to another are supported and lower to
nested SDFGs. *Defining* a ``@dace.program`` inside the body of another
``@dace.program`` is not supported; nested ``def`` blocks without the
decorator are inlined by the Python preprocessor when called.

Lambdas defined as part of a reduction (``dace.reduce``) or other library
nodes that take a WCR are accepted as-is; other lambdas are inlined at their
call site by the preprocessor (see 6.14).

8.7 Class definitions
^^^^^^^^^^^^^^^^^^^^^

``class`` blocks themselves cannot appear inside a ``@dace.program``
(``ClassDef`` is in :data:`~dace.frontend.python.newast.DISALLOWED_STMTS`).
Methods of regular Python classes can however be turned into DaCe programs
by decorating them with ``@dace.method``: the preprocessor binds ``self`` as
part of the program's closure so that attribute accesses on ``self`` resolve
to data containers, scalars, or constants of the enclosing instance.

8.8 Coroutines
^^^^^^^^^^^^^^

Unsupported
