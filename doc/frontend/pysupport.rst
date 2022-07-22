Core Python Language Support
============================

This document describes in detail which features of Python are supported by the Data-Centric Python-Frontend. It does not
include features supported by :doc:`preprocessing` or :doc:`jitoop`.
The comparison is made against the `Python Language Reference <https://docs.python.org/3/reference>`_.

NOTE: This document has to be updated.

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
- def, return, lambda, with

The following keywords are NOT accepted:

- global, nonlocal
- class
- try, except, finally
- raise, yield, pass
- import, from, as
- assert, async, await, del

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
- Subscriptions: Supported on DaCe/Numpy arrays
- Slicings: Supported on DaCe/Numpy arrays
- Calls: Supported for other DaCe programs, and a subset of methods from the DaCe and NumPy modules

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

Supported only for defining WCR/reduction operators

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

Unsupported

7.3 The assert statement
^^^^^^^^^^^^^^^^^^^^^^^^

Unsupported

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
Unsupported

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

Unsupported, including 7.11.1 Future statements

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

Only supported ``with dace.tasklet``

8.6 Function definitions
^^^^^^^^^^^^^^^^^^^^^^^^

Supported only with the ``dace.program`` decorator. Function arguments must be
type-annotated. Nested ``dace.program`` definitions are not supported.

8.7 Class definitions
^^^^^^^^^^^^^^^^^^^^^

See :doc:`preprocessing` and :doc:`jitoop`.

8.8 Coroutines
^^^^^^^^^^^^^^

Unsupported
