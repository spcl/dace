# Python Language Reference Supported Features

This document describes in detail which features of Python are supported by the DaCe Python-Frontend.
The comparison is made against the [Python Language Reference](https://docs.python.org/3/reference/).

## 2 Lexical Analysis
### 2.1 Line Structure
The DaCe Python-Frontend uses the Python AST module to parse code.
Therefore, full support of the line structure section is expected.
However, we explicitly tests for the following subsections (`tests/python_fronted/line_structure_test.py`):
- 2.1.3 Comments
- 2.1.5 Explicit Line Joining
- 2.1.6 Implicit Line Joining
- 2.1.7 Blank Lines

### 2.3 Identifiers and Keywords
The DaCe Python-Frontend uses exactly the same rules for identifiers as normal Python.
However, for a generated SDFG to compile to a target architecture, the identifiers used
must be valid in the output language (e.g., C/C++). We currently provide no automatic
transformation of identifiers used as variable names. Therefore, it is the responsibility
of the user to ensure that the variable names are compatible.
#### 2.3.1 Keywords
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

#### 2.3.2 Reserved classes of identifiers
Reserved class of Python identifiers are not supported. Furthermore, identifiers
starting with double underscore (`__`) are reserved by the SDFG language.

### 2.4 Literals
The DaCe Python-Frontend supports in general the same literals as Python.
However, there is currently limited support for strings and char/byte arrays.
For example, it is not possible to instantiate an (u)int8 array with a string
or byte literal.

### 2.5 Operators
The DaCe Python-Frontend supports all Python operators.
The operators are only supported in the context of arithmetic/logical operations among
scalar values and DaCe (Numpy-compatible) arrays. For example, it is not possible
to concatenate 2 strings with the `+` operator.
The `:=` operator (Named Expression) is parsed as an assignment statement.

### 2.6 Delimiters
The DaCe Python-Frontend supports all Python delimiters. However, not all uses of
those delimiters are supported. For example, we do not support lists, sets, and
dictionaries. Therefore, the delimiters `[, ], {, }` cannot be used to define
those datatypes.

## 6 Expressions
### 6.1 Arithmetic Conversions

The arithmetic conversions for binary operators (except the power operator)
are implemented with explicit casting:
- If any of the operands is of complex type, but the other operand is a float,
  int or bool, then it is casted to the same complex type.
- If any of the operands is of float type, but the other operand is int or bool,
  the it is casted to the same float type.

### 6.2 Atoms
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

### 6.3 Primaries
Similarly to atoms, Python primaries are parsed. However, their intended usage
may not be supported:
- Attribute references: Supported for a subset of the DaCe and Numpy modules
- Subscriptions: Supported on DaCe/Numpy arrays
- Slicings: Supported on DaCe/Numpy arrays
- Calls: Supported for other DaCe programs, and a subset of methods from the
DaCe and Numpy modules

### 6.4 Await expression
Unsupported  

### 6.5 The power operator
Supported, but the case where both operands are integers and the exponent is
negative doesn't return the expected result.

### 6.6 Unary arithmetic and bitwise operations
Supported

### 6.7 Binary arithmetic operations
Supported

### 6.8 Shifting operations
Supported

### 6.9 Binary bitwise operations
Supported

### 6.10 Comparisons
Supported

### 6.11 Boolean operations
Supported

### 6.12 Assignment expressions
Experimental support

### 6.13 Conditional expressions
Supported

### 6.14 Lambdas
Supported only for defining WCR/reduction operators

### 6.15 Expression lists
Supported only for (supported) method arguments that expect a list/tuple

### 6.16 Evaluation order
Supported

### 6.17 Operator precedence
Supported
