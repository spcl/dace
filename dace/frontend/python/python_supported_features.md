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
- def, return, lambda

The following keywords are NOT accepted:
- global, nonlocal
- class
- try, except, finally
- raise, yield, pass
- import, from, as, with
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
