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

### 2.2 Identifiers and Keywords
The DaCe Python-Frontend uses exactly the same rules for identifiers as normal Python.
The following keywords are recognized (for at least a subset of their Python functionality):
- True, False, None
- or, and, not
- if, elif, else
- for, while, break, continue, in
- def, return

The following keywords are NOT accepted:
- global, nonlocal
- class, lambda
- try, except, finally
- raise, yield, pass
- import, from, as, with
- assert, async, await, del
