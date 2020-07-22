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
