# Contribution and Coding Guidelines

DaCe is an open-source project that accepts contributions from any individual or
organization. Below are a set of guidelines that we try our best to follow during
development and the code review process.

## Code Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), with a few notable exceptions:

* **"Power Features"**: We like Python power features. We use them both externally (to make the Python/Numpy frontend easy to use) and internally. Use of any Python power feature that increases productivity is allowed. 
    * **Note on compatibility**: DaCe currently supports Python versions 3.6 and above, please make sure that the feature is supported in those versions, and if not ([dataclasses](https://docs.python.org/3/library/dataclasses.html) for example), please make sure to add a backwards-compatibility dependency. [Example](https://github.com/spcl/dace/blob/205d7c911a74e507d2fcbcc4b6cb5819b026648a/setup.py#L71)
* **Type Hints**: New functions must include proper Python [typing information](https://docs.python.org/3/library/typing.html), in order to support type checking and smoother development.
* **Importing classes and functions directly**: This is disallowed, with the exception of directly importing the following main graph components (which are heavily reused throughout the framework): `SDFG, SDFGState, Memlet, InterstateEdge`.
* **Inline imports**: Imports usually go at the top of a Python file, after the copyright statement and the file docstring. If you must place an `import` statement anywhere else, indicate the reason with an adjacent comment (e.g., `# Avoid import loop`).

For automatic styling, we use the [yapf](https://github.com/google/yapf) file formatter.
**Please run `yapf` before making your pull request ready for review.**

## Dependencies

Please refrain from adding new dependencies unless necessary. If they are necessary,
indicate the reason in the pull request description. Prefer dependencies
available in standard repositories (such as [PyPI](https://pypi.org/)) over git
repositories. Add dependencies both to the [setup.py](setup.py) and [requirements.txt](requirements.txt) files.

## File Heading

In any new file created, add the following copyright statement in the first line:
```
# Copyright 2019-<CURRENT YEAR> ETH Zurich and the DaCe authors. All rights reserved.
```
where `<CURRENT YEAR>` should be replaced with the current year.
In other languages (e.g., C/C++), modify the `#` with the appropriate single-line
comment syntax.

An exception to the rule is empty files (such as _empty_ `__init__.py` files).

## Contributor List

You are welcome to add your name to the [list of contributors](AUTHORS) as part
of the pull request.

## Documentation

Use Python docstrings for function/class/file documentation. Since we use [Sphinx](https://www.sphinx-doc.org/) for our [documentation](https://spcldace.readthedocs.io/), the docstrings can use reStructuredText (rst) for formatting. For a function or a class, document the purpose, all arguments, return value, and notes. [Example](https://github.com/spcl/dace/blob/205d7c911a74e507d2fcbcc4b6cb5819b026648a/dace/dtypes.py#L1146-L1156)

File documentation is mostly one or a few sentences documenting the purpose and elements
in a file. If your file contains more than one class/function, which share a general theme that should be documented, include a docstring for the Python file containing a general explanation (and ideally an example of the topic). [Example](dace/codegen/control_flow.py)

