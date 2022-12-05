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
* **All-inclusive imports**: `import *` is disallowed.
* **Inline imports**: Imports usually go at the top of a Python file, after the copyright statement and the file docstring. If you must place an `import` statement anywhere else, indicate the reason with an adjacent comment (e.g., `# Avoid import loop`).
* **docstrings**: We use [Sphinx](https://www.sphinx-doc.org/) for documentation. Use type hints as much as possible (this will be automatically integrated into the documentation) and the following format:

```python
def example_function(param_a: str, *args: Optional[SDFG]) -> bool:
    """
    Explain what the function does. Note the double line break below, after
    description and before parameter declaration! Without it Sphinx does not work.

    :param param_a: What ``param_a`` does. Double backticks indicate code format in Sphinx.
    :param args: Variable-length arguments are documented just like standard parameters.
    :return: True if example, False otherwise.
    :note: Some notes can be used here. See Sphinx for the full list of available annotations.
    :note: Numbered and itemized lists must also have a blank line and must be indented.

    If you want to include a code sample, use:

    .. code-block:: python

        # Note the empty line above
        example_use = example_function('hello', None, None, SDFG('world'))
    """
    ...
```


For automatic styling, we use the [yapf](https://github.com/google/yapf) file formatter.
**Please run `yapf` before making your pull request ready for review.**

## Tests

We use [pytest](https://www.pytest.org/) for our testing infrastructure. All tests under the `tests/` folder 
(and any subfolders within) are automatically read and run. The files must be under the right subfolder
based on the component being tested (e.g., `tests/sdfg/` for IR-related tests), and must have the right
suffix: either `*_test.py` or `*_cudatest.py`. See [pytest.ini](https://github.com/spcl/dace/blob/master/pytest.ini))
for more information, and for the markers we use to specify software/hardware requirements.

The structure of the test file must follow `pytest` standards (i.e., free functions called `test_*`), and
all tests must also be called at the bottom of the file, under `if __name__ == '__main__':`.

Tests that should be skipped **must** define the `@pytest.mark.skip` decorator with a reason parameter,
and must also be commented out at the main section at the bottom of the file.

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

