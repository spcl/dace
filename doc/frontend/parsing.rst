Parsing Python Programs to SDFGs
================================

Prerequisites
-------------

- :doc:`daceprograms`

Scope
-----

This document describes DaCe's core Python language parser, implemented by the :class:`~dace.frontend.python.newast.ProgramVisitor` class.
The :class:`~dace.frontend.python.newast.ProgramVisitor` supports a restricted subset of Python's features that can be expressed directly as SDFG elements.
A larger subset of the Python language is supported either through code preprocessing (see :doc:`preprocessing`) and/or in JIT mode (see :doc:`jitoop`).

Supported Python Versions
-------------------------

The :class:`~dace.frontend.python.newast.ProgramVisitor` supports exactly the same Python versions as the Data-Centric framework overall: 3.7-3.10.
To add support for newer Python versions, the developer should amend the :class:`~dace.frontend.python.newast.ProgramVisitor`
to handle appropriately any changes to the Python AST (Abstract Syntax Tree) module. More details can be found in the
official `Python documentation <https://docs.python.org/3/library/ast.html>`_.

Main Limitations
----------------

- Classes and object-oriented programing are only supported in JIT mode (see :doc:`jitoop`).
- Python native containers (tuples, lists, sets, and dictionaries) are not supported **directly** as :class:`~dace.data.Data`. Specific instances of them may be **indirectly** supported through code preprocessing (see :doc:`preprocessing`). There is also limited support for specific uses, e.g., as arguments to some methods.
- Only the `range <https://docs.python.org/3/library/stdtypes.html#range>`_, :func:`parrange`, and :func:`~dace.frontend.python.interface.map` iterators are **directly** supported. Other iterators, e.g., `zip` may be **indirectly** supported through code preprocessing (see :doc:`preprocessing`).
- Recursion is not supported.

Parsing Flow
------------

The entry point for parsing a Python program with the :class:`~dace.frontend.python.newast.ProgramVisitor` is the :func:`~dace.frontend.python.newast.parse_dace_program` method.
The Python call tree when calling or compiling a Data-Centric Python program is as follows:

1. :class:`~dace.frontend.python.parser.DaceProgram`
2. :func:`~dace.frontend.python.parser.DaceProgram.__call__`, or :func:`~dace.frontend.python.parser.DaceProgram.compile`, or :func:`~dace.frontend.python.parser.DaceProgram.to_sdfg`
3. :func:`~dace.frontend.python.parser.DaceProgram._parse`
4. :func:`~dace.frontend.python.parser.DaceProgram._generated_pdp`
5. :func:`~dace.frontend.python.newast.parse_dace_program`
6. :func:`~dace.frontend.python.newast.ProgramVisitor.parse_program`

The ProgramVisitor Class
--------------------------------------------------------------

The :class:`~dace.frontend.python.newast.ProgramVisitor` traverses a Data-Centric Python program's AST and constructs
the corresponding :class:`~dace.sdfg.sdfg.SDFG`. The :class:`~dace.frontend.python.newast.ProgramVisitor` inherits from Python's `ast.NodeVisitor <https://docs.python.org/3/library/ast.html#ast.NodeVisitor>`_
class and, therefore, follows the visitor design pattern. The developers are encouraged to accustom themselves with this
programming pattern (see <add-some-resources>), however, the basic functionality is described in <insert-section>.
An object of the :class:`~dace.frontend.python.newast.ProgramVisitor` class is responsible for a single :class:`~dace.sdfg.sdfg.SDFG`
object. While traversing the Python program's AST, if the need for a :class:`~dace.sdfg.nodes.NestedSDFG` arises (see <add-section>), a new
(nested) :class:`~dace.frontend.python.newast.ProgramVisitor` object will be created to handle the corresponsding Python
Abstract Syntax sub-Tree.
