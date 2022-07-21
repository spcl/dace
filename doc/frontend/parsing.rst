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
Abstract Syntax sub-Tree. The :class:`~dace.frontend.python.newast.ProgramVisitor` has the following attributes:

- `filename`: The name of the file containing the Data-Centric Python program.
- `src_line`: The line (in the file) where the Data-Centric Python program is called.
- `src_col`: The column (in the line) where the Data-Centric Python program is called.
- `orig_name`: The name of the Data-Centric Python program.
- `name`: The name of the generated :class:`~dace.sdfg.sdfg.SDFG` object. `name` and `orig_name` differ when generating a :class:`~dace.sdfg.nodes.NestedSDFG`.
- `globals`: The variables defined in the global scope. Typically, these are modules imported and global variables defined in the file containing the Data-Centric Python program. 
- `closure`: The closure of the Data-Centric Python program (see :doc:`preprocessing` and :doc:`jitoop`).
- `nested`: True if generating a :class:`~dace.sdfg.nodes.NestedSDFG`.
- `simplify`: True if the :func:`~dace.sdfg.sdfg.SDFG.simplfy` should be called on the generated :class:`~dace.sdfg.sdfg.SDFG` object.
- `scope_arrays`: The Data-Centric Data (see :mod:`~dace.data`) defined in the parent :class:`~dace.sdfg.sdfg.SDFG` scope.
- `scope_vars`: The variables defined in the parent :class:`~dace.frontend.python.newast.ProgramVisitor` scope.
- `numbers`: DEPRECATED
- `variables`: The variables defined in the current :class:`~dace.frontend.python.newast.ProgramVisitor` scope.
- `accesses`: A dictionary of the accesses to Data defined in a parent :class:`~dace.sdfg.sdfg.SDFG` scope. Used to avoid generating duplicate :class:`~dace.sdfg.nodes.NestedSDFG` connectors for the same Data subsets accessed.
- `views`: A dictionary of Views and the Data subsets viewed. Used to generate Views for Array slices.
- `nested_closure_arrays`: The closure of nested Data-Centric Python programs (see :doc:`preprocessing` and :doc:`jitoop`).
- `annotated_types`: A dictionary from Python variables to Data-Centric datatypes. Used when variables are explicitly type-annotated in the Python code.
- `map_symbols`: The :class:`~dace.sdfg.nodes.Map` symbols defined in the :class:`~dace.sdfg.sdfg.SDFG`. Useful when deciding when an augmented assignment should be implemented with WCR or not.
- `sdfg`: The generated :class:`~dace.sdfg.sdfg.SDFG` object.
- `last_state`: The (current) last :class:`~dace.sdfg.state.SDFGState` object created and added to the :class:`~dace.sdfg.sdfg.SDFG`.
- `inputs`: The input connectors of the generated :class:`~dace.sdfg.nodes.NestedSDFG` and a :class:`~dace.memlet.Memlet`-like representation of the corresponding Data subsets read.
- `outputs`: The output connectors of the generated :class:`~dace.sdfg.nodes.NestedSDFG` and a :class:`~dace.memlet.Memlet`-like representation of the corresponding Data subsets written.
- `current_lineinfo`: The current :class:`~dace.dtypes.DebugInfo`. Used for debugging.
- `modules`: The modules imported in the file of the top-level Data-Centric Python program. Produced by filtering `globals`.
- `loop_idx`: The current scope-depth in a nested loop construct.
- `continue_states`: The generated :class:`~dace.sdfg.state.SDFGState` objects corresponding to Python `continue <https://docs.python.org/3/library/ast.html#ast.Continue>`_ statements. Useful for generating proper nested loop control-flow.
- `break_states`: The generated :class:`~dace.sdfg.state.SDFGState` objects corresponding to Python `break <https://docs.python.org/3/library/ast.html#ast.Break>`_ statements. Useful for generating proper nested loop control-flow.
- `symbols`: The loop symbols defined in the :class:`~dace.sdfg.sdfg.SDFG` object. Useful for memlet/state propagation when multiple loops use the same iteration variable but with different ranges.
- `indirections`: A dictionary from Python code indirection expressions to Data-Centric symbols.


The Visitor Design Pattern
--------------------------

To Nest Or Not To Nest?
-----------------------

Visitor Methods
---------------

Helper Methods
--------------

