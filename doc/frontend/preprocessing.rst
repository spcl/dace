.. _preprocessing:

Preprocessing Python AST
========================

Before the Python frontend lowers a ``@dace.program`` into an SDFG, the function's
abstract syntax tree (AST) is rewritten by a small pipeline of *preprocessing*
passes. Preprocessing has three goals:

1. Resolve every name in the function body to either an SDFG symbol, a data
   container, a closure constant, or an external Python callback.
2. Specialize Python constructs that the SDFG IR does not represent natively
   (for example, list/tuple unpacking, context managers, augmented assignments,
   or f-strings) into a form the parser can handle.
3. Reject programs that use language features which DaCe explicitly does not
   support (see :doc:`pysupport`).

The pipeline is implemented in :mod:`dace.frontend.python.preprocessing` and
is entered through :func:`~dace.frontend.python.preprocessing.preprocess_dace_program`.
The same entry point is reused recursively when one ``@dace.program`` calls
another, which keeps the closure of every nested SDFG self-consistent.

Pipeline overview
-----------------

The passes run in the following order. Most of them are
:class:`ast.NodeTransformer` subclasses; a few are :class:`ast.NodeVisitor`
checks that only validate the program.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pass
     - Role
   * - :class:`~dace.frontend.python.preprocessing.StructTransformer`
     - Rewrites ``dace.struct`` instantiations into the corresponding type
       constructor calls so that downstream passes can see them as plain
       function calls.
   * - :class:`~dace.frontend.python.preprocessing.ModuleResolver`
     - Replaces aliased module references (``import numpy as np``) with their
       canonical names so that later replacement lookups (e.g.,
       ``numpy.dot``) succeed regardless of how the user imported the module.
   * - :class:`~dace.frontend.python.preprocessing.MPIResolver`
     - Recognizes ``mpi4py`` communicator method calls and rewrites them into
       the corresponding ``dace.comm`` library calls. Skipped silently if
       ``mpi4py`` is not installed.
   * - :class:`~dace.frontend.python.preprocessing.ModuloConverter`
     - Normalizes Python's modulo operator to DaCe's C-style semantics (see
       :doc:`pysupport`, section 6.7).
   * - :func:`~dace.frontend.python.preprocessing.find_disallowed_statements`
     - Walks the AST and raises a ``TypeError`` if any statement listed in
       :data:`~dace.frontend.python.newast.DISALLOWED_STMTS` is present
       (``import``, ``try``, ``yield``, ``del``, ...). When disallowed
       statements are found, the function is converted into a Python callback
       instead of being parsed as an SDFG.
   * - :class:`~dace.frontend.python.preprocessing.GlobalResolver`
     - The largest pass. Resolves every free name to one of: a closure
       constant (substituted in-place), a captured array (registered in the
       :class:`~dace.frontend.python.parser.SDFGClosure`), a DaCe symbol, or
       a callable to be inlined or invoked as a callback. ``visit_Assert``
       and ``visit_Raise`` live here as well; ``assert`` is folded statically
       and ``raise`` is replaced by a warning.
   * - :class:`~dace.frontend.python.preprocessing.DisallowedAssignmentChecker`
     - Rejects assignments that would mutate compile-time constants or
       walrus-bound names that are visible from the closure.
   * - :class:`~dace.frontend.python.preprocessing.LoopUnroller`
     - Statically unrolls ``for`` loops whose iterator is a literal sequence
       (``range`` over compile-time bounds, captured tuples, ``enumerate``,
       ``zip``, ``dace.unroll``). Unrolling is what enables many
       metaprogramming patterns that the SDFG IR cannot express directly.
   * - :class:`~dace.frontend.python.preprocessing.ExpressionInliner`
     - Inlines call expressions whose callee is a pure Python function
       captured from the closure (including lambdas), as long as the result
       can be expressed as a single AST expression.
   * - :class:`~dace.frontend.python.preprocessing.ContextManagerInliner`
     - Replaces ``with`` blocks (other than ``with dace.tasklet``) with the
       inlined ``__enter__`` / ``__exit__`` calls of the underlying context
       manager, including handling of early ``return`` / ``break`` /
       ``continue``.
   * - :class:`~dace.frontend.python.preprocessing.ConditionalCodeResolver`
     - Folds ``if`` / ``elif`` branches whose condition is a compile-time
       constant.
   * - :class:`~dace.frontend.python.preprocessing.DeadCodeEliminator`
     - Removes unreachable code after the previous folding step (e.g.,
       statements after a ``return`` in a fully-folded branch).
   * - :class:`~dace.frontend.python.preprocessing.AugAssignExpander`
     - Expands ``a += b`` to ``a = a + b`` for cases where the left-hand
       side is not a simple data access.
   * - :class:`~dace.frontend.python.preprocessing.CallTreeResolver`
     - Discovers all transitively-called ``@dace.program`` and
       ``SDFGConvertible`` objects so that they can be parsed and added to
       the closure.

The closure resolver, conditional resolver, dead-code eliminator, and
expression inliner run *together* in a fixed-point loop: as long as one of
them rewrites the AST, the loop is run again. The maximum number of passes
is controlled by the ``frontend.preprocessing_passes`` configuration entry
(``-1`` means "run until quiescent").

The closure
-----------

The output of preprocessing is two-fold:

* a :class:`~dace.frontend.python.parser.PreprocessedAST` containing the
  rewritten AST, the source file, and the resolved global namespace; and
* a :class:`~dace.frontend.python.parser.SDFGClosure` recording every
  external object that the program needs at call time - captured arrays,
  constants, nested ``@dace.program`` callees, and Python callbacks.

The closure is what allows ``@dace.program`` to behave like a regular
Python function from the caller's point of view while still producing a
fully self-contained SDFG: when the compiled SDFG is invoked, the closure
is used to bind the captured external state to the program's arguments.

Disabling or inspecting preprocessing
-------------------------------------

* Set the configuration option ``frontend.verbose_errors = true`` to see the
  exact pass that raised an exception during preprocessing.
* Set ``frontend.preprocessing_passes`` to a positive integer to cap the
  number of fixed-point iterations (useful when debugging an infinite loop
  in a custom replacement that produces new AST every pass).
* Use :func:`dace.frontend.python.astutils.unparse` to dump the
  intermediate AST after any pass during development.

For background on what comes after preprocessing, see :doc:`parsing` and
:ref:`python-frontend`.
