.. _new-frontend:

Writing a New Frontend
======================

A frontend in DaCe is the component that translates a source program (Python,
Fortran, MLIR, ...) into an SDFG. Existing frontends are organized under
:mod:`dace.frontend` and serve as references when adding a new one. The
goal of this page is to capture the design conventions that have made those
frontends robust enough to ingest large real-world codebases.

The general advice that runs through every section below is: **stay
data-centric**. The SDFG IR is good at expressing parallelism, locality, and
data movement; it is not designed to express pointer arithmetic, exceptional
control flow, or escaping closures. A frontend's job is to massage the input
language until it "speaks" in those terms - without prematurely picking a
schedule - that is what :ref:`transformations <transformations>` are for.

Trust the transformations
-------------------------

Resist the urge to optimize while parsing. Producing a *correct, naive* SDFG
that places statements in distinct states, materializes intermediate buffers,
and uses straightforward maps is almost always the right thing to do; the
:ref:`simplification pipeline <simplify>` will fuse states, eliminate
redundant copies, and inline nested SDFGs. Premature fusion or reordering at
parse time tends to interact badly with later passes and makes the frontend
harder to maintain.

When constructing IR objects, populate ``debuginfo`` with the original source
location. This is much cheaper than reconstructing it later and lets users
correlate generated nodes back to their source. The Python frontend uses
Python AST location information for this purpose; new frontends are encouraged
to follow the same pattern.

Make things data-centric
------------------------

When you encounter a construct that does not map directly to an SDFG node:

* **Preprocess the input AST** before lowering. The Python frontend uses an
  AST-to-AST rewrite pipeline (see :doc:`/frontend/preprocessing`) to desugar
  syntactic elements, inline context managers, unroll metaprogramming loops,
  and fold compile-time constants. New frontends should prefer this strategy
  over teaching the IR builder about every possible source-language quirk.
* **Expose memory access patterns** so that DaCe can reason about them.
  Concrete index expressions (``A[i, j]``) are far more useful than opaque
  pointer arithmetic. If the source language has loop bounds or array shapes
  parameterized by symbols, propagate those symbols through to the SDFG
  rather than substituting them with concrete values.
* **Avoid raw pointers** where possible. The recommended progression is:
  arrays first, then :class:`~dace.data.View` for slices and aliases, then
  (only as a last resort) :class:`~dace.data.Reference` for restricted
  pointer semantics.

Encapsulation, callbacks, and closures
--------------------------------------

Most languages have constructs that the SDFG cannot express directly: I/O,
random number generation, calls into the host runtime, exceptions, etc. The
established pattern is to expose these as **Python callbacks** - opaque
function invocations whose body lives in user-provided Python code - and to
thread a per-program ``__state`` value through the call chain so that
internal state is not lost across calls.

Some practical guidelines:

* Use a single ``__state`` value per stateful resource (a library, a random
  generator, an MPI communicator, ...) so that the SDFG framework can track
  data dependencies on it. Without it, code-generator passes are free to
  reorder calls in ways that violate the source language's semantics.
* If you can prove that two stateful subsystems do not interact, model them
  with separate state values. This unlocks more parallelism downstream.
* Do not lose the closure! When a source-language function captures state
  from its lexical environment, the frontend must arrange for that state to
  flow into the SDFG (typically as additional arguments or as scalar/array
  members of a closure object). The Python frontend's
  :class:`~dace.frontend.python.parser.SDFGClosure` is a useful reference
  for what such an object looks like.

Know your language
------------------

A robust frontend accounts for **every** AST node it might encounter, even
if only by emitting an explicit error. Silent fall-through to "treat this as
a no-op" is the source of most frontend bugs. Maintain an explicit list of
disallowed constructs (see ``DISALLOWED_STMTS`` in
:mod:`dace.frontend.python.newast` for the Python frontend's variant) and
fail fast when one shows up.

When the source language has well-known operators or library calls with a
matching DaCe library node, lower directly to that library node rather than
synthesizing the operation manually. For example, the Python frontend lowers
``@`` and ``numpy.matmul`` to the BLAS ``MatMul`` library node, which gives
downstream optimization a hook to swap in vendor-tuned implementations.

Assumptions
-----------

Two assumptions are particularly important to surface explicitly:

* **Aliasing.** If you have whole-program analysis, you can compute precise
  no-alias information per call. Otherwise, mark potentially-aliasing
  arrays with the :attr:`~dace.data.Data.may_alias` property; the codegen
  honors this when emitting array declarations and the optimizer treats it
  as a barrier.
* **By-reference vs. by-value.** Source languages that pass aggregates by
  reference should be lowered using views; languages that pass them by
  value should produce explicit copies into transients. References are
  available for cases where neither view nor copy is appropriate, but they
  disable some optimizations.

Limit the scope
---------------

Try to build the frontend as a per-scope (per-function, per-translation-unit)
parser. Each scope produces its own SDFG, and inter-scope calls become
nested SDFGs. This mirrors the Python frontend's structure: one
:class:`~dace.frontend.python.newast.ProgramVisitor` per ``@dace.program``,
parented by a single closure. It has two benefits:

* the IR for each scope stays small and easy to debug; and
* nested SDFGs are first-class citizens of the IR and can be inlined,
  duplicated, or specialized by transformations.

For function calls within a scope, prefer creating a nested SDFG over
inlining the body verbatim. The :class:`~dace.transformation.interstate.sdfg_nesting.InlineSDFG`
transformation will inline at the user's request, but reversing an
inlining is much harder.
