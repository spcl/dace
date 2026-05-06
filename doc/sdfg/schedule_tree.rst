.. _schedule_tree:

Schedule Tree
=============

The **schedule tree** is an alternative, tree-shaped intermediate representation of a
DaCe program. It is derived from an :ref:`SDFG <sdfg>` and represents the same computation,
but in a form that resembles structured imperative code rather than a graph of states and
dataflow multigraphs.

A schedule tree is composed of nested ``ScheduleTreeScope`` blocks (loops, branches,
maps, consume scopes, ...) and leaf ``ScheduleTreeNode`` statements (tasklets, copies,
library calls, symbol assignments, views, reference sets, ...). The root of the entire
program is a ``ScheduleTreeRoot``, which additionally carries the unified descriptor
repository (containers, symbols, constants, callback mapping, argument names).

A textual rendering of a schedule tree looks like a Python-like program, for example::

    for i = 0; (i < N); i = (i + 1):
      map j in [0:M]:
        B[i, j] = tasklet(A[i, j])
        C[i, j] = tasklet(...)

The implementation lives in :mod:`dace.sdfg.analysis.schedule_tree`, with the node
hierarchy defined in ``treenodes.py`` and conversion routines in ``sdfg_to_tree.py`` and
``tree_to_sdfg.py``.


Why a Schedule Tree?
--------------------

The SDFG is a graph-based IR that exposes parallelism and data movement explicitly. This
is excellent for transformations that reason about dataflow, scopes, and memory, but it
can be cumbersome for analyses or transformations that are naturally expressed on
*structured control flow* — for example:

* Reasoning about the **execution order** of statements without traversing states and
  inter-state edges.
* Performing **control-flow transformations** such as loop-invariant code motion or
  eliminating empty branches.
* Understanding the program at a **higher level** that closely mirrors the original
  source code.
* Implementing analyses that benefit from a **lexical, tree-shaped view** (e.g., visitor
  patterns analogous to Python's ``ast.NodeVisitor``).

In a schedule tree:

* Control flow is explicit and structured (``ForScope``, ``WhileScope``, ``DoWhileScope``,
  ``IfScope`` / ``ElifScope`` / ``ElseScope``, ``GBlock`` for irreducible flow).
* Dataflow scopes (``MapScope``, ``ConsumeScope``) can be nested within control flow or
  vice versa.
* Nested SDFGs are flattened: data containers from nested SDFGs are dealiased into a
  unified, top-level descriptor repository, so a single naming system applies throughout
  the tree.
* Each node provides ``input_memlets()`` / ``output_memlets()``, which return the set of
  read/write memlets of the node, optionally with memlet propagation across loops and
  maps.


When To Use It
--------------

Prefer the schedule tree representation when:

* You want to develop a frontend that is more naturally expressed on structured control
  flow graphs, for a smoother mapping from source code to IR.
* You want to write a transformation or analysis that is easier to express on
  **structured control flow** than on a state graph.
* You need a **lexical traversal** of the program (the visitor / transformer base
  classes ``ScheduleNodeVisitor`` and ``ScheduleNodeTransformer`` mirror Python's
  ``ast`` API).
* You want to inspect or pretty-print a program in a **human-readable, code-like form**
  (``ScheduleTreeRoot.as_string()``).
* You are reasoning about **input/output sets** of a region of code with memlet
  propagation, taking loop ranges and map parameters into account.

Prefer the SDFG representation when:

* The transformation reasons primarily about **dataflow** between access nodes,
  tasklets, and library nodes.
* You need to apply existing :ref:`pattern-matching transformations <transformations>`
  or :ref:`passes <pass_pipeline>`, most of which are written against the SDFG IR.
* You need to **generate code**: code generation operates on SDFGs (see
  :ref:`codegen`). To generate code from a schedule tree, convert it back to an SDFG
  first.


Converting Between SDFG and Schedule Tree
-----------------------------------------

Every SDFG can be converted to a schedule tree, and a schedule tree can be converted
back to an SDFG. The two conversions are not bit-exact inverses (the round-trip may
introduce fewer states, simplify trivial control flow, etc.), but they preserve program
semantics.

From SDFG to schedule tree::

    from dace.sdfg.analysis.schedule_tree import sdfg_to_tree

    stree = sdfg_to_tree.as_schedule_tree(sdfg)
    # Equivalent shortcut:
    stree = sdfg.as_schedule_tree()

    print(stree.as_string())

By default, the SDFG is deep-copied before conversion. Pass ``in_place=True`` to avoid
the copy; note that an in-place conversion may leave the original SDFG in an
inconsistent state.

From schedule tree back to SDFG::

    from dace.sdfg.analysis.schedule_tree import tree_to_sdfg

    sdfg = tree_to_sdfg.from_schedule_tree(stree)
    # Equivalent shortcut:
    sdfg = stree.as_sdfg(validate=True, simplify=True)

The reconstruction inserts state boundaries where required (e.g., write-after-write,
before labels, around irreducible control flow) according to the configured
``StateBoundaryBehavior``.


Working With Schedule Trees
---------------------------

The node API mirrors Python's ``ast`` module:

* ``ScheduleNodeVisitor`` walks the tree with ``visit_ClassName`` dispatch.
* ``ScheduleNodeTransformer`` walks the tree and lets each ``visit_*`` method return a
  replacement node, ``None`` to delete it, or a list of nodes to splice in.
* ``ScheduleTreeScope.preorder_traversal()`` yields all descendants in pre-order.

A small example that removes empty ``IfScope`` nodes::

    from dace.sdfg.analysis.schedule_tree import treenodes as tn

    class RemoveEmptyIfs(tn.ScheduleNodeTransformer):
        def visit_IfScope(self, node: tn.IfScope):
            if not node.children:
                return None
            return self.generic_visit(node)

    RemoveEmptyIfs().visit(stree)

Some ready-made passes live in :mod:`dace.sdfg.analysis.schedule_tree.passes`, such as
``remove_unused_and_duplicate_labels`` and ``remove_empty_scopes``.

To query memlets of a region with optional propagation::

    inputs = stree.input_memlets()    # Set of read memlets at the root scope
    outputs = stree.output_memlets()  # Set of written memlets at the root scope

    # Keep symbols local to inner scopes instead of propagating them outwards:
    raw_inputs = stree.input_memlets(keep_locals=True)


Caveats
-------

* The schedule tree is intended primarily for **analysis and structured transformation**,
  not for code generation. Convert back to an SDFG before invoking the code generator.
* Conversion **flattens nested SDFGs** and renames their containers to the top-level
  descriptors. If your analysis depends on nested SDFG identity, you must operate on
  the SDFG directly.
* Some constructs that have no structured equivalent (e.g., irreducible control flow)
  are represented inside a ``GBlock`` with explicit ``StateLabel`` and ``GotoNode``
  entries.
* Round-tripping ``SDFG -> schedule tree -> SDFG`` is **semantics-preserving but not
  graph-identical**; expect benign differences such as added/removed empty states or
  reorganized state transitions, especially when ``simplify=True`` is requested in
  ``as_sdfg()``.
