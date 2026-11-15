.. _canonicalize:

Canonicalization and the Canonical Parallel Form
================================================

Canonicalization rewrites an SDFG into the Canonical Parallel Form (CPF), a restricted shape of the SDFG IR that a
program reaches before any target-specific optimization runs. In the CPF, every axis whose iterations are proven
independent is a Map, every loop that stays a loop carries the reason it stays sequential, and every computation the
pipeline recognizes (a reduction, a scan, a contraction, a copy) is a library node. Two programs that compute the
same thing through different spellings of the same loops reach the same form, so a later pass, a code generator, or a
person reading the result sees one shape per computation.

Canonicalization is the first of two steps. It stops where the target begins: wherever the choice between a parallel
and a sequential form is open, it keeps the parallel form, because a CPU, a GPU, and a vectorizer can each still
specialize from it. The second step, :func:`~dace.transformation.passes.canonicalize.finalize.finalize_for_target`,
makes the target decisions (which maps become OpenMP regions or kernels, which library implementation runs, where
transients live).


Using Canonicalization
----------------------

The entry point is :func:`~dace.transformation.passes.canonicalize.pipeline.canonicalize`, which works in place and
returns the same SDFG. ``canonicalize`` followed by ``finalize_for_target`` is the counterpart of
:ref:`auto_optimize <opt_auto>`:

.. code-block:: python

    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu

    sdfg = program.to_sdfg()

    # CPU
    canonicalize(sdfg, target='cpu')
    finalize_for_target(sdfg, 'cpu')

    # GPU: the device move is a separate step, so passes can run between the two
    canonicalize(sdfg, target='gpu')
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, 'gpu')

    sdfg.compile()

The ``target`` argument selects a preset of knobs (``CPU_DEFAULTS`` and ``GPU_DEFAULTS`` in the pipeline module);
an explicit argument overrides the preset. The arguments most callers set are:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Effect
   * - ``target``
     - ``'cpu'`` (default) or ``'gpu'``. Picks the knob preset, for example whether a carried loop is interchanged
       with an enclosing map (on for CPU, off for GPU).
   * - ``lift``, ``lift_copy``, ``semantic_lifting``
     - Recognize contractions, element-wise copies and fills, and the other library operators. All on by default;
       ``semantic_lifting=False`` keeps every loop nest as plain maps (the vectorizer uses it).
   * - ``assume_parallel_guards``
     - Off by default. When on, a loop that is parallel only under a runtime condition becomes the parallel map alone,
       with no sequential fallback. This is unsound if the condition fails at run time.
   * - ``nonnegative_symbols``
     - The symbols the caller knows are never negative. Every proof can use them, and the program checks them at
       entry. Any other symbol has an unknown sign.
   * - ``specialize_constants``
     - Symbol values to bake in before canonicalization, so constant trip counts unroll and small matrix extents
       become concrete.
   * - ``unroll_limit``, ``peel_limit``
     - Unroll constant-trip loops up to this many iterations; peel up to this many iterations to expose a parallel
       remainder. Zero disables either one.
   * - ``validate``, ``validate_all``
     - Validate the result, or the graph after every stage. Both are off by default, since validation is a
       whole-graph walk and the stage units do not validate themselves.
   * - ``stages``
     - Run only the listed stage labels (see below).
   * - ``stats``
     - Record which stage settled each loop and the time spent in the solvers (see below).


What the Form Guarantees
------------------------

A canonicalized SDFG satisfies the following properties. A pass that consumes the CPF can rely on them, and a pass
inside the pipeline must preserve them.

1. **Structured control flow.** Every loop is a :class:`~dace.sdfg.state.LoopRegion` and every branch a
   :class:`~dace.sdfg.state.ConditionalBlock`. The ingest stage raises control flow with
   :class:`~dace.transformation.passes.simplification.control_flow_raising.ControlFlowRaising`, and
   ``RequireStructuredControlFlow`` refuses an SDFG that still has unstructured control flow afterwards.
2. **Parallel by default.** An axis whose iterations are independent is a Map. An axis stays a loop only when its
   carried dependence is proven, or when no analysis reached a verdict. Making a parallel map sequential again for its
   size or its nesting is a decision of the CPU specialization, not of canonicalization.
3. **Recognized computations are library nodes.** Reductions (argmax and argmin included), scans, tensor
   contractions, transposes, symmetric products, rank-k updates, unpivoted LU factorizations, copies, and fills become
   library nodes. Their ``implementation`` stays unset: ``finalize_for_target`` picks the fast implementation per
   target, and code generation expands each node once.
4. **One reduction shape.** A write-conflict resolution (WCR) appears only where a tasklet writes through a chain of
   map exits into an access node. A WCR that is not a reduction is reverted to a plain write.
5. **Guarded parallelism is explicit.** A loop that is parallel only under a runtime fact (no index conflicts in a
   scatter, a bound on a symbol) becomes a conditional block: the parallel map when the condition holds, the original
   loop otherwise. Facts the caller declared (``nonnegative_symbols``) are checked once at program entry.
6. **Few target decisions.** The form contains no OpenMP or kernel-launch choices and no persistent kernels, and
   invariant guards stay pushed into the loops; hoisting them is a step of the target specialization. The
   ``target`` argument changes a few choices between equally parallel forms: whether a carried loop is interchanged
   with an enclosing map, the chunk of a find-first search, and the tile a wavefront skew walks (64 by 64 on both
   targets, with a thread-block diagonal inside a GPU tile). Canonicalization does not tile any other nest.
7. **Deterministic.** The same input SDFG and the same DaCe commit produce the same form, node for node.


Pipeline Stages
---------------

:class:`~dace.transformation.passes.canonicalize.pipeline.CanonicalizationPipeline` runs a fixed recipe of passes,
each tagged with a stage label. :func:`~dace.transformation.passes.canonicalize.pipeline.stage_labels` returns the
labels in order:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Label
     - What the stage does
   * - ``ingest``
     - Raise structured control flow, simplify the parsed program, settle one spelling (no views, no trivial tasklets
       or casts, unique loop iterators, ``while`` loops with a counter as ``for`` loops), and lift a loop that breaks
       on its first hit into a chunked find-first search.
   * - ``normalize_statements``
     - Lower every map to a loop, reroll unrolled loops, undo tiling, and run a fixpoint of induction-variable
       substitution, statement fission, privatization, and invariant motion. Distribute loops into perfect nests and
       permute them so the innermost loop walks the contiguous stride.
   * - ``lift_semantics``
     - Recognize library operators on the normalized nests (see property 3). Some lifts need a shape that a later
       stage creates; they run there and keep this label.
   * - ``cleanup``
     - The structural cleanup between stages: fold symbols, tidy the state machine, inline single-state nested
       SDFGs.
   * - ``derive_parallelism``
     - Convert loops to maps (:class:`~dace.transformation.interstate.loop_to_map.LoopToMap`), break anti-dependences
       by renaming, privatize scalars and arrays, peel loops to expose a parallel remainder, emit guarded maps
       (property 5), and skew a wavefront where a dependence spans two axes.
   * - ``recompose``
     - Collapse and fuse maps and states to a fixpoint, and lift what fusion exposes.
   * - ``fuse``
     - Maximal fusion alone, so a caller that wants to own the fusion decisions can stop before it.
   * - ``finish``
     - Final bookkeeping: drop dead fills and unreferenced transients, inline the remaining control-flow regions,
       emit the entry check of the declared symbol facts, and spell floor division and modulo for the code
       generator.

No label before ``fuse`` recurs after it, so ``canonicalize(sdfg, stages=...)`` with a prefix or a suffix of the
labels runs exactly that part of the recipe. A caller can run the stages up to a label, apply its own passes, and run
the rest.


Dependence Tests
----------------

``LoopToMap`` decides whether a loop's iterations are independent with tests of increasing cost. A loop it converts is
settled at one of three levels:

* **Level 1**: subscript and range tests on the memlets. Most loops settle here.
* **Level 2**: a polyhedral wavefront skew (isl). The skew moves every dependence onto one axis and leaves the other
  axes parallel.
* **Level 3**: an SMT query (Z3) that proves a pair of accesses never conflicts.

A loop can also settle by lifting (it became a library node) or as a guarded loop (property 5). A loop that settles
nowhere stays a :class:`~dace.sdfg.state.LoopRegion`.


Reading the Result
------------------

:func:`dace.codegen.cpf.cpf` renders a canonicalized SDFG as self-contained source: C++ or C (``language='c'``) that a
bare host compiler accepts with OpenMP and no DaCe runtime, or a host and a device unit for ``'hip'`` and ``'cuda'``.
Before rendering, ``AnnotateLoopKinds`` names every loop with a comment, so a ``for`` loop in the output says why it is
there:

* **parallel**: a map; the iterations are independent, and the schedule is a separate decision;
* **sequential**: a carried dependence is proven, and the comment names the access that carries it;
* **unknown**: no test reached a verdict; the loop is kept sequential to be safe and may well be parallel.

Wavefront axes carry one of these words followed by their role (diagonal, front, tile). The comments are notes: no
pass dispatches on them, and they appear only in the standalone rendering.


Statistics and Debugging
------------------------

``canonicalize(sdfg, stats=...)`` records, for every loop, the stage and the pass that settled it, together with the
time spent in the whole pipeline, in Z3, and in isl (:mod:`~dace.transformation.passes.analysis.canon_stats`). A
settled loop's outcome is ``level1``, ``level2``, or ``level3`` (the dependence test above), ``guarded``,
``lift:<pass>`` (it became a library node), or ``restructured:<pass>`` (another rewrite removed it). A loop left in
the form gets the class its rendering comment states, for example ``sequential_witness`` (the carrying access is
known), ``guarded_fallback``, or ``never_examined``. A dict argument is filled with the record; a path gets the
record appended as one JSON line. Without ``stats``, the pipeline does the same work and records nothing.

To find the stage that broke a program, use
:func:`~dace.transformation.passes.canonicalize.debug.first_failing_stage`. It canonicalizes a copy stage by stage,
runs the SDFG on random inputs after every stage, and returns the first stage whose result is invalid or differs from
the input program. ``CanonicalizationPipeline(dump_dir=...)`` saves the SDFG after every stage for inspection, and
``validate_all=True`` validates after every stage.
