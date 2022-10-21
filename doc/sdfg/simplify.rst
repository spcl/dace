.. _simplify:

Simplify Pipeline
=================

Simplify is one of the Pass :ref:`pass_pipeline` designed to reduce the number of elements in an SDFG. By fusing states
together, inlining nested SDFGs, and other passes, a simplified SDFG is more analyzable towards optimization and data-centric
transformation. It also enables language frontends to write safe code (e.g., with one SDFG state per statement) without
worrying about making dataflow explicit.

The simplify pipeline is located in the :class:`~dace.transformation.passes.simplify.SimplifyPass` class. Internally, it
is a fixed-point pipeline (:class:`~dace.transformation.pass_pipeline.FixedPointPipeline`), which means that it will run
in a loop until no more transformations are performed on the input SDFG. This is guaranteed to happen, as no simplify
pass adds any new elements to the graph, only removes them.

By default, simplify runs automatically for every input program. It can also be triggered by the SDFG API, using
``sdfg.simplify()``. Below we discuss how to modify this behavior, gain more insights into the simplification process,
and explain what it does.


Configuring Simplify
--------------------

For debugging purposes, it might be useful to completely disable the automatic simplification process. This can be
configured in the ``.dace.conf`` file by turning :envvar:`optimizer.automatic_simplification` off. Alternatively, you could
set the environment variable ``DACE_optimizer_automatic_simplification=0``, which achieves the same effect (see :ref:`config`).

As simplify runs multiple passes, you may want to inspect what it actually performed on a given graph. To do so through
the API, call ``sdfg.simplify(verbose=True)`` (or set :envvar:`debugprint` to ``verbose``)

For validation, there are two options: ``validate`` controls whether to validate the SDFG (for soundness rather than correct
results) after ``SimplifyPass`` is complete. This is enabled by default. If you wish to validate the SDFG after every
simplify internal pass, set ``validate_all=True`` in the arguments to :func:`~dace.sdfg.sdfg.SDFG.simplify`.


How Simplify Works
------------------

Simplification tries to expose as much dataflow as possible by reducing the number of states and making each state as
large as possible. This both reduces synchronization points (which exist implicitly at the end of each state), and
ensures that dataflow is visible and can be used in transformations.

There is a particular order in which we apply the simplification passes, as a heuristic to maximize the effect and
minimize the number of loops until a fixed-point is reached:

  * We first inline SDFGs with :class:`~dace.transformation.passes.fusion_inline.InlineSDFGs`. We do this first since pre-fused states would normally only contain nested SDFGs and input/output access nodes.
  * :class:`~dace.transformation.passes.scalar_to_symbol.ScalarToSymbolPromotion` then promotes scalar values into symbolic expressions (if they can be represented as such). This clarifies access patterns in programs by making them symbolic, moving indirect memory accesses (e.g., ``A[scalar]``) out of a tasklet and into a memlet if the index is symbolically known.
  * :class:`~dace.transformation.passes.fusion_inline.FuseStates` fuses SDFG states together if their dataflow allows it 
    (i.e., will not create data races). This also merges access nodes between states to form clear data dependencies.
  * After states are fused, we perform optional array inference (:class:`~dace.transformation.passes.optional_arrays.OptionalArrayInference`)
    to understand which of the underlying data cannot be ``None`` (or a null pointer). This allows eliding checks such as
    ``if x is not None`` during optimization, and also creates a richer argument checking scheme that validates array arguments.
  * Since all symbolic expressions have been exposed, we can now perform :class:`~dace.transformation.passes.constant_propagation.ConstantPropagation`
    to propagate constant and symbolic values, and reduce the complexity of the graph. This later also helps in memlet
    intersection checks for automatic parallelization.
  * Following propagation, dead code elimination is provided by two passes: :class:`~dace.transformation.passes.dead_dataflow_elimination.DeadDataflowElimination`
    and :class:`~dace.transformation.passes.dead_state_elimination.DeadStateElimination`. The former removes nodes within
    SDFG states if their results are never used, and the latter checks the state transition conditions and removes states
    that will never be executed (for example, if ``x is None`` for a non-optional array).
  * After constants are propagated, and dead-dataflow/states are removed, many of the symbols on the SDFG will no longer
    be necessary. :class:`~dace.transformation.passes.prune_symbols.RemoveUnusedSymbols` removes those symbols from the graph.
  * :class:`~dace.transformation.passes.array_elimination.ArrayElimination` performs a coarse-grained dead memory elimination
    by removing redundant copies and unnecessary arrays/views.
  * Lastly, memlets with the same source/destination are merged by performing a union on the memlets' subsets in 
    :class:`~dace.transformation.passes.consolidate_edges.ConsolidateEdges`.

Following these passes, we end up reducing the following SDFG components: nested SDFGs, memlets, arrays and scalars,
and SDFG states and nodes in those states.
