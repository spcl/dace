Auto-Optimization Pass (Experimental)
=====================================

Like :ref:`simplify`, auto-optimization is a set of passes applied on an SDFG in a specific order. The goal of this pass
is to provide automatic optimization heuristics. This includes finding parallelism in programs, optimizing common
performance pitfalls (such as excessive allocation/deallocation), finding fast library implementations (e.g., BLAS-optimized
routines for linear algebra operations), and more.

.. warning::
    Auto-optimization depends on the graph and the target platform. As it is still an experimental feature, it is **not** applied
    automatically on every graph and *may break correctness of certain programs*, so use it with caution.


How to trigger auto-optimization
--------------------------------

There are several ways to invoke auto-optimization. First, you can configure it to run on any program in ``.dace.conf``
under ``optimizer.autooptimize`` (or setting the ``DACE_optimizer_autooptimize`` environment variable to ``1``).
Another way to do so is on the ``@dace`` decorator, as shown below:

.. code-block:: python

    import dace

    @dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
    def hello_world(A, B):
        return A @ B


Lastly, it can be triggered on arbitrary SDFGs by calling the auto-optimization pass:

.. code-block:: python

    import dace
    from dace.transformation.auto import auto_optimize as aopt

    opt_sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.CPU)



What does the pass contain?
---------------------------

The auto-optimization pass (:func:`~dace.transformation.auto.auto_optimize.auto_optimize`) includes the following
transformations, applied in this order:

  * Loop-to-map conversion (auto-parallelization): :class:`~dace.transformation.interstate.loop_to_map.LoopToMap`
  * :ref:`simplify`
  * Multi-dimensional :class:`~dace.transformation.dataflow.map_collapse.MapCollapse` to parallelize across multiple dimensions.
  * Greedy subgraph fusion (fusing contents of maps with common dimensions to reduce data movement). See :class:`~dace.transformation.subgraph.subgraph_fusion.SubgraphFusion` for more information.
  * Move loops into maps (when memory access pattern permits) in order to increase the granularity of work threads perform (:class:`~dace.transformation.interstate.move_loop_into_map.MoveLoopIntoMap`).
  * (for FPGAs) Interleave data containers (e.g. arrays) in off-chip memory banks, and use local memory (e.g. BRAM) when possible.
  * Tiling of maps with write-conflict resolution to reduce atomic operations (tile sizes are configurable via 
    ``optimizer.autotile_size``). Partial parallelism (non-conflicting dimensions) can also be extracted to convert 
    atomics to simple updates (configurable in ``optimizer.autotile_partial_parallelism``, True by default).
  * Set all library nodes to expand to fast implementations: first using the ``fast`` expansion if exists, and then via
    heuristics for choosing the fastest library for the target device (e.g., MKL on CPU if available, CUBLAS on GPU).
  * Disable OpenMP sections (usually increases performance at the expense of reducing parallelism within a state).
  * Specialize known symbolic values to the known constants.
  * Move small arrays from heap to stack (threshold is also configurable in ``optimizer.autotile_size``).
  * Make transient data containers' allocation lifetime ``dace.AllocationLifetime.Persistent``, if possible. This moves
    allocation and deallocation out of the critical code path and into the SDFG init/exit functions.

Apart from those, the pass transforms the SDFG to run on the specified platform (e.g., GPU, FPGA).
