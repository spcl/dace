.. _opt_sdfgapi:

Interactive Optimization with the SDFG API
==========================================

Frontends produce a *baseline* SDFG that mirrors the structure of the user's
source program. The Python SDFG API exposes every node, edge, and property
of that graph as a regular Python object, so an SDFG can be optimized
incrementally from a notebook or REPL session without recompiling the
program after every change. This page collects the entry points most often
needed when driving that loop by hand.

Loading and inspecting a baseline
---------------------------------

Any SDFG produced by a frontend or saved to disk can be loaded interactively:

.. code-block:: python

    import dace
    sdfg = dace.SDFG.from_file('matmul.sdfg')          # JSON or .sdfgz
    sdfg.view()                                        # opens VS Code or Jupyter
    sdfg.validate()                                    # raise on inconsistencies

The :meth:`~dace.sdfg.sdfg.SDFG.view` call renders the graph through the
:ref:`SDFV viewer <vscode>`. ``validate()`` will raise descriptive
exceptions on common issues such as missing memlets, dangling connectors, or
undefined symbols.

Operating on the graph
----------------------

The SDFG is just a stateful graph object. The most common operations are:

* Iterating: ``sdfg.states()``, ``sdfg.all_nodes_recursive()``,
  ``sdfg.arrays.items()``.
* Locating nodes by predicate, e.g.
  ``[n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]``.
* Editing properties in place: changing
  :attr:`~dace.sdfg.nodes.Map.schedule`, :attr:`~dace.data.Data.storage`,
  or :attr:`~dace.sdfg.nodes.AccessNode.data`.
* Saving and reloading checkpoints with :meth:`~dace.sdfg.sdfg.SDFG.save`
  followed by :meth:`~dace.sdfg.sdfg.SDFG.from_file`.

Each of these is safe to do inside a live session (even a notebook), as the SDFG
is a pure Python data structure until you call :meth:`~dace.sdfg.sdfg.SDFG.compile`.

Applying transformations interactively
--------------------------------------

The :ref:`transformation API <transformations>` provides three idioms for
firing a transformation from Python:

1. **Pattern matching** with :meth:`~dace.sdfg.sdfg.SDFG.apply_transformations`
   or :meth:`~dace.sdfg.sdfg.SDFG.apply_transformations_repeated` walks the
   graph, finds every match for a transformation class, and applies it.
   These methods accept a list of classes and an ``options`` dictionary,
   making them convenient for batch experiments.

2. **Direct application** via the ``apply_to`` class method targets a
   specific subgraph by passing the matched nodes as keyword arguments.
   This is the form used inside notebooks when you already know the nodes
   you want to operate on:

   .. code-block:: python

       from dace.transformation.dataflow import MapFusion
       MapFusion.apply_to(sdfg, first_map_exit=exit1, array=array_node,
                          second_map_entry=entry2)

3. **Whole-program passes** are run with
   :meth:`~dace.transformation.passes.pipeline.Pipeline.apply_pass` (or
   :meth:`~dace.sdfg.sdfg.SDFG.apply_passes`) and operate on the SDFG
   globally. Common examples include
   :class:`~dace.transformation.passes.scalar_to_symbol.ScalarToSymbolPromotion`
   and :class:`~dace.transformation.passes.transient_reuse.TransientReuse`.

In all three cases the SDFG is mutated in place, so chaining operations is
just calling them in sequence. To revert an experiment, reload from the
saved baseline.

Instrumentation feedback loop
-----------------------------

Interactive optimization works best when paired with
:ref:`runtime instrumentation <profiling>`. Set
:attr:`~dace.sdfg.nodes.Map.instrument` (or
:attr:`~dace.sdfg.sdfg.SDFG.instrument`) on the regions of interest, run
the program once, and inspect
:class:`~dace.codegen.instrumentation.report.InstrumentationReport` to
confirm whether the last transformation actually moved the needle. The
recommended workflow is:

1. Identify the hottest map or library node from a profile.
2. Apply a transformation (or a small sequence) targeting it.
3. Re-run; compare the new instrumentation report against the previous
   one.
4. If the change helped, save the SDFG; otherwise reload and try a
   different sequence.

Composing with the auto-optimizer
---------------------------------

The :ref:`automatic heuristics <opt_auto>` (``dace.transformation.auto.auto_optimize``)
are a useful starting point even when the goal is a fully manual schedule.
A common pattern is to call ``auto_optimize`` once to establish a strong
baseline, save the result, and then drive the remaining optimizations by
hand with the API described above.

For more involved manipulations, such as scripting transformation searches,
see :ref:`sdfg-api` and the ``samples/optimization`` directory in the
repository.
