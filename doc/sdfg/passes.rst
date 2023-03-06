Passes and Pipelines
====================

Passes are an important part of compiler infrastructures, enabling efficient modification of a whole program.
All modifications to an SDFG, including pattern-matching transformations, are performed as passes on the graph.

Passes can be grouped together in Pipelines and depend on each other. The :class:`~dace.transformation.pass_pipeline.Pipeline` 
object, upon applying, will ensure dependencies are met and that passes do not run redundantly.

See more information and examples in :ref:`available_passes`.

.. _pass:

Passes
------

A :class:`~dace.transformation.pass_pipeline.Pass` is an SDFG analysis or manipulation that registers as part of the
SDFG history. Classes that extend Pass can be used for optimization purposes, to collect data on an entire SDFG,
for cleanup, or other uses. 

A Pass is defined by one main method: :func:`~dace.transformation.pass_pipeline.Pass.apply_pass`. This method receives
the SDFG to manipulate/analyze, as well as the previous Pipeline results (if run in the context of a pipeline). 

.. note::
    The return value of a pass serves as a report of the work performed by the pass. A pass returns ``None``
    only if it did not perform any change on the graph. **Always return some object if you made changes to the graph**, even
    if it is an empty dictionary or zero.


An example of a simple pass that only traverses the graph and finds the number of total states is:

.. code-block:: python

    from dace import SDFG
    from dace.transformation import pass_pipeline as ppl
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass
    class CountStates(ppl.Pass):
        """
        Counts states in this SDFG and potentially nested SDFGs.
        (this description will appear in the Visual Studio Code plugin)
        """
        recursive: bool = False  # If True, traverses graph into nested SDFGs

        def modifies(self) -> ppl.Modifies:
            # This is an analysis pass, so it does not modify anything
            return ppl.Modifies.Nothing
        
        def should_reapply(self, modified: ppl.Modifies) -> bool:
            # We should rerun this pass if the state structure has changed
            return modified & ppl.Modifies.States

        def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> int:
            """
            Counts the states and returns the result.
            """
            result = 0

            if self.recursive:
                # If recursive, also counts nested SDFG results
                for sd in sdfg.all_sdfgs_recursive():
                    result += sd.number_of_nodes()
            else:
                # Otherwise, simply count the states in this graph
                result = sdfg.number_of_nodes()

            return result
            

    # To use this pass, we create an object and give it the SDFG, as well as an empty
    # dictionary for previous pipeline results
    result = CountStates(recursive=True).apply_pass(sdfg, {})
    print('SDFG has', result, 'states')


To improve productivity, we provide specific types of Passes that can be extended as necessary, for example :class:`~dace.transformation.pass_pipeline.VisitorPass`:

.. code-block:: python

    class HasWriteConflicts(VisitorPass):
        def __init__(self):
            self.found_wcr = False

        def visit_Memlet(self, memlet: dace.Memlet, parent: dace.SDFGState, pipeline_results: Dict[str, Any]):
            if memlet.wcr:
                self.found_wcr = True

                # If a value is returned, a dictionary key will be filled with the visited object and the value
                return memlet.wcr

    wcr_checker = HasWriteConflicts()
    memlets_with_wcr = wcr_checker.apply_pass(sdfg, {})
    print('SDFG has write-conflicted memlets:', wcr_checker.found_wcr)
    print('Memlets:', memlets_with_wcr)


Other extensible sub-classes are :class:`~dace.transformation.pass_pipeline.StatePass` and :class:`~dace.transformation.pass_pipeline.ScopePass`,
which apply on each state or scope, respectively.

.. _pass_pipeline:

Pipelines
---------

Passes may depend on each other through a :class:`~dace.transformation.pass_pipeline.Pipeline` object.
A pass pipeline contains multiple, potentially dependent Pass objects, and applies them in the correct order.
Each contained pass may depend on other (e.g., analysis) passes, which the pipeline avoids rerunning depending on which
elements were modified by applied passes. An example of a built-in pipeline is the :class:`~dace.transformation.passes.simplify.SimplifyPass`,
which runs multiple complexity reduction passes and may reuse data across them. Prior results of applied passes are contained in
the ``pipeline_results`` argument to ``apply_pass``, which can be used to access previous return values of Passes.

The return value of applying a pipeline is a dictionary whose keys are the Pass subclass names and values are the return
values of each pass.

A Pipeline in itself is a type of Pass, so it can be arbitrarily nested in other Pipelines. Its
dependencies and modified elements are unions of the contained Pass objects.

In every Pass, there are three optional pipeline-related methods that can be implemented:

  * ``depends_on``: Which other passes this pass requires
  * ``modifies``: Which elements of the SDFG does this Pass modify (used to avoid re-applying when unnecessary)
  * ``should_reapply``: Given the modified elements of the SDFG, should this pass be rerun?

So what kind of elements can be modified? We provide a flag object called :class:`~dace.transformation.pass_pipeline.Modifies`
that specifies what type of elements in the graph to include. For example, ``Modifies.Memlets | Modifies.AccessNodes``
tells the system that both were modified.

To use an existing pipeline, all that is necessary is to instantiate it and call it. For example: ``MyPipeline().apply_pass(sdfg, {})``.
To create a new pipeline from existing passes, instantiate the object with a list of Pass objects, or extend the
Pipeline class (e.g., if pipeline order should be modified). For example:

.. code-block:: python

    my_simplify = Pipeline([ScalarToSymbolPromotion(integers_only=False), ConstantPropagation()])
    results = my_simplify.apply_pass(sdfg, {})
    print('Promoted scalars:', results['ScalarToSymbolPromotion'])


