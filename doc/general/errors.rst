.. _errors:

Development Errors and Gotchas
==============================

This section describes common errors and gotchas that you may encounter when extending DaCe.

.. rubric:: Unordered traversal

The default traversal order of DaCe is not guaranteed to be deterministic. This means that if you
write a transformation that depends on the order of nodes in the SDFG, it may not work as expected.
To fix this, you can use the :func:`~dace.sdfg.utils.dfs_topological_sort` function to sort the nodes in a state.

For SDFG state machines, you can also use :func:`~dace.sdfg.analysis.cfg.stateorder_topological_sort`, which will 
traverse the states in the approximate order of execution (i.e., preserving order and entering if/for scopes before 
continuing). 


.. rubric::
    AttributeError

The error manifests as an AttributeError exception and occurs due to trying to access an object's attribute named data
that doesn't exist. For example::
    
    MoveLoopIntoMap::can_be_applied triggered the AttributeError exception: 'NestedSDFG' object has no attribute 'data'

Often, the error is caused by not accounting for certain graph structures. In the example above, the
code assumes an :class:`~dace.sdfg.nodes.AccessNode` would always appear in the context of that graph, but a 
:class:`~dace.sdfg.nodes.NestedSDFG` node can also appear instead, and it does not have an attribute named ``data``.


.. rubric::
    Bad or empty memlet shapes

Memlets in edges have source and destination subsets that can be accessed through :func:`~dace.memlet.Memlet.src_subset`
and :func:`~dace.memlet.Memlet.dst_subset`, respectively. These subsets are represented as :class:`~dace.subsets.Subset`
but could also be ``None`` if the memlet is empty. 
One must take care to check for these cases, as they can happen when map scopes have internal nodes without inputs or
outputs (e.g., when zeroing out an array).

Another source of error is when the memlet source and destination are not consistent with the direction of the edge.
Internally, ``src_subset`` and ``dst_subset`` refer to actual information stored in :attr:`~dace.memlet.Memlet.subset` and
:attr:`~dace.memlet.Memlet.other_subset`. The boolean attribute :attr:`~dace.memlet.Memlet._is_data_src` governs whether the
``src_subset`` points to ``subset``, and it might not be initialized. If the properties 
:attr:`dace.memlet.Memlet.src_subset` and :attr:`dace.memlet.Memlet.dst_subset`
do not return the correct subsets, code trying to find data sources and destinations fails to follow the memlet paths in
the correct direction, ending up in code nodes instead of access nodes. A potential solution is to
try to reconfigure the memlet subsets by calling :func:`dace.memlet.Memlet.try_initialize` with the edge in the state.

