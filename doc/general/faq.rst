.. _faq:

Frequently Asked Questions
==========================

How do I apply a transformation to a specific node or set of nodes?
-------------------------------------------------------------------

Use the ``apply_to`` class method of the transformation. Each transformation
class declares a fixed set of pattern variables (e.g., ``MapFusion`` exposes
``first_map_exit``, ``array``, and ``second_map_entry``); pass the matched
nodes as keyword arguments:

.. code-block:: python

    from dace.transformation.dataflow import MapFusion
    MapFusion.apply_to(sdfg, first_map_exit=exit1, array=array_node,
                       second_map_entry=entry2)

For pattern-based application across the whole SDFG, use
:meth:`~dace.sdfg.sdfg.SDFG.apply_transformations` and
:meth:`~dace.sdfg.sdfg.SDFG.apply_transformations_repeated`. For
whole-program optimizations, run a :ref:`pass <pass>` instead.

When should I use a symbol and when should I use a scalar?
----------------------------------------------------------

See the discussion in :ref:`symbolic-when` for the general guidelines.


How do I perform dynamic memory allocation?
-------------------------------------------

Array shapes in DaCe are always associated with symbolic expressions.
When the size is known only at runtime, declare the array with a symbolic
shape and assign the actual size to that symbol on a state's
*inter-state edge* before the array is used:

.. code-block:: python

    sdfg = dace.SDFG('dyn')
    N = dace.symbol('N')
    sdfg.add_array('A', (N,), dace.float32, transient=True)

    init = sdfg.add_state('init')
    body = sdfg.add_state('body')
    sdfg.add_edge(init, body, dace.InterstateEdge(assignments={'N': 'compute_size()'}))

When the size depends on data within the SDFG, place the allocation inside
a nested SDFG so that its symbol can be assigned from a scalar of the
parent.
