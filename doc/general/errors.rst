.. _errors:

Common Development Errors
=========================

``AttributeError: no attribute 'data'``

The error manifests as an AttributeError exception and occurs due to trying to access an object's attribute named data
that doesn't exist. For example:

WARNING: MoveLoopIntoMap::can_be_applied triggered a AttributeError exception: 'NestedSDFG' object has no attribute 'data'

In DaCe, the attribute data can mostly be found in edges and access nodes. Therefore, the error occurs when attempting
to access the memlet from an object that is not an edge, or the data container's name from an object that is not an
access node. The latter is much more common and usually happens due to misconfigured edges. Memlets in edges have source
and destinations subsets that can be accessed through :func:`dace.memlet.Memlet.src_subset` and :func:`dace.memlet.Memlet.dst_subset`.
However, these subsets are an interface above the actual information stored in :attr:`dace.memlet.Memlet.subset`, :attr:`dace.memlet.Memlet.other_subset`,
and :attr:`Memlet._is_data_src`. If the properties :func:`dace.memlet.Memlet.rc_subset` and :func:`dace.memlet.Memlet.dst_subset`
do not return the correct subsets, code trying to find data sources and destinations fails to follow the memlet paths in
the correct direction, ending up in code nodes instead of access nodes. If this is the case, a potential solution is to
try to reconfigure the memlet subsets by calling :func:`dace.memlet.Memlet.try_initialize`.
