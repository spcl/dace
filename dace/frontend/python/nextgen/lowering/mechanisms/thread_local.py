# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Serialization of the map scopes that touch per-thread storage.

A ``CPU_ThreadLocal`` array is a DIFFERENT array in every thread. A parallel
map over one therefore touches only the slice its own thread happened to
iterate -- which is harmless as long as *everything* touching the array is a
parallel map over the same range, since OpenMP hands each thread the same chunk
every time, so each private copy is written and read consistently. That is what
DaCe's own storage-to-schedule default says too
(``dtypes.STORAGEDEFAULT_SCHEDULE`` maps ``CPU_ThreadLocal`` to
``CPU_Multicore``), and what the stable frontend produces.

The consistency breaks as soon as a plain COPY touches the array. A copy is a
memcpy on whichever thread reaches it -- ordinarily the master, outside any
parallel region -- so it sees only the part of the array that thread wrote.
When that happens, every map touching the array has to be single-threaded too,
so that one thread's copy holds the whole value.

Applied to the finished tree rather than at emission, for two reasons: the
copies that force the decision may be lowered long after the map that fills the
array, and a container's storage is not always known when that map is emitted
-- the inline hint (``b = np.ones(N) @ dace.StorageType.CPU_ThreadLocal``)
moves the storage *after* the creation call is lowered, while the annotated
form (``b: dace.float32[N] @ ...``) declares it before.

Nothing is lost where per-thread storage is actually meant to be used -- inside
a parallel scope -- since a map nested in a ``CPU_Multicore`` one is sequential
by default anyway (``dtypes.SCOPEDEFAULT_SCHEDULE``).
"""
from typing import Set

from dace import data, dtypes
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def serialize_thread_local_scopes(root: tn.ScheduleTreeRoot) -> None:
    """
    Set every map scope that touches a per-thread array which a copy also
    touches to :attr:`~dace.dtypes.ScheduleType.Sequential`, in place.

    :param root: The finished schedule tree.
    """
    thread_local = {
        name
        for name, descriptor in root.containers.items()
        if getattr(descriptor, 'storage', None) == dtypes.StorageType.CPU_ThreadLocal
        and not isinstance(descriptor, data.Scalar)
    }
    if not thread_local:
        return
    copied = set()
    for node in root.preorder_traversal():
        if isinstance(node, tn.CopyNode):
            copied |= {node.target, node.memlet.data} & thread_local
    if not copied:
        return
    for node in root.preorder_traversal():
        if isinstance(node, tn.MapScope) and _touches(node, copied, root):
            node.node.map.schedule = dtypes.ScheduleType.Sequential


def _touches(scope: tn.MapScope, containers: Set[str], root: tn.ScheduleTreeRoot) -> bool:
    """Whether any memlet in a scope's subtree refers to one of ``containers``."""
    for memlet in list(scope.input_memlets(root)) + list(scope.output_memlets(root)):
        if memlet.data in containers:
            return True
    return False
