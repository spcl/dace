# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization of bulk transfers: sequentialize the re-entered ones, then give them memcpy.

The canonical lowering of a copy / zero is the parallel element map on every device, and the
library node's own ``Auto`` selector may only take that away for a PROVABLY sub-threshold count.
Every other reason to run a transfer on one core is a CPU scheduling decision and belongs here:

1. A transfer an enclosing parallel map or a not-provably-short loop re-enters would fork a team
   per entry -- npbench ``stockham_fft``'s inner ``tmp -> D`` copy is entered 349,525 times per
   call. It is pinned ``Sequential`` (the shared
   :func:`~dace.libraries.standard.helper.is_reentered_cpu_transfer` rule, the same one the map
   cost model uses).
2. A ``Sequential`` transfer that is contiguous, same-shaped and packed in one major order is one
   ``std::memcpy`` (or ``std::memset``), not an element loop. Pinning the implementation here is
   what keeps the libc fast path for small and re-entered transfers without ``memcpy`` being
   anyone's default.

Runs wherever transfers exist or are born: the canonicalize ``cpu_specialize`` band, the
``finalize_for_target`` tail, and -- the load-bearing one -- codegen, right after
:class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` turns implicit
access-node-to-access-node edges into copy library nodes and before they are expanded. Only
host-resident transfers are touched, so a GPU graph passing through the same codegen preamble is
left alone.
"""
from typing import Any, Dict, Optional

from dace import SDFG, data, dtypes, properties, symbolic
from dace.transformation import pass_pipeline as ppl

#: Storages a host ``memcpy`` / ``memset`` may dereference, and the only ones this pass decides.
HOST_STORAGES = frozenset({
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_Pinned,
    dtypes.StorageType.CPU_ThreadLocal,
    dtypes.StorageType.Default,
})

#: Schedules that still mean "this transfer opens its own parallel region".
PARALLEL_SCHEDULES = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)

#: Implementations that still mean "nobody has chosen yet".
UNCHOSEN = (None, 'Auto')


def packed_same_layout(inp: data.Data, out: data.Data) -> bool:
    """Whether both descriptors are packed in the same major order.

    :param inp: source descriptor.
    :param out: destination descriptor.
    :returns: ``True`` for both C-contiguous or both Fortran-contiguous.
    """
    return ((inp.is_packed_c_strides() and out.is_packed_c_strides())
            or (inp.is_packed_fortran_strides() and out.is_packed_fortran_strides()))


def copy_endpoints(node, state):
    """``(src desc, src subset, dst desc, dst subset)`` of a copy library node.

    :param node: the copy library node.
    :param state: the state containing it.
    :returns: the two descriptors and their subsets.
    """
    _inp_name, inp, in_subset, _out_name, out, out_subset = node.validate(state.sdfg, state, allow_cross_storage=True)
    return inp, in_subset, out, out_subset


def copy_is_host(node, state) -> bool:
    """Whether both endpoints of a copy live in host memory.

    :param node: the copy library node.
    :param state: the state containing it.
    :returns: ``True`` if no endpoint is device-resident.
    """
    inp, _in_subset, out, _out_subset = copy_endpoints(node, state)
    return {inp.storage, out.storage} <= HOST_STORAGES


def copy_is_one_memcpy(node, state) -> bool:
    """Whether a copy is expressible as a single host ``std::memcpy``.

    These are the shape / contiguity / layout preconditions
    :func:`~dace.libraries.standard.nodes.copy.select_copy_implementation` tests before it
    picks ``MemcpyCPU`` for a provably sub-threshold copy; the size condition is this band's.

    :param node: the copy library node.
    :param state: the state containing it.
    :returns: ``True`` if ``MemcpyCPU`` applies.
    """
    inp, in_subset, out, out_subset = copy_endpoints(node, state)
    if in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1:
        return False
    if len(inp.shape) != len(out.shape):
        return False
    if any(symbolic.inequal_symbols(a, b) for a, b in zip(in_subset.size(), out_subset.size())):
        return False
    return (in_subset.is_contiguous_subset(inp) and out_subset.is_contiguous_subset(out)
            and packed_same_layout(inp, out))


def memset_is_host(node, state) -> bool:
    """Whether a memset writes host memory.

    :param node: the memset library node.
    :param state: the state containing it.
    :returns: ``True`` if the destination is host-resident.
    """
    _out_name, out, _out_subset = node.validate(state.sdfg, state)
    return out.storage in HOST_STORAGES


def memset_is_one_memset(node, state) -> bool:
    """Whether a zeroing is expressible as a single host ``std::memset``.

    :param node: the memset library node.
    :param state: the state containing it.
    :returns: ``True`` if the ``CPU`` implementation applies.
    """
    _out_name, out, out_subset = node.validate(state.sdfg, state)
    return out_subset.num_elements_exact() != 1 and out_subset.is_contiguous_subset(out)


@properties.make_properties
class SpecializeCpuTransfers(ppl.Pass):
    """Sequentialize re-entered host transfers, then collapse the contiguous ones to libc calls."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Specialize every host copy / memset library node in ``sdfg``.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many library nodes were changed, or ``None`` if none were.
        """
        from dace.libraries.standard.helper import is_reentered_cpu_transfer
        from dace.libraries.standard.nodes.copy import CopyLibraryNode
        from dace.libraries.standard.nodes.fill import FillLibraryNode
        changed = 0
        for node, state in sdfg.all_nodes_recursive():
            is_copy = isinstance(node, CopyLibraryNode)
            if not is_copy and not isinstance(node, FillLibraryNode):
                continue
            if not (copy_is_host(node, state) if is_copy else memset_is_host(node, state)):
                continue
            if node.schedule in PARALLEL_SCHEDULES and is_reentered_cpu_transfer(node, state):
                node.schedule = dtypes.ScheduleType.Sequential
                changed += 1
            if node.schedule != dtypes.ScheduleType.Sequential or node.implementation not in UNCHOSEN:
                continue
            if is_copy and copy_is_one_memcpy(node, state):
                node.implementation = 'MemcpyCPU'
                changed += 1
            elif not is_copy and memset_is_one_memset(node, state):
                node.implementation = 'CPU'
                changed += 1
        return changed or None
