# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map of row scans becomes ONE segmented scan before the device offload.

``for i: for j in 1..N: b[i, j] = b[i, j-1] + a[i, j]`` canonicalizes to a map over the rows whose body
stages a row, scans it with a ``Scan`` and applies it. A device-wide scan is a call only host code can
issue, so the offload keeps that map on the host: 13482 rows of ``safety_map_of_scans`` were 13482 trips
of three launches and one ``DeviceScan`` each (296 ms on an MI300 against 23 ms on 16 CPU cores).

The rows are independent, so the body is fissioned into three maps over the rows -- the staging, the
scans and the apply -- with the staged rows expanded to ``[rows, length]``, and the middle map is replaced
by one ``Scan`` with ``segments = rows``. The staging and apply maps collapse into two-dimensional
kernels, and the scans run as one batched device call.
"""
from typing import Any, Dict, List, Optional

from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes.scan import (INIT_CONNECTOR_NAME, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, Scan,
                                                ScanOp, segmented)
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapCollapse, MapFission


def batchable_scan(scan: Scan) -> bool:
    """A scan the segmented lowering takes: the shape ``Scan.segments`` admits."""
    return (not segmented(scan) and scan.chains == 1 and not scan.exclusive and scan.op is not ScanOp.AFFINE
            and INIT_CONNECTOR_NAME not in scan.in_connectors and symbolic.equal_valued(1, scan.stride))


def row_of(memlet: Memlet, sdfg: SDFG, entry: nodes.MapEntry) -> bool:
    """``memlet`` reads or writes whole row ``i - begin`` of a C-contiguous ``[rows, length]`` array, where ``i``
    walks ``begin:begin + rows`` -- the shape :class:`MapFission` gives a staged row it expands."""
    desc = sdfg.arrays[memlet.data]
    begin, end, step = entry.map.range[0]
    if len(desc.shape) != 2 or step != 1 or desc.strides[1] != 1 or symbolic.equal(desc.strides[0],
                                                                                   desc.shape[1]) is not True:
        return False
    if symbolic.equal(desc.shape[0], end - begin + 1) is not True:
        return False
    index = symbolic.pystr_to_symbolic(entry.map.params[0]) - begin
    first, last, _ = memlet.subset[0]
    return (symbolic.equal(first, index) is True and symbolic.equal(last, index) is True
            and memlet.subset[1] == (0, desc.shape[1] - 1, 1))


def replace_row_map(state: SDFGState, entry: nodes.MapEntry, scan: Scan) -> bool:
    """Replace ``map i { scan(row i) }`` by one scan over every row; ``False`` if the map is not that."""
    exit_node = state.exit_node(entry)
    if state.scope_children()[entry] != [scan, exit_node] or len(entry.map.params) != 1:
        return False
    into, out_of = state.in_edges(scan), state.out_edges(scan)
    if len(into) != 1 or len(out_of) != 1:
        return False
    if not row_of(into[0].data, state.sdfg, entry) or not row_of(out_of[0].data, state.sdfg, entry):
        return False
    source = state.memlet_path(into[0])[0]
    target = state.memlet_path(out_of[0])[-1]
    if not isinstance(source.src, nodes.AccessNode) or not isinstance(target.dst, nodes.AccessNode):
        return False
    rows = entry.map.range.num_elements()
    state.remove_nodes_from([entry, exit_node])
    whole = {data: subsets.Range.from_array(state.sdfg.arrays[data]) for data in (source.data.data, target.data.data)}
    state.add_edge(source.src, source.src_conn, scan, INPUT_CONNECTOR_NAME,
                   Memlet(data=source.data.data, subset=whole[source.data.data]))
    state.add_edge(scan, OUTPUT_CONNECTOR_NAME, target.dst, target.dst_conn,
                   Memlet(data=target.data.data, subset=whole[target.data.data]))
    scan.segments = rows
    return True


@properties.make_properties
class BatchRowScans(ppl.Pass):
    """Fission every host map around a row ``Scan`` and batch the scans into one segmented ``Scan``."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Batch the row scans of every top-level map in ``sdfg``.

        :param sdfg: the SDFG to transform, in place, before the device offload.
        :param _pipeline_results: unused.
        :returns: how many maps of scans became one segmented scan, or ``None`` if none did.
        """
        batched = 0
        for state in list(sdfg.states()):
            for entry, scan in self.row_scan_maps(state):
                if not MapFission.can_be_applied_to(state.sdfg, map_entry=entry):
                    continue
                before = set(state.scope_children()[None])
                MapFission.apply_to(state.sdfg, map_entry=entry, save=False)
                scan_entry = state.entry_node(scan)
                if scan_entry is None or not replace_row_map(state, scan_entry, scan):
                    continue
                batched += 1
                # The staging and apply maps are now ``map i { map j }``: one kernel each, not a host loop of launches.
                children = state.scope_children()
                for outer in set(children[None]) - before:
                    inner = [node for node in children.get(outer, ()) if isinstance(node, nodes.MapEntry)]
                    if len(inner) == 1 and MapCollapse.can_be_applied_to(
                            state.sdfg, outer_map_entry=outer, inner_map_entry=inner[0]):
                        MapCollapse.apply_to(state.sdfg, outer_map_entry=outer, inner_map_entry=inner[0], save=False)
        return batched or None

    @classmethod
    def row_scan_maps(cls, state: SDFGState) -> List[tuple]:
        """``(map entry, scan)`` for every top-level one-dimensional map holding a batchable scan."""
        found = []
        children = state.scope_children()
        for entry in children[None]:
            if not isinstance(entry, nodes.MapEntry) or len(entry.map.params) != 1:
                continue
            if entry.map.schedule not in (dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential,
                                          dtypes.ScheduleType.CPU_Multicore):
                continue
            scans = [node for node in children[entry] if isinstance(node, Scan)]
            if len(scans) == 1 and batchable_scan(scans[0]):
                found.append((entry, scans[0]))
        return found
