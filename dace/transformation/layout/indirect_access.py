# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Detect indirect (data-dependent) array accesses for the layout algebra: names the ``(index_array, data_array)`` pair behind each ``A[idx[i]]`` gather/scatter, via symbol promotion (``sym := idx[f(loop_var)]`` on the loop region, then a data memlet ``A[sym]``)."""
import ast
from typing import Dict, List, NamedTuple, Optional

from dace import SDFG
from dace.sdfg.state import LoopRegion


class IndirectAccess(NamedTuple):
    """One indirect access site: ``data_array`` subscripted by ``index_array``'s value; ``kind`` is ``'gather'`` (read) or ``'scatter'`` (write)."""
    index_array: str
    data_array: str
    kind: str


def resolve_index_source(rhs: str, loop_var: str, sdfg: SDFG) -> Optional[str]:
    """Return ``arr`` if ``rhs`` is ``arr[f(loop_var)]`` with ``arr`` a descriptor in ``sdfg``, else ``None``."""
    try:
        tree = ast.parse(str(rhs), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(tree, ast.Subscript) or not isinstance(tree.value, ast.Name):
        return None
    arr = tree.value.id
    if arr not in sdfg.arrays:
        return None
    idx = tree.slice
    if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST (<3.9)
        idx = idx.value
    idx_names = {n.id for n in ast.walk(idx) if isinstance(n, ast.Name)}
    if loop_var not in idx_names:
        return None
    return arr


def index_bindings(region: LoopRegion, sdfg: SDFG) -> Dict[str, str]:
    """``{sym: index_array}`` for every interstate assignment ``sym := index_array[f(loop_var)]`` on ``region``'s edges."""
    loop_var = region.loop_variable
    out: Dict[str, str] = {}
    if not loop_var:
        return out
    for e in region.edges():
        if e.data is None or not e.data.assignments:
            continue
        for lhs, rhs in e.data.assignments.items():
            arr = resolve_index_source(rhs, loop_var, sdfg)
            if arr is not None:
                out[lhs] = arr
    return out


def indirect_accesses(sdfg: SDFG) -> List[IndirectAccess]:
    """Every indirect access site in ``sdfg`` (symbol-promotion form); duplicates removed, order preserved."""
    found: List[IndirectAccess] = []
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            bindings = index_bindings(region, sd)
            if not bindings:
                continue
            for state in region.all_states():
                for node in state.data_nodes():
                    if node.data not in sd.arrays:
                        continue
                    edges = ([(e, 'scatter') for e in state.in_edges(node)] +
                             [(e, 'gather') for e in state.out_edges(node)])
                    for edge, kind in edges:
                        if edge.data is None or edge.data.subset is None:
                            continue
                        for sym in edge.data.subset.free_symbols:
                            idx = bindings.get(str(sym))
                            if idx is not None and idx != node.data:
                                found.append(IndirectAccess(idx, node.data, kind))
    seen = set()
    unique: List[IndirectAccess] = []
    for access in found:
        if access not in seen:
            seen.add(access)
            unique.append(access)
    return unique
