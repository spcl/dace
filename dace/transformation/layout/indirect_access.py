# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Detect indirect (data-dependent) array accesses for the layout algebra.

A sparse layout is, at bottom, an index array whose *values* subscript another array: ``A[idx[i]]``
(a gather), ``y[idx[i]] += ...`` (a scatter). In DaCe every memlet subset is symbolic, so all access
is ``A[sym]``; what distinguishes an indirect access from an ordinary affine ``A[i]`` is how ``sym``
is defined. An indirect access lowers by SYMBOL PROMOTION: an interstate-edge assignment
``sym := idx[f(loop_var)]`` on the enclosing loop region, and then a data-array memlet ``A[sym]``
whose subset free-symbols include ``sym``. This module classifies such accesses and names the
``(index_array, data_array)`` pair, so the brute-force sweep can offer Shuffle / Permute candidates
for the DATA array (:func:`dace.transformation.layout.brute_force.indirection_candidates`).

The bijective layout support for indirection is exactly Shuffle (reorder the data array, composing
the inverse onto the runtime index) and Permute (choose the layout whose block-transfer cost is
invariant to the index distribution). Sparse *format compression* (CSR/ELL packing away zeros) is
NOT a bijective rearrangement and is out of scope of the algebra.

Scope of the detector: the promoted-symbol form above (``for i in range(N): ... A[col[i]]``), which
is what an indirect access in a sequential loop lowers to after ``simplify``. The alternative
``dace.map`` lowering wraps the gather in an ``add_indirection_subgraph`` nested SDFG (the index is
promoted from a temporary inside that nested SDFG); naming its ``(index, data)`` pair requires
tracing the nested SDFG and is a documented extension -- the data array of that form is still
reachable by passing it to ``indirection_candidates`` directly (the k08 mesh-scatter kernel does
exactly this). The symbol-promotion parse mirrors the write-side scatter recognizer
(``passes/scatter_to_guarded_maps.py``); it is kept self-contained here so the layout detector lives
in the layout package.
"""
import ast
from typing import Dict, List, NamedTuple, Optional

from dace import SDFG
from dace.sdfg.state import LoopRegion


class IndirectAccess(NamedTuple):
    """One indirect access site: ``data_array`` is subscripted by the value of ``index_array``.

    ``kind`` is ``'gather'`` for a read (``A[idx[i]]``) or ``'scatter'`` for a write
    (``y[idx[i]] += ...``). The layout lever is on ``data_array`` (Shuffle / Permute); the index
    array is named for provenance and to gate whether a Shuffle candidate is worth generating.
    """
    index_array: str
    data_array: str
    kind: str


def resolve_index_source(rhs: str, loop_var: str, sdfg: SDFG) -> Optional[str]:
    """Return ``arr`` if ``rhs`` is ``arr[f(loop_var)]`` -- ``arr`` a data descriptor in ``sdfg`` and
    the index an expression referencing ``loop_var`` (the bare ``arr[i]`` or a strided
    ``arr[c*i + d]``); ``None`` otherwise. This is the RHS of the promoted indirect-index symbol."""
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
    """``{sym: index_array}`` for every interstate assignment ``sym := index_array[f(loop_var)]`` on
    ``region``'s edges -- the promoted indirect-index symbols of the loop."""
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
    """Every indirect access site in ``sdfg`` (symbol-promotion form): a data-array memlet whose
    subset references a promoted index symbol ``sym := idx[f(loop_var)]``. A read out-edge is a
    ``'gather'``, a write in-edge a ``'scatter'``. Duplicates are removed, order preserved.

    The returned ``(index_array, data_array)`` pairs feed
    :func:`dace.transformation.layout.brute_force.indirection_candidates`, which lays out the data
    array by Shuffle / Permute. See the module docstring for the ``dace.map`` nested-SDFG form the
    detector does not yet name (its data array is passed to ``indirection_candidates`` directly).
    """
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
