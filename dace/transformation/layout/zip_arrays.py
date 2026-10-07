# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ZipArrays -- the Zip layout primitive: fuses F same-shape SoA arrays into one field-minor array ``Z``. Run after ``prepare_for_layout``; homogeneous fields become a plain ``[*S, F]`` array, heterogeneous fields an array of structs."""

import ast
import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import dace
from dace.frontend.python import astutils
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.transformation import pass_pipeline as ppl


class StructMemberRewriter(ast.NodeTransformer):
    """Rewrites tasklet-connector reference ``conn`` to ``conn.field`` (both reads and writes)."""

    def __init__(self, conn_field: Dict[str, str]):
        self.conn_field = conn_field

    def visit_Name(self, node: ast.Name):
        field = self.conn_field.get(node.id)
        if field is None:
            return node
        return ast.copy_location(
            ast.Attribute(value=ast.Name(id=node.id, ctx=ast.Load()), attr=field, ctx=node.ctx), node
        )


@dataclass
class ZipArrays(ppl.Pass):
    """Fuses groups of same-shape arrays into field-minor AoS arrays."""

    def __init__(self, zip_map: Dict[str, List[str]], field_axis: int = None):
        self._zip_map = zip_map
        self._field_axis = field_axis

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _check_nested(self, sdfg: dace.SDFG, fields: List[str]):
        for state in sdfg.states():
            for n in state.nodes():
                if isinstance(n, nd.NestedSDFG):
                    conns = set(n.in_connectors) | set(n.out_connectors)
                    if any(f in conns for f in fields):
                        raise NotImplementedError("ZipArrays: fields passed to nested SDFGs not supported yet")

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for new_name, fields in self._zip_map.items():
            descs = [sdfg.arrays[f] for f in fields]
            shape0 = tuple(descs[0].shape)
            for f, d in zip(fields, descs):
                if tuple(d.shape) != shape0:
                    raise ValueError(f"ZipArrays: field '{f}' shape {tuple(d.shape)} != {shape0}")
            dtypes = {d.dtype for d in descs}
            if len(dtypes) == 1:
                self._zip_homogeneous(sdfg, new_name, fields, descs, shape0)
            else:
                self._check_nested(sdfg, fields)
                self._zip_struct(sdfg, new_name, fields, descs)

            for f in fields:
                sdfg.remove_data(f, validate=False)
        return 0

    def _zip_homogeneous(self, sdfg, new_name, fields, descs, shape0):
        """Same-dtype fields -> one plain array with the field dim at ``field_axis`` (default trailing)."""
        axis = len(shape0) if self._field_axis is None else self._field_axis
        new_shape = list(shape0)
        new_shape.insert(axis, len(fields))
        new_desc = dace.data.Array(
            descs[0].dtype,
            new_shape,
            storage=descs[0].storage,
            transient=all(d.transient for d in descs),
            lifetime=descs[0].lifetime,
        )
        zip_homogeneous_fields(sdfg, new_name, {f: k for k, f in enumerate(fields)}, axis, new_desc)

    def _zip_struct(self, sdfg, new_name, fields, descs):
        """Different-dtype fields -> one contiguous array of structs (true interleaved AoS)."""
        elem = dace.dtypes.struct(f"{new_name}_t", **{f: descs[i].dtype for i, f in enumerate(fields)})
        transient = all(d.transient for d in descs)
        sdfg.add_array(
            new_name,
            list(descs[0].shape),
            elem,
            storage=descs[0].storage,
            transient=transient,
            lifetime=descs[0].lifetime,
            find_new_name=False,
        )
        field_set = set(fields)
        for state in sdfg.states():
            # 1. Rewrite tasklet code (conn -> conn.field) before the edge rename erases the field data name.
            for node in state.nodes():
                if not isinstance(node, nd.Tasklet):
                    continue
                conn_field = {}
                for e in state.in_edges(node):
                    if e.data is not None and e.data.data in field_set and e.dst_conn:
                        conn_field[e.dst_conn] = e.data.data
                for e in state.out_edges(node):
                    if e.data is not None and e.data.data in field_set and e.src_conn:
                        conn_field[e.src_conn] = e.data.data
                if not conn_field:
                    continue
                # Refuse non-Python tasklets: the rewrite parses code as Python, which could silently mis-handle C++.
                if node.code.language != dace.dtypes.Language.Python:
                    raise NotImplementedError(
                        f"ZipArrays: the heterogeneous-struct path rewrites tasklet code as Python, so the "
                        f"{node.code.language.name} tasklet '{node.label}' is unsupported. Use the "
                        f"struct-free homogeneous path (equal field dtypes), which does not touch tasklet code."
                    )
                tree = ast.parse(node.code.as_string)
                StructMemberRewriter(conn_field).visit(tree)
                ast.fix_missing_locations(tree)
                node.code = dace.properties.CodeBlock(astutils.unparse(tree))
                for c in conn_field:  # the field-fed connectors now carry a whole struct element
                    if c in node.in_connectors:
                        node.in_connectors[c] = elem
                    if c in node.out_connectors:
                        node.out_connectors[c] = elem
            # 2. Point every field memlet/access node at the struct array; map-boundary connectors keep their names.
            for edge in state.edges():
                if edge.data is not None and edge.data.data in field_set:
                    edge.data.data = new_name
            for node in state.nodes():
                if isinstance(node, nd.AccessNode) and node.data in field_set:
                    node.data = new_name


def aosoa_layout(sdfg: dace.SDFG, new_name: str, fields: List[str], vector_width: int) -> None:
    """Applies AoSoA layout (Block + Zip) to same-shape, same-dtype ``fields``: ``[.., N] -> [.., N/V, F, V]``."""
    from dace.transformation.layout.split_dimensions import SplitDimensions

    ndim = len(sdfg.arrays[fields[0]].shape)
    masks = [False] * (ndim - 1) + [True]
    factors = [1] * (ndim - 1) + [vector_width]
    SplitDimensions(split_map={f: (list(masks), list(factors)) for f in fields}).apply_pass(sdfg, {})
    # Blocked fields now have ndim+1 dims ([.., N/V, V]); put the field dim just before the lane.
    ZipArrays(zip_map={new_name: fields}, field_axis=ndim).apply_pass(sdfg, {})


def with_field_index(subset: dace.subsets.Range, axis: int, first: int, last: int) -> dace.subsets.Range:
    """``subset`` with the field dimension ``first:last`` inserted at ``axis``."""
    ranges = list(subset.ranges)
    ranges.insert(axis, (first, last, 1))
    return dace.subsets.Range(ranges)


def zip_homogeneous_fields(
    sdfg: dace.SDFG, new_name: str, field_index: Dict[str, int], axis: int, new_desc: dace.data.Array
) -> None:
    """Fuses the same-dtype containers of ``field_index`` into ``new_name`` (described by ``new_desc``), field
    ``f`` at index ``field_index[f]`` of dimension ``axis``. A nested SDFG fed a field is zipped the same way: its
    connectors are the outer containers (nested SDFG contract), so the field connectors become one connector of
    the fused array, carrying the fields' contiguous index range on the field dimension."""
    sdfg.add_datadesc(new_name, copy.deepcopy(new_desc))

    nested = []
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nd.NestedSDFG):
                continue
            in_edges = [e for e in state.in_edges(node) if e.data.data in field_index and e.dst_conn]
            out_edges = [e for e in state.out_edges(node) if e.data.data in field_index and e.src_conn]
            if not in_edges and not out_edges:
                continue
            inner_index = {e.dst_conn: field_index[e.data.data] for e in in_edges}
            for e in out_edges:
                if inner_index.setdefault(e.src_conn, field_index[e.data.data]) != field_index[e.data.data]:
                    raise NotImplementedError("ZipArrays: a nested connector carrying two fields")
            fields_used = sorted(set(inner_index.values()))
            if fields_used != list(range(fields_used[0], fields_used[-1] + 1)):
                raise NotImplementedError("ZipArrays: a nested SDFG fed a non-contiguous set of fields")
            inner_name = dace.data.find_new_name(new_name, node.sdfg.arrays.keys() - inner_index.keys())
            inner_desc = copy.deepcopy(new_desc)
            inner_desc.transient = False
            zip_homogeneous_fields(node.sdfg, inner_name, inner_index, axis, inner_desc)
            for conn in inner_index:
                node.sdfg.remove_data(conn, validate=False)
            nested.append((state, node, in_edges, out_edges, inner_name, fields_used[0], fields_used[-1]))

    for state in sdfg.states():
        for edge in state.edges():
            memlet = edge.data
            if memlet is None or memlet.is_empty():
                continue
            if memlet.data in field_index:
                k = field_index[memlet.data]
                # Preserve wcr: a reduction into a zipped field keeps accumulating (into Z[.., k]).
                edge.data = dace.memlet.Memlet(
                    data=new_name,
                    subset=with_field_index(memlet.subset, axis, k, k),
                    other_subset=None,
                    wcr=memlet.wcr,
                    wcr_nonatomic=memlet.wcr_nonatomic,
                    dynamic=memlet.dynamic,
                )
            elif memlet.other_subset is not None:
                # A copy named after its other end addresses the field through ``other_subset``.
                other = edge.dst if memlet.data != getattr(edge.dst, "data", None) else edge.src
                if isinstance(other, nd.AccessNode) and other.data in field_index:
                    k = field_index[other.data]
                    memlet.other_subset = with_field_index(memlet.other_subset, axis, k, k)
        # Point field access nodes at the fused array; connector names are left as-is.
        for node in state.nodes():
            if isinstance(node, nd.AccessNode) and node.data in field_index:
                node.data = new_name

    for state, node, in_edges, out_edges, inner_name, first, last in nested:
        for edges, is_input in ((in_edges, True), (out_edges, False)):
            if not edges:
                continue
            kept = edges[0]
            for edge in edges[1:]:
                sdutil.remove_edge_and_dangling_path(state, edge)
            for edge in edges:
                if is_input:
                    node.remove_in_connector(edge.dst_conn)
                else:
                    node.remove_out_connector(edge.src_conn)
            if is_input:
                kept.dst_conn = inner_name
                node.add_in_connector(inner_name, force=True)
            else:
                kept.src_conn = inner_name
                node.add_out_connector(inner_name, force=True)
            for edge in state.memlet_path(kept):
                ranges = list(edge.data.subset.ranges)
                ranges[axis] = (first, last, 1)
                edge.data.subset = dace.subsets.Range(ranges)
