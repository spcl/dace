# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""UnzipArrays -- the Unzip layout primitive: project a fused array back to F separate arrays.

The inverse of :class:`~dace.transformation.layout.zip_arrays.ZipArrays`:

  * **Homogeneous** ``Z`` is a plain array ``[*S, F]`` (field-minor AoS). Unzip drops the field
    dimension: an access ``Z[idx, f]`` (constant field ``f``) becomes ``A_f[idx]`` on a fresh array
    ``A_f`` of shape ``[*S]``.
  * **Heterogeneous struct** ``Z`` is a contiguous array of structs (``Array`` with a
    ``dace.dtypes.struct`` dtype -- the true-AoS form ``ZipArrays`` emits). Unzip splits it back into
    the F member arrays: a whole-struct memlet ``Z[idx]`` becomes ``A_f[idx]``, the tasklet code's
    member access ``conn.f`` is reverted to ``conn``, and the struct-typed connectors are retyped to
    the member dtypes.

When the fused array flows WHOLE into a nested SDFG, the single connector is split into F connectors
(one per field) and the inner fused array is unzipped recursively.

Run after ``prepare_for_layout``. ``ZipArrays(...)`` then ``UnzipArrays(...)`` with matching field
lists is a no-op (bit-exact roundtrip).
"""
import ast
from dataclasses import dataclass
from typing import Any, Dict, List

import dace
from dace.frontend.python import astutils
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl


class StructMemberReverter(ast.NodeTransformer):
    """Inverse of ``zip_arrays.StructMemberRewriter``: rewrite a struct member access ``conn.field``
    back to a bare ``conn`` and record which field each connector accessed (``conn -> field``)."""

    def __init__(self, conns):
        self.conns = conns
        self.conn_field = {}

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.conns:
            self.conn_field[node.value.id] = node.attr
            return ast.copy_location(ast.Name(id=node.value.id, ctx=node.ctx), node)
        return node


@dataclass
class UnzipArrays(ppl.Pass):
    """Project fused arrays back into their component field arrays.

    :param unzip_map: ``{fused_name: [field_array_0, field_array_1, ...]}`` -- the same mapping used
                      to build the fused array with :class:`ZipArrays`.
    :param field_axis: position of the field dimension in a homogeneous fused array's shape. ``None``
                       (default) = trailing dimension; pass ``ndim-2`` to unzip an AoSoA ``[*S, F, V]``.
    """

    def __init__(self, unzip_map: Dict[str, List[str]], field_axis: int = None):
        self._unzip_map = unzip_map
        self._field_axis = field_axis

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for fused, fields in self._unzip_map.items():
            desc = sdfg.arrays[fused]
            if isinstance(desc.dtype, dace.dtypes.struct):
                self._unzip_struct(sdfg, fused, fields, desc)
            else:
                axis = len(desc.shape) - 1 if self._field_axis is None else self._field_axis
                self._unzip_homogeneous(sdfg, fused, fields, desc, axis)
        return 0

    # ------------------------------------------------------------------ #
    #  Homogeneous [*S, F] -> F arrays [*S]
    # ------------------------------------------------------------------ #
    def _const_field(self, rng, fused):
        """Extract the constant field index ``k`` from a memlet range ``(k, k, 1)``."""
        b, e, s = rng
        if str(dace.symbolic.simplify(e - b)) != '0' or str(s) != '1':
            raise NotImplementedError(f"UnzipArrays: non-constant field coordinate on '{fused}' ({rng})")
        return int(b)

    def _node_field(self, state, node, fields):
        """The field index of a fused access node, read from its (already-renamed) incident edges."""
        names = {e.data.data for e in state.all_edges(node) if e.data is not None and e.data.data in fields}
        if len(names) != 1:
            raise NotImplementedError(f"UnzipArrays: access node for '{node.data}' spans fields {names}")
        return fields.index(next(iter(names)))

    def _drop_axis(self, ranges, axis):
        return dace.subsets.Range([r for i, r in enumerate(ranges) if i != axis])

    def _unzip_homogeneous(self, sdfg, fused, fields, desc, axis):
        field_shape = [s for i, s in enumerate(desc.shape) if i != axis]
        for f in fields:
            if f not in sdfg.arrays:
                sdfg.add_array(name=f,
                               shape=field_shape,
                               dtype=desc.dtype,
                               storage=desc.storage,
                               transient=desc.transient,
                               lifetime=desc.lifetime,
                               find_new_name=False)

        # Fused array flowing whole into a nested SDFG: split the connector into F field connectors.
        for boundary in list(self._nested_boundaries(sdfg, fused)):
            self._split_nested(sdfg, boundary, fused, fields, axis)

        # A single fused access node fanning multiple fields through a map: split the map's fused
        # connector into one (IN/OUT) pair per field so each field gets its own reservoir edge.
        for state in sdfg.all_states():
            for scope in [n for n in state.nodes() if isinstance(n, nd.MapEntry)]:
                for conn in [c for c in scope.in_connectors]:
                    out_conn = "OUT_" + conn[len("IN_"):]
                    if self._scope_conn_multifield(state, scope, out_conn, fused, axis, is_entry=True):
                        self._split_map_scope(sdfg, state, scope, conn, out_conn, fused, fields, axis, is_entry=True)
            for scope in [n for n in state.nodes() if isinstance(n, nd.MapExit)]:
                for conn in [c for c in scope.out_connectors]:
                    in_conn = "IN_" + conn[len("OUT_"):]
                    if self._scope_conn_multifield(state, scope, in_conn, fused, axis, is_entry=False):
                        self._split_map_scope(sdfg, state, scope, in_conn, conn, fused, fields, axis, is_entry=False)

        # Remaining top-level edges/nodes: constant-field indexed accesses (incl. the Zip-output
        # topology where each field already has its own access node).
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data == fused:
                    if edge.data.other_subset is not None:
                        raise NotImplementedError("UnzipArrays: other_subset memlets are unsupported.")
                    ranges = list(edge.data.subset.ranges)
                    k = self._const_field(ranges[axis], fused)
                    # Preserve wcr: a reduction into an unzipped field keeps accumulating.
                    edge.data = dace.memlet.Memlet(data=fields[k],
                                                   subset=self._drop_axis(ranges, axis),
                                                   other_subset=None,
                                                   wcr=edge.data.wcr,
                                                   wcr_nonatomic=edge.data.wcr_nonatomic,
                                                   dynamic=edge.data.dynamic)
            for node in list(state.nodes()):
                if isinstance(node, nd.AccessNode) and node.data == fused:
                    node.data = fields[self._node_field(state, node, fields)]

        sdfg.remove_data(fused, validate=False)

    def _scope_conn_multifield(self, state, scope, interior_conn, fused, axis, is_entry):
        """True iff the interior edges of one connector carry >1 distinct fused field (so the
        connector must be split). A connector already carrying a single field is left to the plain
        edge/access-node rename."""
        edges = state.out_edges(scope) if is_entry else state.in_edges(scope)
        attr = "src_conn" if is_entry else "dst_conn"
        ks = set()
        for e in edges:
            if getattr(e, attr) == interior_conn and e.data is not None and e.data.data == fused:
                ks.add(self._const_field(list(e.data.subset.ranges)[axis], fused))
        return len(ks) > 1

    def _split_map_scope(self, sdfg, state, scope, in_conn, out_conn, fused, fields, axis, is_entry):
        """Split a map's fused ``(IN_x, OUT_x)`` connector pair into one pair per field. On a
        ``MapEntry`` the reservoir is the IN side (reads); on a ``MapExit`` it is the OUT side."""
        reservoir_conn = in_conn if is_entry else out_conn
        interior_conn = out_conn if is_entry else in_conn
        reservoir = [e for e in state.in_edges(scope) if e.dst_conn == reservoir_conn] if is_entry else \
                    [e for e in state.out_edges(scope) if e.src_conn == reservoir_conn]
        interior = [e for e in state.out_edges(scope) if e.src_conn == interior_conn] if is_entry else \
                   [e for e in state.in_edges(scope) if e.dst_conn == interior_conn]

        # Reroute each interior edge to a per-field connector, keyed by its constant field.
        for e in interior:
            k = self._const_field(list(e.data.subset.ranges)[axis], fused)
            f = fields[k]
            fconn = ("OUT_" if is_entry else "IN_") + f
            mem = dace.memlet.Memlet(data=f,
                                     subset=self._drop_axis(list(e.data.subset.ranges), axis),
                                     wcr=e.data.wcr,
                                     wcr_nonatomic=e.data.wcr_nonatomic,
                                     dynamic=e.data.dynamic)
            if is_entry:
                if fconn not in scope.out_connectors:
                    scope.add_out_connector(fconn)
                state.add_edge(scope, fconn, e.dst, e.dst_conn, mem)
            else:
                if fconn not in scope.in_connectors:
                    scope.add_in_connector(fconn)
                state.add_edge(e.src, e.src_conn, scope, fconn, mem)
            state.remove_edge(e)

        # One reservoir edge per field, to/from a fresh field access node.
        for f in fields:
            interior_side = ("OUT_" if is_entry else "IN_") + f
            has = interior_side in (scope.out_connectors if is_entry else scope.in_connectors)
            if not has:
                continue  # field not routed through this scope
            fconn = ("IN_" if is_entry else "OUT_") + f
            whole = dace.Memlet.from_array(f, sdfg.arrays[f])
            if is_entry:
                scope.add_in_connector(fconn)
                state.add_edge(state.add_read(f), None, scope, fconn, whole)
            else:
                scope.add_out_connector(fconn)
                state.add_edge(scope, fconn, state.add_write(f), None, whole)

        for e in reservoir:
            other = e.src if is_entry else e.dst
            state.remove_edge(e)
            if state.degree(other) == 0:
                state.remove_node(other)
        scope.remove_in_connector(in_conn)
        scope.remove_out_connector(out_conn)

    # ------------------------------------------------------------------ #
    #  Nested-SDFG boundary: split the fused connector into F field connectors
    # ------------------------------------------------------------------ #
    def _nested_boundaries(self, sdfg, fused):
        """Yield ``(state, nsdfg_node, edge, conn, is_input)`` for each whole-array boundary."""
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    for ie in state.in_edges(node):
                        if ie.data is not None and ie.data.data == fused:
                            yield state, node, ie, ie.dst_conn, True
                    for oe in state.out_edges(node):
                        if oe.data is not None and oe.data.data == fused:
                            yield state, node, oe, oe.src_conn, False

    def _split_nested(self, sdfg, boundary, fused, fields, axis):
        state, node, edge, conn, is_input = boundary
        nsdfg = node.sdfg
        inner_fields = [f"{conn}_{k}" for k in range(len(fields))]

        # Recurse: unzip the inner fused array first (it has the same [*S, F] shape/field axis).
        self._unzip_homogeneous(nsdfg, conn, inner_fields, nsdfg.arrays[conn], axis)

        outer_node = edge.src if is_input else edge.dst
        state.remove_edge(edge)
        if is_input:
            node.remove_in_connector(conn)
        else:
            node.remove_out_connector(conn)

        for k, f in enumerate(fields):
            if is_input:
                node.add_in_connector(inner_fields[k])
                an = state.add_read(f)
                state.add_edge(an, None, node, inner_fields[k], dace.Memlet.from_array(f, sdfg.arrays[f]))
            else:
                node.add_out_connector(inner_fields[k])
                an = state.add_write(f)
                state.add_edge(node, inner_fields[k], an, None, dace.Memlet.from_array(f, sdfg.arrays[f]))

        if state.degree(outer_node) == 0:
            state.remove_node(outer_node)

    # ------------------------------------------------------------------ #
    #  Heterogeneous struct -> F arrays (drop the dotted prefix)
    # ------------------------------------------------------------------ #
    def _interior_field(self, state, scope, interior_conn, fields, is_entry):
        """The field a map connector now carries -- read from its interior edges after they have been
        renamed from the fused struct to a member array (the inverse of the zip's connector reuse)."""
        edges = state.out_edges(scope) if is_entry else state.in_edges(scope)
        for e in edges:
            conn = e.src_conn if is_entry else e.dst_conn
            if conn == interior_conn and e.data is not None and e.data.data in fields:
                return e.data.data
        return None

    def _unzip_struct(self, sdfg, fused, fields, desc):
        """Split the contiguous array-of-structs ``fused`` back into its F member arrays -- the exact
        inverse of ``ZipArrays._zip_struct``. Member dtypes come from the struct's ``fields``; each
        tasklet's ``conn.f`` member access is reverted to ``conn`` and the connector retyped."""
        members = dict(desc.dtype.fields)  # {field_name: typeclass}
        for f in fields:
            if f not in sdfg.arrays:
                sdfg.add_array(f,
                               list(desc.shape),
                               members[f],
                               storage=desc.storage,
                               transient=desc.transient,
                               lifetime=desc.lifetime,
                               find_new_name=False)
        for state in sdfg.all_states():
            # 1. Revert each tasklet's member access, retype the struct connector, and rename its
            #    fused edges to the member array the connector accessed.
            for node in [n for n in state.nodes() if isinstance(n, nd.Tasklet)]:
                conns = {e.dst_conn for e in state.in_edges(node) if e.data is not None and e.data.data == fused}
                conns |= {e.src_conn for e in state.out_edges(node) if e.data is not None and e.data.data == fused}
                conns.discard(None)
                if not conns:
                    continue
                tree = ast.parse(node.code.as_string)
                reverter = StructMemberReverter(conns)
                reverter.visit(tree)
                ast.fix_missing_locations(tree)
                node.code = dace.properties.CodeBlock(astutils.unparse(tree))
                conn_field = reverter.conn_field
                for e in state.in_edges(node):
                    if e.data is not None and e.data.data == fused and e.dst_conn in conn_field:
                        e.data.data = conn_field[e.dst_conn]
                        node.in_connectors[e.dst_conn] = members[conn_field[e.dst_conn]]
                for e in state.out_edges(node):
                    if e.data is not None and e.data.data == fused and e.src_conn in conn_field:
                        e.data.data = conn_field[e.src_conn]
                        node.out_connectors[e.src_conn] = members[conn_field[e.src_conn]]
            # 2. Rename the map-boundary edges: the field is the one its paired interior edge carries.
            #    A boundary edge can only be resolved once its INTERIOR edge is renamed, so with a
            #    multi-level map nest an inner boundary must be resolved before its outer one.
            #    Iterate to a fixpoint so the result is independent of node iteration order.
            changed = True
            while changed:
                changed = False
                for scope in [n for n in state.nodes() if isinstance(n, nd.MapEntry)]:
                    for e in list(state.in_edges(scope)):
                        if e.data is None or e.data.data != fused or not e.dst_conn:
                            continue
                        f = self._interior_field(state, scope, "OUT_" + e.dst_conn[len("IN_"):], fields, is_entry=True)
                        if f is not None:
                            e.data.data = f
                            changed = True
                for scope in [n for n in state.nodes() if isinstance(n, nd.MapExit)]:
                    for e in list(state.out_edges(scope)):
                        if e.data is None or e.data.data != fused or not e.src_conn:
                            continue
                        f = self._interior_field(state, scope, "IN_" + e.src_conn[len("OUT_"):], fields, is_entry=False)
                        if f is not None:
                            e.data.data = f
                            changed = True
            # 3. Point each fused access node at the field its (now-renamed) incident edges carry.
            for node in list(state.nodes()):
                if isinstance(node, nd.AccessNode) and node.data == fused:
                    node.data = fields[self._node_field(state, node, fields)]
        sdfg.remove_data(fused, validate=False)
