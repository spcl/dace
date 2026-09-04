# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MpiPackUnpack -- make MPI point-to-point buffers contiguous in the current (permuted/blocked) layout.

After a layout change a halo/buffer subset may become strided; MPI would then need a derived datatype
(2-D-capped, raises higher-D). This pass instead inserts a contiguous transient and a pack/unpack map:
pack ``packed[j] = buf[subset]`` before a send, unpack ``buf[subset] = packed[j]`` after a recv (at the
matching Wait for an async Irecv). MPI then always sends a contiguous buffer -- the new-layout wire order,
identical on every rank because all ranks share the global layout. A buffer already contiguous is left
untouched (identity fast-path). A shuffled buffer is refused: Shuffle and MPI are mutually exclusive.

Run AFTER the layout passes (Permute/Block/apply_assignment).
"""
from dataclasses import dataclass
from typing import Any, Dict

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.layout.phases import program_phases


@dataclass
class MpiPackUnpack(ppl.Pass):
    """Pack/unpack non-contiguous MPI send/recv buffers into contiguous transients (see module docstring)."""

    def __init__(self):
        self._counter = 0

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors
                | ppl.Modifies.Memlets)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        from dace.libraries.mpi.nodes import Send, Isend, Recv, Irecv, Sendrecv
        count = 0
        for phase in program_phases(sdfg):
            for state in phase.states():
                for node in list(state.nodes()):
                    if isinstance(node, (Send, Isend)):
                        count += self._pack_send(sdfg, state, node, "_buffer")
                    elif isinstance(node, Sendrecv):
                        count += self._pack_send(sdfg, state, node, "_inbuffer")
                        count += self._unpack_sync(sdfg, state, node, "_outbuffer")
                    elif isinstance(node, Recv):
                        count += self._unpack_sync(sdfg, state, node, "_buffer")
                    elif isinstance(node, Irecv):
                        count += self._unpack_async(sdfg, state, node, "_buffer")
        return count

    # ---- helpers -------------------------------------------------------------
    def _guard_shuffle(self, arr: str) -> None:
        if arr.startswith("shuffled_"):
            raise NotImplementedError(
                f"MpiPackUnpack: buffer '{arr}' is a shuffled array; Shuffle and MPI are mutually "
                f"exclusive (the shuffled memory order makes the send ambiguous).")

    def _contiguous(self, sdfg: dace.SDFG, memlet) -> bool:
        from dace.libraries.mpi.utils import is_access_contiguous
        return is_access_contiguous(memlet, sdfg.arrays[memlet.data])

    def _require_access_node(self, endpoint, node) -> None:
        # The pack/unpack map reuses the buffer's boundary AccessNode. A buffer produced or consumed
        # directly by a scope (a MapExit/MapEntry) means MPI inside a parallel map -- unsupported: pack
        # or unpack into a contiguous buffer OUTSIDE the map (before the entry / after the exit) instead.
        if not isinstance(endpoint, dace.nodes.AccessNode):
            raise NotImplementedError(
                f"MpiPackUnpack: the buffer of MPI node '{node.name}' is directly a "
                f"{type(endpoint).__name__}, not an AccessNode -- an MPI transfer inside a parallel map is "
                f"not supported. Pack/unpack into a contiguous buffer outside the map (before the map entry "
                f"or after the map exit), then send/recv that. Add in-map support only if a real case needs it.")

    def _add_packed(self, sdfg: dace.SDFG, arr: str, subset) -> str:
        desc = sdfg.arrays[arr]
        name = f"packed_{arr}_{self._counter}"
        self._counter += 1
        # SDFG lifetime, not Scope: an async Isend/Irecv buffer must survive from its state to the
        # matching Wait in a later state (a Scope-lifetime buffer is freed at state exit -> use-after-free).
        sdfg.add_array(name,
                       list(subset.size()),
                       desc.dtype,
                       transient=True,
                       storage=desc.storage,
                       lifetime=dace.dtypes.AllocationLifetime.SDFG)
        return name

    def _copy(self, state, arr, subset, packed, pack, arr_node=None, packed_node=None):
        """``pack``: ``packed[j] = arr[min+j]`` (returns the packed WRITE node); else ``arr[min+j] = packed[j]``
        (returns the arr WRITE node). ``arr_node``/``packed_node`` reuse an existing boundary access node."""
        sizes, mins = subset.size(), subset.min_element()
        params = [f"__j{k}" for k in range(len(sizes))]
        sym = dace.symbolic.symbol
        map_ranges = {params[k]: f"0:{sizes[k]}" for k in range(len(sizes))}
        arr_rng = dace.subsets.Range([(mins[k] + sym(params[k]), mins[k] + sym(params[k]), 1)
                                      for k in range(len(sizes))])
        packed_rng = dace.subsets.Range([(sym(p), sym(p), 1) for p in params])
        if pack:
            pw = packed_node or state.add_access(packed)
            state.add_mapped_tasklet(name=f"pack_{packed}",
                                     map_ranges=map_ranges,
                                     inputs={"__inp": dace.Memlet(data=arr, subset=arr_rng)},
                                     code="__out = __inp",
                                     outputs={"__out": dace.Memlet(data=packed, subset=packed_rng)},
                                     input_nodes={arr: arr_node} if arr_node else None,
                                     output_nodes={packed: pw},
                                     external_edges=True)
            return pw
        aw = arr_node or state.add_write(arr)
        state.add_mapped_tasklet(name=f"unpack_{packed}",
                                 map_ranges=map_ranges,
                                 inputs={"__inp": dace.Memlet(data=packed, subset=packed_rng)},
                                 code="__out = __inp",
                                 outputs={"__out": dace.Memlet(data=arr, subset=arr_rng)},
                                 input_nodes={packed: packed_node} if packed_node else None,
                                 output_nodes={arr: aw},
                                 external_edges=True)
        return aw

    def _pack_send(self, sdfg, state, node, conn) -> int:
        edge = next((e for e in state.in_edges(node) if e.dst_conn == conn), None)
        if edge is None or edge.data is None or edge.data.data is None:
            return 0
        arr, subset, src = edge.data.data, edge.data.subset, edge.src
        self._guard_shuffle(arr)
        if self._contiguous(sdfg, edge.data):
            return 0
        self._require_access_node(src, node)
        packed = self._add_packed(sdfg, arr, subset)
        state.remove_edge(edge)  # free the source node, then feed it into the pack instead of the send
        pw = self._copy(state, arr, subset, packed, pack=True, arr_node=src)
        state.add_edge(pw, None, node, conn, dace.Memlet.from_array(packed, sdfg.arrays[packed]))
        return 1

    def _unpack_sync(self, sdfg, state, node, conn) -> int:
        edge = next((e for e in state.out_edges(node) if e.src_conn == conn), None)
        if edge is None or edge.data is None or edge.data.data is None:
            return 0
        arr, subset, dst = edge.data.data, edge.data.subset, edge.dst
        self._guard_shuffle(arr)
        if self._contiguous(sdfg, edge.data):
            return 0
        self._require_access_node(dst, node)
        packed = self._add_packed(sdfg, arr, subset)
        state.remove_edge(edge)
        pnode = state.add_access(packed)
        state.add_edge(node, conn, pnode, None, dace.Memlet.from_array(packed, sdfg.arrays[packed]))
        self._copy(state, arr, subset, packed, pack=False, arr_node=dst, packed_node=pnode)
        return 1

    def _unpack_async(self, sdfg, state, node, conn) -> int:
        edge = next((e for e in state.out_edges(node) if e.src_conn == conn), None)
        if edge is None or edge.data is None or edge.data.data is None:
            return 0
        arr, subset, dst = edge.data.data, edge.data.subset, edge.dst
        self._guard_shuffle(arr)
        if self._contiguous(sdfg, edge.data):
            return 0
        self._require_access_node(dst, node)
        req = self._request_array(state, node)
        wait_state, _ = self._find_wait(sdfg, req)
        packed = self._add_packed(sdfg, arr, subset)
        state.remove_edge(edge)
        pnode = state.add_access(packed)
        state.add_edge(node, conn, pnode, None, dace.Memlet.from_array(packed, sdfg.arrays[packed]))
        if state.degree(dst) == 0:
            state.remove_node(dst)  # the recv no longer writes it here; the unpack writes arr in a later state
        # add_state_after on the Wait's own graph, not the top SDFG -- the Wait can live inside a LoopRegion.
        post = wait_state.parent_graph.add_state_after(wait_state, f"mpi_unpack_{arr}")
        self._copy(post, arr, subset, packed, pack=False)
        return 1

    def _request_array(self, state, node) -> str:
        for e in state.out_edges(node):
            if e.src_conn == "_request" and e.data is not None and e.data.data is not None:
                return e.data.data
        raise NotImplementedError(f"MpiPackUnpack: Irecv '{node.name}' has no _request edge; cannot find its Wait.")

    def _find_wait(self, sdfg, req: str):
        from dace.libraries.mpi.nodes import Wait, Waitall
        for st in sdfg.all_states():
            for n in st.nodes():
                if isinstance(n, (Wait, Waitall)):
                    for e in st.in_edges(n):
                        if e.dst_conn == "_request" and e.data is not None and e.data.data == req:
                            return st, n
        raise NotImplementedError(
            f"MpiPackUnpack: no Wait/Waitall found for Irecv request '{req}'; cannot place the async unpack.")
