# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Inject explicit copy-in / copy-out states into every body NSDFG.

After :class:`ExpandNestedSDFGInputs` has widened every body NSDFG's
in/out connectors to the full outer array, the body still references
those connectors directly inside its tasklets. This pass lifts each
connector access into a private transient + a single copy state at the
start (input) or end (output) of the body NSDFG, so the inner compute
sees only local transients and the connector boundary becomes one
explicit copy each direction.

Algorithm
---------

For every NSDFG nested inside a Map scope:

* For each ``in_connector``:

  1. Add a transient ``_<conn>_local`` with the same descriptor as the
     connector array.
  2. Rewrite every memlet inside the NSDFG whose ``data == conn`` to
     reference ``_<conn>_local`` instead.
  3. Prepend a new state ``copy_in_<conn>``: ``AN(conn) -> AN(_<conn>_local)``
     with a full-array memlet.

* For each ``out_connector``:

  1. Add a transient ``_<conn>_local`` with the same descriptor as the
     connector array.
  2. Rewrite every memlet inside the NSDFG whose ``data == conn`` to
     reference ``_<conn>_local`` instead.
  3. Append a new state ``copy_out_<conn>``: ``AN(_<conn>_local) -> AN(conn)``
     with a full-array memlet.

* An in/out connector shared by both directions allocates one transient
  shared between the copy-in and copy-out states.

The pass is idempotent: a body NSDFG whose connectors already route
through ``_<conn>_local`` transients is left alone.
"""
import copy
from typing import Dict, Optional

import dace
from dace import properties, subsets
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation import pass_pipeline as ppl


_LOCAL_PREFIX = "_local_"


def _local_name(conn: str) -> str:
    return f"{_LOCAL_PREFIX}{conn}"


def _already_localized(inner: SDFG, conn: str) -> bool:
    """Whether the body has already been rewritten -- the connector array
    is no longer referenced by any inner memlet (every reference points
    at the local transient)."""
    local = _local_name(conn)
    if local not in inner.arrays:
        return False
    for st in inner.all_states():
        for e in st.edges():
            if e.data is not None and e.data.data == conn:
                return False
    return True


def _rewrite_data_references(inner: SDFG, conn: str, local: str) -> None:
    """Rewrite every memlet referencing ``conn`` to reference ``local``
    instead, and rename every AccessNode whose ``data == conn`` to
    ``data == local``. The compute body becomes transient-only."""
    for st in inner.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == conn:
                n.data = local
        for e in st.edges():
            if e.data is not None and e.data.data == conn:
                e.data.data = local


def _add_local_transient(inner: SDFG, conn: str) -> str:
    """Allocate a transient with the same descriptor as the connector
    array. Returns the new array name."""
    local = _local_name(conn)
    if local in inner.arrays:
        return local
    desc = copy.deepcopy(inner.arrays[conn])
    desc.transient = True
    desc.storage = dace.dtypes.StorageType.Register
    inner.add_datadesc(local, desc)
    return local


def _insert_copy_state(inner: SDFG, conn: str, local: str, *, direction: str) -> None:
    """Prepend a copy-in (``direction='in'``) or append a copy-out
    (``direction='out'``) state to the body NSDFG. The copy memlet
    covers the full descriptor extent."""
    assert direction in ("in", "out")
    state_label = f"copy_{direction}_{conn}"
    src_name = conn if direction == "in" else local
    dst_name = local if direction == "in" else conn
    full = subsets.Range.from_array(inner.arrays[conn])
    new_state = inner.add_state(state_label)
    src_an = new_state.add_access(src_name)
    dst_an = new_state.add_access(dst_name)
    new_state.add_edge(src_an, None, dst_an, None, Memlet(data=conn, subset=copy.deepcopy(full)))
    # Splice the new state into the body's CFG.
    if direction == "in":
        old_start = inner.start_block
        if new_state is not old_start:
            # Make the copy-in the new start; the old start node becomes
            # an unconditional successor.
            inner.add_edge(new_state, old_start, dace.InterstateEdge())
            inner.start_block = inner.node_id(new_state)
    else:
        # Append after every sink state.
        sinks = [s for s in inner.nodes() if s is not new_state and inner.out_degree(s) == 0]
        for s in sinks:
            inner.add_edge(s, new_state, dace.InterstateEdge())


@properties.make_properties
class InsertBodyNSDFGCopies(ppl.Pass):
    """Lift every in-map body NSDFG's connector accesses into a private
    transient + one copy state per direction.

    Runs after :class:`ExpandNestedSDFGInputs` (so every connector
    descriptor mirrors the source array) and before the tile descent
    (so the descent sees only local transients in the compute states).
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Inject copy-in / copy-out states for every in-map body NSDFG.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of NSDFGs rewritten, or ``None`` if none.
        """
        rewritten = 0
        for n, g in list(sdfg.all_nodes_recursive()):
            if not isinstance(n, nodes.NestedSDFG):
                continue
            if not isinstance(g, SDFGState):
                continue
            if g.entry_node(n) is None:
                continue  # top-level NSDFG: copy-in/-out lives at the outer level
            inner = n.sdfg
            in_conns = list(n.in_connectors)
            out_conns = list(n.out_connectors)
            seen: Dict[str, str] = {}
            changed = False
            for conn in in_conns:
                if conn not in inner.arrays:
                    continue
                if _already_localized(inner, conn):
                    continue
                local = _add_local_transient(inner, conn)
                _rewrite_data_references(inner, conn, local)
                _insert_copy_state(inner, conn, local, direction="in")
                seen[conn] = local
                changed = True
            for conn in out_conns:
                if conn not in inner.arrays:
                    continue
                if conn in seen:
                    # In-out twin: transient already allocated and inner
                    # memlets already rewritten by the in-pass above.
                    # Only the copy-out state is missing.
                    _insert_copy_state(inner, conn, seen[conn], direction="out")
                    changed = True
                    continue
                if _already_localized(inner, conn):
                    continue
                local = _add_local_transient(inner, conn)
                _rewrite_data_references(inner, conn, local)
                _insert_copy_state(inner, conn, local, direction="out")
                changed = True
            if changed:
                rewritten += 1
        return rewritten or None
