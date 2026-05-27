# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stage ``Tasklet1 -> A(global) -> Tasklet2`` hops through fresh transient scalars.

A non-transient (global / argument) array access node ``A`` that bridges a
producer tasklet ``T1`` and a consumer tasklet ``T2`` forces a global-memory
round-trip and a false serialization that blocks vectorization / tiling. This
pass stages that hop through fresh transient scalars so the producer -> consumer
value flow is decoupled from the global store, while the real store to the
global array is always preserved.

The rewrite has two shapes, selected by the disjointness of the producer subset
``s1`` and the consumer subset ``s2`` (see
``STAGE_GLOBAL_THROUGH_SCALARS_SPEC.md`` for the full behavioural contract):

- **Case A** (``s1`` / ``s2`` PROVABLY DISJOINT): the ``T1 -> T2`` dependency
  through ``A`` is false. ``T1`` writes a fresh scalar ``A1`` which is then
  stored to ``global[s1]``; ``T2`` reads ``global[s2]`` through a fresh scalar
  ``A2``.
- **Case B** (NOT provably disjoint -- a real read-modify-write): the value
  flows ``T1 -> A1 -> T2`` through one transient and is ALSO stored to
  ``global[s1]`` via an assignment tasklet.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import dace
from dace import SDFG, subsets
from dace import dtypes
from dace.memlet import Memlet
from dace.transformation import pass_pipeline as ppl, transformation

#: Storage type used for the staged scalars (kept in registers, never spilled).
_STAGED_SCALAR_STORAGE = dtypes.StorageType.Register

#: Body of the Case-B assignment tasklet that re-stores the staged value.
_ASSIGN_CODE = "_out = _in"

#: Input / output connector names of the Case-B assignment tasklet.
_ASSIGN_IN = "_in"
_ASSIGN_OUT = "_out"


@transformation.explicit_cf_compatible
class StageGlobalArrayThroughScalars(ppl.Pass):
    """Stage tasklet -> global-array -> tasklet hops through fresh transient scalars.

    For each non-transient access node bridging a producer tasklet and a
    consumer tasklet, the producer -> consumer value is routed through fresh
    register scalars instead of round-tripping through the global array. The
    real store to the global array is always preserved.
    """

    #: This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        """Report which SDFG components this pass mutates.

        :returns: ``AccessNodes | Memlets | Descriptors`` -- the pass adds
            transient scalar descriptors and access nodes, and rewires memlets.
        """
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Whether the pass should be re-run after a modification.

        :param modified: The components modified by the preceding pass.
        :returns: ``False`` -- this is a single fixpoint pass.
        """
        return False

    def depends_on(self):
        """Passes that must run before this one.

        :returns: An empty set (the pass is standalone).
        """
        return {}

    # ------------------------------------------------------------------
    # Subset helpers
    # ------------------------------------------------------------------
    def _array_side_subset(self, edge: 'dace.sdfg.graph.MultiConnectorEdge', array_name: str):
        """Return the ``array_name``-side subset carried by ``edge``.

        The memlet on an edge touching the global array stores the global
        subset either in ``subset`` (when the memlet's ``data`` is the global
        array) or in ``other_subset`` (when it describes the other endpoint).
        ``Indices`` subsets are normalized to a ``Range``.

        :param edge: An edge incident to the global access node.
        :param array_name: The global array descriptor name.
        :returns: The array-side subset as a ``Range``, or ``None`` if absent.
        """
        memlet = edge.data
        if memlet is None or memlet.is_empty():
            return None
        if memlet.data == array_name:
            sub = memlet.subset
        else:
            sub = memlet.other_subset
        if sub is None:
            return None
        if isinstance(sub, subsets.Indices):
            return subsets.Range.from_indices(sub)
        return sub

    def _is_single_element(self, sub) -> bool:
        """Whether ``sub`` describes exactly one array element.

        :param sub: A normalized subset (or ``None``).
        :returns: ``True`` iff the subset's symbolic volume is exactly ``1``.
        """
        if sub is None:
            return False
        try:
            return sub.num_elements() == 1
        except Exception:
            return False

    def _carries_data_read(self, edge: 'dace.sdfg.graph.MultiConnectorEdge') -> bool:
        """Whether ``edge`` carries an actual (non-empty) data read.

        Empty memlets (pure ordering / dependency edges) do not move data.

        :param edge: The edge to inspect.
        :returns: ``True`` iff the edge transfers data.
        """
        return edge.data is not None and not edge.data.is_empty()

    # ------------------------------------------------------------------
    # Intervening-write guard (refusal condition 3)
    # ------------------------------------------------------------------
    def _writer_reads_array(self, state: 'dace.SDFGState', tasklet, array_name: str) -> bool:
        """Whether ``tasklet`` reads ``array_name`` (i.e. it is an accumulation step).

        A writer that reads the same global array it writes is part of the
        linear accumulation chain (the canonical ``zqlhs`` reuse), not a
        competing independent store.

        :param state: The state owning ``tasklet``.
        :param tasklet: The producer tasklet of a write edge.
        :param array_name: The global array descriptor name.
        :returns: ``True`` iff ``tasklet`` has an in-edge reading ``array_name``.
        """
        for ie in state.in_edges(tasklet):
            if isinstance(ie.src, dace.nodes.AccessNode) and ie.src.data == array_name:
                return True
        return False

    def _node_value_is_consumed(self, state: 'dace.SDFGState', node) -> bool:
        """Whether ``node``'s stored value is read by some tasklet (a live bridge).

        :param state: The state owning ``node``.
        :param node: A global access node.
        :returns: ``True`` iff a non-empty out-edge feeds a tasklet.
        """
        for oe in state.out_edges(node):
            if isinstance(oe.dst, dace.nodes.Tasklet) and self._carries_data_read(oe):
                return True
        return False

    def _intervening_write_violation(self, state: 'dace.SDFGState', bridge, producer, s1) -> bool:
        """Whether a competing, dead, independent write to ``A.data[s1]`` exists.

        Refusal condition 3: the guard fires when another access node of the
        same global array carries an incoming write whose subset is not provably
        disjoint from ``s1`` and that write is neither this occurrence's
        producer nor part of the linear accumulation chain. A write is part of
        the chain (and thus exempt) when its value is consumed by a tasklet (a
        live bridge) or its writer reads the same global array (an accumulation
        step). A *dead* store fed from elsewhere -- as in a dead overwritten
        store sitting beside the read -- trips the guard.

        :param state: The state owning the occurrence.
        :param bridge: The occurrence's global bridge access node.
        :param producer: The occurrence's producer ``T1``.
        :param s1: The array-side subset written by ``T1``.
        :returns: ``True`` iff the occurrence must be refused.
        """
        array_name = bridge.data
        for node in state.data_nodes():
            if node.data != array_name or node is bridge:
                continue
            for we in state.in_edges(node):
                if not self._carries_data_read(we):
                    continue
                if we.src is producer:
                    continue
                sw = self._array_side_subset(we, array_name)
                if subsets.intersects(sw, s1) is False:
                    # Provably disjoint element -- not a competing write.
                    continue
                # The write overlaps s1. It is exempt when it belongs to the
                # linear accumulation chain (its value is consumed downstream,
                # or its writer reads the same global array).
                if self._node_value_is_consumed(state, node):
                    continue
                if (isinstance(we.src, dace.nodes.Tasklet) and self._writer_reads_array(state, we.src, array_name)):
                    continue
                return True
        return False

    # ------------------------------------------------------------------
    # Occurrence collection
    # ------------------------------------------------------------------
    def _collect_occurrences(self, state: 'dace.SDFGState') -> List[Tuple]:
        """Enumerate every stageable ``T1 -> A(global) -> T2`` occurrence in ``state``.

        :param state: The state to scan.
        :returns: A list of ``(bridge, e1, e2, s1, s2, disjoint)`` tuples for
            every occurrence that survives the refusal checks.
        """
        sdfg = state.sdfg
        occurrences: List[Tuple] = []
        for node in state.data_nodes():
            desc = sdfg.arrays.get(node.data)
            # Refusal 1: only global (non-transient) arrays are staged.
            if desc is None or desc.transient:
                continue
            in_tasklet_edges = [
                e for e in state.in_edges(node) if isinstance(e.src, dace.nodes.Tasklet) and self._carries_data_read(e)
            ]
            out_tasklet_edges = [
                e for e in state.out_edges(node) if isinstance(e.dst, dace.nodes.Tasklet) and self._carries_data_read(e)
            ]
            if not in_tasklet_edges or not out_tasklet_edges:
                continue
            for e1 in in_tasklet_edges:
                for e2 in out_tasklet_edges:
                    occ = self._classify_occurrence(state, node, e1, e2)
                    if occ is not None:
                        occurrences.append(occ)
        return occurrences

    def _classify_occurrence(self, state: 'dace.SDFGState', bridge, e1, e2) -> Optional[Tuple]:
        """Validate and classify a single ``(e1, e2)`` producer/consumer pair.

        :param state: The state owning the occurrence.
        :param bridge: The global bridge access node ``A``.
        :param e1: The producer edge ``T1 -> A``.
        :param e2: The consumer edge ``A -> T2``.
        :returns: A ``(bridge, e1, e2, s1, s2, disjoint)`` tuple, or ``None``
            when the occurrence is refused.
        """
        # Refusal 2: producer / consumer must be tasklets, not access nodes.
        if isinstance(e1.src, dace.nodes.AccessNode) or isinstance(e2.dst, dace.nodes.AccessNode):
            return None
        # Refusal 4: wcr (reduction) edges would lose their accumulation.
        if e1.data.wcr is not None or e2.data.wcr is not None:
            return None

        s1 = self._array_side_subset(e1, bridge.data)
        s2 = self._array_side_subset(e2, bridge.data)
        # Refusal 5: only single-element scalar hops are stageable.
        if not self._is_single_element(s1) or not self._is_single_element(s2):
            return None

        # Refusal 3: a competing dead store to the same element must not exist.
        if self._intervening_write_violation(state, bridge, e1.src, s1):
            return None

        disjoint = subsets.intersects(copy.deepcopy(s1), copy.deepcopy(s2)) is False
        return (bridge, e1, e2, s1, s2, disjoint)

    # ------------------------------------------------------------------
    # Rewriting
    # ------------------------------------------------------------------
    def _apply_case_a(self, sdfg: SDFG, state: 'dace.SDFGState', bridge, e1, e2, s1, s2) -> None:
        """Stage a Case-A (disjoint) occurrence through two transient scalars.

        ``T1`` writes a fresh scalar ``A1`` which is then stored to
        ``global[s1]``; ``T2`` reads ``global[s2]`` through a fresh scalar
        ``A2``. ``A`` becomes a pure pass-through node, removing the false
        ``T1 -> T2`` serialization.

        :param sdfg: The SDFG owning ``state``.
        :param state: The state owning the occurrence.
        :param bridge: The global bridge access node ``A``.
        :param e1: The producer edge ``T1 -> A``.
        :param e2: The consumer edge ``A -> T2``.
        :param s1: The array-side subset written by ``T1``.
        :param s2: The array-side subset read by ``T2``.
        """
        dtype = sdfg.arrays[bridge.data].dtype
        a1_name, _ = sdfg.add_scalar(self._scalar_basename(bridge.data, 's1'),
                                     dtype,
                                     storage=_STAGED_SCALAR_STORAGE,
                                     transient=True,
                                     find_new_name=True)
        a2_name, _ = sdfg.add_scalar(self._scalar_basename(bridge.data, 's2'),
                                     dtype,
                                     storage=_STAGED_SCALAR_STORAGE,
                                     transient=True,
                                     find_new_name=True)

        t1 = e1.src
        t2 = e2.dst
        src_conn = e1.src_conn
        dst_conn = e2.dst_conn

        state.remove_edge(e1)
        state.remove_edge(e2)

        # Producer side: T1 -> A1, then the real store A1 --[A[s1]]--> A.
        a1_node = state.add_access(a1_name)
        state.add_edge(t1, src_conn, a1_node, None, Memlet(f"{a1_name}[0]"))
        state.add_edge(a1_node, None, bridge, None,
                       Memlet(data=bridge.data, subset=copy.deepcopy(s1), other_subset=subsets.Range([(0, 0, 1)])))

        # Consumer side: the load A --[A[s2]]--> A2, then A2 -> T2.
        a2_node = state.add_access(a2_name)
        state.add_edge(bridge, None, a2_node, None,
                       Memlet(data=bridge.data, subset=copy.deepcopy(s2), other_subset=subsets.Range([(0, 0, 1)])))
        state.add_edge(a2_node, None, t2, dst_conn, Memlet(f"{a2_name}[0]"))

    def _apply_case_b(self, sdfg: SDFG, state: 'dace.SDFGState', bridge, e1, e2, s1, s2) -> None:
        """Stage a Case-B (RMW) occurrence through one transient scalar.

        The value flows ``T1 -> A1 -> T2`` through one transient and is ALSO
        stored to ``global[s1]`` via an assignment tasklet, so the global array
        still receives the stored value while the false serialization of
        ``T2``'s read behind the global store is removed.

        :param sdfg: The SDFG owning ``state``.
        :param state: The state owning the occurrence.
        :param bridge: The global bridge access node ``A``.
        :param e1: The producer edge ``T1 -> A``.
        :param e2: The consumer edge ``A -> T2``.
        :param s1: The array-side subset written by ``T1`` (the global store).
        :param s2: The array-side subset read by ``T2`` (unused; RMW value).
        """
        dtype = sdfg.arrays[bridge.data].dtype
        a1_name, _ = sdfg.add_scalar(self._scalar_basename(bridge.data, 'rmw'),
                                     dtype,
                                     storage=_STAGED_SCALAR_STORAGE,
                                     transient=True,
                                     find_new_name=True)

        t1 = e1.src
        t2 = e2.dst
        src_conn = e1.src_conn
        dst_conn = e2.dst_conn

        state.remove_edge(e1)
        state.remove_edge(e2)

        # The RMW value flows producer -> consumer directly through one
        # transient: T1 -> A1, A1 -> T2.
        a1_node = state.add_access(a1_name)
        state.add_edge(t1, src_conn, a1_node, None, Memlet(f"{a1_name}[0]"))
        state.add_edge(a1_node, None, t2, dst_conn, Memlet(f"{a1_name}[0]"))

        # An assignment tasklet re-stores the staged value to the global array,
        # reading the same transient A1.
        assign = state.add_tasklet(f"stage_store_{bridge.data}", {_ASSIGN_IN}, {_ASSIGN_OUT}, _ASSIGN_CODE)
        state.add_edge(a1_node, None, assign, _ASSIGN_IN, Memlet(f"{a1_name}[0]"))
        state.add_edge(assign, _ASSIGN_OUT, bridge, None, Memlet(data=bridge.data, subset=copy.deepcopy(s1)))

        # Preserve the write-after-write store order of the original accumulation
        # chain. The severed read edge ``A@k -> T2`` used to sequence this store
        # ahead of every store the consumer (transitively) produces -- in
        # particular the terminal store that decides the global element's final
        # value. Re-add that happens-before as a tasklet -> tasklet dependency
        # edge ``assign -> T2`` (an empty memlet between adjacent tasklets is the
        # only ordering edge that does not re-introduce a global-node read).
        state.add_edge(assign, None, t2, None, Memlet())

    def _scalar_basename(self, array_name: str, tag: str) -> str:
        """Build a descriptive base name for a staged scalar.

        :param array_name: The global array being staged.
        :param tag: A short role tag (``s1`` / ``s2`` / ``rmw``).
        :returns: The base name (``find_new_name`` disambiguates collisions).
        """
        return f"stage_{array_name}_{tag}"

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def _apply(self, sdfg: SDFG) -> int:
        """Stage every occurrence in ``sdfg`` and recurse into nested SDFGs.

        :param sdfg: The SDFG to transform in place.
        :returns: The number of occurrences rewritten in ``sdfg`` and its
            nested SDFGs.
        """
        count = 0
        for state in sdfg.all_states():
            # Collect on the pristine state so guards see the original graph;
            # occurrences touch disjoint edges, so applying them is independent.
            for bridge, e1, e2, s1, s2, disjoint in self._collect_occurrences(state):
                if disjoint:
                    self._apply_case_a(sdfg, state, bridge, e1, e2, s1, s2)
                else:
                    self._apply_case_b(sdfg, state, bridge, e1, e2, s1, s2)
                count += 1

            # Recurse into the directly-nested SDFGs of this state only; the
            # recursive call handles any deeper nesting (avoids double-counting).
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    count += self._apply(node.sdfg)
        return count

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the staging pass over ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes
            (unused; the pass is standalone).
        :returns: The number of ``T1 -> A(global) -> T2`` occurrences rewritten,
            or ``None`` when nothing changed (Pass-pipeline no-op convention).
        """
        count = self._apply(sdfg)
        if count == 0:
            return None
        return count
