import dace
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from typing import Optional, Dict, Set, Tuple
import copy
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.scalar_fission import PrivatizeScalars
import dace.sdfg.utils as sdutil


class MultiStateScalarWriteError(RuntimeError):
    """Raised when a candidate transient scalar has writes in 2+ states."""
    pass


def _scalar_is_safe_to_eliminate(sdfg: dace.SDFG, scalar_name: str,
                                 current_state: dace.SDFGState) -> bool:
    """Return ``True`` if removing the transient :class:`Scalar` is safe."""
    desc = sdfg.arrays.get(scalar_name)
    if not isinstance(desc, dace.data.Scalar):
        return True

    in_state_writes = sum(1 for n in current_state.nodes()
                         if isinstance(n, dace.nodes.AccessNode) and n.data == scalar_name
                         and len({e for e in current_state.in_edges(n) if e.data is not None}) > 0)
    if in_state_writes > 1:
        raise MultiStateScalarWriteError(
            f"CleanRedundantCopiesAndAssignments: transient scalar {scalar_name!r} has "
            f"{in_state_writes} AccessNodes writes in the same state {current_state.label!r}.")

    for st in sdfg.all_states():
        if st is current_state:
            continue
        _reads, writes = st.read_and_write_sets()
        if scalar_name in writes:
            raise MultiStateScalarWriteError(
                f"CleanRedundantCopiesAndAssignments: transient scalar {scalar_name!r} "
                f"is written in state {st.label!r} as well as {current_state.label!r}.")
    return True

def _has_edge(state: dace.SDFGState, src: dace.nodes.Node, dst: dace.nodes.Node):
    for e in state.edges():
        if e.src == src and e.dst == dst:
            return True
    return False

@dace.properties.make_properties
@explicit_cf_compatible
class CleanRedundantCopiesAndAssignments(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _record_replacement(self, replacements: Dict[str, Tuple[str, str]], scalar_name: str, target_array: str,
                            indices_str: str) -> None:
        if scalar_name in replacements and replacements[scalar_name] != (target_array, indices_str):
            raise ValueError(f"Conflict detected in replacement dictionary: key {scalar_name!r} "
                             f"already exists with target {replacements[scalar_name]!r}.")
        replacements[scalar_name] = (target_array, indices_str)

    @staticmethod
    def _is_assign_tasklet(tasklet: dace.nodes.Tasklet) -> bool:
        if not isinstance(tasklet.code, dace.properties.CodeBlock):
            return False
        src = tasklet.code.as_string.strip().rstrip(';').strip()
        if '=' not in src:
            return False
        lhs, rhs = src.split('=', 1)
        return lhs.strip() in tasklet.out_connectors and rhs.strip() in tasklet.in_connectors

    # Pattern 1: tasklet (t1) -> AN -> assign (t2) -> MapExit
    def _clean_pattern_1(self, state: dace.SDFGState, replacements: Dict[str, Tuple[str, str]],
                         state_reads: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]],
                         state_writes: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]]) -> int:
        sdfg: dace.SDFG = state.sdfg
        modifications = 0
        for me in [n for n in state.nodes() if isinstance(n, dace.nodes.MapExit)]:
            for ie2 in list(state.in_edges(me)):
                if ie2.data.subset and ie2.data.subset.num_elements_exact() == 1:
                    t2 = ie2.src
                    if (isinstance(t2, dace.nodes.Tasklet) and state.in_degree(t2) == 1
                            and self._is_assign_tasklet(t2)):
                        ie1 = state.in_edges(t2)[0]
                        if ie1.data.subset and ie1.data.subset.num_elements_exact() == 1:
                            an = ie1.src
                            if (isinstance(an, dace.nodes.AccessNode) and
                                sdfg.arrays[an.data].transient and
                                isinstance(sdfg.arrays[an.data], dace.data.Scalar)):
                                if state.in_degree(an) == 1:
                                    ie0 = state.in_edges(an)[0]
                                    if ie0.data.subset and ie0.data.subset.num_elements_exact() == 1:
                                        t1 = ie0.src
                                        if isinstance(t1, dace.nodes.Tasklet):
                                            if not _scalar_is_safe_to_eliminate(sdfg, an.data, state):
                                                continue
                                            
                                            me_out = list(state.out_edges(me))
                                            if me_out and me_out[0].data is not None:
                                                target = me_out[0].data.data
                                                idx = ','.join(str(r[0]) for r in me_out[0].data.subset)
                                                self._record_replacement(replacements, an.data, target, idx)
                                                state_writes.setdefault((target, idx), set()).add(t1)

                                            state.remove_edge(ie2)
                                            state.remove_edge(ie1)
                                            state.remove_edge(ie0)
                                            state.add_edge(t1, ie0.src_conn, me, ie2.dst_conn, copy.deepcopy(ie2.data))
                                            if state.degree(t2) == 0: state.remove_node(t2)
                                            if state.degree(an) == 0: state.remove_node(an)
                                            modifications += 1
        return modifications

    # Pattern 2: MapEntry -> AN -> tasklet(s)
    def _clean_pattern_2(self, state: dace.SDFGState, replacements: Dict[str, Tuple[str, str]],
                         state_reads: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]],
                         state_writes: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]]) -> int:
        sdfg: dace.SDFG = state.sdfg
        modifications = 0
        for entry in [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]:
            for oe1 in list(state.out_edges(entry)):
                if oe1.data.subset and oe1.data.subset.num_elements_exact() == 1:
                    an = oe1.dst
                    if (isinstance(an, dace.nodes.AccessNode) and
                        sdfg.arrays[an.data].transient and
                        isinstance(sdfg.arrays[an.data], dace.data.Scalar)):
                        if state.in_degree(an) == 1:
                            out_edges = list(state.out_edges(an))
                            if all(
                                isinstance(oe2.dst, dace.nodes.Tasklet) and
                                oe2.data.subset and
                                oe2.data.subset.num_elements_exact() == 1
                                for oe2 in out_edges
                            ):
                                if not _scalar_is_safe_to_eliminate(sdfg, an.data, state):
                                    continue
                                if oe1.data is not None:
                                    target = oe1.data.data
                                    idx = ','.join(str(r[0]) for r in oe1.data.subset)
                                    self._record_replacement(replacements, an.data, target, idx)
                                    
                                    for oe2 in out_edges:
                                        if isinstance(oe2.dst, dace.nodes.Tasklet):
                                            state_reads.setdefault((target, idx), set()).add(oe2.dst)

                                state.remove_edge(oe1)
                                for oe2 in out_edges:
                                    state.remove_edge(oe2)
                                    state.add_edge(entry, oe1.src_conn, oe2.dst, oe2.dst_conn, copy.deepcopy(oe1.data))
                                state.remove_node(an)
                                modifications += 1
                                if state.degree(entry) == 0:
                                    state.remove_node(entry)
        return modifications

    # Pattern 3: AN1 -> AN2 -> NextNode(s)
    def _clean_pattern_3(self, state: dace.SDFGState, replacements: Dict[str, Tuple[str, str]],
                         state_reads: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]],
                         state_writes: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]]) -> int:
        sdfg: dace.SDFG = state.sdfg
        modifications = 0
        for an2 in [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)]:
            if an2 in state.nodes() and sdfg.arrays[an2.data].transient:
                if state.in_degree(an2) == 1:
                    ie = state.in_edges(an2)[0]
                    if ie.data.subset and ie.data.subset.num_elements_exact() == 1:
                        an1 = ie.src
                        if isinstance(an1, dace.nodes.AccessNode):
                            out_edges = list(state.out_edges(an2))
                            if not out_edges or all(oe.data.subset and oe.data.subset.num_elements_exact() == 1 for oe in out_edges):
                                if not _scalar_is_safe_to_eliminate(sdfg, an2.data, state):
                                    continue
                                indices = ','.join(str(r[0]) for r in ie.data.subset)
                                self._record_replacement(replacements, an2.data, an1.data, indices)
                                
                                for oe in out_edges:
                                    if isinstance(oe.dst, dace.nodes.Tasklet):
                                        state_reads.setdefault((an1.data, indices), set()).add(oe.dst)
                                
                                state.remove_edge(ie)
                                for oe in out_edges:
                                    state.remove_edge(oe)
                                    new_memlet = copy.deepcopy(oe.data)
                                    new_memlet.data = an1.data
                                    new_memlet.subset = dace.subsets.Range.from_string(indices)
                                    state.add_edge(an1, ie.src_conn, oe.dst, oe.dst_conn, new_memlet)
                                state.remove_node(an2)
                                if state.degree(an1) == 0:
                                    state.remove_node(an1)
                                modifications += 1
        return modifications

    # Pattern 4: Tasklet (t) -> Transient Scalar -> Array Node
    def _clean_pattern_4(self, state: dace.SDFGState, replacements: Dict[str, Tuple[str, str]],
                         state_reads: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]],
                         state_writes: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]]) -> int:
        sdfg: dace.SDFG = state.sdfg
        modifications = 0
        for an in [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)]:
            if (an in state.nodes() and
                sdfg.arrays[an.data].transient and
                isinstance(sdfg.arrays[an.data], dace.data.Scalar)):
                if state.in_degree(an) == 1 and state.out_degree(an) == 1:
                    ie = state.in_edges(an)[0]
                    oe = state.out_edges(an)[0]
                    if (ie.data.subset and ie.data.subset.num_elements_exact() == 1 and
                        oe.data.subset and oe.data.subset.num_elements_exact() == 1):
                        t = ie.src
                        arr_node = oe.dst
                        if isinstance(t, dace.nodes.Tasklet) and isinstance(arr_node, dace.nodes.AccessNode):
                            if not _scalar_is_safe_to_eliminate(sdfg, an.data, state):
                                continue
                            target = oe.data.data
                            idx = ','.join(str(r[0]) for r in oe.data.subset)
                            self._record_replacement(replacements, an.data, target, idx)
                            
                            state_writes.setdefault((target, idx), set()).add(t)

                            state.remove_edge(ie)
                            state.remove_edge(oe)
                            state.add_edge(t, ie.src_conn, arr_node, oe.dst_conn, copy.deepcopy(oe.data))
                            state.remove_node(an)
                            modifications += 1
        return modifications

    # Pattern 5: Tasklet (t1) -> Transient Scalar -> Tasklet (t2) -> Array Node
    def _clean_pattern_5(self, state: dace.SDFGState, replacements: Dict[str, Tuple[str, str]],
                         state_reads: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]],
                         state_writes: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]]) -> int:
            sdfg: dace.SDFG = state.sdfg
            modifications = 0
            for an in [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)]:
                if (an in state.nodes() and
                    sdfg.arrays[an.data].transient and
                    isinstance(sdfg.arrays[an.data], dace.data.Scalar)):
                    if state.in_degree(an) == 1 and state.out_degree(an) == 1:
                        ie = state.in_edges(an)[0]
                        oe = state.out_edges(an)[0]
                        if (ie.data.subset and ie.data.subset.num_elements_exact() == 1 and
                            oe.data.subset and oe.data.subset.num_elements_exact() == 1):
                            t1 = ie.src
                            t2 = oe.dst
                            if isinstance(t1, dace.nodes.Tasklet) and isinstance(t2, dace.nodes.Tasklet):
                                if not self._is_assign_tasklet(t2):
                                    continue
                                if state.in_degree(t2) == 1 and state.out_degree(t2) == 1:
                                    oe2 = state.out_edges(t2)[0]
                                    arr_node = oe2.dst
                                    if isinstance(arr_node, dace.nodes.AccessNode) and oe2.data.subset and oe2.data.subset.num_elements_exact() == 1:
                                        if not _scalar_is_safe_to_eliminate(sdfg, an.data, state):
                                            continue
                                        target = oe2.data.data
                                        idx = ','.join(str(r[0]) for r in oe2.data.subset)
                                        self._record_replacement(replacements, an.data, target, idx)
                                        
                                        state_writes.setdefault((target, idx), set()).add(t1)

                                        state.remove_edge(ie)
                                        state.remove_edge(oe)
                                        state.remove_edge(oe2)
                                        state.add_edge(t1, ie.src_conn, arr_node, oe2.dst_conn, copy.deepcopy(oe2.data))
                                        state.remove_node(an)
                                        state.remove_node(t2)
                                        modifications += 1
            return modifications

    def _apply_recursive(self, sdfg: dace.SDFG) -> int:
        if not sdfg:
            return 0

        PrivatizeScalars().apply_pass(sdfg, {})
        modifications = 0
        replacements: Dict[str, Tuple[str, str]] = {}

        for state in list(sdfg.all_states()):
            # CRITICAL FIX 1: Capture the topological order BEFORE any modifications
            # are made to the state graph.
            orig_dfs_nodes = list(sdutil.dfs_topological_sort(state))
            topo_order = {node: i for i, node in enumerate(orig_dfs_nodes)}

            state_reads: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]] = {}
            state_writes: Dict[Tuple[str, str], Set[dace.nodes.Tasklet]] = {}

            modifications += self._clean_pattern_1(state, replacements, state_reads, state_writes)
            modifications += self._clean_pattern_2(state, replacements, state_reads, state_writes)
            modifications += self._clean_pattern_3(state, replacements, state_reads, state_writes)
            modifications += self._clean_pattern_4(state, replacements, state_reads, state_writes)
            modifications += self._clean_pattern_5(state, replacements, state_reads, state_writes)

            # --- Dependency Reconstruction Pipeline ---
            all_tracked_slots = set(state_reads.keys()) | set(state_writes.keys())
            if all_tracked_slots:
                for slot in all_tracked_slots:
                    readers = state_reads.get(slot, set())
                    writers = state_writes.get(slot, set())
                    
                    # Filter out any tasklets that might have been purged/fused by patterns
                    active_involved = {t for t in (readers | writers) if t in state.nodes()}
                    if len(active_involved) < 2:
                        continue

                    # Sort tasks chronologically based on their ORIGINAL graph sequence position
                    sorted_tasklets = sorted(active_involved, key=lambda t: topo_order[t])

                    # CRITICAL FIX 2: Reconstruct complete RAW, WAR, and WAW hazard edges
                    current_readers = set()
                    last_writer = None

                    for t in sorted_tasklets:
                        is_curr_writer = t in writers
                        is_curr_reader = t in readers

                        if is_curr_writer:
                            # WAR Hazard: connect all accumulated readers before this writer
                            for r in current_readers:
                                if r != t and not _has_edge(state, r, t):
                                    state.add_edge(r, None, t, None, dace.Memlet())
                            
                            # WAW Hazard: connect the prior writer to this writer
                            if last_writer and last_writer != t and not _has_edge(state, last_writer, t):
                                state.add_edge(last_writer, None, t, None, dace.Memlet())
                            
                            # Clear structural context for subsequent writers
                            current_readers = set()
                            last_writer = t

                        if is_curr_reader:
                            # RAW Hazard: connect the last known writer to this reader
                            if last_writer and last_writer != t and not _has_edge(state, last_writer, t):
                                state.add_edge(last_writer, None, t, None, dace.Memlet())
                            current_readers.add(t)

            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    modifications += self._apply_recursive(node.sdfg)

        if replacements:
            # 1. Update all interstate edge assignments manually
            for edge in sdfg.all_interstate_edges():
                new_assignments = {}
                for k, v in edge.data.assignments.items():
                    v_str = str(v)
                    for scalar, (arr_name, idx_str) in replacements.items():
                        v_str = v_str.replace(scalar, f"{arr_name}[{idx_str}]")
                    new_assignments[k] = v_str
                edge.data.assignments = new_assignments

            # 2. Update all matching memlets and subsets manually across all states
            for state in sdfg.all_states():
                for edge in state.edges():
                    if edge.data and edge.data.data in replacements:
                        target_array, indices_str = replacements[edge.data.data]
                        edge.data.data = target_array
                        edge.data.subset = dace.subsets.Range.from_string(indices_str)
                for node in state.nodes():
                    if isinstance(node, dace.nodes.AccessNode) and node.data in replacements:
                        target_array, indices_str = replacements[node.data]
                        node.data = target_array

        return modifications

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        tmods = 0
        mods = 1
        while mods:
            mods = self._apply_recursive(sdfg)
            tmods += mods
        return tmods if tmods > 0 else None