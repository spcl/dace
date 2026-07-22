# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Detects write-once-then-read-only transients for const/constexpr emission by the readable code generator."""

import ast
import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from dace import SDFG, SDFGState, data as dt, dtypes, properties, subsets, symbolic
from dace.frontend.python import astutils
from dace.sdfg import nodes as nd, utils as sdutil
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.passes.analysis.analysis import AccessSets, FindAccessNodes, StateReachability
from dace.transformation.passes.inline_tasklet_connectors import tasklet_emits_brace_free

# Result of classifying the value produced by a single writer edge of a descriptor.
WRITER_CONST = 'const'  #: A compile-time constant value (numeric, no data inputs).
WRITER_RUNTIME = 'runtime'  #: A value that depends on other data at runtime.
WRITER_UNKNOWN = 'unknown'  #: Cannot prove either of the above -> leave unmarked.

# Max iteration count for a constant-fill map to be unrolled into per-element writes (see
# _unroll_constant_fill_maps); larger fills stay runtime maps, never const-inited.
CONST_FILL_UNROLL_LIMIT = 16

#: ``{(cfg_id, name)}`` -- const-initializable targets a constant-fill map must ALL write before
#: unrolling pays off. Keyed by cfg_id since the set is built on a throwaway copy, not by traversal order.
PayingTargets = Set[Tuple[int, str]]


@properties.make_properties
@transformation.explicit_cf_compatible
class MarkConstInit(ppl.Pass):
    """Classifies write-once-then-read-only transients for constexpr/const emission: all-constant writers
    (optionally via unrolling a small constant-fill map) become ``constexpr_static``; a single runtime
    writer becomes ``const_runtime``. Conservative and idempotent -- unprovable cases are left unmarked."""

    CATEGORY: str = 'Optimization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # The pass is idempotent (already-marked descriptors are skipped), so it need not be reapplied.
        return False

    def depends_on(self) -> List[Union[type, ppl.Pass]]:
        # FindAccessNodes/AccessSets/StateReachability; all three are nested-SDFG-safe only when indexed
        # defensively (``.get(cfg_id, {})``) -- unlike ScalarWriteShadowScopes, which KeyErrors on nested SDFGs.
        return [FindAccessNodes, AccessSets, StateReachability]

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[int, Dict[str, str]]]:
        """:return: ``{cfg_id: {descriptor name: classification}}`` for marked descriptors, or ``None`` if none."""
        # Unroll static-extent constant-fill maps into per-element writes (only those that pay off --
        # see _paying_fill_targets) so the classifier sees a uniform element-wise tasklet pattern.
        if self._unroll_constant_fill_maps(top_sdfg, self._paying_fill_targets(top_sdfg)):
            # Graph changed -- recompute the stale analyses before classifying.
            pipeline_results = dict(pipeline_results)
            pipeline_results[FindAccessNodes.__name__] = FindAccessNodes().apply_pass(top_sdfg, {})
            pipeline_results[AccessSets.__name__] = AccessSets().apply_pass(top_sdfg, {})
            pipeline_results[StateReachability.__name__] = StateReachability().apply_pass(top_sdfg, {})

        access_nodes_all = pipeline_results[FindAccessNodes.__name__]
        access_sets = pipeline_results[AccessSets.__name__]
        state_reach_all = pipeline_results[StateReachability.__name__]

        result: Dict[int, Dict[str, str]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            # Defensive ``.get`` (never ``[cfg_id]``): a missing/aliased cfg_id yields an empty mapping instead of a
            # KeyError on nested SDFGs.
            access_nodes = access_nodes_all.get(sdfg.cfg_id, {})
            state_reach = state_reach_all.get(sdfg.cfg_id, {})
            marked = self._process_sdfg(sdfg, access_nodes, access_sets, state_reach)
            if marked:
                result[sdfg.cfg_id] = marked

        # No self-validate: codegen.py validates right after this pass runs.
        return result or None

    def report(self, pass_retval: Optional[Dict[int, Dict[str, str]]]) -> Optional[str]:
        if not pass_retval:
            return None
        count = sum(len(v) for v in pass_retval.values())
        return f'MarkConstInit marked {count} descriptor(s) as const-initializable.'

    # ---------------------------------------------------------------------------------------------------------------

    def _fill_map_targets(self, sdfg: SDFG, state: SDFGState, map_entry: nd.MapEntry) -> Set[str]:
        """The descriptor names a constant-fill map writes. Never empty for a map
        :meth:`_is_constant_fill_map` accepts -- that predicate requires a data out-edge."""
        map_exit = state.exit_node(map_entry)
        return {e.data.data for e in state.out_edges(map_exit) if e.data is not None and e.data.data is not None}

    def _fill_map_pays(self, sdfg: SDFG, state: SDFGState, map_entry: nd.MapEntry, paying: PayingTargets) -> bool:
        """True when EVERY name this map writes is const-initializable, so flattening it pays off.
        Explicitly guards the empty-target case -- ``all()`` over empty is vacuously True."""
        names = self._fill_map_targets(sdfg, state, map_entry)
        return bool(names) and all((sdfg.cfg_id, name) in paying for name in names)

    def _candidate_fill_maps(self, top_sdfg: SDFG) -> List[Tuple[SDFG, SDFGState, nd.MapEntry]]:
        """Every top-level constant-fill map in ``top_sdfg``, computed once."""
        found: List[Tuple[SDFG, SDFGState, nd.MapEntry]] = []
        for sdfg in list(top_sdfg.all_sdfgs_recursive()):
            for state in list(sdfg.states()):
                scope = state.scope_dict()
                found.extend((sdfg, state, node) for node in state.nodes() if isinstance(node, nd.MapEntry)
                             and scope[node] is None and self._is_constant_fill_map(sdfg, state, node))
        return found

    def _classify_probe(self, probe: SDFG) -> PayingTargets:
        """Classify an already-unrolled throwaway copy; return the ``constexpr_static`` names keyed by
        ``(cfg_id, name)``. Only ``constexpr_static`` counts -- a fill can never unroll to a
        ``const_runtime`` (that needs the write's state to enclose the reads, which the unroll
        precondition forbids)."""
        access_nodes_all = FindAccessNodes().apply_pass(probe, {})
        access_sets = AccessSets().apply_pass(probe, {})
        state_reach_all = StateReachability().apply_pass(probe, {})

        paying: PayingTargets = set()
        # Materialize before iterating: _process_sdfg mutates the probe as it goes.
        for sdfg in list(probe.all_sdfgs_recursive()):
            marked = self._process_sdfg(sdfg, access_nodes_all.get(sdfg.cfg_id, {}), access_sets,
                                        state_reach_all.get(sdfg.cfg_id, {}))
            for name, kind in (marked or {}).items():
                if kind == 'constexpr_static':
                    paying.add((sdfg.cfg_id, name))
        return paying

    def _paying_fill_targets(self, top_sdfg: SDFG) -> PayingTargets:
        """Decides on throwaway COPIES which constant-fill maps are worth unrolling (the unroll is
        speculative and destructive, so a pass that doesn't apply must not mutate the real SDFG).
        Iterates to a fixed point since one probe's paying set can invalidate another map's target.

        :return: ``{(cfg_id, name)}``; empty when nothing pays or there is no candidate map.
        """
        candidates = self._candidate_fill_maps(top_sdfg)
        if not candidates:
            return set()

        paying: Optional[PayingTargets] = None  # first round: unroll every candidate
        while True:
            probe = copy.deepcopy(top_sdfg)
            if not self._unroll_constant_fill_maps(probe, paying):
                return set()
            found = self._classify_probe(probe)
            if found == paying:
                return found  # fixed point: this set unrolls exactly the maps it was derived from
            paying = found
            # Restricting to ``found`` still unrolls every candidate, so the next probe would rebuild
            # the graph we just classified. Skip it -- this is the common case (no partly-paying map).
            if all(self._fill_map_pays(sdfg, state, entry, found) for sdfg, state, entry in candidates):
                return found

    def _unroll_constant_fill_maps(self, top_sdfg: SDFG, paying: Optional[PayingTargets]) -> bool:
        """Unrolls every static-extent constant-fill map (:meth:`_is_constant_fill_map`) into per-element
        tasklet writes via :class:`MapUnroll`, so the classifier sees element-wise writes.

        :param paying: ``{(cfg_id, name)}`` restricting the unroll to maps whose every target is proven
                       const-initializable. ``None`` unrolls everything -- only safe on a throwaway copy.
        :return: True if any map was unrolled.
        """
        changed = False
        for sdfg in list(top_sdfg.all_sdfgs_recursive()):
            for state in list(sdfg.states()):
                scope = state.scope_dict()
                targets = [
                    node for node in state.nodes()
                    if isinstance(node, nd.MapEntry) and scope[node] is None and self._is_constant_fill_map(
                        sdfg, state, node) and (paying is None or self._fill_map_pays(sdfg, state, node, paying))
                ]
                for map_entry in targets:
                    # Each target is a distinct top-level map in its own scope, so unrolling one
                    # never removes another -- no re-presence check is needed here.
                    MapUnroll.apply_to(sdfg, verify=False, save=False, annotate=False, map_entry=map_entry)
                    changed = True
        return changed

    def _is_constant_fill_map(self, sdfg: SDFG, state: SDFGState, map_entry: nd.MapEntry) -> bool:
        """True for a top-level map that only writes transients from index/constants (no data read),
        with a compile-time-constant range of at most ``CONST_FILL_UNROLL_LIMIT`` iterations."""
        # No data flows into the scope: every value written comes from the index / constants.
        if any(not e.data.is_empty() for e in state.in_edges(map_entry)):
            return False
        map_exit = state.exit_node(map_entry)
        out_edges = [e for e in state.out_edges(map_exit) if e.data is not None and e.data.data is not None]
        if not out_edges:
            return False
        # Only fills of plain transient arrays (never program I/O, streams, views, references, ...).
        for e in out_edges:
            desc = sdfg.arrays.get(e.data.data)
            if not isinstance(desc, (dt.Scalar, dt.Array)) or isinstance(desc, (dt.View, dt.Reference)):
                return False
            if not desc.transient:
                return False
        # A nested-SDFG / library-node body is not a simple fill -- leave it alone.
        body = state.scope_subgraph(map_entry, include_entry=False, include_exit=False).nodes()
        if any(isinstance(n, (nd.NestedSDFG, nd.LibraryNode)) for n in body):
            return False
        # MapUnroll replicates the map's whole weakly connected component. If anything but the fill is
        # in it (e.g. a same-state consumer, as apply_gpu_transformations produces), unrolling
        # duplicates that node and breaks the graph -- so the fill is not a candidate.
        own = set(state.scope_subgraph(map_entry, include_entry=True, include_exit=True).nodes())
        own |= {e.dst for e in out_edges}
        if set(sdutil.weakly_connected_component(state, map_entry).nodes()) - own:
            return False
        # Constant, bounded iteration count.
        count = 1
        for begin, end, step in map_entry.map.range:
            try:
                lo = int(symbolic.evaluate(begin, sdfg.constants))
                hi = int(symbolic.evaluate(end, sdfg.constants))
                st = int(symbolic.evaluate(step, sdfg.constants))
            except (TypeError, ValueError):
                return False
            if st == 0:
                return False
            count *= max(0, (hi - lo) // st + 1)
            if count == 0 or count > CONST_FILL_UNROLL_LIMIT:
                return False
        # Defer the rest to MapUnroll (apply_to uses verify=False, so this is what binds its
        # preconditions). permissive=True because a device map is only unrolled once every target is
        # proven constexpr_static, whose application deletes the writes -- no host code survives.
        return MapUnroll.can_be_applied_to(sdfg, permissive=True, map_entry=map_entry)

    def _process_sdfg(self, sdfg: SDFG, access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode],
                                                                                      Set[nd.AccessNode]]]],
                      access_sets: Dict[Any, Tuple[Set[str], Set[str]]],
                      state_reach: Dict[SDFGState, Set[SDFGState]]) -> Dict[str, str]:
        marked: Dict[str, str] = {}
        symbolic_refs = self._symbolic_data_refs(sdfg)

        # Block-level write states per descriptor, restricted to states this SDFG owns (nested-SDFG-safe -- no cfg_id
        # indexing). Interstate edges only add to read sets, so a name in a state's write set is a real array write.
        write_states: Dict[str, Set[SDFGState]] = defaultdict(set)
        for block, (_, wset) in access_sets.items():
            if isinstance(block, SDFGState) and block.sdfg is sdfg:
                for wname in wset:
                    write_states[wname].add(block)

        for name, desc in list(sdfg.arrays.items()):
            if not self._is_candidate(desc):
                continue
            # Idempotency: never reclassify an already-marked descriptor (const_runtime flag set, or
            # already promoted to a constexpr_static constant).
            if desc.const_init or name in sdfg.constants_prop:
                continue

            classification = self._classify(sdfg, name, desc, access_nodes.get(name, {}), write_states.get(name, set()),
                                            state_reach, symbolic_refs)
            if classification is None:
                continue
            kind, info = classification

            if kind == 'constexpr_static':
                if not self._apply_constexpr_static(sdfg, name, desc, info):
                    continue
            else:
                # const_runtime: flags the scalar so the readable generator fuses its single write into
                # `const T x = expr;`. constexpr_static needs no flag -- it becomes an SDFG constant.
                desc.const_init = True
            marked[name] = kind

        return marked

    def _is_candidate(self, desc: dt.Data) -> bool:
        # A plain transient array/scalar only -- never a view, reference, stream, structure, or
        # container-array (whose element addressing is not the plain flat index).
        # Scope lifetime only: a Persistent/Global/External descriptor lives in the state struct and
        # may hold a different value on each invocation (persistent) or one supplied from outside, so
        # folding its write to a compile-time constant is unsound -- and would leave the descriptor
        # claiming state-struct allocation while its value became a bare constant, which the readable
        # generator's `__state->` access can no longer resolve.
        return (isinstance(desc, (dt.Scalar, dt.Array)) and desc.transient
                and desc.lifetime == dtypes.AllocationLifetime.Scope
                and not isinstance(desc, (dt.View, dt.Reference, dt.Stream, dt.Structure, dt.ContainerArray)))

    def _symbolic_data_refs(self, sdfg: SDFG) -> Set[str]:
        """Names of descriptors read symbolically in an interstate edge or control-flow condition --
        such reads are invisible to ``FindAccessNodes``, so these names are always left unmarked."""
        anames = set(sdfg.arrays.keys())
        refs: Set[str] = set()
        for edge in sdfg.all_interstate_edges():
            refs |= ((edge.data.free_symbols | edge.data.read_symbols()) & anames)
        for cfr in sdfg.all_control_flow_regions():
            refs |= (cfr.used_symbols(all_symbols=True, with_contents=False) & anames)
        return refs

    def _classify(self, sdfg: SDFG, name: str, desc: dt.Data, acc_per_state: Dict[SDFGState, Tuple[Set[nd.AccessNode],
                                                                                                   Set[nd.AccessNode]]],
                  block_write_states: Set[SDFGState], state_reach: Dict[SDFGState, Set[SDFGState]],
                  symbolic_refs: Set[str]) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
        if name in symbolic_refs:
            return None

        # Restrict FindAccessNodes entries to states this SDFG actually owns (guards cfg_id aliasing across cloned
        # nested SDFGs -- a foreign state is simply ignored rather than raising).
        per_state = {state: rw for state, rw in acc_per_state.items() if state.sdfg is sdfg}
        write_sites = [(state, node) for state, (_, writes) in per_state.items() for node in writes]
        read_sites = [(state, node) for state, (reads, _) in per_state.items() for node in reads]

        # Must be written and read somewhere (an unread transient is dead, not const).
        if not write_sites or not read_sites:
            return None

        # Cross-check the write states against AccessSets (block-level, nested-safe): the set of states that write the
        # descriptor must match what FindAccessNodes reports. Any disagreement -> bail conservatively.
        if block_write_states != {state for (state, _) in write_sites}:
            return None

        # (4) All writes strictly before all reads (program order across states, dataflow order within a state).
        if not self._all_writes_before_reads(write_sites, read_sites, state_reach):
            return None

        # (1,2) Collect every write as an edge-level record: (subset, value) with the value classified. A record with a
        # runtime value carries ``value = None``.
        records: List[Tuple[subsets.Subset, Any]] = []
        any_runtime = False
        runtime_fuseable = True
        for state, write_node in write_sites:
            for edge in state.in_edges(write_node):
                if edge.data.is_empty():
                    continue
                subset = self._edge_subset(edge, name)
                if subset is None:
                    return None
                kind, value = self._edge_producer_value(state, edge, name)
                if kind == WRITER_UNKNOWN:
                    return None
                if kind == WRITER_RUNTIME:
                    any_runtime = True
                    records.append((subset, None))
                    # const_runtime fuses the write into ``const T x = expr;`` -- sound only for a
                    # single-assignment, BRACE-FREE tasklet (braces would scope the binding away from other reads).
                    if not (isinstance(edge.src, nd.Tasklet) and self._single_assignment_to(edge.src, edge.src_conn)
                            and tasklet_emits_brace_free(sdfg, state, edge.src)):
                        runtime_fuseable = False
                else:
                    records.append((subset, value))

        if not records:
            return None

        # A runtime value is only handled for a genuine single write; a multi-write mixing in a runtime value is
        # not folded. Sound only when the write is a fuseable tasklet AND its scope encloses every read.
        if any_runtime:
            if len(records) == 1 and runtime_fuseable and self._write_encloses_reads(write_sites[0], read_sites):
                return ('const_runtime', None)
            return None

        # (5) All writes are compile-time constants. Build the initializer.
        if isinstance(desc, dt.Scalar):
            # A scalar has a single element; more than one write to it would necessarily conflict.
            if len(records) != 1:
                return None
            return ('constexpr_static', {'scalar_value': records[0][1]})

        # A concrete shape is required to materialize the constexpr initializer.
        shape = self._concrete_shape(desc)
        if shape is None:
            return None
        array = np.zeros(shape, dtype=desc.dtype.as_numpy_dtype())
        touched = np.zeros(shape, dtype=bool)
        for subset, value in records:
            index = self._subset_to_index(subset, shape)
            if index is None:  # symbolic / non-affine subset
                return None
            if touched[index].any():  # (3) overlapping / conflicting writes
                return None
            touched[index] = True
            array[index] = value
        return ('constexpr_static', {'array_value': array})

    def _all_writes_before_reads(self, write_sites: List[Tuple[SDFGState, nd.AccessNode]],
                                 read_sites: List[Tuple[SDFGState, nd.AccessNode]],
                                 state_reach: Dict[SDFGState, Set[SDFGState]]) -> bool:
        """True if every write is strictly before every read. Cross-state: ``sr`` reachable from ``sw``
        but not vice versa. Same-state: the read is the write node or reachable from it in dataflow."""
        bfs_cache: Dict[nd.AccessNode, Set[nd.AccessNode]] = {}
        for sr, read_node in read_sites:
            for sw, write_node in write_sites:
                if sw is sr:
                    if read_node is write_node:
                        continue
                    if write_node not in bfs_cache:
                        bfs_cache[write_node] = set(sw.bfs_nodes(write_node))
                    if read_node not in bfs_cache[write_node]:
                        return False
                else:
                    if sr not in state_reach.get(sw, set()):
                        return False
                    if sw in state_reach.get(sr, set()):
                        return False
        return True

    def _edge_producer_value(self, state: SDFGState, edge: Any, name: str) -> Tuple[str, Any]:
        """Classifies the value a single write ``edge`` delivers as constant, runtime, or unknown. Only a
        direct tasklet writer is a const/runtime candidate -- a map exit here is always classified runtime."""
        src = edge.src
        if isinstance(src, nd.Tasklet):
            return self._tasklet_value(state, src, edge.src_conn)
        # e.g. a copy from another access node, or a (non-unrolled) map exit -> value from other data.
        return WRITER_RUNTIME, None

    def _tasklet_value(self, state: SDFGState, tasklet: nd.Tasklet, out_conn: Optional[str]) -> Tuple[str, Any]:
        """Classifies the value a tasklet assigns to ``out_conn`` as constant, runtime, or unknown."""
        if tasklet.code.language != dtypes.Language.Python:
            return WRITER_UNKNOWN, None
        data_inputs = [ie for ie in state.in_edges(tasklet) if not ie.data.is_empty()]
        value = self._constant_output_value(tasklet, out_conn)
        if value is None:
            # Non-constant: with data inputs it's runtime; with only symbol inputs (e.g. ``x = ipow(dy, 2)``)
            # a single assignment is still a valid const-binding target, so classify it RUNTIME too.
            if data_inputs or self._single_assignment_to(tasklet, out_conn):
                return WRITER_RUNTIME, None
            return WRITER_UNKNOWN, None
        if data_inputs:
            return WRITER_RUNTIME, None
        return WRITER_CONST, value

    def _single_assignment_to(self, tasklet: nd.Tasklet, out_conn: Optional[str]) -> bool:
        """True only if the tasklet body is exactly one ``out_conn = <expr>`` Python assignment -- the
        sole pattern a ``const T x = <expr>;`` binding can fuse. Fails closed on anything else."""
        if out_conn is None or tasklet.code.language != dtypes.Language.Python:
            return False
        stmts = tasklet.code.code
        if not isinstance(stmts, list) or len(stmts) != 1 or not isinstance(stmts[0], ast.Assign):
            return False
        try:
            targets = [astutils.rname(t) for t in stmts[0].targets]
        except Exception:  # noqa: BLE001 -- an unusual target AST -> not fuseable
            return False
        return out_conn in targets

    def _write_encloses_reads(self, write_site: Tuple[SDFGState, nd.AccessNode],
                              read_sites: List[Tuple[SDFGState, nd.AccessNode]]) -> bool:
        """True if the single write's scope encloses every read, so a fused ``const`` binding is visible
        in-order at each read. Every read must be in the write's state, under an ancestor map scope."""
        state, write_node = write_site
        scope = state.scope_dict()
        write_scope = scope.get(write_node)
        for read_state, read_node in read_sites:
            if read_state is not state:
                return False
            cur = scope.get(read_node)
            enclosed = write_scope is None
            while cur is not None and not enclosed:
                if cur is write_scope:
                    enclosed = True
                cur = scope.get(cur)
            if not enclosed:
                return False
        return True

    def _constant_output_value(self, tasklet: nd.Tasklet, out_conn: Optional[str]) -> Optional[Any]:
        """Returns the compile-time value assigned to ``out_conn`` by the tasklet, or ``None`` if not a constant."""
        if out_conn is None:
            return None
        stmts = tasklet.code.code
        if not isinstance(stmts, list):
            return None
        for stmt in stmts:
            if not isinstance(stmt, ast.Assign):
                continue
            targets = [astutils.rname(t) for t in stmt.targets]
            if out_conn not in targets:
                continue
            try:
                value = astutils.evalnode(stmt.value, {})
            except SyntaxError:
                return None
            if isinstance(value, (bool, int, float, complex)):
                return value
            return None
        return None

    def _edge_subset(self, edge: Any, name: str) -> Optional[subsets.Subset]:
        """Returns the subset of ``name`` written by a single ``edge`` into an access node, or ``None``."""
        memlet = edge.data
        if memlet.data == name and memlet.subset is not None:
            return memlet.subset
        return memlet.dst_subset if memlet.dst_subset is not None else memlet.subset

    def _apply_constexpr_static(self, sdfg: SDFG, name: str, desc: dt.Data, info: Dict[str, Any]) -> bool:
        # Initializer was already materialized during classification; here we just promote it and drop the
        # dead writes. Uses a fresh descriptor copy so it isn't aliased with the live sdfg.arrays[name] object.
        const_desc = copy.deepcopy(desc)
        if 'scalar_value' in info:
            sdfg.add_constant(name, desc.dtype.type(info['scalar_value']), const_desc)
        else:
            sdfg.add_constant(name, info['array_value'], const_desc)
        self._remove_write(sdfg, name)
        return True

    def _concrete_shape(self, desc: dt.Data) -> Optional[Tuple[int, ...]]:
        shape: List[int] = []
        for dim in desc.shape:
            try:
                shape.append(int(dim))
            except (TypeError, ValueError):
                return None
        return tuple(shape)

    def _subset_to_index(self, subset: subsets.Subset, shape: Tuple[int, ...]) -> Optional[Tuple[slice, ...]]:
        if not isinstance(subset, subsets.Range) or len(subset.ranges) != len(shape):
            return None
        index: List[slice] = []
        for start, stop, step in subset.ranges:
            try:
                lo, hi, st = int(start), int(stop), int(step)
            except (TypeError, ValueError):
                return None
            index.append(slice(lo, hi + 1, st))
        return tuple(index)

    def _remove_write(self, sdfg: SDFG, name: str) -> None:
        """Removes the (now dead) runtime write of ``name`` and any producer/access nodes it orphans."""
        for state in sdfg.states():
            for node in list(state.data_nodes()):
                if node.data != name or state.in_degree(node) == 0:
                    continue
                srcs: List[Tuple[nd.Node, Optional[str]]] = []
                for edge in list(state.in_edges(node)):
                    srcs.append((edge.src, edge.src_conn))
                    state.remove_edge(edge)
                for src, conn in srcs:
                    if src not in state.nodes():
                        continue
                    if isinstance(src, nd.Tasklet) and conn is not None:
                        if not any(oe.src_conn == conn for oe in state.out_edges(src)):
                            src.remove_out_connector(conn)
                    self._prune_dead(state, src)
                if node in state.nodes() and (state.in_degree(node) + state.out_degree(node)) == 0:
                    state.remove_node(node)

    def _prune_dead(self, state: SDFGState, node: nd.Node) -> None:
        """Recursively removes a dead producer node (a consumerless tasklet, and any access node it
        orphans). Constant-fill maps are unrolled to tasklets first, so no map-scope pruning is needed."""
        if node not in state.nodes():
            return
        if isinstance(node, nd.Tasklet):
            if state.out_degree(node) != 0:
                return
            external_srcs = [ie.src for ie in state.in_edges(node)]
            state.remove_node(node)
            for src in external_srcs:
                self._prune_dead(state, src)
        elif isinstance(node, nd.AccessNode):
            if (state.in_degree(node) + state.out_degree(node)) == 0:
                state.remove_node(node)
