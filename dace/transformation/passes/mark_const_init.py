# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass that detects write-once-then-read-only transients and classifies them so that the experimental
"readable" code generator may emit them as C++ ``const``/``constexpr`` data.
"""

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

# Upper bound on the iteration count of a constant-fill map that is fully unrolled into per-element
# writes (see _unroll_constant_fill_maps). A larger fill is left as a runtime map (not const-inited),
# so only genuinely small compile-time buffers unroll -- a bigger extent never explodes into a giant
# initializer / many tasklets.
CONST_FILL_UNROLL_LIMIT = 16

#: ``{(cfg_id, descriptor name)}`` -- the const-initializable names a constant-fill map must ALL write
#: before flattening it pays for itself. Keyed by cfg_id, not by position in ``all_sdfgs_recursive``:
#: the set is built on a throwaway copy and applied to the real SDFG, so it must not depend on the two
#: being traversed in the same order.
PayingTargets = Set[Tuple[int, str]]


@properties.make_properties
@transformation.explicit_cf_compatible
class MarkConstInit(ppl.Pass):
    """
    Detects transient descriptors that are initialized (once, or collectively by several element-wise constant writes)
    and thereafter only read, and classifies them so the readable code generator can emit them as ``const``/
    ``constexpr``. Two cases:

    * All writers are pure constant producers (assignment tasklets with no data inputs). A small static-extent
      constant-fill map is unrolled to this element-wise pattern first (:meth:`_unroll_constant_fill_maps`). Classified
      ``constexpr_static``: a compile-time initializer is built from the ``{subset -> value}`` map (unwritten elements
      zero-filled), promoted to an SDFG constant, and the now-dead writes are removed. The descriptor stays in
      ``sdfg.arrays`` -- the code generator skips its allocation.
    * A single writer depends on other data. Classified ``const_runtime``; only the ``const_init`` flag is set and the
      dataflow is left untouched.

    All writes must be proven strictly before every read (cross-state reachability, in-state dataflow order). Anything
    unprovable -- overlapping writes, symbolic shapes, non-affine subsets, a multi-write mixing in a runtime value -- is
    left unmarked. The pass is conservative (soundness over coverage) and idempotent.
    """

    CATEGORY: str = 'Optimization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # The pass is idempotent (already-marked descriptors are skipped), so it need not be reapplied.
        return False

    def depends_on(self) -> List[Union[type, ppl.Pass]]:
        # FindAccessNodes gives per-state read/write AccessNode sets; AccessSets gives block-level read/write data sets
        # (and folds interstate-edge reads); StateReachability gives cross-state program-order reachability. All three
        # are nested-SDFG-safe when the results are indexed defensively (``.get(cfg_id, {})``), unlike
        # ScalarWriteShadowScopes which indexes FindAccessNodes by cfg_id directly and can KeyError on nested SDFGs.
        return [FindAccessNodes, AccessSets, StateReachability]

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[int, Dict[str, str]]]:
        """
        :return: A dictionary mapping each SDFG's ``cfg_id`` to a dictionary of ``{descriptor name: classification}``
                 for the descriptors that were marked, or ``None`` if nothing was marked.
        """
        # Step 0: unroll static-extent constant-fill maps into per-element writes, so uniform and
        # index-dependent fills both reduce to the element-wise tasklet pattern the classifier handles.
        # Only the maps that actually PAY OFF are unrolled -- see _paying_fill_targets.
        if self._unroll_constant_fill_maps(top_sdfg, self._paying_fill_targets(top_sdfg)):
            # The unroll changed the graph, so the analyses this pass consumes are stale: recompute
            # them on the mutated SDFG before classifying.
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

        # No self-validate here: constexpr_static removes dead writes, but codegen.py validates the
        # SDFG right after this pass runs, so a re-validate would be redundant.
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
        """True when EVERY name this fill map writes is const-initializable, so flattening it pays.

        Guards the empty-target case explicitly rather than leaning on ``all(...)`` over an empty set
        being vacuously True: a map with nothing to const-init must never be unrolled, whatever
        :meth:`_is_constant_fill_map` is later loosened to admit.
        """
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
        """Runs the classifier over an already-unrolled throwaway copy and returns the
        ``constexpr_static`` names, keyed by ``(cfg_id, name)``.

        ONLY ``constexpr_static`` counts, and the reason is not the obvious one. A fill map has no
        data inputs, so a symbol-valued one (``beta = N * 2.0``) classifies ``runtime`` per element:
        for N>1 that is a runtime multi-write and declined, so unrolling never pays. At N==1 it
        unrolls to a single runtime write, which looks exactly like a ``const_runtime`` this gate
        would then wrongly hold back -- but it cannot happen. ``const_runtime`` needs the write's
        scope to ENCLOSE every read (:meth:`_write_encloses_reads`: each state is its own ``{ }``
        block, so the reads must be in the write's state), while :meth:`_is_constant_fill_map` only
        admits a fill whose consumer is NOT in its state (else MapUnroll drags the consumer into the
        component it replicates). The two requirements are mutually exclusive, so unrolling a fill can
        never produce a ``const_runtime``.
        """
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
        """Decides, on throwaway COPIES, which constant-fill maps are worth unrolling.

        The unroll is SPECULATIVE: it exists only so the classifier sees element-wise writes, but the
        classifier can still decline the target afterwards (a second write, overlapping subsets, a read
        before the write, ...). Unrolling DESTROYS the map, and with it its schedule -- on a
        GPU-transformed SDFG that schedule is the only thing putting the write on the device, so a
        declined target left flattened strands host tasklets writing GPU_Global memory (an invalid
        SDFG). A pass that does not apply must not mutate, so decide here and mutate only where it pays.

        Only ``constexpr_static`` counts, and NOT merely because it is the classification that wants
        the element-wise form: a ``const_runtime`` is unreachable through the unroll by construction.
        See :meth:`_classify_probe` for the argument.

        Iterated to a FIXED POINT, because one probe is not enough. The first probe unrolls every
        candidate, but the real run unrolls only the paying subset -- two different graphs, which can
        disagree. A map writing both a paying name and a declined one is held back, and its MapExit
        then makes its OTHER target a runtime multi-write, declining a name the first probe had
        accepted. Re-probing with the restricted set catches that. It terminates: unrolling fewer maps
        can only decline more names (a held-back map leaves a MapExit writer, which classifies
        ``runtime``), so the set shrinks monotonically and is bounded below by the empty set.

        :return: ``{(cfg_id, descriptor name)}``. Empty when nothing pays, and when there is no
                 candidate map at all -- the common case, which skips the copies entirely.
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
        """Fully unrolls every static-extent constant-fill map (:meth:`_is_constant_fill_map`) into
        per-element tasklet writes via :class:`MapUnroll`, so the classifier sees the element-wise
        ``arr[0]=..; arr[1]=..`` pattern rather than a map producer.

        :param paying: ``{(cfg_id, name)}`` restricting the unroll to the maps whose EVERY target
                       :meth:`_paying_fill_targets` proved const-initializable -- a map writing one
                       paying name and one declined name is held back, since flattening it would not
                       pay for itself. ``None`` unrolls every candidate, and is ONLY for building that
                       set on a throwaway copy: passing it here for the real SDFG is the speculative
                       unroll this gate exists to prevent.
        :return: True if any map was unrolled (the caller then recomputes its stale analyses).
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
        """True for a top-level map whose iterations only WRITE transient arrays from the index and
        constants (no data read in), with a compile-time-constant range of at most
        ``CONST_FILL_UNROLL_LIMIT`` iterations -- the shape it is safe and worthwhile to unroll."""
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
        # MapUnroll replicates the map's whole WEAKLY CONNECTED COMPONENT, not just the map. Anything
        # else in that component is dragged along and duplicated per element -- above all a CONSUMER
        # reading the filled array in this same state, which is what apply_gpu_transformations always
        # produces by fusing the fill's state into its consumer's. Unrolling that duplicates the access
        # node per element while keeping its original out-edge, so every copy claims to deliver the FULL
        # array; the result is a broken graph ("Isolated node", "Dangling in-connector"). If the fill
        # cannot be flattened on its own, it is not a candidate -- say so here rather than leaving a
        # later classifier to decline it by luck.
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
        # Last, let MapUnroll speak for itself rather than restating its rules here (top-level map,
        # constant range, ...). ``apply_to`` passes verify=False, which SKIPS that check, so asking
        # here is what makes its preconditions bind at all.
        #
        # permissive=True is deliberate and narrow. MapUnroll declines a GPU map by default because
        # flattening a device map strands its body on the host, still touching GPU_Global memory --
        # true of the transformation ALONE. Here it never stands alone: a map is only unrolled once
        # :meth:`_paying_fill_targets` has proven its every target reaches ``constexpr_static``, and
        # applying that promotes the fill to a compile-time constant and deletes the writes outright,
        # so no host code survives to touch device memory. Unrolling a device map that does NOT pay
        # would be exactly the bug the gate exists to prevent -- which is why the permission is granted
        # here, behind that gate, and not in the transformation.
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
                # const_runtime: flag the scalar/array so the readable generator fuses its single
                # write into a `const T x = expr;` binding. constexpr_static needs no flag -- it is
                # an SDFG constant, which the allocator skips.
                desc.const_init = True
            marked[name] = kind

        return marked

    def _is_candidate(self, desc: dt.Data) -> bool:
        # A plain transient array/scalar only -- never a view, reference, stream, structure, or
        # container-array (whose element addressing is not the plain flat index).
        return (isinstance(desc, (dt.Scalar, dt.Array)) and desc.transient
                and not isinstance(desc, (dt.View, dt.Reference, dt.Stream, dt.Structure, dt.ContainerArray)))

    def _symbolic_data_refs(self, sdfg: SDFG) -> Set[str]:
        """Names of descriptors read symbolically in an interstate edge or control-flow condition. Such
        reads are live uses invisible to ``FindAccessNodes`` (access-node only), so any descriptor named
        here is left unmarked -- soundness: a symbolic read must never be mistaken for a dead/absent one.
        """
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
                    # ``const_runtime`` fuses the single write into ``const T x = expr;`` at the write
                    # site. Sound only when the producer is a single-assignment tasklet that emits
                    # BRACE-FREE: a copy/library/map-exit producer has no fuseable statement, and a
                    # tasklet with any non-inlined connector is wrapped in ``{ }`` -- which would trap
                    # the binding so reads elsewhere hit an undeclared identifier. Require both.
                    if not (isinstance(edge.src, nd.Tasklet) and self._single_assignment_to(edge.src, edge.src_conn)
                            and tasklet_emits_brace_free(sdfg, state, edge.src)):
                        runtime_fuseable = False
                else:
                    records.append((subset, value))

        if not records:
            return None

        # A runtime value is only handled for a genuine single write (flags-only const binding); a multi-write pattern
        # that mixes in a runtime value is not folded. The binding is fused at the write site, so it is only sound when
        # the producer is a fuseable tasklet AND the write's scope encloses every read (else the const declaration would
        # not be visible at a read, or would come after it).
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
        but not vice versa (strict program-order domination -- a loop back to ``sw`` fails). Same-state:
        the read node is the write node or reachable from it in dataflow."""
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
        """Classifies the value delivered by a single write ``edge`` as constant, runtime, or unknown.

        Only a direct tasklet writer is a const/runtime candidate. A constant-fill MAP is handled up
        front by :meth:`_unroll_constant_fill_maps` (it becomes per-element tasklet writes before this
        runs), so a map exit reaching here is a non-constant producer -- classified runtime, hence
        never const-inited (and never touched by ``_remove_write``)."""
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
            # A non-constant value. With data inputs it is a runtime value; with only
            # symbol inputs (e.g. ``x = ipow(dy, 2)``) a single assignment to out_conn
            # is still a well-defined runtime value -- a valid const-binding target
            # (``const T x = expr;``), so classify it RUNTIME too. Anything else (a
            # multi-statement body whose locals must stay mutable) stays UNKNOWN.
            if data_inputs or self._single_assignment_to(tasklet, out_conn):
                return WRITER_RUNTIME, None
            return WRITER_UNKNOWN, None
        if data_inputs:
            return WRITER_RUNTIME, None
        return WRITER_CONST, value

    def _single_assignment_to(self, tasklet: nd.Tasklet, out_conn: Optional[str]) -> bool:
        """True only if the tasklet body is exactly one ``out_conn = <expr>`` Python
        assignment -- the sole pattern a ``const T x = <expr>;`` binding can fuse.

        A tasklet may hold arbitrary code, so this FAILS CLOSED: a non-Python body, a
        body that is not a parsed statement list (a raw / unparseable string), anything
        other than a single statement, a statement that is not a plain assignment, or a
        target we cannot resolve, all return False -> the scalar keeps its mutable
        declaration. A write that is a bare function call (``func(x)`` mutating ``x``,
        an ``ast.Expr`` not an ``ast.Assign``) is likewise never const."""
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
        """True if the single write's scope encloses every read, so a ``const`` binding fused at the
        write site is visible in-order at each read. Each state is its own ``{ }`` block, so every read
        must be in the write's state and its map scope must be an ancestor of (or equal to) each read's
        scope (``None`` = state top level, which encloses the whole state)."""
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
        # The initializer was fully materialized during classification; here we only promote it and drop the dead
        # runtime writes. Pass a fresh descriptor copy so the constant's stored descriptor is not aliased with the
        # live ``sdfg.arrays[name]`` object.
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
        """Recursively removes a dead producer node (a now-consumerless tasklet, and any access node
        it orphans). Constant-fill maps are unrolled to tasklets before classification, so a removed
        const write is always a tasklet -- no map-scope pruning is needed here."""
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
