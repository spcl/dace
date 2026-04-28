# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift ``GPU_Shared`` transients out of nested SDFGs into the SDFG that owns
the enclosing GPU_Device map. The framecode allocation walker can then route
the ``__shared__`` declaration into the kernel function body without any
codegen-side fix-ups.

Adds a dependency edge from the kernel ``MapEntry`` to the lifted
``AccessNode`` so the allocation walker pins the array to the kernel scope.
A symmetric dependency edge to ``MapExit`` ensures it stays live through
the kernel.
"""
import copy as _copy
from typing import Any, Dict, List, Tuple

from dace import SDFG, SDFGState, dtypes, properties, nodes
from dace.memlet import Memlet
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class LiftSharedOutOfNestedSDFG(ppl.Pass):
    """Promote every ``GPU_Shared`` transient living inside a nested SDFG that
    sits inside a ``GPU_Device`` map up into the SDFG that owns the kernel
    map. Wires the transient through the NSDFG via an out-connector and adds
    dependency edges to the kernel map's entry/exit so the framecode walker
    pins allocation to the kernel scope."""

    def depends_on(self):
        return set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        # Earlier pipeline passes leave scope_dict caches stale across the
        # SDFG. Invalidate them so downstream lookups (incl. the framecode
        # walker) see correct nesting after we lift.
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                state._clear_scopedict_cache()

        lifted = 0
        worklist: List[Tuple[SDFG, SDFGState, nodes.NestedSDFG, nodes.MapEntry]] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for n in state.nodes():
                    if not isinstance(n, nodes.NestedSDFG):
                        continue
                    kernel_entry = self._enclosing_gpu_device_map(state, n)
                    if kernel_entry is None:
                        continue
                    worklist.append((nsdfg, state, n, kernel_entry))

        for outer_sdfg, outer_state, nsdfg_node, kernel_entry in worklist:
            inner_sdfg: SDFG = nsdfg_node.sdfg
            shared_names = [
                name for name, desc in inner_sdfg.arrays.items()
                if desc.transient and desc.storage == dtypes.StorageType.GPU_Shared
            ]
            for name in shared_names:
                self._lift_one(name, inner_sdfg, nsdfg_node, outer_sdfg, outer_state, kernel_entry)
                lifted += 1

        return {'lifted': lifted} if lifted > 0 else None

    def _enclosing_gpu_device_map(self, state: SDFGState, node: nodes.Node):
        """Return the innermost ``MapEntry`` with ``GPU_Device`` schedule that
        contains ``node`` (by scope membership), or None. Invalidates
        ``scope_dict`` first since upstream pipeline passes can leave it
        stale. Walking data-flow predecessors instead would misclassify
        downstream consumers of a kernel as inside it."""
        state._clear_scopedict_cache()
        sdict = state.scope_dict()
        scope = sdict.get(node)
        while scope is not None:
            if (isinstance(scope, nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.GPU_Device):
                return scope
            scope = sdict.get(scope)
        return None

    def _lift_one(self, name: str, inner_sdfg: SDFG, nsdfg_node: nodes.NestedSDFG, outer_sdfg: SDFG,
                  outer_state: SDFGState, kernel_entry: nodes.MapEntry):
        """Promote ``name`` from ``inner_sdfg`` to ``outer_sdfg`` and wire it
        through ``nsdfg_node``. Topology:

            MapEntry --(empty, dep)--> AN_read --(in:local_gather)--> NSDFG
            NSDFG --(out:local_gather)--> AN_write --(empty, dep)--> MapExit

        The dep edge from MapEntry pins allocation to the kernel scope.
        Two ``AccessNode``s for the same outer descriptor keep the state
        acyclic when the inner mutates the array (DaCe's validator rejects a
        single-AN read+write cycle around an NSDFG)."""
        inner_desc = inner_sdfg.arrays[name]

        # Pick an outer name that doesn't collide.
        outer_name = name
        if outer_name in outer_sdfg.arrays:
            i = 0
            while f'{outer_name}_{i}' in outer_sdfg.arrays:
                i += 1
            outer_name = f'{outer_name}_{i}'

        # Move the descriptor: outer keeps it transient (the actual storage),
        # inner gets a non-transient duplicate (now a connector parameter).
        outer_sdfg.add_datadesc(outer_name, inner_desc, find_new_name=False)
        inner_param_desc = _copy.deepcopy(inner_desc)
        inner_param_desc.transient = False
        del inner_sdfg.arrays[name]
        inner_sdfg.add_datadesc(name, inner_param_desc)

        is_read, is_written = self._classify_usage(inner_sdfg, name)
        full_subset = Range.from_array(inner_desc)
        kernel_exit = outer_state.exit_node(kernel_entry)

        # Read side: MapEntry --(dep)--> AN_read --(in)--> NSDFG.
        if is_read:
            an_read = outer_state.add_access(outer_name)
            outer_state.add_edge(kernel_entry, None, an_read, None, Memlet())
            nsdfg_node.add_in_connector(name, force=True)
            outer_state.add_edge(an_read, None, nsdfg_node, name,
                                 Memlet(data=outer_name, subset=_copy.deepcopy(full_subset)))

        # Write side: NSDFG --(out)--> AN_write --(dep)--> MapExit. ``force=True``
        # is required when the inner both reads and writes the array — the
        # NSDFG node carries the same name in both ``in_connectors`` and
        # ``out_connectors`` (standard DaCe pattern for inout arrays).
        if is_written:
            an_write = outer_state.add_access(outer_name)
            nsdfg_node.add_out_connector(name, force=True)
            outer_state.add_edge(nsdfg_node, name, an_write, None,
                                 Memlet(data=outer_name, subset=_copy.deepcopy(full_subset)))
            outer_state.add_edge(an_write, None, kernel_exit, None, Memlet())

        # If the inner only writes (no read side), still anchor allocation by
        # adding the dep edge from MapEntry to the (write) AccessNode so the
        # framecode walker pins it to the kernel scope.
        if is_written and not is_read:
            outer_state.add_edge(kernel_entry, None, an_write, None, Memlet())

        outer_state._clear_scopedict_cache()

    def _classify_usage(self, inner_sdfg: SDFG, name: str) -> Tuple[bool, bool]:
        """Return ``(is_read, is_written)`` for ``name`` inside ``inner_sdfg``
        by consulting each state's ``read_and_write_sets``. State-level
        read/write sets are the authoritative source DaCe uses elsewhere
        (validation, codegen arg inference) — keeps this pass aligned with
        how the rest of the framework decides connector direction."""
        is_read = False
        is_written = False
        for state in inner_sdfg.states():
            read_set, write_set = state.read_and_write_sets()
            if name in read_set:
                is_read = True
            if name in write_set:
                is_written = True
            if is_read and is_written:
                return True, True
        return is_read, is_written
