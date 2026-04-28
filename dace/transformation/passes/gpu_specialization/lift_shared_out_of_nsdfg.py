# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift ``GPU_Shared`` transients out of nested SDFGs into the SDFG that
owns the enclosing GPU_Device map.

Why: ``__shared__ T name[N]`` is only valid inside a CUDA kernel function.
When a Shared transient lives inside an inner NestedSDFG, the framecode
allocation walker has no signal that the array's home is the kernel — by
default it routes the declaration to whichever scope the inner SDFG
hierarchy points at, which often ends up at the top SDFG. The generated
``__shared__`` declaration then never lands inside any ``__global__``
function, and the inner ``DACE_DFI`` body references an undeclared
identifier (compile error).

This pass promotes the descriptor to the SDFG that owns the kernel
``MapEntry`` and wires it through the NestedSDFG via connectors. A
dependency edge ``MapEntry -> AccessNode`` pins the array to the kernel
scope so the framecode allocates it there. A symmetric edge to
``MapExit`` keeps it live through kernel exit.
"""
import copy as _copy
from typing import Any, Dict, List, Tuple

from dace import SDFG, SDFGState, dtypes, properties, nodes
from dace.memlet import Memlet
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import innermost_enclosing_map


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
        # Refresh scope_dict caches: the framecode walker that runs after
        # this pass keys allocation routing on scope membership.
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
                    kernel_entry = innermost_enclosing_map(state, n, dtypes.ScheduleType.GPU_Device)
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

    def _lift_one(self, name: str, inner_sdfg: SDFG, nsdfg_node: nodes.NestedSDFG, outer_sdfg: SDFG,
                  outer_state: SDFGState, kernel_entry: nodes.MapEntry):
        """Promote ``name`` and wire it through ``nsdfg_node``::

            MapEntry --(empty, dep)--> AN_read --(in:name)--> NSDFG
            NSDFG --(out:name)--> AN_write --(empty, dep)--> MapExit

        Two ``AccessNode``s for the same outer descriptor (read side +
        write side) keep the state acyclic when the inner mutates the
        array — DaCe rejects a single-AN read+write cycle around an NSDFG.
        ``force=True`` on the connector adds is required because the same
        name appears in both ``in_connectors`` and ``out_connectors`` (the
        standard DaCe pattern for inout arrays)."""
        inner_desc = inner_sdfg.arrays[name]

        outer_name = self._pick_outer_name(name, outer_sdfg)
        outer_sdfg.add_datadesc(outer_name, inner_desc, find_new_name=False)
        inner_param_desc = _copy.deepcopy(inner_desc)
        inner_param_desc.transient = False
        del inner_sdfg.arrays[name]
        inner_sdfg.add_datadesc(name, inner_param_desc)

        is_read, is_written = _classify_inner_usage(inner_sdfg, name)
        full_subset = Range.from_array(inner_desc)
        kernel_exit = outer_state.exit_node(kernel_entry)
        an_write = None

        if is_read:
            an_read = outer_state.add_access(outer_name)
            outer_state.add_edge(kernel_entry, None, an_read, None, Memlet())
            nsdfg_node.add_in_connector(name, force=True)
            outer_state.add_edge(an_read, None, nsdfg_node, name,
                                 Memlet(data=outer_name, subset=_copy.deepcopy(full_subset)))

        if is_written:
            an_write = outer_state.add_access(outer_name)
            nsdfg_node.add_out_connector(name, force=True)
            outer_state.add_edge(nsdfg_node, name, an_write, None,
                                 Memlet(data=outer_name, subset=_copy.deepcopy(full_subset)))
            outer_state.add_edge(an_write, None, kernel_exit, None, Memlet())

        # Write-only case still needs allocation anchoring — without a read
        # side the AN_write has no incoming dep from MapEntry, so add one.
        if is_written and not is_read:
            outer_state.add_edge(kernel_entry, None, an_write, None, Memlet())

        outer_state._clear_scopedict_cache()

    @staticmethod
    def _pick_outer_name(name: str, outer_sdfg: SDFG) -> str:
        """Return ``name`` if it's free in ``outer_sdfg``, else ``name_0``,
        ``name_1``, ... so the lift never overwrites an existing descriptor."""
        if name not in outer_sdfg.arrays:
            return name
        i = 0
        while f'{name}_{i}' in outer_sdfg.arrays:
            i += 1
        return f'{name}_{i}'


def _classify_inner_usage(inner_sdfg: SDFG, name: str) -> Tuple[bool, bool]:
    """``(is_read, is_written)`` for ``name`` inside ``inner_sdfg``, via
    each state's ``read_and_write_sets`` (DaCe's authoritative source for
    connector-direction inference)."""
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
