# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Canonicalize nested-SDFG array names so that, across one translation unit, every data name maps to
exactly ONE ``(ndim, strides, offset)`` signature -- and therefore the experimental-readable CPU
generator's per-name ``<name>_idx`` index helper has a single body per output file.

A nested SDFG that receives a view / non-full subset of a parent array under the *same* connector name
gives the inner descriptor different strides / offset -- i.e. a *different-body* ``A_idx`` with the same
base name. That is a hard C++ redefinition that the byte-identical ``deduplicate_functions`` post-pass
cannot collapse. This pass renames the inner (nested) occurrence to a fresh, globally-unique name --
updating the descriptor (``replace_dict``), every internal occurrence, and the owning ``NestedSDFG``
node's connector + incident edge -- so the name -> body map is 1:1 before codegen runs.

Only the readable-codegen preprocessing block runs this pass (see ``dace.codegen.codegen.generate_code``),
so the legacy generator is unaffected.
"""
from typing import Dict, Optional, Set, Tuple

from dace import data as dt
from dace.sdfg.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.pass_pipeline import Modifies


class CanonicalizeNestedIndexNames(ppl.Pass):
    """Rename nested-SDFG arrays so each data name owns a single ``(ndim, strides, offset)`` signature."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> Modifies:
        return Modifies.Descriptors | Modifies.AccessNodes | Modifies.Edges

    def should_reapply(self, modified: Modifies) -> bool:
        return False

    def _signature(self, desc) -> Optional[Tuple]:
        # Mirror ExperimentalCPUCodeGen._register_index_function's dedup key exactly: an ``<name>_idx``
        # body is a function of (ndim, strides, offset). Only Array-like descriptors get an index helper;
        # anything else (Scalar / Stream / Structure / ...) cannot collide on one.
        if not isinstance(desc, dt.Array):
            return None
        return (len(desc.shape), tuple(str(s) for s in desc.strides), tuple(str(o) for o in desc.offset))

    def _all_names(self, sdfg: SDFG) -> Set[str]:
        names: Set[str] = set()
        for sub in sdfg.all_sdfgs_recursive():
            names |= set(sub.arrays.keys())
            names |= set(sub.constants_prop.keys())
            names |= set(str(s) for s in sub.free_symbols)
        return names

    def _unique(self, base: str, used: Set[str]) -> str:
        i = 0
        while True:
            cand = '%s_v%d' % (base, i)
            if cand not in used:
                used.add(cand)
                return cand
            i += 1

    def _rename_in_nested(self, sub: SDFG, old: str, new: str) -> None:
        # Rename the descriptor + every internal occurrence (memlets, access nodes).
        sub.replace_dict({old: new})
        # Fix the owning NestedSDFG node's connector + incident edges (by DaCe convention the outer-side
        # connector name equals the inner data name it binds).
        node = sub.parent_nsdfg_node
        state = sub.parent
        if node is None or state is None:
            return
        # ``force=True``: for an inout connector (same name on both sides) the first add makes ``new``
        # already exist on the node, so the second add would otherwise be refused.
        if old in node.in_connectors:
            ct = node.in_connectors[old]
            node.remove_in_connector(old)
            node.add_in_connector(new, ct, force=True)
            for e in state.in_edges(node):
                if e.dst_conn == old:
                    e.dst_conn = new
        if old in node.out_connectors:
            ct = node.out_connectors[old]
            node.remove_out_connector(old)
            node.add_out_connector(new, ct, force=True)
            for e in state.out_edges(node):
                if e.src_conn == old:
                    e.src_conn = new

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        registry: Dict[str, Tuple] = {}  # name -> signature that owns it
        used: Set[str] = self._all_names(sdfg)
        renamed = 0
        # Top SDFG first (its argument / transient names win), then nested SDFGs depth-first, so a nested
        # occurrence is always the one renamed on a differing-signature collision.
        for sub in sdfg.all_sdfgs_recursive():
            is_root = sub is sdfg
            for name in list(sub.arrays.keys()):
                sig = self._signature(sub.arrays[name])
                if sig is None:
                    continue
                owner = registry.get(name)
                if owner is None:
                    registry[name] = sig
                    continue
                if owner == sig:
                    continue  # identical body -> emission dedup handles it, keep the name
                if is_root:
                    continue  # never rename a root argument / transient
                new = self._unique(name, used)
                self._rename_in_nested(sub, name, new)
                registry[new] = sig
                renamed += 1
        return renamed or None
