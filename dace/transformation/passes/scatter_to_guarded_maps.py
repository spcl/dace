# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end pass: detect scatter loops, guard their index arrays, parallelize.

A *scatter loop* is a ``LoopRegion`` whose body writes to a non-transient array at
an index of the form ``arr[idx[i]]`` -- i.e. the write subset uses a symbol that
the loop's interstate edges bind to a read of another array indexed by the loop
variable. ``LoopToMap`` refuses such loops by default because two iterations may
write the same slot; the user's contract is that ``idx`` is a permutation (no
duplicates → no write-write race), and ``LoopToMap``'s ``permissive`` mode lifts
the loop under that assumption.

This pass operationalises that contract end-to-end:

1. **Detect** every scatter loop in the SDFG by scanning each ``LoopRegion``'s
   interstate-edge assignments for ``sym := idx[loop_var]`` bindings and matching
   them to symbol references in write-memlet subsets to non-transient arrays.
   The union of source-array names is the set of ``idx`` arrays.
2. **Guard** each detected ``idx`` array via
   :func:`~dace.transformation.passes.scatter_conflict_guard.insert_scatter_guard`,
   which inserts an ``IntegerSort`` + adjacent-equal-pair check + ``__builtin_trap()``
   at the earliest legal CFG state.
3. **Parallelize** by applying ``LoopToMap`` in ``permissive`` mode, which lifts
   the scatter loops (and any other previously refused permissive cases) into
   parallel Maps.

The ordering is intentional: guards are emitted *before* permissive lifts, so on
collision the abort fires before any consumer reads the corrupted output.
"""
import ast
from typing import Iterable, Optional, Set

from dace import SDFG, properties
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.scatter_conflict_guard import insert_scatter_guard


@properties.make_properties
@xf.explicit_cf_compatible
class ScatterToGuardedMaps(ppl.Pass):
    """Detect scatter loops, insert per-array runtime guards, then permissively lift to maps.

    Idempotence: the underlying guard utility refuses to emit a second guard for the
    same ``idx`` array (the ``_scatter_guard_sorted_<name>`` transient acts as the
    presence marker). Re-running this pass on an SDFG it has already guarded is a
    no-op for the guard step; the ``LoopToMap`` step still re-applies and is itself
    idempotent on already-lifted Maps.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        """Run the full pipeline. Returns the number of distinct ``idx`` arrays guarded,
        or ``None`` if no scatter loop was found.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap

        idx_arrays = detect_scatter_idx_arrays(sdfg)
        for idx_name in sorted(idx_arrays):
            # ``insert_scatter_guard`` refuses to double-emit; swallow that one case so
            # re-running the pass on an already-guarded SDFG is a no-op.
            try:
                insert_scatter_guard(sdfg, idx_name)
            except ValueError as exc:
                if 'already exists' not in str(exc):
                    raise

        sdfg.apply_transformations_repeated(LoopToMap, permissive=True)
        return len(idx_arrays) or None


def detect_scatter_idx_arrays(sdfg: SDFG) -> Set[str]:
    """Find every ``idx`` array name used as an indirect-write index in any LoopRegion.

    Scans every ``LoopRegion``'s interstate edges for assignments of the form
    ``sym := arr[loop_var]`` (or ``sym := arr[loop_var + const]`` etc.), then
    checks every write-memlet subset inside the region's states for references
    to ``sym``. Whenever a match is found, ``arr`` is added to the result set.

    :param sdfg: The SDFG to scan; nested SDFGs are walked too.
    :returns: Set of ``idx`` array names (resolved against ``sdfg.arrays`` at the
              owning SDFG level).
    """
    idx_arrays: Set[str] = set()
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if not (isinstance(region, LoopRegion) and region.loop_variable):
                continue
            bindings = _collect_indirect_bindings(region, sd)
            if not bindings:
                continue
            for state in region.all_states():
                for node in state.data_nodes():
                    if state.in_degree(node) == 0:
                        continue
                    desc = sd.arrays.get(node.data)
                    if desc is None or getattr(desc, 'transient', False):
                        continue
                    for e in state.in_edges(node):
                        if e.data is None or e.data.subset is None:
                            continue
                        for sym in e.data.subset.free_symbols:
                            arr = bindings.get(str(sym))
                            if arr is not None:
                                idx_arrays.add(arr)
    return idx_arrays


def _collect_indirect_bindings(region: LoopRegion, sdfg: SDFG) -> dict[str, str]:
    """Map each symbol bound by ``region``'s interstate edges to its source data
    array, when the binding is of the form ``sym := arr[loop_var]``.

    Conservative: only the simplest form -- bare ``arr[<loop_var>]`` -- is
    recognised. Compound expressions like ``arr[loop_var] + 1`` are skipped;
    they do not arise from the DaCe Python frontend's scatter lowering, and
    extending the recognition surface risks misclassifying non-scatter
    interstate computations.
    """
    bindings: dict[str, str] = {}
    loop_var = region.loop_variable
    for e in region.edges():
        for lhs, rhs in (e.data.assignments or {}).items():
            arr = _resolve_indirect_source(rhs, loop_var, sdfg)
            if arr is not None:
                bindings[lhs] = arr
    return bindings


def _resolve_indirect_source(rhs_str: str, loop_var: str, sdfg: SDFG) -> Optional[str]:
    """Return ``arr`` if ``rhs_str`` is ``arr[loop_var]`` and ``arr`` is a data
    descriptor in ``sdfg``; ``None`` otherwise.
    """
    try:
        tree = ast.parse(str(rhs_str), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(tree, ast.Subscript):
        return None
    if not isinstance(tree.value, ast.Name):
        return None
    arr = tree.value.id
    if arr not in sdfg.arrays:
        return None
    idx = tree.slice
    # Python <3.9 wraps the slice in ast.Index; unwrap.
    if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
        idx = idx.value
    if not (isinstance(idx, ast.Name) and idx.id == loop_var):
        return None
    return arr


__all__ = ['ScatterToGuardedMaps', 'detect_scatter_idx_arrays']
