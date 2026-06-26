# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Make the initialization of fresh WCR reduction accumulators explicit.

A reduction ``acc[i] += f(i, j)`` writes ``acc`` via a write-conflict-resolution
(WCR) memlet and assumes ``acc`` holds the reduction *identity* (0 for sum, 1 for
product, ...) before the first accumulate. When ``acc`` is a *fresh* transient --
written only by the WCR, never by a plain "seed" write -- nothing in the SDFG
establishes that identity: codegen allocates the transient with an uninitialized
``new T[N]`` and immediately accumulates into it. The reduction therefore reads
uninitialized memory.

In practice this is masked when the allocation happens to return zeroed pages (a
fresh heap allocation usually does), so simple cases appear to work. But it is
undefined behavior and breaks for real:

* a ``Persistent`` accumulator reused across invocations keeps the previous run's
  value, so the second call accumulates on top of the first (a doubled result);
* nesting / memory reuse introduced by later transformations hands the reduction
  dirty memory, producing flaky garbage.

This pass detects fresh WCR accumulators and inserts an explicit map that writes the
reduction identity into the full array in a state placed at the very start of the
(nested) SDFG, so the accumulate always reads a defined value. It is value-preserving:

* a *fresh* accumulator (no plain write) is correctly seeded with the identity;
* an accumulate-onto-existing target (e.g. ``C[i,j] = beta*C[i,j]`` then
  ``C[i,j] += ...``) has a plain seed write, so it is NOT considered fresh and is
  left untouched -- its seed is preserved.

The pass is idempotent: the init it inserts is itself a plain write, so a fixed-point
re-run no longer classifies the accumulator as fresh.
"""
from typing import Dict, Optional

from dace import SDFG, dtypes, properties
from dace import nodes
from dace.frontend import operations
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


@properties.make_properties
@xf.explicit_cf_compatible
class InitializeReductionAccumulators(ppl.Pass):
    """Insert an explicit identity-init for transients written only via WCR (fresh
    reduction accumulators), so a reduction never reads uninitialized memory."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Only relevant when the dataflow / WCR structure changes.
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def depends_on(self):
        return set()

    def _fresh_wcr_targets(self, sdfg: SDFG) -> Dict[str, str]:
        """``{array_name: wcr_lambda}`` for each array written ONLY by WCR edges (at
        least one) and never by a plain write -- i.e. a fresh reduction accumulator."""
        wcr_lambda: Dict[str, str] = {}
        has_plain_write = set()
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode):
                    continue
                for e in state.in_edges(node):
                    if e.data is None or e.data.data is None:
                        continue
                    if e.data.wcr is not None:
                        wcr_lambda.setdefault(node.data, e.data.wcr)
                    else:
                        has_plain_write.add(node.data)
        return {n: w for n, w in wcr_lambda.items() if n not in has_plain_write}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict) -> Optional[int]:
        count = 0
        # Recurse into nested SDFGs (a nested reduction body may own its accumulator).
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.NestedSDFG):
                count += self.apply_pass(node.sdfg, _pipeline_results) or 0

        targets = self._fresh_wcr_targets(sdfg)
        init_state = None
        for name, wcr in targets.items():
            desc = sdfg.arrays.get(name)
            # Only auto-init transient accumulators -- a non-transient output is the
            # caller's buffer and may legitimately carry an incoming seed.
            if desc is None or not desc.transient:
                continue
            redtype = operations.detect_reduction_type(wcr)
            if redtype == dtypes.ReductionType.Custom:
                continue  # unknown identity -- leave as-is
            try:
                identity = dtypes.reduction_identity(desc.dtype, redtype)
            except Exception:
                continue
            if init_state is None:
                init_state = sdfg.add_state_before(sdfg.start_state, label='reduction_init', is_start_block=True)
            self._emit_init_map(sdfg, init_state, name, desc, identity)
            count += 1
        return count or None

    def _emit_init_map(self, sdfg: SDFG, state, name: str, desc, identity) -> None:
        from dace import Memlet
        shape = desc.shape
        idx = [f'_i{d}' for d in range(len(shape))]
        rng = {idx[d]: f'0:{shape[d]}' for d in range(len(shape))}
        me, mx = state.add_map(f'init_{name}', rng)
        tasklet = state.add_tasklet(f'init_{name}', set(), {'__out'}, f'__out = {identity}')
        w = state.add_access(name)
        state.add_edge(me, None, tasklet, None, Memlet())
        state.add_memlet_path(tasklet, mx, w, src_conn='__out', memlet=Memlet(f'{name}[{", ".join(idx)}]'))
