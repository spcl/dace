# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Size a transient whose extent names a symbol read from data by the largest extent that symbol allows.

``cum_x[order - dil_index - diu_index + 2]``, with ``dil_index = dil[p]`` assigned per particle, has a new extent
every iteration: it is allocated per iteration, and no kernel can own it. When every such symbol is nonnegative (every
store into what it is read from is, :func:`~dace.transformation.passes.analysis.array_facts.nonnegative_containers`)
and enters the extent linearly with a nonpositive coefficient, the extent is at most its value with those symbols at
zero (``order + 2``). The transient takes that extent; every access keeps its subset.
"""

from typing import Any

from dace import SDFG, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.sdfg import interstate_bound_names
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import array_facts


def extent_upper_bound(extent: Any, assigned: set[str], nonnegative: set[str]) -> Any | None:
    """``extent`` with every name of ``assigned`` it holds set to zero, when each is in ``nonnegative`` and enters
    linearly with a nonpositive constant coefficient (so zero maximizes it); ``None`` otherwise."""
    expr = symbolic.pystr_to_symbolic(extent)
    for name in sorted({str(s) for s in expr.free_symbols} & assigned):
        coefficients = symbolic.affine_coefficients(expr, name)
        if name not in nonnegative or coefficients is None:
            return None
        lead, rest = coefficients
        if not lead.is_number or lead > 0:
            return None
        expr = rest
    return expr


def keeps_strides(desc: dt.Array, shape: list) -> bool:
    """Whether ``desc`` is packed row-major and ``shape`` changes at most its leading extent, so no stride moves."""
    packed = dt.Array(desc.dtype, desc.shape).strides
    return all(symbolic.equal(a, b) is True for a, b in zip(desc.strides, packed)) and all(
        symbolic.equal(a, b) is True for a, b in zip(desc.shape[1:], shape[1:])
    )


def resize_plan(sdfg: SDFG, name: str, shape: list) -> list[tuple[SDFG, str, list]] | None:
    """Every descriptor that must take ``shape`` with ``sdfg``'s ``name``: the nested SDFG connectors it binds,
    restated in their own symbols; ``None`` if a view or a connector cannot follow without moving a stride."""
    desc = sdfg.arrays[name]
    plan = [(sdfg, name, shape)]
    strides_kept = keeps_strides(desc, shape)
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            # Through the scopes in between: the far end of each memlet path is what reads the descriptor
            ends = [
                (leaf.dst, leaf.dst_conn) if edge.src is node else (leaf.src, leaf.src_conn)
                for edge in state.all_edges(node)
                for leaf in state.memlet_tree(edge).leaves()
            ]
            for other, conn in ends:
                if isinstance(other, nodes.AccessNode) and isinstance(other.desc(sdfg), dt.View) and not strides_kept:
                    return None
                if not isinstance(other, nodes.NestedSDFG):
                    continue
                inner = other.sdfg.arrays[conn]
                if not strides_kept or not keeps_strides(inner, shape):
                    return None
                inverse = {str(value): key for key, value in other.symbol_mapping.items()}
                names = {str(s) for extent in shape for s in symbolic.pystr_to_symbolic(extent).free_symbols}
                if not names <= inverse.keys():
                    return None
                rename = {symbolic.symbol(n): symbolic.symbol(inverse[n]) for n in names}
                inner_shape = [symbolic.pystr_to_symbolic(extent).subs(rename) for extent in shape]
                if any(entry[0] is other.sdfg and entry[1] == conn for entry in plan):
                    continue
                deeper = resize_plan(other.sdfg, conn, inner_shape)
                if deeper is None:
                    return None
                plan += deeper
    return plan


@transformation.explicit_cf_compatible
class BoundDataDependentExtents(ppl.Pass):
    """Give each transient sized by a data-read symbol the extent that bounds it (module docstring)."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def nonnegative(self, sdfg: SDFG, known: dict[SDFG, set[str]]) -> set[str]:
        """The containers and symbols of ``sdfg`` provably nonnegative, given those of the SDFGs around it."""
        if sdfg in known:
            return known[sdfg]
        assumed: set[str] = set()
        node = sdfg.parent_nsdfg_node
        if node is not None:
            outer = self.nonnegative(sdfg.parent_sdfg, known)
            assumed = {e.dst_conn for e in sdfg.parent.in_edges(node) if e.dst_conn and e.data.data in outer}
            assumed |= {k for k, v in node.symbol_mapping.items() if array_facts.parses_nonnegative(str(v), outer)}
        known[sdfg] = array_facts.nonnegative_containers(sdfg, frozenset(assumed))
        return known[sdfg]

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Bound every transient extent the module docstring describes, in ``sdfg`` and its nested SDFGs.

        :returns: How many transients were resized, or ``None`` if none.
        """
        known: dict[SDFG, set[str]] = {}
        resized = 0
        for nested in sdfg.all_sdfgs_recursive():
            assigned = interstate_bound_names(nested)
            if not assigned:
                continue
            for name, desc in nested.arrays.items():
                if (
                    not desc.transient
                    or not isinstance(desc, dt.Array)
                    or isinstance(desc, dt.View)
                    or not {str(s) for s in desc.free_symbols} & assigned
                ):
                    continue
                nonnegative = self.nonnegative(nested, known)
                shape = [extent_upper_bound(extent, assigned, nonnegative) for extent in desc.shape]
                if any(extent is None for extent in shape):
                    continue
                plan = resize_plan(nested, name, shape)
                if plan is None:
                    continue
                for owner, resized_name, resized_shape in plan:
                    owner.arrays[resized_name].set_shape(resized_shape)
                resized += 1
        return resized or None
