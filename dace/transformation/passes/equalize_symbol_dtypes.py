# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Make every symbol name carry the one dtype its scope declares."""

import contextlib
from typing import Any, Dict, Iterator, Mapping, Optional

import sympy

from dace import SDFG, SDFGState, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, sdfg_scope_symbols
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.scopes import state_scope_symbol_tables


def is_scalar_dtype(dtype: dtypes.typeclass) -> bool:
    """A plain scalar typeclass, the only kind a symbol can carry (not a pointer or the typeless ``void``)."""
    return type(dtype) is dtypes.typeclass and dtype.type is not None


def retype_expression(expr: Any, table: Mapping[str, dtypes.typeclass]) -> Any:
    """``expr`` with every symbol of a declared name rebuilt at the declared dtype, keeping its assumptions.

    :param expr: A sympy expression, ``SymExpr``, or a plain number (returned as is).
    :param table: Declared dtype per symbol name, as :func:`~dace.transformation.passes.analysis.scopes.state_scope_symbol_tables` builds it.
    :return: ``expr`` itself when no symbol disagrees with ``table``.
    """
    if isinstance(expr, symbolic.SymExpr):
        exact = retype_expression(expr.expr, table)
        approx = retype_expression(expr.approx, table)
        return expr if exact is expr.expr and approx is expr.approx else symbolic.SymExpr(exact, approx)
    if not isinstance(expr, sympy.Basic):
        return expr
    replacements = {}
    for sym in expr.free_symbols:
        if not symbolic.is_symbol_leaf(sym):
            continue
        declared = table.get(sym.name)
        if declared is None or declared == sym.dtype or not is_scalar_dtype(declared):
            continue
        replacements[sym] = symbolic.symbol(sym.name, declared, **sym.assumptions0)
    return expr.xreplace(replacements) if replacements else expr


def retype_subset(subset: Optional[subsets.Subset], table: Mapping[str, dtypes.typeclass]) -> Optional[subsets.Subset]:
    """``subset`` with its bounds retyped by :func:`retype_expression`; the same object when nothing changes."""
    if not isinstance(subset, subsets.Range):
        return subset
    ranges = [tuple(retype_expression(bound, table) for bound in dim) for dim in subset.ranges]
    if all(old is new for dim, new_dim in zip(subset.ranges, ranges) for old, new in zip(dim, new_dim)):
        return subset
    retyped = subsets.Range(ranges)
    retyped.tile_sizes = list(subset.tile_sizes)
    return retyped


@properties.make_properties
@transformation.explicit_cf_compatible
class EqualizeSymbolDtypes(ppl.Pass):
    """Rebuild each symbol at the dtype its scope declares, so one name is one dtype.

    A symbol's dtype is part of its identity, so a name minted from a bare string in one place and read from a typed
    declaration in another is two unrelated symbols that never cancel. The declaration of a name is the single
    authority: ``sdfg.symbols`` for a free or interstate symbol, the loop or map that binds an iterator or dynamic
    input for a scoped one. This pass rewrites every memlet subset, map range, data descriptor extent and symbol
    mapping that spells a declared name at another dtype.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes | ppl.Modifies.Symbols))

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Retype the whole SDFG tree in place.

        :param sdfg: The SDFG to modify; nested SDFGs are visited too.
        :param _: Pipeline results, unused.
        :return: The number of rewritten expressions, or ``None`` if every symbol already had its declared dtype.
        """
        rewritten = 0
        for owner in sdfg.all_sdfgs_recursive():
            base = sdfg_scope_symbols(owner)
            self.declare(base)
            for region in owner.all_control_flow_regions():
                if isinstance(region, LoopRegion):
                    self.declare(region.new_symbols(base))
            rewritten += self.retype_descriptors(owner, base)
            for state in owner.states():
                rewritten += self.retype_state(owner, state, base)
        return rewritten or None

    @staticmethod
    def declare(declared: Mapping[str, dtypes.typeclass]) -> None:
        """Make the active authority follow the SDFG, so a name that a later pass parses from text comes out at the
        dtype the instances here are rebuilt at."""
        for name, dtype in declared.items():
            if is_scalar_dtype(dtype):
                symbolic.declare_symbol_dtype(name, dtype)

    def retype_descriptors(self, sdfg: SDFG, table: Mapping[str, dtypes.typeclass]) -> int:
        rewritten = 0
        for desc in sdfg.arrays.values():
            shape = [retype_expression(extent, table) for extent in desc.shape]
            strides = [retype_expression(stride, table) for stride in desc.strides]
            total = retype_expression(desc.total_size, table)
            offset = [retype_expression(offs, table) for offs in desc.offset]
            if (
                any(new is not old for new, old in zip(shape, desc.shape))
                or any(new is not old for new, old in zip(strides, desc.strides))
                or total is not desc.total_size
                or any(new is not old for new, old in zip(offset, desc.offset))
            ):
                desc.shape, desc.strides, desc.total_size, desc.offset = shape, strides, total, offset
                rewritten += 1
        return rewritten

    def retype_state(self, sdfg: SDFG, state: SDFGState, base: Mapping[str, dtypes.typeclass]) -> int:
        tables = state_scope_symbol_tables(sdfg, state, dict(base))
        rewritten = 0
        for node in state.nodes():
            if isinstance(node, nodes.EntryNode):
                scoped = tables[node]
                self.declare({name: scoped[name] for name in node.new_symbol_names(sdfg, state) if name in scoped})
            if isinstance(node, nodes.MapEntry):
                table = tables[state.entry_node(node)]
                range_ = retype_subset(node.map.range, table)
                if range_ is not node.map.range:
                    node.map.range = range_
                    rewritten += 1
            elif isinstance(node, nodes.NestedSDFG):
                table = tables[state.entry_node(node)]
                for name, value in node.symbol_mapping.items():
                    # A name mapped onto itself is one symbol across the boundary, so the nested declaration takes the
                    # dtype of the outer one.
                    if str(value) == name and name in node.sdfg.symbols:
                        outer = table.get(name)
                        if outer is not None and is_scalar_dtype(outer) and node.sdfg.symbols[name] != outer:
                            node.sdfg.symbols[name] = outer
                            rewritten += 1
                    retyped = retype_expression(
                        symbolic.pystr_to_symbolic(value) if isinstance(value, str) else value, table
                    )
                    if not isinstance(value, str) and retyped is not value:
                        node.symbol_mapping[name] = retyped
                        rewritten += 1
        for edge in state.edges():
            memlet = edge.data
            if memlet.is_empty():
                continue
            scope = edge.dst if isinstance(edge.src, nodes.EntryNode) else edge.src
            table = tables[state.entry_node(scope)]
            subset = retype_subset(memlet.subset, table)
            other_subset = retype_subset(memlet.other_subset, table)
            volume = retype_expression(memlet.volume, table)
            if subset is not memlet.subset or other_subset is not memlet.other_subset or volume is not memlet.volume:
                memlet.subset, memlet.other_subset, memlet.volume = subset, other_subset, volume
                rewritten += 1
        return rewritten


def equalize(sdfg: SDFG) -> Optional[int]:
    """Rebuild ``sdfg`` at the dtypes it declares, with the symbol-dtype authority following the declarations."""
    with symbolic.serialization_symbol_dtypes({}, inherit=True):
        return EqualizeSymbolDtypes().apply_pass(sdfg, {})


@contextlib.contextmanager
def equalized(sdfg: SDFG) -> Iterator[None]:
    """Run a stage of passes under the dtypes ``sdfg`` declares, with ``sdfg`` rebuilt at them on entry and on exit.

    Every pass of the stage parses names from text, so the names the SDFG declares are made known to the symbol-dtype
    authority first; what the stage mints at another dtype is rebuilt on exit.
    """
    with symbolic.serialization_symbol_dtypes({}, inherit=True):
        EqualizeSymbolDtypes().apply_pass(sdfg, {})
        yield
        EqualizeSymbolDtypes().apply_pass(sdfg, {})
