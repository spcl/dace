# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stores matrices that are arguments of an SDFG in blocks of columns, the layout of matrix-product operands that a
product reads contiguously (see ``dace.libraries.blas.blocked``)."""

import copy
from dataclasses import dataclass, field

import sympy

from dace import SDFG, properties, subsets, symbolic
from dace import data as dt
from dace.libraries.blas.blocked import blocked_shape
from dace.libraries.blas.indexed import index_connector
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import AbstractControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def _multiple_of(expr, block: int) -> bool:
    """Whether an index is provably a multiple of ``block`` (its symbols, and functions of them, are integers)."""
    expr = symbolic.pystr_to_symbolic(expr)
    if not symbolic.issymbolic(expr):
        return int(expr) % block == 0
    integers = {a: sympy.Dummy(integer=True) for a in expr.atoms(sympy.Function) | expr.free_symbols}
    return (sympy.simplify(expr.xreplace(integers)) / block).is_integer is True


@dataclass
class _Container:
    """A container that is a matrix (or a batch of them) to store in blocks: an argument, a view of one, or the array
    of a nested SDFG that one is passed to, with the subset of its parent (in the parent's matrix coordinates) that
    it is."""

    sdfg: SDFG
    name: str
    parent: "_Container | None" = None
    subset: subsets.Range | None = None


@dataclass
class _Plan:
    """What storing an argument in blocks changes: the containers, the memlets (and which of their subsets, in the
    coordinates of which container, and whether exactly), and the matrix-product operands."""

    containers: list = field(default_factory=list)
    memlets: list = field(default_factory=list)  # (memlet, attribute, its subset, exact)
    operands: list = field(default_factory=list)  # (node, connector)
    visited: set = field(default_factory=set)  # (SDFG, name) of the containers


@properties.make_properties
@transformation.explicit_cf_compatible
class BlockMatrixOperands(ppl.Pass):
    """
    Stores matrices that are arguments of an SDFG in blocks of ``block`` columns (see ``dace.libraries.blas.blocked``):
    an argument ``X[..., R, C]`` becomes ``X[..., C / block, R, block]``. Matrix products (``MatMul``, ``Gemm``) that
    read it take it as a blocked operand, and a product that reads ``block`` columns of many rows at a time, as the
    thread-block products of GPU kernels do, then reads contiguous memory.

    The SDFG then takes the arguments in the new layout (as inference engines pack their weights once): the caller
    stores them so, e.g., with the ``Pack`` library node. Every access to an argument must read it: a matrix-product
    operand (rows, and whole blocks of columns), a view or a nested SDFG's array that is such a matrix in turn, or an
    element. Otherwise the pass raises a ``ValueError`` and changes nothing.
    """

    CATEGORY: str = "Layout"

    names = properties.SetProperty(
        element_type=str, default=set(), category="Applicability", desc="The arguments to store in blocks of columns"
    )
    block = properties.Property(dtype=int, default=32, category="Applicability", desc="The columns of a block")

    def __init__(self, names=(), block: int = 32):
        """
        :param names: The arguments to store in blocks of columns.
        :param block: The columns of a block.
        """
        super().__init__()
        self.names = set(names)
        self.block = block

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> dict[str, dt.Array] | None:
        """
        Stores the arguments in blocks.

        :return: The descriptors of the arguments before (in the usual layout), or None if there are none.
        """
        before = {name: copy.deepcopy(sdfg.arrays[name]) for name in sorted(self.names) if name in sdfg.arrays}
        # Everything is planned and computed first, so that the arguments are stored in blocks all or none
        changes = [self._changes(self._plan(sdfg, name)) for name in sorted(self.names)]
        for change in changes:
            change()
        return before or None

    # Planning: what changes, or why it cannot

    def _plan(self, sdfg: SDFG, name: str) -> _Plan:
        if name not in sdfg.arrays:
            raise ValueError(f'"{name}" is not a container of the SDFG')
        desc = sdfg.arrays[name]
        if desc.transient or not isinstance(desc, dt.Array) or isinstance(desc, dt.View):
            raise ValueError(f'"{name}" is not an array argument')
        plan = _Plan()
        self._visit(_Container(sdfg, name), plan)
        return plan

    def _fail(self, container: _Container, reason: str):
        raise ValueError(
            f'"{container.name}" (in "{container.sdfg.label}") cannot be stored in blocks of columns: {reason}'
        )

    def _visit(self, container: _Container, plan: _Plan) -> None:
        sdfg, name, desc = container.sdfg, container.name, container.sdfg.arrays[container.name]
        if len(desc.shape) < 2 or not _multiple_of(desc.shape[-1], self.block):
            self._fail(container, f"its shape {desc.shape} is not a matrix of whole blocks of {self.block} columns")
        key = (id(sdfg), name)
        if key in plan.visited:
            return
        plan.visited.add(key)
        plan.containers.append(container)
        for region in sdfg.all_control_flow_regions(recursive=True):
            if region.sdfg is not sdfg:
                continue
            if any(name in edge.data.read_symbols() for edge in region.edges()):
                self._fail(container, "an interstate edge reads it")
            if not isinstance(region, SDFG) and isinstance(region, AbstractControlFlowRegion):
                if name in region.used_symbols(all_symbols=True, with_contents=False):
                    self._fail(container, f'the control flow of "{region.label}" reads it')
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data != name:
                    continue
                for e in state.in_edges(node):
                    if not e.data.is_empty() and not (isinstance(desc, dt.View) and e.dst_conn == "views"):
                        self._fail(container, f'it is written in "{state.label}"')
                for e in state.out_edges(node):
                    if e.data.is_empty():
                        continue
                    if isinstance(desc, dt.View) and e is sdutil.get_view_edge(state, node):
                        self._fail(container, f'a view of it is written in "{state.label}"')
                    for tree in state.memlet_tree(e).traverse_children(include_self=True):
                        self._visit_edge(container, state, tree.edge, plan)

    def _visit_edge(self, container: _Container, state, edge, plan: _Plan) -> None:
        memlet, dst = edge.data, edge.dst
        # Whether the memlet's ``subset`` is in the container's coordinates (otherwise its ``other_subset``, or all of
        # the container if it has none)
        own = memlet.data == container.name
        ours = memlet.subset if own else memlet.other_subset
        if ours is None:
            ours = subsets.Range.from_array(state.sdfg.arrays[container.name])
        attribute = "subset" if own else "other_subset"
        if isinstance(dst, nodes.EntryNode):  # Within a scope: an over-approximation
            plan.memlets.append((memlet, attribute, ours, False))
            return
        if isinstance(dst, nodes.AccessNode):
            child = dst.data
            if not isinstance(state.sdfg.arrays[child], dt.View) or edge.dst_conn != "views":
                if ours.num_elements() != 1:
                    self._fail(container, f'{ours} of it is copied to "{child}"')
                plan.memlets.append((memlet, attribute, ours, True))  # An element
                return
            plan.memlets.append((memlet, attribute, ours, True))
            theirs = memlet.other_subset if own else memlet.subset
            if theirs is not None:  # In the coordinates of the view, which is stored in blocks too
                plan.memlets.append((memlet, "other_subset" if own else "subset", theirs, True))
            self._visit(_Container(state.sdfg, child, container, copy.deepcopy(ours)), plan)
            return
        subset = ours
        if not own:
            self._fail(container, f'an access in "{state.label}" is not in its coordinates')
        if isinstance(dst, nodes.NestedSDFG):
            plan.memlets.append((memlet, "subset", subset, True))
            if edge.dst_conn in dst.out_connectors:
                self._fail(container, f'"{dst.label}" writes it')
            self._visit(_Container(dst.sdfg, edge.dst_conn, container, copy.deepcopy(subset)), plan)
        elif isinstance(dst, (MatMul, Gemm)) and edge.dst_conn in ("_a", "_b"):
            if index_connector(edge.dst_conn) in dst.in_connectors:
                self._fail(container, f'"{dst.label}" reads it through an index')
            if any(size != 1 for size in subset.size()[:-2]):
                self._fail(container, f'"{dst.label}" reads a batch of matrices of it')
            plan.memlets.append((memlet, "subset", subset, True))
            plan.operands.append((dst, edge.dst_conn))
        elif isinstance(dst, nodes.Tasklet) and subset.num_elements() == 1:
            plan.memlets.append((memlet, "subset", subset, True))
        else:
            self._fail(container, f'"{dst.label}" reads {subset} of it')

    # Applying a plan

    def _changes(self, plan: _Plan):
        """The new descriptors and subsets of a plan, and a function that applies them."""
        converted = []
        for memlet, attribute, subset, exact in plan.memlets:
            converted.append((memlet, attribute, self._converted(subset, exact)))
        descriptors = {}
        for container in plan.containers:
            descriptors[id(container)] = self._descriptor(container, descriptors)

        def apply():
            for container in plan.containers:
                container.sdfg.arrays[container.name] = descriptors[id(container)]
            for memlet, attribute, subset in converted:
                setattr(memlet, attribute, subset)
            for node, connector in plan.operands:
                node.blocked = set(node.blocked) | {connector}

        return apply

    def _converted(self, subset, exact: bool) -> subsets.Range:
        """A subset of a matrix (or a batch of them), in the coordinates of the blocked array."""
        b = self.block
        *batch, rows, (c0, c1, step) = subset.ranges
        if c0 == c1:  # A column
            blocks, inner = (symbolic.int_floor(c0, b),) * 2 + (1,), (c0 % b, c0 % b, 1)
        elif step == 1 and _multiple_of(c0, b) and _multiple_of(c1 + 1, b):  # Whole blocks
            blocks, inner = (sympy.simplify(c0 / b), sympy.simplify((c1 + 1) / b) - 1, 1), (0, b - 1, 1)
        elif not exact:  # The blocks that the columns are in
            blocks, inner = (symbolic.int_floor(c0, b), symbolic.int_floor(c1, b), 1), (0, b - 1, 1)
        else:
            raise ValueError(f"The columns of {subset} are not whole blocks of {b}")
        return subsets.Range([*batch, blocks, rows, inner])

    def _descriptor(self, container: _Container, descriptors: dict) -> dt.Array:
        """The descriptor of a container in blocks: contiguous for an argument, otherwise the parent's subset."""
        desc = copy.deepcopy(container.sdfg.arrays[container.name])
        shape = blocked_shape(desc.shape, self.block)
        if container.parent is None:
            desc.set_shape(shape)
            return desc
        parent = descriptors[id(container.parent)]
        size = container.subset.size()
        if any(symbolic.equal(a, b) is False for a, b in zip(desc.shape[-2:], size[-2:])):
            self._fail(container, f"its shape {desc.shape} does not end with that of {container.subset}")
        # The batch dimensions of the container are those of the subset, or those of more than one element
        batch = list(range(len(size) - 2))
        if len(desc.shape) - 2 != len(batch):
            batch = [d for d in batch if size[d] != 1]
            if len(desc.shape) - 2 != len(batch):
                self._fail(container, f"its shape {desc.shape} is not that of {container.subset}")
        desc.set_shape(shape, strides=[parent.strides[d] for d in batch] + list(parent.strides[-3:]))
        return desc


__all__ = ["BlockMatrixOperands"]
