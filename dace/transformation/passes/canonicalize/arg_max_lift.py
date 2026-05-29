# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift TSVC-style ``if a[i] OP bestv: bestv = a[i]`` argmax/argmin loops to a
:class:`~dace.libraries.standard.nodes.reduce.Reduce` libnode.

Pattern recognised (target: TSVC ``s314`` / ``s316`` value-only argmax/argmin):

.. code-block:: python

    x = a[0]
    for i in range(start, end):
        if a[i] > x:           # or `<`, `>=`, `<=`
            x = a[i]

The Python frontend lowers this to a multi-block loop body shaped like::

    [start_state (empty)]
        | iedge: a_index_sym = a[i]
        v
    [cond_prep_state (empty)]
        | iedge: tmp_sym = (a_index_sym OP carrier)
        v
    [ConditionalBlock(condition = tmp_sym)]
        true-branch -> [write_state: a -> a_index_AN -> assign-tasklet -> carrier_AN]

The pass walks this chain backward from the ConditionalBlock, extracts ``a``,
the carrier (``x``), and the comparison operator, then replaces the loop with a
:class:`Reduce` libnode (``Max`` for ``>`` / ``>=``, ``Min`` for ``<`` / ``<=``)
over ``a[start:end]`` writing to the carrier.

Scope of v1
-----------

* **Value only.** Tracks ``x``; does not yet handle a sibling ``index = i``
  write in the same true-branch (TSVC ``s315``). Index-tracking lift will use
  the existing ``Reduce(Max_Location)`` / ``Reduce(Min_Location)`` libnode
  variants which the CUDA expansion already maps to CUB ``ArgMax`` / ``ArgMin``.

* **Carrier storage.** Scalar (``data.Scalar``) and length-1 array (``shape ==
  (1,)``) carriers are supported. Symbol carriers are out of scope for v1.

* **Direct array reads.** ``a_index_sym = a[i]`` -- no unary transform on the
  gather (``maxv = abs(a[i])`` from TSVC ``s3113`` is a follow-up that emits a
  pre-tasklet computing the transform into a fresh buffer, then a ``Reduce``
  over the buffer).

* **No else branch.** Only the canonical ``if cond: write; else: nothing``
  shape; an else branch with side effects would need separate handling.

Soundness
---------

The rewrite is value-preserving because the original loop's sequential
semantics is exactly the running reduction along ``i``: at each iteration the
carrier holds the running ``max``/``min`` over ``a[start:i+1]``. After the
loop, the carrier equals ``max``/``min`` over ``a[start:end]``, which is what
the :class:`Reduce` libnode computes.

The pre-loop init (``x = a[start]``) is preserved by routing it as a direct
copy *before* the libnode -- the libnode itself runs with ``identity=None``
so its first read seeds itself from ``a[start]``.
"""
import ast
import re
from typing import Any, NamedTuple, Optional, Tuple

import dace
from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState, ControlFlowRegion, ConditionalBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.libraries.standard.nodes.reduce import Reduce


#: Map AST comparison op class -> DaCe reduction type.
_CMP_AST_TO_RTYPE = {
    ast.Gt: dtypes.ReductionType.Max,
    ast.GtE: dtypes.ReductionType.Max,
    ast.Lt: dtypes.ReductionType.Min,
    ast.LtE: dtypes.ReductionType.Min,
}


class _Match(NamedTuple):
    """A successfully matched argmax/argmin loop.

    :param op: ``Max`` or ``Min``.
    :param loop: The :class:`LoopRegion` to rewrite.
    :param parent: ``loop.parent_graph`` (cached).
    :param carrier_name: The carrier scalar's data name (``x`` in s314).
    :param carrier_kind: ``'scalar'`` or ``'length_one_array'``.
    :param carrier_subset: The carrier's single-point subset (``[0]``).
    :param input_array: The reduced-over array's data name (``a`` in s314).
    :param iter_start: Loop start expression.
    :param iter_end: Loop inclusive end expression.
    """
    op: dtypes.ReductionType
    loop: LoopRegion
    parent: ControlFlowRegion
    carrier_name: str
    carrier_kind: str
    carrier_subset: subsets.Range
    input_array: str
    iter_start: Any
    iter_end: Any


@properties.make_properties
@xf.explicit_cf_compatible
class ArgMaxLift(ppl.Pass):
    """Lift TSVC-style argmax/argmin loops to :class:`Reduce` libnodes."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        rewritten = 0
        for sd in sdfg.all_sdfgs_recursive():
            for region in list(sd.all_control_flow_regions()):
                if not (isinstance(region, LoopRegion) and region.loop_variable):
                    continue
                # Stale-snapshot guard: a previous rewrite may have removed this
                # LoopRegion from its parent.
                if region.parent_graph is None or region not in region.parent_graph.nodes():
                    continue
                m = self._match(region, sd)
                if m is None:
                    continue
                self._rewrite(m, sd)
                rewritten += 1
        return rewritten or None

    def _match(self, loop: LoopRegion, sdfg: SDFG) -> Optional[_Match]:
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None
        try:
            if int(symbolic.simplify(stride)) != 1:
                return None
        except (TypeError, ValueError):
            return None

        # Body must hold exactly one ConditionalBlock (with optional empty wrapper states).
        cond_block = None
        for b in loop.nodes():
            if isinstance(b, ConditionalBlock):
                if cond_block is not None:
                    return None
                cond_block = b
            elif isinstance(b, SDFGState):
                if len(b.nodes()) > 0:
                    return None  # any non-empty plain state in the body is unsupported in v1
            else:
                return None
        if cond_block is None:
            return None

        # The conditional must have exactly one (non-else) branch; no else / empty else.
        non_else = [(c, br) for c, br in cond_block.branches if c is not None]
        else_branches = [(c, br) for c, br in cond_block.branches if c is None]
        if len(non_else) != 1:
            return None
        cond_codeblock, true_branch = non_else[0]
        if any(self._branch_has_content(br) for _, br in else_branches):
            return None

        cond_expr_str = cond_codeblock.as_string.strip()
        # The condition is a single symbol set by an iedge upstream.
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', cond_expr_str):
            return None
        tmp_sym_name = cond_expr_str

        # Walk backward through iedges to find the comparison: ``tmp_sym = (g OP c)``.
        op_ast, gather_sym_name, carrier_name = self._resolve_tmp_iedge(loop, cond_block, tmp_sym_name)
        if op_ast is None:
            return None
        op = _CMP_AST_TO_RTYPE[op_ast]

        # Walk one more iedge back to find the gather: ``gather_sym = arr[loop_var]``.
        input_array = self._resolve_gather_iedge(loop, cond_block, gather_sym_name, loop.loop_variable, sdfg)
        if input_array is None:
            return None

        # The true-branch's interstate edges must not carry any assignments --
        # those are independent writes (TSVC s315 puts ``index = i`` on a
        # branch-internal iedge alongside the ``x = a[i]`` chain). The v1
        # rewrite cannot preserve them.
        for e in true_branch.edges():
            if e.data.assignments:
                return None

        # The true-branch must contain a single content state that writes
        # ``carrier = arr[loop_var]`` (additional empty states are allowed --
        # the frontend often emits a trailing ``assign_*`` empty state).
        true_state = self._extract_singleton_state(true_branch)
        if true_state is None:
            return None
        if not self._true_state_writes_carrier_from_array(true_state, carrier_name, input_array,
                                                          loop.loop_variable, sdfg):
            return None

        # Classify the carrier's storage.
        carrier_kind, carrier_subset = self._classify_carrier(carrier_name, sdfg)
        if carrier_kind is None:
            return None

        return _Match(
            op=op,
            loop=loop,
            parent=loop.parent_graph,
            carrier_name=carrier_name,
            carrier_kind=carrier_kind,
            carrier_subset=carrier_subset,
            input_array=input_array,
            iter_start=start,
            iter_end=end,
        )

    # ------------------------- match helpers -------------------------

    def _branch_has_content(self, branch) -> bool:
        if not hasattr(branch, 'nodes'):
            return False
        for n in branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return True
            if isinstance(n, (LoopRegion, ConditionalBlock)):
                return True
        return False

    def _resolve_tmp_iedge(self, loop: LoopRegion, cond_block: ConditionalBlock,
                           tmp_sym: str):
        """Walk in-edges of ``cond_block`` looking for one whose assignment binds
        ``tmp_sym`` to a comparison ``g OP c``. Returns ``(ast_op_cls, g_name,
        c_name)`` or ``(None, None, None)``."""
        for ie in loop.in_edges(cond_block):
            assigns = ie.data.assignments or {}
            rhs = assigns.get(tmp_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode='eval').body
            except SyntaxError:
                continue
            if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
                continue
            op_cls = type(tree.ops[0])
            if op_cls not in _CMP_AST_TO_RTYPE:
                continue
            lhs_name = self._extract_name(tree.left)
            rhs_name = self._extract_name(tree.comparators[0])
            if lhs_name is None or rhs_name is None:
                continue
            return op_cls, lhs_name, rhs_name
        return None, None, None

    def _resolve_gather_iedge(self, loop: LoopRegion, cond_block: ConditionalBlock,
                              gather_sym: str, loop_var: str, sdfg: SDFG) -> Optional[str]:
        """Walk back two levels to find an iedge binding ``gather_sym = arr[loop_var]``.
        Returns ``arr`` if found, else ``None``."""
        # Collect every iedge in the loop and try to find one binding ``gather_sym``.
        for e in loop.all_interstate_edges():
            assigns = e.data.assignments or {}
            rhs = assigns.get(gather_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode='eval').body
            except SyntaxError:
                continue
            if not isinstance(tree, ast.Subscript):
                continue
            if not isinstance(tree.value, ast.Name):
                continue
            arr = tree.value.id
            if arr not in sdfg.arrays:
                continue
            idx = tree.slice
            if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                idx = idx.value
            if not (isinstance(idx, ast.Name) and idx.id == loop_var):
                continue
            return arr
        return None

    def _extract_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _extract_singleton_state(self, branch) -> Optional[SDFGState]:
        if not hasattr(branch, 'nodes'):
            return None
        content_states = [n for n in branch.nodes() if isinstance(n, SDFGState) and len(n.nodes()) > 0]
        if len(content_states) != 1:
            return None
        return content_states[0]

    def _true_state_writes_carrier_from_array(self, state: SDFGState, carrier: str,
                                              array: str, loop_var: str,
                                              sdfg: SDFG) -> bool:
        """Check the true-branch state has the shape ``arr -> arr_index_AN ->
        assign_tasklet -> carrier_AN`` writing ``carrier = arr[loop_var]``."""
        # Single write AccessNode for the carrier.
        carrier_writes = [n for n in state.data_nodes()
                          if n.data == carrier and state.in_degree(n) > 0]
        if len(carrier_writes) != 1:
            return False
        carrier_an = carrier_writes[0]
        # **Strict refusal**: the true-branch may only have ONE terminal
        # AccessNode (the carrier). A second terminal AN means the conditional
        # updates an extra independent value (TSVC s315 ALSO writes ``index =
        # i``), and the v1 rewrite would silently drop it. Reject so the loop
        # stays sequential until index-tracking lift (Max_Location /
        # Min_Location) lands. Intermediate transients in the carrier chain
        # have ``out_degree > 0`` and are not counted here.
        terminal_outs = [n for n in state.data_nodes()
                         if state.in_degree(n) > 0 and state.out_degree(n) == 0]
        if len(terminal_outs) != 1 or terminal_outs[0] is not carrier_an:
            return False
        ins = list(state.in_edges(carrier_an))
        if len(ins) != 1:
            return False
        upstream = ins[0].src
        if not isinstance(upstream, nodes.Tasklet):
            return False
        # Walk through the tasklet to its input AccessNode that should subset ``array[loop_var]``.
        t_ins = list(state.in_edges(upstream))
        if len(t_ins) != 1:
            return False
        copy_an = t_ins[0].src
        if not isinstance(copy_an, nodes.AccessNode):
            return False
        copy_ins = list(state.in_edges(copy_an))
        if len(copy_ins) != 1:
            return False
        if not isinstance(copy_ins[0].src, nodes.AccessNode):
            return False
        if copy_ins[0].src.data != array:
            return False
        # Verify the memlet from the source array references ``[loop_var]``.
        subs = copy_ins[0].data.subset
        if subs is None:
            return False
        loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
        if not any(loop_var_sym in symbolic.pystr_to_symbolic(str(lo)).free_symbols
                   for lo, _, _ in subs.ranges):
            return False
        return True

    def _classify_carrier(self, name: str, sdfg: SDFG) -> Tuple[Optional[str], Optional[subsets.Range]]:
        desc = sdfg.arrays.get(name)
        if desc is None:
            return None, None
        if isinstance(desc, data.Scalar):
            return 'scalar', subsets.Range([(0, 0, 1)])
        if isinstance(desc, data.Array) and tuple(desc.shape) == (1,):
            return 'length_one_array', subsets.Range([(0, 0, 1)])
        return None, None

    # ------------------------- rewrite -------------------------

    def _rewrite(self, m: _Match, sdfg: SDFG):
        """Replace the loop with a :class:`Reduce` libnode."""
        arr_desc = sdfg.arrays[m.input_array]
        # Build the reduce-over subset: ``arr[start:end+1]`` on axis 0 (assuming 1-D arr).
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        # Reduce-state replaces the loop.
        reduce_state = m.parent.add_state(m.loop.label + '_argmax')
        # Re-route edges.
        for ie in list(m.parent.in_edges(m.loop)):
            m.parent.add_edge(ie.src, reduce_state, ie.data)
            m.parent.remove_edge(ie)
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(reduce_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        # Inputs / outputs.
        read = reduce_state.add_read(m.input_array)
        write = reduce_state.add_write(m.carrier_name)

        # ``Reduce`` libnode with identity=None: it seeds from the first read,
        # which matches the kernel's pre-loop ``x = a[start]`` semantics
        # (the first element of the input slice IS that seed).
        wcr_str = 'lambda a, b: max(a, b)' if m.op == dtypes.ReductionType.Max else 'lambda a, b: min(a, b)'
        node = Reduce(name=f'{m.loop.label}_argmax_reduce', wcr=wcr_str, axes=[0], identity=None)
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        reduce_state.add_node(node)
        # Input memlet: ``arr[start : end+1]``.
        end_plus_one = symbolic.simplify(end + 1)
        input_memlet = mm.Memlet(data=m.input_array,
                                 subset=subsets.Range([(start, end_plus_one - 1, 1)]))
        reduce_state.add_edge(read, None, node, '_in', input_memlet)
        # Output memlet: ``carrier[0]``.
        output_memlet = mm.Memlet(data=m.carrier_name, subset=m.carrier_subset)
        reduce_state.add_edge(node, '_out', write, None, output_memlet)
        sdfg.reset_cfg_list()


__all__ = ['ArgMaxLift']
