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

import numpy as np

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
    :param idx_carrier_name: The index carrier symbol (``index`` in s315), when
        the true-branch ALSO tracks the argmax/argmin position; ``None`` for the
        value-only shape. Only the symbol-carrier path supports it (the index is
        bound via an iedge, like the value carrier).
    :param gather_base: Constant term ``b`` of the affine gather index
        ``a[b + c*i]`` (``0`` for the plain ``a[i]`` gather).
    :param gather_coeff: Loop-variable coefficient ``c`` of the affine gather
        index (``1`` for the plain ``a[i]`` gather; a non-unit / symbolic ``c``
        is the strided gather of TSVC s318, where ``c = inc``).
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
    idx_carrier_name: Optional[str] = None
    transform: Optional[str] = None  # unary gather transform ('abs') or None
    gather_base: Any = 0
    gather_coeff: Any = 1


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
        # The comparison ``gather OP carrier`` reaches the ConditionalBlock in
        # one of two shapes, depending on how Simplify folded the body:
        #  (a) indirected -- the condition is a single symbol ``tmp`` bound by an
        #      upstream iedge ``tmp := (g OP c)``; or
        #  (b) inlined -- the comparison sits directly in the condition
        #      ``(g OP c)`` (current canonicalize output for TSVC s314/s316).
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', cond_expr_str):
            op_ast, gather_sym_name, carrier_name, transform = self._resolve_tmp_iedge(loop, cond_block, cond_expr_str)
        else:
            op_ast, gather_sym_name, carrier_name, transform = self._parse_comparison(cond_expr_str)
        if op_ast is None:
            return None
        op = _CMP_AST_TO_RTYPE[op_ast]

        # Walk one more iedge back to find the gather ``gather_sym = arr[b + c*i]``.
        gather = self._resolve_gather_iedge(loop, cond_block, gather_sym_name, loop.loop_variable, sdfg)
        if gather is None:
            return None
        input_array, gather_base, gather_coeff = gather
        # A non-unit / non-zero-base gather (``arr[inc*i]``, TSVC s318) is only
        # supported on the symbol-carrier transform+index path below; every
        # other shape assumes the plain unit ``arr[i]`` gather.
        is_unit_gather = bool(symbolic.simplify(gather_base) == 0 and symbolic.simplify(gather_coeff) == 1)

        # Classify the carrier's storage first; the body-write check differs
        # by case (scalar / length-1 array use state writes; symbol uses an
        # iedge inside the true-branch).
        carrier_kind, carrier_subset = self._classify_carrier(carrier_name, sdfg)
        if carrier_kind is None:
            return None

        idx_carrier_name = None
        if carrier_kind in ('scalar', 'length_one_array'):
            # Data-carrier path: the in-loop write lives on AccessNode chains;
            # the true-branch's iedges must NOT carry assignments (any iedge
            # assignment is an extra write -- e.g. TSVC s315's ``index = i``).
            # A gather transform (``abs``) is only handled on the symbol path.
            if transform is not None or not is_unit_gather:
                return None
            for e in true_branch.edges():
                if e.data.assignments:
                    return None
            true_state = self._extract_singleton_state(true_branch)
            if true_state is None:
                return None
            if not self._true_state_writes_carrier_from_array(true_state, carrier_name, input_array, loop.loop_variable,
                                                              sdfg):
                return None
        else:
            # Symbol-carrier path: the in-loop write is an iedge assignment
            # ``carrier := [f](arr[b+c*i])`` (or ``carrier := [f](gather_sym)``
            # where ``gather_sym`` was bound to ``arr[b+c*i]``) inside the
            # true-branch, OPTIONALLY plus an index carrier ``idx := loop_var``
            # (argmax position, s315). The transform ``f`` (e.g. ``abs``, s3113)
            # must match the one the comparison used. The true-branch states must
            # be empty -- any tasklet / AccessNode work would be a separate side
            # effect the rewrite cannot preserve.
            ok, idx_carrier_name = self._symbol_true_branch_writes_carrier(true_branch,
                                                                           carrier_name,
                                                                           input_array,
                                                                           gather_sym_name,
                                                                           loop.loop_variable,
                                                                           transform=transform)
            if not ok:
                return None
            # An index carrier must itself be a symbol (bound back via iedge).
            if idx_carrier_name is not None and idx_carrier_name not in sdfg.symbols:
                return None
            # A strided gather (s318) is ONLY lowered on the combined
            # transform+index path (``_rewrite_with_transform_and_index``), which
            # materialises ``buf[j] = f(a[b + c*(start-1+j)])`` then ArgReduces.
            # Every other symbol-carrier shape (value-only / index-only /
            # transform-only) assumes the unit ``arr[i]`` gather.
            has_transform_and_index = (transform is not None and idx_carrier_name is not None)
            if not is_unit_gather and not has_transform_and_index:
                return None
            # The strided combined path makes a load-bearing seed assumption
            # (``buf[0]`` stands in for the pre-loop seed at ``base+coeff*(start-1)``
            # and the index init is ``start-1``). Verify it so a mismatched seed
            # is refused rather than mis-lifted.
            if has_transform_and_index and not self._verify_affine_seed(loop, sdfg, carrier_name, idx_carrier_name,
                                                                        input_array, gather_base, gather_coeff, start,
                                                                        transform):
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
            idx_carrier_name=idx_carrier_name,
            transform=transform,
            gather_base=gather_base,
            gather_coeff=gather_coeff,
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

    def _parse_compare_node(self, tree):
        """Extract ``(op_cls, gather_name, carrier_name, transform)`` from a
        :class:`ast.Compare` node ``gather OP carrier`` (or ``f(gather) OP
        carrier``). The gather side may carry a recognised unary transform
        (``abs`` -> ``transform='abs'``); the carrier side must be a bare name.
        Returns ``(None, None, None, None)`` on any mismatch.
        """
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and len(tree.comparators) == 1):
            return None, None, None, None
        op_cls = type(tree.ops[0])
        if op_cls not in _CMP_AST_TO_RTYPE:
            return None, None, None, None
        transform, lhs_name = self._extract_transform(tree.left)
        rhs_name = self._extract_name(tree.comparators[0])
        if lhs_name is None or rhs_name is None:
            return None, None, None, None
        return op_cls, lhs_name, rhs_name, transform

    def _resolve_tmp_iedge(self, loop: LoopRegion, cond_block: ConditionalBlock, tmp_sym: str):
        """Walk in-edges of ``cond_block`` looking for one whose assignment binds
        ``tmp_sym`` to a comparison ``[f](g) OP c``. Returns ``(ast_op_cls,
        g_name, c_name, transform)`` or ``(None, None, None, None)``."""
        for ie in loop.in_edges(cond_block):
            assigns = ie.data.assignments or {}
            rhs = assigns.get(tmp_sym)
            if rhs is None:
                continue
            try:
                tree = ast.parse(str(rhs), mode='eval').body
            except SyntaxError:
                continue
            res = self._parse_compare_node(tree)
            if res[0] is not None:
                return res
        return None, None, None, None

    def _parse_comparison(self, expr_str: str):
        """Parse a comparison ``[f](g) OP c`` inlined directly in the condition.

        Mirrors :meth:`_resolve_tmp_iedge` but on the condition string itself
        (no ``tmp`` iedge indirection). Returns ``(ast_op_cls, g_name, c_name,
        transform)`` or ``(None, None, None, None)``.
        """
        try:
            tree = ast.parse(expr_str, mode='eval').body
        except SyntaxError:
            return None, None, None, None
        return self._parse_compare_node(tree)

    def _resolve_gather_iedge(self, loop: LoopRegion, cond_block: ConditionalBlock, gather_sym: str, loop_var: str,
                              sdfg: SDFG) -> Optional[Tuple[str, Any, Any]]:
        """Walk back two levels to find an iedge binding ``gather_sym = arr[idx]``
        where ``idx`` is an AFFINE function of the loop variable ``b + c*i``.

        Returns ``(arr, base, coeff)`` -- ``base`` (``b``) and ``coeff`` (``c``)
        are sympy expressions, both loop-invariant. The plain ``arr[i]`` gather
        yields ``(arr, 0, 1)``; the strided gather ``arr[inc*i]`` (TSVC s318,
        after ``InductionVariableSubstitution`` closes the secondary IV ``k``)
        yields ``(arr, 0, inc)``. Returns ``None`` if no such iedge exists.
        """
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
            # Only a single 1-D affine index is handled; a multi-dim subscript
            # ``a[i, j]`` (ast.Tuple / Slice) is refused (the 2-D argmax of TSVC
            # s3110 / s13110 is out of scope).
            if isinstance(idx, (ast.Tuple, ast.Slice, ast.List)):
                continue
            try:
                idx_str = ast.unparse(idx)
            except Exception:  # pragma: no cover -- unparse failure is defensive
                continue
            aff = self._affine_index_in_loop_var(idx_str, loop_var, loop)
            if aff is None:
                continue
            base, coeff = aff
            return arr, base, coeff
        return None

    def _affine_index_in_loop_var(self, idx_str: str, loop_var: str, loop: LoopRegion):
        """Decompose a gather index ``idx_str`` as ``base + coeff*loop_var``.

        Returns ``(base, coeff)`` (sympy exprs) iff ``idx_str`` is affine and
        linear in ``loop_var`` AND both ``base`` and ``coeff`` are loop-invariant
        (free of the loop variable and of any symbol reassigned on a body iedge).
        The loop variable MUST appear (a constant index is not a per-iteration
        gather). Returns ``None`` on any mismatch -- the conservative default,
        so a non-affine / loop-carried index is never mis-lifted.
        """
        try:
            idx = symbolic.pystr_to_symbolic(idx_str)
            lv = symbolic.pystr_to_symbolic(loop_var)
        except Exception:  # pragma: no cover -- defensive parse guard
            return None
        if lv not in idx.free_symbols:
            return None
        # Linear-in-``lv`` decomposition: idx == base + coeff*lv with neither
        # term referencing ``lv`` (so no ``lv**2`` etc.). Expand first so
        # ``.coeff`` sees the loop var inside products like ``inc*(i-1)`` (the
        # closed form ``InductionVariableSubstitution`` leaves -- otherwise
        # ``.coeff`` returns 0 for the unexpanded form and the affine match fails).
        idx = idx.expand()
        coeff = idx.coeff(lv, 1)
        base = idx.coeff(lv, 0)
        if symbolic.simplify(idx - (base + coeff * lv)) != 0:
            return None
        if lv in coeff.free_symbols or lv in base.free_symbols:
            return None
        # Every symbol feeding base / coeff must be loop-invariant: not assigned
        # on any body interstate edge (a varying stride/base breaks the closed form).
        body_assigned = set()
        for e in loop.all_interstate_edges():
            body_assigned.update((e.data.assignments or {}).keys())
        for s in set(coeff.free_symbols) | set(base.free_symbols):
            if str(s) in body_assigned:
                return None
        return base, coeff

    def _extract_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        return None

    #: Recognised unary gather transforms ``f(g)`` -> the Python builtin name.
    _SUPPORTED_TRANSFORMS = {'abs'}

    def _extract_transform(self, node) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(transform, name)`` for a possibly-transformed operand.

        A bare ``Name`` ``g`` -> ``(None, 'g')``; a recognised unary call
        ``abs(g)`` -> ``('abs', 'g')``; anything else -> ``(None, None)``.
        """
        if isinstance(node, ast.Name):
            return None, node.id
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in self._SUPPORTED_TRANSFORMS and len(node.args) == 1 and not node.keywords):
            inner = self._extract_name(node.args[0])
            if inner is not None:
                return node.func.id, inner
        return None, None

    def _extract_singleton_state(self, branch) -> Optional[SDFGState]:
        if not hasattr(branch, 'nodes'):
            return None
        content_states = [n for n in branch.nodes() if isinstance(n, SDFGState) and len(n.nodes()) > 0]
        if len(content_states) != 1:
            return None
        return content_states[0]

    def _true_state_writes_carrier_from_array(self, state: SDFGState, carrier: str, array: str, loop_var: str,
                                              sdfg: SDFG) -> bool:
        """Check the true-branch state has the shape ``arr -> arr_index_AN ->
        assign_tasklet -> carrier_AN`` writing ``carrier = arr[loop_var]``."""
        # Single write AccessNode for the carrier.
        carrier_writes = [n for n in state.data_nodes() if n.data == carrier and state.in_degree(n) > 0]
        if len(carrier_writes) != 1:
            return False
        carrier_an = carrier_writes[0]
        # **Strict refusal**: the true-branch may only have ONE terminal
        # AccessNode (the carrier). A second terminal AN means the conditional
        # updates an extra independent value (TSVC s315 ALSO writes ``index =
        # i``), and the v1 rewrite would silently drop it. Intermediate
        # transients in the carrier chain have ``out_degree > 0`` and are not
        # counted here.
        terminal_outs = [n for n in state.data_nodes() if state.in_degree(n) > 0 and state.out_degree(n) == 0]
        if len(terminal_outs) != 1 or terminal_outs[0] is not carrier_an:
            return False
        # Walk back through the chain (AN <- [Tasklet?] <- AN <- ... <- AN(array)).
        # The Python frontend emits an identity ``__out = __inp`` assign tasklet;
        # ``TrivialTaskletElimination`` strips it. Both shapes are accepted.
        source_an = self._walk_back_to_source(state, carrier_an)
        if source_an is None or source_an.data != array:
            return False
        # Verify the memlet from the source array references ``[loop_var]``.
        # Walk forward one edge from source to find the gather memlet.
        out_edges = list(state.out_edges(source_an))
        if not out_edges:
            return False
        loop_var_sym = symbolic.pystr_to_symbolic(loop_var)
        for oe in out_edges:
            if oe.data is None or oe.data.subset is None:
                continue
            if any(loop_var_sym in symbolic.pystr_to_symbolic(str(lo)).free_symbols
                   for lo, _, _ in oe.data.subset.ranges):
                return True
        return False

    def _walk_back_to_source(self, state: SDFGState, carrier_an: nodes.AccessNode) -> Optional[nodes.AccessNode]:
        """Walk back from ``carrier_an`` through a chain of transients and
        identity Tasklets to the source AccessNode (the array we're reading
        from). Returns the source AN, or ``None`` if the chain doesn't form
        a single-source linear path.
        """
        cur = carrier_an
        while True:
            ins = list(state.in_edges(cur))
            if len(ins) != 1:
                return None
            upstream = ins[0].src
            if isinstance(upstream, nodes.Tasklet):
                t_ins = list(state.in_edges(upstream))
                if len(t_ins) != 1:
                    return None
                upstream = t_ins[0].src
                if not isinstance(upstream, nodes.AccessNode):
                    return None
            elif not isinstance(upstream, nodes.AccessNode):
                return None
            # ``upstream`` is now an AccessNode. If it has no in-edges it's the source.
            if state.in_degree(upstream) == 0:
                return upstream
            cur = upstream

    def _classify_carrier(self, name: str, sdfg: SDFG) -> Tuple[Optional[str], Optional[subsets.Range]]:
        desc = sdfg.arrays.get(name)
        if desc is not None:
            if isinstance(desc, data.Scalar):
                return 'scalar', subsets.Range([(0, 0, 1)])
            if isinstance(desc, data.Array) and tuple(desc.shape) == (1, ):
                return 'length_one_array', subsets.Range([(0, 0, 1)])
            return None, None
        # Symbol carrier: present in ``sdfg.symbols`` but not in ``sdfg.arrays``.
        if name in sdfg.symbols:
            return 'symbol', None
        return None, None

    def _rhs_is_value_write(self, rhs_str: str, gather_sym: str, array: str, loop_var: str,
                            transform: Optional[str]) -> bool:
        """True iff ``rhs_str`` is the value-carrier write under ``transform``:
        ``[f](gather_sym)`` or ``[f](array[loop_var])``, where ``f`` is the
        recognised transform (``None`` -> no wrapping call allowed)."""
        try:
            tree = ast.parse(rhs_str, mode='eval').body
        except SyntaxError:
            return False
        if transform is not None:
            if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == transform
                    and len(tree.args) == 1 and not tree.keywords):
                return False
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return False  # an unexpected transform when none was matched
        if isinstance(tree, ast.Name):
            return tree.id == gather_sym
        if isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array:
            idx = tree.slice
            if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
                idx = idx.value
            return isinstance(idx, ast.Name) and idx.id == loop_var
        return False

    def _symbol_true_branch_writes_carrier(self,
                                           true_branch,
                                           carrier: str,
                                           array: str,
                                           gather_sym: str,
                                           loop_var: str,
                                           transform: Optional[str] = None):
        """For the symbol-carrier case, verify the true-branch binds the value
        carrier (``carrier := [f](array[loop_var])`` or ``carrier :=
        [f](gather_sym)``, with the same gather transform ``f`` the comparison
        used) and, optionally, ONE index carrier (``idx := loop_var`` -- the
        argmax position, TSVC s315). The true-branch states must all be empty
        (no tasklet / AccessNode work). Any other iedge assignment is refused.

        :returns: ``(ok, idx_carrier)`` -- ``ok`` is True iff the value carrier
            write was found and every assignment was recognised; ``idx_carrier``
            is the index-carrier symbol name when an ``idx := loop_var`` write is
            present, else ``None``.
        """
        if not hasattr(true_branch, 'edges'):
            return False, None
        carrier_write_seen = False
        idx_carrier = None
        for e in true_branch.edges():
            assigns = e.data.assignments or {}
            for lhs, rhs in assigns.items():
                rhs_str = str(rhs).strip()
                if lhs == carrier:
                    if not self._rhs_is_value_write(rhs_str, gather_sym, array, loop_var, transform):
                        return False, None
                    carrier_write_seen = True
                elif rhs_str == loop_var and idx_carrier is None:
                    # Index carrier := the loop variable (argmax position).
                    idx_carrier = lhs
                else:
                    return False, None  # unrecognised extra write -> refuse
        # All true-branch states must be empty -- any contained tasklet /
        # AccessNode work is a separate side effect.
        for n in true_branch.nodes():
            if isinstance(n, SDFGState) and len(n.nodes()) > 0:
                return False, None
            if not isinstance(n, SDFGState):
                return False, None  # nested control flow not supported in v1
        return carrier_write_seen, idx_carrier

    def _collect_preloop_assignments(self, loop: LoopRegion) -> dict:
        """Walk back the linear pre-loop chain (each block reached by a single
        in-edge) and accumulate interstate-edge assignments. The binding closest
        to the loop wins for each name (it is the value live at loop entry).

        The seed of a TSVC argmax is spread over this chain -- e.g.
        ``block: a_index := a[0]; index := 0`` then ``-> loop: maxv := abs(a_index)``.
        """
        preloop: dict = {}
        parent = loop.parent_graph
        cur = loop
        seen = set()
        while True:
            ins = parent.in_edges(cur)
            if len(ins) != 1:
                break
            e = ins[0]
            for lhs, rhs in (e.data.assignments or {}).items():
                if lhs not in preloop:  # closest-to-loop wins
                    preloop[lhs] = str(rhs)
            if e.src in seen or not isinstance(e.src, (SDFGState, ControlFlowRegion)):
                break
            seen.add(e.src)
            cur = e.src
        return preloop

    def _verify_affine_seed(self, loop: LoopRegion, sdfg: SDFG, value_carrier: str, idx_carrier: str, array: str,
                            base: Any, coeff: Any, start: Any, transform: Optional[str]) -> bool:
        """For the strided transform+index path, verify the pre-loop seed is
        consistent with the buffer the rewrite builds: the value carrier must be
        seeded ``value_carrier := [f](array[Q])`` with ``Q`` equal to the gather's
        seed-iteration position ``base + coeff*(start-1)``, and the index carrier
        ``idx_carrier := (start-1)``.

        This is the load-bearing assumption of
        :meth:`_rewrite_with_transform_and_index` -- the buffer's first element
        ``buf[0] = f(a[base + coeff*(start-1)])`` stands in for the seed, and the
        index bind ``idx_carrier := (start-1) + idx_buf`` yields the seed's init
        index when the seed wins. A loop whose seed sits elsewhere (or whose
        index init != start-1) would be mis-lifted, so refuse it.

        Handles the real frontend shape, where the seed is spread over a pre-loop
        chain with indirection: ``base`` / ``coeff`` carry the secondary-IV symbol
        ``k`` (bound pre-loop to ``inc``), and the value seed is ``maxv :=
        abs(a_index)`` with ``a_index := a[0]`` on an earlier edge. Both the
        position comparison and the gather lookup substitute the chain's bindings.
        """
        preloop = self._collect_preloop_assignments(loop)
        if value_carrier not in preloop or idx_carrier not in preloop:
            return False

        # Pure-symbol pre-loop bindings (skip carriers + array-read bindings like
        # ``a_index := a[0]``) -- these resolve the closed form's IV symbol ``k``.
        subs = {}
        for lhs, rhs in preloop.items():
            if lhs in (value_carrier, idx_carrier):
                continue
            try:
                expr = symbolic.pystr_to_symbolic(rhs)
            except Exception:  # pragma: no cover -- defensive
                continue
            if any(str(s) in sdfg.arrays for s in expr.free_symbols):
                continue  # an array-read binding, not a scalar symbol value
            subs[symbolic.pystr_to_symbolic(lhs)] = expr

        def _resolve(expr):
            try:
                return symbolic.simplify(symbolic.pystr_to_symbolic(str(expr)).subs(subs))
            except Exception:  # pragma: no cover -- defensive
                return None

        seed_pos = _resolve(base + coeff * (start - 1))
        seed_idx = _resolve(start - 1)
        if seed_pos is None or seed_idx is None:
            return False

        # Value seed: ``[f](array[Q])``, possibly indirected through a gather
        # symbol bound on an earlier pre-loop edge (``maxv := abs(a_index)``).
        q = self._seed_gather_position(preloop[value_carrier], array, transform, preloop)
        if q is None:
            return False
        q_resolved = _resolve(q)
        if q_resolved is None or symbolic.simplify(q_resolved - seed_pos) != 0:
            return False
        # Index init must equal start-1.
        idx_resolved = _resolve(preloop[idx_carrier])
        if idx_resolved is None or symbolic.simplify(idx_resolved - seed_idx) != 0:
            return False
        return True

    def _seed_gather_position(self, rhs_str: str, array: str, transform: Optional[str], preloop: dict):
        """If ``rhs_str`` is ``[f](array[P])`` -- or ``[f](g)`` where ``g`` is a
        pre-loop symbol bound to ``array[P]`` -- return the index string ``P``;
        else ``None``."""
        try:
            tree = ast.parse(rhs_str, mode='eval').body
        except SyntaxError:
            return None
        if transform is not None:
            if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == transform
                    and len(tree.args) == 1 and not tree.keywords):
                return None
            tree = tree.args[0]
        elif isinstance(tree, ast.Call):
            return None
        # Follow one level of gather-symbol indirection (``g`` bound to ``a[P]``).
        if isinstance(tree, ast.Name) and tree.id in preloop:
            try:
                tree = ast.parse(preloop[tree.id], mode='eval').body
            except SyntaxError:
                return None
        if not (isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id == array):
            return None
        idx = tree.slice
        if isinstance(idx, ast.Index):  # pragma: no cover -- legacy AST
            idx = idx.value
        try:
            return ast.unparse(idx)
        except Exception:  # pragma: no cover -- defensive
            return None

    # ------------------------- rewrite -------------------------

    def _rewrite(self, m: _Match, sdfg: SDFG):
        """Replace the loop with a :class:`Reduce` (value-only) or
        :class:`ArgReduce` (value + index) libnode."""
        if m.transform is not None and m.idx_carrier_name is not None:
            return self._rewrite_with_transform_and_index(m, sdfg)
        if m.transform is not None:
            return self._rewrite_with_transform(m, sdfg)
        if m.idx_carrier_name is not None:
            return self._rewrite_with_index(m, sdfg)
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)

        # Allocate the output container per carrier kind.
        if m.carrier_kind == 'symbol':
            # Fresh transient scalar -> Reduce output -> iedge bind to the symbol.
            output_dtype = sdfg.symbols[m.carrier_name]
            out_name, _ = sdfg.add_scalar(f'_arg_max_buf_{m.loop.label}',
                                          output_dtype,
                                          transient=True,
                                          find_new_name=True)
            output_subset = subsets.Range([(0, 0, 1)])
        else:
            out_name = m.carrier_name
            output_subset = m.carrier_subset

        # Reduce-state replaces the loop.
        reduce_state = m.parent.add_state(m.loop.label + '_argmax')
        # Re-route inbound edges. For symbol carriers, drop any pre-loop iedge
        # assignment that binds the carrier symbol -- the reduce subsumes it.
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            if m.carrier_kind == 'symbol' and m.carrier_name in new_assigns:
                del new_assigns[m.carrier_name]
            new_iedge = dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns)
            m.parent.add_edge(ie.src, reduce_state, new_iedge)
            m.parent.remove_edge(ie)

        if m.carrier_kind == 'symbol':
            # Insert a bind state AFTER the reduce so the carrier symbol gets
            # re-materialised from the transient scalar before any downstream
            # state references it. Scalars are bare names (no subscript) in
            # iedge assignment RHS expressions.
            bind_state = m.parent.add_state(m.loop.label + '_argmax_bind')
            m.parent.add_edge(reduce_state, bind_state, dace.InterstateEdge(assignments={m.carrier_name: out_name}))
            for oe in list(m.parent.out_edges(m.loop)):
                m.parent.add_edge(bind_state, oe.dst, oe.data)
                m.parent.remove_edge(oe)
        else:
            for oe in list(m.parent.out_edges(m.loop)):
                m.parent.add_edge(reduce_state, oe.dst, oe.data)
                m.parent.remove_edge(oe)

        m.parent.remove_node(m.loop)

        # Inputs / outputs for the libnode.
        read = reduce_state.add_read(m.input_array)
        write = reduce_state.add_write(out_name)

        # For scalar / length-1 array carriers the pre-loop init ``x = a[0]``
        # is already in a prior state writing to the same AccessNode; the
        # libnode's ``identity=None`` semantics fold that pre-existing value
        # into the running reduction (WCR-Max). For symbol carriers we have
        # to include the seed position explicitly in the input slice because
        # the dropped pre-loop iedge no longer materialises the seed.
        wcr_str = 'lambda a, b: max(a, b)' if m.op == dtypes.ReductionType.Max else 'lambda a, b: min(a, b)'
        # Identity (accumulator seed) for the reduction.
        #  * symbol carrier -> the Reduce writes a FRESH transient with no
        #    pre-seeded value (and the input slice already covers the original
        #    seed ``a[start-1]``). ``identity=None`` makes the pure expansion
        #    default the accumulator to 0, which is correct only for a Max over
        #    non-negative data -- it silently corrupts a Min (``min(0, positives)
        #    == 0``) and a Max over all-negative data. Use the proper neutral
        #    element: the dtype's most-negative value for Max, most-positive for
        #    Min (finite extremes, so codegen stays a plain numeric literal).
        #  * scalar / length-1 carrier -> keep ``identity=None``: it WCR-folds
        #    into the pre-loop ``x = a[start]`` seed already in the carrier AN.
        if m.carrier_kind == 'symbol':
            _nt = sdfg.arrays[m.input_array].dtype.type
            if np.issubdtype(_nt, np.floating):
                _info = np.finfo(_nt)
            else:
                _info = np.iinfo(_nt)
            identity = (_info.min if m.op == dtypes.ReductionType.Max else _info.max).item()
        else:
            identity = None
        node = Reduce(name=f'{m.loop.label}_argmax_reduce', wcr=wcr_str, axes=[0], identity=identity)
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        reduce_state.add_node(node)
        if m.carrier_kind == 'symbol':
            # Extend the slice down to ``start - 1`` so a[start - 1] (the seed)
            # is included in the reduction. (TSVC s314 init reads ``a[0]`` for
            # ``start = 1``; same shape generalised.)
            slice_lo = symbolic.simplify(start - 1)
            if slice_lo < 0:
                slice_lo = symbolic.simplify(0)
        else:
            slice_lo = start
        input_memlet = mm.Memlet(data=m.input_array, subset=subsets.Range([(slice_lo, end, 1)]))
        reduce_state.add_edge(read, None, node, '_in', input_memlet)
        output_memlet = mm.Memlet(data=out_name, subset=output_subset)
        reduce_state.add_edge(node, '_out', write, None, output_memlet)
        sdfg.reset_cfg_list()

    def _rewrite_with_index(self, m: _Match, sdfg: SDFG):
        """Replace an argmax/argmin-with-index loop (TSVC s315) with an
        :class:`~dace.libraries.standard.nodes.ArgReduce` libnode.

        The lift mirrors the symbol-carrier value-only path but uses the
        two-output ``ArgReduce`` (value + index). Both outputs are fresh
        transient SCALARS -- ``val_buf`` (the array's dtype) and ``idx_buf``
        (``int64``) -- bound back to the carrier symbols after the reduce:
        ``carrier := val_buf`` and ``idx_carrier := slice_lo + idx_buf`` (the
        ``ArgReduce`` index is slice-local; ``slice_lo`` recovers the
        original-array position). The pre-loop seed iedges binding either
        carrier are dropped -- the reduce subsumes them (the seed position is
        kept by extending the input slice down to ``start - 1``).
        """
        from dace.libraries.standard.nodes import ArgReduce
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        val_buf, _ = sdfg.add_scalar(f'_argmax_val_{m.loop.label}', arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argmax_idx_{m.loop.label}', dtypes.int64, transient=True, find_new_name=True)

        # Include the seed ``a[start-1]`` in the slice (clamped at 0).
        slice_lo = symbolic.simplify(start - 1)
        try:
            if slice_lo < 0:
                slice_lo = symbolic.simplify(0)
        except TypeError:  # symbolic start; assume the seed sits at >= 0
            pass
        lo_is_zero = bool(symbolic.simplify(slice_lo) == 0)

        argmax_state = m.parent.add_state(m.loop.label + '_argreduce')
        # Re-route inbound edges, dropping pre-loop binds of BOTH carriers.
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(m.idx_carrier_name, None)
            m.parent.add_edge(ie.src, argmax_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)

        # Bind state: re-materialise both carrier symbols from the scalars
        # (bare names; the index adds the slice base when the slice is offset).
        idx_rhs = idx_buf if lo_is_zero else f'({symbolic.symstr(slice_lo)} + {idx_buf})'
        bind_state = m.parent.add_state(m.loop.label + '_argreduce_bind')
        m.parent.add_edge(argmax_state, bind_state,
                          dace.InterstateEdge(assignments={
                              m.carrier_name: val_buf,
                              m.idx_carrier_name: idx_rhs
                          }))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        read = argmax_state.add_read(m.input_array)
        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = 'max' if m.op == dtypes.ReductionType.Max else 'min'
        node = ArgReduce(name=f'{m.loop.label}_argreduce', op=op)
        argmax_state.add_node(node)
        argmax_state.add_edge(read, None, node, '_in',
                              mm.Memlet(data=m.input_array, subset=subsets.Range([(slice_lo, end, 1)])))
        argmax_state.add_edge(node, '_out_val', wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, '_out_idx', wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()

    def _rewrite_with_transform(self, m: _Match, sdfg: SDFG):
        """Replace a transformed value-only reduction (TSVC s3113,
        ``maxv = max(|a[i]|)``) with: a map materialising ``buf[j] = f(a[lo+j])``
        into a fresh contiguous transient, then a :class:`Reduce` over ``buf``.

        Only the value-only symbol-carrier shape is handled here (the unit
        ``a[i]`` gather); transform + index (TSVC s318, a strided gather) is
        lowered by :meth:`_rewrite_with_transform_and_index`. The transform
        ``f`` is applied per element in the materialisation map, so the
        reduction itself stays a plain Max/Min fold over a unit-stride buffer.
        """
        from dace.codegen.targets.cpp import sym2cpp
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype

        # Include the seed ``a[start-1]`` (clamped at 0); buf spans the slice.
        slice_lo = symbolic.simplify(start - 1)
        try:
            if slice_lo < 0:
                slice_lo = symbolic.simplify(0)
        except TypeError:  # symbolic start; assume the seed sits at >= 0
            pass
        n_elems = symbolic.simplify(end + 1 - slice_lo)

        buf, _ = sdfg.add_array(f'_argf_buf_{m.loop.label}', [n_elems], arr_dtype, transient=True, find_new_name=True)

        # Materialisation state: buf[_j] = f(input_array[slice_lo + _j]).
        mat_state = m.parent.add_state(m.loop.label + '_argf')
        mat_state.add_mapped_tasklet(
            name='transform_gather',
            map_ranges={'_j': f'0:{sym2cpp(n_elems)}'},
            inputs={'__in': mm.Memlet(data=m.input_array, subset=f'{sym2cpp(slice_lo)} + _j')},
            code=f'__out = {m.transform}(__in)',
            outputs={'__out': mm.Memlet(data=buf, subset='_j')},
            external_edges=True,
        )

        # Reduce over the (contiguous) buffer.
        out_dtype = sdfg.symbols[m.carrier_name]
        out_name, _ = sdfg.add_scalar(f'_argf_val_{m.loop.label}', out_dtype, transient=True, find_new_name=True)
        reduce_state = m.parent.add_state(m.loop.label + '_argf_reduce')

        # Re-route inbound edges to the materialise state, dropping the pre-loop
        # carrier seed (the reduce over buf subsumes it).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            m.parent.add_edge(ie.src, mat_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        m.parent.add_edge(mat_state, reduce_state, dace.InterstateEdge())

        # Bind state: carrier := reduced value.
        bind_state = m.parent.add_state(m.loop.label + '_argf_bind')
        m.parent.add_edge(reduce_state, bind_state, dace.InterstateEdge(assignments={m.carrier_name: out_name}))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        read = reduce_state.add_read(buf)
        write = reduce_state.add_write(out_name)
        wcr_str = 'lambda a, b: max(a, b)' if m.op == dtypes.ReductionType.Max else 'lambda a, b: min(a, b)'
        _info = np.finfo(arr_dtype.type) if np.issubdtype(arr_dtype.type, np.floating) else np.iinfo(arr_dtype.type)
        identity = (_info.min if m.op == dtypes.ReductionType.Max else _info.max).item()
        node = Reduce(name=f'{m.loop.label}_argf_reduce', wcr=wcr_str, axes=[0], identity=identity)
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        reduce_state.add_node(node)
        reduce_state.add_edge(read, None, node, '_in',
                              mm.Memlet(data=buf, subset=subsets.Range([(0, symbolic.simplify(n_elems - 1), 1)])))
        reduce_state.add_edge(node, '_out', write, None, mm.Memlet(data=out_name, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()

    def _rewrite_with_transform_and_index(self, m: _Match, sdfg: SDFG):
        """Replace a transformed argmax/argmin-WITH-INDEX over a (possibly
        strided) gather -- TSVC s318, ``maxv = max(|a[k]|)`` with ``k = inc*i``
        and ``index`` tracking the iteration of the max -- with: a map
        materialising ``buf[j] = f(a[pos_lo + coeff*j])`` into a fresh contiguous
        transient, then an :class:`~dace.libraries.standard.nodes.ArgReduce`
        over ``buf`` yielding the value + the slice-local index.

        Combines the abs-transform materialisation (:meth:`_rewrite_with_transform`)
        with the index-tracking ArgReduce (:meth:`_rewrite_with_index`). The
        gather is affine ``a[gather_base + gather_coeff*i]``; element ``j`` of the
        contiguous buffer maps to iteration ``i = (start-1) + j`` (the seed sits
        at ``i = start-1``, where ``index`` holds its init value). Hence the
        materialised position for ``buf[j]`` is ``pos_lo + coeff*j`` with
        ``pos_lo = base + coeff*(start-1)``, and the recovered iteration index is
        ``index := (start-1) + idx_buf`` (``idx_buf`` is slice-local). Both
        carrier seeds are dropped from the inbound iedges -- the buffer's first
        element subsumes them.
        """
        from dace.codegen.targets.cpp import sym2cpp
        from dace.libraries.standard.nodes import ArgReduce
        start = symbolic.simplify(m.iter_start)
        end = symbolic.simplify(m.iter_end)
        arr_dtype = sdfg.arrays[m.input_array].dtype
        base, coeff = m.gather_base, m.gather_coeff

        # Iterations covered, INCLUDING the seed at ``i = start-1`` (where the
        # index carrier still holds its init value). ``j`` ranges 0..n-1 over
        # ``i = iter_lo + j``.
        iter_lo = symbolic.simplify(start - 1)
        try:
            if iter_lo < 0:
                iter_lo = symbolic.simplify(0)
        except TypeError:  # symbolic start; assume the seed sits at >= 0
            pass
        n_elems = symbolic.simplify(end - iter_lo + 1)
        # Array position of ``buf[j]``: pos(i) = base + coeff*i, i = iter_lo + j.
        pos_lo = symbolic.simplify(base + coeff * iter_lo)

        buf, _ = sdfg.add_array(f'_argfi_buf_{m.loop.label}', [n_elems], arr_dtype, transient=True, find_new_name=True)
        val_buf, _ = sdfg.add_scalar(f'_argfi_val_{m.loop.label}', arr_dtype, transient=True, find_new_name=True)
        idx_buf, _ = sdfg.add_scalar(f'_argfi_idx_{m.loop.label}', dtypes.int64, transient=True, find_new_name=True)

        # Materialisation: buf[_j] = f(input_array[pos_lo + coeff*_j]).
        mat_state = m.parent.add_state(m.loop.label + '_argfi')
        in_subset = f'({sym2cpp(pos_lo)}) + ({sym2cpp(coeff)}) * _j'
        mat_state.add_mapped_tasklet(
            name='transform_gather_idx',
            map_ranges={'_j': f'0:{sym2cpp(n_elems)}'},
            inputs={'__in': mm.Memlet(data=m.input_array, subset=in_subset)},
            code=f'__out = {m.transform}(__in)',
            outputs={'__out': mm.Memlet(data=buf, subset='_j')},
            external_edges=True,
        )

        argmax_state = m.parent.add_state(m.loop.label + '_argfi_reduce')
        # Re-route inbound edges to the materialise state, dropping the pre-loop
        # binds of BOTH carriers (the ArgReduce over buf subsumes them).
        for ie in list(m.parent.in_edges(m.loop)):
            new_assigns = dict(ie.data.assignments or {})
            new_assigns.pop(m.carrier_name, None)
            new_assigns.pop(m.idx_carrier_name, None)
            m.parent.add_edge(ie.src, mat_state,
                              dace.InterstateEdge(condition=ie.data.condition, assignments=new_assigns))
            m.parent.remove_edge(ie)
        m.parent.add_edge(mat_state, argmax_state, dace.InterstateEdge())

        # Bind state: value carrier := val_buf; index carrier := iter_lo + idx_buf
        # (idx_buf is the slice-local position; iter_lo recovers the iteration).
        lo_is_zero = bool(symbolic.simplify(iter_lo) == 0)
        idx_rhs = idx_buf if lo_is_zero else f'({symbolic.symstr(iter_lo)} + {idx_buf})'
        bind_state = m.parent.add_state(m.loop.label + '_argfi_bind')
        m.parent.add_edge(argmax_state, bind_state,
                          dace.InterstateEdge(assignments={
                              m.carrier_name: val_buf,
                              m.idx_carrier_name: idx_rhs
                          }))
        for oe in list(m.parent.out_edges(m.loop)):
            m.parent.add_edge(bind_state, oe.dst, oe.data)
            m.parent.remove_edge(oe)
        m.parent.remove_node(m.loop)

        read = argmax_state.add_read(buf)
        wv = argmax_state.add_write(val_buf)
        wi = argmax_state.add_write(idx_buf)
        op = 'max' if m.op == dtypes.ReductionType.Max else 'min'
        node = ArgReduce(name=f'{m.loop.label}_argfi_argreduce', op=op)
        argmax_state.add_node(node)
        argmax_state.add_edge(read, None, node, '_in',
                              mm.Memlet(data=buf, subset=subsets.Range([(0, symbolic.simplify(n_elems - 1), 1)])))
        argmax_state.add_edge(node, '_out_val', wv, None, mm.Memlet(data=val_buf, subset=subsets.Range([(0, 0, 1)])))
        argmax_state.add_edge(node, '_out_idx', wi, None, mm.Memlet(data=idx_buf, subset=subsets.Range([(0, 0, 1)])))
        sdfg.reset_cfg_list()


__all__ = ['ArgMaxLift']
