# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop to map transformation """

from collections import defaultdict
import copy
import sympy as sp
from typing import Dict, List, Set
import warnings

from dace import data as dt, dtypes, memlet, nodes, sdfg as sd, symbolic, subsets, properties
from dace.sdfg.type_inference import infer_expr_type
from dace.sdfg import graph as gr, nodes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import BreakBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ConditionalBlock
import dace.transformation.helpers as helpers
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


def _check_range(subset, a, itersym, b, step):
    found = False
    for rb, re, _ in subset.ndrange():
        if rb != 0:
            m = rb.match(a * itersym + b)
            if m is None:
                continue
            if (abs(m[a]) >= 1) != True:
                continue
        else:
            m = re.match(a * itersym + b)
            if m is None:
                continue
            if (abs(m[a]) >= 1) != True:
                continue
        found = True
        break
    return found


def _nested_writes_iter_indexed(nsdfg_node, conn, itersym, a, b, step) -> bool:
    """Whether every write to ``conn``'s array *inside* ``nsdfg_node`` is
    indexed by the (mapped) iteration variable.

    A loop body that is a ``NestedSDFG`` propagates a whole-array external
    write memlet (the union over the loop), which hides a per-iteration
    write. This looks past the connector: the inner write subsets are
    rewritten through the node's ``symbol_mapping`` into the outer iteration
    symbol and each must match the same ``a*i+b`` pattern
    :func:`_check_range` enforces. Conservative: requires at least one inner
    write to the array and that *all* of them pass (nested ``NestedSDFG`` s
    are checked recursively, composing the symbol maps).

    Example -- ``for i: if c: b[i] = a[i] + 1`` after a
    ``LoopToMap -> MapToForLoop`` round-trip (the guard forced a
    ``NestedSDFG`` body)::

        for i in 0:N:                      # the loop being re-parallelized
          state:
            a ──► [ NestedSDFG loop_body ] ──► b
                        symbol_mapping {i: i, N: N}
                        │  external write connector memlet:  b[0:N]
                        │  (correct union over the loop -- has no `i`,
                        │   so _check_range(b[0:N]) FAILS -> refuse)
                        └─ inner: if (c):
                                    b[i] = a[i] + 1.0     ◄── real per-
                                                              iteration write

        _nested_writes_iter_indexed walks inside loop_body, finds the inner
        write ``b[i]``, maps it through symbol_mapping ({i: i}) to the outer
        ``b[i]``, and _check_range matches ``1*i + 0`` -> independence
        proven -> LoopToMap fires (the round-trip recovers the map).

    :param nsdfg_node: The ``NestedSDFG`` node feeding the outer write.
    :param conn: The output connector (== inner array name) being written.
    :param itersym: The outer loop iteration symbol.
    :returns: ``True`` iff every inner write to ``conn`` is iter-indexed.
    """
    repl = {symbolic.symbol(k): symbolic.pystr_to_symbolic(str(v)) for k, v in nsdfg_node.symbol_mapping.items()}
    found = False
    for state in nsdfg_node.sdfg.all_states():
        for dn in state.data_nodes():
            if dn.data != conn or state.in_degree(dn) == 0:
                continue
            for e in state.in_edges(dn):
                if e.data is None or e.data.wcr is not None:
                    return False
                if isinstance(e.src, nodes.NestedSDFG):
                    if not _nested_writes_iter_indexed(e.src, e.src_conn, itersym, a, b, step):
                        return False
                    found = True
                    continue
                dst_subset = e.data.get_dst_subset(e, state)
                if dst_subset is None:
                    return False
                outer = copy.deepcopy(dst_subset)
                outer.replace(repl)
                if not _check_range(outer, a, itersym, b, step):
                    return False
                found = True
    return found


def _constant_dim_disjoint(read: subsets.Subset, write: subsets.Subset, dep_indices: Set[int]) -> bool:
    """``True`` iff some iteration-independent dimension carries different
    constants on the read vs the write. Such a dimension makes the two
    accesses pointwise disjoint regardless of what the iteration-dependent
    dimensions look like, so no cross-iteration alias is possible.

    ``aa[0, i]`` (write) vs ``aa[1, i - 1]`` (read) is the canonical case --
    TSVC s132. The two accesses live on disjoint rows.
    """
    rndr = list(read.ndrange())
    wndr = list(write.ndrange())
    if len(rndr) != len(wndr):
        return False
    for i, ((rb, re_, _), (wb, we_, _)) in enumerate(zip(rndr, wndr)):
        if i in dep_indices:
            continue
        if rb != re_ or wb != we_:
            continue
        try:
            diff = symbolic.simplify(rb - wb)
        except Exception:
            continue
        if getattr(diff, 'is_number', False) and diff != 0:
            return True
    return False


def _dependent_indices(itervar: str, subset: subsets.Subset) -> Set[int]:
    """ Finds the indices or ranges of a subset that depend on the iteration
        variable. Returns their index in the subset's indices/ranges list.
    """
    return {
        i
        for i, rng in enumerate(subset.ndrange()) if any(
            symbolic.issymbolic(t) and itervar in {str(s)
                                                   for s in t.free_symbols} for t in rng)
    }


def _sanitize_by_index(indices: Set[int], subset: subsets.Subset) -> subsets.Range:
    """ Keeps the indices or ranges of subsets that are in `indices`. """
    return subsets.Range([t for i, t in enumerate(subset.ndrange()) if i in indices])


def _affine_coeffs(expr, itersym):
    """ Return ``(a, b)`` with ``expr == a*itersym + b``, or ``None`` if
        ``expr`` is not affine in ``itersym``.
    """
    e = sp.expand(symbolic.pystr_to_symbolic(expr))
    a = e.coeff(itersym, 1)
    b = e.coeff(itersym, 0)
    if sp.simplify(e - (a * itersym + b)) != 0:
        return None
    return a, b


def _collect_iedge_bindings(loop: LoopRegion) -> Dict[str, str]:
    """Symbol -> RHS for every assignment carried by an interstate edge inside
    ``loop``'s body. The frontend often binds the per-iteration index of a
    compound write (e.g. ``a[i + M] = ...``) to a fresh symbol on an interstate
    edge -- ``a_slice = i + M`` -- and then writes ``a[a_slice]`` in the next
    state. The subset on the final write thus reads ``a_slice``, not a clean
    ``a*i + b`` form, and the affine matcher gives up. Substituting these
    bindings back lets the matcher see the original linear access.
    """
    out: Dict[str, str] = {}
    for e in loop.all_interstate_edges():
        if not e.data.assignments:
            continue
        for sym, rhs in e.data.assignments.items():
            # Single source of truth: if the same symbol is assigned in
            # multiple places, give up rather than picking one arbitrarily.
            if sym in out and out[sym] != str(rhs):
                out[sym] = None  # ambiguous
            elif sym not in out:
                out[sym] = str(rhs)
    return {k: v for k, v in out.items() if v is not None}


def _subset_with_iedge_subs(subset: subsets.Range, bindings: Dict[str, str]) -> subsets.Range:
    """Return ``subset`` with every reference to a bound symbol replaced by its
    interstate-edge RHS. Bindings whose RHS still references another bound
    symbol are resolved transitively (fixed-point) so chained stagings flatten.

    The returned :class:`subsets.Range` shares no state with ``subset``; safe
    to pass through downstream affine checks.
    """
    if not bindings:
        return subset
    sym_map = {sp.Symbol(k): symbolic.pystr_to_symbolic(v) for k, v in bindings.items()}
    # Transitively resolve: substitute the map into its own RHS until stable.
    for _ in range(8):
        changed = False
        for k, v in list(sym_map.items()):
            nv = v.subs(sym_map)
            if nv != v:
                sym_map[k] = nv
                changed = True
        if not changed:
            break
    new_ranges = []
    for rb, re_, rs in subset.ndrange():
        new_ranges.append((symbolic.simplify(symbolic.pystr_to_symbolic(str(rb)).subs(sym_map)),
                           symbolic.simplify(symbolic.pystr_to_symbolic(str(re_)).subs(sym_map)),
                           symbolic.simplify(symbolic.pystr_to_symbolic(str(rs)).subs(sym_map))))
    return subsets.Range(new_ranges)


def _joint_disjoint_2d(read: subsets.Subset, write: subsets.Subset, itersym, start, stride) -> bool:
    """Joint cross-iteration disjointness for multi-dimensional affine accesses.

    A per-dimension check (:func:`_cross_iter_disjoint`) returns True when
    *some* dimension is disjoint by itself; that misses cases where every
    dimension can individually collide for some ``(p1, p2)`` but the two
    constraints cannot hold *simultaneously*. The canonical case is the
    wavefront ``arr[t-p, p]`` write vs ``arr[t-p, p-1]`` read after skewing:
    dim 0 requires ``p1 = p2`` and dim 1 requires ``p1 = p2 - 1``, both
    satisfiable separately but jointly inconsistent.

    Forms the per-dimension alias equation ``a1*p1 - a2*p2 = b2 - b1`` and
    looks for a pair of dimensions whose 2x2 system has no solution -- either
    determinant ``!= 0`` with non-integer / out-of-range solution, or
    determinant ``== 0`` with the augmented system inconsistent.
    """
    try:
        s_sym = symbolic.simplify(stride)
    except Exception:
        return False
    if not getattr(s_sym, 'is_Integer', False) or int(s_sym) <= 0:
        return False
    rndr = list(read.ndrange())
    wndr = list(write.ndrange())
    if len(rndr) != len(wndr):
        return False
    coeffs = []
    for (rb, re_, _), (wb, we_, _) in zip(rndr, wndr):
        if rb != re_ or wb != we_:
            return False
        f_r = _affine_coeffs(rb, itersym)
        f_w = _affine_coeffs(wb, itersym)
        if f_r is None or f_w is None:
            return False
        a_r, b_r = f_r
        a_w, b_w = f_w
        if not (getattr(a_r, 'is_Integer', False) and getattr(a_w, 'is_Integer', False)):
            return False
        coeffs.append((int(a_w), int(a_r), symbolic.simplify(b_r - b_w)))

    for i in range(len(coeffs)):
        for j in range(i + 1, len(coeffs)):
            a1_w, a1_r, c1 = coeffs[i]
            a2_w, a2_r, c2 = coeffs[j]
            # System: a1_w * p1 - a1_r * p2 = c1
            #         a2_w * p1 - a2_r * p2 = c2
            det = a1_w * (-a2_r) - (-a1_r) * a2_w
            if det != 0:
                # Unique solution -- need both c-values to be numeric to check.
                if not (getattr(c1, 'is_number', False) and getattr(c2, 'is_number', False)):
                    continue
                # Solve via Cramer's: p1 = ((-a2_r)*c1 - (-a1_r)*c2) / det
                num_p1 = (-a2_r) * c1 - (-a1_r) * c2
                num_p2 = a1_w * c2 - a2_w * c1
                if num_p1 % det != 0 or num_p2 % det != 0:
                    return True  # non-integer solution -> disjoint
                p1_val = num_p1 // det
                p2_val = num_p2 // det
                # We don't have bounds at hand, so accept only the strict
                # non-integer case here; that's still enough to catch the
                # wavefront ``det == 0`` family below.
                if p1_val == p2_val:
                    # Same iteration -> not cross-iter alias.
                    return True
            else:
                # Singular system -- consistent iff (a1_w, -a1_r, c1) and
                # (a2_w, -a2_r, c2) are linearly dependent including the c
                # column. Otherwise inconsistent -> no joint solution.
                if not (getattr(c1, 'is_number', False) and getattr(c2, 'is_number', False)):
                    continue
                # Find a non-zero entry in row 1 to compute the ratio.
                if a1_w != 0:
                    if a2_w * c1 != a1_w * c2:
                        return True
                elif a1_r != 0:
                    if (-a2_r) * c1 != (-a1_r) * c2:
                        return True
                # Both rows are (0, 0, c) -- consistent iff c == 0.
                elif c1 != 0 or c2 != 0:
                    return True
    return False


def _cross_iter_disjoint(idx1, idx2, itersym, start, stride) -> bool:
    """True iff a write at ``idx1(i)`` cannot collide with a read at ``idx2(j)``
    for any pair of *distinct* iterations ``i``, ``j`` both in the strided
    iteration set ``{start, start+stride, start+2*stride, ...}``.

    For affine accesses ``idx1 = a1*itersym + b1`` and ``idx2 = a2*itersym + b2``,
    the alias condition ``idx1(i) = idx2(j)`` substituted by ``i = start + s*ki``,
    ``j = start + s*kj`` reduces to ``s * (a1*ki - a2*kj) = (a2-a1)*start + (b2-b1)``.
    The RHS must be divisible by ``s``; if it is not, no integer solution exists
    and the accesses are provably cross-iteration-disjoint. If it is divisible,
    the reduced equation ``a1*ki - a2*kj = rhs/s`` has solutions iff
    ``gcd(a1, a2)`` divides ``rhs/s`` (standard linear Diophantine).

    Plugs the gap when ``propagate_subset`` collapses a strided iteration into
    a stride-1 box on the propagated range -- e.g. ``a[i]`` (writes ``{1,3,...}``)
    vs ``a[i-1]`` (reads ``{0,2,...}``) for ``range(1, N, 2)`` would otherwise
    look like overlapping ``[0..N-1]`` boxes after propagation. (TSVC ``s111``.)
    """
    f1 = _affine_coeffs(idx1, itersym)
    f2 = _affine_coeffs(idx2, itersym)
    if f1 is None or f2 is None:
        return False
    a1, b1 = f1
    a2, b2 = f2
    if not (getattr(a1, 'is_Integer', False) and getattr(a2, 'is_Integer', False)):
        return False
    try:
        s_sym = symbolic.simplify(stride)
    except Exception:
        return False
    if not getattr(s_sym, 'is_Integer', False) or int(s_sym) <= 1:
        return False
    s = int(s_sym)
    try:
        rhs = symbolic.simplify((a2 - a1) * start + (b2 - b1))
    except Exception:
        return False
    if not getattr(rhs, 'is_number', False) or not getattr(rhs, 'is_Integer', False):
        return False
    rhs_i = int(rhs)
    if rhs_i % s != 0:
        return True
    reduced = rhs_i // s
    g = sp.igcd(int(a1), int(a2))
    if g <= 0:
        return False
    return reduced % g != 0


def _ranges_disjoint_by_stride(r1: subsets.Range, r2: subsets.Range) -> bool:
    """Two propagated ranges are disjoint when, on some dimension, their strides
    and starts put them in different residue classes mod ``gcd(stride1, stride2)``.

    The general linear-Diophantine criterion: ``s1 + p1 * k1 == s2 + p2 * k2`` has
    an integer solution iff ``gcd(p1, p2)`` divides ``s2 - s1``. If it does NOT, no
    iteration of either range can hit the same element on that dimension, so the
    multidimensional ranges themselves cannot intersect.

    Plugs the gap in :func:`dace.subsets.Range.intersects`, whose docstring notes
    it does not consider strides. Covers the canonical odd-write / even-read case
    (``a[1:N:2]`` vs ``a[0:N-1:2]`` -- TSVC ``s111`` after frontend lowering).
    """
    if len(r1) != len(r2):
        return False
    for (s1, _e1, p1), (s2, _e2, p2) in zip(r1, r2):
        try:
            p1s = symbolic.simplify(p1)
            p2s = symbolic.simplify(p2)
        except Exception:
            continue
        if not (getattr(p1s, 'is_Integer', False) and getattr(p2s, 'is_Integer', False)):
            continue
        p1i, p2i = int(p1s), int(p2s)
        if p1i <= 0 or p2i <= 0:
            continue
        if p1i == 1 and p2i == 1:
            continue  # stride-1 ranges -- nothing to gain over the existing box check
        g = sp.igcd(p1i, p2i)
        if g <= 1:
            continue
        try:
            diff = symbolic.simplify(s2 - s1)
        except Exception:
            continue
        if not getattr(diff, 'is_number', False) or not getattr(diff, 'is_Integer', False):
            continue
        if int(diff) % g != 0:
            return True
    return False


def _dim_provably_disjoint(idx1, idx2, itersym) -> bool:
    """ True iff ``idx1`` at any iteration can never equal ``idx2`` at any
        iteration, for any integer iterations and any loop bounds.

        Uses the linear-Diophantine solvability criterion: ``a1*i1 + b1 ==
        a2*i2 + b2`` has an integer solution iff ``gcd(a1, a2)`` divides
        ``b2 - b1``. If it does not, the accesses never alias.
    """
    f1 = _affine_coeffs(idx1, itersym)
    f2 = _affine_coeffs(idx2, itersym)
    if f1 is None or f2 is None:
        return False
    a1, b1 = f1
    a2, b2 = f2
    diff = sp.simplify(b2 - b1)
    if not (a1.is_Integer and a2.is_Integer):
        return False
    if a1 == 0 and a2 == 0:
        return diff.is_number and diff != 0
    g = sp.igcd(int(a1), int(a2))
    if g == 0:
        return diff.is_number and diff != 0
    if not diff.is_number:
        return False
    if not diff.is_Integer:
        return True
    return sp.Integer(diff) % g != 0


def _writes_may_overlap(m1: memlet.Memlet, m2: memlet.Memlet, itersym) -> bool:
    """ Conservatively decide whether two write memlets to the same container
        can address the same element on different loop iterations. Returns
        ``False`` only if some subset dimension is provably disjoint (the
        multidimensional element can then never coincide).
    """
    nd1 = list(m1.subset.ndrange())
    nd2 = list(m2.subset.ndrange())
    if len(nd1) != len(nd2):
        return True
    for (b1, e1, _), (b2, e2, _) in zip(nd1, nd2):
        if b1 != e1 or b2 != e2:  # non-point range dimension: cannot decide here
            continue
        # Both writes index this dimension by the same injective function of the
        # iteration variable: a collision there forces the two iterations equal,
        # so the writes can only coincide within one iteration (ordered by program
        # order in the map body), never across distinct iterations.
        coeffs = _affine_coeffs(b1, itersym)
        if coeffs is not None and coeffs[0] != 0 and sp.simplify(b1 - b2) == 0:
            return False
        if _dim_provably_disjoint(b1, b2, itersym):
            return False
    return True


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToMap(xf.MultiStateTransformation):
    """
    Convert a control flow loop into a dataflow map. Currently only supports the simple case where there is no overlap
    between inputs and outputs in the body of the loop, and where the loop body only consists of a single state.
    """

    loop = xf.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):

        def refuse(reason: str) -> bool:
            """Log why this loop stays sequential (keyed by loop label) and refuse."""
            print(f"LoopToMap refused [{self.loop.label}]: {reason}")
            return False

        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return refuse(f"loop information incomplete - start={start}, end={end}, step={step}, itervar={itervar}")

        sset = {}
        sset.update(sdfg.symbols)
        sset.update(sdfg.arrays)
        t = dtypes.result_type_of(infer_expr_type(start, sset), infer_expr_type(step, sset), infer_expr_type(end, sset))
        # We may only convert something to map if the bounds are all integer-derived types. Otherwise most map schedules
        # except for sequential would be invalid.
        if not t in dtypes.INTEGER_TYPES:
            return refuse(f"loop bounds are not integer types - result_type={t}")

        # Loops containing break, continue, or returns may not be turned into a map.
        for blk in self.loop.all_control_flow_blocks():
            if isinstance(blk, (BreakBlock, ContinueBlock, ReturnBlock)):
                if not permissive:
                    return refuse(f"loop body contains a {type(blk).__name__}")

        # We cannot handle symbols read from data containers unless they are scalar.
        for expr in (start, end, step):
            if symbolic.contains_sympy_functions(expr):
                return refuse(f"bound expression reads a non-scalar data container - expr={expr}")

        # Refuse when the loop's range (start/end/step) references a symbol
        # that the loop body itself defines via an interstate-edge
        # assignment. After conversion the body moves into a new
        # ``loop_body`` NestedSDFG and the assignment goes with it, but the
        # Map's range stays at the outer scope; the range then references a
        # symbol only defined inside the new NSDFG, producing a
        # ``Missing symbols on nested SDFG`` validation failure downstream.
        range_syms: Set[str] = set()
        for expr in (start, end, step):
            try:
                range_syms |= {str(s) for s in expr.free_symbols}
            except AttributeError:
                pass
        body_assigned_syms: Set[str] = set()
        for e in self.loop.all_interstate_edges():
            body_assigned_syms.update(e.data.assignments.keys())
        if range_syms & body_assigned_syms:
            return refuse(f"loop range references symbol(s) {range_syms & body_assigned_syms} assigned inside the body")

        _, write_set = self.loop.read_and_write_sets()
        loop_states = set(self.loop.all_states())
        all_loop_blocks = set(self.loop.all_control_flow_blocks())

        # Cannot have StructView in loop body
        for loop_state in loop_states:
            if [n for n in loop_state.data_nodes() if isinstance(n.desc(sdfg), dt.StructureView)]:
                return refuse(f"loop body contains a StructureView in state {loop_state}")

        # Build a substitution map from interstate-edge symbol bindings inside the
        # loop body; used to resolve staged write/read subsets like ``a[a_slice]``
        # where ``a_slice := i + M`` on an interstate edge.
        iedge_bindings = _collect_iedge_bindings(self.loop)

        # Collect symbol reads and writes from inter-state assignments
        in_order_loop_blocks = list(
            cfg_analysis.blockorder_topological_sort(self.loop, recursive=True, ignore_nonstate_blocks=False))
        symbols_that_may_be_used: Set[str] = {itervar}
        used_before_assignment: Set[str] = set()
        for block in in_order_loop_blocks:
            # A symbol read in the block's own dataflow (e.g. a memlet subset
            # ``b[im]``) is read before any symbol the block assigns on its
            # out-edges; if the loop later reassigns it, it is loop-carried. The
            # per-edge ``read_symbols()`` below only sees interstate-edge reads, so
            # fold in these in-state reads.
            try:
                block_reads = {str(s) for s in block.free_symbols}
            except Exception:
                block_reads = set()
            used_before_assignment |= (block_reads - symbols_that_may_be_used)
            for e in block.parent_graph.out_edges(block):
                # Collect read-before-assigned symbols (this works because the states are always in order,
                # see above call to `blockorder_topological_sort`)
                read_symbols = e.data.read_symbols()
                read_symbols -= symbols_that_may_be_used
                used_before_assignment |= read_symbols
                # If symbol was read before it is assigned, the loop cannot be parallel
                assigned_symbols = set()
                for k, v in e.data.assignments.items():
                    try:
                        fsyms = symbolic.pystr_to_symbolic(v).free_symbols
                    except AttributeError:
                        fsyms = set()
                    if not k in fsyms:
                        assigned_symbols.add(k)
                if assigned_symbols & used_before_assignment:
                    return refuse("carried symbol dependency - "
                                  f"{assigned_symbols & used_before_assignment} read before being assigned")

                symbols_that_may_be_used |= e.data.assignments.keys()

        # Get access nodes from other states to isolate local loop variables
        other_access_nodes: Set[str] = set()
        for state in sdfg.states():
            if state in loop_states:
                continue
            other_access_nodes |= set(n.data for n in state.data_nodes() if sdfg.arrays[n.data].transient)
        # Add non-transient nodes from loop state
        for state in loop_states:
            other_access_nodes |= set(n.data for n in state.data_nodes() if not sdfg.arrays[n.data].transient)

        write_memlets: Dict[str, List[memlet.Memlet]] = defaultdict(list)

        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])

        for state in loop_states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                # Take all writes that are not conflicted into consideration
                if dn.data in write_set:
                    for e in state.in_edges(dn):
                        if e.data.dynamic and e.data.wcr is None:
                            # A dynamic write (no WCR) is still safe across
                            # outer-loop iterations if its destination subset
                            # pins an axis to the iteration variable (same
                            # ``a*i+b`` pattern enforced below for non-dynamic
                            # writes). Each iteration then writes to a disjoint
                            # slab -- whether a given lane fires or not
                            # cannot race with another iteration's write.
                            dst_subset = e.data.get_dst_subset(e, state)
                            if not (dst_subset and _check_range(dst_subset, a, itersym, b, step)):
                                return refuse(f"dynamic write to {dn.data} is not indexed by the iteration variable "
                                              f"- dst_subset={dst_subset}")
                        if e.data is None:
                            continue

                        # To be sure that the value is only written at unique
                        # indices per loop iteration, we want to match symbols
                        # of the form "a*i+b" where |a| >= 1, and i is the iteration
                        # variable. The iteration variable must be used.
                        if e.data.wcr is None:
                            dst_subset = e.data.get_dst_subset(e, state)
                            ok = bool(dst_subset) and _check_range(dst_subset, a, itersym, b, step)
                            # A NestedSDFG loop body propagates a whole-array
                            # external write memlet that hides an inner
                            # per-iteration write; look past the connector.
                            if not ok and isinstance(e.src, nodes.NestedSDFG):
                                ok = _nested_writes_iter_indexed(e.src, e.src_conn, itersym, a, b, step)
                            # If the subset references a symbol bound by a loop
                            # interstate edge (the frontend's compound-index
                            # staging shape), substitute and re-check. The
                            # mutation is intentional and persistent: this is
                            # ``can_be_applied`` but rewriting the staged
                            # subset is needed for both downstream affine
                            # disjoint checks (this method) and the final
                            # apply (which reads the same memlet); the iedge
                            # bindings are loop-invariant so the substitution
                            # never changes semantics.
                            if not ok and dst_subset is not None and iedge_bindings:
                                resolved = _subset_with_iedge_subs(dst_subset, iedge_bindings)
                                if _check_range(resolved, a, itersym, b, step):
                                    e.data.subset = resolved
                                    dst_subset = resolved
                                    ok = True
                            if not ok and not permissive:
                                return refuse(f"write to {dn.data} is not uniquely indexed by the iteration variable "
                                              f"(needs an a*i+b subset) - dst_subset={dst_subset}")
                        # End of check

                        write_memlets[dn.data].append(e.data)

        # Two writes with distinct affine subscripts into the same container can
        # hit the same element on different iterations even when each is
        # individually injective in the iteration variable (e.g. ``A[5*i]`` and
        # ``A[3*i]`` collide at ``A[15]``). Parallelizing then reorders the
        # colliding writes. Allow the pair only if some dimension is provably
        # disjoint for all iterations (e.g. ``A[2*i]`` vs ``A[2*i+1]``).
        for data, mmlts in write_memlets.items():
            distinct: Dict[str, memlet.Memlet] = {}
            for m in mmlts:
                if m.wcr is None:
                    distinct.setdefault(str(m.subset), m)
            reps = list(distinct.values())
            for x in range(len(reps)):
                for y in range(x + 1, len(reps)):
                    if _writes_may_overlap(reps[x], reps[y], itersym) and not permissive:
                        return refuse(f"writes {reps[x].subset} and {reps[y].subset} to {data} "
                                      "may overlap across iterations")

        # After looping over relevant writes, consider reads that may overlap
        for state in loop_states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                data = dn.data
                if data in write_memlets:
                    for e in state.out_edges(dn):
                        if e.data is None:
                            continue

                        # If the same container is both read and written, only match if
                        # it read and written at locations that will not create data races
                        src_subset = e.data.get_src_subset(e, state)
                        if not self.test_read_memlet(sdfg, state, e, itersym, itervar, start, end, step, write_memlets,
                                                     e.data, src_subset):
                            return refuse(f"read-after-write conflict on {data} within the loop body "
                                          f"- src_subset={src_subset}")

        # Consider reads in inter-state edges (could be in assignments or in condition)
        isread_set: Set[memlet.Memlet] = set()
        for e in self.loop.all_interstate_edges():
            isread_set |= set(e.data.get_read_memlets(sdfg.arrays))
        for mmlt in isread_set:
            if mmlt.data in write_memlets:
                if not self.test_read_memlet(sdfg, None, None, itersym, itervar, start, end, step, write_memlets, mmlt,
                                             mmlt.subset):
                    return refuse(f"read-after-write conflict on {mmlt.data} via an inter-state edge "
                                  f"- subset={mmlt.subset}")

        # Check that the iteration variable and other symbols are not used on other edges or blocks before they are
        # reassigned.
        in_order_blocks = list(
            cfg_analysis.blockorder_topological_sort(sdfg, recursive=True, ignore_nonstate_blocks=False))
        # First check the outgoing edges of the loop itself.
        reassigned_symbols: Set[str] = None
        for oe in graph.out_edges(self.loop):
            if symbols_that_may_be_used & oe.data.read_symbols():
                return refuse("loop-defined symbol(s) used after the loop on its outgoing edge - "
                              f"{symbols_that_may_be_used & oe.data.read_symbols()}")
            # Check for symbols that are set by all outgoing edges
            # TODO: Handle case of subset of out_edges
            if reassigned_symbols is None:
                reassigned_symbols = set(oe.data.assignments.keys())
            else:
                reassigned_symbols &= oe.data.assignments.keys()
        # Remove reassigned symbols
        if reassigned_symbols is not None:
            symbols_that_may_be_used -= reassigned_symbols
        loop_idx = in_order_blocks.index(self.loop)
        for block in in_order_blocks[loop_idx + 1:]:
            if block in all_loop_blocks:
                continue
            # Don't continue in this direction, as all loop symbols have been reassigned
            if not symbols_that_may_be_used:
                break

            # Check state contents
            if symbols_that_may_be_used & block.free_symbols:
                return refuse(f"loop-defined symbol(s) used after the loop in block {block} - "
                              f"{symbols_that_may_be_used & block.free_symbols}")

            # Check inter-state edges
            reassigned_symbols = None
            for e in block.parent_graph.out_edges(block):
                if symbols_that_may_be_used & e.data.read_symbols():
                    return refuse("loop-defined symbol(s) used after the loop on an inter-state edge - "
                                  f"{symbols_that_may_be_used & e.data.read_symbols()}")

                # Check for symbols that are set by all outgoing edges
                # TODO: Handle case of subset of out_edges
                if reassigned_symbols is None:
                    reassigned_symbols = set(e.data.assignments.keys())
                else:
                    reassigned_symbols &= e.data.assignments.keys()

            # Remove reassigned symbols
            if reassigned_symbols is not None:
                symbols_that_may_be_used -= reassigned_symbols

        return True

    def test_read_memlet(self, sdfg: SDFG, state: SDFGState, edge: gr.MultiConnectorEdge[memlet.Memlet],
                         itersym: symbolic.SymbolicType, itervar: str, start: symbolic.SymbolicType,
                         end: symbolic.SymbolicType, step: symbolic.SymbolicType,
                         write_memlets: Dict[str, List[memlet.Memlet]], mmlt: memlet.Memlet, src_subset: subsets.Range):
        # Import as necessary
        from dace.sdfg.propagation import propagate_subset, align_memlet

        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        data = mmlt.data

        if (mmlt.dynamic and mmlt.src_subset.num_elements() != 1):
            # If pointers are involved, give up
            return False
        if not _check_range(src_subset, a, itersym, b, step):
            # ``_check_range`` only accepts reads that MOVE with the iteration
            # (some dimension ``a*i + b``, ``|a| >= 1``). A read that uses the
            # iteration symbol but does not match that affine form is
            # conservatively a conflict. But a loop-INVARIANT read (no iteration
            # symbol at all) is only a conflict if it actually overlaps a write:
            # ``a[0]`` is safe when the loop writes ``a[1:N]`` (the post-peel
            # ``a[i] = a[0] + b[i]`` remainder), and is a real read-after-write
            # only when it overlaps the write (``a[0]`` vs ``a[0:N]``). Defer
            # both to the propagated-overlap check below.
            if itersym in src_subset.free_symbols:
                return False

        # Always use the source data container for the memlet test
        if state is not None and edge is not None:
            mmlt = align_memlet(state, edge, dst=False)
            data = mmlt.data

        pread = propagate_subset([mmlt], sdfg.arrays[data], [itervar], subsets.Range([(start, end, step)]))
        for candidate in write_memlets[data]:
            # Simple case: read and write are in the same subset
            read = src_subset
            write = candidate.dst_subset
            if read == write:
                continue
            ridx = _dependent_indices(itervar, read)
            widx = _dependent_indices(itervar, write)
            indices = set(ridx) | set(widx)
            if not indices:
                indices = set(range(len(read)))
            # If some iteration-independent dimension carries a different
            # constant on the read vs the write (e.g. ``aa[0, i]`` written,
            # ``aa[1, i-1]`` read), the two accesses are pointwise disjoint
            # there and the rest of the dependence analysis would lose that
            # by sanitising the dimension away.
            if _constant_dim_disjoint(read, write, indices):
                continue
            read = _sanitize_by_index(indices, read)
            write = _sanitize_by_index(indices, write)
            if read == write:
                continue
            # Propagated read does not overlap with propagated write
            pwrite = propagate_subset([candidate],
                                      sdfg.arrays[data], [itervar],
                                      subsets.Range([(start, end, step)]),
                                      use_dst=True)
            t_pread = _sanitize_by_index(indices, pread.src_subset)
            pwrite_san = _sanitize_by_index(indices, pwrite.dst_subset)
            if subsets.intersects(t_pread, pwrite_san) is False:
                continue
            # ``subsets.intersects`` is stride-blind (its own TODO); when
            # propagation collapses a strided iteration set into a stride-1
            # box, two cross-iteration accesses in different residue classes
            # look like overlapping boxes. Defer to a per-iteration
            # stride-aware Diophantine on each dimension: if *some* dimension
            # is provably cross-iteration-disjoint, no pair (i, j) of distinct
            # iterations aliases.
            if _ranges_disjoint_by_stride(t_pread, pwrite_san):
                continue
            itersym_sym = symbolic.pystr_to_symbolic(itervar)
            cross_disjoint = False
            for ridx_d, widx_d in zip(src_subset.ndrange(), candidate.dst_subset.ndrange()):
                rb, re_, _ = ridx_d
                wb, we_, _ = widx_d
                if rb != re_ or wb != we_:
                    continue
                if _cross_iter_disjoint(wb, rb, itersym_sym, start, step):
                    cross_disjoint = True
                    break
            if cross_disjoint:
                continue
            # Joint multi-dim Diophantine: when no single dimension is
            # disjoint by itself but the system across two or more dimensions
            # is inconsistent, the multi-dim alias has no solution -- the
            # wavefront pattern after skewing is the canonical case.
            if _joint_disjoint_2d(src_subset, candidate.dst_subset, itersym_sym, start, step):
                continue
            return False

        return True

    def _is_array_thread_local(self, name: str, itervar: str, sdfg: SDFG, states: List[SDFGState]) -> bool:
        """
        This helper method checks whether an array used exclusively in the body of a detected for-loop is thread-local,
        i.e., its whole range is may be used in every loop iteration, or is can be shared by multiple iterations.

        For simplicity, it is assumed that the for-loop can be safely transformed to a Map. The method applies only to
        bodies that become a NestedSDFG.

        :param name: The name of array.
        :param itervar: The for-loop iteration variable.
        :param sdfg: The SDFG containing the states that comprise the body of the for-loop.
        :param states: A list of states that comprise the body of the for-loop.
        :return: True if the array is thread-local, otherwise False.
        """

        desc = sdfg.arrays[name]
        if not isinstance(desc, dt.Array):
            # Scalars are always thread-local.
            return True
        if itervar in (str(s) for s in desc.free_symbols):
            # If the shape or strides of the array depend on the iteration variable, then the array is thread-local.
            return True
        for state in states:
            for node in state.data_nodes():
                if node.data != name:
                    continue
                for e in state.out_edges(node):
                    src_subset = e.data.get_src_subset(e, state)
                    # If the iteration variable is in the subsets symbols, then the array cannot be thread-local.
                    # Here we use the assumption that the for-loop can be turned to a valid Map, i.e., all other edges
                    # carrying the array depend on the iteration variable in a consistent manner.
                    if src_subset and itervar in src_subset.free_symbols:
                        return False
                for e in state.in_edges(node):
                    dst_subset = e.data.get_dst_subset(e, state)
                    # If the iteration variable is in the subsets symbols, then the array cannot be thread-local.
                    # Here we use the assumption that the for-loop can be turned to a valid Map, i.e., all other edges
                    # carrying the array depend on the iteration variable in a consistent manner.
                    if dst_subset and itervar in dst_subset.free_symbols:
                        return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        from dace.sdfg.propagation import align_memlet

        # Obtain loop information
        itervar = self.loop.loop_variable
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)

        nsdfg = None

        # Nest loop-body states
        states = set(self.loop.all_states())
        # Find read/write sets
        read_set, write_set = set(), set()
        for state in self.loop.all_states():
            rset, wset = state.read_and_write_sets()
            read_set |= rset
            write_set |= wset
            # Add to write set also scalars between tasklets
            for src_node in state.nodes():
                if not isinstance(src_node, nodes.Tasklet):
                    continue
                for dst_node in state.nodes():
                    if src_node is dst_node:
                        continue
                    if not isinstance(dst_node, nodes.Tasklet):
                        continue
                    for e in state.edges_between(src_node, dst_node):
                        if e.data.data and e.data.data in sdfg.arrays:
                            write_set.add(e.data.data)

        # Add headers of any nested loops and conditional blocks
        nodelist = list(self.loop.nodes())
        while nodelist:
            node = nodelist.pop()
            if isinstance(node, (LoopRegion, ConditionalBlock)):
                code_blocks = node.get_meta_codeblocks()
                free_syms = {s for c in code_blocks for s in c.get_free_symbols()}
                free_syms = {s for s in free_syms if s in sdfg.arrays.keys()}
                read_set |= set(free_syms)
                nodelist.extend(node.nodes())

        # Add data from edges
        for edge in self.loop.all_interstate_edges():
            for s in edge.data.free_symbols:
                if s in sdfg.arrays:
                    read_set.add(s)

        # Build mapping of view data to their root data
        view_to_data = {}
        for state in states:
            for node in state.data_nodes():
                if isinstance(sdfg.arrays[node.data], dt.View):
                    root_node = sdutil.get_last_view_node(state, node)
                    assert root_node is not None
                    if node.data in view_to_data:
                        assert view_to_data[node.data] == root_node.data

                    view_to_data[node.data] = root_node.data

        # Find NestedSDFG's / Loop's unique data
        rw_set = read_set | write_set
        unique_set = set()
        for name in rw_set:
            if not sdfg.arrays[name].transient:
                continue
            found = False
            for state in sdfg.states():
                if state in states:
                    continue
                for node in state.nodes():
                    if (isinstance(node, nodes.AccessNode) and node.data == name):
                        found = True
                        break

            iatl_name = name
            if name in view_to_data:
                iatl_name = view_to_data[name]

            if not found and self._is_array_thread_local(iatl_name, itervar, sdfg, states):
                unique_set.add(name)

        # Find NestedSDFG's connectors
        read_set = {n for n in read_set if n not in unique_set or not sdfg.arrays[n].transient}
        write_set = {n for n in write_set if n not in unique_set or not sdfg.arrays[n].transient}

        # Do not route views through the NestedSDFG
        view_set = set(view_to_data.keys())
        read_set -= view_set
        write_set -= view_set

        # Create NestedSDFG and add the loop contents to it. Gaher symbols defined in the NestedSDFG.
        fsymbols = set(sdfg.free_symbols)
        body = graph.add_state_before(self.loop, 'single_state_body')
        nsdfg = SDFG('loop_body', constants=sdfg.constants_prop, parent=body)
        nsdfg.add_node(self.loop.start_block, is_start_block=True)
        nsymbols = dict()
        for block in self.loop.nodes():
            if block is self.loop.start_block:
                continue
            nsdfg.add_node(block)
        for e in self.loop.edges():
            nsymbols.update({s: sdfg.symbols[s] for s in e.data.assignments.keys() if s in sdfg.symbols})
            nsdfg.add_edge(e.src, e.dst, e.data)

        # Add NestedSDFG arrays
        for name in read_set | write_set:
            if '.' in name:
                root_data_name = name.split('.')[0]
                name = root_data_name
            nsdfg.arrays[name] = copy.deepcopy(sdfg.arrays[name])
            nsdfg.arrays[name].transient = False
        for name in unique_set | view_set:
            if '.' in name:
                root_data_name = name.split('.')[0]
                name = root_data_name
            nsdfg.arrays[name] = copy.deepcopy(sdfg.arrays[name])

        # Add NestedSDFG node
        cnode = body.add_nested_sdfg(nsdfg, read_set, write_set)
        if sdfg.parent:
            for s, m in sdfg.parent_nsdfg_node.symbol_mapping.items():
                if s not in cnode.symbol_mapping:
                    cnode.symbol_mapping[s] = symbolic.pystr_to_symbolic(s)
                    nsdfg.symbols[s] = sdfg.symbols[s]
        for name in read_set:
            r = body.add_read(name)
            body.add_edge(r, None, cnode, name, memlet.Memlet.from_array(name, sdfg.arrays[name]))
        for name in write_set:
            w = body.add_write(name)
            body.add_edge(cnode, name, w, None, memlet.Memlet.from_array(name, sdfg.arrays[name]))

        # Fix SDFG symbols
        for sym in sdfg.free_symbols - fsymbols:
            if sym in sdfg.symbols:
                sdfg.remove_symbol(sym)
        for sym, dtype in nsymbols.items():
            nsdfg.symbols[sym] = dtype

        # Symbols that the nested SDFG assigns on its own interstate edges
        # are internal -- they must not be surfaced onto the NestedSDFG
        # node's ``symbol_mapping``. Doing so would make the outer SDFG
        # appear to need them as free symbols; a later pruning pass that
        # removes them from ``sdfg.symbols`` would then desync the mapping
        # (codegen reads ``sdfg.symbols[sym]`` and raises ``KeyError``).
        internally_defined = set()
        for e in nsdfg.all_interstate_edges():
            internally_defined.update(e.data.assignments.keys())

        # Propagate free symbols referenced by nested array shapes/strides/offsets:
        # ``copy.deepcopy`` of the descriptor carries the symbols, but they
        # must be added to the NestedSDFG's symbol mapping.
        for desc in nsdfg.arrays.values():
            for sym in desc.free_symbols:
                sym_name = str(sym)
                if sym_name in internally_defined:
                    continue
                if sym_name in sdfg.symbols:
                    if sym_name not in nsdfg.symbols:
                        nsdfg.symbols[sym_name] = sdfg.symbols[sym_name]
                    if sym_name not in cnode.symbol_mapping:
                        cnode.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)

        # Propagate symbols, where types cannot be inferred
        alltypes = copy.deepcopy(nsdfg.symbols)
        alltypes.update({k: v.dtype for k, v in nsdfg.arrays.items()})
        for e in self.loop.all_interstate_edges():
            for k, v in e.data.assignments.items():
                # Skip if the symbol is already in the SDFG
                if k in nsdfg.symbols:
                    continue

                # Should not happen: Cannot infer type and parent SDFG also does not have an explicit type
                vtype = infer_expr_type(v, alltypes)
                if k not in sdfg.symbols:
                    if vtype is None:
                        warnings.warn(f"Symbol {k} not found in parent SDFG symbols.")
                    continue

                # If the inferred type and the symbol type are the same, skip
                ktype: dtypes.typeclass = sdfg.symbols[k]
                if ktype == vtype:
                    continue

                # Only add explicit type, if it cannot be inferred
                if vtype is None:
                    nsdfg.symbols[k] = ktype

        if (step < 0) == True:
            # If step is negative, we have to flip start and end to produce a correct map with a positive increment.
            start, end, step = end, start, -step

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        # Check intermediate notes
        intermediate_nodes: List[nodes.AccessNode] = []
        for node in body.nodes():
            if isinstance(node, nodes.AccessNode) and body.in_degree(node) > 0 and node not in sink_nodes:
                # Scalars written without WCR must be thread-local
                if isinstance(node.desc(sdfg), dt.Scalar) and any(e.data.wcr is None for e in body.in_edges(node)):
                    continue
                # Arrays written with subsets that do not depend on the loop variable must be thread-local
                map_dependency = False
                for e in body.in_edges(node):
                    subset = e.data.get_dst_subset(e, body)
                    if any(str(s) == itervar for s in subset.free_symbols):
                        map_dependency = True
                        break
                if not map_dependency:
                    continue
                intermediate_nodes.append(node)

        map_node = nodes.Map(body.label + "_map", [itervar], [(start, end, step)])
        entry = nodes.MapEntry(map_node)
        exit = nodes.MapExit(map_node)
        body.add_node(entry)
        body.add_node(exit)

        # If the map uses symbols from data containers, instantiate reads
        containers_to_read = entry.free_symbols & sdfg.arrays.keys()
        # Filter out views
        containers_to_read = {c for c in containers_to_read if not isinstance(sdfg.arrays[c], dt.View)}
        for rd in containers_to_read:
            # We are guaranteed that this is always a scalar, because
            # can_be_applied makes sure there are no sympy functions in each of
            # the loop expresions
            access_node = body.add_read(rd)
            body.add_memlet_path(access_node, entry, dst_conn=rd, memlet=memlet.Memlet(rd))

        # Add views as symbols
        views_to_read = (entry.free_symbols & sdfg.arrays.keys()) - containers_to_read
        view_assignments = {}
        for rd in views_to_read:
            rd_name = f"{rd}_map"
            view_assignments[rd_name] = rd

            rd_sym = symbolic.pystr_to_symbolic(rd)
            rd_name_sym = symbolic.pystr_to_symbolic(rd_name)

            for i in range(len(map_node.range)):
                lb, up, st = map_node.range[i]
                lb = lb.replace(rd_sym, rd_name_sym)
                up = up.replace(rd_sym, rd_name_sym)
                st = st.replace(rd_sym, rd_name_sym)
                map_node.range[i] = (lb, up, st)

        if view_assignments:
            graph.add_state_before(body, "map_views", assignments=view_assignments)

        # Direct edges among source and sink access nodes must pass through a tasklet.
        # We first gather them and handle them later.
        direct_edges: Set[gr.MultiConnectorEdge[memlet.Memlet]] = set()
        for n1 in source_nodes:
            if not isinstance(n1, nodes.AccessNode):
                continue
            for n2 in sink_nodes:
                if not isinstance(n2, nodes.AccessNode):
                    continue
                for e in body.edges_between(n1, n2):
                    e.data.try_initialize(sdfg, body, e)
                    direct_edges.add(e)
                    body.remove_edge(e)

        # Reroute all memlets through the entry and exit nodes
        for n in source_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.out_edges(n):
                    # Fix memlet to contain outer data as subset
                    new_memlet = align_memlet(body, e, dst=False)

                    body.remove_edge(e)
                    body.add_edge_pair(entry, e.dst, n, new_memlet, internal_connector=e.dst_conn)
            else:
                body.add_nedge(entry, n, memlet.Memlet())
        for n in sink_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.in_edges(n):
                    # Fix memlet to contain outer data as subset
                    new_memlet = align_memlet(body, e, dst=True)

                    body.remove_edge(e)
                    body.add_edge_pair(exit, e.src, n, new_memlet, internal_connector=e.src_conn)
            else:
                body.add_nedge(n, exit, memlet.Memlet())
        intermediate_sinks: Dict[str, nodes.AccessNode] = {}
        for n in intermediate_nodes:
            if isinstance(sdfg.arrays[n.data], dt.View):
                continue
            if n.data in intermediate_sinks:
                sink = intermediate_sinks[n.data]
            else:
                sink = body.add_access(n.data)
                intermediate_sinks[n.data] = sink
            helpers.make_map_internal_write_external(sdfg, body, exit, n, sink)

        # Here we handle the direct edges among source and sink access nodes.
        for e in direct_edges:
            src: str = e.src.data
            dst: str = e.dst.data
            if e.data.subset.num_elements() == 1:
                t = body.add_tasklet(f"{n1}_{n2}", {'__inp'}, {'__out'}, "__out =  __inp")
                src_conn, dst_conn = '__out', '__inp'
            else:
                desc = sdfg.arrays[src]
                tname, _ = sdfg.add_transient('tmp',
                                              e.data.src_subset.size(),
                                              desc.dtype,
                                              desc.storage,
                                              find_new_name=True)
                t = body.add_access(tname)
                src_conn, dst_conn = None, None
            body.add_memlet_path(n1,
                                 entry,
                                 t,
                                 memlet=memlet.Memlet(data=src, subset=e.data.src_subset),
                                 dst_conn=dst_conn)
            body.add_memlet_path(t,
                                 exit,
                                 n2,
                                 memlet=memlet.Memlet(data=dst,
                                                      subset=e.data.dst_subset,
                                                      wcr=e.data.wcr,
                                                      wcr_nonatomic=e.data.wcr_nonatomic),
                                 src_conn=src_conn)

        if not source_nodes and not sink_nodes:
            body.add_nedge(entry, exit, memlet.Memlet())

        # Redirect outgoing edges connected to the loop to connect to the body state instead.
        for e in graph.out_edges(self.loop):
            graph.add_edge(body, e.dst, e.data)
        # Delete the loop and connected edges.
        graph.remove_node(self.loop)

        # If this had made a variable a free symbol, we can remove it from the SDFG symbols.
        # Guard both branches with ``in sdfg.symbols`` -- the array-descriptor-symbol
        # propagation above may have already cleared entries that were also free.
        for var in sdfg.free_symbols - fsymbols:
            if var not in sdfg.symbols:
                continue
            if sdfg.parent_nsdfg_node:
                if var not in sdfg.parent_nsdfg_node.symbol_mapping:
                    sdfg.remove_symbol(var)
            else:
                sdfg.remove_symbol(var)

        # Also remove arrays that are unique to the loop body
        for name in unique_set:
            if name in sdfg.arrays:
                sdfg.remove_data(name)

        sdfg.reset_cfg_list()
        for n, p in sdfg.all_nodes_recursive():
            if isinstance(n, nodes.NestedSDFG):
                n.sdfg.parent = p
                n.sdfg.parent_nsdfg_node = n
                n.sdfg.parent_sdfg = p.sdfg

        # Narrow the freshly-created state-level memlets. ``body.add_edge`` uses
        # ``Memlet.from_array`` (full extent) when wiring the new ``NestedSDFG``,
        # which hides the inside subset and stalls downstream passes that look
        # at the carrier's state-level memlets (e.g. ``LoopToScan``'s
        # ``_find_carried_arrays``, ``RedundantArrayCopying*``). Propagating the
        # inside subsets out through the NSDFG + Map narrows them to
        # ``[outer_loop_var, map_range]`` -- the symbol set ``symbols_defined_at``
        # already reports for the ``cnode`` here.
        from dace.sdfg.propagation import propagate_memlets_nested_sdfg, propagate_memlets_state
        propagate_memlets_nested_sdfg(sdfg, body, cnode)
        propagate_memlets_state(sdfg, body)
