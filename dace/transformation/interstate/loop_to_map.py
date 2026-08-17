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
    """Whether every write to ``conn``'s array inside ``nsdfg_node`` is indexed by the mapped
    iteration variable. The external connector memlet is the union over the loop and hides the
    per-iteration write, so rewrite the inner subsets through ``symbol_mapping`` and check each
    against ``a*i+b``. Needs at least one inner write; nested NestedSDFGs recurse.
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


def _nested_reads_match_writes(nsdfg_node, conn, itersym, a, b, step) -> bool:
    """Whether every read of ``conn``'s array inside ``nsdfg_node`` matches the writes' ``a*i+b`` or
    is loop-invariant. Write uniqueness alone is not enough: ``a[i] = ... a[i+1]`` still races.
    """
    repl = {symbolic.symbol(k): symbolic.pystr_to_symbolic(str(v)) for k, v in nsdfg_node.symbol_mapping.items()}
    for state in nsdfg_node.sdfg.all_states():
        for dn in state.data_nodes():
            if dn.data != conn or state.out_degree(dn) == 0:
                continue
            for e in state.out_edges(dn):
                if e.data is None:
                    continue
                if isinstance(e.dst, nodes.NestedSDFG):
                    # The read enters another nested SDFG; descend.
                    if not _nested_reads_match_writes(e.dst, e.dst_conn, itersym, a, b, step):
                        return False
                    continue
                src_subset = e.data.get_src_subset(e, state)
                if src_subset is None:
                    return False
                outer = copy.deepcopy(src_subset)
                outer.replace(repl)
                # Loop-invariant read (no itersym) -- safe, same value every iteration.
                free = set()
                for rb, re, _ in outer.ndrange():
                    for expr in (rb, re):
                        if hasattr(expr, 'free_symbols'):
                            free |= set(expr.free_symbols)
                if itersym not in free:
                    continue
                # itersym-dependent read: must match the writes' a*i+b, else it's a carried read.
                if not _check_range(outer, a, itersym, b, step):
                    return False
    return True


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
    """Return ``(a, b)`` with ``expr == a*itersym + b``, or ``None`` if not affine in ``itersym``.

        Derived, not searched: the derivative is ``a`` (and still mentions ``itersym`` above degree
        one, which is the degree test), the value at zero is ``b``. ``expand`` + ``coeff`` is
        superlinear and hung ``can_be_applied`` on deeply tiled index expressions.
    """
    e = symbolic.pystr_to_symbolic(expr)
    if not e.is_polynomial(itersym):
        return None
    a = sp.diff(e, itersym)
    if itersym in a.free_symbols:  # degree >= 2 -> not affine
        return None
    return a, e.subs(itersym, 0)


def _same_injective_index(idx1, idx2, itersym) -> bool:
    """True iff ``idx1`` and ``idx2`` are the same injective affine ``a*i+b`` (``a != 0``) of ``itersym``."""
    sym = symbolic.pystr_to_symbolic(str(itersym))
    e1 = symbolic.pystr_to_symbolic(str(idx1))
    e2 = symbolic.pystr_to_symbolic(str(idx2))
    coeffs = _affine_coeffs(e1, sym)
    return coeffs is not None and coeffs[0] != 0 and sp.simplify(e1 - e2) == 0


def _dim_provably_disjoint(idx1, idx2, itersym, step=1, start=0) -> bool:
    """True iff ``idx1`` at any iteration can never equal ``idx2`` at any iteration.

        Linear-Diophantine criterion over the iteration counter ``t`` (``i == start + step*t``):
        with ``A_k = a_k*step`` and ``B_k = a_k*start + b_k``, ``A1*t1 + B1 == A2*t2 + B2`` is
        solvable iff ``gcd(A1, A2)`` divides ``B2 - B1``. Ranging over all integers ``t`` is
        conservative w.r.t. the bounded domain, hence sound. ``step``/``start`` default to the
        identity reparameterization.
    """
    f1 = _affine_coeffs(idx1, itersym)
    f2 = _affine_coeffs(idx2, itersym)
    if f1 is None or f2 is None:
        return False
    a1, b1 = f1
    a2, b2 = f2
    if not (a1.is_Integer and a2.is_Integer):
        return False
    step_s = symbolic.pystr_to_symbolic(step)
    start_s = symbolic.pystr_to_symbolic(start)
    # ``expand``, not ``sp.simplify``: only the difference needs cancelling, and only across a
    # product ``Add`` will not flatten. ``simplify`` looked like a hang on tiled write sets.
    A1 = a1 * step_s
    A2 = a2 * step_s
    B1 = a1 * start_s + b1
    B2 = a2 * start_s + b2
    diff = sp.expand(B2 - B1)
    if A1 == 0 and A2 == 0:
        return diff.is_number and diff != 0
    # A strided or offset loop yields symbolic ``A_k`` only when the step/start
    # are symbolic; the gcd criterion then cannot be evaluated -- stay safe.
    if not (A1.is_Integer and A2.is_Integer):
        return False
    g = sp.igcd(int(A1), int(A2))
    if g == 0:
        return diff.is_number and diff != 0
    if not diff.is_number:
        return False
    if not diff.is_Integer:
        return True
    return sp.Integer(diff) % g != 0


def loop_varying_symbols(loop: LoopRegion) -> Set[str]:
    """Symbols that can change while ``loop`` runs, other than its own iterator: nested loop and map
    iterators, and everything its interstate edges assign. Every other symbol a body subset
    mentions holds one value for the whole execution, which is what the disjointness tests need.
    """
    varying: Set[str] = set()
    for cfr in loop.all_control_flow_regions(recursive=True):
        if isinstance(cfr, LoopRegion) and cfr is not loop and cfr.loop_variable:
            varying.add(cfr.loop_variable)
    for state in loop.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                varying.update(node.map.params)
    for e in loop.all_interstate_edges():
        varying.update(e.data.assignments.keys())
    return varying


def _read_write_dims_disjoint(read: subsets.Subset, write: subsets.Subset, itersym, step, start,
                              varying: Set[str]) -> bool:
    """True iff some dimension's read/write point-indices are provably disjoint across every pair of
    in-domain iterations (step-aware, see :func:`_dim_provably_disjoint`).

        Unlike the propagate+intersect fallback this keeps constant disproving dimensions
        (``aa[0, i]`` vs ``aa[1, i-1]``). ``varying`` is :func:`loop_varying_symbols`.
    """
    rnd = list(read.ndrange())
    wnd = list(write.ndrange())
    if len(rnd) != len(wnd) or len(rnd) == 0:
        return False
    for (rb, re_, _), (wb, we_, _) in zip(rnd, wnd):
        if rb != re_ or wb != we_:  # non-point dimension: cannot decide here
            continue
        # SOUNDNESS: only ``itersym`` may vary here (it is reparameterized per access). A
        # body-varying symbol looks constant per dimension yet aliases as it sweeps.
        rw_syms = {s.name
                   for s in symbolic.pystr_to_symbolic(rb).free_symbols
                   } | {s.name
                        for s in symbolic.pystr_to_symbolic(wb).free_symbols}
        if rw_syms & varying:
            continue
        if _dim_provably_disjoint(rb, wb, itersym, step, start):
            return True
    return False


def _read_write_same_iteration(read: subsets.Subset, write: subsets.Subset, itersym) -> bool:
    """True iff some point dimension indexes read and write by the same injective function of
    ``itersym``, so any overlap happens within one iteration, where program order holds.
    """
    rnd = list(read.ndrange())
    wnd = list(write.ndrange())
    if len(rnd) != len(wnd) or len(rnd) == 0:
        return False
    for (rb, re_, _), (wb, we_, _) in zip(rnd, wnd):
        if rb != re_ or wb != we_:  # only point dimensions carry an injective index
            continue
        if _same_injective_index(rb, wb, itersym):
            return True
    return False


def _collision_forces_same_iteration(sub1: subsets.Subset, sub2: subsets.Subset, itersym, varying: Set[str]) -> bool:
    """Prove two point subsets of one container collide only when their iterations coincide.

        Substitute ``itersym`` by ``p`` in ``sub1`` and ``q`` in ``sub2`` and look for rationals
        ``lam_d`` with ``sum_d lam_d * (sub1[d]|p - sub2[d]|q) == p - q`` identically. Such a
        certificate holds for every parameter value, so it is sound; without one the caller keeps
        its may-alias answer. Catches transposed accesses (``cov[i,j]`` vs ``cov[j,i]``), where
        the iteration variable lands in different dimensions.
    """
    nd1 = list(sub1.ndrange())
    nd2 = list(sub2.ndrange())
    if len(nd1) != len(nd2) or len(nd1) == 0:
        return False
    p, q = sp.Dummy('p'), sp.Dummy('q')
    eqs = []
    params: Set[str] = set()
    for (b1, e1, _), (b2, e2, _) in zip(nd1, nd2):
        if b1 != e1 or b2 != e2:  # only point subsets participate in the collision system
            return False
        x1 = symbolic.pystr_to_symbolic(b1)
        x2 = symbolic.pystr_to_symbolic(b2)
        # SOUNDNESS: a symbol shared by both accesses must hold one value for the whole loop;
        # a body-varying one would certify ``p == q`` for accesses that really do alias.
        if any(str(s) in varying for s in x1.free_symbols) or any(str(s) in varying for s in x2.free_symbols):
            return False
        eqs.append(sp.expand(x1.subs(itersym, p) - x2.subs(itersym, q)))
        params |= {s for s in (set(x1.free_symbols) | set(x2.free_symbols)) if str(s) != str(itersym)}
    monomials = [p, q] + sorted(params, key=str)
    # Require every equation affine (total degree <= 1) in {p, q, params}; bail conservatively
    # on anything non-linear (e.g. ``A[i*i]``) where a linear certificate would be unsound.
    for eq in eqs:
        try:
            if sp.Poly(eq, *monomials).total_degree() > 1:
                return False
        except sp.PolynomialError:
            return False
    # Rationals ``lam_d`` with ``sum_d lam_d * eq_d == p - q`` as a polynomial identity: matching
    # monomial coefficients gives a linear system whose solution certifies a collision forces p == q.
    lambdas = list(sp.symbols(f'_l2m_lam0:{len(eqs)}'))
    diff = sp.expand(sum(l * e for l, e in zip(lambdas, eqs)) - (p - q))
    lin_eqs = [diff.coeff(mono) for mono in monomials]
    const = diff
    for mono in monomials:
        const = const.subs(mono, 0)
    lin_eqs.append(const)
    return len(sp.linsolve(lin_eqs, lambdas)) > 0


def _writes_may_overlap(m1: memlet.Memlet, m2: memlet.Memlet, itersym, step, start, varying: Set[str]) -> bool:
    """Whether two writes to the same container may hit one element from different iterations.

        Per dimension first (same injective index, or provably disjoint), then the whole-subset
        collision system. ``varying`` is :func:`loop_varying_symbols`.
    """
    nd1 = list(m1.subset.ndrange())
    nd2 = list(m2.subset.ndrange())
    if len(nd1) != len(nd2):
        return True
    for (b1, e1, _), (b2, e2, _) in zip(nd1, nd2):
        if b1 != e1 or b2 != e2:  # non-point range dimension: cannot decide here
            continue
        # Same injective index in both writes: a collision forces the iterations equal, so the
        # overlap stays inside one iteration, where the map body keeps program order.
        if _same_injective_index(b1, b2, itersym):
            return False
        if _dim_provably_disjoint(b1, b2, itersym, step, start):
            return False
    # No dimension settled it: the iter var may sit in different dimensions of the two writes
    # (a transpose), so try the whole-subset collision system.
    if _collision_forces_same_iteration(m1.subset, m2.subset, itersym, varying):
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

        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        sset = {}
        sset.update(sdfg.symbols)
        sset.update(sdfg.arrays)
        t = dtypes.result_type_of(infer_expr_type(start, sset), infer_expr_type(step, sset), infer_expr_type(end, sset))
        # Bounds must be integer-derived: non-sequential map schedules are otherwise invalid.
        if not t in dtypes.INTEGER_TYPES:
            return False

        # Loops containing break, continue, or returns may not be turned into a map.
        for blk in self.loop.all_control_flow_blocks():
            if isinstance(blk, (BreakBlock, ContinueBlock, ReturnBlock)):
                if not permissive:
                    return False

        # We cannot handle symbols read from data containers unless they are scalar.
        for expr in (start, end, step):
            if symbolic.contains_sympy_functions(expr):
                return False

        # A range symbol the body assigns moves into the loop_body NSDFG while the Map's range
        # stays outside it: ``Missing symbols on nested SDFG`` downstream.
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
            return False

        loop_states = set(self.loop.all_states())
        all_loop_blocks = set(self.loop.all_control_flow_blocks())

        # Cannot have StructView in loop body
        for loop_state in loop_states:
            if [n for n in loop_state.data_nodes() if isinstance(n.desc(sdfg), dt.StructureView)]:
                return False

        # At most one iteration carries no cross-iteration dependence: trivially DOALL, and the
        # analysis below would only be confounded by the clamped ``Min``/``Max`` bound.
        if loop_analysis.loop_provably_at_most_one_iteration(self.loop):
            return True

        # The dominator-heavy sort below only means anything when the body has interstate
        # assignments; with none (most loops) the result is identical, so skip it.
        symbols_that_may_be_used: Set[str] = {itervar}
        used_before_assignment: Set[str] = set()
        if any(e.data.assignments for block in self.loop.all_control_flow_blocks()
               for e in block.parent_graph.out_edges(block)):
            in_order_loop_blocks = list(
                cfg_analysis.blockorder_topological_sort(self.loop, recursive=True, ignore_nonstate_blocks=False))
            for block in in_order_loop_blocks:
                # The sort emits a ConditionalBlock before its branches though its out-edges run
                # after them, so count what every branch of an exhaustive one assigns as defined.
                if isinstance(block, ConditionalBlock) and any(c is None for c, _ in block.branches):
                    per_branch = []
                    for _cond, body in block.branches:
                        assigned_in_branch = set()
                        for inner in body.all_control_flow_blocks():
                            for ie in inner.parent_graph.out_edges(inner):
                                assigned_in_branch |= set(ie.data.assignments.keys())
                        per_branch.append(assigned_in_branch)
                    if per_branch:
                        symbols_that_may_be_used |= set.intersection(*per_branch)

                # ``read_symbols()`` below sees only interstate-edge reads, but a symbol read in
                # the block's own dataflow (``b[im]``) is read before the block's out-edges assign.
                try:
                    block_reads = {str(s) for s in block.free_symbols}
                except Exception:
                    block_reads = set()
                used_before_assignment |= (block_reads - symbols_that_may_be_used)
                for e in block.parent_graph.out_edges(block):
                    # Collect read-before-assigned symbols (states are in order; see
                    # blockorder_topological_sort above).
                    read_symbols = e.data.read_symbols()
                    read_symbols -= symbols_that_may_be_used
                    used_before_assignment |= read_symbols
                    # If symbol was read before it is assigned, the loop cannot be parallel
                    assigned_symbols = set()
                    for k, v in e.data.assignments.items():
                        try:
                            fsyms = {str(s) for s in symbolic.pystr_to_symbolic(v).free_symbols}
                        except AttributeError:
                            fsyms = set()
                        if k in fsyms and k not in symbols_that_may_be_used:
                            # ``k = f(k)`` unassigned earlier this iteration reads the previous
                            # one's value. Affine induction variables are in closed form upstream.
                            return False
                        if k not in fsyms:
                            assigned_symbols.add(k)
                    if assigned_symbols & used_before_assignment:
                        return False

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

        # ``read_and_write_sets()`` walks every state and edge and is only needed below, so a
        # loop refused by the cheaper checks above never pays for it (channel_flow: 41k of 44k).
        _, write_set = self.loop.read_and_write_sets()

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
                        # An empty memlet is an ordering edge, not a write, and its missing
                        # subset would read below as an unindexed whole-array write.
                        if e.data is None or e.data.is_empty():
                            continue
                        if e.data.dynamic and e.data.wcr is None:
                            # A dynamic write whose subset pins an axis to the iter var gives
                            # each iteration a disjoint slab, so a lane firing cannot race.
                            dst_subset = e.data.get_dst_subset(e, state)
                            if not (dst_subset and _check_range(dst_subset, a, itersym, b, step)):
                                return False

                        # Unique write index per iteration: match ``a*i+b``, ``|a| >= 1``, i the
                        # iteration variable (which must be used).
                        if e.data.wcr is None:
                            dst_subset = e.data.get_dst_subset(e, state)
                            ok = bool(dst_subset) and _check_range(dst_subset, a, itersym, b, step)
                            # NestedSDFG body propagates a whole-array external write hiding an
                            # inner per-iteration write; look past the connector.
                            if not ok and isinstance(e.src, nodes.NestedSDFG):
                                ok = _nested_writes_iter_indexed(e.src, e.src_conn, itersym, a, b, step)
                                # Write uniqueness is not enough: an inner read at another iter
                                # position (``a[i+1]`` while writing ``a[i]``) still races.
                                if ok and not _nested_reads_match_writes(e.src, e.src_conn, itersym, a, b, step):
                                    ok = False
                            if not ok and not permissive:
                                return False

                        write_memlets[dn.data].append(e.data)

        # A read of a written array must be loop-invariant or match the writes' ``a*i+b``:
        # ``a[i] = a[i+1]`` races, and the checks below only see overlaps within one iteration.
        for state in loop_states:
            for dn in state.data_nodes():
                data = dn.data
                if data not in write_memlets:
                    continue
                for e in state.out_edges(dn):
                    if e.data is None:
                        continue
                    src_subset = e.data.get_src_subset(e, state)
                    if src_subset is None:
                        continue
                    # Loop-invariant read (no itersym) -- safe, same input every iteration.
                    free = set()
                    for rb, re_, _ in src_subset.ndrange():
                        for expr in (rb, re_):
                            if hasattr(expr, 'free_symbols'):
                                free |= set(expr.free_symbols)
                    if itersym not in free:
                        continue
                    # itersym-dependent read: must match a*i+b like the writes, else this
                    # iteration reads a value another iteration writes.
                    if not _check_range(src_subset, a, itersym, b, step) and not permissive:
                        return False

        # Fixed for the whole loop, so compute once and share with every dependence test below.
        varying = loop_varying_symbols(self.loop)

        # Two individually injective writes still collide across iterations (``A[5*i]`` and
        # ``A[3*i]`` at ``A[15]``); allow only when some dimension is provably disjoint.
        for data, mmlts in write_memlets.items():
            distinct: Dict[str, memlet.Memlet] = {}
            for m in mmlts:
                if m.wcr is None:
                    distinct.setdefault(str(m.subset), m)
            reps = list(distinct.values())
            for x in range(len(reps)):
                for y in range(x + 1, len(reps)):
                    if _writes_may_overlap(reps[x], reps[y], itersym, step, start, varying) and not permissive:
                        return False

        # After looping over relevant writes, consider reads that may overlap
        for state in loop_states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                data = dn.data
                if data in write_memlets:
                    for e in state.out_edges(dn):
                        # As in the write scan: an empty memlet is an ordering edge, not a read.
                        if e.data is None or e.data.is_empty():
                            continue

                        # Container read AND written: match only if the locations can't race.
                        src_subset = e.data.get_src_subset(e, state)
                        if not self.test_read_memlet(sdfg, state, e, itersym, itervar, start, end, step, write_memlets,
                                                     e.data, src_subset, varying):
                            return False

        # Consider reads in inter-state edges (could be in assignments or in condition)
        isread_set: Set[memlet.Memlet] = set()
        for e in self.loop.all_interstate_edges():
            isread_set |= set(e.data.get_read_memlets(sdfg.arrays))
        for mmlt in isread_set:
            if mmlt.data in write_memlets:
                if not self.test_read_memlet(sdfg, None, None, itersym, itervar, start, end, step, write_memlets, mmlt,
                                             mmlt.subset, varying):
                    return False

        # Iteration variable + other symbols must not be used on later edges/blocks before
        # reassignment.
        in_order_blocks = list(
            cfg_analysis.blockorder_topological_sort(sdfg, recursive=True, ignore_nonstate_blocks=False))
        # First check the outgoing edges of the loop itself.
        reassigned_symbols: Set[str] = None
        for oe in graph.out_edges(self.loop):
            if symbols_that_may_be_used & oe.data.read_symbols():
                return False
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
                return False

            # Check inter-state edges
            reassigned_symbols = None
            for e in block.parent_graph.out_edges(block):
                if symbols_that_may_be_used & e.data.read_symbols():
                    return False

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
                         write_memlets: Dict[str, List[memlet.Memlet]], mmlt: memlet.Memlet, src_subset: subsets.Range,
                         varying: Set[str]):
        from dace.sdfg.propagation import propagate_subset, align_memlet

        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        data = mmlt.data

        if (mmlt.dynamic and mmlt.src_subset.num_elements() != 1):
            # If pointers are involved, give up
            return False
        if not _check_range(src_subset, a, itersym, b, step):
            # A loop-invariant read conflicts only if it overlaps a write (``a[0]`` vs ``a[1:N]``
            # does not), so defer to the propagated-overlap check below.
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
            # A one-sided copy memlet (``a[0:N] -> a``) leaves ``dst_subset`` None and carries
            # its subset in ``.subset``.
            write = candidate.dst_subset if candidate.dst_subset is not None else candidate.subset
            if read == write:
                continue
            # Step-aware per-dimension disjointness: no pair of iterations collides, so no
            # cross-iteration RAW. The fallback below drops constant dims and the stride.
            if _read_write_dims_disjoint(read, write, itersym, step, start, varying):
                continue
            # One dimension indexing read and write by the same injective function of the iter
            # var (syrk's ``C[i, :i+1]``) confines the overlap to a single iteration.
            if _read_write_same_iteration(read, write, itersym):
                continue
            # A transpose puts the iter var in different dimensions of read and write: no single
            # dimension settles it, but a distance-0 dependence still lives inside one iteration.
            if _collision_forces_same_iteration(read, write, itersym, varying):
                continue
            ridx = _dependent_indices(itervar, read)
            widx = _dependent_indices(itervar, write)
            indices = set(ridx) | set(widx)
            if not indices:
                indices = set(range(len(read)))
            read = _sanitize_by_index(indices, read)
            write = _sanitize_by_index(indices, write)
            if read == write:
                continue
            # Propagated read does not overlap with propagated write
            pwrite = propagate_subset([candidate],
                                      sdfg.arrays[data], [itervar],
                                      subsets.Range([(start, end, step)]),
                                      use_dst=True)
            t_pread = _sanitize_by_index(indices, pread.src_subset if pread.src_subset is not None else pread.subset)
            pwrite = _sanitize_by_index(indices, pwrite.dst_subset if pwrite.dst_subset is not None else pwrite.subset)
            if subsets.intersects(t_pread, pwrite) is False:
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
                # itersym in the subset means not thread-local; assumes the loop is a valid Map,
                # i.e. every edge carrying the array depends on itersym consistently.
                for e in state.out_edges(node):
                    src_subset = e.data.get_src_subset(e, state)
                    if src_subset and itervar in src_subset.free_symbols:
                        return False
                for e in state.in_edges(node):
                    dst_subset = e.data.get_dst_subset(e, state)
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

        # Headers at EVERY depth: recursing only into LoopRegion / ConditionalBlock stops at a
        # branch, and a container read below that never becomes a connector.
        for block in self.loop.all_control_flow_blocks():
            if isinstance(block, (LoopRegion, ConditionalBlock)):
                free_syms = {s for c in block.get_meta_codeblocks() for s in c.get_free_symbols()}
                read_set |= {s for s in free_syms if s in sdfg.arrays}

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

        # Create NestedSDFG and add the loop contents to it. Gather symbols defined in it.
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
                    # A mapping entry the mapped SDFG never declared (other passes write mapping
                    # entries without one) is not a reason to refuse: type it off the symbol.
                    nsdfg.symbols[s] = sdfg.symbols.get(s, symbolic.symbol(s).dtype)
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

        # Symbols the nested SDFG assigns itself are internal: mapping them makes the outer SDFG
        # appear to need them, and a later pruning pass then desyncs the mapping.
        internally_defined = set()
        for e in nsdfg.all_interstate_edges():
            internally_defined.update(e.data.assignments.keys())

        # Propagate free symbols in nested array shapes/strides/offsets: deepcopy carries them
        # but they must be added to the NestedSDFG's symbol mapping.
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

        # The registrations above can free a symbol after ``add_nested_sdfg`` fixed the mapping;
        # a free symbol missing from the node's mapping fails validation, so self-map leftovers.
        nconnectors = cnode.in_connectors.keys() | cnode.out_connectors.keys()
        for sym in sorted(nsdfg.free_symbols):
            if sym in nconnectors or sym in cnode.symbol_mapping:
                continue
            cnode.symbol_mapping[sym] = symbolic.pystr_to_symbolic(sym)
            if sym not in nsdfg.symbols and sym in sdfg.symbols:
                nsdfg.symbols[sym] = sdfg.symbols[sym]

        if (step < 0) == True:
            # If step is negative, we have to flip start and end to produce a correct map with a positive increment.
            start, end, step = end, start, -step

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        # Check intermediate nodes
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
            # Guaranteed scalar: can_be_applied rejects sympy functions in the loop expressions.
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

        # Direct edges among source and sink access nodes must pass through a tasklet; gather
        # them first. A list, not a set: ``MultiConnectorEdge`` hashes by id(), so set order varies.
        direct_edges: List[gr.MultiConnectorEdge[memlet.Memlet]] = []
        for n1 in source_nodes:
            if not isinstance(n1, nodes.AccessNode):
                continue
            for n2 in sink_nodes:
                if not isinstance(n2, nodes.AccessNode):
                    continue
                for e in body.edges_between(n1, n2):
                    e.data.try_initialize(sdfg, body, e)
                    direct_edges.append(e)
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
                t = body.add_tasklet(f"{src}_{dst}", {'__inp'}, {'__out'}, "__out =  __inp")
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
            # Endpoints must come from ``e``; ``n1``/``n2`` here are the leftover values of the
            # collection loops above, so every edge would be wired to the same last-seen pair.
            body.add_memlet_path(e.src,
                                 entry,
                                 t,
                                 memlet=memlet.Memlet(data=src, subset=e.data.src_subset),
                                 dst_conn=dst_conn)
            body.add_memlet_path(t,
                                 exit,
                                 e.dst,
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

        # Remove any variable this turned into a free symbol. Guard both branches with ``in
        # sdfg.symbols`` -- the array-descriptor-symbol propagation above may have cleared them.
        for var in sdfg.free_symbols - fsymbols:
            if var not in sdfg.symbols:
                continue
            if sdfg.parent_nsdfg_node:
                if var not in sdfg.parent_nsdfg_node.symbol_mapping:
                    sdfg.remove_symbol(var)
            else:
                sdfg.remove_symbol(var)

        # Deregistering does not un-free a symbol whose uses remain; the parent node must map it.
        if sdfg.parent_nsdfg_node is not None:
            pnode = sdfg.parent_nsdfg_node
            pconnectors = pnode.in_connectors.keys() | pnode.out_connectors.keys()
            for var in sorted((sdfg.free_symbols - fsymbols) - pnode.symbol_mapping.keys() - pconnectors):
                pnode.symbol_mapping[var] = symbolic.pystr_to_symbolic(var)

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
