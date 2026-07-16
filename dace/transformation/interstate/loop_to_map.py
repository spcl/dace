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
    """Whether every write to ``conn``'s array INSIDE ``nsdfg_node`` is indexed by the (mapped)
    iteration variable.

    A NestedSDFG loop body propagates a whole-array external write memlet (union over the loop)
    that hides the per-iteration write. Look past the connector: rewrite inner write subsets
    through ``symbol_mapping`` into the outer itersym; each must match the ``a*i+b`` pattern
    :func:`_check_range` enforces. Conservative: needs ≥1 inner write and ALL must pass; nested
    NestedSDFGs recurse, composing symbol maps.

    Example: ``for i: if c: b[i] = a[i] + 1`` after a ``LoopToMap -> MapToForLoop`` round-trip
    (guard forced a NestedSDFG body). The external connector memlet ``b[0:N]`` (union, no ``i``)
    fails _check_range, but the inner ``b[i]`` maps through {i: i} to ``b[i]`` → matches ``1*i+0``
    → independence proven → LoopToMap fires.

    :param nsdfg_node: NestedSDFG node feeding the outer write.
    :param conn: output connector (== inner array name) written.
    :param itersym: outer loop iteration symbol.
    :returns: True iff every inner write to ``conn`` is iter-indexed.
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
    """Whether every read of ``conn``'s array INSIDE ``nsdfg_node`` matches the SAME ``a*i+b``
    pattern as the writes, or is loop-invariant.

    Companion of :func:`_nested_writes_iter_indexed`, which only proves write UNIQUENESS. A
    loop-carried READ at a DIFFERENT iter-indexed position (``a[i] = ... + a[i+1] * ...``) still
    races: iteration ``i`` reads ``a[i+1]`` while ``i+1`` writes it. Conservative: each inner read
    must match ``a*i+b`` OR be loop-invariant (no outer ``itersym``). Nested NestedSDFGs recurse.

    :returns: True if no carried-read pattern found; False if any inner read hits the carrier
              array outside the write's affine form.
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
    """ Return ``(a, b)`` with ``expr == a*itersym + b``, or ``None`` if
        ``expr`` is not affine in ``itersym``.
    """
    e = sp.expand(symbolic.pystr_to_symbolic(expr))
    a = e.coeff(itersym, 1)
    b = e.coeff(itersym, 0)
    if sp.simplify(e - (a * itersym + b)) != 0:
        return None
    return a, b


def _same_injective_index(idx1, idx2, itersym) -> bool:
    """ True iff ``idx1`` and ``idx2`` are the SAME injective affine function ``a*i+b``
        (``a != 0``) of the iteration variable.

        When two accesses index a dimension by such a function, a collision on that dimension
        (``a*p+b == a*q+b``) forces the two iterations to coincide (``p == q``). Any overlap between
        them is therefore confined to a single iteration -- where program order in the map body is
        preserved -- and never becomes a cross-iteration dependency. Used both for write/write
        overlap (:func:`_writes_may_overlap`) and read/write RAW (:func:`_read_write_same_iteration`).

        Both indices and the iteration symbol are re-parsed through the symbol registry (via their
        string form) before the comparison: a read subset and a write subset can carry copies of the
        iteration variable that share the NAME ``i`` but different sympy assumptions, so ``idx1 -
        idx2`` would not simplify to zero and ``coeff`` would not see them as the same symbol.
        Canonicalizing drops the assumption metadata and makes both refer to one registry symbol.
    """
    sym = symbolic.pystr_to_symbolic(str(itersym))
    e1 = symbolic.pystr_to_symbolic(str(idx1))
    e2 = symbolic.pystr_to_symbolic(str(idx2))
    coeffs = _affine_coeffs(e1, sym)
    return coeffs is not None and coeffs[0] != 0 and sp.simplify(e1 - e2) == 0


def _dim_provably_disjoint(idx1, idx2, itersym, step=1, start=0) -> bool:
    """ True iff ``idx1`` at any iteration can never equal ``idx2`` at any
        iteration, for any pair of in-domain iterations and any loop bounds.

        Uses the linear-Diophantine solvability criterion. The iteration
        variable ``i`` only takes the strided values ``start + step * t``
        (``t`` a non-negative integer), so the accesses are reparameterized
        w.r.t. the iteration counter ``t``: ``a * i + b == (a*step)*t +
        (a*start + b)``. Writing ``A_k = a_k*step`` and ``B_k = a_k*start +
        b_k``, ``A1*t1 + B1 == A2*t2 + B2`` has an integer solution iff
        ``gcd(A1, A2)`` divides ``B2 - B1``. If it does not, the accesses
        never alias -- even accounting for the loop's stride (so a stride-2
        ``a[i] = a[i-1] + ...`` writes odd indices / reads even indices and is
        provably disjoint). The Diophantine ranges over ALL integers ``t``,
        which is conservative w.r.t. the bounded iteration domain (a solution
        only outside the domain still reports "may alias"), hence sound.

        ``step``/``start`` default to ``1``/``0`` (identity reparameterization)
        so callers that pass only the affine indices keep the classic behavior.
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
    A1 = sp.simplify(a1 * step_s)
    A2 = sp.simplify(a2 * step_s)
    B1 = sp.simplify(a1 * start_s + b1)
    B2 = sp.simplify(a2 * start_s + b2)
    diff = sp.simplify(B2 - B1)
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


def _read_write_dims_disjoint(read: subsets.Subset, write: subsets.Subset, itersym, step, start) -> bool:
    """ True iff some dimension's read/write point-indices are provably disjoint
        across every pair of in-domain iterations (step-aware
        linear-Diophantine, see :func:`_dim_provably_disjoint`).

        This is the read/write analog of the write/write per-dimension test in
        :func:`_writes_may_overlap`. It additionally accounts for the loop STEP
        (stride-2 write-odds/read-evens) and keeps CONSTANT disproving
        dimensions (``aa[0, i]`` write vs ``aa[1, i-1]`` read -- row 0 can never
        equal row 1), which the propagate+intersect fallback drops when it
        restricts to iteration-dependent dimensions only.
    """
    rnd = list(read.ndrange())
    wnd = list(write.ndrange())
    if len(rnd) != len(wnd) or len(rnd) == 0:
        return False
    for (rb, re_, _), (wb, we_, _) in zip(rnd, wnd):
        if rb != re_ or wb != we_:  # non-point dimension: cannot decide here
            continue
        # SOUNDNESS: only a dimension indexed purely by ``itersym`` (plus
        # literals) yields a valid cross-iteration disjointness verdict. A
        # dimension that also contains an INNER loop variable (``a[i-1, j-1]``
        # vs ``a[i, j]`` where ``j`` ranges) would be misjudged: ``j-1`` and
        # ``j`` look like distinct constants w.r.t. ``i`` yet the sets overlap
        # as ``j`` sweeps -- that is a genuine diagonal recurrence (TSVC s119).
        # Requiring free symbols to be a subset of ``{itersym}`` is
        # conservative (a loop-invariant symbol is also skipped) but sound.
        allowed = {itersym}
        # ``ndrange()`` yields plain ints as well as sympy exprs; ``sympify`` gives both a
        # uniform ``.free_symbols`` (an int has none) without a ``getattr`` guard.
        rw_syms = set(sp.sympify(rb).free_symbols) | set(sp.sympify(wb).free_symbols)
        if not rw_syms <= allowed:
            continue
        if _dim_provably_disjoint(rb, wb, itersym, step, start):
            return True
    return False


def _read_write_same_iteration(read: subsets.Subset, write: subsets.Subset, itersym) -> bool:
    """ True iff some point dimension indexes both ``read`` and ``write`` by the SAME injective
        affine function of the iteration variable (see :func:`_same_injective_index`).

        Then a read/write collision on that dimension forces the reading and writing iterations to
        coincide, so the read and write touch the same element only WITHIN one iteration (where the
        map body preserves program order) and never across iterations. This is the read/write analog
        of the injective-index rule in :func:`_writes_may_overlap`: it recognizes that iteration
        ``i`` reads and writes only its own slab (e.g. syrk's ``C[i, :i+1]`` row), so lifting the
        loop to a DOALL map is safe even though the read and write overlap in-iteration.

        Only ONE such dimension is required: if a collision on dimension ``d`` already forces
        ``p == q``, no pair of distinct iterations can address the same multidimensional element.
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


def _collision_forces_same_iteration(m1: memlet.Memlet, m2: memlet.Memlet, itersym) -> bool:
    """ Prove that two point-subset writes ``m1``, ``m2`` to the same container can only address
        the same element when their loop iterations coincide.

        Substitute the iteration variable by a fresh symbol ``p`` in ``m1`` and ``q`` in ``m2`` and
        build the collision system ``{m1[d]|i=p == m2[d]|i=q  for every dim d}`` (all other symbols
        are free parameters). If this affine system linearly implies ``p == q`` -- i.e. a rational
        combination ``sum_d lam_d * (m1[d]|p - m2[d]|q)`` equals ``p - q`` identically -- then a
        cross-iteration collision is impossible: any overlap between the two writes happens only
        within a single iteration, where program order in the map body is preserved.

        Whenever such a certificate exists, ``m1@p == m2@q`` forces ``p == q``, so the equality
        holds on the whole (affine) solution set and the proof is sound for every parameter value.
        Conservative: returns ``False`` on any non-point subset, any non-affine index, or when no
        certificate is found, so the caller keeps its safe ``may-overlap`` answer.

        This handles transpose/permutation-symmetric writes such as covariance's ``cov[i,j]`` and
        ``cov[j,i]``, where the iteration variable lands in *different* dimensions of the two writes
        so no single dimension is provably disjoint, yet a collision forces ``i == j`` (the diagonal
        of one iteration).
    """
    nd1 = list(m1.subset.ndrange())
    nd2 = list(m2.subset.ndrange())
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
    # Look for rationals ``lam_d`` with ``sum_d lam_d * eq_d == p - q`` as a polynomial identity in
    # {p, q, params}. Matching every monomial's coefficient to that of ``p - q`` gives a linear
    # system in the ``lam_d``; a solution is a soundness certificate that a collision forces p == q.
    lambdas = list(sp.symbols(f'_l2m_lam0:{len(eqs)}'))
    diff = sp.expand(sum(l * e for l, e in zip(lambdas, eqs)) - (p - q))
    lin_eqs = [diff.coeff(mono) for mono in monomials]
    const = diff
    for mono in monomials:
        const = const.subs(mono, 0)
    lin_eqs.append(const)
    return len(sp.linsolve(lin_eqs, lambdas)) > 0


def _writes_may_overlap(m1: memlet.Memlet, m2: memlet.Memlet, itersym, step=1, start=0) -> bool:
    """ Conservatively decide whether two write memlets to the same container
        can address the same element on different loop iterations. Returns
        ``False`` only if some subset dimension is provably disjoint (the
        multidimensional element can then never coincide), or if a collision
        provably forces the two iterations to coincide (see
        :func:`_collision_forces_same_iteration`).

        ``step``/``start`` describe the loop's strided iteration domain and are
        threaded into the per-dimension disjointness test, so a step-4 unrolled
        body's writes ``a[i], a[i+1], a[i+2], a[i+3]`` (all distinct modulo 4)
        are recognised as disjoint.
    """
    nd1 = list(m1.subset.ndrange())
    nd2 = list(m2.subset.ndrange())
    if len(nd1) != len(nd2):
        return True
    for (b1, e1, _), (b2, e2, _) in zip(nd1, nd2):
        if b1 != e1 or b2 != e2:  # non-point range dimension: cannot decide here
            continue
        # Both writes index this dim by the same injective function of the iter var: a collision
        # forces the two iterations equal, so they coincide only within one iteration (program
        # order in the map body), never across distinct iterations.
        if _same_injective_index(b1, b2, itersym):
            return False
        if _dim_provably_disjoint(b1, b2, itersym, step, start):
            return False
    # No single dimension settled it. Fall back to the whole-subset collision system: the iter var
    # may appear in different dimensions of the two writes (a transpose), yet a collision can still
    # force the two iterations to coincide.
    if _collision_forces_same_iteration(m1, m2, itersym):
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
            """Refuse the match. Reason dropped; diagnostics live in the pipeline driver."""
            return False

        # A loop pinned sequential is a deliberate fallback (the else branch of an
        # ``if cond: parallel else: sequential`` specialization); never parallelize it.
        if self.loop.pinned_sequential:
            return refuse("loop is pinned sequential")

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
        # Bounds must be integer-derived: non-sequential map schedules are otherwise invalid.
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

        # Refuse when the range (start/end/step) references a symbol the body defines via an
        # interstate assignment. After conversion the body → a ``loop_body`` NSDFG (assignment
        # goes with it) but the Map's range stays outer, referencing a symbol only defined inside
        # the NSDFG → ``Missing symbols on nested SDFG`` downstream.
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

        loop_states = set(self.loop.all_states())
        all_loop_blocks = set(self.loop.all_control_flow_blocks())

        # Cannot have StructView in loop body
        for loop_state in loop_states:
            if [n for n in loop_state.data_nodes() if isinstance(n.desc(sdfg), dt.StructureView)]:
                return refuse(f"loop body contains a StructureView in state {loop_state}")

        # A loop that provably runs at most once carries no cross-iteration dependence, so it is
        # trivially DOALL -- accept here, skipping the dependence analysis below (which a clamped
        # ``Max``/``Min`` bound would otherwise confound). This maps the single-iteration middle
        # segment a range split leaves behind (e.g. the ``{x}`` clamp of the s1113 broadcast split),
        # where the dependence analysis has nothing to prove. The structural guards above still gate.
        if loop_analysis.loop_provably_at_most_one_iteration(self.loop):
            return True

        # Collect symbol reads and writes from inter-state assignments. The read-before-assigned
        # analysis needs the loop's blocks in topological order, but that (dominator-heavy) sort is
        # only meaningful when the loop body actually has inter-state assignments. A loop whose
        # interstate edges carry none -- the common single-statement post-MapToForLoop case, ~all of
        # a stencil's probes -- has nothing to check, so skip the sort and leave
        # ``symbols_that_may_be_used`` at its initial ``{itervar}`` (identical to what the loop below
        # would produce with no assignments).
        symbols_that_may_be_used: Set[str] = {itervar}
        used_before_assignment: Set[str] = set()
        if any(e.data.assignments for block in self.loop.all_control_flow_blocks()
               for e in block.parent_graph.out_edges(block)):
            in_order_loop_blocks = list(
                cfg_analysis.blockorder_topological_sort(self.loop, recursive=True, ignore_nonstate_blocks=False))
            for block in in_order_loop_blocks:
                # A symbol read in the block's own dataflow (e.g. a memlet subset ``b[im]``) is read
                # before any symbol the block assigns on its out-edges; if the loop later reassigns it,
                # it is loop-carried. The per-edge ``read_symbols()`` below only sees interstate-edge
                # reads, so fold in these in-state reads.
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
                            # Self-recurrent assignment (k = f(k), e.g. k = k + inc) whose ``k`` has
                            # NOT been (re)assigned earlier this iteration is a loop-carried recurrence:
                            # each iteration reads the previous value, so the loop cannot be
                            # parallelized. Affine induction variables are substituted to a closed form
                            # upstream, so a self-recurrence that survives to here is a genuine carried
                            # dependency. If ``k`` was already assigned earlier in the iteration (e.g.
                            # reset ``k = 0`` then ``k = k + 1``) it is a loop-local counter, not carried,
                            # so it is handled by the read-before-assignment check below instead.
                            return refuse(f"self-recurrent carried symbol '{k}' (assignment {k} = {v})")
                        if k not in fsyms:
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

        # read_and_write_sets() walks every state/edge of the loop and is only needed from here
        # on (the per-array write analysis below). Computing it lazily -- after the cheaper
        # bound/break/StructView/carried-symbol refusals above -- lets a loop refused by any of
        # those return without paying for it (on channel_flow that is ~41k of ~44k probes).
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
                        if e.data.dynamic and e.data.wcr is None:
                            # Dynamic write (no WCR) is safe across iterations if its dst subset
                            # pins an axis to the iter var (same ``a*i+b`` as non-dynamic below):
                            # each iteration writes a disjoint slab, so a lane firing or not can't
                            # race another iteration's write.
                            dst_subset = e.data.get_dst_subset(e, state)
                            if not (dst_subset and _check_range(dst_subset, a, itersym, b, step)):
                                return refuse(f"dynamic write to {dn.data} is not indexed by the iteration variable "
                                              f"- dst_subset={dst_subset}")
                        if e.data is None:
                            continue

                        # Unique write index per iteration: match ``a*i+b``, ``|a| >= 1``, i the
                        # iteration variable (which must be used).
                        if e.data.wcr is None:
                            dst_subset = e.data.get_dst_subset(e, state)
                            ok = bool(dst_subset) and _check_range(dst_subset, a, itersym, b, step)
                            # NestedSDFG body propagates a whole-array external write hiding an
                            # inner per-iteration write; look past the connector.
                            if not ok and isinstance(e.src, nodes.NestedSDFG):
                                ok = _nested_writes_iter_indexed(e.src, e.src_conn, itersym, a, b, step)
                                # NSDFG descent only proves WRITE uniqueness. A carried READ at a
                                # DIFFERENT iter position (``a[i+1]`` while writing ``a[i]``) is a
                                # forward/backward dependence that races. Require every inner read
                                # of ``conn`` to match the writes' ``a*i+b`` (or be loop-invariant).
                                if ok and not _nested_reads_match_writes(e.src, e.src_conn, itersym, a, b, step):
                                    ok = False
                            if not ok and not permissive:
                                return refuse(f"write to {dn.data} is not uniquely indexed by the iteration variable "
                                              f"(needs an a*i+b subset) - dst_subset={dst_subset}")

                        write_memlets[dn.data].append(e.data)

        # Carried-read check: for every array also written, each in-loop READ subset must be
        # loop-invariant (no itersym) or match the writes' ``a*i+b``. A read at a DIFFERENT iter
        # offset (``a[i+1]`` while writing ``a[i]``) is a carried dependency that races (iter ``i``
        # reads ``a[i+1]`` while ``i+1`` writes it). The same-iteration disjoint check below only
        # catches within-ONE-iteration overlaps; cross-iteration carries slip through.
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
                        return refuse(f"read of {data} at {src_subset} is iter-indexed but does not match the "
                                      f"write pattern a*i+b -- loop-carried forward/backward dependency")

        # Two distinct-affine writes to the same container can hit the same element on different
        # iterations even when each is individually injective (``A[5*i]`` and ``A[3*i]`` collide
        # at ``A[15]``); parallelizing reorders them. Allow only if some dim is provably disjoint
        # for all iterations (``A[2*i]`` vs ``A[2*i+1]``).
        for data, mmlts in write_memlets.items():
            distinct: Dict[str, memlet.Memlet] = {}
            for m in mmlts:
                if m.wcr is None:
                    distinct.setdefault(str(m.subset), m)
            reps = list(distinct.values())
            for x in range(len(reps)):
                for y in range(x + 1, len(reps)):
                    if _writes_may_overlap(reps[x], reps[y], itersym, step, start) and not permissive:
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

                        # Container read AND written: match only if the locations can't race.
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

        # Iteration variable + other symbols must not be used on later edges/blocks before
        # reassignment.
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
        from dace.sdfg.propagation import propagate_subset, align_memlet

        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        data = mmlt.data

        if (mmlt.dynamic and mmlt.src_subset.num_elements() != 1):
            # If pointers are involved, give up
            return False
        if not _check_range(src_subset, a, itersym, b, step):
            # ``_check_range`` accepts only reads that MOVE with the iteration (some dim
            # ``a*i+b``, ``|a| >= 1``). An itersym read not matching that is conservatively a
            # conflict. A loop-INVARIANT read (no itersym) is a conflict only if it overlaps a
            # write: ``a[0]`` is safe against writes to ``a[1:N]`` but not ``a[0:N]``. Defer both
            # to the propagated-overlap check below.
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
            # A one-sided copy memlet (e.g. a whole-array ``a[0:N] -> a`` boundary
            # passthrough) carries its subset in ``.subset`` and leaves
            # ``.dst_subset`` None; fall back to ``.subset`` so the dependency test
            # doesn't crash on ``None.ndrange()``.
            write = candidate.dst_subset if candidate.dst_subset is not None else candidate.subset
            if read == write:
                continue
            # Step-aware per-dimension disjointness: if any dimension's read/write
            # indices can never coincide for any pair of in-domain iterations
            # (linear-Diophantine over the strided iteration counter), the accesses
            # never alias -- no cross-iteration RAW. This is strictly more precise
            # than the propagate+intersect fallback below, which drops constant
            # disproving dims and ignores the loop stride.
            if _read_write_dims_disjoint(read, write, itersym, step, start):
                continue
            # Same-iteration collision: if some point dimension indexes both the read and the write
            # by the same injective function of the iter var (e.g. syrk's ``C[i, :i+1]`` -- row ``i``
            # read and written by iteration ``i``), a collision forces the read and write iterations
            # to coincide. The overlap is then confined to one iteration (program order in the map
            # body preserves it) and is never a cross-iteration RAW. Mirrors the write/write
            # injective-index rule in :func:`_writes_may_overlap`.
            if _read_write_same_iteration(read, write, itersym):
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
                for e in state.out_edges(node):
                    src_subset = e.data.get_src_subset(e, state)
                    # itersym in the subset → not thread-local. Assumes the loop is a valid Map
                    # (all edges carrying the array depend on itersym consistently).
                    if src_subset and itervar in src_subset.free_symbols:
                        return False
                for e in state.in_edges(node):
                    dst_subset = e.data.get_dst_subset(e, state)
                    # itersym in the subset → not thread-local. Assumes the loop is a valid Map
                    # (all edges carrying the array depend on itersym consistently).
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

        # Symbols the nested SDFG assigns on its own interstate edges are internal → keep off the
        # node's ``symbol_mapping``. Surfacing them makes the outer SDFG appear to need them as
        # free symbols; a later pruning pass removing them from ``sdfg.symbols`` desyncs the
        # mapping (codegen ``sdfg.symbols[sym]`` → KeyError).
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
