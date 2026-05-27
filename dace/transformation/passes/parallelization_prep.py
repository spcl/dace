# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone parallelization-preparation passes.

These rewrite loops so that ``LoopToMap`` can parallelize more of them. They are
plain :class:`~dace.transformation.pass_pipeline.Pass` objects so the
``parallelize`` pipeline (and anyone else) can just compose them:

- :class:`ShortLoopUnroll` -- fully unroll constant-trip loops with at most
  ``unroll_limit`` iterations, turning small recurrence / reduction loops into
  inline straight-line code instead of atomically-parallelized maps.
- :class:`BestEffortLoopPeeling` -- search front/back/both peels of 1..``peel_limit``
  boundary iterations, keep the one that unblocks the most maps, revert if none
  helps, and prune the now-dead boundary guard from the remainder.

Transformation classes are imported lazily inside the methods: importing them at
module load would cycle (this package is imported by the transformations those
imports pull in).
"""
import copy
from typing import Any, Dict, Optional

from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl

#: Default trip-count threshold below which a constant-trip loop is unrolled.
DEFAULT_UNROLL_LIMIT = 8
#: Default maximum number of iterations peeled (per side) when searching for a
#: peel that unblocks parallelization.
DEFAULT_PEEL_LIMIT = 8
#: Names under which a (floor) modulo may be defined in a subset expression. The
#: peel modulo-rewrite recognises all of these -- ``sympy.Mod`` (the ``%`` operator)
#: and the equivalent helper-function spellings -- so it folds a wrap-around access
#: regardless of which representation introduced it.
_MODULO_FUNC_NAMES = frozenset({'Mod', 'py_mod', 'Modulo', 'mod', 'floor_mod'})


def _loops(sdfg: SDFG):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _unique_block_label(sdfg: SDFG, base: str) -> str:
    """A control-flow-block label not currently used anywhere in ``sdfg``."""
    import itertools
    existing = {b.label for b in sdfg.all_control_flow_blocks()}
    for n in itertools.count():
        cand = f'{base}_p{n}'
        if cand not in existing:
            return cand


def _constant_trip_count(loop: LoopRegion, sdfg: SDFG) -> Optional[int]:
    """The exact iteration count of ``loop`` if it is constant, else ``None``
    (matches ``len(range(0, end - start + 1, stride))``, i.e. LoopUnroll's count)."""
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or not loop.loop_variable:
        return None
    if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
        return None
    try:
        stride_val = int(symbolic.evaluate(stride, sdfg.constants))
        diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
    except (TypeError, ValueError):
        return None
    if stride_val <= 0 or diff <= 0:
        return None
    return len(range(0, diff, stride_val))


@properties.make_properties
class ShortLoopUnroll(ppl.Pass):
    """Fully unroll every constant-trip loop with at most ``unroll_limit`` iterations."""

    CATEGORY: str = 'Optimization Preparation'

    unroll_limit = properties.Property(
        dtype=int,
        default=DEFAULT_UNROLL_LIMIT,
        desc='Fully unroll constant-trip loops with at most this many iterations (0 disables).')

    def __init__(self, unroll_limit: int = DEFAULT_UNROLL_LIMIT):
        self.unroll_limit = unroll_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Unroll short constant-trip loops; returns the number unrolled or None.

        Re-collects after each unroll since unrolling rewrites the control-flow
        structure (and may expose newly-constant inner loops).
        """
        if self.unroll_limit <= 0:
            return None
        from dace.transformation.interstate.loop_unroll import LoopUnroll
        unrolled = 0
        changed = True
        while changed:
            changed = False
            for loop in _loops(sdfg):
                trip = _constant_trip_count(loop, sdfg)
                if trip is None or trip > self.unroll_limit:
                    continue
                try:
                    LoopUnroll().apply_to(sdfg=loop.sdfg, loop=loop)
                except Exception:
                    continue  # not unrollable in this context; leave it for LoopToMap
                unrolled += 1
                changed = True
                break
        return unrolled or None


@properties.make_properties
class BestEffortLoopPeeling(ppl.Pass):
    """Best-effort loop peeling that unblocks parallelization.

    For each of front / back / both and each peel count ``k`` in
    ``1..peel_limit``, peel ``k`` boundary iterations off the loops and run a
    cheap candidate check (scalar fission -> symbol propagation -> constant
    propagation), then COUNT the loops ``LoopToMap`` *could* parallelize via
    ``can_be_applied_to`` -- WITHOUT applying it (peeling is a preparation pass;
    the actual parallelization is the pipeline's job). Keep the single peel that
    yields the most mappable loops; if none beats the no-peel baseline, leave the
    SDFG unpeeled. The search runs on ``copy.deepcopy`` copies (revertible by
    construction); only the winning peel is applied to the real SDFG.
    """

    CATEGORY: str = 'Optimization Preparation'

    peel_limit = properties.Property(
        dtype=int,
        default=DEFAULT_PEEL_LIMIT,
        desc='Try peeling 1..peel_limit iterations from the front, the back, and both; keep the '
        'peel that produces the most maps and revert if none beats the no-peel baseline (0 disables).')

    def __init__(self, peel_limit: int = DEFAULT_PEEL_LIMIT):
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _peel_one_loop(self, sdfg: SDFG, loop: LoopRegion, count: int, direction: str) -> bool:
        """Peel ``count`` iterations off ``loop`` (``'front'`` / ``'back'`` /
        ``'both'``). Returns whether it peeled. Does NOT prune -- callers prune.

        Uses ``verify=False`` so loops with symbolic bounds (where peeling a
        boundary is most useful) are not rejected by ``LoopUnroll``'s
        constant-size gate; infeasible loops simply raise and are skipped.
        """
        from dace.transformation.interstate.loop_peeling import LoopPeeling
        sides = {'front': [True], 'back': [False], 'both': [True, False]}[direction]
        # A loop short enough to be fully consumed by the peel is the unroll
        # pass's job, not peeling's.
        trip = _constant_trip_count(loop, sdfg)
        if trip is not None and trip <= count * len(sides):
            return False
        did = False
        for idx, begin in enumerate(sides):
            try:
                if idx > 0:
                    # LoopPeeling names the peeled iteration regions after the loop
                    # label; a second peel on the same loop would reuse those names.
                    # Relabel the remainder loop first so front/back peels differ.
                    loop.label = _unique_block_label(loop.sdfg, loop.label)
                # Properties must go through ``options=`` -- bare kwargs are not
                # applied, leaving ``count`` at LoopUnroll's default 0 (a no-op).
                LoopPeeling().apply_to(sdfg=loop.sdfg,
                                       loop=loop,
                                       verify=False,
                                       options={
                                           'count': count,
                                           'begin': begin
                                       })
                did = True
            except Exception:
                continue
        return did

    def _peel_loops(self, sdfg: SDFG, count: int, direction: str) -> int:
        """Peel every peelable loop in ``sdfg`` (used directly in tests). Prunes
        the dead boundary guards a peel leaves behind."""
        peeled = sum(int(self._peel_one_loop(sdfg, loop, count, direction)) for loop in _loops(sdfg))
        if peeled:
            self._clean_peeled_remainder(sdfg)
        return peeled

    def _isolate_loop(self, loop: LoopRegion, sdfg: SDFG):
        """Build a throwaway mini-SDFG containing only a copy of ``loop`` and the
        arrays / symbols it references, so the peel search can experiment cheaply
        without deep-copying the whole SDFG. Returns ``(mini, mini_loop)`` or
        ``(None, None)`` if the nest cannot be isolated cleanly."""
        import dace
        from dace import serialize
        try:
            mini = dace.SDFG(sdfg.name + '_peelprobe')
            for sname, stype in sdfg.symbols.items():
                mini.add_symbol(sname, stype)
            # Arrays referenced by the loop body and its interstate edges.
            needed = set()
            for st in loop.all_states():
                for n in st.data_nodes():
                    needed.add(n.data)
                for e in st.edges():
                    if e.data is not None and e.data.data is not None:
                        needed.add(e.data.data)
            for e in loop.all_interstate_edges():
                needed |= e.data.used_arrays(sdfg.arrays)
            for name in needed:
                if name in sdfg.arrays and name not in mini.arrays:
                    mini.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]))
            # Copy the loop region via a JSON round-trip (reparents to `mini`).
            mini_loop = serialize.from_json(serialize.to_json(loop), context={'sdfg': mini})
            mini.add_node(mini_loop, is_start_block=True)
            mini.reset_cfg_list()
            mini.validate()
            return mini, mini_loop
        except Exception:
            return None, None

    def _best_peel_for(self, loop: LoopRegion, sdfg: SDFG):
        """Find the ``(count, direction)`` peel that unblocks the most maps for
        ``loop``, experimenting on an isolated mini-SDFG; ``None`` if no peel
        beats leaving it alone (or the loop already maps)."""
        # Cheap pre-filter: a loop LoopToMap can already map needs no peel. This
        # skips the (comparatively expensive) isolate-and-search for the common
        # case, leaving the search only for genuinely stuck loops.
        from dace.transformation.interstate.loop_to_map import LoopToMap
        try:
            if LoopToMap.can_be_applied_to(sdfg, loop=loop):
                return None
        except Exception:
            pass
        mini, _ = self._isolate_loop(loop, sdfg)
        if mini is None:
            return None
        base = copy.deepcopy(mini)
        try:
            baseline = self._mappable_loop_count(base)
        except Exception:
            return None

        best_count, best = baseline, None
        for direction in ('front', 'back', 'both'):
            for count in range(1, self.peel_limit + 1):
                cand = copy.deepcopy(mini)
                cloops = _loops(cand)
                if not cloops:
                    break
                if not self._peel_one_loop(cand, cloops[0], count, direction):
                    continue
                self._clean_peeled_remainder(cand)
                try:
                    cand.validate()  # only a peel that stays valid is a working parameter
                    n_mappable = self._mappable_loop_count(cand)
                except Exception:
                    continue
                if n_mappable > best_count:
                    best_count, best = n_mappable, (count, direction)
        return best

    def _equality_guard_values(self, loop: LoopRegion):
        """Loop-invariant values ``x`` for which the body has an ``if i == x`` guard
        (or ``x == i``). These are index-set-split points: a special-case iteration
        that blocks parallelization can be carved out as [start, x-1] + {x} + [x+1,
        end] (a boundary ``x`` simply drops the empty side -- see
        :meth:`_split_loop_at`). ``x`` must not depend on the loop variable."""
        import ast
        from dace.sdfg.state import ConditionalBlock
        ivar_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
        values = []
        for cb in [b for b in loop.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]:
            for cond, _ in cb.branches:
                if cond is None:
                    continue
                try:
                    node = cond.code[0]
                except (AttributeError, IndexError, TypeError):
                    continue
                if isinstance(node, ast.Expr):
                    node = node.value
                if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
                    continue
                left, right = node.left, node.comparators[0]

                def is_ivar(n):
                    return isinstance(n, ast.Name) and n.id == loop.loop_variable

                if is_ivar(left) and not is_ivar(right):
                    other = right
                elif is_ivar(right) and not is_ivar(left):
                    other = left
                else:
                    continue
                try:
                    x = symbolic.pystr_to_symbolic(ast.unparse(other))
                except Exception:
                    continue
                if ivar_sym not in x.free_symbols and x not in values:
                    values.append(x)
        return values

    def _split_loop_at(self, sdfg: SDFG, loop: LoopRegion, x) -> bool:
        """Index-set-split ``loop`` at iteration ``x`` into the range segments
        [start, x-1], the single iteration {x}, and [x+1, end] -- each a clone of the
        body, wired in sequence in place of the loop. A boundary ``x`` drops the side
        that would be empty: ``x == start`` (e.g. ``i == 0``) has no preceding
        iterations, so only {x} + [x+1, end] are emitted; ``x == end`` (the last
        iteration) has no following iterations, so only [start, x-1] + {x}. The
        enclosing-range-aware ``LiftTrivialIf`` (run by
        :meth:`_clean_peeled_remainder` afterwards) then resolves the ``if i == x``
        guard per segment: a contradiction in the range segments, a tautology in the
        single middle iteration. Unit stride only. Returns whether it split."""
        import copy
        from dace.properties import CodeBlock
        from dace.sdfg.sdfg import InterstateEdge
        from dace.transformation.passes.analysis import loop_analysis
        stride = loop_analysis.get_loop_stride(loop)
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        try:
            if stride is None or int(symbolic.evaluate(stride, sdfg.constants)) != 1:
                return False
        except (TypeError, ValueError):
            return False
        if start is None or end is None:
            return False
        ivar = loop.loop_variable
        parent = loop.parent_graph
        # Drop a range segment that is provably empty at a boundary split point.
        want_before = symbolic.simplify(x - start) != 0  # x != start -> [start, x-1] is non-empty
        want_after = symbolic.simplify(x - end) != 0  # x != end   -> [x+1, end] is non-empty

        def clone_segment() -> LoopRegion:
            seg = copy.deepcopy(loop)
            seg.label = _unique_block_label(sdfg, loop.label)
            parent.add_node(seg)  # register so the next unique-label query sees it
            return seg

        chain = []
        if want_before:
            before = clone_segment()
            before.loop_condition = CodeBlock(f'{ivar} < ({x})')  # [start, x-1]
            chain.append(before)
        at = clone_segment()  # {x}: a single iteration
        at.init_statement = CodeBlock(f'{ivar} = ({x})')
        at.loop_condition = CodeBlock(f'{ivar} < ({x}) + 1')
        chain.append(at)
        if want_after:
            after = clone_segment()
            after.init_statement = CodeBlock(f'{ivar} = ({x}) + 1')  # [x+1, end], original condition
            chain.append(after)

        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop
        for ie in in_edges:
            parent.add_edge(ie.src, chain[0], ie.data)
            parent.remove_edge(ie)
        for prev, nxt in zip(chain, chain[1:]):
            parent.add_edge(prev, nxt, InterstateEdge())
        for oe in out_edges:
            parent.add_edge(chain[-1], oe.dst, oe.data)
            parent.remove_edge(oe)
        parent.remove_node(loop)
        if is_start:
            parent.start_block = parent.node_id(chain[0])
        parent.reset_cfg_list()
        return True

    def _best_split_for(self, loop: LoopRegion, sdfg: SDFG):
        """The ``if i == x`` guard value whose index-set split unblocks the most maps
        for ``loop`` (probed on an isolated copy), or ``None`` if splitting does not
        help or the loop already maps."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        try:
            if LoopToMap.can_be_applied_to(sdfg, loop=loop):
                return None
        except Exception:
            pass
        guards = self._equality_guard_values(loop)
        if not guards:
            return None
        mini, _ = self._isolate_loop(loop, sdfg)
        if mini is None:
            return None
        try:
            baseline = self._mappable_loop_count(copy.deepcopy(mini))
        except Exception:
            return None
        best_count, best = baseline, None
        for x in guards:
            cand = copy.deepcopy(mini)
            cloops = _loops(cand)
            if not cloops or not self._split_loop_at(cand, cloops[0], x):
                continue
            self._clean_peeled_remainder(cand)
            try:
                cand.validate()
                n_mappable = self._mappable_loop_count(cand)
            except Exception:
                continue
            if n_mappable > best_count:
                best_count, best = n_mappable, x
        return best

    def _prune_dead_loop_branches(self, sdfg: SDFG):
        """Collapse the boundary guards a peel leaves in the remainder loop body so
        what is left is clean affine code LoopToMap can map.

        Delegates to the iteration-range-aware :class:`LiftTrivialIf`: a guard that
        is false over the whole remainder range (``if i == N-1`` once the remainder
        is ``[0, N-2]``, the ``i == 0`` / ``i == 1`` arms of an if/elif chain peeled
        off the front) is a contradiction whose branch is dropped, and the surviving
        always-true guard (the if/elif's else body, lowered to ``if not(i==0): ...``)
        is a tautology that gets lifted up. The frontend lowers an if/elif chain to
        *nested* conditionals, and one ``LiftTrivialIf`` pass collapses one nesting
        level (lifting the outer guard exposes the inner one), so it is run to a
        fixpoint -- the conditional count strictly decreases, so this terminates --
        and ``EmptyStateElimination`` then tidies the empty boundary states a lift
        leaves behind."""
        from dace.sdfg.state import ConditionalBlock
        from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf

        def num_conditionals() -> int:
            return sum(1 for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock))

        lift = LiftTrivialIf()
        prev = num_conditionals()
        while prev > 0:
            lift.apply_pass(sdfg, {})
            cur = num_conditionals()
            if cur >= prev:
                break  # no further trivial conditional to collapse
            prev = cur
        EmptyStateElimination().apply_pass(sdfg, {})

    def _clean_peeled_remainder(self, sdfg: SDFG):
        """Post-peel cleanup so the remainder body is affine and mappable: collapse
        the now-dead boundary guards (:meth:`_prune_dead_loop_branches`) and rewrite
        the modulo subsets the peel made affine over the shortened range
        (:meth:`_rewrite_modulo_over_range`)."""
        self._prune_dead_loop_branches(sdfg)
        self._rewrite_modulo_over_range(sdfg)

    @staticmethod
    def _provably_nonneg(x) -> bool:
        """Whether ``x`` is provably ``>= 0`` -- a concrete non-negative number once
        simplified (the deciding differences of a range bound reduce to numbers)."""
        s = symbolic.simplify(x)
        return s.is_number and s >= 0

    def _nonneg_assuming_large_modulus(self, x, m) -> bool:
        """Whether ``x`` is provably ``>= 0`` given the modulus ``m`` is at least
        ``peel_limit + 1``. Peeling a symbolic loop by ``k <= peel_limit`` iterations
        already presupposes the loop runs at least ``k`` times (the peeled iterations
        execute unconditionally), so when ``m`` is the trip count it is ``> peel_limit``.
        Handles a concrete number, or an expression affine and non-decreasing in the
        single modulus symbol ``m`` with no other free symbols (e.g. ``m - 1 >= 0``)."""
        import sympy
        s = symbolic.simplify(x)
        if s.is_number:
            return bool(s >= 0)
        if not isinstance(m, sympy.Symbol):
            return False
        c1 = s.coeff(m, 1)
        c0 = s - c1 * m
        if (m in c1.free_symbols or m in c0.free_symbols or c0.free_symbols or c1.free_symbols
                or not (c1.is_number and c0.is_number)):
            return False  # not affine in m alone
        if c1 < 0:
            return False  # decreasing in m -> not bounded below by the m-minimum
        return bool(c0 + c1 * (self.peel_limit + 1) >= 0)

    def _enclosing_loop_ranges(self, block) -> Dict[Any, Any]:
        """``{loop_variable: (start, end)}`` for every ``LoopRegion`` enclosing
        ``block``, with inclusive bounds. Empty for a peeled iteration region (no
        enclosing loop), whose body holds a fixed, already-substituted index."""
        from dace.sdfg.state import LoopRegion
        from dace.transformation.passes.analysis import loop_analysis
        ranges: Dict[Any, Any] = {}
        graph = block.parent_graph
        seen = set()
        while graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            if isinstance(graph, LoopRegion) and graph.loop_variable:
                start = loop_analysis.get_init_assignment(graph)
                end = loop_analysis.get_loop_end(graph)
                if start is not None and end is not None:
                    ranges[symbolic.pystr_to_symbolic(graph.loop_variable)] = (start, end)
            graph = getattr(graph, 'parent_graph', None)
        return ranges

    @staticmethod
    def _modulo_operands(node):
        """``(arg, m)`` if ``node`` is a modulo of two operands -- either
        ``sympy.Mod`` (the ``%`` operator) or a recognised floor-mod helper-function
        call (see :data:`_MODULO_FUNC_NAMES`) -- else ``None``. Lets the rewrite
        accept whichever spelling defines the modulo."""
        import sympy
        if isinstance(node, sympy.Mod) and len(node.args) == 2:
            return node.args[0], node.args[1]
        name = getattr(getattr(node, 'func', None), '__name__', None)
        if name in _MODULO_FUNC_NAMES and len(node.args) == 2:
            return node.args[0], node.args[1]
        return None

    @staticmethod
    def _modulo_nodes(expr) -> set:
        """Every modulo subexpression of ``expr``: both the ``%`` operator
        (``sympy.Mod``) and the floor-mod helper-function spellings (see
        :data:`_MODULO_FUNC_NAMES`), so a wrap-around index is found regardless of
        which representation introduced it."""
        import sympy
        mods = set(expr.atoms(sympy.Mod))
        mods |= {f for f in expr.atoms(sympy.Function) if getattr(f.func, '__name__', None) in _MODULO_FUNC_NAMES}
        return mods

    def _modulo_to_affine(self, mod, ranges: Dict[Any, Any]):
        """If ``mod`` is a modulo ``arg % m`` (operator or helper function) with
        ``arg`` affine in the enclosing loop variables and provably confined to a
        single band ``[t*m, (t+1)*m - 1]`` over their ranges, return the equivalent
        affine ``arg - t*m`` (so the result lands in ``[0, m-1]`` -- i.e. ``arg % m``);
        else ``None``. This makes a peeled wrap-around access affine WITHOUT relying
        on C's truncated ``%`` (which would mis-handle a negative or beyond-``m``
        argument): the remainder of ``arr[(i + k) % N]`` rewrites to ``i + k`` (band
        ``t = 0``), and a peeled wrapping iteration ``Mod(N, N)`` rewrites to ``0``
        (band ``t = 1``), ``Mod(-1, N)`` to ``N - 1`` (band ``t = -1``). ``m`` may be
        symbolic."""
        operands = self._modulo_operands(mod)
        if operands is None:
            return None
        arg, m = operands
        # Reduce ``arg`` to its extremes over the enclosing loop box: affine in each
        # ranged variable, so the min/max sit at the range ends (per slope sign). A
        # peeled region has no enclosing loop, so ``arg`` is already a fixed point.
        lo = hi = arg
        for lv, (start, end) in ranges.items():
            if lv not in arg.free_symbols:
                continue
            a = arg.coeff(lv, 1)
            if lv in a.free_symbols or lv in (arg - a * lv).free_symbols:
                return None  # not affine in this loop variable
            if self._provably_nonneg(a):
                lo, hi = lo.subs(lv, start), hi.subs(lv, end)
            elif self._provably_nonneg(-a):
                lo, hi = lo.subs(lv, end), hi.subs(lv, start)
            else:
                return None  # indeterminate slope sign
        # Find the band ``t`` such that ``arg - t*m`` stays in ``[0, m-1]`` over the
        # whole range. Peeling shifts the argument by a bounded number of strides, so
        # the band index is within +/-(peel_limit + 1).
        for t in range(-(self.peel_limit + 1), self.peel_limit + 2):
            if (self._nonneg_assuming_large_modulus(lo - t * m, m)
                    and self._nonneg_assuming_large_modulus(m - 1 - (hi - t * m), m)):
                return arg - t * m
        return None

    def _loop_own_ranges(self, loop: LoopRegion) -> Dict[Any, Any]:
        """``{loop_variable: (start, end)}`` for ``loop`` itself (inclusive), the
        single-entry range box :meth:`_modulo_to_affine` reduces a modulo argument
        over; empty if the bounds are not recoverable."""
        from dace.transformation.passes.analysis import loop_analysis
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None or not loop.loop_variable:
            return {}
        return {symbolic.pystr_to_symbolic(loop.loop_variable): (start, end)}

    def _has_wrapping_modulo(self, loop: LoopRegion) -> bool:
        """Whether ``loop``'s body holds a memlet-subset modulo ``Mod(arg, m)`` whose
        ``arg`` is affine in the loop variable yet spans more than one band over the
        loop's full range -- so it does *not* reduce to a single affine index there
        (:meth:`_modulo_to_affine` returns ``None``). Such a wrap reads/writes the
        wrong element under C's truncated ``%`` at the boundary iteration, and peeling
        that boundary lets the band fold make both the remainder and the peeled
        iteration affine and floor-correct. A non-affine (e.g. data-dependent) modulo
        argument is ignored: no bounded peel could ever fold it."""
        import sympy
        from dace import subsets
        ranges = self._loop_own_ranges(loop)
        if not ranges:
            return False
        (ivar, _), = ranges.items()
        for st in loop.all_states():
            for e in st.edges():
                if e.data is None:
                    continue
                for r in (e.data.subset, e.data.other_subset):
                    if not isinstance(r, subsets.Range):
                        continue
                    for rng in r.ranges:
                        for expr in rng:
                            if not isinstance(expr, sympy.Basic):
                                continue
                            for mod in self._modulo_nodes(expr):
                                operands = self._modulo_operands(mod)
                                if operands is None:
                                    continue
                                arg = operands[0]
                                if ivar not in arg.free_symbols:
                                    continue
                                a = arg.coeff(ivar, 1)
                                if ivar in a.free_symbols or ivar in (arg - a * ivar).free_symbols:
                                    continue  # arg not affine in the loop variable
                                if self._modulo_to_affine(mod, ranges) is None:
                                    return True  # affine but genuinely wrapping
        return False

    def _best_modulo_peel_for(self, loop: LoopRegion, sdfg: SDFG):
        """The smallest ``(count, direction)`` peel that turns a genuinely-wrapping
        body modulo into an affine, floor-correct index for ``loop``, probed on an
        isolated copy; ``None`` if the loop has no such wrap or no bounded peel folds
        it. Unlike :meth:`_best_peel_for`, this runs *even when* ``LoopToMap`` already
        maps the loop: a wrap-around read maps as-is but then emits C's truncated
        ``%``, computing the wrong boundary value, so the peel is for correctness, not
        to unblock a map. A peel is accepted as soon as it removes every wrapping
        modulo (so no C ``%`` survives) while staying valid -- folding the wrap to its
        affine, floor-correct form is value-preserving whether or not it also maps."""
        if not self._has_wrapping_modulo(loop):
            return None
        mini, _ = self._isolate_loop(loop, sdfg)
        if mini is None:
            return None
        # Smallest peel first (count ascending, single-side before both) so the
        # correctness fix touches as few iterations as possible.
        for count in range(1, self.peel_limit + 1):
            for direction in ('front', 'back', 'both'):
                cand = copy.deepcopy(mini)
                cloops = _loops(cand)
                if not cloops:
                    return None
                if not self._peel_one_loop(cand, cloops[0], count, direction):
                    continue
                self._clean_peeled_remainder(cand)
                try:
                    cand.validate()
                    if any(self._has_wrapping_modulo(l) for l in _loops(cand)):
                        continue  # a wrapping modulo still survives -> C ``%`` remains
                except Exception:
                    continue
                return (count, direction)
        return None

    def _rewrite_modulo_over_range(self, sdfg: SDFG):
        """Rewrite every ``Mod(arg, m)`` memlet subset to an equivalent affine form
        when ``arg`` provably stays in one band over its enclosing loops' ranges (see
        :meth:`_modulo_to_affine`). A peel that removes the wrapping boundary
        iterations leaves a remainder whose wrap-around modulo is the identity, and
        the peeled iterations themselves carry the wrapped constant index -- both are
        folded here so the body is affine and C's truncated ``%`` is never emitted."""
        import sympy
        from dace import subsets
        for st in sdfg.all_states():
            ranges = self._enclosing_loop_ranges(st)
            repl: Dict[Any, Any] = {}
            ranges_seen = []
            for e in st.edges():
                if e.data is None:
                    continue
                for r in (e.data.subset, e.data.other_subset):
                    if not isinstance(r, subsets.Range):
                        continue
                    ranges_seen.append(r)
                    for rng in r.ranges:
                        for expr in rng:
                            if not isinstance(expr, sympy.Basic):
                                continue
                            for mod in self._modulo_nodes(expr):
                                if mod in repl:
                                    continue
                                affine = self._modulo_to_affine(mod, ranges)
                                if affine is not None:
                                    repl[mod] = affine
            if repl:
                for r in ranges_seen:
                    r.replace(repl)

    def _mappable_loop_count(self, candidate: SDFG) -> int:
        """Cheap proxy for "does the peel unblock parallelization?": run scalar
        fission -> symbol propagation -> constant propagation (no reduction passes),
        then COUNT the loops ``LoopToMap`` *could* parallelize -- via
        ``can_be_applied_to``, WITHOUT applying it. Peeling is a preparation pass;
        the actual ``LoopToMap`` is the pipeline's ``parallelize`` stage, so the
        search only probes ``can_be_applied``. Mutates ``candidate`` (the prep)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.constant_propagation import ConstantPropagation
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        PrivatizeScalars().apply_pass(candidate, {})
        SymbolPropagation().apply_pass(candidate, {})
        ConstantPropagation().apply_pass(candidate, {})
        count = 0
        for loop in _loops(candidate):
            try:
                if LoopToMap.can_be_applied_to(candidate, loop=loop):
                    count += 1
            except Exception:
                pass
        return count

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Unblock stuck loops, choosing per loop (on an isolated copy of just that
        nest, so the search is cheap and revertible) between an index-set split at an
        interior ``if i == x`` guard and a best-effort boundary peel, applying the
        winner to the real SDFG. Returns the number of loops rewritten or None.

        Splitting is tried first: an interior equality guard (``if i == x`` with ``x``
        away from the boundary) is carved out as [start, x-1] + {x} + [x+1, end], which
        no bounded boundary peel could reach. Next a wrapping-modulo peel fixes a
        boundary that reads/writes the wrong element under C's truncated ``%`` (a
        correctness fix that runs even when the loop already maps). Whatever stays
        stuck then goes through the front/back/both peel search."""
        if self.peel_limit <= 0 or not _loops(sdfg):
            return None
        applied = 0
        # 1. Index-set splitting for interior equality guards.
        for loop in list(_loops(sdfg)):
            x = self._best_split_for(loop, sdfg)
            if x is not None and self._split_loop_at(sdfg, loop, x):
                self._clean_peeled_remainder(sdfg)
                applied += 1
        # 2. Peel a genuinely-wrapping body modulo to its floor-correct affine form,
        #    even for loops LoopToMap already maps (the wrap-around access otherwise
        #    emits C's truncated ``%`` and computes the wrong boundary value).
        for loop in list(_loops(sdfg)):
            best = self._best_modulo_peel_for(loop, sdfg)
            if best is not None and self._peel_one_loop(sdfg, loop, *best):
                self._clean_peeled_remainder(sdfg)
                applied += 1
        # 3. Best-effort boundary peel for the loops splitting did not resolve.
        for loop in list(_loops(sdfg)):
            best = self._best_peel_for(loop, sdfg)
            if best is not None and self._peel_one_loop(sdfg, loop, *best):
                self._clean_peeled_remainder(sdfg)
                applied += 1
        return applied or None
