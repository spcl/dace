# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse two consecutive loops with identical bodies over adjacent index ranges.

A hand-tiled loop nest arrives split into a *main-body* loop over the largest
multiple-of-``K`` prefix and a step-1 *remainder* loop over the trailing
``< K`` elements::

    for i in range(0, (N // K) * K):   # main body (after re-roll: unit stride)
        acc += a[i]
    for i in range((N // K) * K, N):   # remainder
        acc += a[i]

Both loops run the *same* body over *adjacent* index ranges, so together they
sweep ``range(0, N)``. Left split, a reduction like the one above lifts to two
separate ``Reduce`` library nodes writing the same accumulator -- a shape whose
seed does not chain across the two nodes (the second re-seeds the accumulator
and drops the first's partial sum). Fusing the two loops back into one
``for i in range(0, N)`` removes the split so a single ``Reduce`` is lifted and
the result is exact.

The rewrite is unconditionally value-preserving: concatenating two disjoint,
adjacent iteration ranges ``[A, B)`` and ``[B, C)`` under the *same* stride and
the *same* body executes the body for exactly the same index sequence in the
same order as a single ``[A, C)`` loop, so every loop-carried value is
identical. The match is deliberately conservative -- both loops must be
single-state, unit-stride, directly consecutive (one plain interstate edge
between them, nothing else), and structurally identical up to their iteration
variable -- so it fires only on the re-rolled tile/remainder shape and its kin,
never on unrelated adjacent loops.
"""
import re
from typing import List, Optional, Tuple

import sympy

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes.analysis import loop_analysis

#: Placeholder the iteration variable is normalised to when comparing two loop
#: bodies, so ``a[_loop_it_0]`` and ``a[_loop_it_1]`` compare equal.
_ITER_PLACEHOLDER = '__lv__'

#: Placeholder every body-local scratch transient name is normalised to, so two
#: bodies differing only in a frontend-generated intermediate name (e.g.
#: ``s0_plus_a_slice`` vs ``s0_plus_a_slice_0``) compare equal. The carried
#: accumulator and the read/written arrays -- which are referenced OUTSIDE the
#: body and so are not body-local -- keep their real names and must match.
_SCRATCH_PLACEHOLDER = '__scratch__'


def _int_floor_to_sympy(expr):
    """Rewrite DaCe ``int_floor(a, b)`` sub-terms as SymPy ``floor(a / b)``.

    ``int_floor`` is opaque to :func:`sympy.simplify`, so a genuine identity
    such as ``int_floor(N - 11, 11) + 1 == int_floor(N, 11)`` is left unproven.
    SymPy's ``floor`` knows ``floor(x + k) == floor(x) + k`` for integer ``k``,
    which is exactly what an adjacency check between a tile bound and a
    remainder start needs.
    """
    return expr.replace(lambda x: hasattr(x, 'func') and x.func.__name__ == 'int_floor' and len(x.args) == 2,
                        lambda x: sympy.floor(x.args[0] / x.args[1]))


def _symbolically_equal(a, b) -> bool:
    """Whether two symbolic expressions are provably equal, floor-aware.

    :param a: First expression.
    :param b: Second expression.
    :returns: ``True`` iff ``a - b`` simplifies to zero (after rewriting
              ``int_floor`` to SymPy ``floor``), else ``False``.
    """
    try:
        diff = symbolic.simplify(symbolic.pystr_to_symbolic(a) - symbolic.pystr_to_symbolic(b))
        if diff == 0:
            return True
        return symbolic.simplify(_int_floor_to_sympy(diff)) == 0
    except Exception:
        return False


def _normalize(text: str, loop_var: str) -> str:
    """Replace whole-word occurrences of ``loop_var`` in ``text`` with the
    canonical placeholder so two bodies differing only in iterator name match."""
    return re.sub(r'\b%s\b' % re.escape(loop_var), _ITER_PLACEHOLDER, text)


def _canon_data(name: str, local_scratch: set) -> str:
    """Map a body-local scratch transient to the canonical placeholder; leave
    carried / external names (accumulator, arrays) untouched."""
    return _SCRATCH_PLACEHOLDER if name in local_scratch else name


def _node_key(node, loop_var: str, local_scratch: set) -> Tuple:
    """A structural key for a body node, iterator- and scratch-name-independent."""
    if isinstance(node, nodes.AccessNode):
        return ('access', _canon_data(node.data, local_scratch))
    if isinstance(node, nodes.Tasklet):
        return ('tasklet', _normalize(node.code.as_string.strip(), loop_var))
    return ('other', type(node).__name__)


@explicit_cf_compatible
class FuseConsecutiveLoops(ppl.Pass):
    """Fuse two directly-consecutive, identical-bodied, unit-stride loops whose
    iteration ranges are adjacent (``[A, B)`` followed by ``[B, C)``) into a
    single loop over ``[A, C)``."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Fuse every matching consecutive-loop pair in ``sdfg`` and its nested
        SDFGs, repeating until no pair matches (a chain of tiles collapses one
        adjacency per sweep).

        :param sdfg: SDFG to mutate in place.
        :returns: The number of fusions performed, or ``None`` if none.
        """
        fused = 0
        for sd in sdfg.all_sdfgs_recursive():
            changed = True
            while changed:
                changed = False
                for cfg in list(sd.all_control_flow_regions(recursive=True)):
                    if self._fuse_one(cfg):
                        fused += 1
                        changed = True
                        break
        return fused or None

    def _fuse_one(self, cfg: ControlFlowRegion) -> bool:
        """Find and fuse one consecutive-loop pair inside ``cfg``.

        :param cfg: The control-flow region to search (one level; loops nested
                    deeper are reached via ``all_control_flow_regions``).
        :returns: ``True`` if a pair was fused.
        """
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out_edges = cfg.out_edges(first)
            if len(out_edges) != 1:
                continue
            link = out_edges[0]
            second = link.dst
            if not isinstance(second, LoopRegion) or second is first:
                continue
            if len(cfg.in_edges(second)) != 1:
                continue
            # The connecting edge must be pure sequencing: no assignments and a
            # trivial condition, so nothing runs (or is decided) between them.
            if link.data.assignments:
                continue
            if link.data.condition is not None and link.data.condition.as_string not in ('1', 'True'):
                continue
            if self._adjacent_identical(first, second):
                self._merge(cfg, first, second, link)
                return True
        return False

    def _adjacent_identical(self, first: LoopRegion, second: LoopRegion) -> bool:
        """Whether ``first`` then ``second`` are unit-stride, structurally
        identical, and cover adjacent index ranges ``[A, B)`` and ``[B, C)``."""
        for loop in (first, second):
            if not loop.loop_variable:
                return False
            stride = loop_analysis.get_loop_stride(loop)
            if stride is None or symbolic.simplify(stride) != 1:
                return False
            start = loop_analysis.get_init_assignment(loop)
            end = loop_analysis.get_loop_end(loop)
            if start is None or end is None:
                return False
        # Adjacency: first's exclusive end (last value + 1) == second's start.
        first_end_excl = symbolic.simplify(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(first)) + 1)
        second_start = loop_analysis.get_init_assignment(second)
        if not _symbolically_equal(first_end_excl, second_start):
            return False
        return self._bodies_match(first, second)

    def _single_body_state(self, loop: LoopRegion) -> Optional[SDFGState]:
        """The loop's one non-empty body state, or ``None`` if the body is not a
        single compute state (empty connective states are tolerated)."""
        blocks = list(loop.nodes())
        if not all(isinstance(b, SDFGState) for b in blocks):
            return None
        non_empty = [b for b in blocks if b.nodes()]
        if len(non_empty) != 1:
            return None
        return non_empty[0]

    def _bodies_match(self, first: LoopRegion, second: LoopRegion) -> bool:
        """Whether the two loops' single body states are identical up to their
        iteration variable and body-local scratch names (same nodes, same edges,
        same memlets)."""
        s1 = self._single_body_state(first)
        s2 = self._single_body_state(second)
        if s1 is None or s2 is None:
            return False
        sig1 = self._state_signature(s1, first.loop_variable, self._local_scratch(first, s1))
        sig2 = self._state_signature(s2, second.loop_variable, self._local_scratch(second, s2))
        return sig1 == sig2

    def _local_scratch(self, loop: LoopRegion, body_state: SDFGState) -> set:
        """Transient data names used ONLY inside ``body_state`` -- i.e. not
        referenced by any other block of the owning SDFG (not carried across
        iterations, not read/written outside the loop). These are frontend
        scratch intermediates whose names are irrelevant to what the body
        computes and so are normalised away when comparing two bodies."""
        root = loop
        while root.parent_graph is not None:
            root = root.parent_graph
        external = set()
        for st in root.all_states():
            if st is body_state:
                continue
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode):
                    external.add(n.data)
        local = set()
        for n in body_state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data not in external:
                desc = root.arrays.get(n.data)
                if desc is not None and desc.transient:
                    local.add(n.data)
        return local

    def _state_signature(self, state: SDFGState, loop_var: str, local_scratch: set) -> Tuple:
        """An iterator- and scratch-name-independent structural signature of a
        body state: its sorted node keys and its sorted edge descriptors
        (endpoints, connectors, memlet data/subset/wcr)."""
        node_sig = sorted(_node_key(n, loop_var, local_scratch) for n in state.nodes())
        edge_sig = []
        for e in state.edges():
            subset = _normalize(str(e.data.subset), loop_var) if (e.data and e.data.subset is not None) else ''
            data_name = _canon_data(e.data.data, local_scratch) if (e.data is not None and e.data.data) else ''
            wcr = str(e.data.wcr) if e.data is not None else ''
            edge_sig.append((_node_key(e.src, loop_var, local_scratch), e.src_conn,
                             _node_key(e.dst, loop_var, local_scratch), e.dst_conn, data_name, subset, wcr))
        return (tuple(node_sig), tuple(sorted(edge_sig)))

    def _merge(self, cfg: ControlFlowRegion, first: LoopRegion, second: LoopRegion, link) -> None:
        """Extend ``first`` over the union range and splice ``second`` out.

        ``first`` keeps its body and iterator; only its exclusive upper bound is
        widened to ``second``'s. ``second``'s successor edges are re-homed onto
        ``first`` and ``second`` is removed.

        :param cfg: The region owning both loops.
        :param first: The surviving loop (its body is kept).
        :param second: The loop to remove (its body is a duplicate of ``first``).
        :param link: The ``first -> second`` sequencing edge (removed with ``second``).
        """
        var = first.loop_variable
        new_end_excl = symbolic.simplify(symbolic.pystr_to_symbolic(loop_analysis.get_loop_end(second)) + 1)
        first.loop_condition = dace.properties.CodeBlock(f"{var} < ({symbolic.symstr(new_end_excl)})")

        out_edges = list(cfg.out_edges(second))
        cfg.remove_node(second)  # also drops the ``first -> second`` link edge
        for e in out_edges:
            cfg.add_edge(first, e.dst, e.data)


__all__ = ['FuseConsecutiveLoops']
