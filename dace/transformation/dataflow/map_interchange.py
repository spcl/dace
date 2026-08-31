# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the map interchange transformation. """

from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace import properties, subsets, symbolic
from dace.properties import make_properties
from dace.symbolic import symlist
from dace.transformation import transformation
from dace.sdfg.propagation import propagate_memlet
import sympy


@make_properties
class MapInterchange(transformation.SingleStateTransformation):
    """ Implements the map-interchange transformation.

        Map-interchange takes two nested maps and interchanges their position.
    """

    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    transform_bounds = properties.Property(
        dtype=bool,
        default=False,
        desc='Also interchange a NON-RECTANGULAR nest, by transforming the bounds. A plain swap is '
        'only legal when the inner range does not mention the outer parameter, which refuses every '
        'triangular / trapezoidal nest -- and those are exactly the nests whose traversal order is '
        'worth fixing (TSVC s1232 keeps a stride-LEN_2D innermost loop because of it). Off by '
        'default: the rewrite derives new bounds and, when the slope is symbolic, records a '
        'runtime assumption, neither of which a caller asking for a plain swap expects.')

    def trapezoid(self):
        """``(param, lo, hi, slope, base)`` when the nest is trapezoidal and this rewrite handles it.

        The shape is ``for p in [plo, phi]: for q in [base + slope*p, qhi]`` with unit steps -- the
        inner range's START leans on the outer parameter and nothing else does. ``None`` when the
        nest is rectangular (a plain swap already applies), when the dependence sits somewhere this
        rewrite cannot inverte (the inner END, the step), or when the slope is zero.
        """
        outer, inner = self.outer_map_entry.map, self.inner_map_entry.map
        if len(outer.params) != 1 or len(inner.params) != 1:
            return None
        p = symbolic.pystr_to_symbolic(outer.params[0])
        plo, phi, pstep = outer.range[0]
        qlo, qhi, qstep = inner.range[0]
        if symbolic.simplify(pstep - 1) != 0 or symbolic.simplify(qstep - 1) != 0:
            return None
        # The outer range must not lean on the inner parameter, and the inner END must not lean on
        # the outer one: this rewrite inverts a lower bound, nothing else.
        q = symbolic.pystr_to_symbolic(inner.params[0])
        if any(q in getattr(e, 'free_symbols', set()) for e in (plo, phi)):
            return None
        if p in getattr(qhi, 'free_symbols', set()):
            return None
        if not isinstance(qlo, sympy.Basic) or p not in qlo.free_symbols:
            return None
        slope = qlo.coeff(p, 1)
        base = symbolic.simplify(qlo - slope * p)
        if p in getattr(slope, 'free_symbols', set()) or p in getattr(base, 'free_symbols', set()):
            return None  # not affine in the outer parameter
        if slope == 0 or slope.is_negative:
            # A zero slope is the rectangular case; a negative one inverts to a different bound
            # shape (a MAX over an int_ceil) that this rewrite does not emit.
            return None
        return outer.params[0], plo, phi, slope, base

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # TODO: Assuming that the subsets on the edges between the two map
        # entries/exits are the union of separate inner subsets, is it possible
        # that inverting these edges breaks the continuity of union? What about
        # the opposite?

        # Check the edges between the entries of the two maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry

        # Check that inner map range is independent of outer range -- unless the caller asked for
        # the bound-transforming form and the nest is one this rewrite can invert.
        map_deps = set()
        for s in inner_map_entry.map.range:
            map_deps |= set(map(str, symlist(s)))
        if any(dep in outer_map_entry.map.params for dep in map_deps):
            if not self.transform_bounds or self.trapezoid() is None:
                return False

        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for e in graph.out_edges(outer_map_entry):
            if e.dst != inner_map_entry:
                return False
        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for e in graph.in_edges(inner_map_entry):
            if e.src != outer_map_entry:
                return False
            # Check that dynamic input range memlets are independent of
            # first map range
            if e.dst_conn and not e.dst_conn.startswith('IN_'):
                memlet_deps = set()
                for s in e.data.subset:
                    memlet_deps |= set(map(str, symlist(s)))
                if any(dep in outer_map_entry.map.params for dep in memlet_deps):
                    return False

        # Check the edges between the exits of the two maps.
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for e in graph.out_edges(inner_map_exit):
            if e.dst != outer_map_exit:
                return False
        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for e in graph.in_edges(outer_map_exit):
            if e.src != inner_map_exit:
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        # Read the trapezoid BEFORE the swap: it is a property of the nest as matched, and the swap
        # is what invalidates it.
        trapezoid = self.trapezoid() if self.transform_bounds else None
        # Extract the parameters and ranges of the inner/outer maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Switch connectors
        outer_map_entry.in_connectors, inner_map_entry.in_connectors = \
            inner_map_entry.in_connectors, outer_map_entry.in_connectors
        outer_map_entry.out_connectors, inner_map_entry.out_connectors = \
            inner_map_entry.out_connectors, outer_map_entry.out_connectors
        outer_map_exit.in_connectors, inner_map_exit.in_connectors = \
            inner_map_exit.in_connectors, outer_map_exit.in_connectors
        outer_map_exit.out_connectors, inner_map_exit.out_connectors = \
            inner_map_exit.out_connectors, outer_map_exit.out_connectors

        # Get edges between the map entries and exits.
        entry_edges = graph.edges_between(outer_map_entry, inner_map_entry)
        exit_edges = graph.edges_between(inner_map_exit, outer_map_exit)
        for e in entry_edges + exit_edges:
            graph.remove_edge(e)

        # Change source and destination of edges.
        sdutil.change_edge_dest(graph, outer_map_entry, inner_map_entry)
        sdutil.change_edge_src(graph, inner_map_entry, outer_map_entry)
        sdutil.change_edge_dest(graph, inner_map_exit, outer_map_exit)
        sdutil.change_edge_src(graph, outer_map_exit, inner_map_exit)

        # Add edges between the map entries and exits.
        new_entry_edges = []
        new_exit_edges = []
        for e in entry_edges:
            new_entry_edges.append(graph.add_edge(e.dst, e.src_conn, e.src, e.dst_conn, e.data))
        for e in exit_edges:
            new_exit_edges.append(graph.add_edge(e.dst, e.src_conn, e.src, e.dst_conn, e.data))

        # Repropagate memlets in modified region
        for e in new_entry_edges:
            path = graph.memlet_path(e)
            index = next(i for i, edge in enumerate(path) if e is edge)
            if index < len(path) - 1:
                edge_to_propagate = path[index + 1]
            else:
                edge_to_propagate = e
            e.data.subset = propagate_memlet(graph, edge_to_propagate.data, outer_map_entry, True).subset
        for e in new_exit_edges:
            path = graph.memlet_path(e)
            index = next(i for i, edge in enumerate(path) if e is edge)
            if index > 0:
                edge_to_propagate = path[index - 1]
            else:
                edge_to_propagate = e
            e.data.subset = propagate_memlet(graph, edge_to_propagate.data, outer_map_exit, True).subset

        if trapezoid is not None:
            self.rewrite_trapezoid_bounds(sdfg, trapezoid)

    def rewrite_trapezoid_bounds(self, sdfg: SDFG, trapezoid) -> None:
        """Re-derive both ranges after swapping a nest whose inner start leaned on the outer param.

        ``for p in [plo, phi]: for q in [base + slope*p, qhi]`` covers
        ``{(p, q) : plo <= p <= phi, base + slope*p <= q <= qhi}``. Swapped, the same set is
        ``{(q, p) : base + slope*plo <= q <= qhi, plo <= p <= min(phi, (q - base) // slope)}`` --
        for a fixed ``q`` the constraint ``slope*p <= q - base`` is exactly ``p <= floor((q-base)/
        slope)``, and no ``q`` below ``base + slope*plo`` admits any ``p``. The iteration SET is
        therefore identical; only its traversal order changes, which is what a Map leaves free.

        A symbolic ``slope`` must be ``>= 1`` for the inverted bound to mean anything (it is a
        divisor). That is recorded as a tracked assumption rather than assumed silently, so
        :func:`~dace.transformation.passes.canonicalize.assume_symbols_nonnegative.insert_assumption_guards`
        emits the matching runtime trap -- the same discipline the modular-wrap split uses for
        ``K < N``.
        """
        from dace.transformation.passes.canonicalize.tracked_assumptions import record_assumption

        param, plo, phi, slope, base = trapezoid
        # After the swap the node that WAS inner sits outermost, still carrying its own map.
        new_outer, new_inner = self.inner_map_entry.map, self.outer_map_entry.map
        q = symbolic.pystr_to_symbolic(new_outer.params[0])
        _, qhi, _ = new_outer.range[0]
        new_outer.range = subsets.Range([(symbolic.simplify(base + slope * plo), qhi, 1)])
        reach = symbolic.int_floor(q - base, slope)
        new_inner.range = subsets.Range([(plo, sympy.Min(phi, reach), 1)])
        if not slope.is_number:
            record_assumption(sdfg, sympy.Ge(slope, 1))

    @staticmethod
    def annotates_memlets():
        return True
