""" Functionality relating to Memlet propagation (deducing external memlets
    from internal memory accesses and scope ranges). """

import copy
import itertools
import functools
import networkx as nx
import sympy
import unittest
import math

from dace import data, subsets, symbolic, types
from dace.memlet import Memlet
from dace.graph import nodes, nxutil
from dace.graph.graph import OrderedMultiDiGraph
from dace.transformation import pattern_matching


class MemletPattern(object):
    """ A pattern match on a memlet subset that can be used for propagation. 
    """
    s_patterns = []
    s_dependencies = {}

    @staticmethod
    def patterns():
        return [p() for p in MemletPattern.s_patterns]

    @staticmethod
    def register_pattern(clazz, depends=None):
        if not issubclass(clazz, MemletPattern):
            raise TypeError
        MemletPattern.s_patterns.append(clazz)

    @staticmethod
    def unregister_pattern(clazz):
        if not issubclass(clazz, MemletPattern):
            raise TypeError
        MemletPattern.s_patterns.remove(clazz)

    ####################################################

    def match(self, expressions, variable_context, node_range, orig_edges):
        raise NotImplementedError

    def propagate(self, array, expressions, node_range):
        raise NotImplementedError


class SeparableMemletPattern(object):
    """ Memlet pattern that can be applied to each of the dimensions 
        separately. """

    s_smpatterns = []

    @staticmethod
    def register_pattern(cls):
        if not issubclass(cls, SeparableMemletPattern): raise TypeError
        if cls not in SeparableMemletPattern.s_smpatterns:
            SeparableMemletPattern.s_smpatterns.append(cls)

    @staticmethod
    def unregister_pattern(cls):
        SeparableMemletPattern.s_smpatterns.remove(cls)

    def match(self, dim_exprs, variable_context, node_range, orig_edges,
              dim_index, total_dims):
        raise NotImplementedError

    def propagate(self, array, dim_exprs, node_range):
        raise NotImplementedError


class SeparableMemlet(MemletPattern):
    """ Meta-memlet pattern that applies all separable memlet patterns. """

    def match(self, expressions, variable_context, node_range, orig_edges):
        # Assuming correct dimensionality in each of the expressions
        data_dims = len(expressions[0])
        self.patterns_per_dim = [None] * data_dims

        overapprox_range = subsets.Range([(rb.approx if isinstance(
            rb, symbolic.SymExpr) else rb, re.approx if isinstance(
                re, symbolic.SymExpr) else re, rs.approx if isinstance(
                    rs, symbolic.SymExpr) else rs)
                                          for rb, re, rs in node_range])

        for dim in range(data_dims):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[dim], symbolic.SymExpr):
                    dexprs.append(expr[dim].approx)
                elif isinstance(expr[dim], tuple):
                    dexprs.append(
                        (expr[dim][0].approx
                         if isinstance(expr[dim][0], symbolic.SymExpr) else
                         expr[dim][0], expr[dim][1].approx
                         if isinstance(expr[dim][1], symbolic.SymExpr) else
                         expr[dim][1], expr[dim][2].approx
                         if isinstance(expr[dim][2],
                                       symbolic.SymExpr) else expr[dim][2]))
                else:
                    dexprs.append(expr[dim])

            for pattern_class in SeparableMemletPattern.s_smpatterns:
                smpattern = pattern_class()
                if smpattern.match(dexprs, variable_context, overapprox_range,
                                   orig_edges, dim, data_dims):
                    self.patterns_per_dim[dim] = smpattern
                    break

        return None not in self.patterns_per_dim

    def propagate(self, array, expressions, node_range):
        result = [(None, None, None)] * len(self.patterns_per_dim)

        overapprox_range = subsets.Range([(rb.approx if isinstance(
            rb, symbolic.SymExpr) else rb, re.approx if isinstance(
                re, symbolic.SymExpr) else re, rs.approx if isinstance(
                    rs, symbolic.SymExpr) else rs)
                                          for rb, re, rs in node_range])

        for i, smpattern in enumerate(self.patterns_per_dim):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[i], symbolic.SymExpr):
                    dexprs.append(expr[i].approx)
                elif isinstance(expr[i], tuple):
                    dexprs.append((expr[i][0].approx if isinstance(
                        expr[i][0],
                        symbolic.SymExpr) else expr[i][0], expr[i][1].approx
                                   if isinstance(expr[i][1], symbolic.SymExpr)
                                   else expr[i][1], expr[i][2].approx
                                   if isinstance(expr[i][2], symbolic.SymExpr)
                                   else expr[i][2], expr.tile_sizes[i]))
                else:
                    dexprs.append(expr[i])

            result[i] = smpattern.propagate(array, dexprs, overapprox_range)

        # TODO(later): Not necessarily Range (general integer sets)
        return subsets.Range(result)


MemletPattern.register_pattern(SeparableMemlet)


class AffineSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches affine expressions, i.e.,
        of the form `a * {index} + b`.
    """

    def match(self, dim_exprs, variable_context, node_range, orig_edges,
              dim_index, total_dims):

        params = variable_context[-1]  # Why only last element?
        # Create wildcards for multiplication and addition
        a = sympy.Wild('a', exclude=params)
        b = sympy.Wild('b', exclude=params)

        self.param = None
        self.paramind = None
        self.mult = None
        self.add_min = None
        self.add_max = None
        self.constant_min = None
        self.constant_max = None

        # Obtain vector length
        self.veclen = None
        if dim_index == total_dims - 1:
            for e in orig_edges:
                self.veclen = e.veclen
        if self.veclen is None:
            self.veclen = 1
        ######################

        # Special case: Get the total internal access range
        # If this range matches (0, rs), we say that the propagated skip is 1
        self.internal_range = set()

        for dexpr in dim_exprs:
            subexprs = None
            step = None
            if isinstance(dexpr, sympy.Basic):  # Affine index
                subexprs = [dexpr]

            elif isinstance(dexpr, tuple) and len(dexpr) == 3:  # Affine range
                subexprs = [dexpr[0], dexpr[1]]
                step = dexpr[2]

            if subexprs is None:  # Something else
                return False

            for i, subexpr in enumerate(subexprs):
                # Try to match an affine expression with a parameter
                param = None
                pind = -1
                for indp, p in enumerate(params):
                    matches = subexpr.match(a * p + b)
                    if param is None and matches is None:
                        continue
                    elif param is not None and matches is not None:
                        return False  # Only one parameter may match
                    elif matches is not None:
                        multiplier = matches[a]
                        addition = matches[b]
                        param = p
                        pind = indp

                if param is None:
                    return False  # A parameter must match
                if self.param is not None and param != self.param:
                    return False  # There can only be one parameter
                if self.mult is not None and multiplier != self.mult:
                    return False  # Multiplier must be the same

                self.param = param
                self.paramind = pind
                self.multiplier = multiplier

                # If this is one expression
                if len(subexprs) == 1:
                    self.internal_range.add(addition)
                elif i == 0:  # Range begin
                    brb = addition
                elif i == 1:  # Range end
                    bre = addition

            if len(subexprs) > 1:
                self.internal_range.add((brb, bre))

            if step is not None:
                if self.param in step.free_symbols:
                    return False  # Step must be independent of parameter

            node_rb, node_re, node_rs = node_range[self.paramind]
            if node_rs != 1:
                # Map ranges where the last index is not known
                # exactly are not supported by this pattern.
                return False

        if self.param is None:  # and self.constant_min is None:
            return False

        return True

    def propagate(self, array, dim_exprs, node_range):
        # Compute last index in map according to range definition
        node_rb, node_re, node_rs = node_range[self.paramind]  # node_rs = 1
        node_rlen = node_re - node_rb + 1

        if isinstance(dim_exprs, list):
            dim_exprs = dim_exprs[0]

        if isinstance(dim_exprs, tuple):

            if len(dim_exprs) == 3:
                rb, re, rs = dim_exprs
                rt = '1'
            elif len(dim_exprs) == 4:
                rb, re, rs, rt = dim_exprs
            else:
                raise NotImplementedError

            rb = symbolic.pystr_to_symbolic(rb).expand()
            re = symbolic.pystr_to_symbolic(re).expand()
            rs = symbolic.pystr_to_symbolic(rs).expand()
            rt = symbolic.pystr_to_symbolic(rt).expand()
        else:
            rb, re = (dim_exprs.expand(), dim_exprs.expand())
            rs = 1
            rt = 1

        result_begin = rb.subs(self.param, node_rb).expand()
        result_end = re.subs(self.param, node_re).expand()

        # Experimental
        # This should be using sympy.floor
        memlet_start_pts = ((re - rt + 1 - rb) / rs) + 1
        memlet_rlen = memlet_start_pts.expand() * rt
        interval_len = (result_end - result_begin + 1) * self.veclen
        num_elements = node_rlen * memlet_rlen

        if (interval_len == num_elements
                or interval_len.expand() == num_elements):
            # Continuous access
            result_skip = 1
            result_tile = 1
        else:
            if rt == 1:
                result_skip = (result_end - result_begin - re + rb) / (
                    node_re - node_rb)
                try:
                    if result_skip < 1:
                        result_skip = 1
                except:
                    pass
                result_tile = result_end - result_begin + 1 - (
                    node_rlen - 1) * result_skip
            else:
                candidate_skip = rs
                candidate_tile = rt * node_rlen
                candidate_lstart_pt = result_end - result_begin + 1 - candidate_tile
                if (candidate_lstart_pt / (num_elements / candidate_tile - 1)
                    ).simplify() == candidate_skip:
                    result_skip = rs
                    result_tile = rt * node_rlen
                else:
                    result_skip = rs / node_rlen
                    result_tile = rt

            if result_skip == result_tile or result_skip == 1:
                result_skip = 1
                result_tile = 1

        result_begin = sympy.simplify(result_begin)
        result_end = sympy.simplify(result_end)
        result_skip = sympy.simplify(result_skip)
        result_tile = sympy.simplify(result_tile)

        return (result_begin, result_end, result_skip, result_tile)


SeparableMemletPattern.register_pattern(AffineSMemlet)


class ModuloSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches modulo expressions, i.e.,
        of the form `f(x) % N`.

        Acts as a meta-pattern: Finds the underlying pattern for `f(x)`.
    """

    def match(self, dim_exprs, variable_context, node_range, orig_edges,
              dim_index, total_dims):
        # Pattern does not support unions of expressions
        if len(dim_exprs) > 1: return False
        dexpr = dim_exprs[0]
        # Pattern does not support ranges
        if not isinstance(dexpr, sympy.Basic): return False

        # Create wildcards
        val = sympy.Wild('val')
        mod = sympy.Wild('mod', exclude=variable_context[-1])

        # Try to match an affine expression
        matches = dexpr.match(val % mod)
        if matches is None or len(matches) != 2:
            return False

        self.subexpr = matches[val]
        self.modulo = matches[mod]

        self.subpattern = None
        for pattern_class in SeparableMemletPattern.s_smpatterns:
            smpattern = pattern_class()
            if smpattern.match([self.subexpr], variable_context, node_range,
                               orig_edges, dim_index, total_dims):
                self.subpattern = smpattern

        return self.subpattern is not None

    def propagate(self, array, dim_exprs, node_range):
        se_range = self.subpattern.propagate(array, [self.subexpr], node_range)

        # Apply modulo on start and end ranges
        try:
            if se_range[0] < 0:
                se_range = (0, self.modulo, se_range[2])
        except TypeError:  # cannot determine truth value of Relational
            print('WARNING: Cannot evaluate relational %s, assuming true.' %
                  (se_range[0] < 0))
        try:
            if se_range[1] > self.modulo:
                se_range = (0, self.modulo, se_range[2])
        except TypeError:  # cannot determine truth value of Relational
            print('WARNING: Cannot evaluate relational %s, assuming true.' %
                  (se_range[1] > self.modulo))

        return se_range


SeparableMemletPattern.register_pattern(ModuloSMemlet)


class ConstantSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches constant (i.e., unrelated to 
        current scope) expressions.
    """

    def match(self, dim_exprs, variable_context, node_range, orig_edges,
              dim_index, total_dims):
        # Pattern does not support unions of expressions. TODO: Support
        if len(dim_exprs) > 1: return False
        dexpr = dim_exprs[0]

        # Create a wildcard that excludes current map's parameters
        cst = sympy.Wild('cst', exclude=variable_context[-1])

        # Range case
        if isinstance(dexpr, tuple) and len(dexpr) == 3:
            # Try to match a constant expression for the range
            for rngelem in dexpr:
                if types.isconstant(rngelem):
                    continue

                matches = rngelem.match(cst)
                if matches is None or len(matches) != 1:
                    return False
                if not matches[cst].is_constant():
                    return False

        else:  # Single element case
            # Try to match a constant expression
            if not types.isconstant(dexpr):
                matches = dexpr.match(cst)
                if matches is None or len(matches) != 1:
                    return False
                if not matches[cst].is_constant():
                    return False

        return True

    def propagate(self, array, dim_exprs, node_range):
        if isinstance(dim_exprs[0], tuple):
            return dim_exprs[0]  # Already in range format
        # Convert index to range format
        return (dim_exprs[0], dim_exprs[0], 1)


SeparableMemletPattern.register_pattern(ConstantSMemlet)


class GenericSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that detects any expression, and propagates 
        interval bounds. Used as a last resort. """

    def match(self, dim_exprs, variable_context, node_range, orig_edges,
              dim_index, total_dims):

        self.params = variable_context[-1]

        # Always matches
        return True

    def propagate(self, array, dim_exprs, node_range):

        result_begin = None
        result_end = None

        # Iterate over the node dimensions
        for idx, node_r in enumerate(node_range):

            # Get dimension range
            if len(node_r) == 3:
                node_rb, node_re, node_rs = node_r
            elif len(node_r) == 4:
                node_rb, node_re, node_rs, _ = node_r
            else:
                raise NotImplementedError

            # Get true range end
            lastindex = node_re
            if node_rs != 1:
                lastindex = symbolic.pystr_to_symbolic(
                    '%s + int_floor(%s - %s, %s) * %s' %
                    (symbolic.symstr(node_rb), symbolic.symstr(node_re),
                     symbolic.symstr(node_rb), symbolic.symstr(node_rs),
                     symbolic.symstr(node_rs)))

            if isinstance(dim_exprs, list):
                dim_exprs = dim_exprs[0]

            if isinstance(dim_exprs, tuple):

                if len(dim_exprs) == 3:
                    rb, re, rs = dim_exprs
                elif len(dim_exprs) == 4:
                    rb, re, rs, _ = dim_exprs
                else:
                    raise NotImplementedError

                rb = symbolic.pystr_to_symbolic(rb)
                re = symbolic.pystr_to_symbolic(re)
                rs = symbolic.pystr_to_symbolic(rs)

            else:
                rb, re = (dim_exprs, dim_exprs)

            if result_begin is None:
                result_begin = rb.subs(self.params[idx], node_rb)
            else:
                result_begin = result_begin.subs(self.params[idx], node_rb)
            if result_end is None:
                result_end = re.subs(self.params[idx], lastindex)
            else:
                result_end = result_end.subs(self.params[idx], lastindex)

        result_skip = 1
        result_tile = 1

        return (result_begin, result_end, result_skip, result_tile)


SeparableMemletPattern.register_pattern(GenericSMemlet)


def _subexpr(dexpr, repldict):
    if isinstance(dexpr, tuple):
        return tuple(_subexpr(d, repldict) for d in dexpr)
    elif isinstance(dexpr, symbolic.SymExpr):
        return dexpr.expr.subs(repldict)
    else:
        return dexpr.subs(repldict)


class ConstantRangeMemlet(MemletPattern):
    """ Memlet pattern that matches arbitrary expressions with constant range.
    """

    def match(self, expressions, variable_context, node_range, orig_edges):
        constant_range = True
        for dim in node_range:
            for rngelem in dim:  # For (begin, end, skip)
                if not types.isconstant(rngelem) and not isinstance(
                        rngelem, sympy.Number):
                    constant_range = False
                    break
        if not constant_range:
            return False

        self.params = variable_context[-1]

        return True

    # TODO: An integer set library should shine here (unify indices)
    def propagate(self, array, expressions, node_range):
        rng = [(None, None, 1)] * len(array.shape)
        node_range_gen = (range(rb, re, rs) for rb, re, rs in node_range)
        for ndind in itertools.product(*tuple(node_range_gen)):
            repldict = {p: ndind[i] for i, p in enumerate(self.params)}
            for expr in expressions:
                for dim, dexpr in enumerate(expr):
                    evaldexpr = _subexpr(dexpr, repldict)
                    rb, re, rs = rng[dim]
                    if rb is None:
                        rng[dim] = (evaldexpr, evaldexpr, 1)
                    else:
                        if evaldexpr < rb:
                            rng[dim] = (evaldexpr, re, rs)
                        if evaldexpr > re:  # The +1 is because ranges are exclusive
                            rng[dim] = (rb, evaldexpr, rs)

        return subsets.Range(rng)


# ConstantRangeMemlet is slow, so it should be evaluated last
MemletPattern.register_pattern(ConstantRangeMemlet)


def propagate_labels_sdfg(sdfg):
    """ Propagates memlets throughout an entire given SDFG. 
        @note: This is an in-place operation on the SDFG.
    """
    for state in sdfg.nodes():
        _propagate_labels(state, sdfg)


def _propagate_labels(g, sdfg):
    """ Propagates memlets throughout one SDFG state. 
        @param g: The state to propagate in.
        @param sdfg: The SDFG in which the state is situated.
        @note: This is an in-place operation on the SDFG state.
    """
    patterns = MemletPattern.patterns()

    # Algorithm:
    # 1. Start propagating information from tasklets outwards (their edges
    #    are hardcoded).
    #    NOTE: This process can be performed in parallel.
    # 2. Traverse the neighboring nodes (topological sort, first forward to
    #    outputs and then backward to inputs).
    #    There are four possibilities:
    #    a. If the neighboring node is a tasklet, skip (such edges are
    #       immutable)
    #    b. If the neighboring node is an array, make sure it is the correct
    #       array. Otherwise, throw a mismatch exception.
    #    c. If the neighboring node is a scope node, and its other edges are
    #       not set, set the results per-array, using the union of the
    #       obtained ranges in the previous depth.
    #    d. If the neighboring node is a scope node, and its other edges are
    #       already set, verify the results per-array, using the union of the
    #       obtained ranges in the previous depth.
    #    NOTE: The SDFG creation process ensures that all edges in the
    #          multigraph are tagged with the appropriate array. In any case
    #          of ambiguity, the function raises an exception.
    # 3. For each edge in the multigraph, collect results and group by array assigned to edge.
    #    Accumulate information about each array in the target node.
    scope_dict = g.scope_dict()

    def stop_at(parent, child):
        # Transients should only propagate in the direction of the
        # non-transient data
        if isinstance(parent,
                      nodes.AccessNode) and parent.desc(sdfg).transient:
            for _, _, _, _, memlet in g.edges_between(parent, child):
                if parent.data != memlet.data:
                    return True
            return False
        if isinstance(child, nodes.AccessNode):
            return False
        return True

    array_data = {}  # type: dict(node -> dict(data -> list(Subset)))
    tasklet_nodes = [
        node for node in g.nodes() if (isinstance(node, nodes.CodeNode) or (
            isinstance(node, nodes.AccessNode) and node.desc(sdfg).transient))
    ]
    # Step 1: Direction - To output
    for start_node in tasklet_nodes:
        for node in nxutil.dfs_topological_sort(
                g, start_node, condition=stop_at):
            _propagate_node(sdfg, g, node, array_data, patterns, scope_dict,
                            True)
    # Step 1: Direction - To input
    array_data = {}
    g.reverse()
    for node in nxutil.dfs_topological_sort(
            g, tasklet_nodes, condition=stop_at):
        _propagate_node(sdfg, g, node, array_data, patterns, scope_dict)

    # To support networkx 1.11
    g.reverse()


# External API
def propagate_memlet(dfg_state, memlet: Memlet, scope_node: nodes.EntryNode,
                     union_inner_edges: bool):
    """ Tries to propagate a memlet through a scope (computes the image of 
        the memlet function applied on an integer set of, e.g., a map range) 
        and returns a new memlet object.
        @param dfg_state: An SDFGState object representing the graph.
        @param memlet: The memlet adjacent to the scope node from the inside.
        @param scope_node: A scope entry or exit node.
        @param union_inner_edges: True if the propagation should take other
                                  neighboring internal memlets within the same
                                  scope into account.
    """
    if isinstance(scope_node, nodes.EntryNode):
        neighboring_edges = dfg_state.out_edges(scope_node)
    elif isinstance(scope_node, nodes.ExitNode):
        neighboring_edges = dfg_state.in_edges(scope_node)
    else:
        raise TypeError('Trying to propagate through a non-scope node')

    # Find other adjacent edges within the connected to the scope node
    # and union their subsets
    if union_inner_edges:
        aggdata = [
            e.data for e in neighboring_edges
            if e.data.data == memlet.data and e.data != memlet
        ]
    else:
        aggdata = []

    aggdata.append(memlet)

    new_subset = _propagate_edge(dfg_state.parent, None,
                                 scope_node, None, memlet, aggdata,
                                 MemletPattern.patterns(), None)

    new_memlet = copy.copy(memlet)
    new_memlet.subset = new_subset
    new_memlet.other_subset = None

    # Number of accesses in the propagated memlet is the sum of the internal
    # number of accesses times the size of the map range set
    new_memlet.num_accesses = (
        sum(m.num_accesses for m in aggdata) * functools.reduce(
            lambda a, b: a * b, scope_node.map.range.size(), 1))

    return new_memlet


def _propagate_node(sdfg,
                    g,
                    node,
                    array_data,
                    patterns,
                    scope_dict,
                    write=False):
    # Step 2: Propagate edges
    # If this is a tasklet, we only propagate to adjacent nodes and not modify edges
    # Special case: starting from reduction, no need for external nodes to compute edges
    if (not isinstance(node, nodes.CodeNode)
            and not isinstance(node, nodes.AccessNode) and node in array_data):
        # Otherwise (if primitive), use current node information and accumulated data
        # on arrays to set the memlets per edge
        for _, _, target, _, memlet in g.out_edges(node):
            # Option (a)
            if (isinstance(target, nodes.CodeNode)):
                continue

            if not isinstance(memlet, Memlet):
                raise AttributeError('Edge does not contain a memlet')

            aggdata = None
            if node in array_data:
                if memlet.data in array_data[node]:
                    aggdata = array_data[node][memlet.data]

            wcr = None
            if aggdata is not None:
                for m in aggdata:
                    if m.wcr is not None:
                        wcr = (m.wcr, m.wcr_identity)
                        break

            # Compute candidate edge
            candidate = _propagate_edge(sdfg, g, node, target, memlet, aggdata,
                                        patterns, not write)
            if candidate is None:
                continue

            # Option (b)
            if isinstance(target, nodes.AccessNode):
                # Check for data mismatch
                if target.data != memlet.data:  #and not target.desc.transient:
                    raise LookupError(
                        'Mismatch between edge data %s and data node %s' %
                        (memlet.data, target.data))

            # Options (c), (d)
            else:
                pass

            # Set new edge value
            memlet.subset = candidate

            # Number of accesses in the propagated memlet is the sum of the internal
            # number of accesses times the size of the map range set
            memlet.num_accesses = (
                sum(m.num_accesses for m in aggdata) * functools.reduce(
                    lambda a, b: a * b, node.map.range.size(), 1))

            # Set WCR, if necessary
            if wcr is not None:
                memlet.wcr, memlet.wcr_identity = wcr

    # Step 3: Accumulate edge information in adjacent node, grouped by array
    for _, _, target, _, memlet in g.out_edges(node):
        if (isinstance(target, nodes.CodeNode)):
            continue

        if not isinstance(memlet, Memlet):
            raise AttributeError('Edge does not contain a memlet')

        # Transients propagate only towards the data they are writing to
        if isinstance(node, nodes.AccessNode) and node.data == memlet.data:
            continue

        # No data
        if memlet.subset is None:
            continue
        #if isinstance(memlet, subsets.SequentialDependency):
        #    continue

        # Accumulate data information on target node
        if target not in array_data:
            array_data[target] = {}
        if memlet.data not in array_data[target]:
            array_data[target][memlet.data] = []
        array_data[target][memlet.data].append(memlet)


def _propagate_edge(sdfg, g, u, v, memlet, aggdata, patterns, reversed):
    if ((isinstance(u, nodes.EntryNode) or isinstance(u, nodes.ExitNode))):
        mapnode = u.map

        if aggdata is None:
            return None

        # Collect data about edge
        data = memlet.data
        expr = [edge.subset for edge in aggdata]

        if memlet.data not in sdfg.arrays:
            raise KeyError('Data descriptor (Array, Stream) "%s" not defined '
                           'in SDFG.' % memlet.data)

        for pattern in patterns:
            if pattern.match(
                    expr,
                [[symbolic.pystr_to_symbolic(p) for p in mapnode.params]],
                    mapnode.range, aggdata):  # Only one level of context
                return pattern.propagate(sdfg.arrays[memlet.data], expr,
                                         mapnode.range)

        # No patterns found. Emit a warning and propagate the entire array
        print('WARNING: Cannot find appropriate memlet pattern to propagate %s'
              % str(expr))

        return subsets.Range.from_array(sdfg.arrays[memlet.data])
    elif isinstance(u, nodes.ConsumeEntry) or isinstance(u, nodes.ConsumeExit):

        # Nothing to analyze/propagate in consume
        return subsets.Range.from_array(sdfg.arrays[memlet.data])

    else:
        raise NotImplementedError('Unimplemented primitive: %s' % type(u))
