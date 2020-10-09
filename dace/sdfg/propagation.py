# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Functionality relating to Memlet propagation (deducing external memlets
    from internal memory accesses and scope ranges). """

import copy
import itertools
import functools
import sympy
import warnings

from dace import registry, subsets, symbolic, dtypes
from dace.memlet import Memlet
from dace.sdfg import nodes


@registry.make_registry
class MemletPattern(object):
    """
    A pattern match on a memlet subset that can be used for propagation.
    """
    def can_be_applied(self, expressions, variable_context, node_range,
                       orig_edges):
        raise NotImplementedError

    def propagate(self, array, expressions, node_range):
        raise NotImplementedError


@registry.make_registry
class SeparableMemletPattern(object):
    """ Memlet pattern that can be applied to each of the dimensions 
        separately. """
    def can_be_applied(self, dim_exprs, variable_context, node_range,
                       orig_edges, dim_index, total_dims):
        raise NotImplementedError

    def propagate(self, array, dim_exprs, node_range):
        raise NotImplementedError


@registry.autoregister
class SeparableMemlet(MemletPattern):
    """ Meta-memlet pattern that applies all separable memlet patterns. """
    def can_be_applied(self, expressions, variable_context, node_range,
                       orig_edges):
        # Assuming correct dimensionality in each of the expressions
        data_dims = len(expressions[0])
        self.patterns_per_dim = [None] * data_dims

        overapprox_range = subsets.Range([
            (rb.approx if isinstance(rb, symbolic.SymExpr) else rb,
             re.approx if isinstance(re, symbolic.SymExpr) else re,
             rs.approx if isinstance(rs, symbolic.SymExpr) else rs)
            for rb, re, rs in node_range
        ])

        for dim in range(data_dims):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[dim], symbolic.SymExpr):
                    dexprs.append(expr[dim].approx)
                elif isinstance(expr[dim], tuple):
                    dexprs.append(
                        (expr[dim][0].approx if isinstance(
                            expr[dim][0], symbolic.SymExpr) else expr[dim][0],
                         expr[dim][1].approx if isinstance(
                             expr[dim][1], symbolic.SymExpr) else expr[dim][1],
                         expr[dim][2].approx if isinstance(
                             expr[dim][2], symbolic.SymExpr) else expr[dim][2]))
                else:
                    dexprs.append(expr[dim])

            for pattern_class in SeparableMemletPattern.extensions().keys():
                smpattern = pattern_class()
                if smpattern.can_be_applied(dexprs, variable_context,
                                            overapprox_range, orig_edges, dim,
                                            data_dims):
                    self.patterns_per_dim[dim] = smpattern
                    break

        return None not in self.patterns_per_dim

    def propagate(self, array, expressions, node_range):
        result = [(None, None, None)] * len(self.patterns_per_dim)

        overapprox_range = subsets.Range([
            (rb.approx if isinstance(rb, symbolic.SymExpr) else rb,
             re.approx if isinstance(re, symbolic.SymExpr) else re,
             rs.approx if isinstance(rs, symbolic.SymExpr) else rs)
            for rb, re, rs in node_range
        ])

        for i, smpattern in enumerate(self.patterns_per_dim):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[i], symbolic.SymExpr):
                    dexprs.append(expr[i].approx)
                elif isinstance(expr[i], tuple):
                    dexprs.append(
                        (expr[i][0].approx if isinstance(
                            expr[i][0], symbolic.SymExpr) else expr[i][0],
                         expr[i][1].approx if isinstance(
                             expr[i][1], symbolic.SymExpr) else expr[i][1],
                         expr[i][2].approx if isinstance(
                             expr[i][2], symbolic.SymExpr) else expr[i][2],
                         expr.tile_sizes[i]))
                else:
                    dexprs.append(expr[i])

            result[i] = smpattern.propagate(array, dexprs, overapprox_range)

        # TODO(later): Not necessarily Range (general integer sets)
        return subsets.Range(result)


@registry.autoregister
class AffineSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches affine expressions, i.e.,
        of the form `a * {index} + b`.
    """
    def can_be_applied(self, dim_exprs, variable_context, node_range,
                       orig_edges, dim_index, total_dims):

        params = variable_context[-1]
        defined_vars = variable_context[-2]
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
                    if p not in subexpr.free_symbols:
                        continue
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
                if (symbolic.issymbolic(step)
                        and self.param in step.free_symbols):
                    return False  # Step must be independent of parameter

            node_rb, node_re, node_rs = node_range[self.paramind]
            if node_rs != 1:
                # Map ranges where the last index is not known
                # exactly are not supported by this pattern.
                return False
            if (any(s not in defined_vars for s in node_rb.free_symbols)
                    or any(s not in defined_vars
                           for s in node_re.free_symbols)):
                # Cannot propagate variables only defined in this scope (e.g.,
                # dynamic map ranges)
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
        interval_len = (result_end - result_begin + 1)
        num_elements = node_rlen * memlet_rlen

        if (interval_len == num_elements
                or interval_len.expand() == num_elements):
            # Continuous access
            result_skip = 1
            result_tile = 1
        else:
            if rt == 1:
                result_skip = (result_end - result_begin - re + rb) / (node_re -
                                                                       node_rb)
                try:
                    if result_skip < 1:
                        result_skip = 1
                except:
                    pass
                result_tile = result_end - result_begin + 1 - (node_rlen -
                                                               1) * result_skip
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


@registry.autoregister
class ModuloSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches modulo expressions, i.e.,
        of the form `f(x) % N`.

        Acts as a meta-pattern: Finds the underlying pattern for `f(x)`.
    """
    def can_be_applied(self, dim_exprs, variable_context, node_range,
                       orig_edges, dim_index, total_dims):
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
            if smpattern.can_be_applied([self.subexpr], variable_context,
                                        node_range, orig_edges, dim_index,
                                        total_dims):
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


@registry.autoregister
class ConstantSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches constant (i.e., unrelated to 
        current scope) expressions.
    """
    def can_be_applied(self, dim_exprs, variable_context, node_range,
                       orig_edges, dim_index, total_dims):
        # Pattern does not support unions of expressions. TODO: Support
        if len(dim_exprs) > 1: return False
        dexpr = dim_exprs[0]

        # Create a wildcard that excludes current map's parameters
        cst = sympy.Wild('cst', exclude=variable_context[-1])

        # Range case
        if isinstance(dexpr, tuple) and len(dexpr) == 3:
            # Try to match a constant expression for the range
            for rngelem in dexpr:
                if dtypes.isconstant(rngelem):
                    continue

                matches = rngelem.match(cst)
                if matches is None or len(matches) != 1:
                    return False
                if not matches[cst].is_constant():
                    return False

        else:  # Single element case
            # Try to match a constant expression
            if not dtypes.isconstant(dexpr):
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


@registry.autoregister
class GenericSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that detects any expression, and propagates 
        interval bounds. Used as a last resort. """
    def can_be_applied(self, dim_exprs, variable_context, node_range,
                       orig_edges, dim_index, total_dims):
        dims = []
        for dim in dim_exprs:
            if isinstance(dim, tuple):
                dims.extend(dim)
            else:
                dims.append(dim)

        self.params = variable_context[-1]
        defined_vars = variable_context[-2]

        used_symbols = set()
        for dim in dims:
            if symbolic.issymbolic(dim):
                used_symbols.update(dim.free_symbols)

        if (used_symbols & set(self.params)
                and any(s not in defined_vars
                        for s in node_range.free_symbols)):
            # Cannot propagate symbols that are undefined in the outer range
            # (e.g., dynamic map ranges).
            return False

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


def _subexpr(dexpr, repldict):
    if isinstance(dexpr, tuple):
        return tuple(_subexpr(d, repldict) for d in dexpr)
    elif isinstance(dexpr, symbolic.SymExpr):
        return dexpr.expr.subs(repldict)
    else:
        return dexpr.subs(repldict)


@registry.autoregister
class ConstantRangeMemlet(MemletPattern):
    """ Memlet pattern that matches arbitrary expressions with constant range.
    """
    def can_be_applied(self, expressions, variable_context, node_range,
                       orig_edges):
        constant_range = True
        for dim in node_range:
            for rngelem in dim:  # For (begin, end, skip)
                if not dtypes.isconstant(rngelem) and not isinstance(
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


def propagate_memlets_sdfg(sdfg):
    """ Propagates memlets throughout an entire given SDFG. 
        :note: This is an in-place operation on the SDFG.
    """
    for state in sdfg.nodes():
        propagate_memlets_state(sdfg, state)


def propagate_memlets_state(sdfg, state):
    """ Propagates memlets throughout one SDFG state. 
        :param sdfg: The SDFG in which the state is situated.
        :param state: The state to propagate in.
        :note: This is an in-place operation on the SDFG state.
    """
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

    # First, propagate nested SDFGs in a bottom-up fashion
    for node in state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            propagate_memlets_sdfg(node.sdfg)

    # Process scopes from the leaves upwards
    propagate_memlets_scope(sdfg, state, state.scope_leaves())


def propagate_memlets_scope(sdfg, state, scopes):
    """ 
    Propagate memlets from the given scopes outwards. 
    :param sdfg: The SDFG in which the scopes reside.
    :param state: The SDFG state in which the scopes reside.
    :param scopes: The ScopeTree object or a list thereof to start from.
    :note: This operation is performed in-place on the given SDFG.
    """
    from dace.sdfg.scope import ScopeTree

    if isinstance(scopes, ScopeTree):
        scopes_to_process = [scopes]
    else:
        scopes_to_process = scopes

    next_scopes = set()

    # Process scopes from the inputs upwards, propagating edges at the
    # entry and exit nodes
    while len(scopes_to_process) > 0:
        for scope in scopes_to_process:
            if scope.entry is None:
                continue

            # Propagate out of entry
            _propagate_node(state, scope.entry)

            # Propagate out of exit
            _propagate_node(state, scope.exit)

            # Add parent to next frontier
            next_scopes.add(scope.parent)
        scopes_to_process = next_scopes
        next_scopes = set()


def _propagate_node(dfg_state, node):
    if isinstance(node, nodes.EntryNode):
        internal_edges = [
            e for e in dfg_state.out_edges(node)
            if e.src_conn and e.src_conn.startswith('OUT_')
        ]
        external_edges = [
            e for e in dfg_state.in_edges(node)
            if e.dst_conn and e.dst_conn.startswith('IN_')
        ]
    else:
        internal_edges = [
            e for e in dfg_state.in_edges(node)
            if e.dst_conn and e.dst_conn.startswith('IN_')
        ]
        external_edges = [
            e for e in dfg_state.out_edges(node)
            if e.src_conn and e.src_conn.startswith('OUT_')
        ]

    for edge in external_edges:
        if edge.data.is_empty():
            new_memlet = Memlet()
        else:
            internal_edge = next(e for e in internal_edges
                                 if e.data.data == edge.data.data)
            new_memlet = propagate_memlet(dfg_state, internal_edge.data, node,
                                          True)
        edge._data = new_memlet


# External API
def propagate_memlet(dfg_state,
                     memlet: Memlet,
                     scope_node: nodes.EntryNode,
                     union_inner_edges: bool,
                     arr=None):
    """ Tries to propagate a memlet through a scope (computes the image of 
        the memlet function applied on an integer set of, e.g., a map range) 
        and returns a new memlet object.
        :param dfg_state: An SDFGState object representing the graph.
        :param memlet: The memlet adjacent to the scope node from the inside.
        :param scope_node: A scope entry or exit node.
        :param union_inner_edges: True if the propagation should take other
                                  neighboring internal memlets within the same
                                  scope into account.
    """
    if isinstance(scope_node, nodes.EntryNode):
        entry_node = scope_node
        neighboring_edges = dfg_state.out_edges(scope_node)
    elif isinstance(scope_node, nodes.ExitNode):
        entry_node = dfg_state.scope_dict()[scope_node]
        neighboring_edges = dfg_state.in_edges(scope_node)
    else:
        raise TypeError('Trying to propagate through a non-scope node')
    if memlet.is_empty():
        return Memlet()

    sdfg = dfg_state.parent
    scope_node_symbols = set(conn for conn in entry_node.in_connectors
                             if not conn.startswith('IN_'))
    defined_vars = [
        symbolic.pystr_to_symbolic(s)
        for s in dfg_state.symbols_defined_at(entry_node).keys()
        if s not in scope_node_symbols
    ]

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

    if arr is None:
        if memlet.data not in sdfg.arrays:
            raise KeyError('Data descriptor (Array, Stream) "%s" not defined '
                           'in SDFG.' % memlet.data)
        arr = sdfg.arrays[memlet.data]

    # Propagate subset
    if isinstance(entry_node, nodes.MapEntry):
        mapnode = entry_node.map

        variable_context = [
            defined_vars,
            [symbolic.pystr_to_symbolic(p) for p in mapnode.params]
        ]

        new_subset = None
        for md in aggdata:
            tmp_subset = None
            for pclass in MemletPattern.extensions():
                pattern = pclass()
                if pattern.can_be_applied([md.subset], variable_context,
                                          mapnode.range, [md]):
                    tmp_subset = pattern.propagate(arr, [md.subset],
                                                   mapnode.range)
                    break
            else:
                # No patterns found. Emit a warning and propagate the entire
                # array
                warnings.warn('Cannot find appropriate memlet pattern to '
                              'propagate %s through %s' %
                              (str(md.subset), str(mapnode.range)))
                tmp_subset = subsets.Range.from_array(arr)

            # Union edges as necessary
            if new_subset is None:
                new_subset = tmp_subset
            else:
                old_subset = new_subset
                new_subset = subsets.union(new_subset, tmp_subset)
                if new_subset is None:
                    warnings.warn('Subset union failed between %s and %s ' %
                                  (old_subset, tmp_subset))
                    break

        # Some unions failed
        if new_subset is None:
            new_subset = subsets.Range.from_array(arr)

        assert new_subset is not None

    elif isinstance(entry_node, nodes.ConsumeEntry):
        # Nothing to analyze/propagate in consume
        new_subset = subsets.Range.from_array(arr)
    else:
        raise NotImplementedError('Unimplemented primitive: %s' %
                                  type(entry_node))
    ### End of subset propagation

    new_memlet = copy.copy(memlet)
    new_memlet.subset = new_subset
    new_memlet.other_subset = None

    # Number of accesses in the propagated memlet is the sum of the internal
    # number of accesses times the size of the map range set (unbounded dynamic)
    new_memlet.num_accesses = (
        sum(m.num_accesses for m in aggdata) *
        functools.reduce(lambda a, b: a * b, scope_node.map.range.size(), 1))
    if any(m.dynamic for m in aggdata):
        new_memlet.dynamic = True
    elif symbolic.issymbolic(new_memlet.num_accesses) and any(
            s not in defined_vars
            for s in new_memlet.num_accesses.free_symbols):
        new_memlet.dynamic = True
        new_memlet.num_accesses = 0

    return new_memlet
