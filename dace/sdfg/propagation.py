# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Functionality relating to Memlet propagation (deducing external memlets
    from internal memory accesses and scope ranges). """

import copy
from dace.symbolic import issymbolic, pystr_to_symbolic
from dace.sdfg.nodes import AccessNode
import itertools
import functools
import sympy
from sympy import ceiling
from sympy.concrete.summations import Sum
import warnings

from dace import registry, subsets, symbolic, dtypes, data
from dace.memlet import Memlet
from dace.sdfg import nodes
from typing import List, Set


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
                subexprs = [dexpr, dexpr]

            elif isinstance(dexpr, tuple) and len(dexpr) == 3:  # Affine range
                subexprs = [dexpr[0], dexpr[1]]
                step = dexpr[2]

            if subexprs is None:  # Something else
                return False

            for i, subexpr in enumerate(subexprs):
                if not issymbolic(subexpr):
                    subexpr = pystr_to_symbolic(subexpr)

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
            result_begin = subexprs[0].subs(self.param, node_rb).expand()
            if node_rs != 1:
                # Special case: i:i+stride for a begin:end:stride range
                if node_rb == result_begin and bre + 1 == node_rs and step == 1:
                    pass
                else:
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

        # Special case: i:i+stride for a begin:end:stride range
        if (node_rb == result_begin and (re - rb + 1) == node_rs and rs == 1
                and rt == 1):
            return (node_rb, node_re, 1, 1)

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


def propagate_states(sdfg) -> None:
    """
    Annotate the states of an SDFG with the number of executions.

    :param sdfg: The SDFG to annotate.
    """
    from dace.sdfg.sdfg import InterstateEdge
    from dace.transformation.interstate.branch_annotation import AnnotateBranch
    from dace.transformation.interstate.loop_annotation import AnnotateLoop

    # Clean up the state machine by separating combined condition and assignment
    # edges.
    for e in list(sdfg.edges()):
        if e.data.assignments and not e.data.is_unconditional():
            tmpstate = sdfg.add_state()
            sdfg.add_edge(
                e.src,
                tmpstate,
                InterstateEdge(condition=e.data.condition)
            )
            sdfg.add_edge(
                tmpstate,
                e.dst,
                InterstateEdge(assignments=e.data.assignments)
            )
            sdfg.remove_edge(e)

    # Initialize states.
    for v in sdfg.nodes():
        v.visited = False

    # Annotate for-loops with ranges and find loop guards.
    sdfg.apply_transformations_repeated(AnnotateLoop)

    # Annotate branch constructs.
    sdfg.apply_transformations_repeated(AnnotateBranch)

    # Identify and annotate any un-annotated loops (e.g. while loops).
    unannotated_cycle_states = []
    for cycle in sdfg.find_cycles():
        has_loop_guard = False
        no_unannotated_states = True
        for v in cycle:
            if not v.ranges:
                no_unannotated_states = False
            if getattr(v, 'is_loop_guard', False):
                has_loop_guard = True
        if not (has_loop_guard and no_unannotated_states):
            # This loop is no fully annotated for loop.
            unannotated_cycle_states.extend(cycle)

    # Keep track of branches still needing to be traversed.
    remaining_branches = []
    # Keep track of encountered nested loop variables.
    itvar_stack = []
    # Keep track of states that fully merge a previous conditional split. We do
    # this so we can remove the dynamic executions flag for those states.
    full_merge_states = set()

    state = sdfg.start_state
    proposed_executions = 1
    proposed_dynamic = False
    while state is not None:
        out_degree = sdfg.out_degree(state)
        out_edges = sdfg.out_edges(state)

        if state.visited:
            if proposed_executions == 0 and proposed_dynamic:
                state.executions = proposed_executions
                state.dynamic_executions = proposed_dynamic
            elif getattr(state, 'is_loop_guard', False):
                # If we encounter a loop guard that's already been visited,
                # we've finished traversing a loop and can remove that loop's
                # iteration variable from the stack. We additively merge the
                # number of executions.
                if state.executions != 0 and not state.dynamic_executions:
                    state.executions += proposed_executions
                itvar_stack.pop()
            else:
                state.executions = sympy.Max(
                    state.executions,
                    proposed_executions
                ).doit()
                if state in full_merge_states:
                    state.dynamic_executions = False
                else:
                    state.dynamic_executions = (
                        state.dynamic_executions or proposed_dynamic
                    )
        elif proposed_dynamic and proposed_executions == 0:
            # Dynamic unbounded.
            state.visited = True
            state.executions = proposed_executions
            state.dynamic_executions = proposed_dynamic
            # This gets pushed through to all children unconditionally.
            if len(out_edges) > 0:
                next_edge = out_edges.pop()
                for oedge in out_edges:
                    remaining_branches.append({
                        'state': oedge.dst,
                        'proposed_executions': proposed_executions,
                        'proposed_dynamic': proposed_dynamic,
                    })
                state = next_edge.dst
                continue
        else:
            state.visited = True
            if state in full_merge_states:
                # If this state fully merges a conditional branch, this turns
                # dynamic executions back off.
                proposed_dynamic = False
            state.executions = proposed_executions
            state.dynamic_executions = proposed_dynamic

            if out_degree == 1:
                # Continue with the only child branch.
                state = out_edges[0].dst
                continue
            elif out_degree > 1:
                if getattr(state, 'is_loop_guard', False):
                    itvar = symbolic.symbol(state.itvar)
                    loop_range = state.ranges[state.itvar]
                    start = loop_range[0][0]
                    stop = loop_range[0][1]
                    stride = loop_range[0][2]

                    # Calculate the number of loop executions.
                    # This resolves ranges based on the order of iteration
                    # variables pushed on to the stack if we're in a nested
                    # loop.
                    loop_executions = Sum(
                        ceiling(1 / stride),
                        (itvar, start, stop)
                    )
                    for outer_itvar_string in reversed(itvar_stack):
                        outer_range = state.ranges[
                            outer_itvar_string
                        ]
                        outer_start = outer_range[0][0]
                        outer_stop = outer_range[0][1]
                        outer_stride = outer_range[0][2]
                        outer_itvar = symbolic.symbol(outer_itvar_string)
                        loop_executions = Sum(
                            ceiling(loop_executions / outer_stride),
                            (outer_itvar, outer_start, outer_stop)
                        )
                    loop_executions = loop_executions.doit()

                    itvar_stack.append(state.itvar)

                    loop_state = state.condition_edge.dst
                    end_state = (
                        out_edges[0].dst if out_edges[1].dst == loop_state
                        else out_edges[1].dst
                    )

                    remaining_branches.append({
                        'state': end_state,
                        'proposed_executions': state.executions,
                        'proposed_dynamic': proposed_dynamic,
                    })

                    # Traverse down the loop.
                    state = loop_state
                    proposed_executions = loop_executions
                    continue

                # Conditional split or unannotated (dynamic unbounded) loop.
                unannotated_loop_edge = None
                for oedge in out_edges:
                    if oedge.dst in unannotated_cycle_states:
                        # This is an unannotated loop going down this branch.
                        unannotated_loop_edge = oedge

                if unannotated_loop_edge is not None:
                    # Traverse as an unbounded loop.
                    out_edges.remove(unannotated_loop_edge)
                    for oedge in out_edges:
                        remaining_branches.append({
                            'state': oedge.dst,
                            'proposed_executions': state.executions,
                            'proposed_dynamic': False,
                        })
                    state.is_loop_guard = True
                    proposed_executions = 0
                    proposed_dynamic = True
                    state = unannotated_loop_edge.dst
                    continue
                else:
                    # Traverse as a conditional split.
                    proposed_executions = state.executions
                    proposed_dynamic = True
                    if (getattr(state, 'full_merge_state', None) is not None and
                        not state.dynamic_executions):
                        full_merge_states.add(state.full_merge_state)
                    for oedge in out_edges:
                        remaining_branches.append({
                            'state': oedge.dst,
                            'proposed_executions': proposed_executions,
                            'proposed_dynamic': proposed_dynamic,
                        })

        if len(remaining_branches) > 0:
            # Traverse down next remaining branch.
            branch = remaining_branches.pop()
            state = branch['state']
            proposed_executions = branch['proposed_executions']
            proposed_dynamic = branch['proposed_dynamic']
        else:
            # No more remaining branch, done.
            state = None
            break


def propagate_memlets_sdfg(sdfg, border_memlets=None):
    """ Propagates memlets throughout an entire given SDFG. 
        :param border_memlets: A map between connectors to and from this SDFG
                               (if it's a nested SDFG), to their connected
                               'border' memlets.
        :note: This is an in-place operation on the SDFG.
    """
    for state in sdfg.nodes():
        propagate_memlets_state(sdfg, state)

    propagate_states(sdfg)

    if border_memlets is not None:
        # For each state, go through all access nodes corresponding to any in-
        # or out-connectors to and from this SDFG. Given those access nodes,
        # collect the corresponding memlets and use them to calculate the
        # memlet volume and subset corresponding to the outside memlet attached
        # to that connector. This is passed out via `border_memlets` and
        # propagated along from there.
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, AccessNode):
                    for direction in border_memlets:
                        if (node.label in border_memlets[direction]):
                            memlet = border_memlets[direction][node.label]

                            # Collect the edges to/from this access node,
                            # depending on the direction the connector leads in.
                            edges = []
                            if direction == 'in':
                                edges = state.out_edges(node)
                            elif direction == 'out':
                                edges = state.in_edges(node)

                            # Collect all memlets belonging to this access node,
                            # and accumulate the total volume between them.
                            memlets = []
                            for edge in edges:
                                inside_memlet = edge.data
                                memlets.append(inside_memlet)

                                if memlet is None:
                                    # Use the first encountered memlet as a
                                    # 'border' memlet and accumulate the sum
                                    # on it.
                                    memlet = copy.deepcopy(inside_memlet)
                                    memlet.dynamic = False
                                    memlet.volume = 0
                                    memlet.subset = None
                                    memlet.other_subset = None
                                    memlet._is_data_src = True
                                    border_memlets[direction][
                                        node.label
                                    ] = memlet

                                if memlet.dynamic and memlet.volume == 0:
                                    # Dynamic unbounded - this won't change.
                                    continue
                                elif ((inside_memlet.dynamic and
                                       inside_memlet.volume == 0) or
                                      (state.dynamic_executions and
                                       state.executions == 0)):
                                    # At least one dynamic unbounded memlet lets
                                    # the sum be dynamic unbounded.
                                    memlet.dynamic = True
                                    memlet.volume = 0
                                else:
                                    memlet.volume += (
                                        inside_memlet.volume * state.executions
                                    )
                                    memlet.dynamic = (
                                        memlet.dynamic or
                                        inside_memlet.dynamic or
                                        state.dynamic_executions
                                    )

                            # Given all of this access nodes' memlets, propagate
                            # the subset according to the state's variable
                            # ranges.
                            if len(memlets) > 0:
                                params = []
                                ranges = []
                                for symbol in state.ranges:
                                    params.append(symbol)
                                    ranges.append(state.ranges[symbol][0])

                                if len(params) == 0 or len(ranges) == 0:
                                    params = ['dummy']
                                    ranges = [(0, 0, 1)]

                                # Propagate the subset based on the direction
                                # this memlet is pointing. If we're accessing
                                # from an incoming connector, propagate the
                                # source subset, if we're going to an outgoing
                                # connector, propagate the destination subset.
                                use_dst = False
                                if direction == 'out':
                                    use_dst = True
                                subset = propagate_subset(
                                    memlets,
                                    sdfg.arrays[node.label],
                                    params,
                                    subsets.Range(ranges),
                                    defined_variables=None,
                                    use_dst=use_dst
                                ).subset

                                # If the border memlet already has a set range,
                                # compute the union of the ranges to merge the
                                # subsets.
                                if memlet.subset is not None:
                                    if memlet.subset.dims() != subset.dims():
                                        raise ValueError('Cannot merge subset '
                                                            'ranges of unequal '
                                                            'dimension!')
                                    else:
                                        memlet.subset = subsets.union(
                                            memlet.subset,
                                            subset
                                        )
                                else:
                                    memlet.subset = subset


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

    from dace.transformation.helpers import unsqueeze_memlet

    # First, propagate nested SDFGs in a bottom-up fashion
    for node in state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            # Build a map of connectors to associated 'border' memlets inside
            # the nested SDFG. This map will be populated with memlets once they
            # get propagated in the SDFG.
            border_memlets = {
                'in': {},
                'out': {},
            }
            for connector in node.in_connectors:
                border_memlets['in'][connector] = None
            for connector in node.out_connectors:
                border_memlets['out'][connector] = None

            # Propagate memlets inside the nested SDFG.
            propagate_memlets_sdfg(node.sdfg, border_memlets)

            # Make sure any potential NSDFG symbol mapping is correctly reversed
            # when propagating out.
            for direction in border_memlets:
                for connector in border_memlets[direction]:
                    border_memlet = border_memlets[direction][connector]
                    if border_memlet is not None:
                        border_memlet.substitute_symbol(node.symbol_mapping)

            # Propagate the inside 'border' memlets outside the SDFG by
            # offsetting, and unsqueezing if necessary.
            for iedge in state.in_edges(node):
                if iedge.dst_conn in border_memlets['in']:
                    internal_memlet = border_memlets['in'][iedge.dst_conn]
                    if internal_memlet is None:
                        continue
                    iedge._data = unsqueeze_memlet(
                        internal_memlet,
                        iedge.data,
                        True
                    )
                    if symbolic.issymbolic(iedge.data.volume):
                        if any(str(s) not in sdfg.symbols
                               for s in iedge.data.volume.free_symbols):
                            iedge.data.volume = 0
                            iedge.data.dynamic = True
            for oedge in state.out_edges(node):
                if oedge.src_conn in border_memlets['out']:
                    internal_memlet = border_memlets['out'][oedge.src_conn]
                    if internal_memlet is None:
                        continue
                    oedge._data = unsqueeze_memlet(
                        internal_memlet,
                        oedge.data,
                        True
                    )
                    if symbolic.issymbolic(oedge.data.volume):
                        if any(str(s) not in sdfg.symbols
                               for s in oedge.data.volume.free_symbols):
                            oedge.data.volume = 0
                            oedge.data.dynamic = True

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
        entry_node = dfg_state.entry_node(scope_node)
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
        return propagate_subset(aggdata, arr, mapnode.params, mapnode.range,
                                defined_vars, use_dst=False)

    elif isinstance(entry_node, nodes.ConsumeEntry):
        # Nothing to analyze/propagate in consume
        new_memlet = copy.copy(memlet)
        new_memlet.subset = subsets.Range.from_array(arr)
        new_memlet.other_subset = None
        new_memlet.volume = 0
        new_memlet.dynamic = True
        return new_memlet
    else:
        raise NotImplementedError('Unimplemented primitive: %s' %
                                  type(entry_node))


# External API
def propagate_subset(
        memlets: List[Memlet],
        arr: data.Data,
        params: List[str],
        rng: subsets.Subset,
        defined_variables: Set[symbolic.SymbolicType] = None,
        use_dst: bool = False) -> Memlet:
    """ Tries to propagate a list of memlets through a range (computes the 
        image of the memlet function applied on an integer set of, e.g., a 
        map range) and returns a new memlet object.
        :param memlets: The memlets to propagate.
        :param arr: Array descriptor for memlet (used for obtaining extents).
        :param params: A list of variable names.
        :param rng: A subset with dimensionality len(params) that contains the
                    range to propagate with.
        :param defined_variables: A set of symbols defined that will remain the
                                  same throughout propagation. If None, assumes
                                  that all symbols outside of `params` have been
                                  defined.
        :return: Memlet with propagated subset and volume.
    """
    # Argument handling
    if defined_variables is None:
        # Default defined variables is "everything but params"
        defined_variables = set()
        defined_variables |= rng.free_symbols
        for memlet in memlets:
            defined_variables |= memlet.free_symbols
        defined_variables -= set(params)
        defined_variables = set(
            symbolic.pystr_to_symbolic(p) for p in defined_variables)

    # Propagate subset
    variable_context = [
        defined_variables, [symbolic.pystr_to_symbolic(p) for p in params]
    ]

    new_subset = None
    for md in memlets:
        tmp_subset = None

        subset = None
        if use_dst and md.dst_subset is not None:
            subset = md.dst_subset
        elif not use_dst and md.src_subset is not None:
            subset = md.src_subset
        else:
            subset = md.subset

        for pclass in MemletPattern.extensions():
            pattern = pclass()
            if pattern.can_be_applied([subset], variable_context, rng, [md]):
                tmp_subset = pattern.propagate(arr, [subset], rng)
                break
        else:
            # No patterns found. Emit a warning and propagate the entire
            # array
            warnings.warn('Cannot find appropriate memlet pattern to '
                          'propagate %s through %s' %
                          (str(subset), str(rng)))
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
    ### End of subset propagation

    # Create new memlet
    new_memlet = copy.copy(memlets[0])
    new_memlet.subset = new_subset
    new_memlet.other_subset = None

    # Propagate volume:
    # Number of accesses in the propagated memlet is the sum of the internal
    # number of accesses times the size of the map range set (unbounded dynamic)
    new_memlet.volume = (sum(m.volume for m in memlets) *
                         functools.reduce(lambda a, b: a * b, rng.size(), 1))
    if any(m.dynamic for m in memlets):
        new_memlet.dynamic = True
    elif symbolic.issymbolic(new_memlet.volume) and any(
            s not in defined_variables for s in new_memlet.volume.free_symbols):
        new_memlet.dynamic = True
        new_memlet.volume = 0

    return new_memlet
