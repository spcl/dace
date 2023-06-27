# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functionality relating to Memlet propagation (deducing external memlets
from internal memory accesses and scope ranges).
"""

from collections import defaultdict, deque
import copy
from dace.symbolic import issymbolic, pystr_to_symbolic, simplify
import itertools
import functools
from dace.transformation.pass_pipeline import Modifies, Pass
import sympy
from sympy import ceiling
from sympy.concrete.summations import Sum
import warnings
import networkx as nx

from dace import registry, subsets, symbolic, dtypes, data
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFGState, graph as gr
import dace
from typing import List, Set, Type, Union
from dace.sdfg.analysis import cfg
from dace.transformation import pass_pipeline as ppl
from typing import Any, Dict, Iterator, List, Optional, Set, Type, Union


approximation_dict = {}
# dictionary that maps loop headers to "border memlets" that are written to in the corresponding loop
loop_write_dict: dict[SDFGState, dict[str, Memlet]] = {}
loop_dict: dict[SDFGState, (SDFGState, SDFGState, list[SDFGState], str, subsets.Range)] = {}



@registry.make_registry
class MemletPattern(object):
    """
    A pattern match on a memlet subset that can be used for propagation.
    """

    def can_be_applied(self, expressions, variable_context, node_range, orig_edges):
        raise NotImplementedError

    def propagate(self, array, expressions, node_range):
        raise NotImplementedError


@registry.make_registry
class SeparableMemletPattern(object):
    """ Memlet pattern that can be applied to each of the dimensions 
        separately. """

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):
        raise NotImplementedError

    def propagate(self, array, dim_exprs, node_range):
        raise NotImplementedError


@registry.autoregister
class SeparableMemlet(MemletPattern):
    """ Meta-memlet pattern that applies all separable memlet patterns. """

    def can_be_applied(self, expressions, variable_context, node_range, orig_edges):
        # Assuming correct dimensionality in each of the expressions
        data_dims = len(expressions[0])
        self.patterns_per_dim = [None] * data_dims


        # Return False if index appears in multiple dimensions
        params = variable_context[-1]
        for expr in expressions:
            for param in params:
                occured_before = False
                for dim in range(data_dims):
                    free_symbols = []
                    if isinstance(expr[dim], symbolic.SymExpr):
                        free_symbols += expr[dim].expr.free_symbols
                    elif isinstance(expr[dim], tuple):
                        free_symbols += expr[dim][0].expr.free_symbols if isinstance(expr[dim][0], symbolic.SymExpr) else [expr[dim][0]]
                        free_symbols += expr[dim][1].expr.free_symbols if isinstance(expr[dim][1], symbolic.SymExpr) else [expr[dim][1]]
                        free_symbols += expr[dim][2].expr.free_symbols if isinstance(expr[dim][2], symbolic.SymExpr) else [expr[dim][2]]
                    else:
                        free_symbols += [expr[dim]]
                    
                    if param in free_symbols:
                        if occured_before:
                            return False
                        occured_before = True
                        


        overapprox_range = subsets.Range([(rb.expr if isinstance(rb, symbolic.SymExpr) else rb,
                                           re.expr if isinstance(re, symbolic.SymExpr) else re,
                                           rs.expr if isinstance(rs, symbolic.SymExpr) else rs)
                                          for rb, re, rs in node_range])

        for dim in range(data_dims):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[dim], symbolic.SymExpr):
                    dexprs.append(expr[dim].expr)
                elif isinstance(expr[dim], tuple):
                    dexprs.append((expr[dim][0].expr if isinstance(expr[dim][0], symbolic.SymExpr) else expr[dim][0],
                                   expr[dim][1].expr if isinstance(expr[dim][1], symbolic.SymExpr) else expr[dim][1],
                                   expr[dim][2].expr if isinstance(expr[dim][2], symbolic.SymExpr) else expr[dim][2]))
                else:
                    dexprs.append(expr[dim])

            for pattern_class in SeparableMemletPattern.extensions().keys():
                smpattern = pattern_class()
                if smpattern.can_be_applied(dexprs, variable_context, overapprox_range, orig_edges, dim, data_dims):
                    self.patterns_per_dim[dim] = smpattern
                    break

        return None not in self.patterns_per_dim

    def propagate(self, array, expressions, node_range):
        result = [(None, None, None)] * len(self.patterns_per_dim)

        overapprox_range = subsets.Range([(rb.expr if isinstance(rb, symbolic.SymExpr) else rb,
                                           re.expr if isinstance(re, symbolic.SymExpr) else re,
                                           rs.expr if isinstance(rs, symbolic.SymExpr) else rs)
                                          for rb, re, rs in node_range])

        for i, smpattern in enumerate(self.patterns_per_dim):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[i], symbolic.SymExpr):
                    dexprs.append(expr[i].expr)
                elif isinstance(expr[i], tuple):
                    dexprs.append((expr[i][0].expr if isinstance(expr[i][0], symbolic.SymExpr) else expr[i][0],
                                   expr[i][1].expr if isinstance(expr[i][1], symbolic.SymExpr) else expr[i][1],
                                   expr[i][2].expr if isinstance(expr[i][2], symbolic.SymExpr) else expr[i][2],
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
    # FIXME: this returns overapproximations/falls back to the full array range. Even for seemingly
    # manageable cases like A[3 * i]

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):

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
                if (symbolic.issymbolic(step) and self.param in step.free_symbols):
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
                    or any(s not in defined_vars for s in node_re.free_symbols)):
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

        # Special case: multiplier < 0
        if (self.multiplier < 0) == True:
            result_begin, result_end = result_end, result_begin

        # TODO: What is this??? Makes no sense for underapproximation!! Also why is 1 added to re-rb
        # Special case: i:i+stride for a begin:end:stride range
        if (node_rb == result_begin and (re - rb + 1) == node_rs and rs == 1 and rt == 1):
            return (node_rb, node_re, 1, 1)

        # Experimental
        # This should be using sympy.floor
        memlet_start_pts = ((re - rt + 1 - rb) / rs) + 1
        memlet_rlen = memlet_start_pts.expand() * rt
        interval_len = (result_end - result_begin + 1)
        num_elements = node_rlen * memlet_rlen

        if (interval_len == num_elements or interval_len.expand() == num_elements):
            # Continuous access
            result_skip = 1
            result_tile = 1
        else:
            if rt == 1:
                result_skip = (result_end - result_begin - re + rb) / (node_re - node_rb)
                try:
                    if result_skip < 1:
                        result_skip = 1
                except:
                    pass
                result_tile = result_end - result_begin + 1 - (node_rlen - 1) * result_skip
            else:
                candidate_skip = rs
                candidate_tile = rt * node_rlen
                candidate_lstart_pt = result_end - result_begin + 1 - candidate_tile
                if simplify(candidate_lstart_pt / (num_elements / candidate_tile - 1)) == candidate_skip:
                    result_skip = rs
                    result_tile = rt * node_rlen
                else:
                    result_skip = rs / node_rlen
                    result_tile = rt

            if result_skip == result_tile or result_skip == 1:
                result_skip = 1
                result_tile = 1

        result_begin = simplify(result_begin)
        result_end = simplify(result_end)
        result_skip = simplify(result_skip)
        result_tile = simplify(result_tile)

        return (result_begin, result_end, result_skip, result_tile)


@registry.autoregister
class ModuloSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches modulo expressions, i.e.,
        of the form `f(x) % N`.

        Acts as a meta-pattern: Finds the underlying pattern for `f(x)`.
    """

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):
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
            if smpattern.can_be_applied([self.subexpr], variable_context, node_range, orig_edges, dim_index,
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
            print('WARNING: Cannot evaluate relational %s, assuming true.' % (se_range[0] < 0))
        try:
            if se_range[1] > self.modulo:
                se_range = (0, self.modulo, se_range[2])
        except TypeError:  # cannot determine truth value of Relational
            print('WARNING: Cannot evaluate relational %s, assuming true.' % (se_range[1] > self.modulo))

        return se_range


@registry.autoregister
class ConstantSMemlet(SeparableMemletPattern):
    """ Separable memlet pattern that matches constant (i.e., unrelated to 
        current scope) expressions.
    """

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):
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

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):
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
                and any(symbolic.pystr_to_symbolic(s) not in defined_vars for s in node_range.free_symbols)):
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

            if (node_rs < 0) == True:
                node_rb, node_re, node_rs = node_re, node_rb, -node_rs

            # Get true range end
            pos_firstindex = node_rb
            neg_firstindex = node_re
            pos_lastindex = node_re
            neg_lastindex = node_rb
            if node_rs != 1:
                pos_lastindex = symbolic.pystr_to_symbolic(
                    '%s + int_floor(%s - %s, %s) * %s' %
                    (symbolic.symstr(node_rb, cpp_mode=False), symbolic.symstr(node_re, cpp_mode=False),
                     symbolic.symstr(node_rb, cpp_mode=False), symbolic.symstr(
                         node_rs, cpp_mode=False), symbolic.symstr(node_rs, cpp_mode=False)))
                neg_firstindex = pos_lastindex

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

            # Support for affine expressions with a negative multiplier
            firstindex = pos_firstindex
            lastindex = pos_lastindex
            a = sympy.Wild('a', exclude=self.params)
            b = sympy.Wild('b', exclude=self.params)
            if result_begin is None:
                matches = rb.match(a * self.params[idx] + b)
            else:
                matches = result_begin.match(a * self.params[idx] + b)
            if matches and (matches[a] < 0) == True:
                firstindex = neg_firstindex
            if result_end is None:
                matches = re.match(a * self.params[idx] + b)
            else:
                matches = result_end.match(a * self.params[idx] + b)
            if matches and (matches[a] < 0) == True:
                lastindex = neg_lastindex

            if result_begin is None:
                result_begin = rb.subs(self.params[idx], firstindex)
            else:
                result_begin = result_begin.subs(self.params[idx], firstindex)
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

    def can_be_applied(self, expressions, variable_context, node_range, orig_edges):
        constant_range = True
        for dim in node_range:
            for rngelem in dim:  # For (begin, end, skip)
                if not dtypes.isconstant(rngelem) and not isinstance(rngelem, sympy.Number):
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

class UnderapproximateWrites(ppl.Pass):

    def depends_on(self) -> Set[Type[Pass] | Pass]:
        # pass does not depend on any other pass so far
        # TODO: changes this for later versions
        return super().depends_on()
    
    def modifies(self) -> Modifies:
        return ppl.Modifies.Nothing
    
    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Symbols | ppl.Modifies.Nodes
    
    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> Dict[gr.MultiConnectorEdge[Memlet], gr.MultiConnectorEdge[Memlet]]:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """
        approximation_dict.clear()
        loop_write_dict.clear()
        loop_dict.clear()

        # fill the approximation dictionary with the original edges as keys and the edges with the
        # approximated memlets as values
        for (edge, parent) in sdfg.all_edges_recursive():
            if isinstance(parent, SDFGState):
                approximation_dict[edge] = copy.deepcopy(edge.data)
                if not isinstance(approximation_dict[edge].subset, subsets.Subsetlist) and not approximation_dict[edge].subset == None:
                    approximation_dict[edge].subset = subsets.Subsetlist([approximation_dict[edge].subset])
                if not isinstance(approximation_dict[edge].dst_subset, subsets.Subsetlist) and not approximation_dict[edge].dst_subset == None:
                    approximation_dict[edge].dst_subset = subsets.Subsetlist([approximation_dict[edge].dst_subset])
                if not isinstance(approximation_dict[edge].src_subset, subsets.Subsetlist) and not approximation_dict[edge].src_subset == None:
                    approximation_dict[edge].src_subset = subsets.Subsetlist([approximation_dict[edge].src_subset])

        self.propagate_memlets_sdfg(sdfg)

        # TODO: make sure that no Memlet contains None as subset in the first place
        for entry in approximation_dict.values():
            if entry.subset is None:
                entry.subset = subsets.Subsetlist([])
        return  {
            "approximation": approximation_dict,
            "loop_approximation": loop_write_dict,
            "loops" : loop_dict
        }



    def _annotate_loop_ranges(self, sdfg, unannotated_cycle_states) -> Dict[SDFGState, tuple[SDFGState, SDFGState, list[SDFGState], str, subsets.Range]]:
            """
            Modified version of _annotate_loop_ranges from dace.sdfg.propagation
            that also returns the identified loops in a dictionary

            Annotate each valid for loop construct with its loop variable ranges.
            :param sdfg: The SDFG in which to look.
            :param unannotated_cycle_states: List of states in cycles without valid
                                            for loop ranges.
            :return: dictionary mapping loop headers to first state in the loop,
                    the set of states enclosed by the loop, the itearation variable,
                    the range of the iterator variable
            """

            # We import here to avoid cyclic imports.
            from dace.transformation.interstate.loop_detection import find_for_loop
            from dace.sdfg import utils as sdutils

            # dictionary mapping loop headers to beginstate, loopstates, looprange
            identified_loops = {}
            for cycle in sdfg.find_cycles():
                # In each cycle, try to identify a valid loop guard state.
                guard = None
                begin = None
                itvar = None
                for v in cycle:
                    # Try to identify a valid for-loop guard.
                    in_edges = sdfg.in_edges(v)
                    out_edges = sdfg.out_edges(v)

                    # A for-loop guard has two or more incoming edges (1 increment and
                    # n init, all identical), and exactly two outgoing edges (loop and
                    # exit loop).
                    if len(in_edges) < 2 or len(out_edges) != 2:
                        continue

                    # All incoming guard edges must set exactly one variable and it must
                    # be the same for all of them.
                    itvars = set()
                    for iedge in in_edges:
                        if len(iedge.data.assignments) > 0:
                            if not itvars:
                                itvars = set(iedge.data.assignments.keys())
                            else:
                                itvars &= set(iedge.data.assignments.keys())
                        else:
                            itvars = None
                            break
                    if not itvars or len(itvars) > 1:
                        continue
                    itvar = next(iter(itvars))
                    itvarsym = pystr_to_symbolic(itvar)

                    # The outgoing edges must be negations of one another.
                    if out_edges[0].data.condition_sympy() != (sympy.Not(out_edges[1].data.condition_sympy())):
                        continue

                    # Make sure the last state of the loop (i.e. the state leading back
                    # to the guard via 'increment' edge) is part of this cycle. If not,
                    # we're looking at the guard for a nested cycle, which we ignore for
                    # this cycle.
                    increment_edge = None
                    for iedge in in_edges:
                        if itvarsym in pystr_to_symbolic(iedge.data.assignments[itvar]).free_symbols:
                            increment_edge = iedge
                            break
                    if increment_edge is None:
                        continue
                    if increment_edge.src not in cycle:
                        continue

                    # One of the child states must be in the loop (loop begin), and the
                    # other one must be outside the cycle (loop exit).
                    loop_state = None
                    exit_state = None
                    if out_edges[0].dst in cycle and out_edges[1].dst not in cycle:
                        loop_state = out_edges[0].dst
                        exit_state = out_edges[1].dst
                    elif out_edges[1].dst in cycle and out_edges[0].dst not in cycle:
                        loop_state = out_edges[1].dst
                        exit_state = out_edges[0].dst
                    if loop_state is None or exit_state is None:
                        continue

                    # This is a valid guard state candidate.
                    guard = v
                    begin = loop_state
                    break

                if guard is not None and begin is not None and itvar is not None:
                    # A guard state was identified, see if it has valid for-loop ranges
                    # and annotate the loop as such.

                    # Ensure that this guard's loop wasn't annotated yet.



                    loop_state_list = []
                    res = find_for_loop(sdfg, guard, begin, itervar=itvar)
                    if res is None:
                        # No range detected, mark as unbounded.
                        unannotated_cycle_states.extend(cycle)
                    else:
                        itervar, rng, (start_states, last_loop_state) = res

                        # Make sure the range is flipped in a direction such that the
                        # stride is positive (in order to match subsets.Range).
                        start, stop, stride = rng
                        # This inequality needs to be checked exactly like this due to
                        # constraints in sympy/symbolic expressions, do not simplify!!!
                        if (stride < 0) == True:
                            rng = (stop, start, -stride)

                        loop_states = sdutils.dfs_conditional(sdfg, sources=[begin], condition=lambda _, child: child != guard)
                        
                        if itvar not in begin.ranges:
                            # only do this if loop has not been annotated yet
                            for v in loop_states:
                                v.ranges[itervar] = subsets.Range([rng])
                                loop_state_list.append(v)
                            guard.ranges[itervar] = subsets.Range([rng])
                            guard.condition_edge = sdfg.edges_between(guard, begin)[0]
                            guard.is_loop_guard = True
                            guard.itvar = itervar
                            identified_loops[guard] = (begin, last_loop_state, loop_state_list, itvar, subsets.Range([rng]))

                else:
                    # There's no guard state, so this cycle marks all states in it as
                    # dynamically unbounded.
                    unannotated_cycle_states.extend(cycle)

            return identified_loops


    def propagate_states(self, sdfg) -> None:
        """
        Annotate the states of an SDFG with the number of executions.

        Algorithm:
        
            1. Clean up the state machine by splitting condition and assignment edges
            into separate edes with a dummy state in between.
            2. Detect and annotate any for-loop constructs with their corresponding loop
            variable ranges.
            3. Start traversing the state machine from the start state (start state
            gets executed once by default). At every state, check the following:

                a. The state was already visited -> in this case it can either be the
                guard of a loop we're returning to - in which case the number of
                executions is additively combined - or it is a state that can be
                reached through multiple paths (e.g. if/else branches), in which case
                the number of executions is equal to the maximum number of executions
                for each incoming path (in case this fully merges a previously
                branched out tree again, the number of executions isn't dynamic
                anymore). In both cases we override the calculated number of
                executions if we're propagating dynamic unbounded. This DFS traversal
                is complete and we continue with the next unvisited state.
                b. We're propagating dynamic unbounded -> this overrides every
                calculated number of executions, so this gets unconditionally
                propagated to all child states.
                c. None of the above, the next regular traversal step is executed:

                    1. If there is no further outgoing edge, this DFS traversal is done and we continue with the next
                    unvisited state.
                    2. If there is one outgoing edge, we continue propagating the
                    same number of executions to the child state. If the transition
                    to the child state is conditional, the current state might be
                    an implicit exit state, in which case we mark the next state as
                    dynamic to signal that it's an upper bound.
                    3. If there is more than one outgoing edge we:

                        a. Check if it's an annotated loop guard with a range. If
                        so, we calculate the number of executions for the loop
                        and propagate this down the loop.
                        b. Check if it's a loop that hasn't been unannotated, which
                        means it's unbounded. In this case we propagate dynamic
                        unbounded down the loop.
                        c. Otherwise this must be a conditional branch, so this
                        state's number of executions is given to all child states
                        as an upper bound.

            4. The traversal ends when all reachable states have been visited at least
            once.

        :param sdfg: The SDFG to annotate.
        :note: This operates on the SDFG in-place.
        """

        # We import here to avoid cyclic imports.
        from dace.sdfg import InterstateEdge
        from dace.transformation.helpers import split_interstate_edges
        from dace.sdfg.analysis import cfg

        # Reset the state edge annotations (which may have changed due to transformations)
        self.reset_state_annotations(sdfg)

        # Clean up the state machine by separating combined condition and assignment
        # edges.
        # TODO: this modifies the sdfg. Find a solution s.t. it leaves the sdfg unmodified
        split_interstate_edges(sdfg)

        # To enable branch annotation, we add a temporary exit state that connects
        # to all child-less states. With this, we can use the dominance frontier
        # to determine a full-merge state for branches.
        temp_exit_state = None
        for s in sdfg.nodes():
            if sdfg.out_degree(s) == 0:
                if temp_exit_state is None:
                    temp_exit_state = sdfg.add_state('__dace_brannotate_exit')
                sdfg.add_edge(s, temp_exit_state, InterstateEdge())

        dom_frontier = cfg.acyclic_dominance_frontier(sdfg)

        # Find any valid for loop constructs and annotate the loop ranges. Any other
        # cycle should be marked as unannotated.
        unannotated_cycle_states = []
        self._annotate_loop_ranges(sdfg, unannotated_cycle_states)

        # Keep track of states that fully merge a previous conditional split. We do
        # this so we can remove the dynamic executions flag for those states.
        full_merge_states = set()

        visited_states = set()

        traversal_q = deque()
        traversal_q.append((sdfg.start_state, 1, False, []))
        while traversal_q:
            (state, proposed_executions, proposed_dynamic, itvar_stack) = traversal_q.pop()

            out_degree = sdfg.out_degree(state)
            out_edges = sdfg.out_edges(state)

            # Check if the traversal reached a state that's already been visited
            # (ends traversal), or if the number of executions being propagated is
            # dynamic unbounded. Otherwise, continue regular traversal.
            if state in visited_states:
                # This state has already been visited.
                if proposed_executions == 0 and proposed_dynamic:
                    state.executions = proposed_executions
                    state.dynamic_executions = proposed_dynamic
                elif getattr(state, 'is_loop_guard', False):
                    # If we encounter a loop guard that's already been visited,
                    # we've finished traversing a loop and can remove that loop's
                    # iteration variable from the stack. We additively merge the
                    # number of executions.
                    if not (state.executions == 0 and state.dynamic_executions):
                        state.executions += proposed_executions
                else:
                    # If we have already visited this state, but it is NOT a loop
                    # guard, this means that we can reach this state via multiple
                    # different paths. If so, the number of executions for this
                    # state is given by the maximum number of executions among each
                    # of the paths reaching it. If the state additionally completely
                    # merges a previously branched out state tree, we know that the
                    # number of executions isn't dynamic anymore.
                    # The only exception to this rule: If the state is in an
                    # unannotated loop, i.e. should be annotated as dynamic
                    # unbounded instead, we do that.
                    if (state in unannotated_cycle_states):
                        state.executions = 0
                        state.dynamic_executions = True
                    else:
                        state.executions = sympy.Max(state.executions, proposed_executions).doit()
                        if state in full_merge_states:
                            state.dynamic_executions = False
                        else:
                            state.dynamic_executions = (state.dynamic_executions or proposed_dynamic)
            elif proposed_dynamic and proposed_executions == 0:
                # We're propagating a dynamic unbounded number of executions, which
                # always gets propagated unconditionally. Propagate to all children.
                visited_states.add(state)
                state.executions = proposed_executions
                state.dynamic_executions = proposed_dynamic
                # This gets pushed through to all children unconditionally.
                if len(out_edges) > 0:
                    for oedge in out_edges:
                        traversal_q.append((oedge.dst, proposed_executions, proposed_dynamic, itvar_stack))
            else:
                # If the state hasn't been visited yet and we're not propagating a
                # dynamic unbounded number of executions, we calculate the number of
                # executions for the next state(s) and continue propagating.
                visited_states.add(state)
                if state in full_merge_states:
                    # If this state fully merges a conditional branch, this turns
                    # dynamic executions back off.
                    proposed_dynamic = False
                state.executions = proposed_executions
                state.dynamic_executions = proposed_dynamic

                if out_degree == 1:
                    # Continue with the only child state.
                    if not out_edges[0].data.is_unconditional():
                        # If the transition to the child state is based on a
                        # condition, this state could be an implicit exit state. The
                        # child state's number of executions is thus only given as
                        # an upper bound and marked as dynamic.
                        proposed_dynamic = True
                    traversal_q.append((out_edges[0].dst, proposed_executions, proposed_dynamic, itvar_stack))
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
                        loop_executions = ceiling(((stop + 1) - start) / stride)
                        for outer_itvar_string in reversed(itvar_stack):
                            outer_range = state.ranges[outer_itvar_string]
                            outer_start = outer_range[0][0]
                            outer_stop = outer_range[0][1]
                            outer_stride = outer_range[0][2]
                            outer_itvar = symbolic.pystr_to_symbolic(outer_itvar_string)
                            exec_repl = loop_executions.subs({outer_itvar: (outer_itvar * outer_stride + outer_start)})
                            loop_executions = Sum(exec_repl,
                                                (outer_itvar, 0, ceiling((outer_stop - outer_start) / outer_stride)))
                        loop_executions = loop_executions.doit()

                        loop_state = state.condition_edge.dst
                        end_state = (out_edges[0].dst if out_edges[1].dst == loop_state else out_edges[1].dst)

                        traversal_q.append((end_state, state.executions, proposed_dynamic, itvar_stack))
                        traversal_q.append((loop_state, loop_executions, proposed_dynamic, itvar_stack + [state.itvar]))
                    else:
                        # Conditional split or unannotated (dynamic unbounded) loop.
                        unannotated_loop_edge = None
                        for oedge in out_edges:
                            if oedge.dst in unannotated_cycle_states:
                                # This is an unannotated loop down this branch.
                                unannotated_loop_edge = oedge

                        if unannotated_loop_edge is not None:
                            # Traverse as an unbounded loop.
                            out_edges.remove(unannotated_loop_edge)
                            for oedge in out_edges:
                                traversal_q.append((oedge.dst, state.executions, False, itvar_stack))
                            traversal_q.append((unannotated_loop_edge.dst, 0, True, itvar_stack))
                        else:
                            # Traverse as a conditional split.
                            proposed_executions = state.executions
                            proposed_dynamic = True

                            # Get the dominance frontier for each child state and
                            # merge them into one common frontier, representing the
                            # branch's immediate post-dominator. If a state has no
                            # dominance frontier, add the state itself to the
                            # frontier. This takes care of the case where a branch
                            # is fully merged, but one branch contains no states.
                            common_frontier = set()
                            for oedge in out_edges:
                                frontier = dom_frontier[oedge.dst]
                                if not frontier:
                                    frontier = {oedge.dst}
                                common_frontier |= frontier

                                # Continue traversal for each child.
                                traversal_q.append((oedge.dst, proposed_executions, proposed_dynamic, itvar_stack))

                            # If the whole branch is not dynamic, and the
                            # common frontier is exactly one state, we know that
                            # the branch merges again at that state.
                            if not state.dynamic_executions and len(common_frontier) == 1:
                                full_merge_states.add(list(common_frontier)[0])

        # If we had to create a temporary exit state, we remove it again here.
        if temp_exit_state is not None:
            sdfg.remove_node(temp_exit_state)


    def propagate_memlets_nested_sdfg(self, parent_sdfg, parent_state, nsdfg_node):
        """
        Propagate memlets out of a nested sdfg.

        :param parent_sdfg: The parent SDFG this nested SDFG is in.
        :param parent_state: The state containing this nested SDFG.
        :param nsdfg_node: The NSDFG node containing this nested SDFG.
        :note: This operates in-place on the parent SDFG.
        """
        # We import late to avoid cyclic imports here.
        from dace.transformation.helpers import unsqueeze_memlet

        # Build a map of connectors to associated 'border' memlets inside
        # the nested SDFG. This map will be populated with memlets once they
        # get propagated in the SDFG.
        border_memlets = {
            'in': {},
            'out': {},
        }
        for connector in nsdfg_node.in_connectors:
            border_memlets['in'][connector] = None
        for connector in nsdfg_node.out_connectors:
            border_memlets['out'][connector] = None

        sdfg: dace.SDFG = nsdfg_node.sdfg
        outer_symbols = parent_state.symbols_defined_at(nsdfg_node)
        # TODO: change this later to merge branches with full writes whose path conditions form a tautology 

        # add dummy state as a sink
        dummy_sink = sdfg.add_state("dummy_state")
        for sink_node in sdfg.sink_nodes():
            if not sink_node is dummy_sink:
                sdfg.add_edge(sink_node, dummy_sink, dace.sdfg.InterstateEdge())
        
        # get all the nodes that are executed unconditionally in the cfg a.k.a nodes that dominate the sink states
        dominators = cfg.all_dominators(sdfg)
        states = dominators[dummy_sink]

        # remove dummy state 
        sdfg.remove_node(dummy_sink)
        


        # For each state, go through all access nodes corresponding to any in- or
        # out-connectors to and from this SDFG. Given those access nodes, collect
        # the corresponding memlets and use them to calculate the memlet volume and
        # subset corresponding to the outside memlet attached to that connector.
        # This is passed out via `border_memlets` and propagated along from there.
        for state in states:
            for node in state.data_nodes():
                for direction in border_memlets:
                    if (node.label not in border_memlets[direction]):
                        continue

                    memlet = border_memlets[direction][node.label]

                    # Collect the edges to/from this access node, depending on the
                    # direction the connector leads in.
                    edges = []
                    if direction == 'in':
                        edges = state.out_edges(node)
                    elif direction == 'out':
                        edges = state.in_edges(node)

                    # Collect all memlets belonging to this access node, and
                    # accumulate the total volume between them.
                    memlets = []
                    for edge in edges:
                        inside_memlet = approximation_dict[edge]
                        memlets.append(inside_memlet)

                        if memlet is None:
                            # Use the first encountered memlet as a 'border' memlet
                            # and accumulate the sum on it.
                            memlet = Memlet(data=inside_memlet.data, volume=0)
                            memlet._is_data_src = True
                            border_memlets[direction][node.label] = memlet

                    # Given all of this access nodes' memlets, propagate the subset
                    # according to the state's variable ranges.
                    if len(memlets) > 0:
                        params = []
                        ranges = []
                        for symbol in state.ranges:
                            params.append(symbol)
                            ranges.append(state.ranges[symbol][0])

                        if len(params) == 0 or len(ranges) == 0:
                            params = ['__dace_dummy']
                            ranges = [(0, 0, 1)]

                        # Propagate the subset based on the direction this memlet is
                        # pointing. If we're accessing from an incoming connector,
                        # propagate the source subset, if we're going to an outgoing
                        # connector, propagate the destination subset.
                        use_dst = False
                        if direction == 'out':
                            use_dst = True
                        array = sdfg.arrays[node.label]
                        subset = self.propagate_subset(memlets, array, params, subsets.Range(ranges), use_dst=use_dst).subset
                        if not subset:
                            continue
                        # If the border memlet already has a set range, compute the
                        # union of the ranges to merge the subsets.
                        if memlet.subset is not None:
                            if memlet.subset.dims() != subset.dims():
                                raise ValueError('Cannot merge subset ranges of unequal dimension!')
                            else:
                                memlet.subset = subsets.list_union(memlet.subset, subset)
                        else:
                            memlet.subset = subset

            if state in loop_write_dict.keys():
                for node_label, loop_memlet in loop_write_dict[state].items():
                    if (node_label not in border_memlets["out"]):
                        continue
                    memlet = border_memlets["out"][node_label]

                    if memlet is None:
                        # Use the first encountered memlet as a 'border' memlet
                        # and accumulate the sum on it.
                        memlet = Memlet(data=loop_memlet.data, volume=0)
                        memlet._is_data_src = True
                        border_memlets["out"][node_label] = memlet

                    params = []
                    ranges = []
                    for symbol in state.ranges:
                        params.append(symbol)
                        ranges.append(state.ranges[symbol][0])

                    if len(params) == 0 or len(ranges) == 0:
                        params = ['__dace_dummy']
                        ranges = [(0, 0, 1)]

                    # Propagate the subset based on the direction this memlet is
                    # pointing. If we're accessing from an incoming connector,
                    # propagate the source subset, if we're going to an outgoing
                    # connector, propagate the destination subset.
                    use_dst = True
                    array = sdfg.arrays[node_label]
                    subset = self.propagate_subset([loop_memlet], array, params, subsets.Range(ranges), use_dst=use_dst).subset

                    # If the border memlet already has a set range, compute the
                    # union of the ranges to merge the subsets.
                    if memlet.subset is not None:
                        if memlet.subset.dims() != subset.dims():
                            raise ValueError('Cannot merge subset ranges of unequal dimension!')
                        else:
                            memlet.subset = subsets.list_union(memlet.subset, subset)
                    else:
                        memlet.subset = subset
                    



        # Make sure any potential NSDFG symbol mapping is correctly reversed
        # when propagating out.
        for direction in border_memlets:
            for connector in border_memlets[direction]:
                border_memlet = border_memlets[direction][connector]
                if border_memlet is not None:
                    border_memlet.replace(nsdfg_node.symbol_mapping)

                    # Also make sure that there's no symbol in the border memlet's
                    # range that only exists inside the nested SDFG. If that's the
                    # case, use an empty set to stay correct.

                    if border_memlet.src_subset is not None:
                        if isinstance(border_memlet.src_subset, subsets.Subsetlist):
                            _subsets = border_memlet.src_subset.subset_list
                        else:
                            _subsets = [border_memlet.src_subset]
                        for i, subset in enumerate(_subsets):
                            for rng in subset:
                                fall_back = False
                                for item in rng:
                                    if any(str(s) not in outer_symbols.keys() for s in item.free_symbols):
                                        fall_back = True
                                        break
                                if fall_back:
                                    _subsets[i] = None
                                    break
                        border_memlet.src_subset = subsets.Subsetlist(_subsets)
                    if border_memlet.dst_subset is not None:
                        if isinstance(border_memlet.dst_subset, subsets.Subsetlist):
                            _subsets = border_memlet.dst_subset.subset_list
                        else:
                            _subsets = [border_memlets.dst_subset]
                        for i, subset in enumerate(_subsets):
                            for rng in subset:
                                fall_back = False
                                for item in rng:
                                    if any(str(s) not in outer_symbols.keys() for s in item.free_symbols):
                                        fall_back = True
                                        break
                                if fall_back:
                                    _subsets[i] = None
                                    break
                        border_memlet.dst_subset = subsets.Subsetlist(_subsets)

        # TODO: Make sure that this makes sense, especially the part in the try clause
        # TODO: if oedge has no corresponding border memlet assign empty memlet to it. 
        # TODO: Verify: Is this correct? What if there is nestedSDFG <--> nestedSDFG


        # Propagate the inside 'border' memlets outside the SDFG by
        # offsetting, and unsqueezing if necessary.
        for edge in parent_state.in_edges(nsdfg_node):
            in_memlet = approximation_dict[edge]
            if edge.dst_conn in border_memlets['in']:
                internal_memlet = border_memlets['in'][edge.dst_conn]
                # iterate over all the subset in the Subsetlist of internal
                if internal_memlet is None:
                    in_memlet.subset = None
                    in_memlet.src_subset = None
                    approximation_dict[edge] = in_memlet
                    continue

                # handle subsets that aren't subsetlists yet
                if isinstance(internal_memlet.subset, subsets.Subsetlist):
                    _subsets = internal_memlet.subset.subset_list
                else:
                    _subsets = [internal_memlet.subset]

                if isinstance(in_memlet.subset, subsets.Subsetlist):
                    in_memlet.subset = in_memlet.subset.subset_list[0]
                if isinstance(in_memlet.dst_subset, subsets.Subsetlist):
                    in_memlet.dst_subset = in_memlet.dst_subset.subset_list[0]
                if isinstance(in_memlet.src_subset, subsets.Subsetlist):
                    in_memlet.src_subset = in_memlet.src_subset.subset_list[0]
                
                tmp_memlet = Memlet(
                    data = internal_memlet.data,
                    subset = internal_memlet.subset,
                    other_subset= internal_memlet.other_subset,
                    volume = internal_memlet.volume,
                    dynamic=internal_memlet.dynamic,
                    wcr=internal_memlet.wcr,
                    wcr_nonatomic=internal_memlet.wcr_nonatomic,
                    allow_oob=internal_memlet.allow_oob
                )

                for j, subset in enumerate(_subsets):
                    if subset is None:
                        continue
                    tmp_memlet.subset = subset
                    try:
                        unsqueezed_memlet = unsqueeze_memlet(tmp_memlet, in_memlet, False)
                        # If no appropriate memlet found, use array dimension
                        for i, (rng, s) in enumerate(zip(tmp_memlet.subset, parent_sdfg.arrays[unsqueezed_memlet.data].shape)):
                            if rng[1] + 1 == s:
                                unsqueezed_memlet.subset[i] = (unsqueezed_memlet.subset[i][0], s - 1, 1)
                            if symbolic.issymbolic(unsqueezed_memlet.volume):
                                if any(str(s) not in outer_symbols for s in unsqueezed_memlet.volume.free_symbols):
                                    unsqueezed_memlet.subset = None
                                    break
                        subset = unsqueezed_memlet.subset
                    except (ValueError, NotImplementedError):
                        # In any case of memlets that cannot be unsqueezed fall back to empty subset
                        subset = None
                    _subsets[j] = subset

                # if the unsqueezing failed for all the subsets of the border memlet return empty subset
                if all(s is None for s in _subsets):
                    in_memlet.subset = None
                    in_memlet.src_subset = None
                    approximation_dict[edge] = in_memlet
                else:
                    in_memlet = unsqueezed_memlet
                    in_memlet.subset = subsets.Subsetlist(_subsets)
                    approximation_dict[edge] = in_memlet
                
        for edge in parent_state.out_edges(nsdfg_node):
            out_memlet = approximation_dict[edge]
            if edge.src_conn in border_memlets['out']:
                internal_memlet = border_memlets['out'][edge.src_conn]

                if internal_memlet is None:
                    out_memlet.subset = None
                    out_memlet.dst_subset = None
                    approximation_dict[edge] = out_memlet
                    continue
                if isinstance(internal_memlet.subset, subsets.Subsetlist):
                    _subsets = internal_memlet.subset.subset_list
                else:
                    _subsets = [internal_memlet.subset]

                if isinstance(out_memlet.subset, subsets.Subsetlist):
                    out_memlet.subset = out_memlet.subset.subset_list[0]
                if isinstance(out_memlet.dst_subset, subsets.Subsetlist):
                    out_memlet.dst_subset = out_memlet.dst_subset.subset_list[0]
                if isinstance(out_memlet.src_subset, subsets.Subsetlist):
                    out_memlet.src_subset = out_memlet.src_subset.subset_list[0]
                    
                tmp_memlet = Memlet(
                    data = internal_memlet.data,
                    subset = internal_memlet.subset,
                    other_subset= internal_memlet.other_subset,
                    volume = internal_memlet.volume,
                    dynamic=internal_memlet.dynamic,
                    wcr=internal_memlet.wcr,
                    wcr_nonatomic=internal_memlet.wcr_nonatomic,
                    allow_oob=internal_memlet.allow_oob
                )

                for j, subset in enumerate(_subsets):
                    if subset is None:
                        continue
                    tmp_memlet.subset = subset
                    try:
                        unsqueezed_memlet = unsqueeze_memlet(tmp_memlet, out_memlet, False)
                        # If no appropriate memlet found, use array dimension
                        for i, (rng, s) in enumerate(zip(tmp_memlet.subset, parent_sdfg.arrays[unsqueezed_memlet.data].shape)):
                            if rng[1] + 1 == s:
                                unsqueezed_memlet.subset[i] = (unsqueezed_memlet.subset[i][0], s - 1, 1)
                            if symbolic.issymbolic(unsqueezed_memlet.volume):
                                if any(str(s) not in outer_symbols for s in unsqueezed_memlet.volume.free_symbols):
                                    unsqueezed_memlet.subset = None
                        subset = unsqueezed_memlet.subset
                    except (ValueError, NotImplementedError):
                        # In any case of memlets that cannot be unsqueezed (i.e.,
                        # reshapes), use dynamic unbounded memlets.
                        subset = None
                    _subsets[j] = subset

                if all(s is None for s in _subsets):
                    out_memlet.subset = None
                    out_memlet.dst_subset = None
                else:
                    out_memlet = unsqueezed_memlet
                    out_memlet.subset = subsets.Subsetlist(_subsets)
                    approximation_dict[edge] = out_memlet


    def reset_state_annotations(self, sdfg):
        """ Resets the state (loop-related) annotations of an SDFG.

            :note: This operation is shallow (does not go into nested SDFGs).
        """
        for state in sdfg.nodes():
            state.executions = 0
            state.dynamic_executions = True
            state.ranges = {}
            state.condition_edge = None
            state.is_loop_guard = False
            state.itervar = None


    def propagate_memlets_sdfg(self, sdfg):
        """ Propagates memlets throughout an entire given SDFG. 
        
            :note: This is an in-place operation on the SDFG.
        """
        # Reset previous annotations first
        #TODO: This is important for the loop analysis. Is just running this again OK?
        self.reset_state_annotations(sdfg)

        for state in sdfg.nodes():
            self.propagate_memlets_state(sdfg, state)


        loops = self._annotate_loop_ranges(sdfg,[])
        loop_dict.update(loops)

        
        self.propagate_memlet_loop(sdfg, loops)


        # TODO: Since we are not interested in the number of executions this is probably not necessary,
        # but the loop annotation is definitely helpful
        self.propagate_states(sdfg)

    def propagate_memlet_loop(self, sdfg, loops: dict, loopheader:SDFGState = None):
        # for each state in the loop body
            # propagate each memlet out of the loop and store it in a border memlet similar to a nested sdfg
            # maybe in a map that maps loop heads to border memlets
            # if the state is again a loop head nested in the current loop
                # call propagate_memlet_loop on the nested loop

        # difference to propagate_memlets_nested_sdfg is that there are no border memlets

        def filter_subsets(itvar: str, s_itvars: List[str], range: subsets.Range, memlet: Memlet):
            # if loop range is symbolic -> only propagate subsets that contain the iterator as a symbol
            # if loop range is constant (and not empty) only propagate subsets that have iterator variables of current or surrounding loops in their definition   
            
            if memlet.subset is None:
                return None
              
            if(range.free_symbols):
                if isinstance(memlet.subset, subsets.Subsetlist):
                    filtered_subsets = [s for s in memlet.subset.subset_list if itvar in s.free_symbols]
                else:
                    filtered_subsets = [s for s in [memlet.subset] if itvar in s.free_symbols]

            # range is constant
            else:
                if isinstance(memlet.subset, subsets.Subsetlist):
                    filtered_subsets = [s for s in memlet.subset.subset_list if (s_itvar in s.free_symbols for s_itvar in s_itvars )]
                else:
                    filtered_subsets = [s for s in [memlet.subset] if (s_itvar in s.free_symbols for s_itvar in s_itvars )]

            return filtered_subsets
        
        if not loops:
            return
        
        # No loopheader was passed as an argument so we find the outermost loops
        if loopheader == None:
            top_loopheaders = []
            for current_loop_header, current_loop in loops.items():
                for other_loop_header, other_loop in loops.items():
                    if other_loop_header is current_loop_header:
                        continue
                    _, _, other_loop_states, _, _ = other_loop
                    if current_loop_header in other_loop_states:
                        break
                else:
                    top_loopheaders.append(current_loop_header)

            if not top_loopheaders:
                return
            
            for loopheader in top_loopheaders:
                self.propagate_memlet_loop(sdfg, loops, loopheader)

            return
        

        current_loop = loops[loopheader]
        begin, last_loop_state, loop_states, itvar, rng = current_loop

        

        if rng.num_elements() == 0:
            return
        
        dominators = cfg.all_dominators(sdfg)

        if any(begin not in dominators[s] and not begin is s for s in loop_states):
            return
        
        border_memlets = defaultdict(None)
        ignore = []

        # TODO: change this later for more precision
        # get all the nodes that are executed unconditionally in the cfg a.k.a nodes that dominate the sink states
        states = dominators[last_loop_state].intersection(set(loop_states))
        states.add(last_loop_state)

        for state in states:
            if state in ignore:
                continue

            # recursively propagate nested loops and ignore the nested states
            if state in loops.keys():
                self.propagate_memlet_loop(sdfg, loops, state)
                _, _, nested_loop_states, _, _ = loops[state]
                ignore += nested_loop_states

            # iterate over the data_nodes that are actually in the current state
            # plus the data_nodes that are overwritten in the corresponding loop body
            # if the state is a loop header
            surrounding_itvars = state.ranges.keys()
            # iterate over acccessnodes in the state
            for node in state.data_nodes():
                # no writes associated with this access node
                if state.in_degree(node) == 0:
                    continue

                edges = state.in_edges(node)
                memlet = border_memlets.get(node.label)
                memlets = []

                # collect all the subsets of the incoming memlets for the current access node
                for edge in edges:
                    inside_memlet = copy.copy(approximation_dict[edge])

                    filtered_subsets = filter_subsets(itvar, surrounding_itvars, rng, inside_memlet)
                    if not filtered_subsets:
                        continue

                    inside_memlet.subset = subsets.Subsetlist(filtered_subsets)
                    memlets.append(inside_memlet)
                    if memlet is None:
                        # Use the first encountered memlet as a 'border' memlet
                        # and accumulate the sum on it.
                        memlet = Memlet(data=inside_memlet.data, volume=0)
                        memlet._is_data_src = True
                        border_memlets[node.label] = memlet
                    
                self.propagate_loop_subset(sdfg, memlets, memlet, sdfg.arrays[node.label], itvar, rng)

            if state not in loop_write_dict.keys():
                continue
            # iterate over the accessnodes in the loopheader dict
            for node_label, border_memlet in loop_write_dict[state].items():
                filtered_subsets = filter_subsets(itvar, surrounding_itvars, rng, border_memlet)
                if not filtered_subsets:
                    continue
                
                border_memlet.subset = subsets.Subsetlist(filtered_subsets)
                memlet = border_memlets.get(node_label)
                if memlet is None:
                    # Use the first encountered memlet as a 'border' memlet
                    # and accumulate the sum on it.
                    memlet = Memlet(data=border_memlet.data, volume=0)
                    memlet._is_data_src = True
                    border_memlets[node_label] = memlet

                self.propagate_loop_subset(sdfg, [border_memlet], memlet, sdfg.arrays[node_label], itvar, rng)
        
        loop_write_dict[loopheader] = border_memlets


    def propagate_loop_subset(self, sdfg: dace.SDFG, memlets: List[Memlet], dst_memlet: Memlet, arr:dace.data.Array, itvar:str, rng:subsets.Subset ):
        if len(memlets) > 0:
            params = [itvar]
            ranges = [rng]

            # TODO: Is this even necessary. Can't hurt i guess
            if len(params) == 0 or len(ranges) == 0:
                params = ['__dace_dummy']
                ranges = [(0, 0, 1)]

            use_dst = True
            subset = self.propagate_subset(memlets, arr, params, rng, use_dst=use_dst).subset

            # If the border memlet already has a set range, compute the
            # union of the ranges to merge the subsets.
            if dst_memlet.subset is not None:
                if dst_memlet.subset.dims() != subset.dims():
                    raise ValueError('Cannot merge subset ranges of unequal dimension!')
                else:
                    dst_memlet.subset = subsets.list_union(dst_memlet.subset, subset)
            else:
                dst_memlet.subset = subset

    def propagate_memlets_state(self, sdfg, state):
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

                # Propagate memlets inside the nested SDFG.
                self.propagate_memlets_sdfg(node.sdfg)

                # Propagate memlets out of the nested SDFG.
                self.propagate_memlets_nested_sdfg(sdfg, state, node)

        # Process scopes from the leaves upwards
        self.propagate_memlets_scope(sdfg, state, state.scope_leaves())


    def propagate_memlets_scope(self, sdfg, state, scopes, propagate_entry=True, propagate_exit=True):
        """ 
        Propagate memlets from the given scopes outwards. 

        :param sdfg: The SDFG in which the scopes reside.
        :param state: The SDFG state in which the scopes reside.
        :param scopes: The ScopeTree object or a list thereof to start from.
        :param propagate_entry: If False, skips propagating out of the scope entry node.
        :param propagate_exit: If False, skips propagating out of the scope exit node.
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

                # TODO: Maybe only propagate in one direction since we are only interested in writes
                # Propagate out of entry
                if propagate_entry:
                    self._propagate_node(state, scope.entry)

                # Propagate out of exit
                if propagate_exit:
                    self._propagate_node(state, scope.exit)

                # Add parent to next frontier
                next_scopes.add(scope.parent)
            scopes_to_process = next_scopes
            next_scopes = set()


    def _propagate_node(self, dfg_state, node):
        if isinstance(node, nodes.EntryNode):
            internal_edges = [e for e in dfg_state.out_edges(node) if e.src_conn and e.src_conn.startswith('OUT_')]
            external_edges = [e for e in dfg_state.in_edges(node) if e.dst_conn and e.dst_conn.startswith('IN_')]
            geticonn = lambda e: e.src_conn[4:]
            geteconn = lambda e: e.dst_conn[3:]
            use_dst = False
        else:
            internal_edges = [e for e in dfg_state.in_edges(node) if e.dst_conn and e.dst_conn.startswith('IN_')]
            external_edges = [e for e in dfg_state.out_edges(node) if e.src_conn and e.src_conn.startswith('OUT_')]
            geticonn = lambda e: e.dst_conn[3:]
            geteconn = lambda e: e.src_conn[4:]
            use_dst = True

        # TODO: generalize this for subsetLists (?)
        for edge in external_edges:
            if approximation_dict[edge].is_empty():
                new_memlet = Memlet()
            else:
                internal_edge = next(e for e in internal_edges if geticonn(e) == geteconn(edge))
                aligned_memlet = self.align_memlet(dfg_state, internal_edge, dst=use_dst)
                new_memlet = self.propagate_memlet(dfg_state, aligned_memlet, node, True, connector=geteconn(edge))
            approximation_dict[edge] = new_memlet


    def align_memlet(self, state, e: gr.MultiConnectorEdge[Memlet], dst: bool) -> Memlet:
        """Takes Multiconnectoredge in DFG """

        is_src = e.data._is_data_src
        # Memlet is already aligned
        if is_src is None or (is_src and not dst) or (not is_src and dst):
            res = approximation_dict[e]
            return res

        # Data<->Code memlets always have one data container
        mpath = state.memlet_path(e)
        if not isinstance(mpath[0].src, nodes.AccessNode) or not isinstance(mpath[-1].dst, nodes.AccessNode):
            return approximation_dict[e]

        # Otherwise, find other data container
        result = copy.deepcopy(approximation_dict[e])
        if dst:
            node = mpath[-1].dst
        else:
            node = mpath[0].src

        # Fix memlet fields
        result.data = node.data
        result.subset = approximation_dict[e].other_subset
        result.other_subset = approximation_dict[e].subset
        result._is_data_src = not is_src
        return result


    # External API
    def propagate_memlet(self, dfg_state,
                        memlet: Memlet,
                        scope_node: nodes.EntryNode,
                        union_inner_edges: bool,
                        arr=None,
                        connector=None):
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
        use_dst = False
        if isinstance(scope_node, nodes.EntryNode):
            use_dst = False
            entry_node = scope_node
            neighboring_edges = dfg_state.out_edges(scope_node)
            if connector is not None:
                neighboring_edges = [e for e in neighboring_edges if e.src_conn and e.src_conn[4:] == connector]
        elif isinstance(scope_node, nodes.ExitNode):
            use_dst = True
            entry_node = dfg_state.entry_node(scope_node)
            neighboring_edges = dfg_state.in_edges(scope_node)
            if connector is not None:
                neighboring_edges = [e for e in neighboring_edges if e.dst_conn and e.dst_conn[3:] == connector]
        else:
            raise TypeError('Trying to propagate through a non-scope node')
        if memlet.is_empty():
            return Memlet()

        sdfg = dfg_state.parent
        scope_node_symbols = set(conn for conn in entry_node.in_connectors if not conn.startswith('IN_'))
        defined_vars = [
            symbolic.pystr_to_symbolic(s) for s in (dfg_state.symbols_defined_at(entry_node).keys()
                                                    | sdfg.constants.keys()) if s not in scope_node_symbols
        ]

        # Find other adjacent edges within the connected to the scope node
        # and union their subsets
        if union_inner_edges:
            aggdata = [approximation_dict[e]for e in neighboring_edges if approximation_dict[e].data == memlet.data and approximation_dict[e] != memlet]
        else:
            aggdata = []

        aggdata.append(memlet)

        if arr is None:
            if memlet.data not in sdfg.arrays:
                raise KeyError('Data descriptor (Array, Stream) "%s" not defined in SDFG.' % memlet.data)

            # FIXME: A memlet alone (without an edge) cannot figure out whether it is data<->data or data<->code
            #        so this test cannot be used
            # If the data container is not specified on the memlet, use other data
            # if memlet._is_data_src is not None:
            #     if use_dst and memlet._is_data_src:
            #         raise ValueError('Cannot propagate memlet - source data container given but destination is necessary')
            #     elif not use_dst and not memlet._is_data_src:
            #         raise ValueError('Cannot propagate memlet - destination data container given but source is necessary')

            arr = sdfg.arrays[memlet.data]

        # Propagate subset
        if isinstance(entry_node, nodes.MapEntry):
            mapnode = entry_node.map
            return self.propagate_subset(aggdata, arr, mapnode.params, mapnode.range, defined_vars, use_dst=use_dst)

        elif isinstance(entry_node, nodes.ConsumeEntry):
            # Nothing to analyze/propagate in consume
            new_memlet = copy.copy(memlet)
            new_memlet.subset = subsets.Range.from_array(arr)
            new_memlet.other_subset = None
            new_memlet.volume = 0
            new_memlet.dynamic = True
            return new_memlet
        else:
            raise NotImplementedError('Unimplemented primitive: %s' % type(entry_node))


    # External API
    def propagate_subset(self, memlets: List[Memlet],
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
            :param use_dst: Whether to propagate the memlets' dst subset or use the
                            src instead, depending on propagation direction.
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
            defined_variables = set(symbolic.pystr_to_symbolic(p) for p in defined_variables)

        # Propagate subset
        variable_context = [defined_variables, [symbolic.pystr_to_symbolic(p) for p in params]]

        new_subset = None
        # TODO: For each memlet propagate each Subset in the Subsetlist instead of only one subset
        for md in memlets:
            if md.is_empty():
                continue

            _subsets = None
            if use_dst and md.dst_subset is not None:
                _subsets = copy.deepcopy(md.dst_subset)
            elif not use_dst and md.src_subset is not None:
                _subsets = copy.deepcopy(md.src_subset)
            else:
                _subsets = copy.deepcopy(md.subset)
            
            if isinstance(_subsets, subsets.Subsetlist):
                _subsets = _subsets.subset_list
            else:
                _subsets = [_subsets]
            
            if len(list(set(_subsets) - set([None]))) == 0 or _subsets == None:
                break

            # iterate over all the subsets in the Subsetlist of the current memlet and
            # try to apply a memletpattern. If no pattern matches fall back to the empty set
            for i, subset in enumerate(_subsets):
                # find a pattern for the current subset
                for pclass in MemletPattern.extensions():
                    pattern = pclass()
                    if pattern.can_be_applied([subset], variable_context, rng, [md]):
                        subset = pattern.propagate(arr, [subset], rng)
                        break
                else:
                    # No patterns found. Underapproximate the subset with an empty subset (so None)
                    warnings.warn('Cannot find appropriate memlet pattern to '
                                'propagate %s through %s' % (str(subset), str(rng)))
                    subset = None
                # FIXME: this overwrites the subset in the original edge. Make a deepcopy
                _subsets[i] = subset
    
            # Union edges as necessary
            if new_subset is None:
                new_subset = subsets.Subsetlist(_subsets)
            else:
                old_subset = new_subset
                new_subset = subsets.list_union(new_subset, subsets.Subsetlist(_subsets))
                if new_subset is None:
                    warnings.warn('Subset union failed between %s and %s ' % (old_subset, _subsets))
                    break


        # Create new memlet
        new_memlet = copy.copy(memlets[0])
        new_memlet.subset = new_subset
        new_memlet.other_subset = None

        # Propagate volume:
        # We do not care about the volume and if it is dynamic
        # Number of accesses in the propagated memlet is the sum of the internal
        # number of accesses times the size of the map range set (unbounded dynamic)
        new_memlet.volume = 0
        
        
        # if any(m.dynamic for m in memlets):
        #     new_memlet.dynamic = True
        # elif symbolic.issymbolic(new_memlet.volume) and any(s not in defined_variables
        #                                                     for s in new_memlet.volume.free_symbols):
        #     new_memlet.dynamic = True
        #     new_memlet.volume = 0

        return new_memlet


    def _freesyms(self, expr):
        """ 
        Helper function that either returns free symbols for sympy expressions
        or an empty set if constant.
        """
        if isinstance(expr, sympy.Basic):
            return expr.free_symbols
        return {}
