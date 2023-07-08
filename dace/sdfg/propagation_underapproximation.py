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
loop_dict: dict[SDFGState, (SDFGState, SDFGState,
                            list[SDFGState], str, subsets.Range)] = {}
iteration_variables: dict[dace.SDFG, set[str]] = {}


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

        # get iteration variables that should be propagated
        params = variable_context[-1]
        # get other iteration variables that should not be propagated
        other_params = variable_context[-3]

        # Return False if iteration variable appears in multiple dimensions 
        # or if two iteration variables appear in the same dimension
        for expr in expressions:
            for param in params:
                occured_before = False
                for dim in range(data_dims):
                    # collect free_symbols in current dimension
                    free_symbols = []
                    curr_dim_expr = expr[dim]
                    if isinstance(curr_dim_expr, symbolic.SymExpr):
                        free_symbols += curr_dim_expr.expr.free_symbols
                    elif isinstance(curr_dim_expr, tuple):
                        free_symbols += curr_dim_expr[0].expr.free_symbols if isinstance(
                            curr_dim_expr[0], symbolic.SymExpr) else list(pystr_to_symbolic(curr_dim_expr[0]).expand().free_symbols)
                        free_symbols += curr_dim_expr[1].expr.free_symbols if isinstance(
                            curr_dim_expr[1], symbolic.SymExpr) else list(pystr_to_symbolic(curr_dim_expr[1]).expand().free_symbols)
                        free_symbols += curr_dim_expr[2].expr.free_symbols if isinstance(
                            curr_dim_expr[2], symbolic.SymExpr) else list(pystr_to_symbolic(curr_dim_expr[2]).expand().free_symbols)
                    else:
                        free_symbols += [curr_dim_expr]

                    if param in free_symbols:
                        if occured_before:
                            return False
                        occured_before = True
                
                    for other_param in set(params) | set(other_params):
                        if other_param is param:
                            continue
                        if other_param in free_symbols and param in free_symbols:
                            return False


        overapprox_range = subsets.Range([(rb.expr if isinstance(rb, symbolic.SymExpr) else rb,
                                           re.expr if isinstance(
                                               re, symbolic.SymExpr) else re,
                                           rs.expr if isinstance(rs, symbolic.SymExpr) else rs)
                                          for rb, re, rs in node_range])

        for dim in range(data_dims):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[dim], symbolic.SymExpr):
                    dexprs.append(expr[dim].expr)
                elif isinstance(expr[dim], tuple):
                    dexprs.append((expr[dim][0].expr if isinstance(expr[dim][0], symbolic.SymExpr) else expr[dim][0],
                                   expr[dim][1].expr if isinstance(
                                       expr[dim][1], symbolic.SymExpr) else expr[dim][1],
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
                                           re.expr if isinstance(
                                               re, symbolic.SymExpr) else re,
                                           rs.expr if isinstance(rs, symbolic.SymExpr) else rs)
                                          for rb, re, rs in node_range])

        for i, smpattern in enumerate(self.patterns_per_dim):

            dexprs = []
            for expr in expressions:
                if isinstance(expr[i], symbolic.SymExpr):
                    dexprs.append(expr[i].expr)
                elif isinstance(expr[i], tuple):
                    dexprs.append((expr[i][0].expr if isinstance(expr[i][0], symbolic.SymExpr) else expr[i][0],
                                   expr[i][1].expr if isinstance(
                                       expr[i][1], symbolic.SymExpr) else expr[i][1],
                                   expr[i][2].expr if isinstance(
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
        of the form `a * {index} + b`. Only works for ranges like (a * i + b : a * i + b : s)
    """

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
                # if the range does not represent a single index return False
                # TODO: remove this for more precise analyisis
                if not subexprs[0] == subexprs[1] or step != 1:
                    return False

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

        result_skip = self.multiplier * node_rs
        result_tile = 1

        # TODO: Okay this is basically for cases like tiling, we can ignore this for now
        # Special case: i:i+stride for a begin:end:stride range
        # if (node_rb == result_begin and (re - rb + 1) == node_rs and rs == 1 and rt == 1):
        #     return (node_rb, node_re, 1, 1)

        # Experimental
        # This should be using sympy.floor
        # memlet_start_pts = ((re - rt + 1 - rb) / rs) + 1
        # memlet_rlen = memlet_start_pts.expand() * rt
        # interval_len = (result_end - result_begin + 1)
        # num_elements = node_rlen * memlet_rlen

        # if (interval_len == num_elements or interval_len.expand() == num_elements):
        #     # Continuous access
        #     result_skip = 1
        #     result_tile = 1
        # else:
        #     if rt == 1:
        #         result_skip = (result_end - result_begin - re + rb) / (node_re - node_rb)
        #         try:
        #             if result_skip < 1:
        #                 result_skip = 1
        #         except:
        #             pass
        #         result_tile = result_end - result_begin + 1 - (node_rlen - 1) * result_skip
        #     else:
        #         candidate_skip = rs
        #         candidate_tile = rt * node_rlen
        #         candidate_lstart_pt = result_end - result_begin + 1 - candidate_tile
        #         if simplify(candidate_lstart_pt / (num_elements / candidate_tile - 1)) == candidate_skip:
        #             result_skip = rs
        #             result_tile = rt * node_rlen
        #         else:
        #             result_skip = rs / node_rlen
        #             result_tile = rt

        #     if result_skip == result_tile or result_skip == 1:
        #         result_skip = 1
        #         result_tile = 1

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
        if len(dim_exprs) > 1:
            return False
        dexpr = dim_exprs[0]
        # Pattern does not support ranges
        if not isinstance(dexpr, sympy.Basic):
            return False

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

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):
        # Pattern does not support unions of expressions. TODO: Support
        if len(dim_exprs) > 1:
            return False
        dexpr = dim_exprs[0]

        free_symbols = set()
        for expr in dexpr:
            free_symbols |= expr.free_symbols
        for var in variable_context[-1]:
            if var in free_symbols:
                return False

        # # Create a wildcard that excludes current map's parameters
        # cst = sympy.Wild('cst', exclude=variable_context[-1] + list(variable_context[-2]))

        # # Range case
        # if isinstance(dexpr, tuple) and len(dexpr) == 3:
        #     # Try to match a constant expression for the range
        #     for rngelem in dexpr:
        #         if dtypes.isconstant(rngelem):
        #             continue

        #         matches = rngelem.match(cst)
        #         if matches is None or len(matches) != 1:
        #             return False
        #         if not matches[cst].is_constant():
        #             return False

        # else:  # Single element case
        #     # Try to match a constant expression
        #     if not dtypes.isconstant(dexpr):
        #         matches = dexpr.match(cst)
        #         if matches is None or len(matches) != 1:
        #             return False
        #         if not matches[cst].is_constant():
        #             return False

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

        return False
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
        # TODO: change this for later versions
        return super().depends_on()

    def modifies(self) -> Modifies:
        return ppl.Modifies.Everything

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
        iteration_variables.clear()

        # fill the approximation dictionary with the original edges as keys and the edges with the
        # approximated memlets as values
        for (edge, parent) in sdfg.all_edges_recursive():
            if isinstance(parent, SDFGState):
                approximation_dict[edge] = copy.deepcopy(edge.data)
                if not isinstance(approximation_dict[edge].subset, subsets.Subsetlist) and not approximation_dict[edge].subset == None:
                    approximation_dict[edge].subset = subsets.Subsetlist(
                        [approximation_dict[edge].subset])
                if not isinstance(approximation_dict[edge].dst_subset, subsets.Subsetlist) and not approximation_dict[edge].dst_subset == None:
                    approximation_dict[edge].dst_subset = subsets.Subsetlist(
                        [approximation_dict[edge].dst_subset])
                if not isinstance(approximation_dict[edge].src_subset, subsets.Subsetlist) and not approximation_dict[edge].src_subset == None:
                    approximation_dict[edge].src_subset = subsets.Subsetlist(
                        [approximation_dict[edge].src_subset])

        self.propagate_memlets_sdfg(sdfg)

        # TODO: make sure that no Memlet contains None as subset in the first place
        for entry in approximation_dict.values():
            if entry.subset is None:
                entry.subset = subsets.Subsetlist([])
        return {
            "approximation": approximation_dict,
            "loop_approximation": loop_write_dict,
            "loops": loop_dict
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

                    loop_states = sdutils.dfs_conditional(
                        sdfg, sources=[begin], condition=lambda _, child: child != guard)

                    if itvar not in begin.ranges:
                        # only do this if loop has not been annotated yet
                        for v in loop_states:
                            v.ranges[itervar] = subsets.Range([rng])
                            loop_state_list.append(v)
                        guard.ranges[itervar] = subsets.Range([rng])
                        guard.condition_edge = sdfg.edges_between(guard, begin)[
                            0]
                        guard.is_loop_guard = True
                        guard.itvar = itervar
                        identified_loops[guard] = (
                            begin, last_loop_state, loop_state_list, itvar, subsets.Range([rng]))

            else:
                # There's no guard state, so this cycle marks all states in it as
                # dynamically unbounded.
                unannotated_cycle_states.extend(cycle)

        return identified_loops


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
                sdfg.add_edge(sink_node, dummy_sink,
                              dace.sdfg.InterstateEdge())

        # get all the nodes that are executed unconditionally in the cfg a.k.a nodes that dominate the sink states
        dominators = cfg.all_dominators(sdfg)
        states = dominators[dummy_sink]

        # remove dummy state
        sdfg.remove_node(dummy_sink)

        # For each state, go through all access nodes corresponding to any in- or
        # out-connectors to and from this SDFG. Given those access nodes, collect
        # the corresponding memlets and use them to calculate the
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
                            memlet = Memlet(data=inside_memlet.data)
                            memlet._is_data_src = True
                            border_memlets[direction][node.label] = memlet

                    # Given all of this access nodes' memlets, propagate the subset
                    # according to the state's variable ranges.
                    if len(memlets) > 0:
                        params = []
                        ranges = []

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
                        subset = self.propagate_subset(
                            memlets, array, params, subsets.Range(ranges), use_dst=use_dst).subset
                        if not subset:
                            continue
                        # If the border memlet already has a set range, compute the
                        # union of the ranges to merge the subsets.
                        if memlet.subset is not None:
                            if memlet.subset.dims() != subset.dims():
                                raise ValueError(
                                    'Cannot merge subset ranges of unequal dimension!')
                            else:
                                memlet.subset = subsets.list_union(
                                    memlet.subset, subset)
                        else:
                            memlet.subset = subset

            # propagate the memlets for each loop
            if state in loop_write_dict.keys():
                for node_label, loop_memlet in loop_write_dict[state].items():
                    if (node_label not in border_memlets["out"]):
                        continue
                    memlet = border_memlets["out"][node_label]

                    if memlet is None:
                        # Use the first encountered memlet as a 'border' memlet
                        # and accumulate the sum on it.
                        memlet = Memlet(data=loop_memlet.data)
                        memlet._is_data_src = True
                        border_memlets["out"][node_label] = memlet

                    params = []
                    ranges = []
                    if len(params) == 0 or len(ranges) == 0:
                        params = ['__dace_dummy']
                        ranges = [(0, 0, 1)]

                    use_dst = True
                    array = sdfg.arrays[node_label]
                    subset = self.propagate_subset(
                        [loop_memlet], array, params, subsets.Range(ranges), use_dst=use_dst).subset

                    # If the border memlet already has a set range, compute the
                    # union of the ranges to merge the subsets.
                    if memlet.subset is not None:
                        if memlet.subset.dims() != subset.dims():
                            raise ValueError(
                                'Cannot merge subset ranges of unequal dimension!')
                        else:
                            memlet.subset = subsets.list_union(
                                memlet.subset, subset)
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

                self.unsqueeze_memlet_subsetList(
                    internal_memlet, in_memlet, parent_sdfg)

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

                self.unsqueeze_memlet_subsetList(
                    internal_memlet, out_memlet, parent_sdfg)

                approximation_dict[edge] = out_memlet

    def unsqueeze_memlet_subsetList(self, internal_memlet: Memlet, external_memlet: Memlet, parent_sdfg: dace.SDFG):
        """helper method that tries to unsqueeze a memlet in a nested SDFG. If it fails it falls back to an empty memlet."""

        from dace.transformation.helpers import unsqueeze_memlet

        if isinstance(external_memlet.subset, subsets.Subsetlist):
            external_memlet.subset = external_memlet.subset.subset_list[0]
        if isinstance(external_memlet.dst_subset, subsets.Subsetlist):
            external_memlet.dst_subset = external_memlet.dst_subset.subset_list[0]
        if isinstance(external_memlet.src_subset, subsets.Subsetlist):
            external_memlet.src_subset = external_memlet.src_subset.subset_list[0]

        if isinstance(internal_memlet.subset, subsets.Subsetlist):
            _subsets = internal_memlet.subset.subset_list
        else:
            _subsets = [internal_memlet.subset]

        tmp_memlet = Memlet(
            data=internal_memlet.data,
            subset=internal_memlet.subset,
            other_subset=internal_memlet.other_subset
        )

        for j, subset in enumerate(_subsets):
            if subset is None:
                continue
            tmp_memlet.subset = subset
            try:
                unsqueezed_memlet = unsqueeze_memlet(
                    tmp_memlet, external_memlet, False)
                # If no appropriate memlet found fall back to empty subset
                # for i, (rng, s) in enumerate(zip(tmp_memlet.subset, parent_sdfg.arrays[unsqueezed_memlet.data].shape)):
                #     if rng[1] + 1 == s:
                #         unsqueezed_memlet.subset = None
                #         break

                subset = unsqueezed_memlet.subset
            except (ValueError, NotImplementedError):
                # In any case of memlets that cannot be unsqueezed (i.e.,
                # reshapes), use dynamic unbounded memlets.
                subset = None
            _subsets[j] = subset

        if all(s is None for s in _subsets):
            external_memlet.subset = None
            external_memlet.dst_subset = None
        else:
            external_memlet = unsqueezed_memlet
            external_memlet.subset = subsets.Subsetlist(_subsets)

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
        from dace.transformation.helpers import split_interstate_edges

        # Reset previous annotations first
        # TODO: This is important for the loop analysis. Is just running this again OK?
        self.reset_state_annotations(sdfg)
        split_interstate_edges(sdfg)
        loops = self._annotate_loop_ranges(sdfg, [])
        loop_dict.update(loops)

        for state in sdfg.nodes():
            self.propagate_memlets_state(sdfg, state)

        self.propagate_memlet_loop(sdfg, loops)

    def propagate_memlet_loop(self, sdfg, loops:  dict[SDFGState, (SDFGState, SDFGState, list[SDFGState], str, subsets.Range)], loopheader: SDFGState = None):
        """
        Propagate Memlets recursively out of loop constructs with representative border memlets, 
        similar to propagate_memlets_nested_sdfg(). 
        Only states that are executed unconditionally are considered. 
        Loops containing a break are ignored. 
        The results are written to the global loop_write_dict

        :param sdfg: The SDFG the loops are contained in.
        :param loops: dictionary that maps each for-loop construct to a tuple consisting of first state in the loop, the last state in the loop
                    the set of states enclosed by the loop, the itearation variable and the range of the iterator variable
        :param loopheader: a loopheader to start the propagation with. If no parameter is given, propagate_memlet_loop will be called recursively on the outermost loopheaders

        """

        def filter_subsets(itvar: str, s_itvars: List[str], range: subsets.Range, memlet: Memlet):
            # helper method that filters out subsets that do not depend on the iteration variable if the iteration range is symbolic

            # if loop range is symbolic -> only propagate subsets that contain the iterator as a symbol
            # if loop range is constant (and not empty, which is already verified) always propagate all subsets out

            if memlet.subset is None:
                return None
            filtered_subsets = memlet.subset.subset_list if isinstance(memlet.subset, subsets.Subsetlist) else [memlet.subset]

            # range contains symbols
            if (range.free_symbols):
                filtered_subsets = [s for s in filtered_subsets if itvar in s.free_symbols]

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

            # propagate out of loops recursively
            for loopheader in top_loopheaders:
                self.propagate_memlet_loop(sdfg, loops, loopheader)
            return

        current_loop = loops[loopheader]
        begin, last_loop_state, loop_states, itvar, rng = current_loop
        if rng.num_elements() == 0:
            return

        # make sure there is no break out of the loop
        dominators = cfg.all_dominators(sdfg)
        if any(begin not in dominators[s] and not begin is s for s in loop_states):
            return

        border_memlets = defaultdict(None)
        
        # FIXME: the calculation of the relevant states seems to be off!
        # get all the nodes that are executed unconditionally in the cfg a.k.a nodes that dominate the sink states
        states = dominators[last_loop_state].intersection(set(loop_states))
        states.add(last_loop_state)
        # maintain a list of states that are part of nested loops
        ignore = []
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
            # TODO: this is wrong if the state is a loop guard, because it also includes the current itvar, subtract the current itvar
            surrounding_itvars = state.ranges.keys()
            # iterate over acccessnodes in the state
            for node in state.data_nodes():
                # no writes associated with this access node
                if state.in_degree(node) == 0:
                    continue
                edges = state.in_edges(node)
                # get the current border memlet for this data node
                border_memlet = border_memlets.get(node.label)
                memlets = []

                # collect all the subsets of the incoming memlets for the current access node
                for edge in edges:
                    inside_memlet = copy.copy(approximation_dict[edge])

                    # filter out subsets that could become empty depending on assignments of symbols
                    filtered_subsets = filter_subsets(
                        itvar, surrounding_itvars, rng, inside_memlet)
                    if not filtered_subsets:
                        continue

                    inside_memlet.subset = subsets.Subsetlist(filtered_subsets)
                    memlets.append(inside_memlet)
                    if border_memlet is None:
                        # Use the first encountered memlet as a 'border' memlet
                        # and accumulate the sum on it.
                        border_memlet = Memlet(data=inside_memlet.data)
                        border_memlet._is_data_src = True
                        border_memlets[node.label] = border_memlet

                self.propagate_loop_subset(
                    sdfg, memlets, border_memlet, sdfg.arrays[node.label], itvar, rng)

            if state not in loop_write_dict.keys():
                continue
            # propagate the border memlets of nested loop
            for node_label, other_border_memlet in loop_write_dict[state].items():
                filtered_subsets = filter_subsets(
                    itvar, surrounding_itvars, rng, other_border_memlet)
                if not filtered_subsets:
                    continue

                other_border_memlet.subset = subsets.Subsetlist(filtered_subsets)
                border_memlet = border_memlets.get(node_label)
                if border_memlet is None:
                    # Use the first encountered memlet as a 'border' memlet
                    # and accumulate the sum on it.
                    border_memlet = Memlet(data=other_border_memlet.data)
                    border_memlet._is_data_src = True
                    border_memlets[node_label] = border_memlet

                self.propagate_loop_subset(
                    sdfg, [other_border_memlet], border_memlet, sdfg.arrays[node_label], itvar, rng)

        loop_write_dict[loopheader] = border_memlets

    def propagate_loop_subset(self, 
                              sdfg: dace.SDFG, 
                              memlets: List[Memlet], 
                              dst_memlet: Memlet, 
                              arr: dace.data.Array, 
                              itvar: str, 
                              rng: subsets.Subset,
                              loop_nest_itvars: Set[str] = set()):
        if len(memlets) > 0:
            params = [itvar]
            ranges = [rng]

            # TODO: Is this even necessary. Can't hurt i guess
            if len(params) == 0 or len(ranges) == 0:
                params = ['__dace_dummy']
                ranges = [(0, 0, 1)]
            
            # get all the other iteration variables surrounding this memlet
            surrounding_itvars = iteration_variables[sdfg] if sdfg in iteration_variables.keys() else set()
            surrounding_itvars |=  loop_nest_itvars

            use_dst = True
            subset = self.propagate_subset(
                memlets, arr, params, rng, use_dst=use_dst, surrounding_itvars= surrounding_itvars).subset

            if subset is None or not len(subset.subset_list):
                return
            # If the border memlet already has a set range, compute the
            # union of the ranges to merge the subsets.
            if dst_memlet.subset is not None:
                if dst_memlet.subset.dims() != subset.dims():
                    raise ValueError(
                        'Cannot merge subset ranges of unequal dimension!')
                else:
                    dst_memlet.subset = subsets.list_union(
                        dst_memlet.subset, subset)
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
                # TODO: collect iteration variables from surrounding maps, SDFG and state.
                # apply variable mapping and save it in the global variable dict

                map_ivars = self.collect_iteration_variables(sdfg, state, node)
                sdfg_ivars = iteration_variables[sdfg] if sdfg in iteration_variables.keys(
                ) else set()
                state_ivars = state.ranges.keys()
                it_vars = map_ivars | sdfg_ivars | state_ivars

                def mapfunction(mapping, symbol):
                    if symbol in mapping.keys():
                        return mapping[symbol]
                    else:
                        return None

                # apply symbol mapping of nested SDFG
                symbol_mapping = node.symbol_mapping
                iteration_variables[node.sdfg] = set(map(
                    lambda x: mapfunction(symbol_mapping, x), it_vars))

                # Propagate memlets inside the nested SDFG.
                self.propagate_memlets_sdfg(node.sdfg)

                # Propagate memlets out of the nested SDFG.
                self.propagate_memlets_nested_sdfg(sdfg, state, node)

        # Process scopes from the leaves upwards
        self.propagate_memlets_scope(sdfg, state, state.scope_leaves())

    def collect_iteration_variables(self, sdfg: dace.SDFG, state: SDFGState, node: nodes.NestedSDFG) -> set[str]:
        scope_dict = state.scope_dict()
        current_scope: nodes.EntryNode = scope_dict[node]
        params = set()
        while (current_scope):
            # TODO: what about consume?
            mapnode: nodes.Map = current_scope.map
            params.update(set(mapnode.params))
            current_scope = scope_dict[current_scope]

        return params

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

        surrounding_map_vars = {}

        # collect all the iteration variables of the surrounding maps
        # TODO: make this more efficient. collect the variables from the root down. also i do this twice here 
        while len(scopes_to_process) > 0:
            for scope in scopes_to_process:
                if scope is None:
                    continue
                next_scope = scope
                while(next_scope):
                    next_scope = next_scope.parent
                    if next_scope is None:
                        break
                    curr_entry = next_scope.entry
                    if scope not in surrounding_map_vars.keys():
                        surrounding_map_vars[scope] = set()
                    if isinstance(curr_entry, nodes.MapEntry):
                        surrounding_map_vars[scope] |= set(curr_entry.map.params)
                next_scopes.add(scope.parent)
            scopes_to_process = next_scopes
            next_scopes = set()

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

                map_vars = surrounding_map_vars[scope] if scope in surrounding_map_vars.keys() else set()
                sdfg_vars = iteration_variables[sdfg] if sdfg in iteration_variables.keys() else set()
                loop_vars = state.ranges.keys()

                surrounding_vars = map_vars | sdfg_vars | loop_vars

                # TODO: Maybe only propagate in one direction since we are only interested in writes
                # Propagate out of entry
                if propagate_entry:
                    self._propagate_node(state, scope.entry, surrounding_vars)

                # Propagate out of exit
                if propagate_exit:
                    self._propagate_node(state, scope.exit, surrounding_vars)

                # Add parent to next frontier
                next_scopes.add(scope.parent)
            scopes_to_process = next_scopes
            next_scopes = set()

    def _propagate_node(self, dfg_state, node, surrounding_itvars: set[str] = None):
        if isinstance(node, nodes.EntryNode):
            internal_edges = [e for e in dfg_state.out_edges(
                node) if e.src_conn and e.src_conn.startswith('OUT_')]
            external_edges = [e for e in dfg_state.in_edges(
                node) if e.dst_conn and e.dst_conn.startswith('IN_')]

            def geticonn(e): return e.src_conn[4:]
            def geteconn(e): return e.dst_conn[3:]
            use_dst = False
        else:
            internal_edges = [e for e in dfg_state.in_edges(
                node) if e.dst_conn and e.dst_conn.startswith('IN_')]
            external_edges = [e for e in dfg_state.out_edges(
                node) if e.src_conn and e.src_conn.startswith('OUT_')]

            def geticonn(e): return e.dst_conn[3:]
            def geteconn(e): return e.src_conn[4:]
            use_dst = True

        # TODO: generalize this for subsetLists (?)
        for edge in external_edges:
            if approximation_dict[edge].is_empty():
                new_memlet = Memlet()
            else:
                internal_edge = next(
                    e for e in internal_edges if geticonn(e) == geteconn(edge))
                aligned_memlet = self.align_memlet(
                    dfg_state, internal_edge, dst=use_dst)
                new_memlet = self.propagate_memlet(
                    dfg_state, aligned_memlet, node, True, connector=geteconn(edge), surrounding_itvars = surrounding_itvars)
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
                         connector=None,
                         surrounding_itvars: set[str] = None):
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
                neighboring_edges = [
                    e for e in neighboring_edges if e.src_conn and e.src_conn[4:] == connector]
        elif isinstance(scope_node, nodes.ExitNode):
            use_dst = True
            entry_node = dfg_state.entry_node(scope_node)
            neighboring_edges = dfg_state.in_edges(scope_node)
            if connector is not None:
                neighboring_edges = [
                    e for e in neighboring_edges if e.dst_conn and e.dst_conn[3:] == connector]
        else:
            raise TypeError('Trying to propagate through a non-scope node')
        if memlet.is_empty():
            return Memlet()

        sdfg = dfg_state.parent
        scope_node_symbols = set(
            conn for conn in entry_node.in_connectors if not conn.startswith('IN_'))
        defined_vars = [
            symbolic.pystr_to_symbolic(s) for s in (dfg_state.symbols_defined_at(entry_node).keys()
                                                    | sdfg.constants.keys()) if s not in scope_node_symbols
        ]

        # Find other adjacent edges within the connected to the scope node
        # and union their subsets
        if union_inner_edges:
            aggdata = [approximation_dict[e]for e in neighboring_edges if approximation_dict[e].data ==
                       memlet.data and approximation_dict[e] != memlet]
        else:
            aggdata = []

        aggdata.append(memlet)

        if arr is None:
            if memlet.data not in sdfg.arrays:
                raise KeyError(
                    'Data descriptor (Array, Stream) "%s" not defined in SDFG.' % memlet.data)

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
            return self.propagate_subset(aggdata, arr, mapnode.params, mapnode.range, defined_vars, use_dst=use_dst, surrounding_itvars = surrounding_itvars)

        elif isinstance(entry_node, nodes.ConsumeEntry):
            # Nothing to analyze/propagate in consume
            new_memlet = copy.copy(memlet)
            new_memlet.subset = subsets.Range.from_array(arr)
            new_memlet.other_subset = None
            return new_memlet
        else:
            raise NotImplementedError(
                'Unimplemented primitive: %s' % type(entry_node))

    # External API

    def propagate_subset(self, memlets: List[Memlet],
                         arr: data.Data,
                         params: List[str],
                         rng: subsets.Subset,
                         defined_variables: Set[symbolic.SymbolicType] = None,
                         use_dst: bool = False,
                         surrounding_itvars: Set[str] = set()) -> Memlet:
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
            :param surrounding_itvars:  set of iterator variables that surround the memlet
                                        but are not propagated in this call
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
            defined_variables = set(symbolic.pystr_to_symbolic(p)
                                    for p in defined_variables)

        # Propagate subset
        variable_context = [[symbolic.pystr_to_symbolic(p) for p in surrounding_itvars], defined_variables, [
            symbolic.pystr_to_symbolic(p) for p in params]]

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
                continue

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
                new_subset = subsets.list_union(
                    new_subset, subsets.Subsetlist(_subsets))
                if new_subset is None:
                    warnings.warn('Subset union failed between %s and %s ' % (
                        old_subset, _subsets))
                    break

        # Create new memlet
        new_memlet = copy.copy(memlets[0])
        new_memlet.subset = new_subset
        new_memlet.other_subset = None
        return new_memlet

    def _freesyms(self, expr):
        """ 
        Helper function that either returns free symbols for sympy expressions
        or an empty set if constant.
        """
        if isinstance(expr, sympy.Basic):
            return expr.free_symbols
        return {}
