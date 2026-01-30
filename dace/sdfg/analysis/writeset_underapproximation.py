# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass derived from ``propagation.py`` that under-approximates write-sets of for-loops and Maps in an SDFG.
"""

import copy
from dataclasses import dataclass, field
import itertools
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

import sympy

import dace
from dace import SDFG, Memlet, data, dtypes, registry, subsets, symbolic
from dace.sdfg import SDFGState
from dace.sdfg import graph
from dace.sdfg import graph as gr
from dace.sdfg import nodes, scope
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.nodes import AccessNode, NestedSDFG
from dace.sdfg.state import LoopRegion
from dace.symbolic import issymbolic, pystr_to_symbolic, simplify
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.pass_pipeline import Modifies


@registry.make_registry
class UnderapproximationMemletPattern(object):
    """
    A pattern match on a memlet subset that can be used for propagation.
    """

    def can_be_applied(self, expressions, variable_context, node_range, orig_edges):
        raise NotImplementedError

    def propagate(self, array, expressions, node_range):
        raise NotImplementedError


@registry.make_registry
class SeparableUnderapproximationMemletPattern(object):
    """ Memlet pattern that can be applied to each of the dimensions
        separately. """

    def can_be_applied(self, dim_exprs, variable_context, node_range, orig_edges, dim_index, total_dims):
        raise NotImplementedError

    def propagate(self, array, dim_exprs, node_range):
        raise NotImplementedError


@registry.autoregister
class SeparableUnderapproximationMemlet(UnderapproximationMemletPattern):
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
        if not self._iteration_variables_appear_only_once(data_dims, expressions, other_params, params):
            return False

        node_range = self._make_range(node_range)

        for dim in range(data_dims):
            dexprs = []
            for expr in expressions:
                expr_dim = expr[dim]
                if isinstance(expr_dim, symbolic.SymExpr):
                    dexprs.append(expr_dim.expr)
                elif isinstance(expr_dim, tuple):
                    dexprs.append((expr_dim[0].expr if isinstance(expr_dim[0], symbolic.SymExpr) else expr_dim[0],
                                   expr_dim[1].expr if isinstance(expr_dim[1], symbolic.SymExpr) else expr_dim[1],
                                   expr_dim[2].expr if isinstance(expr_dim[2], symbolic.SymExpr) else expr_dim[2]))
                else:
                    dexprs.append(expr_dim)

            for pattern_class in SeparableUnderapproximationMemletPattern.extensions().keys():
                smpattern = pattern_class()
                if smpattern.can_be_applied(dexprs, variable_context, node_range, orig_edges, dim, data_dims):
                    self.patterns_per_dim[dim] = smpattern
                    break

        return None not in self.patterns_per_dim

    def _iteration_variables_appear_only_once(self, data_dims, expressions, other_params, params):
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
                            curr_dim_expr[0], symbolic.SymExpr) else list(
                                pystr_to_symbolic(curr_dim_expr[0]).expand().free_symbols)
                        free_symbols += curr_dim_expr[1].expr.free_symbols if isinstance(
                            curr_dim_expr[1], symbolic.SymExpr) else list(
                                pystr_to_symbolic(curr_dim_expr[1]).expand().free_symbols)
                        free_symbols += curr_dim_expr[2].expr.free_symbols if isinstance(
                            curr_dim_expr[2], symbolic.SymExpr) else list(
                                pystr_to_symbolic(curr_dim_expr[2]).expand().free_symbols)
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
        return True

    def _make_range(self, node_range):
        return subsets.Range([
            (rb.expr if isinstance(rb, symbolic.SymExpr) else rb, re.expr if isinstance(re, symbolic.SymExpr) else re,
             rs.expr if isinstance(rs, symbolic.SymExpr) else rs) for rb, re, rs in node_range
        ])

    def propagate(self, array, expressions, node_range):
        result = [(None, None, None)] * len(self.patterns_per_dim)

        node_range = self._make_range(node_range)

        for i, smpattern in enumerate(self.patterns_per_dim):

            dexprs = []
            for expr in expressions:
                expr_i = expr[i]
                if isinstance(expr_i, symbolic.SymExpr):
                    dexprs.append(expr_i.expr)
                elif isinstance(expr_i, tuple):
                    dexprs.append(
                        (expr_i[0].expr if isinstance(expr_i[0], symbolic.SymExpr) else expr_i[0],
                         expr_i[1].expr if isinstance(expr_i[1], symbolic.SymExpr) else expr_i[1],
                         expr_i[2].expr if isinstance(expr_i[2], symbolic.SymExpr) else expr_i[2], expr.tile_sizes[i]))
                else:
                    dexprs.append(expr_i)

            result[i] = smpattern.propagate(array, dexprs, node_range)

        # TODO(later): Not necessarily Range (general integer sets)
        return subsets.Range(result)


@registry.autoregister
class AffineUnderapproximationSMemlet(SeparableUnderapproximationMemletPattern):
    """
    Separable memlet pattern that matches affine expressions, i.e., of the
    form `a * {index} + b`. Only works for expressions like (a * i + b : a * i + b : 1)
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
                # if step of subscript expression is not 1 back off
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
        # parameter range
        node_rb, node_re, node_rs = node_range[self.paramind]  # node_rs = 1

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

            # subscript expression
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

        result_begin = simplify(result_begin)
        result_end = simplify(result_end)
        result_skip = simplify(result_skip)
        result_tile = simplify(result_tile)

        return (result_begin, result_end, result_skip, result_tile)


@registry.autoregister
class ConstantUnderapproximationSMemlet(SeparableUnderapproximationMemletPattern):
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
            if isinstance(expr, sympy.Basic):
                free_symbols |= expr.free_symbols
            else:
                continue
        for var in variable_context[-1]:
            if var in free_symbols:
                return False

        return True

    def propagate(self, array, dim_exprs, node_range):
        if isinstance(dim_exprs[0], tuple):
            return dim_exprs[0]  # Already in range format
        # Convert index to range format
        return (dim_exprs[0], dim_exprs[0], 1)


def _subexpr(dexpr, repldict):
    if isinstance(dexpr, tuple):
        return tuple(_subexpr(d, repldict) for d in dexpr)
    elif isinstance(dexpr, symbolic.SymExpr):
        return dexpr.expr.subs(repldict)
    else:
        return dexpr.subs(repldict)


@registry.autoregister
class ConstantRangeUnderapproximationMemlet(UnderapproximationMemletPattern):
    """
    Memlet pattern that matches arbitrary expressions with constant range.
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


def _find_unconditionally_executed_states(sdfg: SDFG) -> Set[SDFGState]:
    """
    Returns all states that are executed unconditionally in an SDFG
    """
    dummy_sink = sdfg.add_state("dummy_state")
    for sink_node in sdfg.sink_nodes():
        if sink_node is not dummy_sink:
            sdfg.add_edge(sink_node, dummy_sink, dace.sdfg.InterstateEdge())
    # get all the nodes that are executed unconditionally in the state-machine a.k.a nodes
    # that dominate the sink states
    dominators = cfg_analysis.all_dominators(sdfg)
    states = dominators[dummy_sink]
    # remove dummy state
    sdfg.remove_node(dummy_sink)
    return states


def _unsqueeze_memlet_subsetunion(internal_memlet: Memlet, external_memlet: Memlet, parent_sdfg: dace.SDFG,
                                  nsdfg: NestedSDFG) -> Memlet:
    """
    Helper method that tries to unsqueeze a memlet, containing a SubsetUnion as subset, in
    a nested SDFG. If it fails it falls back to an empty memlet.

    :param internal_memlet: The internal memlet to unsqueeze.
    :param
    """

    from dace.transformation.helpers import unsqueeze_memlet

    if isinstance(external_memlet.subset, subsets.SubsetUnion):
        external_memlet.subset = external_memlet.subset.subset_list[0]
    if isinstance(external_memlet.dst_subset, subsets.SubsetUnion):
        external_memlet.dst_subset = external_memlet.dst_subset.subset_list[0]
    if isinstance(external_memlet.src_subset, subsets.SubsetUnion):
        external_memlet.src_subset = external_memlet.src_subset.subset_list[0]
    if isinstance(internal_memlet.subset, subsets.SubsetUnion):
        _subsets = internal_memlet.subset.subset_list
    else:
        _subsets = [internal_memlet.subset]

    tmp_memlet = Memlet(data=internal_memlet.data,
                        subset=internal_memlet.subset,
                        other_subset=internal_memlet.other_subset)

    internal_array = nsdfg.sdfg.arrays[internal_memlet.data]
    external_array = parent_sdfg.arrays[external_memlet.data]

    for j, subset in enumerate(_subsets):
        if subset is None:
            continue
        tmp_memlet.subset = subset
        try:
            unsqueezed_memlet = unsqueeze_memlet(tmp_memlet,
                                                 external_memlet,
                                                 False,
                                                 internal_offset=internal_array.offset,
                                                 external_offset=external_array.offset)
            subset = unsqueezed_memlet.subset
        except (ValueError, NotImplementedError):
            # In any case of memlets that cannot be unsqueezed (i.e.,
            # reshapes), use empty memlets.
            subset = None
        _subsets[j] = subset

    # if all subsets are empty make memlet empty
    if all(s is None for s in _subsets):
        external_memlet.subset = None
        external_memlet.other_subset = None
    else:
        external_memlet = unsqueezed_memlet
        external_memlet.subset = subsets.SubsetUnion(_subsets)

    return external_memlet


def _freesyms(expr):
    """
    Helper function that either returns free symbols for sympy expressions
    or an empty set if constant.
    """
    if isinstance(expr, sympy.Basic):
        return expr.free_symbols
    return {}


def _collect_iteration_variables(state: SDFGState, node: nodes.NestedSDFG) -> Set[str]:
    """
    Helper method which finds all the iteration variables that
    surround a nested SDFG in a state.

    :param state: The state in which the nested SDFG resides
    :param node: The nested SDFG that the surrounding iteration
                variables need to be found for
    :return: The set of iteration variables surrounding the nested SDFG
    """
    scope_dict = state.scope_dict()
    current_scope: nodes.EntryNode = scope_dict[node]
    params = set()
    while current_scope:
        mapnode: nodes.Map = current_scope.map
        params.update(set(mapnode.params))
        current_scope = scope_dict[current_scope]

    return params


def _collect_itvars_scope(scopes: Union[scope.ScopeTree, List[scope.ScopeTree]]) -> Dict[scope.ScopeTree, Set[str]]:
    """
    Helper method which finds all surrounding iteration variables for each scope

    :param scopes: A List of scope trees or a single scopetree to analize
    :return: A dictionary mapping each ScopeTree object in scopes to the
            list of iteration variables surrounding it
    """
    if isinstance(scopes, scope.ScopeTree):
        scopes_to_process = [scopes]
    else:
        scopes_to_process = scopes

    next_scopes = set()
    surrounding_map_vars = {}
    while len(scopes_to_process) > 0:
        for scope_node in scopes_to_process:
            if scope_node is None:
                continue
            next_scope = scope_node
            while next_scope:
                next_scope = next_scope.parent
                if next_scope is None:
                    break
                curr_entry = next_scope.entry
                if scope_node not in surrounding_map_vars:
                    surrounding_map_vars[scope_node] = set()
                if isinstance(curr_entry, nodes.MapEntry):
                    surrounding_map_vars[scope_node] |= set(curr_entry.map.params)
            next_scopes.add(scope_node.parent)
        scopes_to_process = next_scopes
        next_scopes = set()
    return surrounding_map_vars


def _map_header_to_parent_headers(
    loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]]
) -> Dict[SDFGState, Set[SDFGState]]:
    """
    Given the loops of an SDFG returns a mapping that maps each loop to its parents in the loop
    nest tree.
    """
    mapping = {}
    for header, loop in loops.items():
        _, _, loop_states, _, _ = loop
        for state in loop_states:
            if state not in mapping:
                mapping[state] = set()
            if state in loops:
                mapping[state].add(header)
    return mapping


def _generate_loop_nest_tree(
    loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]]
) -> Dict[SDFGState, Set[SDFGState]]:
    """
    Given the loops of an SDFG returns the loop nest trees in the SDFG represented by a dictionary.
    """
    header_parents_mapping = _map_header_to_parent_headers(loops)
    tree_dict: Dict[SDFGState, Set[SDFGState]] = {}
    for header, loop in loops.items():
        _, _, loop_states, _, _ = loop
        tree_dict[header] = set()
        for state in loop_states:
            # if the state is a loop header and no parent header is a child of header state is a direct child
            if state in loops and len(set(loop_states).intersection(header_parents_mapping[state])) == 0:
                tree_dict[header].add(state)
    return tree_dict


def _postorder_traversal(root: SDFGState, loop_nest_tree: Dict[SDFGState, Set[SDFGState]]) -> List[SDFGState]:
    """
    Given a loop nest tree in the form of a dictionary and the root of the tree, returns the DFS
    traversal order of that tree starting from the root.
    """
    post_order_list = []
    if root is None:
        return []
    stack = [root]
    last = None

    while stack:
        root = stack[-1]
        if root in loop_nest_tree:
            children = loop_nest_tree[root]
        else:
            children = []
        if not children or last is not None and (last in children):
            post_order_list.append(root)
            stack.pop()
            last = root
        # if not, push children in stack
        else:
            for child in children:
                stack.append(child)
    return post_order_list


def _find_loop_nest_roots(loop_nest_tree: Dict[SDFGState, Set[SDFGState]]) -> Set[SDFGState]:
    """
    Given the loop nest trees in an SDFG in the form of a dictionary, returns the root nodes of
    all loop nest trees in that SDFG.
    """
    all_nodes = set()
    child_nodes = set()

    for parent, children in loop_nest_tree.items():
        all_nodes.add(parent)
        all_nodes.update(children)
        child_nodes.update(children)
    roots = all_nodes - child_nodes
    return roots


def _filter_undefined_symbols(border_memlet: Memlet, outer_symbols: Dict[str, dtypes.typeclass]):
    '''
    Helper method that filters out subsets containing symbols which are not defined
    outside a nested SDFG.

    :note: This function operates in-place on the given memlet.
    '''
    if border_memlet.src_subset is not None:
        if isinstance(border_memlet.src_subset, subsets.SubsetUnion):
            _subsets = border_memlet.src_subset.subset_list
        else:
            _subsets = [border_memlet.src_subset]
        for i, subset in enumerate(_subsets):
            for rng in subset:
                fall_back = False
                for item in rng:
                    if any(str(s) not in outer_symbols for s in item.free_symbols):
                        fall_back = True
                        break
                if fall_back:
                    _subsets[i] = None
                    break
        border_memlet.src_subset = subsets.SubsetUnion(_subsets)
    if border_memlet.dst_subset is not None:
        if isinstance(border_memlet.dst_subset, subsets.SubsetUnion):
            _subsets = border_memlet.dst_subset.subset_list
        else:
            _subsets = [border_memlet.dst_subset]
        for i, subset in enumerate(_subsets):
            for rng in subset:
                fall_back = False
                for item in rng:
                    if any(str(s) not in outer_symbols for s in item.free_symbols):
                        fall_back = True
                        break
                if fall_back:
                    _subsets[i] = None
                    break
        border_memlet.dst_subset = subsets.SubsetUnion(_subsets)


def _merge_subsets(subset_a: subsets.Subset, subset_b: subsets.Subset) -> subsets.SubsetUnion:
    """
    Helper function that merges two subsets to a SubsetUnion and throws
    an error if the subsets have different dimensions
    """
    if subset_a is not None:
        if subset_a.dims() != subset_b.dims():
            raise ValueError('Cannot merge subset ranges of unequal dimension!')
        return subsets.list_union(subset_a, subset_b)
    else:
        return subset_b


@dataclass
class UnderapproximateWritesDict:
    approximation: Dict[graph.Edge, Memlet] = field(default_factory=dict)
    loop_approximation: Dict[SDFGState, Dict[str, Memlet]] = field(default_factory=dict)
    loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str,
                                 subsets.Range]] = field(default_factory=dict)


@transformation.explicit_cf_compatible
class UnderapproximateWrites(ppl.Pass):

    # Dictionary mapping each edge to a copy of the memlet of that edge with its write set underapproximated.
    approximation_dict: Dict[graph.Edge, Memlet]
    # Dictionary that maps loop headers to "border memlets" that are written to in the corresponding loop.
    loop_write_dict: Dict[SDFGState, Dict[str, Memlet]]
    # Dictionary containing information about the for loops in the SDFG.
    loop_dict: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]]
    # Dictionary mapping each nested SDFG to the iteration variables surrounding it.
    iteration_variables: Dict[SDFG, Set[str]]
    # Mapping of state to the iteration variables surrounding them, including the ones from surrounding SDFGs.
    ranges_per_state: Dict[SDFGState, Dict[str, subsets.Range]]

    def __init__(self):
        super().__init__()
        self.approximation_dict = {}
        self.loop_write_dict = {}
        self.loop_dict = {}
        self.iteration_variables = {}
        self.ranges_per_state = defaultdict(lambda: {})

    def modifies(self) -> Modifies:
        return ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply.
        return modified & ppl.Modifies.Everything

    def apply_pass(self, top_sdfg: dace.SDFG, _) -> Dict[int, UnderapproximateWritesDict]:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is
                                populated with prior Pass results as ``{Pass subclass name:
                                returned object from pass}``. If not run in a pipeline, an
                                empty dictionary is expected.
        :return: A dictionary containing three dictionaries with analysis data:
            - 'approximation': A dictionary mapping each edge to a copy of the memlet of that edge
                                with its write set underapproximated
            - 'loop_approximation': A dictionary mapping each identified for-loop in the SDFG to
                                    its under-approximated write-set
            - 'loops': A dictionary containing information about the identified for-loops in the
                        SDFG. It maps loop guard-states to the first state in the loop,
                        the set of states enclosed by the loop, the itearation variable and
                        the range of the iteration variable

        :notes: The only modification this pass performs on the SDFG is splitting interstate
                edges.
        """
        result = defaultdict(lambda: UnderapproximateWritesDict())

        for sdfg in top_sdfg.all_sdfgs_recursive():
            # Clear the global dictionaries.
            self.approximation_dict = {}
            self.loop_write_dict = {}
            self.loop_dict = {}
            self.iteration_variables = {}
            self.ranges_per_state = defaultdict(lambda: {})

            # fill the approximation dictionary with the original edges as keys and the edges with the
            # approximated memlets as values
            for (edge, parent) in sdfg.all_edges_recursive():
                if isinstance(parent, SDFGState):
                    self.approximation_dict[edge] = copy.deepcopy(edge.data)
                    if not isinstance(self.approximation_dict[edge].subset,
                                      subsets.SubsetUnion) and self.approximation_dict[edge].subset:
                        self.approximation_dict[edge].subset = subsets.SubsetUnion(
                            [self.approximation_dict[edge].subset])
                    if not isinstance(self.approximation_dict[edge].dst_subset,
                                      subsets.SubsetUnion) and self.approximation_dict[edge].dst_subset:
                        self.approximation_dict[edge].dst_subset = subsets.SubsetUnion(
                            [self.approximation_dict[edge].dst_subset])
                    if not isinstance(self.approximation_dict[edge].src_subset,
                                      subsets.SubsetUnion) and self.approximation_dict[edge].src_subset:
                        self.approximation_dict[edge].src_subset = subsets.SubsetUnion(
                            [self.approximation_dict[edge].src_subset])

            self._underapproximate_writes_sdfg(sdfg)

            # Replace None with empty SubsetUnion in each Memlet
            for entry in self.approximation_dict.values():
                if entry.subset is None:
                    entry.subset = subsets.SubsetUnion([])

            result[sdfg.cfg_id].approximation = self.approximation_dict
            result[sdfg.cfg_id].loop_approximation = self.loop_write_dict
            result[sdfg.cfg_id].loops = self.loop_dict

        return result

    def _underapproximate_writes_sdfg(self, sdfg: SDFG):
        """
        Underapproximates write-sets of loops, maps and nested SDFGs in the given SDFG.
        """
        from dace.transformation.helpers import split_interstate_edges
        from dace.transformation.passes.analysis import loop_analysis

        split_interstate_edges(sdfg)
        loops = self._find_for_loops(sdfg)
        self.loop_dict.update(loops)

        for region in sdfg.all_control_flow_regions():
            if isinstance(region, LoopRegion):
                start = loop_analysis.get_init_assignment(region)
                stop = loop_analysis.get_loop_end(region)
                stride = loop_analysis.get_loop_stride(region)
                for state in region.all_states():
                    self.ranges_per_state[state][region.loop_variable] = subsets.Range([(start, stop, stride)])

            for state in region.all_states():
                self._underapproximate_writes_state(sdfg, state)

        self._underapproximate_writes_loops(loops, sdfg)

    def _find_for_loops(
            self, sdfg: SDFG) -> Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]]:
        """
        Modified version of _annotate_loop_ranges from dace.sdfg.propagation
        that returns the identified loops in a dictionary and stores the found iteration variables
        in the global ranges_per_state dictionary.

        :param sdfg: The SDFG in which to look.
        :return: dictionary mapping loop headers to first state in the loop,
                the set of states enclosed by the loop, the itearation variable,
                the range of the iteration variable
        """

        # We import here to avoid cyclic imports.
        from dace.sdfg import utils as sdutils
        from dace.transformation.interstate.loop_detection import find_for_loop

        # dictionary mapping loop headers to beginstate, loopstates, looprange
        identified_loops = {}
        for cycle in sdfg.find_cycles():
            # In each cycle, try to identify a valid loop guard state.
            guard = None
            begin = None
            itvar = None
            for state in cycle:
                # Try to identify a valid for-loop guard.
                in_edges = sdfg.in_edges(state)
                out_edges = sdfg.out_edges(state)

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
                    if itvarsym in _freesyms(pystr_to_symbolic(iedge.data.assignments[itvar])):
                        increment_edge = iedge
                        break
                if increment_edge is None or increment_edge.src not in cycle:
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
                guard = state
                begin = loop_state
                break

            if guard is not None and begin is not None and itvar is not None:
                # A guard state was identified, see if it has valid for-loop ranges
                # and annotate the loop as such.

                loop_state_list = []
                res = find_for_loop(sdfg, guard, begin, itervar=itvar)
                if res is None:
                    continue
                itervar, rng, (_, last_loop_state) = res
                # Make sure the range is flipped in a direction such that the
                # stride is positive (in order to match subsets.Range).
                start, stop, stride = rng
                # This inequality needs to be checked exactly like this due to
                # constraints in sympy/symbolic expressions, do not simplify!!!
                if (stride < 0) == True:
                    rng = (stop, start, -stride)
                loop_states = sdutils.dfs_conditional(sdfg, sources=[begin], condition=lambda _, child: child != guard)

                if itvar not in self.ranges_per_state[begin]:

                    for loop_state in loop_states:
                        self.ranges_per_state[loop_state][itervar] = subsets.Range([rng])
                        loop_state_list.append(loop_state)
                    self.ranges_per_state[guard][itervar] = subsets.Range([rng])
                    identified_loops[guard] = (begin, last_loop_state, loop_state_list, itvar, subsets.Range([rng]))

        return identified_loops

    def _underapproximate_writes_loops(self, loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str,
                                                                          subsets.Range]], sdfg: SDFG):
        """
        Helper function that calls underapproximate_writes_loops on all the loops in the SDFG in
        bottom up order of the loop nests.
        """
        loop_nest_tree = _generate_loop_nest_tree(loops)
        root_loop_headers = _find_loop_nest_roots(loop_nest_tree)
        for root in root_loop_headers:
            post_order_traversal = _postorder_traversal(root, loop_nest_tree)
            for loop_header in post_order_traversal:
                self._underapproximate_writes_loop(sdfg, loops, loop_header)

    def _underapproximate_writes_state(self, sdfg: SDFG, state: SDFGState):
        """ Propagates memlets throughout one SDFG state.

            :param sdfg: The SDFG in which the state is situated.
            :param state: The state to propagate in.
        """

        # Algorithm:
        # 1. Start propagating information from tasklets outwards (their edges
        #    are hardcoded).
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
        # 3. For each edge in the multigraph, store the results in the global dictionary
        #    approximation_dict

        # First, propagate nested SDFGs in a bottom-up fashion
        dnodes: Set[nodes.AccessNode] = set()
        for node in state.nodes():
            if isinstance(node, AccessNode):
                dnodes.add(node)
            elif isinstance(node, nodes.NestedSDFG):
                self._find_live_iteration_variables(node, sdfg, state)

                # Propagate memlets inside the nested SDFG.
                self._underapproximate_writes_sdfg(node.sdfg)

                # Propagate memlets out of the nested SDFG.
                self._underapproximate_writes_nested_sdfg(sdfg, state, node)

        # Process scopes from the leaves upwards
        self._underapproximate_writes_scope(sdfg, state, state.scope_leaves())

        # Make sure any scalar writes are also added if they have not been processed yet.
        for dn in dnodes:
            desc = sdfg.data(dn.data)
            if isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and desc.total_size == 1):
                for iedge in state.in_edges(dn):
                    if not iedge in self.approximation_dict:
                        self.approximation_dict[iedge] = copy.deepcopy(iedge.data)
                        self.approximation_dict[iedge]._edge = iedge

    def _find_live_iteration_variables(self, nsdfg: nodes.NestedSDFG, sdfg: SDFG, state: SDFGState):
        """
        Helper method that collects all iteration variables of surrounding maps and loops of a
        given nested SDFG and stores them in the global iteration_variables dictionary after
        applying the symbol-mapping of the nested SDFG.
        """

        def symbol_map(mapping, symbol):
            if symbol in mapping:
                return mapping[symbol]
            return None

        map_iteration_variables = _collect_iteration_variables(state, nsdfg)
        sdfg_iteration_variables = self.iteration_variables[sdfg] if sdfg in self.iteration_variables else set()
        state_iteration_variables = self.ranges_per_state[state].keys()
        iteration_variables_local = (map_iteration_variables | sdfg_iteration_variables | state_iteration_variables)
        mapped_iteration_variables = set(map(lambda x: symbol_map(nsdfg.symbol_mapping, x), iteration_variables_local))
        if mapped_iteration_variables:
            self.iteration_variables[nsdfg.sdfg] = mapped_iteration_variables

    def _underapproximate_writes_nested_sdfg(
        self,
        parent_sdfg: SDFG,
        parent_state: SDFGState,
        nsdfg_node: NestedSDFG,
    ):
        """
        Propagate writes out of a nested sdfg. Only considers memlets in states that are
        executed unconditionally. The results are stored in the global approximation_dict

        :param parent_sdfg: The parent SDFG this nested SDFG is in.
        :param parent_state: The state containing this nested SDFG.
        :param nsdfg_node: The NSDFG node containing this nested SDFG.
        """

        def _init_border_memlet(template_memlet: Memlet, node_label: str):
            '''
            Creates a Memlet with the same data as the template_memlet, stores it in the
            border_memlets dictionary and returns it.
            '''
            border_memlet = Memlet(data=template_memlet.data)
            border_memlet._is_data_src = True
            border_memlets[node_label] = border_memlet
            return border_memlet

        # Build a map of connectors to associated 'border' memlets inside
        # the nested SDFG. This map will be populated with memlets once they
        # get propagated in the SDFG.
        border_memlets = {}
        for connector in nsdfg_node.out_connectors:
            border_memlets[connector] = None

        outer_symbols = parent_state.symbols_defined_at(nsdfg_node)
        # For each state, go through all access nodes corresponding to any
        # out-connector from this SDFG. Given those access nodes, collect
        # the corresponding memlets and use them to calculate the
        # subset corresponding to the outside memlet attached to that connector.
        # This is passed out via `border_memlets` and propagated along from there.
        states = _find_unconditionally_executed_states(nsdfg_node.sdfg)
        for state in states:
            for node in state.data_nodes():
                if node.label not in border_memlets:
                    continue
                # Get the edges to this access node
                edges = state.in_edges(node)
                border_memlet = border_memlets[node.label]

                # Collect all memlets belonging to this access node
                memlets = []
                for edge in edges:
                    inside_memlet = self.approximation_dict[edge]
                    memlets.append(inside_memlet)
                    # initialize border memlet if it does not exist already
                    if border_memlet is None:
                        border_memlet = _init_border_memlet(inside_memlet, node.label)

                # Given all of this access nodes' memlets union all the subsets to one SubsetUnion
                if len(memlets) > 0:
                    subset = subsets.SubsetUnion([])
                    for memlet in memlets:
                        subset = subsets.list_union(subset, memlet.subset)
                    # compute the union of the ranges to merge the subsets.
                    border_memlet.subset = _merge_subsets(border_memlet.subset, subset)

            # collect the memlets for each loop in the NSDFG
            if state in self.loop_write_dict:
                for node_label, loop_memlet in self.loop_write_dict[state].items():
                    if node_label not in border_memlets:
                        continue
                    border_memlet = border_memlets[node_label]
                    # initialize border memlet if it does not exist already
                    if border_memlet is None:
                        border_memlet = _init_border_memlet(loop_memlet, node_label)
                    # compute the union of the ranges to merge the subsets.
                    border_memlet.subset = _merge_subsets(border_memlet.subset, loop_memlet.subset)

        # Make sure any potential NSDFG symbol mapping is correctly reversed
        # when propagating out.
        for connector in border_memlets:
            border_memlet = border_memlets[connector]
            if not border_memlet:
                continue
            border_memlet.replace(nsdfg_node.symbol_mapping)
            # filter out subsets that use symbols that are not defined outside of the nsdfg
            _filter_undefined_symbols(border_memlet, outer_symbols)

        # Propagate the inside 'border' memlets outside the SDFG by
        # offsetting, and unsqueezing if necessary.
        for edge in parent_state.out_edges(nsdfg_node):
            out_memlet = self.approximation_dict[edge]
            if edge.src_conn in border_memlets:
                internal_memlet = border_memlets[edge.src_conn]
                if internal_memlet is None:
                    out_memlet.subset = None
                    out_memlet.dst_subset = None
                    self.approximation_dict[edge] = out_memlet
                    continue
                out_memlet = _unsqueeze_memlet_subsetunion(internal_memlet, out_memlet, parent_sdfg, nsdfg_node)
                self.approximation_dict[edge] = out_memlet

    def _underapproximate_writes_loop(self, sdfg: SDFG, loops: Dict[SDFGState,
                                                                    Tuple[SDFGState, SDFGState, List[SDFGState], str,
                                                                          subsets.Range]], loop_header: SDFGState):
        """
        Propagate Memlets recursively out of loop constructs with representative border memlets,
        similar to propagate_memlets_nested_sdfg. Only states that are executed unconditionally
        are considered. Loops containing breaks are ignored. The results are stored in the
        global loop_write_dict.

        :param sdfg: The SDFG the loops are contained in.
        :param loops: dictionary that maps each for-loop construct to a tuple consisting of first
                    state in the loop, the last state in the loop the set of states enclosed by
                    the loop, the itearation variable and the range of the iterator variable
        :param loop_header: a loopheader to start the propagation with. If no parameter is given,
                    propagate_memlet_loop will be called recursively on the outermost loopheaders
        """

        def _init_border_memlet(template_memlet: Memlet, node_label: str):
            '''
            Creates a Memlet with the same data as the template_memlet, stores it in the
            border_memlets dictionary and returns it.
            '''
            border_memlet = Memlet(data=template_memlet.data)
            border_memlet._is_data_src = True
            border_memlets[node_label] = border_memlet
            return border_memlet

        def filter_subsets(itvar: str, itrange: subsets.Range, memlet: Memlet) -> List[subsets.Subset]:
            # helper method that filters out subsets that do not depend on the iteration variable
            # if the iteration range is symbolic

            # if loop range is symbolic
            # -> only propagate subsets that contain the iterator as a symbol
            # if loop range is constant (and not empty, which is already verified)
            # -> always propagate all subsets out
            if memlet.subset is None:
                return []
            result = memlet.subset.subset_list if isinstance(memlet.subset, subsets.SubsetUnion) else [memlet.subset]
            # range contains symbols
            if itrange.free_symbols:
                result = [s for s in result if itvar in s.free_symbols]
            return result

        current_loop = loops[loop_header]
        begin, last_loop_state, loop_states, itvar, rng = current_loop
        if rng.num_elements() == 0:
            return
        # make sure there is no break out of the loop
        dominators = cfg_analysis.all_dominators(sdfg)
        if any(begin not in dominators[s] and not begin is s for s in loop_states):
            return
        border_memlets = defaultdict(None)
        # get all the nodes that are executed unconditionally in the cfg
        # a.k.a nodes that dominate the sink states
        states = dominators[last_loop_state].intersection(set(loop_states))
        states.update([loop_header, last_loop_state])

        for state in states:
            # iterate over the data_nodes that are actually in the current state
            # plus the data_nodes that are overwritten in the corresponding loop body
            # if the state is a loop header
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
                    inside_memlet = copy.copy(self.approximation_dict[edge])
                    # filter out subsets that could become empty depending on assignments
                    # of symbols
                    filtered_subsets = filter_subsets(itvar, rng, inside_memlet)
                    if not filtered_subsets:
                        continue

                    inside_memlet.subset = subsets.SubsetUnion(filtered_subsets)
                    memlets.append(inside_memlet)
                    if border_memlet is None:
                        border_memlet = _init_border_memlet(inside_memlet, node.label)

                self._underapproximate_writes_loop_subset(sdfg, memlets, border_memlet, sdfg.arrays[node.label], itvar,
                                                          rng)

            if state not in self.loop_write_dict:
                continue
            # propagate the border memlets of nested loop
            for node_label, other_border_memlet in self.loop_write_dict[state].items():
                # filter out subsets that could become empty depending on symbol assignments
                filtered_subsets = filter_subsets(itvar, rng, other_border_memlet)
                if not filtered_subsets:
                    continue

                other_border_memlet.subset = subsets.SubsetUnion(filtered_subsets)
                border_memlet = border_memlets.get(node_label)
                if border_memlet is None:
                    border_memlet = _init_border_memlet(other_border_memlet, node_label)

                self._underapproximate_writes_loop_subset(sdfg, [other_border_memlet], border_memlet,
                                                          sdfg.arrays[node_label], itvar, rng)

        self.loop_write_dict[loop_header] = border_memlets

    def _underapproximate_writes_loop_subset(self,
                                             sdfg: dace.SDFG,
                                             memlets: List[Memlet],
                                             dst_memlet: Memlet,
                                             arr: dace.data.Array,
                                             itvar: str,
                                             rng: subsets.Subset,
                                             loop_nest_itvars: Union[Set[str], None] = None):
        """
        Helper function that takes a list of (border) memlets, propagates them out of a
        loop-construct and summarizes them to one Memlet. The result is written back to dst_memlet

        :param sdfg: The SDFG the memlets reside in
        :param memlets: A list of memlets to propagate
        :param arr: The array the memlets write to
        :param itvar: The iteration variable of the loop the memlets are propagated out of
        :param rng: The iteration range of the iteration variable
        :param loop_nest_itvars: A set of iteration variables of surrounding loops
        """
        if not loop_nest_itvars:
            loop_nest_itvars = set()
        if len(memlets) > 0:
            params = [itvar]
            # get all the other iteration variables surrounding this memlet
            surrounding_itvars = self.iteration_variables[sdfg] if sdfg in self.iteration_variables else set()
            if loop_nest_itvars:
                surrounding_itvars |= loop_nest_itvars

            subset = self._underapproximate_subsets(memlets,
                                                    arr,
                                                    params,
                                                    rng,
                                                    use_dst=True,
                                                    surrounding_itvars=surrounding_itvars).subset

            if subset is None or len(subset.subset_list) == 0:
                return
            # compute the union of the ranges to merge the subsets.
            dst_memlet.subset = _merge_subsets(dst_memlet.subset, subset)

    def _underapproximate_writes_scope(self, sdfg: SDFG, state: SDFGState, scopes: Union[scope.ScopeTree,
                                                                                         List[scope.ScopeTree]]):
        """
        Propagate memlets from the given scopes outwards.

        :param sdfg: The SDFG in which the scopes reside.
        :param state: The SDFG state in which the scopes reside.
        :param scopes: The ScopeTree object or a list thereof to start from.
        """

        # for each map scope find the iteration variables of surrounding maps
        surrounding_map_vars: Dict[scope.ScopeTree, Set[str]] = _collect_itvars_scope(scopes)
        if isinstance(scopes, scope.ScopeTree):
            scopes_to_process = [scopes]
        else:
            scopes_to_process = scopes

        # Process scopes from the inputs upwards, propagating edges at the
        # entry and exit nodes
        next_scopes = set()
        while len(scopes_to_process) > 0:
            for scope_node in scopes_to_process:
                if scope_node.entry is None:
                    continue

                surrounding_iteration_variables = self._collect_iteration_variables_scope_node(
                    scope_node, sdfg, state, surrounding_map_vars)
                self._underapproximate_writes_node(state, scope_node.exit, surrounding_iteration_variables)
                # Add parent to next frontier
                next_scopes.add(scope_node.parent)
            scopes_to_process = next_scopes
            next_scopes = set()

    def _collect_iteration_variables_scope_node(self, scope_node: scope.ScopeTree, sdfg: SDFG, state: SDFGState,
                                                surrounding_map_vars: Dict[scope.ScopeTree, Set[str]]) -> Set[str]:
        map_iteration_variables = surrounding_map_vars[scope_node] if scope_node in surrounding_map_vars else set()
        sdfg_iteration_variables = self.iteration_variables[sdfg] if sdfg in self.iteration_variables else set()
        loop_iteration_variables = self.ranges_per_state[state].keys()
        surrounding_iteration_variables = (map_iteration_variables | sdfg_iteration_variables
                                           | loop_iteration_variables)
        return surrounding_iteration_variables

    def _underapproximate_writes_node(self,
                                      dfg_state: SDFGState,
                                      node: Union[nodes.EntryNode, nodes.ExitNode],
                                      surrounding_itvars: Union[Set[str], None] = None):
        """
        Helper method which propagates all memlets attached to a map scope out of the map scope.
        Can be used for both propagation directions. The propagated memlets are stored in the
        global approximation dictonary.

        :param dfg_state: The state the map resides in
        :param node: Either an entry or an exit node of a map scope
        :param surrounding_itvars: Iteration variables that surround the map scope
        """
        if isinstance(node, nodes.EntryNode):
            internal_edges = [e for e in dfg_state.out_edges(node) if e.src_conn and e.src_conn.startswith('OUT_')]
            external_edges = [e for e in dfg_state.in_edges(node) if e.dst_conn and e.dst_conn.startswith('IN_')]

            def geticonn(e):
                return e.src_conn[4:]

            def geteconn(e):
                return e.dst_conn[3:]

            use_dst = False
        else:
            internal_edges = [e for e in dfg_state.in_edges(node) if e.dst_conn and e.dst_conn.startswith('IN_')]
            external_edges = [e for e in dfg_state.out_edges(node) if e.src_conn and e.src_conn.startswith('OUT_')]

            def geticonn(e):
                return e.dst_conn[3:]

            def geteconn(e):
                return e.src_conn[4:]

            use_dst = True

        for edge in external_edges:
            if self.approximation_dict[edge].is_empty():
                new_memlet = Memlet()
            else:
                internal_edge = next(e for e in internal_edges if geticonn(e) == geteconn(edge))
                aligned_memlet = self._align_memlet(dfg_state, internal_edge, dst=use_dst)
                new_memlet = self._underapproximate_memlets(dfg_state,
                                                            aligned_memlet,
                                                            node,
                                                            True,
                                                            connector=geteconn(edge),
                                                            surrounding_itvars=surrounding_itvars)
            new_memlet._edge = edge
            self.approximation_dict[edge] = new_memlet

    def _align_memlet(self, state: SDFGState, edge: gr.MultiConnectorEdge[Memlet], dst: bool) -> Memlet:
        """
        Takes Multiconnectoredge containing Memlet in DFG and swaps subset and other_subset of
        Memlet if it "points" in the wrong direction

        :param state: The state the memlet resides in
        :param edge: The edge containing the memlet that needs to be aligned
        :param dst: True if Memlet should "point" to destination

        :return: Aligned memlet
        """

        is_src = edge.data._is_data_src
        # Memlet is already aligned
        if is_src is None or (is_src and not dst) or (not is_src and dst):
            res = self.approximation_dict[edge]
            return res

        # Data<->Code memlets always have one data container
        mpath = state.memlet_path(edge)
        if not isinstance(mpath[0].src, AccessNode) or not isinstance(mpath[-1].dst, AccessNode):
            return self.approximation_dict[edge]

        # Otherwise, find other data container
        result = copy.deepcopy(self.approximation_dict[edge])
        if dst:
            node = mpath[-1].dst
        else:
            node = mpath[0].src

        # Fix memlet fields
        result.data = node.data
        result.subset = self.approximation_dict[edge].other_subset
        result.other_subset = self.approximation_dict[edge].subset
        result._is_data_src = not is_src
        return result

    def _underapproximate_memlets(self,
                                  dfg_state,
                                  memlet: Memlet,
                                  scope_node: Union[nodes.EntryNode, nodes.ExitNode],
                                  union_inner_edges: bool,
                                  arr: Union[dace.data.Array, None] = None,
                                  connector=None,
                                  surrounding_itvars: Union[Set[str], None] = None):
        """ Tries to underapproximate a memlet through a scope (computes an underapproximation
            of the image of the memlet function applied on an integer set of, e.g., a map range)
            and returns a new memlet object.

            :param dfg_state: An SDFGState object representing the graph.
            :param memlet: The memlet adjacent to the scope node from the inside.
            :param scope_node: A scope entry or exit node.
            :param union_inner_edges: True if the propagation should take other
                                    neighboring internal memlets within the same
                                    scope into account.
        """
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
        defined_vars = {
            symbolic.pystr_to_symbolic(s)
            for s in (dfg_state.symbols_defined_at(entry_node).keys() | sdfg.constants.keys())
            if s not in scope_node_symbols
        }

        # Find other adjacent edges within the connected to the scope node
        # and union their subsets
        if union_inner_edges:
            aggdata = [
                self.approximation_dict[e] for e in neighboring_edges
                if self.approximation_dict[e].data == memlet.data and self.approximation_dict[e] != memlet
            ]
        else:
            aggdata = []

        aggdata.append(memlet)

        if arr is None:
            if memlet.data not in sdfg.arrays:
                raise KeyError('Data descriptor (Array, Stream) "%s" not defined in SDFG.' % memlet.data)

            # FIXME: A memlet alone (without an edge) cannot figure out whether it is data<->data or data<->code
            #        so this test cannot be used
            arr = sdfg.arrays[memlet.data]

        # Propagate subset
        if isinstance(entry_node, nodes.MapEntry):
            mapnode = entry_node.map
            return self._underapproximate_subsets(aggdata,
                                                  arr,
                                                  mapnode.params,
                                                  mapnode.range,
                                                  defined_vars,
                                                  use_dst=use_dst,
                                                  surrounding_itvars=surrounding_itvars)

        elif isinstance(entry_node, nodes.ConsumeEntry):
            # Nothing to analyze/propagate in consume
            new_memlet = copy.copy(memlet)
            new_memlet.subset = subsets.Range.from_array(arr)
            new_memlet.other_subset = None
            return new_memlet
        else:
            raise NotImplementedError('Unimplemented primitive: %s' % type(entry_node))

    def _underapproximate_subsets(self,
                                  memlets: List[Memlet],
                                  arr: data.Data,
                                  params: List[str],
                                  rng: subsets.Subset,
                                  defined_variables: Union[Set[symbolic.SymbolicType], None] = None,
                                  use_dst: bool = False,
                                  surrounding_itvars: Union[Set[str], None] = None) -> Memlet:
        """ Tries to underapproximate a list of memlets through a range (underapproximates
            the image of the memlet function applied on an integer set of, e.g., a
            map range) and returns a new memlet object.

            :param memlets: The memlets to propagate.
            :param arr: Array descriptor for memlet (used for obtaining extents).
            :param params: A list of variable names.
            :param rng: A subset with dimensionality len(params) that contains the
                        range to propagate with.
            :param defined_variables: A set of symbols defined that will remain the
                                    same throughout underapproximation. If None, assumes
                                    that all symbols outside of `params` have been
                                    defined.
            :param use_dst: Whether to underapproximate the memlets' dst subset or use the
                            src instead, depending on propagation direction.
            :param surrounding_itvars:  set of iteration variables that surround the memlet
                                        but are not considered for the underapproximation in
                                        this call
            :return: Memlet with underapproximated subset.
        """
        if not surrounding_itvars:
            surrounding_itvars = set()
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
        variable_context = [[symbolic.pystr_to_symbolic(p) for p in surrounding_itvars], defined_variables,
                            [symbolic.pystr_to_symbolic(p) for p in params]]

        new_subset = None
        for memlet in memlets:
            if memlet.is_empty():
                continue

            _subsets = None
            if use_dst and memlet.dst_subset is not None:
                _subsets = copy.deepcopy(memlet.dst_subset)
            elif not use_dst and memlet.src_subset is not None:
                _subsets = copy.deepcopy(memlet.src_subset)
            else:
                _subsets = copy.deepcopy(memlet.subset)

            if isinstance(_subsets, subsets.SubsetUnion):
                _subsets = _subsets.subset_list
            else:
                _subsets = [_subsets]

            if len(list(set(_subsets) - set([None]))) == 0 or _subsets is None:
                continue

            # iterate over all the subsets in the SubsetUnion of the current memlet and
            # try to apply a memletpattern. If no pattern matches fall back to the empty set
            for i, subset in enumerate(_subsets):
                # find a pattern for the current subset
                for pclass in UnderapproximationMemletPattern.extensions():
                    pattern = pclass()
                    if pattern.can_be_applied([subset], variable_context, rng, [memlet]):
                        subset = pattern.propagate(arr, [subset], rng)
                        break
                else:
                    # No patterns found. Underapproximate the subset with an empty subset (so None)
                    subset = None
                _subsets[i] = subset

            # Union edges as necessary
            if new_subset is None:
                new_subset = subsets.SubsetUnion(_subsets)
            else:
                old_subset = new_subset
                new_subset = subsets.list_union(new_subset, subsets.SubsetUnion(_subsets))
                if new_subset is None:
                    warnings.warn('Subset union failed between %s and %s ' % (old_subset, _subsets))
                    break

        # Create new memlet
        new_memlet = copy.copy(memlets[0])
        new_memlet.subset = new_subset
        new_memlet.other_subset = None
        return new_memlet
