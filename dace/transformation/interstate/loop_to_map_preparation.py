# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import ast

import sympy

from dace import properties
from dace import sdfg as sd
from dace import symbolic
from dace.frontend.python import astutils
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


class ASTFinReplaceFuncAccess(ast.NodeTransformer):

    def __init__(self, access: str, target_symbol: str):
        self.access = access
        self.target_symbol = target_symbol
        self.found_expression = None

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == self.access:
            self.found_expression = astutils.unparse(node)
            new_node = ast.Name(self.target_symbol, node.ctx)
            new_node = ast.copy_location(new_node, node)
            return new_node
        return self.generic_visit(node)


_builtin_userfunctions = {
    'int_floor', 'int_ceil', 'abs', 'Abs', 'min', 'Min', 'max', 'Max', 'not', 'Not', 'Eq', 'NotEq', 'Ne', 'AND', 'OR',
    'pow', 'round'
}

def get_data_access(expr):
    """ Returns True if expression contains Sympy functions. """
    if symbolic.is_sympy_userfunction(expr):
        if str(expr.func) in _builtin_userfunctions:
            return None
        return [str(expr.func)]
    if not isinstance(expr, sympy.Basic):
        return None
    retval = None
    for arg in expr.args:
        rval = get_data_access(arg)
        if rval is not None:
            if retval is not None:
                retval.extend(rval)
            else:
                retval = rval
    return retval


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToMapPeparation(xf.MultiStateTransformation):

    loop = xf.PatternNode(LoopRegion)

    _data_accesses = {
        'start': None,
        'end': None,
    }

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive = False):
        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        if symbolic.contains_sympy_functions(step):
            return False
        found_something = False
        if symbolic.contains_sympy_functions(start):
            data_access = get_data_access(start)
            if data_access is None or len(data_access) > 1 or data_access[0] not in sdfg.arrays:
                return False
            else:
                self._data_accesses['start'] = data_access[0]
                found_something = True
        if symbolic.contains_sympy_functions(end):
            data_access = get_data_access(end)
            if data_access is None or len(data_access) > 1 or data_access[0] not in sdfg.arrays:
                return False
            else:
                self._data_accesses['end'] = data_access[0]
                found_something = True
        return found_something

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        for tgt in ['start', 'end']:
            if self._data_accesses[tgt]:
                tgt_symbol = '__' + self.loop.label + '_ReadOnly_' + tgt + '_' + self._data_accesses[tgt]
                code_block = self.loop.init_statement if tgt == 'start' else self.loop.loop_condition
                for stmt in code_block.code:
                    afr = ASTFinReplaceFuncAccess(self._data_accesses[tgt], tgt_symbol)
                    afr.visit(stmt)
                    if afr.found_expression:
                        if graph.start_block is self.loop:
                            graph.add_state_before(self.loop)
                        for iedge in graph.in_edges(self.loop):
                            iedge.data.assignments[tgt_symbol] = afr.found_expression
        