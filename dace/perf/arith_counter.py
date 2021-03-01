import ast
import astunparse
import dace
from dace.sdfg.graph import SubgraphView
from dace.sdfg.scope import ScopeTree as Scope
from dace.symbolic import pystr_to_symbolic
from dace.libraries.blas import MatMul, Transpose
import sympy
import sys
from typing import Any, Dict, List, Union

INDENT = 0
iprint = lambda *args: print('  ' * INDENT, *args)


def count_matmul(node: MatMul, symbols: Dict[str, Any],
                 state: dace.SDFGState) -> int:
    A_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_a')
    B_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_b')
    C_memlet = next(e for e in state.out_edges(node) if e.src_conn == '_c')
    result = 2  # Multiply, add
    # Batch
    if len(C_memlet.data.subset) == 3:
        result *= symeval(C_memlet.data.subset.size()[0], symbols)
    # M*N
    result *= symeval(C_memlet.data.subset.size()[-2], symbols)
    result *= symeval(C_memlet.data.subset.size()[-1], symbols)
    # K
    result *= symeval(A_memlet.data.subset.size()[-1], symbols)
    return result


# Do not use O(x) or Order(x) in sympy, it's not working as intended
bigo = sympy.Function('bigo')

# Mapping between library node types and arithmetic operations
LIBNODES_TO_ARITHMETICS = {MatMul: count_matmul, Transpose: lambda *args: 0}

# Mapping between python functions and their arithmetic operations
PYFUNC_TO_ARITHMETICS = {
    'math.sin': 1,
    'math.cos': 1,
    'math.exp': 1,
    'math.tanh': 1,
    'math.sqrt': 1,
    'min': 0,
    'max': 0
}


def count_arithmetic_ops(sdfg: dace.SDFG,
                         symbols: Dict[str, Any] = None) -> int:
    result = 0
    symbols = symbols or {}
    for state in sdfg.nodes():
        result += count_arithmetic_ops_state(state, symbols)
    return result


def symeval(val: Any, symbols: Dict[str, Any]):
    # Replace in two steps to avoid double-replacement
    rep1 = {
        pystr_to_symbolic(kk): pystr_to_symbolic('__REPLSYM_' + kk)
        for kk in symbols.keys()
    }
    rep2 = {
        pystr_to_symbolic('__REPLSYM_' + kk): vv
        for kk, vv in symbols.items()
    }
    return val.subs(rep1).subs(rep2)


def evalsyms(base: Dict[str, Any], new: Dict[str, Any]):
    result = {}
    for k, v, in new.items():
        result[k] = symeval(v, base)
    return result


def count_arithmetic_ops_state(state: dace.SDFGState,
                               symbols: Dict[str, Any] = None) -> int:
    global INDENT
    symbols = symbols or {}
    stree_root = state.scope_tree()[None]
    sdict = state.scope_children()
    result = 0

    def traverse(scope: Scope) -> int:
        global INDENT
        result = 0
        repetitions = 1
        if scope.entry is not None:
            repetitions = scope.entry.map.range.num_elements()

        for node in sdict[scope.entry]:
            node_result = 0
            if isinstance(node, dace.nodes.NestedSDFG):
                # Use already known symbols + symbols from mapping
                nested_syms = {}
                nested_syms.update(symbols)
                nested_syms.update(evalsyms(symbols, node.symbol_mapping))

                #iprint('SUBSDFG', node.sdfg.name, nested_syms)
                INDENT += 1
                node_result += count_arithmetic_ops(node.sdfg, nested_syms)
                INDENT -= 1

            elif isinstance(node, dace.nodes.Tasklet):
                if node.code.language == dace.Language.CPP:
                    for oedge in state.out_edges(node):
                        node_result += bigo(oedge.data.num_accesses)
                else:
                    node_result += count_arithmetic_ops_code(
                        node.code.code)
            elif isinstance(node, dace.libraries.standard.nodes.reduce.Reduce):
                node_result += state.in_edges(node)[
                    0].data.subset.num_elements() * count_arithmetic_ops_code(
                        node.wcr)
            elif isinstance(node, dace.nodes.LibraryNode):
                node_result += LIBNODES_TO_ARITHMETICS[type(node)](node,
                                                                   symbols,
                                                                   state)
            # Augment list by WCR edges
            if isinstance(node, (dace.nodes.CodeNode, dace.nodes.AccessNode)):
                for oedge in state.out_edges(node):
                    if oedge.data.wcr is not None:
                        node_result += count_arithmetic_ops_code(
                            oedge.data.wcr)
            if node_result != 0 and INDENT <= 1:
                if isinstance(node, dace.nodes.NestedSDFG) and INDENT == 0:
                    iprint('*',
                           type(node).__name__, node.sdfg.name, ': `',
                           node_result, '`')
                elif isinstance(node, dace.nodes.Tasklet) and INDENT == 1:
                    iprint('*',
                           type(node).__name__, node, ': `', node_result, '`')
            result += node_result

        # Add children scopes
        for child in scope.children:
            if INDENT == 0:
                iprint('Scope', child.entry)
            INDENT += 1
            result += traverse(child)
            INDENT -= 1

        return repetitions * result

    # Traverse from root scope
    return traverse(stree_root)


def count_arithmetic_ops_subgraph(subgraph: SubgraphView, state: dace.SDFGState,
                               symbols: Dict[str, Any] = None) -> int:
    global INDENT
    stree_root = state.scope_tree()[None]
    sdict = state.scope_children()
    result = 0
    symbols = symbols or {}


    def traverse(scope: Scope) -> int:
        global INDENT
        result = 0
        repetitions = 1
        if scope.entry is not None:
            repetitions = scope.entry.map.range.num_elements()

        for node in sdict[scope.entry]:
            if node in subgraph:
                node_result = 0
                if isinstance(node, dace.nodes.NestedSDFG):
                    # Use already known symbols + symbols from mapping
                    nested_syms = {}
                    nested_syms.update(symbols)
                    nested_syms.update(evalsyms(symbols, node.symbol_mapping))

                    #iprint('SUBSDFG', node.sdfg.name, nested_syms)
                    INDENT += 1
                    # use regular count_arithmetic_ops
                    node_result += count_arithmetic_ops(node.sdfg, nested_syms)
                    INDENT -= 1
                elif isinstance(node, dace.nodes.LibraryNode):
                    node_result += LIBNODES_TO_ARITHMETICS[type(node)](node,
                                                                       symbols,
                                                                       state)
                elif isinstance(node, dace.nodes.Tasklet):
                    if node._code['language'] == dace.Language.CPP:
                        for oedge in state.out_edges(node):
                            node_result += bigo(oedge.data.num_accesses)
                    else:
                        node_result += count_arithmetic_ops_code(
                            node._code['code_or_block'])
                elif isinstance(node, dace.libraries.standard.nodes.reduce.Reduce):
                    node_result += state.in_edges(node)[
                        0].data.subset.num_elements() * count_arithmetic_ops_code(
                            node.wcr)

                # Augment list by WCR edges
                if isinstance(node, (dace.nodes.CodeNode, dace.nodes.AccessNode)):
                    for oedge in state.out_edges(node):
                        if oedge.data.wcr is not None:
                            node_result += count_arithmetic_ops_code(
                                oedge.data.wcr)
                if node_result != 0 and INDENT <= 1:
                    if isinstance(node, dace.nodes.NestedSDFG) and INDENT == 0:
                        iprint('*',
                               type(node).__name__, node.sdfg.name, ': `',
                               node_result, '`')
                    elif isinstance(node, dace.nodes.Tasklet) and INDENT == 1:
                        iprint('*',
                               type(node).__name__, node, ': `', node_result, '`')
                result += node_result

        # Add children scopes
        for child in scope.children:
            if INDENT == 0:
                iprint('Scope', child.entry)
            INDENT += 1
            result += traverse(child)
            INDENT -= 1

        return repetitions * result

    # Traverse from root scope
    return traverse(stree_root)



def count_arithmetic_ops_code(
        code_or_block: Union[List[ast.AST], ast.AST, str]) -> int:
    ctr = ArithmeticCounter()
    if isinstance(code_or_block, (tuple, list)):
        for stmt in code_or_block:
            ctr.visit(stmt)
    elif isinstance(code_or_block, str):
        ctr.visit(ast.parse(code_or_block))
    else:
        ctr.visit(code_or_block)

    return ctr.count


class ArithmeticCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, ast.MatMult):
            raise NotImplementedError('MatMult op count requires shape '
                                      'inference')
        self.count += 1
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.count += 1
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        fname = astunparse.unparse(node.func)[:-1]
        if fname not in PYFUNC_TO_ARITHMETICS:
            print('WARNING: Unrecognized python function "%s"' % fname)
            return self.generic_visit(node)
        self.count += PYFUNC_TO_ARITHMETICS[fname]
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        return self.visit_BinOp(node)

    def visit_For(self, node: ast.For):
        raise NotImplementedError

    def visit_While(self, node: ast.While):
        raise NotImplementedError


def test_counter():
    ctr = ArithmeticCounter()
    ctr.visit(ast.parse('out = -inp'))
    assert ctr.count == 1
    ctr.count = 0
    ctr.visit(ast.parse('out = -inp + 5'))
    assert ctr.count == 2
    ctr.count = 0
    ctr.visit(ast.parse('out = math.exp(inp) * inp + a + b'))
    assert ctr.count == 4
    ctr.count = 0
    ctr.visit(ast.parse('''
out = -b * 5
out2 += a * b
    '''))
    assert ctr.count == 4
    ctr.count = 0
    ctr.visit(ast.parse('lambda a,b: a+b'))
    assert ctr.count == 1
    ctr.count = 0
    ctr.visit(ast.parse('lambda a,b: b if a > b else a'))
    assert ctr.count == 0
    ctr.count = 0


if __name__ == '__main__':
    test_counter()
    if len(sys.argv) != 2:
        print('USAGE: %s <SDFG FILE>' % sys.argv[0])
        exit(1)

    sdfg = dace.SDFG.from_file(sys.argv[1])
    ops = count_arithmetic_ops(sdfg)
    print('Total floating-point arithmetic operations', ops)
