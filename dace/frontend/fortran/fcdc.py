#fparser imports
from dataclasses import astuple

from venv import create
from fparser.api import parse
from fparser.two.Fortran2003 import *
from fparser.two.Fortran2008 import *
from fparser.two.parser import *
from fparser.two.utils import *
from fparser.two.symbol_table import *
import os
from fparser.common.readfortran import FortranStringReader, FortranFileReader

#dace imports
import dace
from dace.sdfg import *
from dace.data import Scalar
from dace.sdfg import SDFG
from dace.sdfg.nodes import Tasklet
from dace import dtypes
from dace.properties import CodeBlock

import ast_components
from ast_components import *

from typing import List, Tuple, Set


def add_tasklet(substate: SDFGState, name: str, vars_in: Set[str],
                vars_out: Set[str], code: str, debuginfo: list, source: str):
    tasklet = substate.add_tasklet(name="T" + name,
                                   inputs=vars_in,
                                   outputs=vars_out,
                                   code=code,
                                   debuginfo=dace.DebugInfo(
                                       start_line=debuginfo[0],
                                       start_column=debuginfo[1],
                                       filename=source),
                                   language=dace.Language.Python)
    return tasklet


def add_memlet_read(substate: SDFGState, var_name: str, tasklet: Tasklet,
                    dest_conn: str, memlet_range: str):
    src = substate.add_access(var_name)
    if memlet_range != "":
        substate.add_memlet_path(src,
                                 tasklet,
                                 dst_conn=dest_conn,
                                 memlet=dace.Memlet(expr=var_name,
                                                    subset=memlet_range))
    else:
        substate.add_memlet_path(src,
                                 tasklet,
                                 dst_conn=dest_conn,
                                 memlet=dace.Memlet(expr=var_name))


def add_memlet_write(substate: SDFGState, var_name: str, tasklet: Tasklet,
                     source_conn: str, memlet_range: str):
    dst = substate.add_write(var_name)
    if memlet_range != "":
        substate.add_memlet_path(tasklet,
                                 dst,
                                 src_conn=source_conn,
                                 memlet=dace.Memlet(expr=var_name,
                                                    subset=memlet_range))
    else:
        substate.add_memlet_path(tasklet,
                                 dst,
                                 src_conn=source_conn,
                                 memlet=dace.Memlet(expr=var_name))


def add_simple_state_to_sdfg(state: SDFGState, top_sdfg: SDFG,
                             state_name: str):
    if state.last_sdfg_states.get(top_sdfg) is not None:
        substate = top_sdfg.add_state(state_name)
    else:
        substate = top_sdfg.add_state(state_name, is_start_state=True)
    finish_add_state_to_sdfg(state, top_sdfg, substate)
    return substate


def finish_add_state_to_sdfg(state: SDFGState, top_sdfg: SDFG,
                             substate: SDFGState):
    if state.last_sdfg_states.get(top_sdfg) is not None:
        top_sdfg.add_edge(state.last_sdfg_states[top_sdfg], substate,
                          dace.InterstateEdge())
    state.last_sdfg_states[top_sdfg] = substate


class TaskletWriter:
    def __init__(self, outputs: List[str], outputs_changes: List[str]):
        self.outputs = outputs
        self.outputs_changes = outputs_changes

        self.ast_elements = {
            BinOp_Node: self.binop2string,
            Name_Node: self.name2string,
            Int_Literal_Node: self.intlit2string,
            Real_Literal_Node: self.floatlit2string,
            UnOp_Node: self.unop2string,
            Array_Subscript_Node: self.arraysub2string,
        }

    def write_tasklet_code(self, node: Node):
        if node.__class__ in self.ast_elements:
            text = self.ast_elements[node.__class__](node)
            #print("RET TW:",text)
            #    text = text.replace("][", ",")
            return text
        else:

            print("ERROR:", node.__class__.__name__)

    def arraysub2string(self, node: Array_Subscript_Node):
        str_to_return = self.write_tasklet_code(
            node.name) + "[" + self.write_tasklet_code(node.indices[0])
        for i in node.indices[1:]:
            str_to_return += ", " + self.write_tasklet_code(i) + "]"
        return str_to_return

    def name2string(self, node: Name_Node):
        return_value = node.name

        if len(self.outputs) > 0:
            #print("TASK WRITER:",node.name,self.outputs[0],self.outputs_changes[0])
            if node.name == self.outputs[0]:
                if self.outputs[0] != self.outputs_changes[0]:
                    return_value = self.outputs_changes[0]
                self.outputs.pop(0)
                self.outputs_changes.pop(0)
            #print("RETURN VALUE:",return_value)
        return str(return_value)

    def intlit2string(self, node: Int_Literal_Node):

        return "".join(map(str, node.value))

    def floatlit2string(self, node: Real_Literal_Node):

        return "".join(map(str, node.value))

    def unop2string(self, node: UnOp_Node):
        op = node.op
        if op == ".NOT.":
            op = "not "
        return op + self.write_tasklet_code(node.lval)

    def binop2string(self, node: BinOp_Node):
        #print("BL: ",self.write_tasklet_code(node.lvalue))
        #print("RL: ",self.write_tasklet_code(node.rvalue))
        # print(node.op)
        op = node.op
        if op == ".EQ.":
            op = "=="
        if op == ".AND.":
            op = " and "
        if op == ".OR.":
            op = " or "
        if op == ".NE.":
            op = "!="
        if op == "/=":
            op = "!="
        if op == ".NOT.":
            op = "!"
        if op == ".LE.":
            op = "<="
        if op == ".GE.":
            op = ">="
        if op == ".LT.":
            op = "<"
        if op == ".GT.":
            op = ">"
        # if op == "&&":
        #    op=" and "
        # if self.write_tasklet_code(node.lvalue) is None:
        #    a=1
        # if self.write_tasklet_code(node.rvalue) is None:
        #    a=1
        return self.write_tasklet_code(
            node.lval) + op + self.write_tasklet_code(node.rval)


def generate_memlet(op, top_sdfg, state):
    if state.name_mapping.get(top_sdfg).get(op.name) is not None:
        shape = top_sdfg.arrays[state.name_mapping[top_sdfg][op.name]].shape
    elif state.name_mapping.get(state.globalsdfg).get(op.name) is not None:
        shape = state.globalsdfg.arrays[state.name_mapping[state.globalsdfg][
            op.name]].shape
    else:
        raise NameError("Variable name not found: ", op.name)
    # print("SHAPE:")
    # print(shape)
    tmp_node = op
    indices = []
    while isinstance(tmp_node, Array_Subscript_Node):
        if isinstance(tmp_node.index, Name_Node):
            indices.append(state.name_mapping[top_sdfg][tmp_node.index.name])
        elif isinstance(tmp_node.index, Int_Literal_Node):
            indices.append("".join(map(str, tmp_node.index.value)))
        tmp_node = tmp_node.unprocessed_name
    # for i in indices:
    # print("INDICES:",i)
    memlet = '0'
    if len(shape) == 1:
        if shape[0] == 1:
            return memlet
    from dace import subsets
    all_indices = indices + [None] * (len(shape) - len(indices))
    subset = subsets.Range([(i, i, 1) if i is not None else (1, s, 1)
                            for i, s in zip(all_indices, shape)])
    return subset


class Context:
    def __init__(self, name):
        self.name = name
        self.constants = {}
        self.symbols = []
        self.containers = []
        self.read_vars = []
        self.written_vars = []


class NameMap(dict):
    def __getitem__(self, k):
        assert isinstance(k, SDFG)
        if k not in self:
            self[k] = {}

        return super().__getitem__(k)

    def get(self, k):
        return self[k]

    def __setitem__(self, k, v) -> None:
        assert isinstance(k, SDFG)
        return super().__setitem__(k, v)


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    if not hasattr(node, "_fields"):
        a = 1
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    #print("CLASS: ",node.__class__)
    #if isinstance(node,DeclRefExpr):
    #print("NAME: ", node.name)

    for name, field in iter_fields(node):
        #print("NASME:",name)
        if isinstance(field, Node):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, Node):
                    yield item


class NodeVisitor(object):
    def visit(self, node):
        # print(node.__class__.__name__)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Node):
                        self.visit(item)
            elif isinstance(value, Node):
                self.visit(value)


class NodeTransformer(NodeVisitor):
    """
    A :class:`NodeVisitor` subclass that walks the abstract syntax tree and
    allows modification of nodes.

    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace or remove the old node.  If the return value of
    the visitor method is ``None``, the node will be removed from its location,
    otherwise it is replaced with the return value.  The return value may be the
    original node in which case no replacement takes place.

    Here is an example transformer that rewrites all occurrences of name lookups
    (``foo``) to ``data['foo']``::

       class RewriteName(NodeTransformer):

           def visit_Name(self, node):
               return copy_location(Subscript(
                   value=Name(id='data', ctx=Load()),
                   slice=Index(value=Str(s=node.id)),
                   ctx=node.ctx
               ), node)

    Keep in mind that if the node you're operating on has child nodes you must
    either transform the child nodes yourself or call the :meth:`generic_visit`
    method for the node first.

    For nodes that were part of a collection of statements (that applies to all
    statement nodes), the visitor may also return a list of nodes rather than
    just a single node.

    Usually you use the transformer like this::

       node = YourTransformer().visit(node)
    """
    def as_list(self, x):
        if isinstance(x, list):
            return x
        if x is None:
            return []
        return [x]

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, Node):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, Node):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class FindInputNodesVisitor(NodeVisitor):
    def __init__(self):
        self.nodes: List[Name_Node] = []

    def visit_Name_Node(self, node: Name_Node):
        self.nodes.append(node)

    def visit_BinOp_Node(self, node: BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, Name_Node):
                pass

        else:
            self.visit(node.lval)
        self.visit(node.rval)


class FindOutputNodesVisitor(NodeVisitor):
    def __init__(self):
        self.nodes: List[Name_Node] = []

    def visit_BinOp_Node(self, node: BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, Name_Node):
                self.nodes.append(node.lval)
            elif isinstance(node.lval, Array_Subscript_Node):
                self.nodes.append(node.lval.name)
            self.visit(node.rval)


# TODO rewrite this
class CallToArray(NodeTransformer):
    def __init__(self, funcs=[]):
        self.funcs = funcs

    def visit_Call_Expr_Node(self, node: Call_Expr_Node):

        self.count = self.count + 1 if hasattr(self, "count") else 0
        tmp = self.count
        if node.name in [
                "malloc", "exp", "pow", "sqrt", "cbrt", "max", "abs", "min",
                "dace_sum", "dace_sign", "tanh"
        ]:
            args2 = []
            for i in node.args:
                arg = CallToArray(self.funcs).visit(i)
                args2.append(arg)
            node.args = args2
            return node
        if node.name in self.funcs:
            args2 = []
            if hasattr(node, "args"):
                for i in node.args:
                    arg = CallToArray(self.funcs).visit(i)
                    args2.append(arg)
            node.args = args2
            return node
        indices = [CallToArray(self.funcs).visit(i) for i in node.args]
        return Array_Subscript_Node(name=node.name, indices=indices)


class AST_translator:
    def __init__(self, ast: InternalFortranAst, source):
        self.tables = ast.tables
        self.name_mapping = NameMap()
        self.contexts = {}
        self.file_name = source
        self.all_array_names = []
        self.last_sdfg_states = {}
        self.ast_elements = {
            #WhileStmt: self.while2sdfg,
            #DoStmt: self.do2sdfg,
            #RetStmt: self.ret2sdfg,
            #IfStmt: self.ifstmt2sdfg,
            #ForStmt: self.forstmt2sdfg,
            #BasicBlock: self.basicblock2sdfg,
            #FunctionSubprogram: self.funcdecl2sdfg,
            #SubroutineSubprogram: self.subroutine2sdfg,
            BinOp_Node: self.binop2sdfg,
            Decl_Stmt_Node: self.declstmt2sdfg,
            Var_Decl_Node: self.vardecl2sdfg,
            #Constant_Decl_Node: self.const2sdfg,
            #Parm_Decl_Node: self.parmdecl2sdfg,
            #Type_Decl_Node: self.typedecl2sdfg,
            #Call_Expr_Node: self.call2sdfg,
            #AllocList: self.alloclist2sdfg,
            #ContinueStmt: self.cont2sdfg,
            #GotoStmt: self.goto2sdfg,
            Program_Node: self.ast2sdfg
        }
        self.fortrantypes2dacetypes = {
            "DOUBLE": dace.float64,
            "REAL": dace.float32,
            "INTEGER": dace.int32,
            "BOOL": dace.int32,
        }

    def get_dace_type(self, type):
        if isinstance(type, str):
            return self.fortrantypes2dacetypes[type]

    def get_name_mapping_in_context(self, sdfg: SDFG):
        a = self.name_mapping[self.globalsdfg].copy()
        if sdfg is not self.globalsdfg:
            a.update(self.name_mapping[sdfg])
        return a

    def get_arrays_in_context(self, sdfg: SDFG):
        a = self.globalsdfg.arrays.copy()
        if sdfg is not self.globalsdfg:
            a.update(sdfg.arrays)
        return a

    def get_memlet_range(self, sdfg: SDFG, variables: List[Node],
                         var_name: str, var_name_tasklet: str) -> str:
        var = self.get_arrays_in_context(sdfg).get(var_name)

        if len(var.shape) == 0:
            return ""

        if (len(var.shape) == 1 and var.shape[0] == 1):
            return "0"

        for o_v in variables:
            if o_v.name == var_name_tasklet:
                return generate_memlet(o_v, sdfg, self)

    def translate(self, node: Node, sdfg: SDFG):
        if node.__class__ in self.ast_elements:
            self.ast_elements[node.__class__](node, sdfg)
        elif isinstance(node, list):
            for i in node:
                self.translate(i, sdfg)
        else:
            print("WARNING:", node.__class__.__name__)

    def ast2sdfg(self, node: Program_Node, sdfg: SDFG):
        self.globalsdfg = sdfg
        for i in node.main_program.specification_part.typedecls:
            self.translate(i, sdfg)
        for i in node.main_program.specification_part.symbols:
            self.translate(i, sdfg)
        for i in node.main_program.specification_part.specifications:
            self.translate(i, sdfg)
        self.translate(node.main_program.execution_part.execution, sdfg)

    #TODO REWRITE THIS nicely
    def binop2sdfg(self, node: BinOp_Node, sdfg: SDFG):
        print(node)

        outputnodefinder = FindOutputNodesVisitor()
        outputnodefinder.visit(node)
        output_vars = outputnodefinder.nodes
        output_names = []
        output_names_tasklet = []

        for i in output_vars:
            mapped_name = self.get_name_mapping_in_context(sdfg).get(i.name)
            arrays = self.get_arrays_in_context(sdfg)

            if mapped_name in arrays and mapped_name not in output_names:
                output_names.append(mapped_name)
                output_names_tasklet.append(i.name)

        inputnodefinder = FindInputNodesVisitor()
        inputnodefinder.visit(node)
        input_vars = inputnodefinder.nodes
        input_names = []
        input_names_tasklet = []

        for i in input_vars:
            mapped_name = self.get_name_mapping_in_context(sdfg).get(i.name)
            arrays = self.get_arrays_in_context(sdfg)
            if i.name in sdfg.symbols:
                continue
            if mapped_name in arrays and mapped_name not in input_names:
                input_names.append(mapped_name)
                input_names_tasklet.append(i.name)

        substate = add_simple_state_to_sdfg(
            self, sdfg, "_state_l" + str(node.line_number[0]) + "_c" +
            str(node.line_number[1]))

        output_names_changed = [o_t + "_out" for o_t in output_names_tasklet]

        #output_names_dict = {on: dace.pointer(dace.int32) for on in output_names_changed}

        tasklet = add_tasklet(
            substate,
            "_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]),
            input_names_tasklet, output_names_changed, "text",
            node.line_number, self.file_name)

        for i, j in zip(input_names, input_names_tasklet):
            memlet_range = self.get_memlet_range(sdfg, input_vars, i, j)
            add_memlet_read(substate, i, tasklet, j, memlet_range)

        for i, j, k in zip(output_names, output_names_tasklet,
                           output_names_changed):

            memlet_range = self.get_memlet_range(sdfg, output_vars, i, j)
            add_memlet_write(substate, i, tasklet, k, memlet_range)

        tw = TaskletWriter(output_names_tasklet, output_names_changed)
        # print("BINOP:",output_names,output_names_tasklet,output_names_changed)
        text = tw.write_tasklet_code(node)
        # print("BINOPTASKLET:",text)
        tasklet.code = CodeBlock(text, dace.Language.Python)

    def declstmt2sdfg(self, node: Decl_Stmt_Node, sdfg: SDFG):
        for i in node.vardecl:
            self.translate(i, sdfg)

    def vardecl2sdfg(self, node: Var_Decl_Node, sdfg: SDFG):
        #if the sdfg is the toplevel-sdfg, the variable is a global variable
        transient = sdfg is not self.globalsdfg
        # find the type
        datatype = self.get_dace_type(node.type)
        # get the dimensions
        if node.sizes is not None:
            sizes = []
            offset = []
            offset_value = -1
            for i in node.sizes:
                tw = TaskletWriter([], [])
                text = tw.write_tasklet_code(i)
                sizes.append(dace.symbolic.pystr_to_symbolic(text))
                offset.append(offset_value)

        else:
            sizes = None
        # create and check name
        if self.name_mapping[sdfg].get(node.name) is not None:
            raise ValueError("Name already defined in this scope")
        if node.name in sdfg.symbols:
            raise ValueError("Name already defined as symbol")
        self.name_mapping[sdfg][node.name] = sdfg._find_new_name(node.name)

        if sizes is None:
            sdfg.add_scalar(self.name_mapping[sdfg][node.name],
                            dtype=datatype,
                            transient=transient)
        else:
            strides = [dace.data._prod(sizes[:i]) for i in range(len(sizes))]
            sdfg.add_array(self.name_mapping[sdfg][node.name],
                           shape=sizes,
                           dtype=datatype,
                           offset=offset,
                           strides=strides,
                           transient=transient)
        #This might no longer be necessary
        self.all_array_names.append(self.name_mapping[sdfg][node.name])
        if self.contexts.get(sdfg.name) is None:
            self.contexts[sdfg.name] = Context(name=sdfg.name)
        if node.name not in self.contexts[sdfg.name].containers:
            self.contexts[sdfg.name].containers.append(node.name)


if __name__ == "__main__":
    parser = ParserFactory().create(std="f2008")
    testname = "loop3"
    reader = FortranFileReader(
        os.path.realpath("/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" +
                         testname + ".f90"))
    ast = parser(reader)
    tables = SYMBOL_TABLES
    table = tables.lookup("CLOUDPROGRAM")
    own_ast = InternalFortranAst(ast, tables)
    #own_ast.list_tables()
    program = own_ast.create_ast(ast)
    fd = []
    program = CallToArray(fd).visit(program)
    Ast2Sdfg = AST_translator(
        own_ast,
        "/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" + testname + ".f90")
    sdfg = SDFG("top_level")
    Ast2Sdfg.translate(program, sdfg)

    sdfg.validate()
    sdfg.compile()
    # node_list = walk(ast)
    # node_types = []
    # for i in node_list:
    #     if type(i).__name__ not in node_types and i is not None and type(
    #             i) != type("string"):
    #         if type(i) == tuple:
    #             print(i)
    #         node_types.append(type(i).__name__)

    #print(ast)
