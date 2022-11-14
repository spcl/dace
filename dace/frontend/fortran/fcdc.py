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
from ast_trasforms import *
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


def get_name(node: Node):
    if isinstance(node, Name_Node):
        return node.name
    elif isinstance(node, Array_Subscript_Node):
        return node.name.name
    else:
        raise NameError("Name not found")


class TaskletWriter:
    def __init__(self, outputs: List[str], outputs_changes: List[str]):
        self.outputs = outputs
        self.outputs_changes = outputs_changes

        self.ast_elements = {
            BinOp_Node: self.binop2string,
            Name_Node: self.name2string,
            Name_Range_Node: self.name2string,
            Int_Literal_Node: self.intlit2string,
            Real_Literal_Node: self.floatlit2string,
            Bool_Literal_Node: self.boollit2string,
            UnOp_Node: self.unop2string,
            Array_Subscript_Node: self.arraysub2string,
            Parenthesis_Expr_Node: self.parenthesis2string,
            Call_Expr_Node: self.call2string,
            ParDecl_Node: self.pardecl2string,
        }

    def pardecl2string(self, node: ParDecl_Node):
        return "ERROR" + node.type

    def write_code(self, node: Node):
        if node.__class__ in self.ast_elements:
            text = self.ast_elements[node.__class__](node)
            if text is None:
                raise NameError("Error in code generation")
            #print("RET TW:",text)
            #    text = text.replace("][", ",")
            return text
        elif isinstance(node, str):
            return node
        else:

            print("ERROR:", node.__class__.__name__)

    def arraysub2string(self, node: Array_Subscript_Node):
        str_to_return = self.write_code(node.name) + "[" + self.write_code(
            node.indices[0])
        for i in node.indices[1:]:
            str_to_return += ", " + self.write_code(i)
        str_to_return += "]"
        return str_to_return

    def name2string(self, node):
        if isinstance(node, str):
            return node

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

    def boollit2string(self, node: Bool_Literal_Node):

        return str(node.value)

    def unop2string(self, node: UnOp_Node):
        op = node.op
        if op == ".NOT.":
            op = "not "
        return op + self.write_code(node.lval)

    def parenthesis2string(self, node: Parenthesis_Expr_Node):
        return "(" + self.write_code(node.expr) + ")"

    def call2string(self, node: Call_Expr_Node):
        if node.name.name == "dace_epsilon":
            return str(sys.float_info.min)
        return_str = self.write_code(node.name) + "(" + self.write_code(
            node.args[0])
        for i in node.args[1:]:
            return_str += ", " + self.write_code(i)
        return_str += ")"
        return return_str

    def binop2string(self, node: BinOp_Node):
        #print("BL: ",self.write_code(node.lvalue))
        #print("RL: ",self.write_code(node.rvalue))
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
        # if self.write_code(node.lvalue) is None:
        #    a=1
        # if self.write_code(node.rvalue) is None:
        #    a=1
        left = self.write_code(node.lval)
        right = self.write_code(node.rval)
        return left + op + right


def generate_memlet(op, top_sdfg, state):
    if state.name_mapping.get(top_sdfg).get(get_name(op)) is not None:
        shape = top_sdfg.arrays[state.name_mapping[top_sdfg][get_name(
            op)]].shape
    elif state.name_mapping.get(state.globalsdfg).get(
            get_name(op)) is not None:
        shape = state.globalsdfg.arrays[state.name_mapping[state.globalsdfg][
            get_name(op)]].shape
    else:
        raise NameError("Variable name not found: ", get_name(op))
    # print("SHAPE:")
    # print(shape)
    indices = []
    if isinstance(op, Array_Subscript_Node):
        for i in op.indices:
            tw = TaskletWriter([], [])
            text = tw.write_code(i)
            #This might need to be replaced with the name in the context of the top/current sdfg
            indices.append(dace.symbolic.pystr_to_symbolic(text))
    memlet = '0'
    if len(shape) == 1:
        if shape[0] == 1:
            return memlet
    from dace import subsets
    all_indices = indices + [None] * (len(shape) - len(indices))
    subset = subsets.Range([(i, i, 1) if i is not None else (1, s, 1)
                            for i, s in zip(all_indices, shape)])
    return subset


class ProcessedWriter(TaskletWriter):
    def __init__(self, sdfg: SDFG, mapping):
        self.sdfg = sdfg
        self.mapping = mapping
        self.ast_elements = {
            BinOp_Node: self.binop2string,
            Name_Node: self.name2string,
            Name_Range_Node: self.namerange2string,
            Int_Literal_Node: self.intlit2string,
            Real_Literal_Node: self.floatlit2string,
            Bool_Literal_Node: self.boollit2string,
            UnOp_Node: self.unop2string,
            Array_Subscript_Node: self.arraysub2string,
            Parenthesis_Expr_Node: self.parenthesis2string,
            Call_Expr_Node: self.call2string,
            ParDecl_Node: self.pardecl2string,
        }

    def name2string(self, node: Name_Node):
        name = node.name
        for i in self.sdfg.arrays:
            sdfg_name = self.mapping.get(self.sdfg).get(name)
            if sdfg_name == i:
                name = i
                break
        return name

    def namerange2string(self, node: Name_Range_Node):
        name = node.name
        if name == "f2dace_MAX":
            arr = self.sdfg.arrays.get(
                self.mapping[self.sdfg][node.arrname.name])
            name = str(arr.shape[node.pos])
            return name
        else:
            return self.name2string(node)


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


class ModuleMap(dict):
    def __getitem__(self, k):
        assert isinstance(k, Module_Node)
        if k not in self:
            self[k] = {}

        return super().__getitem__(k)

    def get(self, k):
        return self[k]

    def __setitem__(self, k, v) -> None:
        assert isinstance(k, Module_Node)
        return super().__setitem__(k, v)


class AST_translator:
    def __init__(self, ast: InternalFortranAst, source):
        self.tables = ast.tables
        self.top_level = None
        self.globalsdfg = None
        self.functions_and_subroutines = ast.functions_and_subroutines
        self.name_mapping = NameMap()
        self.contexts = {}
        self.libstates = []
        self.file_name = source
        self.all_array_names = []
        self.last_sdfg_states = {}
        self.last_loop_continues = {}
        self.last_loop_breaks = {}
        self.last_returns = {}
        self.module_vars = []
        self.libraries = {}
        self.last_call_expression = {}
        self.ast_elements = {
            If_Stmt_Node: self.ifstmt2sdfg,
            For_Stmt_Node: self.forstmt2sdfg,
            Map_Stmt_Node: self.forstmt2sdfg,
            Execution_Part_Node: self.basicblock2sdfg,
            Subroutine_Subprogram_Node: self.subroutine2sdfg,
            BinOp_Node: self.binop2sdfg,
            Decl_Stmt_Node: self.declstmt2sdfg,
            Var_Decl_Node: self.vardecl2sdfg,
            Symbol_Decl_Node: self.symbol2sdfg,
            Symbol_Array_Decl_Node: self.symbolarray2sdfg,
            Call_Expr_Node: self.call2sdfg,
            Program_Node: self.ast2sdfg,
            Write_Stmt_Node: self.write2sdfg,
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
        for i in node.modules:
            for j in i.specification_part.typedecls:
                self.translate(j, sdfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))
            for j in i.specification_part.symbols:
                self.translate(j, sdfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))
            for j in i.specification_part.specifications:
                self.translate(j, sdfg)
                for k in j.vardecl:
                    self.module_vars.append((k.name, i.name))

        for i in node.main_program.specification_part.typedecls:
            self.translate(i, sdfg)
        for i in node.main_program.specification_part.symbols:
            self.translate(i, sdfg)
        for i in node.main_program.specification_part.specifications:
            self.translate(i, sdfg)
        self.translate(node.main_program.execution_part.execution, sdfg)

    def basicblock2sdfg(self, node: Execution_Part_Node, sdfg: SDFG):
        for i in node.execution:
            self.translate(i, sdfg)

    def write2sdfg(self, node: Write_Stmt_Node, sdfg: SDFG):

        print(node)

    def ifstmt2sdfg(self, node: If_Stmt_Node, sdfg: SDFG):

        name = "If_l_" + str(node.line_number[0]) + "_c_" + str(
            node.line_number[1])
        begin_state = add_simple_state_to_sdfg(self, sdfg, "Begin" + name)
        guard_substate = sdfg.add_state("Guard" + name)
        sdfg.add_edge(begin_state, guard_substate, dace.InterstateEdge())

        condition = ProcessedWriter(sdfg,
                                    self.name_mapping).write_code(node.cond)

        body_ifstart_state = sdfg.add_state("BodyIfStart" + name)
        self.last_sdfg_states[sdfg] = body_ifstart_state
        self.translate(node.body, sdfg)
        final_substate = sdfg.add_state("MergeState" + name)

        sdfg.add_edge(guard_substate, body_ifstart_state,
                      dace.InterstateEdge(condition))

        if self.last_sdfg_states[sdfg] not in [
                self.last_loop_breaks.get(sdfg),
                self.last_loop_continues.get(sdfg),
                self.last_returns.get(sdfg)
        ]:
            body_ifend_state = add_simple_state_to_sdfg(
                self, sdfg, "BodyIfEnd" + name)
            sdfg.add_edge(body_ifend_state, final_substate,
                          dace.InterstateEdge())

        if len(node.body_else.execution) > 0:
            name_else = "Else_l_" + str(node.line_number[0]) + "_c_" + str(
                node.line_number[1])
            body_elsestart_state = sdfg.add_state("BodyElseStart" + name_else)
            self.last_sdfg_states[sdfg] = body_elsestart_state
            self.translate(node.body_else, sdfg)
            body_elseend_state = add_simple_state_to_sdfg(
                self, sdfg, "BodyElseEnd" + name_else)
            sdfg.add_edge(guard_substate, body_elsestart_state,
                          dace.InterstateEdge("not (" + condition + ")"))
            sdfg.add_edge(body_elseend_state, final_substate,
                          dace.InterstateEdge())
        else:
            sdfg.add_edge(guard_substate, final_substate,
                          dace.InterstateEdge("not (" + condition + ")"))
        self.last_sdfg_states[sdfg] = final_substate

    def forstmt2sdfg(self, node: For_Stmt_Node, sdfg: SDFG):

        declloop = False
        name = "FOR_l_" + str(node.line_number[0]) + "_c_" + str(
            node.line_number[1])
        begin_state = add_simple_state_to_sdfg(self, sdfg, "Begin" + name)
        guard_substate = sdfg.add_state("Guard" + name)
        final_substate = sdfg.add_state("Merge" + name)
        self.last_sdfg_states[sdfg] = final_substate
        decl_node = node.init
        entry = {}
        if isinstance(decl_node, BinOp_Node):
            if sdfg.symbols.get(decl_node.lval.name) is not None:
                iter_name = decl_node.lval.name
            elif self.name_mapping[sdfg].get(decl_node.lval.name) is not None:
                iter_name = self.name_mapping[sdfg][decl_node.lval.name]
            else:
                raise ValueError("Unknown variable " + decl_node.lval.name)
            entry[iter_name] = ProcessedWriter(
                sdfg, self.name_mapping).write_code(decl_node.rval)

        sdfg.add_edge(begin_state, guard_substate,
                      dace.InterstateEdge(assignments=entry))

        condition = ProcessedWriter(sdfg,
                                    self.name_mapping).write_code(node.cond)

        increment = "i+0+1"
        if isinstance(node.iter, BinOp_Node):
            increment = ProcessedWriter(sdfg, self.name_mapping).write_code(
                node.iter.rval)
        entry = {iter_name: increment}

        begin_loop_state = sdfg.add_state("BeginLoop" + name)
        end_loop_state = sdfg.add_state("EndLoop" + name)
        self.last_sdfg_states[sdfg] = begin_loop_state
        self.last_loop_continues[sdfg] = end_loop_state
        self.translate(node.body, sdfg)

        sdfg.add_edge(self.last_sdfg_states[sdfg], end_loop_state,
                      dace.InterstateEdge())
        sdfg.add_edge(guard_substate, begin_loop_state,
                      dace.InterstateEdge(condition))
        sdfg.add_edge(end_loop_state, guard_substate,
                      dace.InterstateEdge(assignments=entry))
        sdfg.add_edge(guard_substate, final_substate,
                      dace.InterstateEdge("not (" + condition + ")"))
        self.last_sdfg_states[sdfg] = final_substate

    def symbol2sdfg(self, node: Symbol_Decl_Node, sdfg: SDFG):
        if self.contexts.get(sdfg.name) is None:
            self.contexts[sdfg.name] = Context(name=sdfg.name)
        if self.contexts[sdfg.name].constants.get(node.name) is None:
            if isinstance(node.init, Int_Literal_Node) or isinstance(
                    node.init, Real_Literal_Node):
                self.contexts[sdfg.name].constants[node.name] = node.init.value
            if isinstance(node.init, Name_Node):
                self.contexts[sdfg.name].constants[node.name] = self.contexts[
                    sdfg.name].constants[node.init.name]
        datatype = self.get_dace_type(node.type)
        if node.name not in sdfg.symbols:
            sdfg.add_symbol(node.name, datatype)
            if self.last_sdfg_states.get(sdfg) is None:
                bstate = sdfg.add_state("SDFGbegin", is_start_state=True)
                self.last_sdfg_states[sdfg] = bstate
            if node.init is not None:
                substate = sdfg.add_state("Dummystate_" + node.name)
                increment = TaskletWriter([], []).write_code(node.init)

                entry = {node.name: increment}
                sdfg.add_edge(self.last_sdfg_states[sdfg], substate,
                              dace.InterstateEdge(assignments=entry))
                self.last_sdfg_states[sdfg] = substate

    def symbolarray2sdfg(self, node: Symbol_Array_Decl_Node, sdfg: SDFG):
        return NotImplementedError(
            "Symbol_Decl_Node not implemented. This should be done via a transformation that itemizes the constant array."
        )

    def subroutine2sdfg(self, node: Subroutine_Subprogram_Node, sdfg: SDFG):

        if node.execution_part is None:
            return

        inputnodefinder = FindInputs()
        inputnodefinder.visit(node)
        input_vars = inputnodefinder.nodes
        outputnodefinder = FindOutputs()
        outputnodefinder.visit(node)
        output_vars = outputnodefinder.nodes

        parameters = node.args.copy()

        new_sdfg = dace.SDFG(node.name.name)
        substate = add_simple_state_to_sdfg(self, sdfg,
                                            "state" + node.name.name)
        variables_in_call = []
        if self.last_call_expression.get(sdfg) is not None:
            variables_in_call = self.last_call_expression[sdfg]

        # Sanity check to make sure the parameter numbers match
        if not ((len(variables_in_call) == len(parameters)) or
                (len(variables_in_call) == len(parameters) + 1
                 and not isinstance(node.result_type, Void))):
            for i in variables_in_call:
                print("VAR CALL: ", i)
            for j in parameters:
                print("LOCAL TO UPDATE: ", j)
            raise ValueError(
                "number of parameters does not match the function signature")

        # creating new arrays for nested sdfg
        inouts_in_new_sdfg = []

        views = []
        ind_count = 0

        var2 = []
        literals = []
        literal_values = []
        par2 = []

        symbol_arguments = []

        for arg_i, variable in enumerate(variables_in_call):
            # print(i.__class__)
            if isinstance(variable, Name_Node):
                varname = variable.name
            elif isinstance(variable, Array_Subscript_Node):
                varname = variable.name.name
            if isinstance(variable, Literal) or varname == "LITERAL":
                literals.append(parameters[arg_i])
                literal_values.append(variable)
                continue
            elif varname in sdfg.symbols:
                symbol_arguments.append((parameters[arg_i], variable))
                continue

            par2.append(parameters[arg_i])
            var2.append(variable)

        variables_in_call = var2
        parameters = par2
        assigns = []
        for lit, litval in zip(literals, literal_values):
            local_name = lit
            #self.translate(local_name, new_sdfg)
            #print("LOCAL_NAME SPECIAL: ",local_name.name,local_name.__class__)
            # self.name_mapping[(new_sdfg, local_name.name)] = find_new_array_name(self.all_array_names,
            #                                                                     local_name.name)
            #self.all_array_names.append(self.name_mapping[(new_sdfg, local_name.name)])
            assigns.append(
                BinOp_Node(lval=Name_Node(name=local_name.name),
                           rval=litval,
                           op="=",
                           line_number=node.line_number))

        for parameter, symbol in symbol_arguments:
            #self.translate(parameter, new_sdfg)
            if parameter.name != symbol.name:
                assigns.append(
                    BinOp_Node(lval=Name_Node(name=parameter.name),
                               rval=Name_Node(name=symbol.name),
                               op="=",
                               line_number=node.line_number))

        for variable_in_call in variables_in_call:
            all_arrays = self.get_arrays_in_context(sdfg)

            sdfg_name = self.name_mapping.get(sdfg).get(
                get_name(variable_in_call))
            globalsdfg_name = self.name_mapping.get(self.globalsdfg).get(
                get_name(variable_in_call))
            matched = False
            for array_name, array in all_arrays.items():
                if array_name in [sdfg_name]:
                    matched = True
                    local_name = parameters[variables_in_call.index(
                        variable_in_call)]
                    self.name_mapping[new_sdfg][
                        local_name.name] = new_sdfg._find_new_name(
                            local_name.name)
                    self.all_array_names.append(
                        self.name_mapping[new_sdfg][local_name.name])

                    inouts_in_new_sdfg.append(
                        self.name_mapping[new_sdfg][local_name.name])

                    indices = 0
                    index_list = []
                    shape = []
                    tmp_node = variable_in_call
                    strides = list(array.strides)
                    offsets = list(array.offset)
                    mysize = 1
                    if isinstance(variable_in_call, Array_Subscript_Node):
                        for i in variable_in_call.indices:
                            if isinstance(i, ParDecl_Node):
                                if i.type == "ALL":
                                    shape.append(array.shape[indices])
                                    mysize = mysize * array.shape[indices]
                                else:
                                    raise NotImplementedError(
                                        "Index in ParDecl should be ALL")
                            else:
                                text = ProcessedWriter(
                                    sdfg, self.name_mapping).write_code(i)
                                index_list.append(
                                    dace.symbolic.pystr_to_symbolic(text))
                                strides.pop(indices)
                                offsets.pop(indices)
                            indices = indices + 1

                    if isinstance(variable_in_call, Name_Node):
                        shape = list(array.shape)
                    if shape == () or shape == (
                            1, ) or shape == [] or shape == [1]:
                        new_sdfg.add_scalar(
                            self.name_mapping[new_sdfg][local_name.name],
                            array.dtype, array.storage)
                    else:
                        if not isinstance(variable_in_call, Name_Node):
                            viewname, view = sdfg.add_view(
                                array_name + "_view_" + str(len(views)),
                                shape,
                                array.dtype,
                                storage=array.storage,
                                strides=strides,
                                offset=offsets)
                            from dace import subsets

                            all_indices = index_list + [None] * (
                                len(array.shape) - len(index_list))
                            subset = subsets.Range([
                                (i, i, 1) if i is not None else (1, s, 1)
                                for i, s in zip(all_indices, array.shape)
                            ])
                            smallsubset = subsets.Range([(1, s, 1)
                                                         for s in shape])

                            memlet = dace.Memlet(
                                f'{array_name}[{subset}]->{smallsubset}')
                            memlet2 = dace.Memlet(
                                f'{viewname}[{smallsubset}]->{subset}')
                            r = substate.add_read(array_name)
                            wv = substate.add_write(viewname)
                            rv = substate.add_read(viewname)
                            w = substate.add_write(array_name)
                            substate.add_edge(r, None, wv, 'views',
                                              copy.deepcopy(memlet))
                            substate.add_edge(rv, 'views2', w, None,
                                              copy.deepcopy(memlet2))

                            views.append([array_name, wv, rv])

                        new_sdfg.add_array(
                            self.name_mapping[new_sdfg][local_name.name],
                            shape,
                            array.dtype,
                            array.storage,
                            strides=strides,
                            offset=offsets)
            if not matched:
                for array_name, array in all_arrays.items():
                    if array_name in [globalsdfg_name]:
                        local_name = parameters[variables_in_call.index(
                            variable_in_call)]
                        self.name_mapping[new_sdfg][
                            local_name.name] = new_sdfg._find_new_name(
                                local_name.name)
                        self.all_array_names.append(
                            self.name_mapping[new_sdfg][local_name.name])

                        inouts_in_new_sdfg.append(
                            self.name_mapping[new_sdfg][local_name.name])

                        indices = 0
                        if isinstance(variable_in_call, Array_Subscript_Node):
                            indices = len(variable_in_call.indices)

                        shape = array.shape[indices:]

                        if shape == () or shape == (1, ):
                            new_sdfg.add_scalar(
                                self.name_mapping[new_sdfg][local_name.name],
                                array.dtype, array.storage)
                        else:
                            new_sdfg.add_array(
                                self.name_mapping[new_sdfg][local_name.name],
                                shape,
                                array.dtype,
                                array.storage,
                                strides=array.strides,
                                offset=array.offset)

        # Preparing symbol dictionary for nested sdfg
        sym_dict = {}
        for i in sdfg.symbols:
            sym_dict[i] = i
        #print("FUNC: ",sym_dict)
        #if sdfg is not self.globalsdfg:

        write_names = list(dict.fromkeys([i.name for i in output_vars]))
        read_names = list(dict.fromkeys([i.name for i in input_vars]))
        not_found_write_names = []
        not_found_read_names = []
        for i in write_names:
            if self.name_mapping[new_sdfg].get(i) is None:
                not_found_write_names.append(i)
        for i in read_names:
            if self.name_mapping[new_sdfg].get(i) is None:
                not_found_read_names.append(i)

        for i in self.libstates:
            self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
            self.all_array_names.append(self.name_mapping[new_sdfg][i])
            inouts_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
            new_sdfg.add_scalar(self.name_mapping[new_sdfg][i],
                                dace.int32,
                                transient=False)
        addedmemlets = []
        globalmemlets = []
        for i in not_found_read_names:
            if i in [a[0] for a in self.module_vars]:
                if self.name_mapping[sdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    addedmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    inouts_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    array_in_global = sdfg.arrays[self.name_mapping[sdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i],
                                            array_in_global.dtype,
                                            transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
                elif self.name_mapping[self.globalsdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    globalmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    inouts_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    array_in_global = self.globalsdfg.arrays[self.name_mapping[
                        self.globalsdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i],
                                            array_in_global.dtype,
                                            transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
        for i in not_found_write_names:
            if i in not_found_read_names:
                continue
            if i in [a[0] for a in self.module_vars]:
                if self.name_mapping[sdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    addedmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    inouts_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    array = sdfg.arrays[self.name_mapping[sdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i],
                                            array_in_global.dtype,
                                            transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)
                elif self.name_mapping[self.globalsdfg].get(i) is not None:
                    self.name_mapping[new_sdfg][i] = new_sdfg._find_new_name(i)
                    globalmemlets.append(i)
                    self.all_array_names.append(self.name_mapping[new_sdfg][i])
                    inouts_in_new_sdfg.append(self.name_mapping[new_sdfg][i])
                    array = self.globalsdfg.arrays[self.name_mapping[
                        self.globalsdfg][i]]
                    if isinstance(array_in_global, Scalar):
                        new_sdfg.add_scalar(self.name_mapping[new_sdfg][i],
                                            array_in_global.dtype,
                                            transient=False)
                    elif array_in_global.type == "Array":
                        new_sdfg.add_array(self.name_mapping[new_sdfg][i],
                                           array_in_global.shape,
                                           array_in_global.dtype,
                                           array_in_global.storage,
                                           transient=False,
                                           strides=array_in_global.strides,
                                           offset=array_in_global.offset)

        #print(inouts_in_new_sdfg)

        internal_sdfg = substate.add_nested_sdfg(new_sdfg,
                                                 sdfg,
                                                 inouts_in_new_sdfg,
                                                 inouts_in_new_sdfg,
                                                 symbol_mapping=sym_dict)
        #if sdfg is not self.globalsdfg:
        for i in self.libstates:
            memlet = "0"
            add_memlet_write(substate, self.name_mapping[sdfg][i],
                             internal_sdfg, self.name_mapping[new_sdfg][i],
                             memlet)
            add_memlet_read(substate, self.name_mapping[sdfg][i],
                            internal_sdfg, self.name_mapping[new_sdfg][i],
                            memlet)

        for i in variables_in_call:

            local_name = parameters[variables_in_call.index(i)]
            if self.name_mapping.get(sdfg).get(get_name(i)) is not None:
                var = sdfg.arrays.get(self.name_mapping[sdfg][get_name(i)])
                mapped_name = self.name_mapping[sdfg][get_name(i)]
            # TODO: FIx symbols in function calls
            elif get_name(i) in sdfg.symbols:
                var = get_name(i)
                mapped_name = get_name(i)
            elif self.name_mapping.get(self.globalsdfg).get(
                    get_name(i)) is not None:
                var = self.globalsdfg.arrays.get(
                    self.name_mapping[self.globalsdfg][get_name(i)])
                mapped_name = self.name_mapping[self.globalsdfg][get_name(i)]
            else:
                raise NameError("Variable name not found: " + get_name(i))

            # print("Context change:",i.name," ",var.shape)
            if not hasattr(var, "shape") or len(var.shape) == 0:
                memlet = ""
            elif (len(var.shape) == 1 and var.shape[0] == 1):
                memlet = "0"
            else:
                memlet = generate_memlet(i, sdfg, self)
            # print("MEMLET: "+memlet)
            found = False
            for elem in views:
                if mapped_name == elem[0]:
                    found = True
                    memlet = subsets.Range([
                        (1, s, 1) for s in sdfg.arrays[elem[1].label].shape
                    ])
                    substate.add_memlet_path(
                        internal_sdfg,
                        elem[2],
                        src_conn=self.name_mapping[new_sdfg][local_name.name],
                        memlet=dace.Memlet(expr=elem[1].label, subset=memlet))
                    substate.add_memlet_path(
                        elem[1],
                        internal_sdfg,
                        dst_conn=self.name_mapping[new_sdfg][local_name.name],
                        memlet=dace.Memlet(expr=elem[1].label, subset=memlet))

            if not found:
                add_memlet_write(substate, mapped_name, internal_sdfg,
                                 self.name_mapping[new_sdfg][local_name.name],
                                 memlet)
                add_memlet_read(substate, mapped_name, internal_sdfg,
                                self.name_mapping[new_sdfg][local_name.name],
                                memlet)

        for i in addedmemlets:

            memlet = generate_memlet(Name_Node(name=i), sdfg, self)
            add_memlet_write(substate, self.name_mapping[sdfg][i],
                             internal_sdfg, self.name_mapping[new_sdfg][i],
                             memlet)
            add_memlet_read(substate, self.name_mapping[sdfg][i],
                            internal_sdfg, self.name_mapping[new_sdfg][i],
                            memlet)
        for i in globalmemlets:

            memlet = generate_memlet(Name_Node(name=i), sdfg, self)
            add_memlet_write(substate, self.name_mapping[self.globalsdfg][i],
                             internal_sdfg, self.name_mapping[new_sdfg][i],
                             memlet)
            add_memlet_read(substate, self.name_mapping[self.globalsdfg][i],
                            internal_sdfg, self.name_mapping[new_sdfg][i],
                            memlet)

        # make_nested_sdfg_with_context_change(sdfg, new_sdfg, node.name, used_vars, self)

        if node.execution_part is not None:
            for j in node.specification_part.uses:
                for k in j.list:
                    if self.contexts.get(new_sdfg.name) is None:
                        self.contexts[new_sdfg.name] = Context(
                            name=new_sdfg.name)
                    if self.contexts[new_sdfg.name].constants.get(
                            get_name(k)) is None and self.contexts[
                                self.globalsdfg.name].constants.get(
                                    get_name(k)) is not None:
                        self.contexts[new_sdfg.name].constants[get_name(
                            k)] = self.contexts[
                                self.globalsdfg.name].constants[get_name(k)]

                    print(get_name(k))
                    pass
            for j in node.specification_part.specifications:
                self.declstmt2sdfg(j, new_sdfg)
            for i in assigns:
                self.translate(i, new_sdfg)
            self.translate(node.execution_part, new_sdfg)

    #TODO REWRITE THIS nicely
    def binop2sdfg(self, node: BinOp_Node, sdfg: SDFG):
        #print(node)

        calls = FindFunctionCalls()
        calls.visit(node)
        if len(calls.nodes) == 1:
            augmented_call = calls.nodes[0]
            if augmented_call.name.name not in [
                    "sqrt", "exp", "pow", "max", "min", "abs", "tanh"
            ]:
                augmented_call.args.append(node.lval)
                augmented_call.hasret = True
                self.call2sdfg(augmented_call, sdfg)
                return

        outputnodefinder = FindOutputs()
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

        inputnodefinder = FindInputs()
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

        #output_names_changed = [o_t + "_out" for o_t in output_names_tasklet]
        output_names_changed = [o_t for o_t in output_names_tasklet]
        #output_names_dict = {on: dace.pointer(dace.int32) for on in output_names_changed}
        """ tasklet = add_tasklet(
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
            add_memlet_write(substate, i, tasklet, k, memlet_range) """
        tasklet = add_tasklet(
            substate,
            "_l" + str(node.line_number[0]) + "_c" + str(node.line_number[1]),
            input_names, output_names, "text", node.line_number,
            self.file_name)

        for i, j in zip(input_names, input_names):
            memlet_range = self.get_memlet_range(sdfg, input_vars, i, j)
            add_memlet_read(substate, i, tasklet, j, memlet_range)

        for i, j, k in zip(output_names, output_names, output_names):

            memlet_range = self.get_memlet_range(sdfg, output_vars, i, j)
            add_memlet_write(substate, i, tasklet, k, memlet_range)
        tw = ProcessedWriter(sdfg, self.name_mapping)
        # print("BINOP:",output_names,output_names_tasklet,output_names_changed)
        text = tw.write_code(node)
        # print("BINOPTASKLET:",text)
        tasklet.code = CodeBlock(text, dace.Language.Python)

    def call2sdfg(self, node: Call_Expr_Node, sdfg: SDFG):
        self.last_call_expression[sdfg] = node.args
        match_found = False
        rettype = "INTEGER"
        hasret = False
        if node.name in self.functions_and_subroutines:
            for i in self.top_level.function_definitions:
                if i.name == node.name:
                    self.function2sdfg(i, sdfg)
                    return
            for i in self.top_level.subroutine_definitions:
                if i.name == node.name:
                    self.subroutine2sdfg(i, sdfg)
                    return
            for j in self.top_level.modules:
                for i in j.function_definitions:
                    if i.name == node.name:
                        self.function2sdfg(i, sdfg)
                        return
                for i in j.subroutine_definitions:
                    if i.name == node.name:
                        self.subroutine2sdfg(i, sdfg)
                        return
        else:
            #TODO rewrite this
            libstate = self.libraries.get(node.name.name)
            if not isinstance(rettype, Void) and hasattr(node, "hasret"):
                if node.hasret:
                    hasret = True
                    retval = node.args.pop(len(node.args) - 1)
            if node.name == "free":
                return
            input_names_tasklet = {}
            output_names_tasklet = []
            input_names = []
            output_names = []
            special_list_in = {}
            special_list_out = []
            if libstate is not None:
                #print("LIBSTATE:", libstate)
                special_list_in[self.name_mapping[sdfg][libstate] +
                                "_task"] = dace.pointer(
                                    sdfg.arrays.get(self.name_mapping[sdfg]
                                                    [libstate]).dtype)
                special_list_out.append(self.name_mapping[sdfg][libstate] +
                                        "_task_out")
            used_vars = [
                node for node in walk(node) if isinstance(node, Name_Node)
            ]

            for i in used_vars:
                for j in sdfg.arrays:
                    if self.name_mapping.get(sdfg).get(
                            i.name) == j and j not in input_names:
                        elem = sdfg.arrays.get(j)
                        scalar = False
                        if len(elem.shape) == 0:
                            scalar = True
                        elif (len(elem.shape) == 1 and elem.shape[0] == 1):
                            scalar = True
                        if not scalar and not node.name.name in [
                                "fprintf", "printf"
                        ]:
                            #    print("ADDING!",
                            #          not node.name.name in ["fprintf", "printf"],
                            #          not scalar)
                            output_names.append(j)
                            output_names_tasklet.append(i.name)
                        #print("HERE: ", elem.__class__, j, scalar,
                        #      node.name.name)

                        input_names_tasklet[i.name] = dace.pointer(elem.dtype)
                        input_names.append(j)

            output_names_changed = []
            for o, o_t in zip(output_names, output_names_tasklet):
                # changes=False
                # for i,i_t in zip(input_names,input_names_tasklet):
                #    if o_t==i_t:
                #        var=sdfg.arrays.get(i)
                #        if len(var.shape) == 0 or (len(var.shape) == 1 and var.shape[0] is 1):
                output_names_changed.append(o_t + "_out")

            tw = TaskletWriter(output_names_tasklet.copy(),
                               output_names_changed.copy())
            if not isinstance(rettype, Void) and hasret:
                special_list_in[retval.name] = dace.pointer(
                    self.get_dace_type(rettype))
                # special_list_in.append(retval.name)
                special_list_out.append(retval.name + "_out")
                text = tw.write_code(
                    BinOp_Node(lval=retval,
                               op="=",
                               rval=node,
                               line_number=node.line_number))

            else:
                text = tw.write_code(node)
            substate = add_simple_state_to_sdfg(
                self, sdfg, "_state" + str(node.line_number[0]))

            tasklet = add_tasklet(substate, str(node.line_number[0]), {
                **input_names_tasklet,
                **special_list_in
            }, output_names_changed + special_list_out, "text",
                                  node.line_number, self.file_name)
            if libstate is not None:
                add_memlet_read(substate, self.name_mapping[sdfg][libstate],
                                tasklet,
                                self.name_mapping[sdfg][libstate] + "_task",
                                "0")

                add_memlet_write(
                    substate, self.name_mapping[sdfg][libstate], tasklet,
                    self.name_mapping[sdfg][libstate] + "_task_out", "0")
            if not isinstance(rettype, Void) and hasret:
                add_memlet_read(substate, self.name_mapping[sdfg][retval.name],
                                tasklet, retval.name, "0")

                add_memlet_write(substate,
                                 self.name_mapping[sdfg][retval.name], tasklet,
                                 retval.name + "_out", "0")

            for i, j in zip(input_names, input_names_tasklet):
                memlet_range = self.get_memlet_range(sdfg, used_vars, i, j)
                add_memlet_read(substate, i, tasklet, j, memlet_range)

            for i, j, k in zip(output_names, output_names_tasklet,
                               output_names_changed):

                memlet_range = self.get_memlet_range(sdfg, used_vars, i, j)
                add_memlet_write(substate, i, tasklet, k, memlet_range)

            setattr(tasklet, "code", CodeBlock(text, dace.Language.Python))

    def declstmt2sdfg(self, node: Decl_Stmt_Node, sdfg: SDFG):
        for i in node.vardecl:
            self.translate(i, sdfg)

    def vardecl2sdfg(self, node: Var_Decl_Node, sdfg: SDFG):
        #if the sdfg is the toplevel-sdfg, the variable is a global variable
        transient = True
        #transient = sdfg is not self.globalsdfg
        # find the type
        datatype = self.get_dace_type(node.type)
        # get the dimensions
        if node.sizes is not None:
            sizes = []
            offset = []
            offset_value = -1
            for i in node.sizes:
                tw = TaskletWriter([], [])
                text = tw.write_code(i)
                sizes.append(dace.symbolic.pystr_to_symbolic(text))
                offset.append(offset_value)

        else:
            sizes = None
        # create and check name
        if self.name_mapping[sdfg].get(node.name) is not None:
            return
            #raise ValueError("Name already defined in this scope")
        if node.name in sdfg.symbols:
            return
            #raise ValueError("Name already defined as symbol")
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


def create_sdfg_from_string(
    source_string: str,
    sdfg_name: str,
):
    parser = ParserFactory().create(std="f2008")
    reader = FortranStringReader(source_string)
    ast = parser(reader)
    tables = SYMBOL_TABLES
    own_ast = InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)
    functions_and_subroutines_builder = FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.nodes
    program = functionStatementEliminator(program)
    program = CallToArray(
        functions_and_subroutines_builder.nodes).visit(program)
    program = CallExtractor().visit(program)
    program = SignToIf().visit(program)
    program = ArrayToLoop().visit(program)
    program = SumToLoop().visit(program)
    program = ForDeclarer().visit(program)
    program = IndexExtractor().visit(program)
    ast2sdfg = AST_translator(own_ast, __file__)
    sdfg = SDFG(sdfg_name)
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.NestedSDFG):
            if 'test_function' in node.sdfg.name:
                sdfg = node.sdfg
                break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()
    return sdfg


if __name__ == "__main__":
    parser = ParserFactory().create(std="f2008")
    #testname = "int_assign"

    #testname = "arrayrange1"
    testname = "cloudscexp2"
    reader = FortranFileReader(
        os.path.realpath("/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" +
                         testname + ".f90"))
    ast = parser(reader)
    tables = SYMBOL_TABLES
    table = tables.lookup("CLOUDPROGRAM")
    own_ast = InternalFortranAst(ast, tables)
    #own_ast.list_tables()
    program = own_ast.create_ast(ast)
    functions_and_subroutines_builder = FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.nodes
    program = functionStatementEliminator(program)
    program = CallToArray(
        functions_and_subroutines_builder.nodes).visit(program)
    program = CallExtractor().visit(program)
    program = SignToIf().visit(program)
    program = ArrayToLoop().visit(program)
    program = SumToLoop().visit(program)
    program = ForDeclarer().visit(program)
    ast2sdfg = AST_translator(
        own_ast,
        "/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" + testname + ".f90")
    sdfg = SDFG("top_level")
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    sdfg.validate()
    sdfg.save("/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" + testname +
              "_initial.sdfg")
    sdfg.simplify(verbose=True)
    sdfg.save("/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" + testname +
              "_simplify.sdfg")
    from dace.transformation.auto import auto_optimize as aopt
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    sdfg.save("/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" + testname +
              "_optimized.sdfg")
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
