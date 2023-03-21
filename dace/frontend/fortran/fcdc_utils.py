# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.api import parse
import os
import sys
from fparser.common.readfortran import FortranStringReader, FortranFileReader

#dace imports
from dace import subsets
from dace.data import Scalar
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace import Memlet
from dace.sdfg.nodes import Tasklet
from dace import dtypes
from dace import symbolic as sym
from dace import DebugInfo as di
from dace import Language as lang
from dace.properties import CodeBlock
from numpy import finfo as finf
from numpy import float as fl

from dace.frontend.fortran import ast_internal_classes
from typing import List, Set


def add_tasklet(substate: SDFGState, name: str, vars_in: Set[str], vars_out: Set[str], code: str, debuginfo: list,
                source: str):
    tasklet = substate.add_tasklet(name="T" + name,
                                   inputs=vars_in,
                                   outputs=vars_out,
                                   code=code,
                                   debuginfo=di(start_line=debuginfo[0], start_column=debuginfo[1], filename=source),
                                   language=lang.Python)
    return tasklet


def add_memlet_read(substate: SDFGState, var_name: str, tasklet: Tasklet, dest_conn: str, memlet_range: str):
    src = substate.add_access(var_name)
    if memlet_range != "":
        substate.add_memlet_path(src, tasklet, dst_conn=dest_conn, memlet=Memlet(expr=var_name, subset=memlet_range))
    else:
        substate.add_memlet_path(src, tasklet, dst_conn=dest_conn, memlet=Memlet(expr=var_name))


def add_memlet_write(substate: SDFGState, var_name: str, tasklet: Tasklet, source_conn: str, memlet_range: str):
    dst = substate.add_write(var_name)
    if memlet_range != "":
        substate.add_memlet_path(tasklet, dst, src_conn=source_conn, memlet=Memlet(expr=var_name, subset=memlet_range))
    else:
        substate.add_memlet_path(tasklet, dst, src_conn=source_conn, memlet=Memlet(expr=var_name))


def add_simple_state_to_sdfg(state: SDFGState, top_sdfg: SDFG, state_name: str):
    if state.last_sdfg_states.get(top_sdfg) is not None:
        substate = top_sdfg.add_state(state_name)
    else:
        substate = top_sdfg.add_state(state_name, is_start_state=True)
    finish_add_state_to_sdfg(state, top_sdfg, substate)
    return substate


def finish_add_state_to_sdfg(state: SDFGState, top_sdfg: SDFG, substate: SDFGState):
    if state.last_sdfg_states.get(top_sdfg) is not None:
        top_sdfg.add_edge(state.last_sdfg_states[top_sdfg], substate, InterstateEdge())
    state.last_sdfg_states[top_sdfg] = substate


def get_name(node: ast_internal_classes.FNode):
    if isinstance(node, ast_internal_classes.Name_Node):
        return node.name
    elif isinstance(node, ast_internal_classes.Array_Subscript_Node):
        return node.name.name
    else:
        raise NameError("Name not found")


class TaskletWriter:
    """
    Class that writes a python tasklet from a node
    :param outputs: list of output variables
    :param outputs_changes: list of names output variables should be changed to
    :param input: list of input variables
    :param input_changes: list of names input variables should be changed to
    :param sdfg: sdfg the tasklet will be part of
    :param name_mapping: mapping of names in the code to names in the sdfg
    :return: python code for a tasklet, as a string
    """
    def __init__(self,
                 outputs: List[str],
                 outputs_changes: List[str],
                 sdfg: SDFG = None,
                 name_mapping=None,
                 input: List[str] = None,
                 input_changes: List[str] = None):
        self.outputs = outputs
        self.outputs_changes = outputs_changes
        self.sdfg = sdfg
        self.mapping = name_mapping
        self.input = input
        self.input_changes = input_changes

        self.ast_elements = {
            ast_internal_classes.BinOp_Node: self.binop2string,
            ast_internal_classes.Name_Node: self.name2string,
            ast_internal_classes.Name_Range_Node: self.name2string,
            ast_internal_classes.Int_Literal_Node: self.intlit2string,
            ast_internal_classes.Real_Literal_Node: self.floatlit2string,
            ast_internal_classes.Bool_Literal_Node: self.boollit2string,
            ast_internal_classes.UnOp_Node: self.unop2string,
            ast_internal_classes.Array_Subscript_Node: self.arraysub2string,
            ast_internal_classes.Parenthesis_Expr_Node: self.parenthesis2string,
            ast_internal_classes.Call_Expr_Node: self.call2string,
            ast_internal_classes.ParDecl_Node: self.pardecl2string,
        }

    def pardecl2string(self, node: ast_internal_classes.ParDecl_Node):
        #At this point in the process, the should not be any ParDecl nodes left in the AST - they should have been replaced by the appropriate ranges
        return f"ERROR{node.type}"

    def write_code(self, node: ast_internal_classes.FNode):
        """
        :param node: node to write code for
        :return: python code for the node, as a string
        :note Main function of the class, writes the code for a node
        :note If the node is a string, it is returned as is
        :note If the node is not a string, it is checked if it is in the ast_elements dictionary
        :note If it is, the appropriate function is called with the node as an argument, leading to a recursive traversal of the tree spanned by the node
        :note If it not, an error is raised

        """
        if node.__class__ in self.ast_elements:
            text = self.ast_elements[node.__class__](node)
            if text is None:
                raise NameError("Error in code generation")

            return text
        elif isinstance(node, str):
            return node
        else:
            raise NameError("Error in code generation" + node.__class__.__name__)

    def arraysub2string(self, node: ast_internal_classes.Array_Subscript_Node):
        str_to_return = self.write_code(node.name) + "[" + self.write_code(node.indices[0])
        for i in node.indices[1:]:
            str_to_return += ", " + self.write_code(i)
        str_to_return += "]"
        return str_to_return

    def name2string(self, node):
        if isinstance(node, str):
            return node

        return_value = node.name
        name = node.name
        for i in self.sdfg.arrays:
            sdfg_name = self.mapping.get(self.sdfg).get(name)
            if sdfg_name == i:
                name = i
                break

        if len(self.outputs) > 0:
            if name == self.outputs[0]:
                if self.outputs[0] != self.outputs_changes[0]:
                    name = self.outputs_changes[0]
                self.outputs.pop(0)
                self.outputs_changes.pop(0)

        if self.input is not None and len(self.input) > 0:
            if name == self.input[0]:
                if self.input[0] != self.input_changes[0]:
                    name = self.input_changes[0]
                else:
                    pass
                self.input.pop(0)
                self.input_changes.pop(0)
        return name

    def intlit2string(self, node: ast_internal_classes.Int_Literal_Node):

        return "".join(map(str, node.value))

    def floatlit2string(self, node: ast_internal_classes.Real_Literal_Node):

        return "".join(map(str, node.value))

    def boollit2string(self, node: ast_internal_classes.Bool_Literal_Node):

        return str(node.value)

    def unop2string(self, node: ast_internal_classes.UnOp_Node):
        op = node.op
        if op == ".NOT.":
            op = "not "
        return op + self.write_code(node.lval)

    def parenthesis2string(self, node: ast_internal_classes.Parenthesis_Expr_Node):
        return "(" + self.write_code(node.expr) + ")"

    def call2string(self, node: ast_internal_classes.Call_Expr_Node):
        # This is a replacement for the epsilon function in fortran
        if node.name.name == "__dace_epsilon":
            return str(finf(fl).eps)
        return_str = self.write_code(node.name) + "(" + self.write_code(node.args[0])
        for i in node.args[1:]:
            return_str += ", " + self.write_code(i)
        return_str += ")"
        return return_str

    def binop2string(self, node: ast_internal_classes.BinOp_Node):

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
        #TODO Add list of missing operators

        left = self.write_code(node.lval)
        right = self.write_code(node.rval)
        if op != "=":
            return "(" + left + op + right + ")"
        else:
            return left + op + right


def generate_memlet(op, top_sdfg, state):
    if state.name_mapping.get(top_sdfg).get(get_name(op)) is not None:
        shape = top_sdfg.arrays[state.name_mapping[top_sdfg][get_name(op)]].shape
    elif state.name_mapping.get(state.globalsdfg).get(get_name(op)) is not None:
        shape = state.globalsdfg.arrays[state.name_mapping[state.globalsdfg][get_name(op)]].shape
    else:
        raise NameError("Variable name not found: ", get_name(op))
    indices = []
    if isinstance(op, ast_internal_classes.Array_Subscript_Node):
        for i in op.indices:
            tw = TaskletWriter([], [], top_sdfg, state.name_mapping)
            text = tw.write_code(i)
            #This might need to be replaced with the name in the context of the top/current sdfg
            indices.append(sym.pystr_to_symbolic(text))
    memlet = '0'
    if len(shape) == 1:
        if shape[0] == 1:
            return memlet

    all_indices = indices + [None] * (len(shape) - len(indices))
    subset = subsets.Range([(i, i, 1) if i is not None else (1, s, 1) for i, s in zip(all_indices, shape)])
    return subset


class ProcessedWriter(TaskletWriter):
    """
    This class is derived from the TaskletWriter class and is used to write the code of a tasklet that's on an interstate edge rather than a computational tasklet.
    :note The only differences are in that the names for the sdfg mapping are used, and that the indices are considered to be one-bases rather than zero-based. 
    """
    def __init__(self, sdfg: SDFG, mapping):
        self.sdfg = sdfg
        self.mapping = mapping
        self.ast_elements = {
            ast_internal_classes.BinOp_Node: self.binop2string,
            ast_internal_classes.Name_Node: self.name2string,
            ast_internal_classes.Name_Range_Node: self.namerange2string,
            ast_internal_classes.Int_Literal_Node: self.intlit2string,
            ast_internal_classes.Real_Literal_Node: self.floatlit2string,
            ast_internal_classes.Bool_Literal_Node: self.boollit2string,
            ast_internal_classes.UnOp_Node: self.unop2string,
            ast_internal_classes.Array_Subscript_Node: self.arraysub2string,
            ast_internal_classes.Parenthesis_Expr_Node: self.parenthesis2string,
            ast_internal_classes.Call_Expr_Node: self.call2string,
            ast_internal_classes.ParDecl_Node: self.pardecl2string,
        }

    def name2string(self, node: ast_internal_classes.Name_Node):
        name = node.name
        for i in self.sdfg.arrays:
            sdfg_name = self.mapping.get(self.sdfg).get(name)
            if sdfg_name == i:
                name = i
                break
        return name

    def arraysub2string(self, node: ast_internal_classes.Array_Subscript_Node):
        str_to_return = self.write_code(node.name) + "[(" + self.write_code(node.indices[0]) + "+1)"
        for i in node.indices[1:]:
            str_to_return += ",( " + self.write_code(i) + "+1)"
        str_to_return += "]"
        return str_to_return

    def namerange2string(self, node: ast_internal_classes.Name_Range_Node):
        name = node.name
        if name == "f2dace_MAX":
            arr = self.sdfg.arrays.get(self.mapping[self.sdfg][node.arrname.name])
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
        assert isinstance(k, ast_internal_classes.Module_Node)
        if k not in self:
            self[k] = {}

        return super().__getitem__(k)

    def get(self, k):
        return self[k]

    def __setitem__(self, k, v) -> None:
        assert isinstance(k, ast_internal_classes.Module_Node)
        return super().__setitem__(k, v)
