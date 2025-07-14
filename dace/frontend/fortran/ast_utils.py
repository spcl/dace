# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import Counter, defaultdict
from itertools import chain
from typing import List, Set, Iterator, Type, TypeVar, Dict, Tuple, Iterable, Union, Optional

import networkx as nx
from fparser.two.Fortran2003 import Module_Stmt, Name, Interface_Block, Subroutine_Stmt, Specification_Part, Module, \
    Derived_Type_Def, Function_Stmt, Interface_Stmt, Function_Body, Type_Name, Rename, Entity_Decl, Kind_Selector, \
    Intrinsic_Type_Spec, Use_Stmt, Declaration_Type_Spec
from fparser.two.Fortran2008 import Type_Declaration_Stmt, Procedure_Stmt
from fparser.two.utils import Base
from numpy import finfo as finf
from numpy import float64 as fl

from dace import DebugInfo as di
from dace import Language as lang
from dace import Memlet
from dace import data as dat
from dace import dtypes
# dace imports
from dace import subsets
from dace import symbolic as sym
from dace.frontend.fortran import ast_internal_classes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg.nodes import Tasklet

fortrantypes2dacetypes = {
    "DOUBLE": dtypes.float64,
    "REAL": dtypes.float32,
    "INTEGER": dtypes.int32,
    "INTEGER8": dtypes.int64,
    "CHAR": dtypes.int8,
    "LOGICAL": dtypes.int32,  # This is a hack to allow fortran to pass through external C
    "Unknown": dtypes.float64,  # TMP hack unti lwe have a proper type inference
}


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
    found = False
    if isinstance(substate.parent.arrays[var_name], dat.View):
        for i in substate.data_nodes():
            if i.data == var_name and len(substate.out_edges(i)) == 0:
                src = i
                found = True
                break
    if not found:
        src = substate.add_read(var_name)

    # src = substate.add_access(var_name)
    if memlet_range != "":
        substate.add_memlet_path(src, tasklet, dst_conn=dest_conn, memlet=Memlet(expr=var_name, subset=memlet_range))
    else:
        substate.add_memlet_path(src, tasklet, dst_conn=dest_conn, memlet=Memlet(expr=var_name))
    return src


def add_memlet_write(substate: SDFGState, var_name: str, tasklet: Tasklet, source_conn: str, memlet_range: str):
    found = False
    if isinstance(substate.parent.arrays[var_name], dat.View):
        for i in substate.data_nodes():
            if i.data == var_name and len(substate.in_edges(i)) == 0:
                dst = i
                found = True
                break
    if not found:
        dst = substate.add_write(var_name)
    # dst = substate.add_write(var_name)
    if memlet_range != "":
        substate.add_memlet_path(tasklet, dst, src_conn=source_conn, memlet=Memlet(expr=var_name, subset=memlet_range))
    else:
        substate.add_memlet_path(tasklet, dst, src_conn=source_conn, memlet=Memlet(expr=var_name))
    return dst


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
    if isinstance(node, ast_internal_classes.Actual_Arg_Spec_Node):
        actual_node = node.arg
    else:
        actual_node = node    
    if isinstance(actual_node, ast_internal_classes.Name_Node):
        return actual_node.name
    elif isinstance(actual_node, ast_internal_classes.Array_Subscript_Node):
        return actual_node.name.name
    elif isinstance(actual_node, ast_internal_classes.Data_Ref_Node):
        view_name = actual_node.parent_ref.name
        while isinstance(actual_node.part_ref, ast_internal_classes.Data_Ref_Node):
            if isinstance(actual_node.part_ref.parent_ref, ast_internal_classes.Name_Node):
                view_name = view_name + "_" + actual_node.part_ref.parent_ref.name
            elif isinstance(actual_node.part_ref.parent_ref, ast_internal_classes.Array_Subscript_Node):
                view_name = view_name + "_" + actual_node.part_ref.parent_ref.name.name
            actual_node = actual_node.part_ref
        view_name = view_name + "_" + get_name(actual_node.part_ref)
        return view_name

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

    # store all tasklets generated for quick retrieval of assignments
    TASKLETS_CREATED: Dict[SDFG, Dict[str, Tuple[str, ast_internal_classes.FNode]]] = defaultdict(lambda: defaultdict())

    def __init__(self,
                 outputs: List[str],
                 outputs_changes: List[str],
                 sdfg: SDFG = None,
                 name_mapping=None,
                 input: List[str] = None,
                 input_changes: List[str] = None,
                 placeholders={},
                 placeholders_offsets={},
                 rename_dict=None
                 ):
        self.outputs = outputs
        self.outputs_changes = outputs_changes
        self.sdfg = sdfg
        self.placeholders = placeholders
        self.placeholders_offsets = placeholders_offsets
        self.mapping = name_mapping
        self.input = input
        self.input_changes = input_changes
        self.rename_dict = rename_dict
        self.depth = 0
        self.data_ref_stack = []

        self.ast_elements = {
            ast_internal_classes.BinOp_Node: self.binop2string,
            ast_internal_classes.Actual_Arg_Spec_Node: self.actualarg2string,
            ast_internal_classes.Name_Node: self.name2string,
            ast_internal_classes.Name_Range_Node: self.name2string,
            ast_internal_classes.Int_Literal_Node: self.intlit2string,
            ast_internal_classes.Real_Literal_Node: self.floatlit2string,
            ast_internal_classes.Double_Literal_Node: self.doublelit2string,
            ast_internal_classes.Bool_Literal_Node: self.boollit2string,
            ast_internal_classes.Char_Literal_Node: self.charlit2string,
            ast_internal_classes.UnOp_Node: self.unop2string,
            ast_internal_classes.Array_Subscript_Node: self.arraysub2string,
            ast_internal_classes.Parenthesis_Expr_Node: self.parenthesis2string,
            ast_internal_classes.Call_Expr_Node: self.call2string,
            ast_internal_classes.ParDecl_Node: self.pardecl2string,
            ast_internal_classes.Data_Ref_Node: self.dataref2string,
            ast_internal_classes.Array_Constructor_Node: self.arrayconstructor2string,
        }

    def pardecl2string(self, node: ast_internal_classes.ParDecl_Node):
        # At this point in the process, the should not be any ParDecl nodes left in the AST - they should have been replaced by the appropriate ranges
        return '0'
        #raise NameError("Error in code generation")
        return f"ERROR{node.type}"

    def actualarg2string(self, node: ast_internal_classes.Actual_Arg_Spec_Node):
        return self.write_code(node.arg)
    
    def arrayconstructor2string(self, node: ast_internal_classes.Array_Constructor_Node):
        str_to_return = "[ "
        for i in node.value_list:
            str_to_return += self.write_code(i) + ", "
        str_to_return = str_to_return[:-2]
        str_to_return += " ]"
        return str_to_return

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
        self.depth += 1
        if node.__class__ in self.ast_elements:
            text = self.ast_elements[node.__class__](node)
            if text is None:
                raise NameError("Error in code generation")
            if "ERRORALL" in text and self.depth == 1:
                print(text)
                #raise NameError("Error in code generation")
            self.depth -= 1
            return text
        elif isinstance(node, int):
            self.depth -= 1
            return str(node)
        elif isinstance(node, str):
            self.depth -= 1
            return node
        elif isinstance(node, sym.symbol):
            string_name = str(node)
            string_to_return = self.write_code(ast_internal_classes.Name_Node(name=string_name))
            self.depth -= 1
            return string_to_return
        else:
            raise NameError("Error in code generation: " + node.__class__.__name__)

    def dataref2string(self, node: ast_internal_classes.Data_Ref_Node):
        part1=self.write_code(node.parent_ref)
        if isinstance(node.parent_ref, ast_internal_classes.Name_Node):
            self.data_ref_stack.append(node.parent_ref)
        elif isinstance(node.parent_ref, ast_internal_classes.Array_Subscript_Node):
            self.data_ref_stack.append(node.parent_ref.name)
        else:
            raise TypeError("Error in code generation, expected Name_Node or Array_Subscript_Node in dataref parent")

        ret=part1 + "." + self.write_code(node.part_ref)
        self.data_ref_stack.pop()
        return ret

    def arraysub2string(self, node: ast_internal_classes.Array_Subscript_Node):
        local_name=node.name.name
        local_name_node=node.name
        #special handling if the array is in a structure - we must get the view to the member
        if len(self.data_ref_stack)>0:
            name_prefix=""
            for i in self.data_ref_stack:
                name_prefix+=self.write_code(i)+"_"
            local_name=name_prefix+local_name    
        if self.mapping.get(self.sdfg).get(local_name) is not None:
            if self.sdfg.arrays.get(self.mapping.get(self.sdfg).get(local_name)) is not None:
                arr = self.sdfg.arrays[self.mapping.get(self.sdfg).get(local_name)]
                if arr.shape is None or (len(arr.shape) == 1 and arr.shape[0] == 1):
                    return self.write_code(local_name_node)
            else:
                raise NameError("Variable name not found: ", node.name.name) 
        else:
            raise NameError("Variable name not found: ", node.name.name)
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
        if hasattr(node, "isStructMember"):
            if node.isStructMember:
                return node.name

        if self.rename_dict is not None and str(name) in self.rename_dict:
            return self.write_code(self.rename_dict[str(name)])
        if self.placeholders.get(name) is not None:
            location = self.placeholders.get(name)
            sdfg_name = self.mapping.get(self.sdfg).get(location[0])
            if sdfg_name is None:
                return name
            else:
                if self.sdfg.arrays[sdfg_name].shape is None or (
                        len(self.sdfg.arrays[sdfg_name].shape) == 1 and self.sdfg.arrays[sdfg_name].shape[0] == 1):
                    return "1"
                size = self.sdfg.arrays[sdfg_name].shape[location[1]]
                return self.write_code(str(size))

        if self.placeholders_offsets.get(name) is not None:
            location = self.placeholders_offsets.get(name)
            sdfg_name = self.mapping.get(self.sdfg).get(location[0])
            if sdfg_name is None:
                return name
            else:
                if self.sdfg.arrays[sdfg_name].shape is None or (
                        len(self.sdfg.arrays[sdfg_name].shape) == 1 and self.sdfg.arrays[sdfg_name].shape[0] == 1):
                    return "0"
                offset = self.sdfg.arrays[sdfg_name].offset[location[1]]
                return self.write_code(str(offset))
        if self.sdfg is not None:
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
        # Typecheck and crash early if unexpected.
        assert hasattr(node, 'value')
        lit = node.value
        assert isinstance(lit, str)

        # Fortran "real literals" may have an additional suffix at the end.
        # Examples:
        # valid: 1.0 => 1
        # valid: 1. => 1
        # valid: 1.e5 => 1e5
        # valid: 1.d5 => 1e5
        # valid: 1._kinder => 1 (precondition: somewhere earlier, `integer, parameter :: kinder=8`)
        # valid: 1.e5_kinder => 1e5
        # not valid: 1.d5_kinder => 1e5
        # TODO: Is there a complete spec of the structure of real literals?
        if '_' in lit:
            # First, deal with kind specification and remove it altogether, since we know the type anyway.
            parts = lit.split('_')
            assert 1 <= len(parts) <= 2, f"{lit} is not a valid fortran literal."
            lit = parts[0]
            assert 'd' not in lit, f"{lit} is not a valid fortran literal."
        if 'd' in lit:
            # Again, since we know the type anyway, here we just make the s/d/e/ replacement.
            lit = lit.replace('d', 'e')
        return f"{float(lit)}"

    def doublelit2string(self, node: ast_internal_classes.Double_Literal_Node):

        return "".join(map(str, node.value))

    def charlit2string(self, node: ast_internal_classes.Char_Literal_Node):
        return "".join(map(str, node.value))

    def boollit2string(self, node: ast_internal_classes.Bool_Literal_Node):
        assert node.value in {'0', '1'},\
            f"`{node.value}` is not a valid respresentation: use `0` for falsey values, and `1` for truthy values."
        return node.value

    def unop2string(self, node: ast_internal_classes.UnOp_Node):
        op = node.op
        if op == '.NOT.':
            # NOTE: DaCe cannot handle boolean expressions correctly, so this workaround.
            return '(1 - ' + self.write_code(node.lval) + ')'
        return op + self.write_code(node.lval)

    def parenthesis2string(self, node: ast_internal_classes.Parenthesis_Expr_Node):
        return "(" + self.write_code(node.expr) + ")"

    def call2string(self, node: ast_internal_classes.Call_Expr_Node):
        # This is a replacement for the epsilon function in fortran
        if node.name.name == "__dace_epsilon":
            return str(finf(fl).eps)
        if node.name.name == "pow":
            return "( " + self.write_code(node.args[0]) + " ** " + self.write_code(node.args[1]) + "  )"
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
        # TODO Add list of missing operators

        left = self.write_code(node.lval)
        right = self.write_code(node.rval)
        if op != "=":
            return "(" + left + op + right + ")"
        else:
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                TaskletWriter.TASKLETS_CREATED[self.sdfg][node.lval.name] = (right, node.rval)
            return left + op + right


def generate_memlet(op, top_sdfg, state, offset_normalization=False,mapped_name=None):
    if mapped_name is None:
        if state.name_mapping.get(top_sdfg).get(get_name(op)) is not None:
            shape = top_sdfg.arrays[state.name_mapping[top_sdfg][get_name(op)]].shape
        elif state.name_mapping.get(state.globalsdfg).get(get_name(op)) is not None:
            shape = state.globalsdfg.arrays[state.name_mapping[state.globalsdfg][get_name(op)]].shape
        else:
            raise NameError("Variable name not found: ", get_name(op))
    else:
        
        shape = top_sdfg.arrays[state.name_mapping[top_sdfg][mapped_name]].shape
        
    indices = []
    if isinstance(op, ast_internal_classes.Array_Subscript_Node):
        for idx, i in enumerate(op.indices):
            if isinstance(i, ast_internal_classes.ParDecl_Node):
                if i.type == 'ALL':
                    indices.append(None)
                else:
                    tw = TaskletWriter([], [], top_sdfg, state.name_mapping, placeholders=state.placeholders,
                                       placeholders_offsets=state.placeholders_offsets)
                    text_start = tw.write_code(i.range[0])
                    text_end = tw.write_code(i.range[1])
                    symb_start = sym.pystr_to_symbolic(text_start+"-1")
                    symb_end = sym.pystr_to_symbolic(text_end+"-1")
                    indices.append([symb_start, symb_end])
            else:
                tw = TaskletWriter([], [], top_sdfg, state.name_mapping, placeholders=state.placeholders,
                                   placeholders_offsets=state.placeholders_offsets)
                text = tw.write_code(i)
                # This might need to be replaced with the name in the context of the top/current sdfg
                indices.append([sym.pystr_to_symbolic(text), sym.pystr_to_symbolic(text)])
    memlet = '0'
    if len(shape) == 1:
        if shape[0] == 1:
            return memlet

    all_indices = indices + [None] * (len(shape) - len(indices))
    if offset_normalization:
        subset = subsets.Range(
            [(i[0], i[1], 1) if i is not None else (0, s - 1, 1) for i, s in zip(all_indices, shape)])
    else:
        subset = subsets.Range([(i[0], i[1], 1) if i is not None else (1, s, 1) for i, s in zip(all_indices, shape)])
    return subset


def generate_memlet_view(op, top_sdfg, state, offset_normalization=False,mapped_name=None,view_name=None,was_data_ref=False):
    if mapped_name is None:
        if state.name_mapping.get(top_sdfg).get(get_name(op)) is not None:
            shape = top_sdfg.arrays[state.name_mapping[top_sdfg][get_name(op)]].shape
        elif state.name_mapping.get(state.globalsdfg).get(get_name(op)) is not None:
            shape = state.globalsdfg.arrays[state.name_mapping[state.globalsdfg][get_name(op)]].shape
        else:
            raise NameError("Variable name not found: ", get_name(op))
    else:
        
        shape = top_sdfg.arrays[state.name_mapping[top_sdfg][mapped_name]].shape
        view_shape=top_sdfg.arrays[view_name].shape
        if len(view_shape)!=len(shape):
            was_data_ref=False
        else:
            was_data_ref=True
        
        
    indices = []
    skip=[]
    if isinstance(op, ast_internal_classes.Array_Subscript_Node):
        for idx, i in enumerate(op.indices):
            if isinstance(i, ast_internal_classes.ParDecl_Node):
                if i.type == 'ALL':
                    indices.append(None)
                else:
                    tw = TaskletWriter([], [], top_sdfg, state.name_mapping, placeholders=state.placeholders,
                                       placeholders_offsets=state.placeholders_offsets)
                    text_start = tw.write_code(i.range[0])
                    text_end = tw.write_code(i.range[1])
                    symb_start = sym.pystr_to_symbolic(text_start+"-1")
                    symb_end = sym.pystr_to_symbolic(text_end+"-1")
                    indices.append([symb_start, symb_end])
            else:
                tw = TaskletWriter([], [], top_sdfg, state.name_mapping, placeholders=state.placeholders,
                                       placeholders_offsets=state.placeholders_offsets)
                text = tw.write_code(i)
                symb = sym.pystr_to_symbolic(text)
                if was_data_ref:
                    indices.append([symb, symb])
                skip.append(idx)
    memlet = '0'
    if len(shape) == 1:
        if shape[0] == 1:
            return memlet
    tmp_shape = []
    for idx,i in enumerate(shape):
        if idx in skip:
            if was_data_ref:
                tmp_shape.append(1)
        else:
            tmp_shape.append(i)


    all_indices = indices + [None] * (len(shape) - len(indices)-len(skip))
    if offset_normalization:
        subset = subsets.Range(
            [(i[0], i[1], 1) if i is not None else (0, s - 1, 1) for i, s in zip(all_indices, tmp_shape)])
    else:
        subset = subsets.Range([(i[0], i[1], 1) if i is not None else (1, s, 1) for i, s in zip(all_indices, tmp_shape)])
    return subset

class ProcessedWriter(TaskletWriter):
    """
    This class is derived from the TaskletWriter class and is used to write the code of a tasklet that's on an interstate edge rather than a computational tasklet.
    :note The only differences are in that the names for the sdfg mapping are used, and that the indices are considered to be one-bases rather than zero-based. 
    """

    def __init__(self, sdfg: SDFG, mapping, placeholders, placeholders_offsets, rename_dict):
        self.sdfg = sdfg
        self.depth = 0
        self.mapping = mapping
        self.placeholders = placeholders
        self.placeholders_offsets = placeholders_offsets
        self.rename_dict = rename_dict
        self.data_ref_stack = []
        self.ast_elements = {
            ast_internal_classes.BinOp_Node: self.binop2string,
            ast_internal_classes.Actual_Arg_Spec_Node: self.actualarg2string,
            ast_internal_classes.Name_Node: self.name2string,
            ast_internal_classes.Name_Range_Node: self.namerange2string,
            ast_internal_classes.Int_Literal_Node: self.intlit2string,
            ast_internal_classes.Real_Literal_Node: self.floatlit2string,
            ast_internal_classes.Double_Literal_Node: self.doublelit2string,
            ast_internal_classes.Bool_Literal_Node: self.boollit2string,
            ast_internal_classes.Char_Literal_Node: self.charlit2string,
            ast_internal_classes.UnOp_Node: self.unop2string,
            ast_internal_classes.Array_Subscript_Node: self.arraysub2string,
            ast_internal_classes.Parenthesis_Expr_Node: self.parenthesis2string,
            ast_internal_classes.Call_Expr_Node: self.call2string,
            ast_internal_classes.ParDecl_Node: self.pardecl2string,
            ast_internal_classes.Data_Ref_Node: self.dataref2string,
        }

    def name2string(self, node: ast_internal_classes.Name_Node):
        name = node.name
        if name in self.rename_dict:
            return str(self.rename_dict[name])
        for i in self.sdfg.arrays:
            sdfg_name = self.mapping.get(self.sdfg).get(name)
            if sdfg_name == i:
                name = i
                break
        return name

    def arraysub2string(self, node: ast_internal_classes.Array_Subscript_Node):
        str_to_return = self.write_code(node.name) + "[(" + self.write_code(node.indices[0]) + ")"
        for i in node.indices[1:]:
            str_to_return += ",( " + self.write_code(i) + ")"
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


class FunctionSubroutineLister:
    def __init__(self):
        self.list_of_functions = []
        self.names_in_functions = {}
        self.list_of_subroutines = []
        self.names_in_subroutines = {}
        self.list_of_types = []
        self.names_in_types = {}

        self.list_of_module_vars = []
        self.interface_blocks: Dict[str, List[Name]] = {}

    def get_functions_and_subroutines(self, node: Base):
        for i in node.children:
            if isinstance(i, Subroutine_Stmt):
                subr_name = singular(children_of_type(i, Name)).string
                self.names_in_subroutines[subr_name] = list_descendent_names(node)
                self.names_in_subroutines[subr_name] += list_descendent_typenames(node)
                self.list_of_subroutines.append(subr_name)
            elif isinstance(i, Type_Declaration_Stmt):
                if isinstance(node, Specification_Part) and isinstance(node.parent, Module):
                    self.list_of_module_vars.append(i)
            elif isinstance(i, Derived_Type_Def):
                name = i.children[0].children[1].string
                self.names_in_types[name] = list_descendent_names(i)
                self.names_in_types[name] += list_descendent_typenames(i)
                self.list_of_types.append(name)


            elif isinstance(i, Function_Stmt):
                fn_name = singular(children_of_type(i, Name)).string
                self.names_in_functions[fn_name] = list_descendent_names(node)
                self.names_in_functions[fn_name] += list_descendent_typenames(node)
                self.list_of_functions.append(fn_name)
            elif isinstance(i, Interface_Block):
                name = None
                functions = []
                for j in i.children:
                    if isinstance(j, Interface_Stmt):
                        list_of_names = list_descendent_names(j)
                        if len(list_of_names) == 1:
                            name = list_of_names[0]
                    elif isinstance(j, Function_Body):
                        fn_stmt = singular(children_of_type(j, Function_Stmt))
                        fn_name = singular(children_of_type(fn_stmt, Name))
                        if fn_name not in functions:
                            functions.append(fn_name)
                    elif isinstance(j, Procedure_Stmt):
                        for k in j.children:
                            if k.__class__.__name__ == "Procedure_Name_List":
                                for n in children_of_type(k, Name):
                                    if n not in functions:
                                        functions.append(n)
                if len(functions) > 0:
                    if name is None:
                        # Anonymous interface can show up multiple times.
                        name = ''
                        if name not in self.interface_blocks:
                            self.interface_blocks[name] = []
                        self.interface_blocks[name].extend(functions)
                    else:
                        assert name not in self.interface_blocks
                        self.interface_blocks[name] = functions
            elif isinstance(i, Base):
                self.get_functions_and_subroutines(i)


def list_descendent_typenames(node: Base) -> List[str]:
    def _list_descendent_typenames(_node: Base, _list_of_names: List[str]) -> List[str]:
        for c in _node.children:
            if isinstance(c, Type_Name):
                if c.string not in _list_of_names:
                    _list_of_names.append(c.string)
            elif isinstance(c, Base):
                _list_descendent_typenames(c, _list_of_names)
        return _list_of_names

    return _list_descendent_typenames(node, [])


def list_descendent_names(node: Base) -> List[str]:
    def _list_descendent_names(_node: Base, _list_of_names: List[str]) -> List[str]:
        for c in _node.children:
            if isinstance(c, Name):
                if c.string not in _list_of_names:
                    _list_of_names.append(c.string)
            elif isinstance(c, Base):
                _list_descendent_names(c, _list_of_names)
        return _list_of_names

    return _list_descendent_names(node, [])


def get_defined_modules(node: Base) -> List[str]:
    def _get_defined_modules(_node: Base, _defined_modules: List[str]) -> List[str]:
        for m in _node.children:
            if isinstance(m, Module_Stmt):
                _defined_modules.extend(c.string for c in m.children if isinstance(c, Name))
            elif isinstance(m, Base):
                _get_defined_modules(m, _defined_modules)
        return _defined_modules

    return _get_defined_modules(node, [])


class UseAllPruneList:
    def __init__(self, module: str, identifiers: List[str]):
        """
        Keeps a list of referenced identifiers to intersect with the identifiers available in the module.
        WARN: The list of referenced identifiers is taken from the scope of the invocation of "use", but may not be
        entirely reliable. The parser should be able to function without this pruning (i.e., by really importing all).
        """
        self.module = module
        self.identifiers = identifiers


def get_used_modules(node: Base) -> Tuple[List[str], Dict[str, List[Union[UseAllPruneList, Base]]]]:
    used_modules: List[str] = []
    objects_in_use: Dict[str, List[Union[UseAllPruneList, Base]]] = {}

    def _get_used_modules(_node: Base):
        for m in _node.children:
            if not isinstance(m, Base):
                continue
            if not isinstance(m, Use_Stmt):
                # Subtree may have `use` statements.
                _get_used_modules(m)
                continue
            nature, _, mod_name, _, olist = m.children
            if nature is not None:
                # TODO: Explain why intrinsic nodes are avoided.
                if nature.string.lower() == "intrinsic":
                    continue

            mod_name = mod_name.string
            used_modules.append(mod_name)
            olist = atmost_one(children_of_type(m, 'Only_List'))
            if not olist:
                # TODO: Have better/clearer semantics.
                if mod_name not in objects_in_use:
                    objects_in_use[mod_name] = []
                # A list of identifiers referred in the context of `_node`. If it's a specification part, then the
                # context is its parent. If it's a module or a program, then `_node` itself is the context.
                refs = list_descendent_names(_node.parent if isinstance(_node, Specification_Part) else _node)
                # Add a special symbol to indicate that everything needs to be imported.
                objects_in_use[mod_name].append(UseAllPruneList(mod_name, refs))
            else:
                assert all(isinstance(c, (Name, Rename)) for c in olist.children)
                used = [c if isinstance(c, Name) else c.children[2] for c in olist.children]
                if not used:
                    continue
                # Merge all the used item in one giant list.
                if mod_name not in objects_in_use:
                    objects_in_use[mod_name] = []
                extend_with_new_items_from(objects_in_use[mod_name], used)
                assert len(set([str(o) for o in objects_in_use[mod_name]])) == len(objects_in_use[mod_name])

    _get_used_modules(node)
    return used_modules, objects_in_use


def parse_module_declarations(program):
    module_level_variables = {}

    for module in program.modules:

        module_name = module.name.name
        from dace.frontend.fortran.ast_transforms import ModuleVarsDeclarations

        visitor = ModuleVarsDeclarations()  # module_name)
        if module.specification_part is not None:
            visitor.visit(module.specification_part)
            module_level_variables = {**module_level_variables, **visitor.scope_vars}

    return module_level_variables


def validate_internal_ast(prog: ast_internal_classes.Program_Node):
    # A variable should not be redeclared in the same context.
    occurences = {}
    for fn in mywalk(prog,
                     (ast_internal_classes.Subroutine_Subprogram_Node, ast_internal_classes.Main_Program_Node)):
        # Execution-part is included in case some declaration is still there.
        decls = [v.name for d in chain(mywalk(fn.specification_part, ast_internal_classes.Decl_Stmt_Node),
                                       mywalk(fn.execution_part, ast_internal_classes.Decl_Stmt_Node))
                 for v in d.vardecl]
        counts = Counter(decls)
        counts = Counter({k: v for k, v in counts.items() if v > 1})
        if counts:
            occurences[fn.name.name] = counts
    if occurences:
        msg = '\n'.join(f"{k}: {c}" for k, c in occurences.items())
        raise ValueError(f"A variable should not be redeclared in the same context; got:\n{msg}")

    # Execution-part should not contain any declaration.
    occurences = {}
    for ex in mywalk(prog, ast_internal_classes.Execution_Part_Node):
        decls = [v.name for d in mywalk(ex, ast_internal_classes.Decl_Stmt_Node) for v in d.vardecl]
        if decls:
            occurences[ex.parent.name.name] = decls
    if occurences:
        msg = '\n'.join(f"{k}: {c}" for k, c in occurences.items())
        raise ValueError(f"execution-part should not contain any declaration; got\n{msg}")


def match_callsite_args_to_function_args(
        fn: ast_internal_classes.Subroutine_Subprogram_Node,
        call: ast_internal_classes.Call_Expr_Node) \
        -> Dict[str, ast_internal_classes.FNode]:
    fargs, cargs = fn.args, call.args
    out: Dict[str, ast_internal_classes.FNode] = {}

    # Once we start with keyword arguments, everything that comes after must be keyword arguments.
    kwzone = False
    while fargs and cargs:
        fa, fargs = fargs[0], fargs[1:]
        ca, cargs = cargs[0], cargs[1:]
        if isinstance(ca, ast_internal_classes.Actual_Arg_Spec_Node):
            kwzone = True
        if kwzone:
            # TODO: This should be the case but we do not handle it correctly when converting functions to subroutines.
            if isinstance(ca, ast_internal_classes.Actual_Arg_Spec_Node):
                kw, ca = ca.arg_name, ca.arg
                assert kw.name == fa.name
        out[fa.name] = ca
    # TODO: We assume any extra argument added by the current transforms (that called this helper) is added to the end
    #  of `fn.args` or `call.args`, whichever was visited first. But we still should check if other arguments are in
    #  order.
    return out


T = TypeVar('T')


def iter_fields(node: ast_internal_classes.FNode):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_attributes(node: ast_internal_classes.FNode):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._attributes``
    that is present on *node*.
    """
    for field in node._attributes:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node: ast_internal_classes.FNode):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """

    for name, field in iter_fields(node):
        # print("NASME:",name)
        if isinstance(field, ast_internal_classes.FNode):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast_internal_classes.FNode):
                    yield item


def mywalk(node,
           types: Union[None, Type[ast_internal_classes.FNode], Tuple[Type[ast_internal_classes.FNode], ...]] = None):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        if not types or isinstance(node, types):
            yield node


def singular(items: Iterator[T]) -> T:
    """
    Asserts that any given iterator or generator `items` has exactly 1 item and returns that.
    """
    it = atmost_one(items)
    assert it is not None, f"`items` must not be empty."
    return it


def atmost_one(items: Iterator[T]) -> Optional[T]:
    """
    Asserts that any given iterator or generator `items` has exactly 1 item and returns that.
    """
    # We might get one item.
    try:
        it = next(items)
    except StopIteration:
        # No items found.
        return None
    # But not another one.
    try:
        nit = next(items)
    except StopIteration:
        # I.e., we must have exhausted the iterator.
        return it
    raise ValueError(f"`items` must have at most 1 item, got: {it}, {nit}, ...")


def children_of_type(node: Base, typ: Union[str, Type[T], Tuple[Type, ...]]) -> Iterator[T]:
    """
    Returns a generator over the children of `node` that are of type `typ`.
    """
    if isinstance(typ, str):
        return (c for c in node.children if type(c).__name__ == typ)
    else:
        return (c for c in node.children if isinstance(c, typ))


def extend_with_new_items_from(lst: List[T], items: Iterable[T]):
    """
    Extends the list `lst` with new items from `items` (i.e., if it does not exist there already).
    """
    for it in items:
        if it not in lst:
            lst.append(it)


class TempName(object):
    _instance = None
    _counter = None

    def __new__(cls):
        if not getattr(cls, '_instance'):
            cls._instance = super(TempName, cls).__new__(cls)
            cls._instance._counter = 0
        return cls._instance

    @staticmethod
    def get_name(tag: str = 'tmp'):
        tmp = TempName()
        name, tmp._counter = f"{tag}_{tmp._counter}", tmp._counter + 1
        return name

def is_literal(node: ast_internal_classes.FNode) -> bool:
    return isinstance(node, (ast_internal_classes.Int_Literal_Node, ast_internal_classes.Double_Literal_Node, ast_internal_classes.Real_Literal_Node, ast_internal_classes.Bool_Literal_Node))

