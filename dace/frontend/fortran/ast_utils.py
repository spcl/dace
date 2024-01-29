# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.api import parse
import os
import sys
from fparser.common.readfortran import FortranStringReader, FortranFileReader

from dace.frontend.fortran import ast_components

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
from numpy import float64 as fl

from dace.frontend.fortran import ast_internal_classes
from typing import List, Set
import networkx as nx

fortrantypes2dacetypes = {
    "DOUBLE": dtypes.float64,
    "REAL": dtypes.float32,
    "INTEGER": dtypes.int32,
    "INTEGER8": dtypes.int64,
    "CHAR": dtypes.int8,
    "BOOL": dtypes.int32,  #This is a hack to allow fortran to pass through external C 
    "Unknown": dtypes.float64, # TMP hack unti lwe have a proper type inference
}


def eliminate_dependencies(dep_graph:nx.digraph.DiGraph):
    simple_graph = nx.DiGraph()
    simplify_order=(list(nx.topological_sort(dep_graph)))
    actually_used_in_module={}
    for i in simplify_order:
        
        res=dep_graph.nodes.get(i).get("info_list")
        if simple_graph.has_node(i)==True:
            simple_graph.add_node(i,info_list=res)
        out_edges=dep_graph.out_edges(i)
        in_edges=dep_graph.in_edges(i)
        in_names=[]
        out_names=[]
        
        for j in in_edges:
            in_names_local_obj=dep_graph.get_edge_data(j[0],j[1])["obj_list"]
            # if in_names_local_obj is not None:
            #   for k in in_names_local_obj:
            #     if k.__class__.__name__!="Name":
            #         print("Assumption failed: Object list contains non-name node")
            in_names_local=[]
            if in_names_local_obj is not None:
                for k in in_names_local_obj:
                    if k.__class__.__name__=="Name":
                        in_names_local.append(k.string)
                    elif k.__class__.__name__=="Rename":
                        in_names_local.append(k.children[1].string)    
            if in_names_local is not None:
                for k in in_names_local:
                    if k not in in_names:
                        in_names.append(k)
        for j in out_edges:
            out_names_local_obj=dep_graph.get_edge_data(j[0],j[1])["obj_list"]
            out_names_local=[]
            if out_names_local_obj is not None:
                for k in out_names_local_obj:
                    if k.__class__.__name__=="Name":
                        out_names_local.append(k.string)
                    elif k.__class__.__name__=="Rename":
                        out_names_local.append(k.children[1].string)        
            
            if out_names_local is not None:
                for k in out_names_local:
                    if k not in out_names:
                        out_names.append(k)
        actually_used=[]
        for name in in_names:
            actually_used.append(name)
        changed=True
        if res is not None:
         while(changed):
            changed=False 
            for used_name in actually_used:
                for var in res.list_of_module_vars:
                    for ii in var.children:
                        if ii.__class__.__name__=="Entity_Decl_List":
                                for iii in ii.children:
                                    if iii.__class__.__name__=="Entity_Decl":
                                        if iii.children[0].string==used_name:
                                            for j in var.children:
                                                #print("USED: "+ used_name)
                                                nl = NameLister()
                                                nl.get_names(iii)
                                                for nameininit in nl.list_of_names:
                                                    if nameininit not in actually_used:
                                                        actually_used.append(nameininit)
                                                        changed=True
                                                if j.__class__.__name__=="Declaration_Type_Spec":
                                                    for k in j.children:
                                                        if k.__class__.__name__=="Type_Name":
                                                            if k.string not in actually_used:
                                                                actually_used.append(k.string)
                                                                changed=True

                if used_name in res.list_of_types:
                    for used_in_type in res.names_in_types[used_name]:
                        if used_in_type not in actually_used and used_in_type in res.list_of_types:
                            actually_used.append(used_in_type)
                            changed=True
                if used_name in res.list_of_functions:
                    for used_in_function in res.names_in_functions[used_name]:
                        if used_in_function not in actually_used and (used_in_function in res.list_of_functions or used_in_function in res.list_of_subroutines):                            
                            actually_used.append(used_in_function)
                            changed=True            
                if used_name in res.list_of_subroutines:
                    for used_in_subroutine in res.names_in_subroutines[used_name]:
                        if used_in_subroutine not in actually_used and (used_in_subroutine in res.list_of_functions or used_in_subroutine in res.list_of_subroutines):                            
                            actually_used.append(used_in_subroutine)
                            changed=True
                           
        not_used=[]
        if res is not None:
            for j in out_names:
                if i==simplify_order[0]:
                    for tname in res.list_of_types:
                        if j in res.names_in_types[tname]:
                            if j not in actually_used:
                                actually_used.append(j)
                    for fname in res.list_of_functions:
                        if j in res.names_in_functions[fname]:
                            if j not in actually_used:
                                actually_used.append(j)
                    for sname in res.list_of_subroutines:
                        if j in res.names_in_subroutines[sname]:
                            if j not in actually_used:
                                actually_used.append(j)
                else:
                    changed=True
                    while(changed):
                        changed=False 
                        for tname in res.list_of_types:
                            if tname in in_names or tname in actually_used:
                                if j in res.names_in_types[tname]:
                                    if j not in actually_used:
                                        actually_used.append(j)
                                        changed=True
                        for fname in res.list_of_functions:
                            if fname in in_names or fname in actually_used:
                                if j in res.names_in_functions[fname]:
                                    if j not in actually_used:
                                        actually_used.append(j)
                                        changed=True
                                    for k in res.names_in_functions[fname]:
                                            if k in res.list_of_functions:
                                                if k not in actually_used:
                                                    actually_used.append(k)
                                                    changed=True
                                    for k in res.names_in_functions[fname]:
                                            if k in res.list_of_subroutines:
                                                if k not in actually_used:
                                                    actually_used.append(k)
                                                    changed=True                

                        for sname in res.list_of_subroutines:
                            if sname in in_names or sname in actually_used:
                                if j in res.names_in_subroutines[sname]:
                                    if j not in actually_used:
                                        actually_used.append(j)
                                        changed=True
                                    for k in res.names_in_subroutines[sname]:
                                            if k in res.list_of_functions:
                                                if k not in actually_used:
                                                    actually_used.append(k)
                                                    changed=True
                                    for k in res.names_in_subroutines[sname]:
                                            if k in res.list_of_subroutines:
                                                if k not in actually_used:
                                                    actually_used.append(k)
                                                    changed=True    


            
            for j in out_names:
                if j not in actually_used:
                    not_used.append(j)
            #if not_used!=[]:
            #print(i)                
            #print("NOT USED: "+ str(not_used))  
       
        out_edges=dep_graph.out_edges(i)
        out_names=[]
        for j in out_edges:
            out_names_local_obj=dep_graph.get_edge_data(j[0],j[1])["obj_list"]
            out_names_local=[]
            if out_names_local_obj is not None:
                for k in out_names_local_obj:
                    if k.__class__.__name__=="Name":
                        out_names_local.append(k.string)
                    elif k.__class__.__name__=="Rename":
                        out_names_local.append(k.children[1].string)   
                    
            new_out_names_local=[]
            if len(out_names_local)==0:
                continue
            for k in out_names_local:
                if k not in not_used:
                    for kk in out_names_local_obj:
                        if kk.__class__.__name__=="Name":
                            if kk.string==k:
                                new_out_names_local.append(kk)
                                break
                        elif kk.__class__.__name__=="Rename":
                            if kk.children[1].string==k:
                                new_out_names_local.append(kk)
                                break    
            if len(new_out_names_local)>0:
                if simple_graph.has_node(i)==False:
                    if i!=simplify_order[0]:              
                        continue
                    else:
                        simple_graph.add_node(i,info_list=res)
                if simple_graph.has_node(j[1])==False:
                    simple_graph.add_node(j[1])
                simple_graph.add_edge(i,j[1],obj_list=new_out_names_local)
        actually_used_in_module[i]=actually_used        
    print(simple_graph)
    return simple_graph,actually_used_in_module
    


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
    return src    


def add_memlet_write(substate: SDFGState, var_name: str, tasklet: Tasklet, source_conn: str, memlet_range: str):
    dst = substate.add_write(var_name)
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
            ast_internal_classes.Char_Literal_Node: self.charlit2string,
            ast_internal_classes.UnOp_Node: self.unop2string,
            ast_internal_classes.Array_Subscript_Node: self.arraysub2string,
            ast_internal_classes.Parenthesis_Expr_Node: self.parenthesis2string,
            ast_internal_classes.Call_Expr_Node: self.call2string,
            ast_internal_classes.ParDecl_Node: self.pardecl2string,
            ast_internal_classes.Data_Ref_Node: self.dataref2string,
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

    def dataref2string(self, node: ast_internal_classes.Data_Ref_Node):
        return self.write_code(node.parent_ref) + "." + self.write_code(node.part_ref)

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
    
    def charlit2string(self, node: ast_internal_classes.Char_Literal_Node):
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
        if node.name.name == "pow":
            return " ( " + self.write_code(node.args[0]) + " ** " + self.write_code(node.args[1]) + "  ) "
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



class UseModuleLister:
    def __init__(self):
        self.list_of_modules = []
        self.objects_in_use={}

    def get_used_modules(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Use_Stmt":
                if i.children[0] is not None:
                    if i.children[0].string.lower()=="intrinsic":
                        continue
                for j in i.children:
                    if j.__class__.__name__ == "Name":
                        self.list_of_modules.append(j.string)
                        for k in i.children:
                            if k.__class__.__name__ == "Only_List":
                                self.objects_in_use[j.string] = k

            else:
                self.get_used_modules(i)


class DefModuleLister:
    def __init__(self):
        self.list_of_modules = []

    def get_defined_modules(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Module_Stmt":
                for j in i.children:
                    if j.__class__.__name__ == "Name":
                        self.list_of_modules.append(j.string)
            else:
                self.get_defined_modules(i)


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


class UseModuleLister:
    def __init__(self):
        self.list_of_modules = []
        self.objects_in_use={}

    def get_used_modules(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Use_Stmt":
                if i.children[0] is not None:
                    if i.children[0].string.lower()=="intrinsic":
                        continue
                for j in i.children:
                    if j.__class__.__name__ == "Name":
                        self.list_of_modules.append(j.string)
                        for k in i.children:
                            if k.__class__.__name__ == "Only_List":
                                self.objects_in_use[j.string] = k

            else:
                self.get_used_modules(i)


class FunctionSubroutineLister:
    def __init__(self):
        self.list_of_functions = []
        self.names_in_functions = {}
        self.list_of_subroutines=[]
        self.names_in_subroutines={}
        self.list_of_types=[]
        self.names_in_types={}
        self.list_of_module_vars=[]
        

    def get_functions_and_subroutines(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Subroutine_Stmt":
                for j in i.children:
                    if j.__class__.__name__ == "Name":
                        nl = NameLister()
                        nl.get_names(node)
                        tnl = TypeNameLister()
                        tnl.get_typenames(node)
                        self.names_in_subroutines[j.string] = nl.list_of_names
                        self.names_in_subroutines[j.string] += tnl.list_of_typenames
                        self.list_of_subroutines.append(j.string)
            elif i.__class__.__name__ == "Type_Declaration_Stmt":
                if node.__class__.__name__ == "Specification_Part":
                    if node.parent.__class__.__name__ == "Module":
                        
                        self.list_of_module_vars.append(i)
                        

            elif i.__class__.__name__ == "Derived_Type_Def":
                        name=i.children[0].children[1].string
                        nl = NameLister()
                        nl.get_names(i)
                        tnl = TypeNameLister()
                        tnl.get_typenames(i)
                        self.names_in_types[name] = nl.list_of_names
                        self.names_in_types[name] += tnl.list_of_typenames
                        self.list_of_types.append(name)            
            elif i.__class__.__name__ == "Function_Stmt":
                for j in i.children:
                    if j.__class__.__name__ == "Name":
                        nl = NameLister()
                        nl.get_names(node)
                        tnl = TypeNameLister()
                        tnl.get_typenames(node)
                        self.names_in_functions[j.string] = nl.list_of_names
                        self.names_in_functions[j.string] += tnl.list_of_typenames
                        self.list_of_functions.append(j.string)
                
            else:
                self.get_functions_and_subroutines(i)

class TypeNameLister:
    def __init__(self):
        self.list_of_typenames = []
        
        

    def get_typenames(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Type_Name":
                if i.string not in self.list_of_typenames:
                    self.list_of_typenames.append(i.string)    
            else:
                self.get_typenames(i)

class NameLister:
    def __init__(self):
        self.list_of_names = []
        
        

    def get_names(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Name":
                if i.string not in self.list_of_names:
                    self.list_of_names.append(i.string)    
            else:
                self.get_names(i)

class DefModuleLister:
    def __init__(self):
        self.list_of_modules = []

    def get_defined_modules(self, node):
        if node is None:
            return
        if not hasattr(node, "children"):
            return
        for i in node.children:
            if i.__class__.__name__ == "Module_Stmt":
                for j in i.children:
                    if j.__class__.__name__ == "Name":
                        self.list_of_modules.append(j.string)
            else:
                self.get_defined_modules(i)

def parse_module_declarations(internal_ast: ast_components.InternalFortranAst, ast, parsed_modules):

    module_level_variables = {}

    for module_name, module in parsed_modules.items():

        from dace.frontend.fortran.ast_transforms import ModuleVarsDeclarations

        visitor = ModuleVarsDeclarations() #module_name)
        if module.specification_part is not None:
            visitor.visit(module.specification_part)
            module_level_variables = {**module_level_variables, **visitor.scope_vars}

    return module_level_variables
