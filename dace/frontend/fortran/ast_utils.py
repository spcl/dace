# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.
from itertools import chain
from typing import List, Set, Iterator, Type, TypeVar, Dict, Tuple, Iterable, Union

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
    "BOOL": dtypes.int32,  # This is a hack to allow fortran to pass through external C
    "Unknown": dtypes.float64,  # TMP hack unti lwe have a proper type inference
}


def eliminate_dependencies(dep_graph: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, List]]:
    simple_graph = nx.DiGraph()
    simplify_order = list(nx.topological_sort(dep_graph))
    actually_used_in_module = {}
    for i in simplify_order:
        res = dep_graph.nodes.get(i).get("info_list")
        if simple_graph.has_node(i):
            simple_graph.add_node(i, info_list=res)
        in_names = []
        out_names = []

        for j in dep_graph.in_edges(i):
            in_names_local_obj = dep_graph.get_edge_data(j[0], j[1])["obj_list"]
            # if in_names_local_obj is not None:
            #   for k in in_names_local_obj:
            #     if k.__class__.__name__!="Name":
            #         print("Assumption failed: Object list contains non-name node")
            in_names_local = []
            if in_names_local_obj is not None:
                for k in in_names_local_obj:
                    if isinstance(k, Name):
                        in_names_local.append(k.string)
                    elif isinstance(k, Rename):
                        in_names_local.append(k.children[2].string)
                        in_names_local.append(k.children[1].string)
            if in_names_local is not None:
                for k in in_names_local:
                    if k not in in_names:
                        in_names.append(k)
        for _, dep in dep_graph.out_edges(i):
            out_names_local_obj = dep_graph.get_edge_data(i, dep).get('obj_list', [])
            out_names_local = []
            dep_info = dep_graph.nodes.get(dep).get('info_list')
            canonical_used_objs = []
            for used_obj in out_names_local_obj:
                if isinstance(used_obj, UseAllPruneList):
                    # We have a special type indicating that everything needs to be imported.
                    if not dep_info:
                        # If there is nothing to import, then do nothing.
                        continue
                    assert used_obj.module == dep
                    assert isinstance(dep_info, FunctionSubroutineLister)
                    list_of_module_vars = []
                    for type_stmt in dep_info.list_of_module_vars:
                        for entity_decl_list in children_of_type(type_stmt, 'Entity_Decl_List'):
                            for entity_decl in children_of_type(entity_decl_list, Entity_Decl):
                                entity_name = singular(children_of_type(entity_decl, Name)).string
                                if entity_name in used_obj.identifiers:
                                    list_of_module_vars.append(entity_name)
                    used_names = list(Name(name) for name in chain(dep_info.list_of_functions,
                                                                   dep_info.list_of_subroutines,
                                                                   list_of_module_vars,
                                                                   dep_info.list_of_types))
                    extend_with_new_items_from(canonical_used_objs, used_names)
                else:
                    extend_with_new_items_from(canonical_used_objs, [used_obj])
            out_names_local_obj = canonical_used_objs
            for k in out_names_local_obj:
                if isinstance(k, Name):
                    out_names_local.append(k.string)
                elif isinstance(k, Rename):
                    out_names_local.append(k.children[1].string)
                    out_names_local.append(k.children[2].string)
            for k in out_names_local:
                if k not in out_names:
                    out_names.append(k)
        actually_used = []
        for name in in_names:
            actually_used.append(name)
        changed = True
        if res is not None:
            while changed:
                changed = False
                for used_name in actually_used:
                    for var in res.list_of_module_vars:
                        for ii in var.children:
                            # Support scenario where an imported symbol appears in kind selector
                            # Example: REAL(KIND=JPRB) :: R2ES
                            # Where JPRB is imported from another module.
                            if isinstance(ii, Intrinsic_Type_Spec):
                                for iii in ii.children:
                                    if isinstance(iii, Kind_Selector):
                                        name_to_check = iii.children[1].string
                                        if name_to_check not in actually_used:
                                            actually_used.append(name_to_check)
                            elif ii.__class__.__name__ == "Entity_Decl_List":
                                for iii in ii.children:
                                    if isinstance(iii, Entity_Decl):
                                        name_to_check = iii.children[0].string
                                        for type_name in res.list_of_types:
                                            for used_in_type in res.names_in_types[type_name]:
                                                if name_to_check == used_in_type:
                                                    if name_to_check not in actually_used:
                                                        actually_used.append(name_to_check)
                                        for type_name in res.list_of_subroutines:
                                            for used_in_type in res.names_in_subroutines[type_name]:
                                                if name_to_check == used_in_type:
                                                    if name_to_check not in actually_used:
                                                        actually_used.append(name_to_check)

                                        if iii.children[0].string == used_name:
                                            for j in var.children:
                                                list_of_names = list_descendent_names(iii)
                                                for nameininit in list_of_names:
                                                    if nameininit not in actually_used:
                                                        actually_used.append(nameininit)
                                                        changed = True
                                                if isinstance(j, Declaration_Type_Spec):
                                                    for k in j.children:
                                                        if isinstance(k, Type_Name):
                                                            if k.string not in actually_used:
                                                                actually_used.append(k.string)
                                                                changed = True

                    if used_name in res.list_of_types:
                        for used_in_type in res.names_in_types[used_name]:
                            if used_in_type not in actually_used and used_in_type in res.list_of_types:
                                actually_used.append(used_in_type)
                                changed = True
                    if used_name in res.list_of_functions:
                        for used_in_function in res.names_in_functions[used_name]:
                            if used_in_function not in actually_used and (
                                    used_in_function in res.list_of_functions or used_in_function in res.list_of_subroutines):
                                actually_used.append(used_in_function)
                                changed = True
                    if used_name in res.list_of_subroutines:
                        for used_in_subroutine in res.names_in_subroutines[used_name]:
                            if used_in_subroutine not in actually_used and (
                                    used_in_subroutine in res.list_of_functions or used_in_subroutine in res.list_of_subroutines):
                                actually_used.append(used_in_subroutine)
                                changed = True

        not_used = []
        if res is not None:
            for j in out_names:
                if i == simplify_order[0]:
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
                    changed = True
                    while changed:
                        changed = False
                        for tname in res.list_of_types:
                            if tname in in_names or tname in actually_used:
                                if j in res.names_in_types[tname]:
                                    if j not in actually_used:
                                        actually_used.append(j)
                                        changed = True
                        for fname in res.list_of_functions:
                            if fname in in_names or fname in actually_used:
                                if j in res.names_in_functions[fname]:
                                    if j not in actually_used:
                                        actually_used.append(j)
                                        changed = True
                                    for k in res.names_in_functions[fname]:
                                        if k in res.list_of_functions:
                                            if k not in actually_used:
                                                actually_used.append(k)
                                                changed = True
                                    for k in res.names_in_functions[fname]:
                                        if k in res.list_of_subroutines:
                                            if k not in actually_used:
                                                actually_used.append(k)
                                                changed = True

                        for sname in res.list_of_subroutines:
                            if sname in in_names or sname in actually_used:
                                if j in res.names_in_subroutines[sname]:
                                    if j not in actually_used:
                                        actually_used.append(j)
                                        changed = True
                                    for k in res.names_in_subroutines[sname]:
                                        if k in res.list_of_functions:
                                            if k not in actually_used:
                                                actually_used.append(k)
                                                changed = True
                                    for k in res.names_in_subroutines[sname]:
                                        if k in res.list_of_subroutines:
                                            if k not in actually_used:
                                                actually_used.append(k)
                                                changed = True

            for j in out_names:
                if j not in actually_used:
                    not_used.append(j)
            # if not_used!=[]:
            # print(i)
            # print("NOT USED: "+ str(not_used))

        for _, dep in dep_graph.out_edges(i):
            out_names_local_obj = dep_graph.get_edge_data(i, dep).get('obj_list', [])
            out_names_local = []
            dep_info = dep_graph.nodes.get(dep).get('info_list')
            canonical_used_objs = []
            for used_obj in out_names_local_obj:
                if isinstance(used_obj, UseAllPruneList):
                    # We have a special type indicating that everything needs to be imported.
                    if not dep_info:
                        # If there is nothing to import, then do nothing.
                        continue
                    assert used_obj.module == dep
                    assert isinstance(dep_info, FunctionSubroutineLister)
                    list_of_module_vars = []
                    for type_stmt in dep_info.list_of_module_vars:
                        for entity_decl_list in children_of_type(type_stmt, 'Entity_Decl_List'):
                            for entity_decl in children_of_type(entity_decl_list, Entity_Decl):
                                entity_name = singular(children_of_type(entity_decl, Name)).string
                                if entity_name in used_obj.identifiers:
                                    list_of_module_vars.append(entity_name)
                    used_names = list(Name(name) for name in chain(dep_info.list_of_functions,
                                                                   dep_info.list_of_subroutines,
                                                                   list_of_module_vars,
                                                                   dep_info.list_of_types))
                    extend_with_new_items_from(canonical_used_objs, used_names)
                else:
                    extend_with_new_items_from(canonical_used_objs, [used_obj])
            out_names_local_obj = canonical_used_objs
            for k in out_names_local_obj:
                if isinstance(k, Name):
                    out_names_local.append(k.string)
                elif isinstance(k, Rename):
                    out_names_local.append(k.children[1].string)
                    out_names_local.append(k.children[2].string)

            new_out_names_local = []
            if len(out_names_local) == 0:
                continue
            for k in out_names_local:
                if k not in not_used:
                    for kk in out_names_local_obj:
                        if isinstance(kk, Name):
                            if kk.string == k:
                                new_out_names_local.append(kk)
                                break
                        elif isinstance(kk, Rename):
                            if kk.children[1].string == k:
                                new_out_names_local.append(kk)
                                break
            if len(new_out_names_local) > 0:
                if not simple_graph.has_node(i):
                    if i != simplify_order[0]:
                        continue
                    else:
                        simple_graph.add_node(i, info_list=res)
                if not simple_graph.has_node(dep):
                    simple_graph.add_node(dep)
                simple_graph.add_edge(i, dep, obj_list=new_out_names_local)
        actually_used_in_module[i] = actually_used
        # print(simple_graph)

    # Verify that all `UseAllPruneList` items are accounted for in the simplified graph.
    for _, _, uses in simple_graph.edges.data('obj_list'):
        assert not uses or all(isinstance(u, Base) for u in uses)

    return simple_graph, actually_used_in_module


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
    if isinstance(node, ast_internal_classes.Name_Node):
        return node.name
    elif isinstance(node, ast_internal_classes.Array_Subscript_Node):
        return node.name.name
    elif isinstance(node, ast_internal_classes.Data_Ref_Node):
        view_name = node.parent_ref.name
        while isinstance(node.part_ref, ast_internal_classes.Data_Ref_Node):
            if isinstance(node.part_ref.parent_ref, ast_internal_classes.Name_Node):
                view_name = view_name + "_" + node.part_ref.parent_ref.name
            elif isinstance(node.part_ref.parent_ref, ast_internal_classes.Array_Subscript_Node):
                view_name = view_name + "_" + node.part_ref.parent_ref.name.name
            node = node.part_ref
        view_name = view_name + "_" + get_name(node.part_ref)
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
        }

    def pardecl2string(self, node: ast_internal_classes.ParDecl_Node):
        # At this point in the process, the should not be any ParDecl nodes left in the AST - they should have been replaced by the appropriate ranges
        # raise NameError("Error in code generation")
        return f"ERROR{node.type}"

    def actualarg2string(self, node: ast_internal_classes.Actual_Arg_Spec_Node):
        return self.write_code(node.arg)

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
                raise NameError("Error in code generation")
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
        # TODO Add list of missing operators

        left = self.write_code(node.lval)
        right = self.write_code(node.rval)
        if op != "=":
            return "(" + left + op + right + ")"
        else:
            return left + op + right


def generate_memlet(op, top_sdfg, state, offset_normalization=False):
    if state.name_mapping.get(top_sdfg).get(get_name(op)) is not None:
        shape = top_sdfg.arrays[state.name_mapping[top_sdfg][get_name(op)]].shape
    elif state.name_mapping.get(state.globalsdfg).get(get_name(op)) is not None:
        shape = state.globalsdfg.arrays[state.name_mapping[state.globalsdfg][get_name(op)]].shape
    else:
        raise NameError("Variable name not found: ", get_name(op))
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
                    symb_start = sym.pystr_to_symbolic(text_start)
                    symb_end = sym.pystr_to_symbolic(text_end)
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
        self.functions_and_subroutines_in_types= []
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
            if m.children[0] is not None:
                # TODO: Explain why intrinsic nodes are avoided.
                if m.children[0].string.lower() == "intrinsic":
                    continue

            mod_name = singular(children_of_type(m, Name)).string
            used_modules.append(mod_name)
            olist = list(children_of_type(m, 'Only_List'))
            assert len(olist) <= 1, f"{m} can have at most one only-list, got {olist}."
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
                olist = olist[0]
                if not olist.children:
                    continue
                # Merge all the used item in one giant list.
                if mod_name not in objects_in_use:
                    objects_in_use[mod_name] = []
                extend_with_new_items_from(objects_in_use[mod_name], olist.children)
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


T = TypeVar('T')


def singular(items: Iterator[T]) -> T:
    """
    Asserts that any given iterator or generator `items` has exactly 1 item and returns that.
    """
    # We should be able to get one item.
    try:
        it = next(items)
    except StopIteration:
        # No items found.
        raise
    # But not another one.
    try:
        nit = next(items)
    except StopIteration:
        # I.e., we must have exhausted the iterator.
        return it
    raise ValueError(f"`items` must have only 1 item, got: {it}, {nit}, ...")


def children_of_type(node: Base, typ: Union[str, Type[T]]) -> Iterator[T]:
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
