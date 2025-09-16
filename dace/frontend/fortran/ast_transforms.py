# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import re
import warnings
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Set, Union, Type

import sympy as sp

from dace import symbolic as sym
from dace.frontend.fortran import ast_internal_classes, ast_utils
from dace.frontend.fortran.ast_desugaring import ConstTypeInjection
from dace.frontend.fortran.ast_internal_classes import Var_Decl_Node, Name_Node, Int_Literal_Node, Data_Ref_Node, \
    Execution_Part_Node, Array_Subscript_Node, Bool_Literal_Node
from dace.frontend.fortran.ast_utils import mywalk, iter_fields, iter_attributes, TempName, singular, atmost_one, \
    match_callsite_args_to_function_args


class NeedsTypeInferenceException(BaseException):

    def __init__(self, func_name, line_number):
        self.line_number = line_number
        self.func_name = func_name


class Structure:

    def __init__(self, name: str):
        self.vars: Dict[str, Union[ast_internal_classes.Symbol_Decl_Node, ast_internal_classes.Var_Decl_Node]] = {}
        self.name = name


class Structures:

    def __init__(self, definitions: List[ast_internal_classes.Derived_Type_Def_Node]):
        self.structures: Dict[str, Structure] = {}
        self.parse(definitions)

    def parse(self, definitions: List[ast_internal_classes.Derived_Type_Def_Node]):

        for structure in definitions:

            struct = Structure(name=structure.name.name)
            if structure.component_part is not None:
                if structure.component_part.component_def_stmts is not None:
                    for statement in structure.component_part.component_def_stmts:
                        if isinstance(statement, ast_internal_classes.Data_Component_Def_Stmt_Node):
                            for var in statement.vars.vardecl:
                                struct.vars[var.name] = var

            self.structures[structure.name.name] = struct

    def is_struct(self, type_name: str):
        return type_name in self.structures

    def get_definition(self, type_name: str):
        return self.structures[type_name]

    def find_definition(self,
                        scope_vars,
                        node: ast_internal_classes.Data_Ref_Node,
                        variable_name: Optional[ast_internal_classes.Name_Node] = None):

        # we assume starting from the top (left-most) data_ref_node
        # for struct1 % struct2 % struct3 % var
        # we find definition of struct1, then we iterate until we find the var

        # find the top structure
        top_ref = node
        while isinstance(top_ref.parent_ref, ast_internal_classes.Data_Ref_Node):
            top_ref = top_ref.parent_ref

        struct_type = scope_vars.get_var(node.parent, ast_utils.get_name(top_ref.parent_ref)).type
        struct_def = self.structures[struct_type]

        # cur_node = node
        cur_node = top_ref

        while True:

            prev_node = cur_node
            cur_node = cur_node.part_ref

            if isinstance(cur_node, ast_internal_classes.Array_Subscript_Node):
                struct_def = self.structures[struct_type]
                cur_var = struct_def.vars[cur_node.name.name]
                node = cur_node
                break

            elif isinstance(cur_node, ast_internal_classes.Name_Node):
                struct_def = self.structures[struct_type]
                cur_var = struct_def.vars[cur_node.name]
                break

            if isinstance(cur_node.parent_ref.name, ast_internal_classes.Name_Node):

                if variable_name is not None and cur_node.parent_ref.name.name == variable_name.name:
                    return struct_def, struct_def.vars[cur_node.parent_ref.name.name], prev_node

                struct_type = struct_def.vars[cur_node.parent_ref.name.name].type
            else:

                if variable_name is not None and cur_node.parent_ref.name == variable_name.name:
                    return struct_def, struct_def.vars[cur_node.parent_ref.name], prev_node

                struct_type = struct_def.vars[cur_node.parent_ref.name].type
            struct_def = self.structures[struct_type]

        return struct_def, cur_var, prev_node


class NodeVisitor(object):
    """
    A base node visitor class for Fortran ASTs.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    def visit(self, node: ast_internal_classes.FNode):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast_internal_classes.FNode):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value)


class NodeTransformer(NodeVisitor):
    """
    A base node visitor that walks the abstract syntax tree and allows
    modification of nodes.
    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace old nodes.
    """

    def as_list(self, x):
        if isinstance(x, list):
            return x
        if x is None:
            return []
        return [x]

    def generic_visit(self, node: ast_internal_classes.FNode):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_internal_classes.FNode):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_internal_classes.FNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_internal_classes.FNode):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class Flatten_Classes(NodeTransformer):

    def __init__(self, classes: List[ast_internal_classes.Derived_Type_Def_Node]):
        self.classes = classes
        self.current_class = None

    def visit_Derived_Type_Def_Node(self, node: ast_internal_classes.Derived_Type_Def_Node):
        self.current_class = node
        return_node = self.generic_visit(node)
        # self.current_class=None
        return return_node

    def visit_Module_Node(self, node: ast_internal_classes.Module_Node):
        self.current_class = None
        return self.generic_visit(node)

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        node: ast_internal_classes.Subroutine_Subprogram_Node = self.generic_visit(node)
        print("Subroutine: ", node.name.name)
        if not self.current_class:
            return node
        for i in self.classes:
            if not i.is_class or i.name.name != self.current_class.name.name:
                continue
            for j in i.procedure_part.procedures:
                name = None
                if j.name.name == node.name.name:
                    name = i.name.name + "_" + node.name.name
                elif hasattr(j, "args") and j.args[2] is not None:
                    if j.args[2].name == node.name.name:
                        name = i.name.name + "_" + j.name.name
                if name:
                    return ast_internal_classes.Subroutine_Subprogram_Node(
                        name=ast_internal_classes.Name_Node(name=name, type=node.type),
                        args=node.args,
                        specification_part=node.specification_part,
                        execution_part=node.execution_part,
                        internal_subprogram_part=node.execution_part,
                        mandatory_args_count=node.mandatory_args_count,
                        optional_args_count=node.optional_args_count,
                        elemental=node.elemental,
                        line_number=node.line_number)

        return node

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if not self.current_class:
            return self.generic_visit(node)
        for i in self.classes:
            if not i.is_class or i.name.name != self.current_class.name.name:
                continue
            for j in i.procedure_part.procedures:
                if j.name.name == node.name.name:
                    name = i.name.name + "_" + j.name.name
                    return ast_internal_classes.Call_Expr_Node(name=ast_internal_classes.Name_Node(
                        name=name, type=node.type, args=node.args, line_number=node.line_number),
                                                               args=node.args,
                                                               type=node.type,
                                                               subroutine=node.subroutine,
                                                               line_number=node.line_number,
                                                               parent=node.parent)
        return self.generic_visit(node)


class FindFunctionAndSubroutines(NodeVisitor):
    """
    Finds all function and subroutine names in the AST
    :return: List of names
    """

    def __init__(self):
        self.names: List[ast_internal_classes.Name_Node] = []
        self.module_based_names: Dict[str, List[ast_internal_classes.Name_Node]] = {}
        self.nodes: Dict[str, ast_internal_classes.FNode] = {}
        self.iblocks: Dict[str, List[str]] = {}
        self.current_module = "_dace_default"
        self.module_based_names[self.current_module] = []

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        ret = node.name
        ret.elemental = node.elemental
        self.names.append(ret)
        assert ret.name not in self.nodes
        self.nodes[ret.name] = node
        self.module_based_names[self.current_module].append(ret)
        if node.internal_subprogram_part:
            self.visit(node.internal_subprogram_part)

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):
        ret = node.name
        ret.elemental = node.elemental
        self.names.append(ret)
        assert ret.name not in self.nodes
        self.nodes[ret.name] = node
        self.module_based_names[self.current_module].append(ret)
        if node.internal_subprogram_part:
            self.visit(node.internal_subprogram_part)

    def visit_Module_Node(self, node: ast_internal_classes.Module_Node):
        self.iblocks.update(node.interface_blocks)
        self.current_module = node.name.name
        self.module_based_names[self.current_module] = []
        self.generic_visit(node)

    @staticmethod
    def from_node(node: ast_internal_classes.FNode) -> 'FindFunctionAndSubroutines':
        v = FindFunctionAndSubroutines()
        v.visit(node)
        return v


class FindNames(NodeVisitor):

    def __init__(self):
        self.names: List[str] = []

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        self.names.append(node.name)

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.names.append(node.name.name)
        for i in node.indices:
            self.visit(i)


class FindDefinedNames(NodeVisitor):

    def __init__(self):
        self.names: List[str] = []
        self.specs: {} = {}

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        self.names.append(node.name)
        self.specs[node.name] = node


class FindInputs(NodeVisitor):
    """
    Finds all inputs (reads) in the AST node and its children
    :return: List of names
    """

    def __init__(self):

        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        if node.specification_part is not None:
            self.visit(node.specification_part)
        if node.execution_part is not None:
            self.visit(node.execution_part)

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        self.nodes.append(node)

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.nodes.append(node.name)
        for i in node.indices:
            self.visit(i)

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):
        if isinstance(node.parent_ref, ast_internal_classes.Name_Node):
            self.nodes.append(node.parent_ref)
        elif isinstance(node.parent_ref, ast_internal_classes.Array_Subscript_Node):
            self.nodes.append(node.parent_ref.name)
        if isinstance(node.parent_ref, ast_internal_classes.Array_Subscript_Node):
            for i in node.parent_ref.indices:
                self.visit(i)
        if isinstance(node.part_ref, ast_internal_classes.Array_Subscript_Node):
            for i in node.part_ref.indices:
                self.visit(i)
        elif isinstance(node.part_ref, ast_internal_classes.Data_Ref_Node):
            self.visit_Blunt_Data_Ref_Node(node.part_ref)

    def visit_Blunt_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):
        if isinstance(node.parent_ref, ast_internal_classes.Array_Subscript_Node):
            for i in node.parent_ref.indices:
                self.visit(i)
        if isinstance(node.part_ref, ast_internal_classes.Array_Subscript_Node):
            for i in node.part_ref.indices:
                self.visit(i)
        elif isinstance(node.part_ref, ast_internal_classes.Data_Ref_Node):
            self.visit_Blunt_Data_Ref_Node(node.part_ref)

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                pass
            elif isinstance(node.lval, ast_internal_classes.Array_Subscript_Node):
                for i in node.lval.indices:
                    self.visit(i)
            elif isinstance(node.lval, ast_internal_classes.Data_Ref_Node):
                # if isinstance(node.lval.parent_ref, ast_internal_classes.Name_Node):
                #    self.nodes.append(node.lval.parent_ref)
                if isinstance(node.lval.parent_ref, ast_internal_classes.Array_Subscript_Node):
                    # self.nodes.append(node.lval.parent_ref.name)
                    for i in node.lval.parent_ref.indices:
                        self.visit(i)
                if isinstance(node.lval.part_ref, ast_internal_classes.Data_Ref_Node):
                    self.visit_Blunt_Data_Ref_Node(node.lval.part_ref)
                elif isinstance(node.lval.part_ref, ast_internal_classes.Array_Subscript_Node):
                    for i in node.lval.part_ref.indices:
                        self.visit(i)

        else:
            if isinstance(node.lval, ast_internal_classes.Data_Ref_Node):
                if isinstance(node.lval.parent_ref, ast_internal_classes.Name_Node):
                    self.nodes.append(node.lval.parent_ref)
                elif isinstance(node.lval.parent_ref, ast_internal_classes.Array_Subscript_Node):
                    self.nodes.append(node.lval.parent_ref.name)
                    for i in node.lval.parent_ref.indices:
                        self.visit(i)
                if isinstance(node.lval.part_ref, ast_internal_classes.Data_Ref_Node):
                    self.visit_Blunt_Data_Ref_Node(node.lval.part_ref)
                elif isinstance(node.lval.part_ref, ast_internal_classes.Array_Subscript_Node):
                    for i in node.lval.part_ref.indices:
                        self.visit(i)
            else:
                self.visit(node.lval)
        if isinstance(node.rval, ast_internal_classes.Data_Ref_Node):
            if isinstance(node.rval.parent_ref, ast_internal_classes.Name_Node):
                self.nodes.append(node.rval.parent_ref)
            elif isinstance(node.rval.parent_ref, ast_internal_classes.Array_Subscript_Node):
                self.nodes.append(node.rval.parent_ref.name)
                for i in node.rval.parent_ref.indices:
                    self.visit(i)
            if isinstance(node.rval.part_ref, ast_internal_classes.Data_Ref_Node):
                self.visit_Blunt_Data_Ref_Node(node.rval.part_ref)
            elif isinstance(node.rval.part_ref, ast_internal_classes.Array_Subscript_Node):
                for i in node.rval.part_ref.indices:
                    self.visit(i)
        else:
            self.visit(node.rval)


class FindOutputs(NodeVisitor):
    """
    Finds all outputs (writes) in the AST node and its children
    :return: List of names
    """

    def __init__(self, thourough=False):
        self.thourough = thourough
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        for i in node.args:
            if isinstance(i, ast_internal_classes.Name_Node):
                if self.thourough:
                    self.nodes.append(i)
            elif isinstance(i, ast_internal_classes.Array_Subscript_Node):
                if self.thourough:
                    self.nodes.append(i.name)
                for j in i.indices:
                    self.visit(j)
            elif isinstance(i, ast_internal_classes.Data_Ref_Node):
                if isinstance(i.parent_ref, ast_internal_classes.Name_Node):
                    if self.thourough:
                        self.nodes.append(i.parent_ref)
                elif isinstance(i.parent_ref, ast_internal_classes.Array_Subscript_Node):
                    if self.thourough:
                        self.nodes.append(i.parent_ref.name)
                    for j in i.parent_ref.indices:
                        self.visit(j)
                self.visit(i.part_ref)
            self.visit(i)

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                self.nodes.append(node.lval)
            elif isinstance(node.lval, ast_internal_classes.Array_Subscript_Node):
                self.nodes.append(node.lval.name)
            elif isinstance(node.lval, ast_internal_classes.Data_Ref_Node):
                if isinstance(node.lval.parent_ref, ast_internal_classes.Name_Node):
                    self.nodes.append(node.lval.parent_ref)
                elif isinstance(node.lval.parent_ref, ast_internal_classes.Array_Subscript_Node):
                    self.nodes.append(node.lval.parent_ref.name)
                    for i in node.lval.parent_ref.indices:
                        self.visit(i)

            self.visit(node.rval)


class StructLister(NodeVisitor):
    """
    Fortran does not differentiate between arrays and functions.
    We need to go over and convert all function calls to arrays.
    So, we create a closure of all math and defined functions and
    create array expressions for the others.
    """

    def __init__(self):

        self.structs = []
        self.names = []

    def visit_Derived_Type_Def_Node(self, node: ast_internal_classes.Derived_Type_Def_Node):
        self.names.append(node.name.name)
        if node.procedure_part is not None:
            if len(node.procedure_part.procedures) > 0:
                node.is_class = True
                self.structs.append(node)
                return
        node.is_class = False
        self.structs.append(node)


class StructDependencyLister(NodeVisitor):

    def __init__(self, names=None):
        self.names = names
        self.structs_used = []
        self.is_pointer = []
        self.pointer_names = []

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        if node.type in self.names:
            self.structs_used.append(node.type)
            self.is_pointer.append(node.alloc)
            self.pointer_names.append(node.name)


class StructMemberLister(NodeVisitor):

    def __init__(self):
        self.members = []
        self.is_pointer = []
        self.pointer_names = []

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        self.members.append(node.type)
        self.is_pointer.append(node.alloc)
        self.pointer_names.append(node.name)


class FindStructDefs(NodeVisitor):

    def __init__(self, name=None):
        self.name = name
        self.structs = []

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        if node.type == self.name:
            self.structs.append(node.name)


class FindStructUses(NodeVisitor):

    def __init__(self, names=None, target=None):
        self.names = names
        self.target = target
        self.nodes = []

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

        if isinstance(node.parent_ref, ast_internal_classes.Name_Node):
            parent_name = node.parent_ref.name
        elif isinstance(node.parent_ref, ast_internal_classes.Array_Subscript_Node):
            parent_name = node.parent_ref.name.name
        elif isinstance(node.parent_ref, ast_internal_classes.Data_Ref_Node):
            raise NotImplementedError("Data ref node not implemented for not name or array")
            self.visit(node.parent_ref)
            parent_name = None
        else:

            raise NotImplementedError("Data ref node not implemented for not name or array")
        if isinstance(node.part_ref, ast_internal_classes.Name_Node):
            part_name = node.part_ref.name
        elif isinstance(node.part_ref, ast_internal_classes.Array_Subscript_Node):
            part_name = node.part_ref.name.name
        elif isinstance(node.part_ref, ast_internal_classes.Data_Ref_Node):
            self.visit(node.part_ref)
            if isinstance(node.part_ref.parent_ref, ast_internal_classes.Name_Node):
                part_name = node.part_ref.parent_ref.name
            elif isinstance(node.part_ref.parent_ref, ast_internal_classes.Array_Subscript_Node):
                part_name = node.part_ref.parent_ref.name.name

        else:
            raise NotImplementedError("Data ref node not implemented for not name or array")
        if part_name == self.target and parent_name in self.names:
            self.nodes.append(node)


class StructPointerChecker(NodeVisitor):

    def __init__(self, parent_struct, pointed_struct, pointer_name, structs_lister, struct_dep_graph, analysis):
        self.parent_struct = [parent_struct]
        self.pointed_struct = [pointed_struct]
        self.pointer_name = [pointer_name]
        self.nodes = []
        self.connection = []
        self.structs_lister = structs_lister
        self.struct_dep_graph = struct_dep_graph
        if analysis == "full":
            start_idx = 0
            end_idx = 1
            while start_idx != end_idx:
                for i in struct_dep_graph.in_edges(self.parent_struct[start_idx]):
                    found = False
                    for parent, child in zip(self.parent_struct, self.pointed_struct):
                        if i[0] == parent and i[1] == child:
                            found = True
                            break
                    if not found:
                        self.parent_struct.append(i[0])
                        self.pointed_struct.append(i[1])
                        self.pointer_name.append(struct_dep_graph.get_edge_data(i[0], i[1])["point_name"])
                        end_idx += 1
                start_idx += 1

    def visit_Main_Program_Node(self, node: ast_internal_classes.Main_Program_Node):
        for parent, pointer in zip(self.parent_struct, self.pointer_name):
            finder = FindStructDefs(parent)
            finder.visit(node.specification_part)
            struct_names = finder.structs
            use_finder = FindStructUses(struct_names, pointer)
            use_finder.visit(node.execution_part)
            self.nodes += use_finder.nodes
            self.connection.append([parent, pointer, struct_names, use_finder.nodes])

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        for parent, pointer in zip(self.parent_struct, self.pointer_name):

            finder = FindStructDefs(parent)
            if node.specification_part is not None:
                finder.visit(node.specification_part)
            struct_names = finder.structs
            use_finder = FindStructUses(struct_names, pointer)
            if node.execution_part is not None:
                use_finder.visit(node.execution_part)
            self.nodes += use_finder.nodes
            self.connection.append([parent, pointer, struct_names, use_finder.nodes])


class StructPointerEliminator(NodeTransformer):

    def __init__(self, parent_struct, pointed_struct, pointer_name):
        self.parent_struct = parent_struct
        self.pointed_struct = pointed_struct
        self.pointer_name = pointer_name

    def visit_Derived_Type_Def_Node(self, node: ast_internal_classes.Derived_Type_Def_Node):
        if node.name.name == self.parent_struct:
            newnode = ast_internal_classes.Derived_Type_Def_Node(name=node.name, parent=node.parent)
            component_part = ast_internal_classes.Component_Part_Node(component_def_stmts=[], parent=node.parent)
            for i in node.component_part.component_def_stmts:

                vardecl = []
                for k in i.vars.vardecl:
                    if k.name == self.pointer_name and k.alloc == True and k.type == self.pointed_struct:
                        continue
                    else:
                        vardecl.append(k)
                if vardecl != []:
                    component_part.component_def_stmts.append(
                        ast_internal_classes.Data_Component_Def_Stmt_Node(vars=ast_internal_classes.Decl_Stmt_Node(
                            vardecl=vardecl, parent=node.parent),
                                                                          parent=node.parent))
            newnode.component_part = component_part
            return newnode
        else:
            return node


class CallToArray(NodeTransformer):
    """
    Fortran does not differentiate between arrays and functions.
    We need to go over and convert all function calls to arrays.
    So, we create a closure of all math and defined functions and
    create array expressions for the others.
    """

    def __init__(self, funcs: FindFunctionAndSubroutines, dict=None):
        self.funcs = funcs
        self.rename_dict = dict

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        self.excepted_funcs = [
            "malloc", "pow", "cbrt", "__dace_sign", "__dace_allocated", "tanh", "atan2", "__dace_epsilon",
            "__dace_exit", "surrtpk", "surrtab", "surrtrf", "abor1", *FortranIntrinsics.function_names()
        ]
        #

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if isinstance(node.name, str):
            return node
        assert node.name is not None, f"not a valid call expression, got: {node} / {type(node)}"
        name = node.name.name

        found_in_names = name in [i.name for i in self.funcs.names]
        found_in_renames = False
        if self.rename_dict is not None:
            for k, v in self.rename_dict.items():
                for original_name, replacement_names in v.items():
                    if isinstance(replacement_names, str):
                        if name == replacement_names:
                            found_in_renames = True
                            module = k
                            original_one = original_name
                            node.name.name = original_name
                            print(f"Found {name} in {module} with original name {original_one}")
                            break
                    elif isinstance(replacement_names, list):
                        for repl in replacement_names:
                            if name == repl:
                                found_in_renames = True
                                module = k
                                original_one = original_name
                                node.name.name = original_name
                                print(f"Found in list {name} in {module} with original name {original_one}")
                                break
                    else:
                        raise ValueError(f"Invalid type {type(replacement_names)} for {replacement_names}")

        # TODO Deconproc is a special case, we need to handle it differently - this is just s quick workaround
        if name.startswith(
                "__dace_"
        ) or name in self.excepted_funcs or found_in_renames or found_in_names or name in self.funcs.iblocks:
            processed_args = []
            for i in node.args:
                arg = CallToArray(self.funcs, self.rename_dict).visit(i)
                processed_args.append(arg)
            node.args = processed_args
            return node
        indices = [CallToArray(self.funcs, self.rename_dict).visit(i) for i in node.args]
        # Array subscript cannot be empty.
        assert indices
        return ast_internal_classes.Array_Subscript_Node(name=node.name,
                                                         type=node.type,
                                                         indices=indices,
                                                         line_number=node.line_number)


class ArgumentExtractor(NodeTransformer):
    """
    Uses the ArgumentExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """

    def __init__(self, program):
        self._count = 0
        ParentScopeAssigner().visit(program)
        # For a nesting of execution parts (rare, but in case it happens), after visiting each direct child of it,
        # `self.execution_preludes[-1]` will contain all the temporary variable assignments necessary for that node.
        self.execution_preludes: List[List[ast_internal_classes.BinOp_Node]] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        DIRECTLY_REFERNCEABLE = (ast_internal_classes.Name_Node, ast_internal_classes.Literal,
                                 ast_internal_classes.Array_Subscript_Node, ast_internal_classes.Data_Ref_Node)

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in [
                "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            return self.generic_visit(node)
        result = ast_internal_classes.Call_Expr_Node(name=node.name,
                                                     args=[],
                                                     line_number=node.line_number,
                                                     type=node.type,
                                                     subroutine=node.subroutine,
                                                     parent=node.parent)

        for i, arg in enumerate(node.args):
            # Ensure we allow to extract function calls from arguments
            if (isinstance(arg, DIRECTLY_REFERNCEABLE) or (isinstance(arg, ast_internal_classes.Actual_Arg_Spec_Node)
                                                           and isinstance(arg.arg, DIRECTLY_REFERNCEABLE))):
                # If it is a node type that's allowed to be directly referenced in a (possibly keyworded) function
                # argument, then we keep the node as is.
                result.args.append(arg)
                continue

            # These needs to be extracted, so register a temporary variable.
            tmpname = TempName.get_name('tmp_arg')
            decl = ast_internal_classes.Decl_Stmt_Node(
                vardecl=[ast_internal_classes.Var_Decl_Node(name=tmpname, type='VOID', sizes=None, init=None)])
            node.parent.specification_part.specifications.append(decl)

            if isinstance(arg, ast_internal_classes.Actual_Arg_Spec_Node):
                self.generic_visit(arg.arg)
                result.args.append(
                    ast_internal_classes.Actual_Arg_Spec_Node(arg_name=arg.arg_name,
                                                              arg=ast_internal_classes.Name_Node(name=tmpname,
                                                                                                 type=arg.arg.type)))
                asgn = ast_internal_classes.BinOp_Node(op="=",
                                                       lval=ast_internal_classes.Name_Node(name=tmpname,
                                                                                           type=arg.arg.type),
                                                       rval=arg.arg,
                                                       line_number=node.line_number,
                                                       parent=node.parent)
            else:
                self.generic_visit(arg)
                result.args.append(ast_internal_classes.Name_Node(name=tmpname, type=arg.type))
                asgn = ast_internal_classes.BinOp_Node(op="=",
                                                       lval=ast_internal_classes.Name_Node(name=tmpname, type=arg.type),
                                                       rval=arg,
                                                       line_number=node.line_number,
                                                       parent=node.parent)

            self.execution_preludes[-1].append(asgn)
        return result

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        self.execution_preludes.append([])
        for ex in node.execution:
            ex = self.visit(ex)
            newbody.extend(reversed(self.execution_preludes[-1]))
            newbody.append(ex)
            self.execution_preludes[-1].clear()
        self.execution_preludes.pop()
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class FunctionCallTransformer(NodeTransformer):

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
            if hasattr(node.rval, "subroutine"):
                if node.rval.subroutine is True:
                    return self.generic_visit(node)
            if node.rval.name.name.find("__dace_") != -1:
                return self.generic_visit(node)
            if node.rval.name.name == "pow":
                return self.generic_visit(node)
            if node.op != "=":
                return self.generic_visit(node)
            args = node.rval.args
            lval = node.lval
            args.append(lval)
            return (ast_internal_classes.Call_Expr_Node(type=node.rval.type,
                                                        name=ast_internal_classes.Name_Node(name=node.rval.name.name +
                                                                                            "_srt",
                                                                                            type=node.rval.type),
                                                        args=args,
                                                        subroutine=True,
                                                        line_number=node.line_number,
                                                        parent=node.parent))

        else:
            return self.generic_visit(node)


class NameReplacer(NodeTransformer):
    """
    Replaces all occurences of a name with another name
    """

    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        if node.name == self.old_name:
            return ast_internal_classes.Name_Node(name=self.new_name, type=node.type)
        else:
            return self.generic_visit(node)


class ArrayDimensionSymbolsMapper(NodeTransformer):

    def __init__(self):
        # The dictionary that maps a symbol for array dimension information to a tuple of type and component.
        # ASSUMPTION: The type name must be globally unique.
        self.array_dims_symbols: Dict[str, Tuple[str, str]] = {}
        self.cur_type = None

    def visit_Derived_Type_Def_Node(self, node: ast_internal_classes.Derived_Type_Def_Node):
        self.cur_type = node
        out = self.generic_visit(node)
        self.cur_type = None
        return out

    def visit_Data_Component_Def_Stmt_Node(self, node: ast_internal_classes.Data_Component_Def_Stmt_Node):
        assert self.cur_type
        for v in node.vars.vardecl:
            if not isinstance(v, ast_internal_classes.Symbol_Decl_Node):
                continue
            assert v.name not in self.array_dims_symbols
            self.array_dims_symbols[v.name] = (self.cur_type.name.name, v.name)
        return self.generic_visit(node)


CONFIG_INJECTOR_SIZE_PATTERN = re.compile(r"__f2dace_SA_(?P<comp>[a-zA-Z0-9_]+)_d_(?P<num>[0-9]*)")
CONFIG_INJECTOR_OFFSET_PATTERN = re.compile(r"__f2dace_SOA_(?P<comp>[a-zA-Z0-9_]+)_d_(?P<num>[0-9]*)")


class ArrayDimensionConfigInjector(NodeTransformer):

    def __init__(self, array_dims_info: ArrayDimensionSymbolsMapper, cfg: List[ConstTypeInjection]):
        self.cfg: Dict[str, str] = {}  # Maps the array dimension symbols to their fixed values.
        self.in_exec_depth = 0  # Whether the visitor is in code (i.e., not declarations) and at what depth.

        for c in cfg:
            assert c.scope_spec is None  # Cannot support otherwise.
            typ = c.type_spec[-1]  # We assume globally unique typenames for these configuration objects.
            comp = c.component_spec[-1]
            if not comp.endswith('_s'):
                continue
            comp = comp.removesuffix('_s')
            size_match = CONFIG_INJECTOR_SIZE_PATTERN.match(comp)
            offset_match = CONFIG_INJECTOR_OFFSET_PATTERN.match(comp)
            if size_match:
                marker = 'SA'
                comp, num = size_match.groups()
            elif offset_match:
                marker = 'SOA'
                comp, num = offset_match.groups()
            else:
                continue
            for k, v in array_dims_info.array_dims_symbols.items():
                if v[0] == typ and v[1].startswith(f"__f2dace_{marker}_{comp}_d_{num}_s_"):
                    assert k not in self.cfg
                    self.cfg[k] = c.value

    def visit_Execution_Part_Node(self, expart: Execution_Part_Node):
        self.in_exec_depth += 1
        out = self.generic_visit(expart)
        self.in_exec_depth -= 1
        return out

    def visit_Data_Ref_Node(self, dref: Data_Ref_Node):
        if isinstance(dref.part_ref, (Array_Subscript_Node, Data_Ref_Node)):
            return self.generic_visit(dref)
        assert isinstance(dref.part_ref, Name_Node)
        if self.in_exec_depth > 0 and dref.part_ref.name in self.cfg:
            val = self.cfg[dref.part_ref.name]
            if val in {'true', 'false'}:
                return Bool_Literal_Node(val)
            else:
                return Int_Literal_Node(val)
        return dref

    def visit_Name_Node(self, name: Name_Node):
        if self.in_exec_depth > 0 and name.name in self.cfg:
            return Int_Literal_Node(self.cfg[name.name])
        return name

    def visit_Var_Decl_Node(self, var: Var_Decl_Node):
        if var.sizes is None:
            return var

        def _name_of(_z) -> Optional[str]:
            if isinstance(_z, str):
                return _z
            elif isinstance(_z, Name_Node):
                return _z.name
            return None

        def _maybe_intval_of(_z, to_type: Type):
            z_name = _name_of(_z)
            return to_type(self.cfg[z_name]) if z_name in self.cfg else _z

        var.sizes = [_maybe_intval_of(z, Int_Literal_Node) for z in var.sizes]
        var.offsets = [_maybe_intval_of(z, int) for z in var.offsets]
        return var


class FunctionToSubroutineDefiner(NodeTransformer):
    """
    Transforms all function definitions into subroutine definitions
    """

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):
        assert node.ret
        ret = node.ret
        ret_name, subr_name = f"{node.name.name}__ret", f"{node.name.name}_srt"

        found = False
        if node.specification_part:
            for j in node.specification_part.specifications:
                for k in j.vardecl:
                    if node.ret is not None:
                        if k.name == ret.name:
                            j.vardecl[j.vardecl.index(k)].name = ret_name
                            found = True
                            break
                    if k.name == node.name.name:
                        j.vardecl[j.vardecl.index(k)].name = ret_name
                        found = True
                        break

        if not found:
            var = ast_internal_classes.Var_Decl_Node(name=ret_name, type='VOID')
            stmt_node = ast_internal_classes.Decl_Stmt_Node(vardecl=[var], line_number=node.line_number)

            if node.specification_part is not None:
                node.specification_part.specifications.append(stmt_node)
            else:
                node.specification_part = ast_internal_classes.Specification_Part_Node(specifications=[stmt_node],
                                                                                       symbols=None,
                                                                                       interface_blocks=None,
                                                                                       uses=None,
                                                                                       typedecls=None,
                                                                                       enums=None)

        # We should always be able to tell a functions return _variable_ (i.e., not type, which we also should be able
        # to tell).
        assert node.ret
        execution_part = NameReplacer(ret.name, ret_name).visit(node.execution_part)
        args = node.args
        args.append(ast_internal_classes.Name_Node(name=ret_name, type=node.type))
        return ast_internal_classes.Subroutine_Subprogram_Node(name=ast_internal_classes.Name_Node(name=subr_name,
                                                                                                   type=node.type),
                                                               args=args,
                                                               specification_part=node.specification_part,
                                                               execution_part=execution_part,
                                                               internal_subprogram_part=node.internal_subprogram_part,
                                                               subroutine=True,
                                                               line_number=node.line_number,
                                                               elemental=node.elemental)


class CallExtractorNodeLister(NodeVisitor):
    """
    Finds all function calls in the AST node and its children that have to be extracted into independent expressions
    """

    def __init__(self, root=None):
        self.root = root
        self.nodes: List[ast_internal_classes.Call_Expr_Node] = []

    def visit_For_Stmt_Node(self, node: ast_internal_classes.For_Stmt_Node):
        self.generic_visit(node.init)
        self.generic_visit(node.cond)
        return

    def visit_If_Stmt_Node(self, node: ast_internal_classes.If_Stmt_Node):
        self.generic_visit(node.cond)
        return

    def visit_While_Stmt_Node(self, node: ast_internal_classes.While_Stmt_Node):
        self.generic_visit(node.cond)
        return

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        stop = False
        if self.root == node:
            return self.generic_visit(node)
        if isinstance(self.root, ast_internal_classes.BinOp_Node):
            if node == self.root.rval and isinstance(self.root.lval, ast_internal_classes.Name_Node):
                return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                stop = True

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if not stop and node.name.name not in [
                "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            self.nodes.append(node)
        # return self.generic_visit(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class CallExtractor(NodeTransformer):
    """
    Uses the CallExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """

    def __init__(self, ast, count=0):
        self.count = count

        self.functions_by_name: Dict[str, ast_internal_classes.Subroutine_Subprogram_Node] = {
            f.name.name: f
            for f in mywalk(ast, (ast_internal_classes.Subroutine_Subprogram_Node,
                                  ast_internal_classes.Function_Subprogram_Node))
        }

        ParentScopeAssigner().visit(ast)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in [
                "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                return self.generic_visit(node)
        if not hasattr(self, "count"):
            self.count = 0
        else:
            self.count = self.count + 1
        tmp = self.count

        # for i, arg in enumerate(node.args):
        #    # Ensure we allow to extract function calls from arguments
        #    node.args[i] = self.visit(arg)

        return ast_internal_classes.Name_Node(name="tmp_call_" + str(tmp - 1))

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):

        oldbody = node.execution
        changes_made = True
        while changes_made:
            changes_made = False
            newbody = []
            for child in oldbody:
                lister = CallExtractorNodeLister(child)
                lister.visit(child)
                res = lister.nodes

                if len(res) > 0:
                    changes_made = True
                    # Variables are counted from 0...end, starting from main node, to all calls nested
                    # in main node arguments.
                    # However, we need to define nested ones first.
                    # We go in reverse order, counting from end-1 to 0.
                    temp = self.count + len(res) - 1
                    for i in reversed(range(0, len(res))):
                        if res[i].type == 'VOID' and res[i].name.name in self.functions_by_name:
                            fn = self.functions_by_name[res[i].name.name]
                            res[i].type = fn.type
                        node.parent.specification_part.specifications.append(
                            ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                ast_internal_classes.Var_Decl_Node(
                                    name="tmp_call_" + str(temp), type=res[i].type, sizes=None, init=None)
                            ]))
                        newbody.append(
                            ast_internal_classes.BinOp_Node(op="=",
                                                            lval=ast_internal_classes.Name_Node(name="tmp_call_" +
                                                                                                str(temp),
                                                                                                type=res[i].type),
                                                            rval=res[i],
                                                            line_number=child.line_number,
                                                            parent=child.parent))
                        temp = temp - 1
                if isinstance(child, ast_internal_classes.Call_Expr_Node):
                    new_args = []
                    for i in child.args:
                        new_args.append(self.visit(i))
                    new_child = ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                    subroutine=child.subroutine,
                                                                    name=child.name,
                                                                    args=new_args,
                                                                    line_number=child.line_number,
                                                                    parent=child.parent)
                    newbody.append(new_child)
                elif isinstance(child, ast_internal_classes.BinOp_Node):
                    if isinstance(child.lval, ast_internal_classes.Name_Node) and isinstance(
                            child.rval, ast_internal_classes.Call_Expr_Node):
                        new_args = []
                        for i in child.rval.args:
                            new_args.append(self.visit(i))
                        new_child = ast_internal_classes.Call_Expr_Node(type=child.rval.type,
                                                                        subroutine=child.rval.subroutine,
                                                                        name=child.rval.name,
                                                                        args=new_args,
                                                                        line_number=child.rval.line_number,
                                                                        parent=child.rval.parent)
                        newbody.append(
                            ast_internal_classes.BinOp_Node(op=child.op,
                                                            lval=child.lval,
                                                            rval=new_child,
                                                            line_number=child.line_number,
                                                            parent=child.parent))
                    else:
                        newbody.append(self.visit(child))
                else:
                    newbody.append(self.visit(child))
            oldbody = newbody

        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ParentScopeAssigner(NodeVisitor):
    """
        For each node, it assigns its parent scope - program, subroutine, function.

        If the parent node is one of the "parent" types, we assign it as the parent.
        Otherwise, we look for the parent of my parent to cover nested AST nodes within
        a single scope.
    """

    def __init__(self):
        pass

    def visit(self, node: ast_internal_classes.FNode, parent_node: Optional[ast_internal_classes.FNode] = None):

        parent_node_types = [
            ast_internal_classes.Subroutine_Subprogram_Node, ast_internal_classes.Function_Subprogram_Node,
            ast_internal_classes.Main_Program_Node, ast_internal_classes.Module_Node
        ]

        if parent_node is not None and type(parent_node) in parent_node_types:
            node.parent = parent_node
        elif parent_node is not None:
            node.parent = parent_node.parent

        # Copied from `generic_visit` to recursively parse all leafs
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item, node)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value, node)

        return node


class ModuleVarsDeclarations(NodeVisitor):
    """
        Creates a mapping (scope name, variable name) -> variable declaration.

        The visitor is used to access information on variable dimension, sizes, and offsets.
    """

    def __init__(self):  # , module_name: str):

        self.scope_vars: Dict[Tuple[str, str], ast_internal_classes.FNode] = {}

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        var_name = node.name
        self.scope_vars[var_name] = node

    def visit_Symbol_Decl_Node(self, node: ast_internal_classes.Symbol_Decl_Node):
        var_name = node.name
        self.scope_vars[var_name] = node


class ScopeVarsDeclarations(NodeVisitor):
    """
        Creates a mapping (scope name, variable name) -> variable declaration.

        The visitor is used to access information on variable dimension, sizes, and offsets.
    """

    def __init__(self, ast):

        self.scope_vars: Dict[Tuple[str, str], ast_internal_classes.FNode] = {}
        if hasattr(ast, "module_declarations"):
            self.module_declarations = ast.module_declarations
        else:
            self.module_declarations = {}

    def get_var(self, scope: Optional[Union[ast_internal_classes.FNode, str]],
                variable_name: str) -> ast_internal_classes.FNode:

        if scope is not None and self.contains_var(scope, variable_name):
            return self.scope_vars[(self._scope_name(scope), variable_name)]
        elif variable_name in self.module_declarations:
            return self.module_declarations[variable_name]
        else:
            raise RuntimeError(
                f"Couldn't find the declaration of variable {variable_name} in function {self._scope_name(scope)}!")

    def contains_var(self, scope: ast_internal_classes.FNode, variable_name: str) -> bool:
        return (self._scope_name(scope), variable_name) in self.scope_vars

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):

        parent_name = self._scope_name(node.parent)
        var_name = node.name
        self.scope_vars[(parent_name, var_name)] = node

    def visit_Symbol_Decl_Node(self, node: ast_internal_classes.Symbol_Decl_Node):

        parent_name = self._scope_name(node.parent)
        var_name = node.name
        self.scope_vars[(parent_name, var_name)] = node

    def _scope_name(self, scope: ast_internal_classes.FNode) -> str:
        if isinstance(scope, ast_internal_classes.Main_Program_Node):
            return scope.name.name.name
        elif isinstance(scope, str):
            return scope
        else:
            return scope.name.name


class IndexExtractorNodeLister(NodeVisitor):
    """
    Finds all array subscript expressions in the AST node and its children that have to be extracted into independent expressions
    """

    def __init__(self):
        self.nodes: List[ast_internal_classes.Array_Subscript_Node] = []
        self.current_parent: Optional[ast_internal_classes.Data_Ref_Node] = None

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["pow", "atan2", "tanh", *FortranIntrinsics.retained_function_names()]:
            return self.generic_visit(node)
        else:
            for arg in node.args:
                self.visit(arg)
            return

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        old_current_parent = self.current_parent
        self.current_parent = None
        for i in node.indices:
            self.visit(i)
        self.current_parent = old_current_parent

        self.nodes.append((node, self.current_parent))

        # disable structure parent node for array indices

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

        set_node = False
        if self.current_parent is None:
            self.current_parent = node
            set_node = True
        self.visit(node.parent_ref)
        self.visit(node.part_ref)

        if set_node:
            set_node = False
            self.current_parent = None

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class IndexExtractor(NodeTransformer):
    """
    Uses the IndexExtractorNodeLister to find all array subscript expressions
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the index expression with the variable.

    Before parsing the AST, the transformation first runs:
    - ParentScopeAssigner to ensure that each node knows its scope assigner.
    - ScopeVarsDeclarations to aggregate all variable declarations for each function.
    """

    def __init__(self, ast: ast_internal_classes.FNode, normalize_offsets: bool = False, count=0):

        self.count = count
        self.normalize_offsets = normalize_offsets
        self.program = ast
        self.replacements = {}

        if normalize_offsets:
            ParentScopeAssigner().visit(ast)
            self.scope_vars = ScopeVarsDeclarations(ast)
            self.scope_vars.visit(ast)
            self.structures = ast.structures

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["pow", "atan2", "tanh", *FortranIntrinsics.retained_function_names()]:
            return self.generic_visit(node)
        else:

            new_args = []
            for arg in node.args:
                new_args.append(self.visit(arg))
            node.args = new_args
            return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        new_indices = []

        for i in node.indices:
            new_indices.append(self.visit(i))

        tmp = self.count
        newer_indices = []
        for i in new_indices:
            if isinstance(i, ast_internal_classes.ParDecl_Node):
                newer_indices.append(i)
            else:
                if isinstance(i, ast_internal_classes.Name_Node):
                    if i.name.startswith("tmp_index_"):
                        newer_indices.append(i)
                        continue
                newer_indices.append(ast_internal_classes.Name_Node(name="tmp_index_" + str(tmp)))
                self.replacements["tmp_index_" + str(tmp)] = (i, node.name.name)
                tmp = tmp + 1
        self.count = tmp

        return ast_internal_classes.Array_Subscript_Node(name=node.name,
                                                         type=node.type,
                                                         indices=newer_indices,
                                                         line_number=node.line_number)

    def visit_Specification_Part_Node(self, node: ast_internal_classes.Specification_Part_Node):
        newspec = []
        for child in node.specifications:
            lister = IndexExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            temp = self.count
            tmp_child = self.visit(child)
            if res is not None:
                for j, parent_node in res:
                    for idx, i in enumerate(j.indices):

                        if isinstance(i, ast_internal_classes.ParDecl_Node):
                            continue
                        elif isinstance(i, ast_internal_classes.Name_Node):
                            if i.name.startswith("tmp_index_"):
                                continue

                        tmp_name = "tmp_index_" + str(temp)
                        temp = temp + 1
                        if self.normalize_offsets:
                            var_name = ""
                            if isinstance(j, ast_internal_classes.Name_Node):
                                var_name = j.name
                                variable = self.scope_vars.get_var(child.parent, var_name)
                            elif parent_node is not None:
                                struct, variable, _ = self.structures.find_definition(
                                    self.scope_vars, parent_node, j.name)
                                var_name = j.name.name
                            else:
                                var_name = j.name.name
                                variable = self.scope_vars.get_var(child.parent, var_name)

                            offset = variable.offsets[idx]
                            if not isinstance(offset, ast_internal_classes.FNode):
                                # check if offset is a number
                                try:
                                    offset = int(offset)
                                except:
                                    raise ValueError(f"Offset {offset} is not a number")
                                offset = ast_internal_classes.Int_Literal_Node(value=str(offset))
                            newspec.append(
                                ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                    ast_internal_classes.Var_Decl_Node(name=tmp_name,
                                                                       type="INTEGER",
                                                                       sizes=None,
                                                                       init=ast_internal_classes.BinOp_Node(
                                                                           op="-",
                                                                           lval=self.replacements[tmp_name][0],
                                                                           rval=offset,
                                                                           line_number=child.line_number,
                                                                           parent=child.parent),
                                                                       line_number=child.line_number)
                                ],
                                                                    line_number=child.line_number))

                        else:
                            newspec.append(
                                ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                    ast_internal_classes.Var_Decl_Node(
                                        name=tmp_name,
                                        type="INTEGER",
                                        sizes=None,
                                        init=ast_internal_classes.BinOp_Node(
                                            op="-",
                                            lval=self.replacements[tmp_name][0],
                                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                            line_number=child.line_number,
                                            parent=child.parent),
                                        line_number=child.line_number)
                                ],
                                                                    line_number=child.line_number))
                        self.replacements.pop(tmp_name)

            newspec.append(tmp_child)

        return ast_internal_classes.Specification_Part_Node(specifications=newspec,
                                                            typedecls=node.typedecls,
                                                            symbols=node.symbols,
                                                            uses=node.uses,
                                                            enums=node.enums,
                                                            line_number=node.line_number)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = IndexExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            temp = self.count

            tmp_child = self.visit(child)
            if res is not None:
                for j, parent_node in res:
                    for idx, i in enumerate(j.indices):

                        if isinstance(i, ast_internal_classes.ParDecl_Node):
                            continue
                        elif isinstance(i, ast_internal_classes.Name_Node):
                            if i.name.startswith("tmp_index_"):
                                continue

                        tmp_name = "tmp_index_" + str(temp)
                        temp = temp + 1
                        newbody.append(
                            ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                ast_internal_classes.Var_Decl_Node(name=tmp_name,
                                                                   type="INTEGER",
                                                                   sizes=None,
                                                                   init=None,
                                                                   line_number=child.line_number)
                            ],
                                                                line_number=child.line_number))
                        if self.normalize_offsets:

                            # Find the offset of a variable to which we are assigning
                            var_name = ""
                            if isinstance(j, ast_internal_classes.Name_Node):
                                var_name = j.name
                                variable = self.scope_vars.get_var(child.parent, var_name)
                            elif parent_node is not None:
                                struct, variable, _ = self.structures.find_definition(
                                    self.scope_vars, parent_node, j.name)
                                var_name = j.name.name
                            else:
                                var_name = j.name.name
                                variable = self.scope_vars.get_var(child.parent, var_name)

                            offset = variable.offsets[idx]

                            # it can be a symbol, an operator, or a value

                            if not isinstance(offset, ast_internal_classes.FNode):
                                # check if offset is a number
                                try:
                                    offset = int(offset)
                                except:
                                    raise ValueError(f"Offset {offset} is not a number")
                                offset = ast_internal_classes.Int_Literal_Node(value=str(offset))
                            newbody.append(
                                ast_internal_classes.BinOp_Node(op="=",
                                                                lval=ast_internal_classes.Name_Node(name=tmp_name),
                                                                rval=ast_internal_classes.BinOp_Node(
                                                                    op="-",
                                                                    lval=self.replacements[tmp_name][0],
                                                                    rval=offset,
                                                                    line_number=child.line_number,
                                                                    parent=child.parent),
                                                                line_number=child.line_number))
                        else:
                            newbody.append(
                                ast_internal_classes.BinOp_Node(
                                    op="=",
                                    lval=ast_internal_classes.Name_Node(name=tmp_name),
                                    rval=ast_internal_classes.BinOp_Node(
                                        op="-",
                                        lval=self.replacements[tmp_name][0],
                                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                        line_number=child.line_number,
                                        parent=child.parent),
                                    line_number=child.line_number,
                                    parent=child.parent))
                        self.replacements.pop(tmp_name)

            newbody.append(tmp_child)
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class SignToIf(NodeTransformer):

    def __init__(self, ast):
        self.ast = ast

    """
    Transforms all sign expressions into if statements
    """

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node) and node.rval.name.name == "__dace_sign":
            args = node.rval.args
            lval = node.lval
            cond = ast_internal_classes.BinOp_Node(op=">=",
                                                   rval=ast_internal_classes.Real_Literal_Node(value="0.0"),
                                                   lval=args[1],
                                                   line_number=node.line_number,
                                                   parent=node.parent)

            abs_name = self.ast.intrinsic_handler.replace_function_name(ast_internal_classes.Name_Node(name="ABS"))

            body_if = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.Call_Expr_Node(name=abs_name,
                                                                                         type="DOUBLE",
                                                                                         args=[copy.deepcopy(args[0])],
                                                                                         line_number=node.line_number,
                                                                                         parent=node.parent,
                                                                                         subroutine=False),
                                                line_number=node.line_number,
                                                parent=node.parent)
            ])
            body_else = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.UnOp_Node(
                                                    op="-",
                                                    type="VOID",
                                                    lval=ast_internal_classes.Call_Expr_Node(
                                                        name=abs_name,
                                                        args=[copy.deepcopy(args[0])],
                                                        type="DOUBLE",
                                                        subroutine=False,
                                                        line_number=node.line_number,
                                                        parent=node.parent),
                                                    line_number=node.line_number,
                                                    parent=node.parent),
                                                line_number=node.line_number,
                                                parent=node.parent)
            ])
            return (ast_internal_classes.If_Stmt_Node(cond=cond,
                                                      body=body_if,
                                                      body_else=body_else,
                                                      line_number=node.line_number,
                                                      parent=node.parent))

        else:
            return self.generic_visit(node)


def optionalArgsHandleFunction(fn: ast_internal_classes.Subroutine_Subprogram_Node):
    # TODO: Determine the arguments' optionality during construction and keep updated.
    fn.optional_args = []
    if fn.specification_part is None:
        return 0
    for spec in fn.specification_part.specifications:
        for var in spec.vardecl:
            if var.optional:
                fn.optional_args.append((var.name, var.type))

    vardecls, new_args = [], []
    args_names = {a.name for a in fn.args}
    optional_args_names = {a[0] for a in fn.optional_args}
    for arg in fn.args:
        if arg.name not in optional_args_names:
            continue
        name = f"__f2dace_OPTIONAL_{arg.name}"
        if name in args_names:
            continue
        var = ast_internal_classes.Var_Decl_Node(name=name,
                                                 type='LOGICAL',
                                                 alloc=False,
                                                 sizes=None,
                                                 offsets=None,
                                                 kind=None,
                                                 optional=False,
                                                 init=None,
                                                 line_number=fn.line_number,
                                                 parent=fn)
        new_args.append(ast_internal_classes.Name_Node(name=name, line_number=fn.line_number, parent=fn))
        vardecls.append(var)

    if vardecls:
        fn.args.extend(new_args)
        fn.specification_part.specifications.append(
            ast_internal_classes.Decl_Stmt_Node(vardecl=vardecls, line_number=fn.line_number, parent=fn))

    return len(new_args)


class OptionalArgsTransformer(NodeTransformer):

    def __init__(self, funcs_with_opt_args):
        self.funcs_with_opt_args = funcs_with_opt_args

    def visit_Call_Expr_Node(self, call: ast_internal_classes.Call_Expr_Node):
        fn_def = self.funcs_with_opt_args.get(call.name.name)
        if not fn_def:
            return self.generic_visit(call)
        assert fn_def.optional_args

        # Basic assumption for positioanl arguments
        # Optional arguments follow the mandatory ones
        # We use that to determine which optional arguments are missing
        optional_args = len(fn_def.optional_args)
        should_be_args = len(fn_def.args)
        mandatory_args = should_be_args - optional_args * 2
        present_args = len(call.args)

        # Remove the deduplicated variable entries acting as flags for optional args
        missing_args_count = should_be_args - present_args
        if missing_args_count == 0:
            return self.generic_visit(call)

        new_args: List[Optional[ast_internal_classes.FNode]] = [None] * should_be_args
        # The mandatory arguments should be left as is.
        for i in range(mandatory_args):
            new_args[i] = call.args[i]

        optional_args_names = [a[0] for a in fn_def.optional_args]
        for i in range(mandatory_args, len(call.args)):
            current_arg = call.args[i]
            if isinstance(current_arg, ast_internal_classes.Actual_Arg_Spec_Node):
                # TODO: We are moving keyworded arguments into their new positions as positional arguments. But what
                #  about the arguments that are not present in the call statement at all?
                name = current_arg.arg_name
                new_pos = mandatory_args + optional_args_names.index(name.name)
                new_args[new_pos] = current_arg.arg
            else:
                # Positional arguments are left as is.
                new_args[i] = current_arg

        for i in range(mandatory_args, mandatory_args + optional_args):
            relative_position = i - mandatory_args
            if new_args[i] is None:
                dtype = fn_def.optional_args[relative_position][1]
                if dtype == 'INTEGER':
                    new_args[i] = ast_internal_classes.Int_Literal_Node(value='0')
                elif dtype == 'LOGICAL':
                    new_args[i] = ast_internal_classes.Bool_Literal_Node(value='0')
                elif dtype == 'DOUBLE':
                    new_args[i] = ast_internal_classes.Real_Literal_Node(value='0')
                elif dtype == 'CHAR':
                    new_args[i] = ast_internal_classes.Char_Literal_Node(value='0')
                else:
                    raise NotImplementedError()
                new_args[i + optional_args] = ast_internal_classes.Bool_Literal_Node(value='0')
            else:
                new_args[i + optional_args] = ast_internal_classes.Bool_Literal_Node(value='1')

        call.args = new_args
        return self.generic_visit(call)


def optionalArgsExpander(program: ast_internal_classes.Program_Node):
    """
    Adds to each optional arg a logical value specifying its status.
    Eliminates function statements from the AST
    :param program: The AST to be transformed
    :return: The transformed AST
    :note Should only be used on the program node
    """
    modified_functions = {}
    for fn in mywalk(program, ast_internal_classes.Subroutine_Subprogram_Node):
        if optionalArgsHandleFunction(fn):
            modified_functions[fn.name.name] = fn

    return OptionalArgsTransformer(modified_functions).visit(program)


class AllocatableReplacerTransformer(NodeTransformer):

    def __init__(self, program: ast_internal_classes.Program_Node):
        ParentScopeAssigner().visit(program)

        # TODO: This assumes globally unique function names. We can make it more general by keeping track of the
        #  module etc. too.
        self.functions_by_name: Dict[str, ast_internal_classes.Subroutine_Subprogram_Node] = {
            f.name.name: f
            for f in mywalk(program, ast_internal_classes.Subroutine_Subprogram_Node)
        }

        # For a nesting of execution parts (rare, but in case it happens), after visiting each direct child of it,
        # `self.execution_preludes[-1]` will contain all the temporary variable assignments necessary for that node.
        self.execution_preludes: List[List[ast_internal_classes.BinOp_Node]] = []

    @staticmethod
    def _allocated_flag(var: ast_internal_classes.Var_Decl_Node) -> str:
        assert isinstance(var, ast_internal_classes.Var_Decl_Node) and var.alloc
        return f"__f2dace_ALLOCATED_{var.name}"

    def visit_Call_Expr_Node(self, call: ast_internal_classes.Call_Expr_Node):
        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if call.name.name in FortranIntrinsics.retained_function_names():
            return self.generic_visit(call)
        assert call.name.name in self.functions_by_name
        fn = self.functions_by_name[call.name.name]
        callee_to_caller_argmap = match_callsite_args_to_function_args(fn, call)

        # Since we are here, this call expression does not have the extra arguments yet. So, for each callee argument in
        # order, we will add the extra argument in order. Note that we may not have updated the callee definition yet,
        # but this will be resolved when we visit that definition.
        for fa in fn.args:
            # The declaration must exist somewhere in the function specification parts.
            fadecl: ast_internal_classes.Var_Decl_Node = atmost_one(
                v for v in mywalk(fn.specification_part, ast_internal_classes.Var_Decl_Node) if v.name == fa.name)
            if not fadecl or not fadecl.alloc:
                continue
            ca = callee_to_caller_argmap[fa.name]
            # TODO: We assume that `ca` is a variable declared inside the call-site (i.e. not somewhere above). We
            #  should do proper scope search instead.
            if isinstance(ca, (ast_internal_classes.Name_Node, ast_internal_classes.Array_Subscript_Node)):
                cadecl: ast_internal_classes.Var_Decl_Node = singular(
                    v for v in mywalk(ca.parent.specification_part, ast_internal_classes.Var_Decl_Node)
                    if v.name == ca.name)
            elif isinstance(ca, ast_internal_classes.Data_Ref_Node):
                raise NotImplementedError
            # There must be an allocated flag for it too. We may not have defined it yet, but we will by the time we are
            # done with visiting the callee definition.
            alloc_flag = ast_internal_classes.Name_Node(name=self._allocated_flag(cadecl),
                                                        type='LOGICAL',
                                                        line_number=call.line_number,
                                                        parent=call.parent)
            call.args.append(alloc_flag)
        return self.generic_visit(call)

    def visit_Subroutine_Subprogram_Node(self, fn: ast_internal_classes.Subroutine_Subprogram_Node):
        # Since we are here, this function defintion does not have the extra variables and arguments yet. So, first we
        # declare the extra variables for all allocated variables.
        if fn.specification_part:
            for fvdecl in mywalk(fn.specification_part, ast_internal_classes.Var_Decl_Node):
                assert isinstance(fvdecl, ast_internal_classes.Var_Decl_Node)
                if not fvdecl.alloc:
                    continue
                alloc_flag = self._allocated_flag(fvdecl)
                decl = ast_internal_classes.Var_Decl_Node(name=alloc_flag,
                                                          type='LOGICAL',
                                                          init=ast_internal_classes.Bool_Literal_Node('0'),
                                                          line_number=fn.line_number,
                                                          parent=fn.parent)
                fn.specification_part.specifications.append(ast_internal_classes.Decl_Stmt_Node(vardecl=[decl]))
        # Then, for each argument in order, we will add the extra argument in order. Note that we may not have updated
        # any of the call-sites yet, but this will be resolved when we visit those call expressions.
        for fa in fn.args:
            # The declaration must exist somewhere in the function specification parts.
            fadecl: ast_internal_classes.Var_Decl_Node = atmost_one(
                v for v in mywalk(fn.specification_part, ast_internal_classes.Var_Decl_Node) if v.name == fa.name)
            if not fadecl or not fadecl.alloc:
                continue
            # We need to make it an argument.
            alloc_flag = self._allocated_flag(fadecl)
            fn.args.append(
                ast_internal_classes.Name_Node(name=alloc_flag,
                                               type='LOGICAL',
                                               line_number=fn.line_number,
                                               parent=fn.parent))
        return self.generic_visit(fn)

    def visit_Allocate_Stmt_Node(self, alloc: ast_internal_classes.Allocate_Stmt_Node):
        for av in alloc.allocation_list:
            # The declaration must exist somewhere in the function specification parts.
            avdecl: ast_internal_classes.Var_Decl_Node = singular(
                v for v in mywalk(av.parent.specification_part, ast_internal_classes.Var_Decl_Node)
                if v.name == av.name.name)
            # TODO: Here we are setting only the `ALLOCATED` flag for the array, but there are other operations missing,
            #  e.g., setting the sizes, offsets, and the actual memory allocation itself.
            asgn = ast_internal_classes.BinOp_Node(
                lval=ast_internal_classes.Name_Node(name=self._allocated_flag(avdecl)),
                op='=',
                rval=ast_internal_classes.Bool_Literal_Node(value='1'),
                line_number=alloc.line_number,
                parent=alloc.parent)
            self.execution_preludes[-1].append(asgn)

    def visit_Deallocate_Stmt_Node(self, dealloc: ast_internal_classes.Deallocate_Stmt_Node):
        for dv in dealloc.deallocation_list:
            # The declaration must exist somewhere in the function specification parts.
            dvdecl: ast_internal_classes.Var_Decl_Node = singular(
                v for v in mywalk(dv.parent.specification_part, ast_internal_classes.Var_Decl_Node)
                if v.name == dv.name)
            # TODO: Here we are setting only the `ALLOCATED` flag for the array, but there are other operations missing,
            #  e.g., the actual memory deallocation itself.
            asgn = ast_internal_classes.BinOp_Node(
                lval=ast_internal_classes.Name_Node(name=self._allocated_flag(dvdecl)),
                op='=',
                rval=ast_internal_classes.Bool_Literal_Node(value='0'),
                line_number=dealloc.line_number,
                parent=dealloc.parent)
            self.execution_preludes[-1].append(asgn)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        self.execution_preludes.append([])
        for ex in node.execution:
            ex = self.visit(ex)
            newbody.extend(reversed(self.execution_preludes[-1]))
            newbody.append(ex)
            self.execution_preludes[-1].clear()
        self.execution_preludes.pop()
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


def par_Decl_Range_Finder(node: ast_internal_classes.Array_Subscript_Node,
                          ranges: list,
                          rangeslen: list,
                          count: int,
                          newbody: list,
                          scope_vars: ScopeVarsDeclarations,
                          structures: Structures,
                          declaration=True,
                          main_iterator_ranges: Optional[list] = None,
                          allow_scalars=False):
    """
    Helper function for the transformation of array operations and sums to loops
    :param node: The AST to be transformed
    :param ranges: The ranges of the loop
    :param rangeslength: The length of ranges of the loop
    :param rangepos: The positions of the ranges
    :param count: The current count of the loop
    :param newbody: The new basic block that will contain the loop
    :param main_iterator_ranges: When parsing right-hand side of equation, use access to main loop range
    :return: Ranges, rangepos, newbody
    """

    rangepos = []
    currentindex = 0
    indices = []
    name_chain = []
    if isinstance(node, ast_internal_classes.Data_Ref_Node):

        # we assume starting from the top (left-most) data_ref_node
        # for struct1 % struct2 % struct3 % var
        # we find definition of struct1, then we iterate until we find the var

        struct_type = scope_vars.get_var(node.parent, node.parent_ref.name).type
        struct_def = structures.structures[struct_type]
        cur_node = node
        name_chain = [cur_node.parent_ref]
        while True:
            cur_node = cur_node.part_ref
            if isinstance(cur_node, ast_internal_classes.Data_Ref_Node):
                name_chain.append(cur_node.parent_ref)

            if isinstance(cur_node, ast_internal_classes.Array_Subscript_Node):
                struct_def = structures.structures[struct_type]
                offsets = struct_def.vars[cur_node.name.name].offsets
                node = cur_node
                break

            elif isinstance(cur_node, ast_internal_classes.Name_Node):
                struct_def = structures.structures[struct_type]

                var_def = struct_def.vars[cur_node.name]
                offsets = var_def.offsets

                # FIXME: is this always a desired behavior?

                # if we are passed a name node in the context of parDeclRange,
                # then we assume it should be a total range across the entire array
                array_sizes = var_def.sizes
                assert array_sizes is not None

                dims = len(array_sizes)
                node = ast_internal_classes.Array_Subscript_Node(
                    name=cur_node,
                    parent=node.parent,
                    type=var_def.type,
                    indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * dims)

                break

            struct_type = struct_def.vars[cur_node.parent_ref.name].type
            struct_def = structures.structures[struct_type]

    else:
        offsets = scope_vars.get_var(node.parent, node.name.name).offsets

    for idx, i in enumerate(node.indices):

        if isinstance(i, ast_internal_classes.ParDecl_Node):

            if i.type == "ALL":
                lower_boundary = None
                if offsets[idx] != 1:
                    # support symbols and integer literals
                    if isinstance(offsets[idx], (ast_internal_classes.Name_Node, ast_internal_classes.Data_Ref_Node,
                                                 ast_internal_classes.BinOp_Node)):
                        lower_boundary = offsets[idx]
                    else:
                        # check if offset is a number
                        try:
                            offset_value = int(offsets[idx])
                        except:
                            raise ValueError(f"Offset {offsets[idx]} is not a number")
                        lower_boundary = ast_internal_classes.Int_Literal_Node(value=str(offset_value))
                else:
                    lower_boundary = ast_internal_classes.Int_Literal_Node(value="1")

                first = True
                if len(name_chain) >= 1:
                    for i in name_chain:
                        if first:
                            first = False
                            array_name = i.name
                        else:
                            array_name = array_name + "_" + i.name
                    array_name = array_name + "_" + node.name.name
                else:
                    array_name = node.name.name
                upper_boundary = ast_internal_classes.Name_Range_Node(name="f2dace_MAX",
                                                                      type="INTEGER",
                                                                      arrname=ast_internal_classes.Name_Node(
                                                                          name=array_name,
                                                                          type="VOID",
                                                                          line_number=node.line_number),
                                                                      pos=idx)
                """
                    When there's an offset, we add MAX_RANGE + offset.
                    But since the generated loop has `<=` condition, we need to subtract 1.
                """
                if offsets[idx] != 1:

                    # support symbols and integer literals
                    if isinstance(offsets[idx], (ast_internal_classes.Name_Node, ast_internal_classes.Data_Ref_Node,
                                                 ast_internal_classes.BinOp_Node)):
                        offset = offsets[idx]
                    else:
                        try:
                            offset_value = int(offsets[idx])
                        except:
                            raise ValueError(f"Offset {offsets[idx]} is not a number")
                        offset = ast_internal_classes.Int_Literal_Node(value=str(offset_value))

                    upper_boundary = ast_internal_classes.BinOp_Node(lval=upper_boundary, op="+", rval=offset)
                    upper_boundary = ast_internal_classes.BinOp_Node(
                        lval=upper_boundary, op="-", rval=ast_internal_classes.Int_Literal_Node(value="1"))

                ranges.append([lower_boundary, upper_boundary])
                rangeslen.append(-1)

            else:
                ranges.append([i.range[0], i.range[1]])
                lower_boundary = i.range[0]

                start = 0
                if isinstance(i.range[0], ast_internal_classes.Int_Literal_Node):
                    start = int(i.range[0].value)
                else:
                    start = i.range[0]

                end = 0
                if isinstance(i.range[1], ast_internal_classes.Int_Literal_Node):
                    end = int(i.range[1].value)
                else:
                    end = i.range[1]

                if isinstance(end, int) and isinstance(start, int):
                    rangeslen.append(end - start + 1)
                else:
                    add = ast_internal_classes.BinOp_Node(lval=start,
                                                          op="+",
                                                          rval=ast_internal_classes.Int_Literal_Node(value="1"))
                    substr = ast_internal_classes.BinOp_Node(lval=end, op="-", rval=add)
                    rangeslen.append(substr)

            rangepos.append(currentindex)
            if declaration:
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(name="tmp_parfor_" + str(count + len(rangepos) - 1),
                                                              type="INTEGER",
                                                              sizes=None,
                                                              init=None,
                                                              parent=node.parent,
                                                              line_number=node.line_number)
                    ]))
            """
                To account for ranges with different starting offsets inside the same loop,
                we need to adapt array accesses.
                The main loop iterator is already initialized with the lower boundary of the dominating array.

                Thus, if the offset is the same, the index is just "tmp_parfor".
                Otherwise, it is "tmp_parfor - tmp_parfor_lower_boundary + our_lower_boundary"
            """

            if declaration:
                """
                    For LHS, we don't need to adjust - we dictate the loop iterator.
                """

                indices.append(ast_internal_classes.Name_Node(name="tmp_parfor_" + str(count + len(rangepos) - 1)))
            else:
                """
                    For RHS, we adjust starting array position by taking consideration the initial value
                    of the loop iterator.

                    Offset is handled by always subtracting the lower boundary.
                """
                current_lower_boundary = main_iterator_ranges[currentindex][0]

                indices.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="tmp_parfor_" +
                                                                                        str(count + len(rangepos) - 1)),
                                                    op="+",
                                                    rval=ast_internal_classes.BinOp_Node(lval=lower_boundary,
                                                                                         op="-",
                                                                                         rval=current_lower_boundary,
                                                                                         parent=node.parent),
                                                    parent=node.parent))
            currentindex += 1

        elif allow_scalars:

            ranges.append([i, i])
            rangeslen.append(1)
            indices.append(i)
            currentindex += 1
        else:
            indices.append(i)

    node.indices = indices


class ReplaceArrayConstructor(NodeTransformer):

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):

        if isinstance(node.rval, ast_internal_classes.Array_Constructor_Node):
            assigns = []
            for i in range(len(node.rval.value_list)):
                assigns.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Array_Subscript_Node(
                        name=node.lval,
                        indices=[ast_internal_classes.Int_Literal_Node(value=str(i + 1))],
                        type=node.type,
                        parent=node.parent),
                                                    op="=",
                                                    rval=node.rval.value_list[i],
                                                    line_number=node.line_number,
                                                    parent=node.parent,
                                                    typ=node.type))
            return ast_internal_classes.Execution_Part_Node(execution=assigns)
        return self.generic_visit(node)


class RenameVar(NodeTransformer):

    def __init__(self, oldname: str, newname: str):
        self.oldname = oldname
        self.newname = newname

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        return ast_internal_classes.Name_Node(name=self.newname) if node.name == self.oldname else node


class PartialRenameVar(NodeTransformer):

    def __init__(self, oldname: str, newname: str):
        self.oldname = oldname
        self.newname = newname

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        if hasattr(node, "type"):
            return ast_internal_classes.Name_Node(name=node.name.replace(self.oldname, self.newname),
                                                  parent=node.parent,
                                                  type=node.type) if self.oldname in node.name else node
        else:
            type = "VOID"
            return ast_internal_classes.Name_Node(name=node.name.replace(self.oldname, self.newname),
                                                  parent=node.parent,
                                                  type="VOID") if self.oldname in node.name else node

    def visit_Symbol_Decl_Node(self, node: ast_internal_classes.Symbol_Decl_Node):
        return ast_internal_classes.Symbol_Decl_Node(name=node.name.replace(self.oldname, self.newname),
                                                     type=node.type,
                                                     sizes=node.sizes,
                                                     init=node.init,
                                                     line_number=node.line_number,
                                                     kind=node.kind,
                                                     alloc=node.alloc,
                                                     offsets=node.offsets)


class IfConditionExtractor(NodeTransformer):
    """
    Ensures that each loop iterator is unique by extracting the actual iterator and assigning it to a uniquely named local variable
    """

    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:

            if isinstance(child, ast_internal_classes.If_Stmt_Node):
                if isinstance(child.cond, ast_internal_classes.BinOp_Node):
                    if child.cond.op == "==" and isinstance(child.cond.rval, ast_internal_classes.Int_Literal_Node):
                        if isinstance(child.cond.lval, ast_internal_classes.Name_Node):
                            if child.cond.lval.name == "jb_var_2205":
                                newbody.append(child)
                                continue

                old_cond = child.cond
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Var_Decl_Node(
                            name="_if_cond_" + str(self.count), type="INTEGER", sizes=None, init=None)
                    ]))
                newbody.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_if_cond_" +
                                                                                        str(self.count)),
                                                    op="=",
                                                    rval=old_cond,
                                                    line_number=child.line_number,
                                                    parent=child.parent))
                newcond = ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_if_cond_" +
                                                                                              str(self.count)),
                                                          op="==",
                                                          rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                          line_number=child.line_number,
                                                          parent=old_cond.parent)
                self.count += 1
                newifbody = self.visit(child.body)
                newelsebody = self.visit(child.body_else)

                newif = ast_internal_classes.If_Stmt_Node(cond=newcond,
                                                          body=newifbody,
                                                          body_else=newelsebody,
                                                          line_number=child.line_number,
                                                          parent=child.parent)

                newbody.append(newif)

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class WhileConditionExtractor(NodeTransformer):
    """
    Ensures that each loop iterator is unique by extracting the actual iterator and assigning it to a uniquely named local variable
    """

    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:

            if isinstance(child, ast_internal_classes.While_Stmt_Node):

                old_cond = child.cond
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Var_Decl_Node(
                            name="_while_cond_" + str(self.count), type="INTEGER", sizes=None, init=None)
                    ]))
                newbody.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_while_cond_" +
                                                                                        str(self.count)),
                                                    op="=",
                                                    rval=copy.deepcopy(old_cond),
                                                    line_number=child.line_number,
                                                    parent=child.parent))
                newcond = ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_while_cond_" +
                                                                                              str(self.count)),
                                                          op="==",
                                                          rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                          line_number=child.line_number,
                                                          parent=old_cond.parent)

                old_count = self.count
                self.count += 1
                newwhilebody = self.visit(child.body)
                newwhilebody.execution.append(
                    ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_while_cond_" +
                                                                                        str(old_count)),
                                                    op="=",
                                                    rval=copy.deepcopy(old_cond),
                                                    line_number=child.line_number,
                                                    parent=child.parent))

                newwhile = ast_internal_classes.While_Stmt_Node(cond=newcond,
                                                                body=newwhilebody,
                                                                line_number=child.line_number,
                                                                parent=child.parent)

                newbody.append(newwhile)

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ForDeclarer(NodeTransformer):
    """
    Ensures that each loop iterator is unique by extracting the actual iterator and assigning it to a uniquely named local variable
    """

    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Map_Stmt_Node):
                newbody.append(self.visit(child))
                continue
            if isinstance(child, ast_internal_classes.For_Stmt_Node):
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(
                            name="_for_it_" + str(self.count), type="INTEGER", sizes=None, init=None)
                    ]))
                final_assign = ast_internal_classes.BinOp_Node(lval=child.init.lval,
                                                               op="=",
                                                               rval=child.cond.rval,
                                                               line_number=child.line_number,
                                                               parent=child.parent)
                newfbody = RenameVar(child.init.lval.name, "_for_it_" + str(self.count)).visit(child.body)
                newcond = RenameVar(child.cond.lval.name, "_for_it_" + str(self.count)).visit(child.cond)
                newiter = RenameVar(child.iter.lval.name, "_for_it_" + str(self.count)).visit(child.iter)
                newinit = child.init
                newinit.lval = RenameVar(child.init.lval.name, "_for_it_" + str(self.count)).visit(child.init.lval)

                newfor = ast_internal_classes.For_Stmt_Node(init=newinit,
                                                            cond=newcond,
                                                            iter=newiter,
                                                            body=newfbody,
                                                            line_number=child.line_number,
                                                            parent=child.parent)
                self.count += 1
                newfor = self.visit(newfor)
                newbody.append(newfor)

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ElementalFunctionExpander(NodeTransformer):
    "Makes elemental functions into normal functions by creating a loop around thme if they are called with arrays"

    def __init__(self, func_list: list, ast):
        assert ast is not None
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)
        self.ast = ast

        self.func_list = func_list
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Call_Expr_Node):
                arrays = False
                sizes = None
                for i in self.func_list:
                    if child.name.name == i.name or child.name.name == i.name + "_srt":
                        print("F: " + child.name.name)
                        if hasattr(i, "elemental"):
                            print("El: " + str(i.elemental))
                            if i.elemental is True:
                                if len(child.args) > 0:
                                    for j in child.args:
                                        if isinstance(j, ast_internal_classes.Array_Subscript_Node):
                                            pardecls = [
                                                k for k in mywalk(j) if isinstance(k, ast_internal_classes.ParDecl_Node)
                                            ]
                                            if len(pardecls) > 0:
                                                arrays = True
                                                break
                                        elif isinstance(j, ast_internal_classes.Name_Node):

                                            var_def = self.scope_vars.get_var(child.parent, j.name)

                                            if var_def.sizes is not None:
                                                if len(var_def.sizes) > 0:
                                                    sizes = var_def.sizes
                                                    arrays = True
                                                    break

                if not arrays:
                    newbody.append(self.visit(child))
                else:
                    newbody.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Var_Decl_Node(
                                name="_for_elem_it_" + str(self.count), type="INTEGER", sizes=None, init=None)
                        ]))
                    newargs = []
                    # The range must be determined! It's currently hard set to 10
                    if sizes is not None:
                        if len(sizes) > 0:
                            shape = sizes
                        if len(sizes) > 1:
                            raise NotImplementedError("Only 1D arrays are supported")
                    # shape = ["10"]
                    for i in child.args:
                        if isinstance(i, ast_internal_classes.Name_Node):
                            newargs.append(
                                ast_internal_classes.Array_Subscript_Node(
                                    name=i,
                                    indices=[ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count))],
                                    line_number=child.line_number,
                                    type=i.type))
                            if i.name.startswith("tmp_call_"):
                                for j in newbody:
                                    if isinstance(j, ast_internal_classes.Decl_Stmt_Node):
                                        if j.vardecl[0].name == i.name:
                                            newbody[newbody.index(j)].vardecl[0].sizes = shape
                                            break
                        elif isinstance(i, ast_internal_classes.Array_Subscript_Node):
                            raise NotImplementedError("Not yet supported")
                            pardecl = [k for k in mywalk(i) if isinstance(k, ast_internal_classes.ParDecl_Node)]
                            if len(pardecl) != 1:
                                raise NotImplementedError("Only 1d array subscripts are supported")
                            ranges = []
                            rangesrval = []
                            par_Decl_Range_Finder(i, rangesrval, [], self.count, newbody, self.scope_vars,
                                                  self.ast.structures, False, ranges)
                            newargs.append(
                                ast_internal_classes.Array_Subscript_Node(
                                    name=i.name,
                                    indices=[ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count))],
                                    line_number=child.line_number,
                                    type=i.type))
                        else:
                            raise NotImplementedError("Only name nodes and array subscripts are supported")

                    newbody.append(
                        ast_internal_classes.For_Stmt_Node(
                            init=ast_internal_classes.BinOp_Node(
                                lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                op="=",
                                rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                line_number=child.line_number,
                                parent=child.parent),
                            cond=ast_internal_classes.BinOp_Node(
                                lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                op="<=",
                                rval=shape[0],
                                line_number=child.line_number,
                                parent=child.parent),
                            body=ast_internal_classes.Execution_Part_Node(execution=[
                                ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                    name=child.name,
                                                                    args=newargs,
                                                                    line_number=child.line_number,
                                                                    parent=child.parent,
                                                                    subroutine=child.subroutine)
                            ]),
                            line_number=child.line_number,
                            iter=ast_internal_classes.BinOp_Node(
                                lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                op="=",
                                rval=ast_internal_classes.BinOp_Node(
                                    lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                    op="+",
                                    rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                    parent=child.parent),
                                line_number=child.line_number,
                                parent=child.parent)))
                    self.count += 1

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class TypeInference(NodeTransformer):
    """
    """

    def __init__(self, ast, assert_voids=True, assign_scopes=True, scope_vars=None):
        self.assert_voids = assert_voids

        self.ast = ast
        if assign_scopes:
            ParentScopeAssigner().visit(ast)
        # if scope_vars is None:
        # we must always recompute, things might have changed
        if (True):
            self.scope_vars = ScopeVarsDeclarations(ast)
            self.scope_vars.visit(ast)
        else:
            self.scope_vars = scope_vars
        self.structures = ast.structures

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):

        # We always retrieve the newest type information since it could have been modified.
        try:
            var_def = self.scope_vars.get_var(node.parent, node.name)

            node.type = var_def.type
            node.sizes = var_def.sizes

            node.offsets = var_def.offsets

        except Exception as e:
            print(f"Ignore type inference for {node.name}")
            print(e)

        return node

    def visit_Name_Range_Node(self, node: ast_internal_classes.Name_Range_Node):
        node.sizes = []
        node.offsets = [1]
        return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        var_def = self.scope_vars.get_var(node.parent, node.name.name)
        node.type = var_def.type

        new_sizes = []
        new_indices = []
        for i, idx in enumerate(node.indices):

            idx = self.visit(idx)
            new_indices.append(idx)

            if isinstance(idx, ast_internal_classes.ParDecl_Node):

                if idx.type == 'ALL':
                    new_sizes.append(var_def.sizes[i])
                else:
                    new_sizes.append(
                        ast_internal_classes.BinOp_Node(
                            op='+',
                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                            lval=ast_internal_classes.Parenthesis_Expr_Node(
                                expr=ast_internal_classes.BinOp_Node(op='-', rval=idx.range[0], lval=idx.range[1])),
                            type="INTEGER"))
            else:

                # we also have non-contiguous arrays here!
                # (1) if the index is an array or another object that has non-trivial size:
                # (1a) if size is not 1D, then we throw an error - we don't support that.
                # (1b) if size is 1D, then size of this object becomes the new size of the array in this dimension
                #
                # (2) for all other cases, it's mostly "1"

                sizes = self._get_sizes(idx)

                if len(sizes) > 1:
                    raise NotImplementedError()

                if len(sizes) == 1:
                    new_sizes.append(sizes[0])
                else:
                    new_sizes.append(ast_internal_classes.Int_Literal_Node(value="1"))

        all_const = True
        for size in new_sizes:
            if not isinstance(size, ast_internal_classes.Int_Literal_Node) or not size.value == "1":
                all_const = False
        if all_const:
            new_sizes = []

        node.indices = new_indices
        node.sizes = new_sizes
        node.offsets = var_def.offsets

        return node

    def visit_Parenthesis_Expr_Node(self, node: ast_internal_classes.Parenthesis_Expr_Node):

        node.expr = self.visit(node.expr)
        node.type = node.expr.type
        node.sizes = self._get_sizes(node.expr)
        node.offsets = self._get_offsets(node.expr)
        return node

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        """
            Simple implementation of type promotion in binary ops.
        """

        node.lval = self.visit(node.lval)
        node.rval = self.visit(node.rval)
        """
            Handle promotion of numeric types.
        """

        type_hierarchy = ['VOID', 'LOGICAL', 'CHAR', 'INTEGER', 'REAL', 'DOUBLE']

        lval_type = self._get_type(node.lval)
        rval_type = self._get_type(node.rval)
        if lval_type in type_hierarchy and rval_type in type_hierarchy:

            idx_left = type_hierarchy.index(lval_type)
            idx_right = type_hierarchy.index(rval_type)

            node.type = type_hierarchy[max(idx_left, idx_right)]

        else:

            node.type = lval_type

        is_assignment = isinstance(node.lval, (ast_internal_classes.Name_Node))

        should_be_updated = False
        if node.lval.type == 'VOID':
            should_be_updated = True

        if self._get_sizes(node.lval) is None:
            should_be_updated = True

        # We do NOT overwrite size of a varible if it has been defined by the user.
        # We only do it for temporaries introduced by our transformations.
        # At this moment, the only way to distinguish between them is the `tmp_` prefix.
        if isinstance(node.lval, ast_internal_classes.Name_Node) and node.lval.name.startswith("tmp_"):
            should_be_updated = True

        if node.op == '=' and is_assignment and node.rval.type != 'VOID' and should_be_updated:

            lval_definition = self.scope_vars.get_var(node.parent, node.lval.name)

            lval_definition.type = node.type
            lval_definition.sizes = self._get_sizes(node.rval)
            lval_definition.offsets = self._get_offsets(node.rval)

            node.lval.type = node.type
            node.lval.sizes = lval_definition.sizes
            node.lval.offsets = lval_definition.offsets

        else:

            # We handle the following cases:
            #
            # (1)   Both sides of the binop have known types
            # (1a)  Both are not scalars - we take the lval for simplicity.
            #       The array must have same sizes, otherwise the program is malformed.
            #       But we can't determine this as sizes might be symbolic.
            # (1b)  One side is scalar and the other one is not - we take the array size.
            # (1c)  Both sides are scalar - trivial
            #
            # (2)   Only left or rval have determined sizes - we take that side.
            #
            # (3)   No sizes are known - we leave it like that.
            #       We need more information to determine that.

            left_size = self._get_sizes(node.lval) if node.lval.type != 'VOID' else None
            right_size = self._get_sizes(node.rval) if node.rval.type != 'VOID' else None

            if left_size is not None and right_size is not None:

                if len(left_size) > 0:
                    node.sizes = self._get_sizes(node.lval)
                    node.offsets = self._get_offsets(node.lval)
                elif len(right_size) > 0:
                    node.sizes = self._get_sizes(node.rval)
                    node.offsets = self._get_offsets(node.rval)
                else:
                    node.sizes = self._get_sizes(node.lval)
                    node.offsets = self._get_offsets(node.lval)

            elif left_size is not None:

                node.sizes = self._get_sizes(node.lval)
                node.offsets = self._get_offsets(node.lval)

            elif right_size is not None:

                node.sizes = self._get_sizes(node.rval)
                node.offsets = self._get_offsets(node.rval)

            else:
                node.sizes = None
                node.offsets = None

        if node.type == 'VOID':
            print("Couldn't infer the type for binop!")

        return node

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

        # if node.type != 'VOID':
        #    return node

        node.parent_ref = self.visit(node.parent_ref)
        node.part_ref = self.visit(node.part_ref)

        struct, variable, prev_part_ref = self.structures.find_definition(self.scope_vars, node)

        if prev_part_ref.part_ref.type != 'VOID':
            node.type = prev_part_ref.part_ref.type

        node.sizes = prev_part_ref.part_ref.sizes
        node.offsets = prev_part_ref.part_ref.offsets
        if node.sizes is None:
            node.sizes = []
            variable.sizes = []
            node.offsets = []
            variable.offsets = []

        return node

    def visit_Actual_Arg_Spec_Node(self, node: ast_internal_classes.Actual_Arg_Spec_Node):
        node.arg = self.visit(node.arg)

        func_arg_name_type = self._get_type(node.arg)
        if func_arg_name_type == 'VOID':

            func_arg = self.scope_vars.get_var(node.parent, node.arg.name)
            node.type = func_arg.type
            node.arg.type = func_arg.type
            node.sizes = self._get_sizes(func_arg)
            node.arg.sizes = self._get_sizes(func_arg)

        else:
            node.type = func_arg_name_type
            node.sizes = self._get_sizes(node.arg)

        return node

    def visit_UnOp_Node(self, node: ast_internal_classes.UnOp_Node):
        node.lval = self.visit(node.lval)
        if node.lval.type != 'VOID':
            node.type = node.lval.type
            node.sizes = self._get_sizes(node.lval)
            node.offsets = self._get_offsets(node.lval)
        return node

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        from dace.frontend.fortran.intrinsics import FortranIntrinsics

        new_args = []
        for arg in node.args:
            new_args.append(self.visit(arg))
        node.args = new_args

        sizes, offsets, return_type = FortranIntrinsics.output_size(node)
        if sizes is not None:
            node.sizes = sizes
            node.offsets = offsets
        else:
            node.sizes = None
            node.offsets = None

        if return_type != 'VOID':
            node.type = return_type

        return node

    def _get_type(self, node):

        if isinstance(node, ast_internal_classes.Int_Literal_Node):
            return 'INTEGER'
        elif isinstance(node, ast_internal_classes.Real_Literal_Node):
            return 'REAL'
        elif isinstance(node, ast_internal_classes.Bool_Literal_Node):
            return 'LOGICAL'
        else:
            return node.type

    def visit_Int_Literal_Node(self, node: ast_internal_classes.Int_Literal_Node):
        node.sizes = []
        node.offsets = [1]
        return node

    def visit_Double_Literal_Node(self, node: ast_internal_classes.Double_Literal_Node):
        node.sizes = []
        node.offsets = [1]
        return node

    def visit_Real_Literal_Node(self, node: ast_internal_classes.Real_Literal_Node):
        node.sizes = []
        node.offsets = [1]
        return node

    def visit_Bool_Literal_Node(self, node: ast_internal_classes.Bool_Literal_Node):
        node.sizes = []
        node.offsets = [1]
        return node

    def _get_offsets(self, node):

        if isinstance(node, (ast_internal_classes.Int_Literal_Node, ast_internal_classes.Real_Literal_Node,
                             ast_internal_classes.Bool_Literal_Node)):
            node.offsets = [1]
        return node.offsets

    def _get_sizes(self, node):

        if isinstance(node, (ast_internal_classes.Int_Literal_Node, ast_internal_classes.Real_Literal_Node,
                             ast_internal_classes.Bool_Literal_Node)):
            node.sizes = []
        return node.sizes


class PointerRemoval(NodeTransformer):

    def __init__(self):
        self.nodes = {}

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        if node.name.name in self.nodes:
            original_ref_node = self.nodes[node.name.name]

            cur_ref_node = original_ref_node
            new_ref_node = ast_internal_classes.Data_Ref_Node(parent_ref=cur_ref_node.parent_ref,
                                                              part_ref=None,
                                                              type=cur_ref_node.type,
                                                              line_number=cur_ref_node.line_number)
            newer_ref_node = new_ref_node

            while isinstance(cur_ref_node.part_ref, ast_internal_classes.Data_Ref_Node):
                cur_ref_node = cur_ref_node.part_ref
                newest_ref_node = ast_internal_classes.Data_Ref_Node(parent_ref=cur_ref_node.parent_ref,
                                                                     part_ref=None,
                                                                     type=cur_ref_node.type,
                                                                     line_number=cur_ref_node.line_number)
                newer_ref_node.part_ref = newest_ref_node
                newer_ref_node = newest_ref_node

            node.name = cur_ref_node.part_ref
            newer_ref_node.part_ref = node
            return new_ref_node
        else:
            return self.generic_visit(node)

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        if node.name in self.nodes:
            return self.nodes[node.name]
        return node

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Pointer_Assignment_Stmt_Node):
                self.nodes[child.name_pointer.name] = child.name_target
            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)

    def visit_Specification_Part_Node(self, node: ast_internal_classes.Specification_Part_Node):

        newspec = []

        symbols_to_remove = set()

        for i in node.specifications:

            if not isinstance(i, ast_internal_classes.Decl_Stmt_Node):
                newspec.append(self.visit(i))
            else:

                newdecls = []
                for var_decl in i.vardecl:

                    if var_decl.name in self.nodes:
                        if var_decl.sizes is not None:
                            for symbol in var_decl.sizes:
                                symbols_to_remove.add(symbol.name)
                        if var_decl.offsets is not None:
                            for symbol in var_decl.offsets:
                                symbols_to_remove.add(symbol.name)

                    else:
                        newdecls.append(var_decl)
                if len(newdecls) > 0:
                    newspec.append(ast_internal_classes.Decl_Stmt_Node(vardecl=newdecls))

        if node.symbols is not None:
            new_symbols = []
            for symbol in node.symbols:
                if symbol.name not in symbols_to_remove:
                    new_symbols.append(symbol)
        else:
            new_symbols = None

        return ast_internal_classes.Specification_Part_Node(specifications=newspec,
                                                            symbols=new_symbols,
                                                            typedecls=node.typedecls,
                                                            uses=node.uses,
                                                            enums=node.enums)


class ArgumentPruner(NodeVisitor):

    def __init__(self, funcs):

        self.funcs = funcs

        self.parsed_funcs: Dict[str, List[int]] = {}

        self.used_names = set()
        self.declaration_names = set()

        self.used_in_all_functions: Set[str] = set()

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        # if node.name not in self.used_names:
        #    print(f"Used name {node.name}")
        self.used_names.add(node.name)

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        self.declaration_names.add(node.name)

        # visit also sizes and offsets
        self.generic_visit(node)

    def generic_visit(self, node: ast_internal_classes.FNode):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value)

        for field, value in iter_attributes(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value)

    def _visit_function(self, node: ast_internal_classes.FNode):

        old_used_names = self.used_names
        self.used_names = set()
        self.declaration_names = set()

        self.visit(node.specification_part)

        self.visit(node.execution_part)

        new_args = []
        removed_args = []
        for idx, arg in enumerate(node.args):

            if not isinstance(arg, ast_internal_classes.Name_Node):
                raise NotImplementedError()

            if arg.name not in self.used_names:
                # print(f"Pruning argument {arg.name} of function {node.name.name}")
                removed_args.append(idx)
            else:
                # print(f"Leaving used argument {arg.name} of function {node.name.name}")
                new_args.append(arg)
        self.parsed_funcs[node.name.name] = removed_args

        declarations_to_remove = set()
        for x in self.declaration_names:
            if x not in self.used_names:
                # print(f"Marking removal variable {x}")
                declarations_to_remove.add(x)
            # else:
            # print(f"Keeping used variable {x}")

        for decl_stmt_node in node.specification_part.specifications:

            newdecl = []
            for decl in decl_stmt_node.vardecl:

                if not isinstance(decl, ast_internal_classes.Var_Decl_Node):
                    raise NotImplementedError()

                if decl.name not in declarations_to_remove:
                    # print(f"Readding declared variable {decl.name}")
                    newdecl.append(decl)
                # else:
                #    print(f"Pruning unused but declared variable {decl.name}")
            decl_stmt_node.vardecl = newdecl

        self.used_in_all_functions.update(self.used_names)
        self.used_names = old_used_names

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):

        if node.name.name not in self.parsed_funcs:
            self._visit_function(node)

        to_remove = self.parsed_funcs[node.name.name]
        for idx in reversed(to_remove):
            # print(f"Prune argument {node.args[idx].name} in {node.name.name}")
            del node.args[idx]

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):

        if node.name.name not in self.parsed_funcs:
            self._visit_function(node)

        to_remove = self.parsed_funcs[node.name.name]
        for idx in reversed(to_remove):
            del node.args[idx]

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        if node.name.name not in self.parsed_funcs:

            if node.name.name in self.funcs:
                self._visit_function(self.funcs[node.name.name])
            else:

                # now add actual arguments to the list of used names
                for arg in node.args:
                    self.visit(arg)

                return

        to_remove = self.parsed_funcs[node.name.name]
        for idx in reversed(to_remove):
            del node.args[idx]

        # now add actual arguments to the list of used names
        for arg in node.args:
            self.visit(arg)


class IfEvaluator(NodeTransformer):

    def __init__(self):
        self.replacements = 0

    def visit_If_Stmt_Node(self, node):
        try:
            text = ast_utils.TaskletWriter({}, {}).write_code(node.cond)
        except:
            text = None
            return self.generic_visit(node)
        # print(text)
        try:
            evaluated = sym.evaluate(sym.pystr_to_symbolic(text), {})
        except:
            # print("Failed: " + text)
            return self.generic_visit(node)

        if evaluated == sp.true:
            print("Expr: " + text + " eval to True replace")
            self.replacements += 1
            return node.body
        elif evaluated == sp.false:
            print("Expr: " + text + " eval to False replace")
            self.replacements += 1
            return node.body_else

        return self.generic_visit(node)


class AssignmentLister(NodeTransformer):

    def __init__(self, correction=[]):
        self.simple_assignments = []
        self.correction = correction

    def reset(self):
        self.simple_assignments = []

    def visit_BinOp_Node(self, node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                for i in self.correction:
                    if node.lval.name == i[0]:
                        node.rval.value = i[1]
            self.simple_assignments.append((node.lval, node.rval))
        return node


class AssignmentPropagator(NodeTransformer):

    def __init__(self, simple_assignments):
        self.simple_assignments = simple_assignments
        self.replacements = 0

    def visit_If_Stmt_Node(self, node):
        test = self.generic_visit(node)
        return ast_internal_classes.If_Stmt_Node(line_number=node.line_number,
                                                 cond=test.cond,
                                                 body=test.body,
                                                 body_else=test.body_else)

    def generic_visit(self, node: ast_internal_classes.FNode):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_internal_classes.FNode):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_internal_classes.FNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_internal_classes.FNode):
                done = False
                if isinstance(node, ast_internal_classes.BinOp_Node):
                    if node.op == "=":
                        if old_value == node.lval:
                            new_node = self.visit(old_value)
                            done = True
                if not done:
                    for i in self.simple_assignments:
                        if old_value == i[0]:
                            old_value = i[1]
                            self.replacements += 1
                            break
                        elif (isinstance(old_value, ast_internal_classes.Name_Node)
                              and isinstance(i[0], ast_internal_classes.Name_Node)):
                            if old_value.name == i[0].name:
                                old_value = i[1]
                                self.replacements += 1
                                break
                        elif (isinstance(old_value, ast_internal_classes.Data_Ref_Node)
                              and isinstance(i[0], ast_internal_classes.Data_Ref_Node)):
                            if (isinstance(old_value.part_ref, ast_internal_classes.Name_Node)
                                    and isinstance(i[0].part_ref, ast_internal_classes.Name_Node)
                                    and isinstance(old_value.parent_ref, ast_internal_classes.Name_Node)
                                    and isinstance(i[0].parent_ref, ast_internal_classes.Name_Node)):
                                if (old_value.part_ref.name == i[0].part_ref.name
                                        and old_value.parent_ref.name == i[0].parent_ref.name):
                                    old_value = i[1]
                                    self.replacements += 1
                                    break

                    new_node = self.visit(old_value)

                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class ReplaceImplicitParDecls(NodeTransformer):

    def __init__(self, scope_vars, structures):
        self.scope_vars = scope_vars
        self.structures = structures

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        return node

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

        _, var_def, last_data_ref_node = self.structures.find_definition(self.scope_vars, node)

        if var_def.sizes is None or len(var_def.sizes) == 0:
            return node

        if not isinstance(last_data_ref_node.part_ref, ast_internal_classes.Name_Node):
            return node

        last_data_ref_node.part_ref = ast_internal_classes.Array_Subscript_Node(
            name=last_data_ref_node.part_ref,
            parent=node.parent,
            type=var_def.type,
            indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var_def.sizes))

        return node

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        args = []
        for arg in node.args:
            args.append(self.visit(arg))
        node.args = args

        return node

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):

        var = self.scope_vars.get_var(node.parent, node.name)
        if var.sizes is not None and len(var.sizes) > 0:

            indices = [ast_internal_classes.ParDecl_Node(type='ALL')] * len(var.sizes)
            return ast_internal_classes.Array_Subscript_Node(name=node,
                                                             type=var.type,
                                                             parent=node.parent,
                                                             indices=indices,
                                                             line_number=node.line_number)
        else:
            return node


class ReplaceStructArgsLibraryNodesVisitor(NodeVisitor):
    """
    Finds all intrinsic operations that have to be transformed to loops in the AST
    """

    def __init__(self):
        self.nodes: List[ast_internal_classes.FNode] = []

        self.FUNCS_TO_REPLACE = ["transpose", "matmul"]

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        name = node.name.name.split('__dace_')
        if len(name) == 2 and name[1].lower() in self.FUNCS_TO_REPLACE:
            self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class ReplaceStructArgsLibraryNodes(NodeTransformer):

    def __init__(self, ast):

        self.ast = ast
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)
        self.structures = ast.structures

        self.counter = 0

        FUNCS_TO_REPLACE = ["transpose", "matmul"]

    # FIXME: copy-paste from intrinsics
    def _parse_struct_ref(self, node: ast_internal_classes.Data_Ref_Node) -> ast_internal_classes.FNode:

        # we assume starting from the top (left-most) data_ref_node
        # for struct1 % struct2 % struct3 % var
        # we find definition of struct1, then we iterate until we find the var

        struct_type = self.scope_vars.get_var(node.parent, node.parent_ref.name).type
        struct_def = self.ast.structures.structures[struct_type]
        cur_node = node

        while True:
            cur_node = cur_node.part_ref

            if isinstance(cur_node, ast_internal_classes.Array_Subscript_Node):
                struct_def = self.ast.structures.structures[struct_type]
                return struct_def.vars[cur_node.name.name]

            elif isinstance(cur_node, ast_internal_classes.Name_Node):
                struct_def = self.ast.structures.structures[struct_type]
                return struct_def.vars[cur_node.name]

            elif isinstance(cur_node, ast_internal_classes.Data_Ref_Node):
                struct_type = struct_def.vars[cur_node.parent_ref.name].type
                struct_def = self.ast.structures.structures[struct_type]

            else:
                raise NotImplementedError()

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):

        newbody = []

        for child in node.execution:

            lister = ReplaceStructArgsLibraryNodesVisitor()
            lister.visit(child)
            res = lister.nodes

            if res is None or len(res) == 0:
                newbody.append(self.visit(child))
                continue

            for call_node in res:

                args = []
                for arg in call_node.args:

                    if isinstance(arg, ast_internal_classes.Data_Ref_Node):

                        var = self._parse_struct_ref(arg)
                        tmp_var_name = f"tmp_libnode_{self.counter}"

                        node.parent.specification_part.specifications.append(
                            ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                ast_internal_classes.Var_Decl_Node(
                                    name=tmp_var_name, type=var.type, sizes=var.sizes, offsets=var.offsets, init=None)
                            ]))

                        # No need to create array subscript node - Array2Loop will catch that.
                        dest_node = ast_internal_classes.Name_Node(name=tmp_var_name,
                                                                   parent=call_node.parent,
                                                                   type=var.type)

                        if isinstance(arg.part_ref, ast_internal_classes.Name_Node):
                            arg.part_ref = ast_internal_classes.Array_Subscript_Node(
                                name=arg.part_ref,
                                parent=call_node.parent,
                                type=arg.part_ref.type,
                                indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var.sizes))

                        newbody.append(
                            ast_internal_classes.BinOp_Node(op="=",
                                                            lval=dest_node,
                                                            rval=arg,
                                                            line_number=child.line_number,
                                                            parent=child.parent))

                        self.counter += 1

                        args.append(ast_internal_classes.Name_Node(name=tmp_var_name, type=var.type))

                    else:
                        args.append(arg)

                call_node.args = args

            newbody.append(child)

        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ParDeclOffsetNormalizer(NodeTransformer):

    def __init__(self, ast):
        self.ast = ast
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)
        self.structures = ast.structures

        self.data_ref_stack = []

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

        # struct, variable, last_var = self.structures.find_definition(
        #    self.scope_vars, node
        # )

        # if not isinstance(last_var.part_ref, ast_internal_classes.Array_Subscript_Node):
        #    return node

        self.data_ref_stack.append(copy.deepcopy(node))
        node.part_ref = self.visit(node.part_ref)
        self.data_ref_stack.pop()

        return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        if len(self.data_ref_stack) > 0:
            _, array_var, _ = self.structures.find_definition(self.scope_vars, self.data_ref_stack[-1], node.name)
        else:
            array_var = self.scope_vars.get_var(node.parent, node.name.name)

        indices = []
        for idx, actual_index in enumerate(node.indices):

            self.current_offset = array_var.offsets[idx]
            if isinstance(self.current_offset, int):
                self.current_offset = ast_internal_classes.Int_Literal_Node(value=str(self.current_offset))
            indices.append(self.visit(actual_index))

        self.current_offset = None
        node.indices = indices
        return node

    def visit_ParDecl_Node(self, node: ast_internal_classes.ParDecl_Node):

        if self.current_offset is None:
            return node

        if node.type != 'RANGE':
            return node

        new_ranges = []
        for r in node.range:

            if r is None:
                new_ranges.append(r)
            else:
                # lower_boundary - offset + 1
                # we add +1 because offset normalization is applied later on
                new_ranges.append(
                    ast_internal_classes.BinOp_Node(op='+',
                                                    lval=ast_internal_classes.Int_Literal_Node(value="0"),
                                                    rval=ast_internal_classes.BinOp_Node(op='-',
                                                                                         lval=r,
                                                                                         rval=self.current_offset,
                                                                                         type=r.type),
                                                    type=r.type))

        node = ast_internal_classes.ParDecl_Node(type='RANGE', range=new_ranges)

        return node


class ArrayLoopLister(NodeVisitor):

    def __init__(self, scope_vars, structures):
        self.nodes: List[ast_internal_classes.Array_Subscript_Node] = []
        self.dataref_nodes: List[Tuple[ast_internal_classes.Data_Ref_Node,
                                       ast_internal_classes.Array_Subscript_Node]] = []

        self.scopes_vars = scope_vars
        self.structures = structures

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.nodes.append(node)

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):
        _, var_def, last_data_ref_node = self.structures.find_definition(self.scopes_vars, node)

        if isinstance(last_data_ref_node.part_ref, ast_internal_classes.Array_Subscript_Node):
            self.dataref_nodes.append((node, last_data_ref_node.part_ref))


class ArrayLoopExpander(NodeTransformer):
    """
        Transforms the AST by removing array expressions and replacing them with loops.

        This transformation is used as a general parent for replacing AST statements with a loop.
        In practice, we offer two child transformations that rely on this:
        - ArrayToLoop that replaces vectorizable operations like `a = b + c`, where all variables are arrays.
        - ElementalIntrinsicExpander that replaces calls to elemental functions with array arguments.

        Both transformations inherit from this base, and they define all *lister* classes that decide which
        AST nodes should be transformed.

    """

    @staticmethod
    def lister_type() -> Type:
        pass

    def __init__(self, ast, functions=None):
        self.count = 0

        self.ast = ast
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)

        self.functions = functions

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child_ in node.execution:
            lister = self.lister_type()(self.scope_vars, self.ast.structures, self.functions)
            lister.visit(child_)
            res = lister.nodes
            res_range = lister.range_nodes

            if res is None or len(res) == 0:
                newbody.append(self.visit(child_))
                continue

            # if res is not None and len(res) > 0:
            for child in res:

                if isinstance(child, ast_internal_classes.BinOp_Node):

                    current = child.lval
                    ranges = []
                    par_Decl_Range_Finder(current, ranges, [], self.count, newbody, self.scope_vars,
                                          self.ast.structures, True)

                    # if res_range is not None and len(res_range) > 0:

                    # catch cases where an array is used as name, without range expression
                    visitor = ReplaceImplicitParDecls(self.scope_vars, self.ast.structures)
                    child.rval = visitor.visit(child.rval)

                    rval_lister = ArrayLoopLister(self.scope_vars, self.ast.structures)
                    rval_lister.visit(child.rval)

                elif isinstance(child, ast_internal_classes.Call_Expr_Node):

                    # Important! We assume that all ranges are the same.
                    # No support for arguments with differnet dimensionality

                    all_ranges = []
                    for arg in child.args:

                        current = arg
                        ranges = []
                        # if we parsed an arg and it's still a name node, then it's a scalar
                        # we don't process it.
                        if not isinstance(arg, ast_internal_classes.Name_Node):
                            par_Decl_Range_Finder(current, ranges, [], self.count, newbody, self.scope_vars,
                                                  self.ast.structures, True)
                            all_ranges.append(ranges)
                        else:
                            all_ranges.append([])

                    for ranges in all_ranges[1:]:

                        if len(ranges) != len(all_ranges[0]):
                            warnings.warn(
                                f"Mismatch between dimensionality of array expansion, in line: {child.line_number}")

                    # For simplicity, the range is dictated by the first array
                    ranges = None
                    for r in all_ranges:
                        if len(r) > 0:
                            ranges = r
                            break

                    # catch cases where an array is used as name, without range expression
                    args = []
                    for arg in child.args:
                        visitor = ReplaceImplicitParDecls(self.scope_vars, self.ast.structures)
                        args.append(visitor.visit(arg))
                    child.args = args

                    rval_lister = ArrayLoopLister(self.scope_vars, self.ast.structures)
                    for arg in child.args:
                        rval_lister.visit(arg)

                else:
                    raise NotImplementedError()

                # rvals = [i for i in mywalk(child.rval) if isinstance(i, ast_internal_classes.Array_Subscript_Node)]
                for i in rval_lister.nodes:
                    rangesrval = []

                    par_Decl_Range_Finder(i, rangesrval, [], self.count, newbody, self.scope_vars, self.ast.structures,
                                          False, ranges)
                    for i, j in zip(ranges, rangesrval):
                        if i != j:
                            if isinstance(i, list) and isinstance(j, list) and len(i) == len(j):
                                for k, l in zip(i, j):
                                    if k != l:
                                        if isinstance(k, ast_internal_classes.Name_Range_Node) and isinstance(
                                                l, ast_internal_classes.Name_Range_Node):
                                            if k.name != l.name:
                                                raise NotImplementedError("Ranges must be the same")
                                        else:
                                            # this is not actually illegal.
                                            # raise NotImplementedError("Ranges must be the same")
                                            continue
                            else:
                                raise NotImplementedError("Ranges must be identical")

                for dataref in rval_lister.dataref_nodes:
                    rangesrval = []

                    i = dataref[0]

                    par_Decl_Range_Finder(i, rangesrval, [], self.count, newbody, self.scope_vars, self.ast.structures,
                                          False, ranges)
                    for i, j in zip(ranges, rangesrval):
                        if i != j:
                            if isinstance(i, list) and isinstance(j, list) and len(i) == len(j):
                                for k, l in zip(i, j):
                                    if k != l:
                                        if isinstance(k, ast_internal_classes.Name_Range_Node) and isinstance(
                                                l, ast_internal_classes.Name_Range_Node):
                                            if k.name != l.name:
                                                raise NotImplementedError("Ranges must be the same")
                                        else:
                                            # this is not actually illegal.
                                            # raise NotImplementedError("Ranges must be the same")
                                            continue
                            else:
                                raise NotImplementedError("Ranges must be identical")

                range_index = 0

                if isinstance(child, ast_internal_classes.BinOp_Node):
                    body = ast_internal_classes.BinOp_Node(lval=current,
                                                           op="=",
                                                           rval=child.rval,
                                                           line_number=child.line_number,
                                                           parent=child.parent)
                elif isinstance(child, ast_internal_classes.Call_Expr_Node):
                    body = child

                for i in ranges:
                    initrange = i[0]
                    finalrange = i[1]
                    init = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="=",
                        rval=initrange,
                        line_number=child.line_number,
                        parent=child.parent)
                    cond = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="<=",
                        rval=finalrange,
                        line_number=child.line_number,
                        parent=child.parent)
                    iter = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="=",
                        rval=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                            op="+",
                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                            parent=child.parent),
                        line_number=child.line_number,
                        parent=child.parent)
                    current_for = ast_internal_classes.Map_Stmt_Node(
                        init=init,
                        cond=cond,
                        iter=iter,
                        body=ast_internal_classes.Execution_Part_Node(execution=[body]),
                        line_number=child.line_number,
                        parent=child.parent)
                    body = current_for
                    range_index += 1

                newbody.append(body)

                self.count = self.count + range_index
            # else:
            #    newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ArrayLoopNodeLister(NodeVisitor):
    """
    Finds all array operations that have to be transformed to loops in the AST

    For each binary operator, we first check if LHS has indices with pardecl.
    If yes, then this is a candidate for replacement.

    If not, then we check if LHS is a name of an array or data reference pointing to an array.
    We explicitly exclude here expressions `arr = func(input)` as these have different replacement rules.

    If array is but implicit, e.g., just array name, then we have to replace it with a pardecl operation such that
    the main processing function can replace them with proper accesess dependent on loop iterators.
    """

    def __init__(self, scope_vars, structures, _):
        self.nodes: List[ast_internal_classes.FNode] = []
        self.range_nodes: List[ast_internal_classes.FNode] = []

        self.scope_vars = scope_vars
        self.structures = structures

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        rval_pardecls = [i for i in mywalk(node.rval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        lval_pardecls = [i for i in mywalk(node.lval) if isinstance(i, ast_internal_classes.ParDecl_Node)]

        if not lval_pardecls:

            # Handle edge case - the left hand side is an array
            # But we don't have a pardecl.
            #
            # This means that we target a NameNode that refers to an array
            # Same logic applies to structures
            #
            # BUT: we explicitly exclude patterns like arr = func()
            if isinstance(node.lval,
                          (ast_internal_classes.Name_Node, ast_internal_classes.Data_Ref_Node)) and not isinstance(
                              node.rval, ast_internal_classes.Call_Expr_Node):

                if isinstance(node.lval, ast_internal_classes.Name_Node):

                    var = self.scope_vars.get_var(node.lval.parent, node.lval.name)
                    if var.sizes is None or len(var.sizes) == 0:
                        return

                    node.lval = ast_internal_classes.Array_Subscript_Node(
                        name=node.lval,
                        parent=node.parent,
                        type=var.type,
                        indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var.sizes))

                else:
                    _, var_def, last_data_ref_node = self.structures.find_definition(self.scope_vars, node.lval)

                    if var_def.sizes is None or len(var_def.sizes) == 0:
                        return

                    if not isinstance(last_data_ref_node.part_ref, ast_internal_classes.Name_Node):
                        return

                    last_data_ref_node.part_ref = ast_internal_classes.Array_Subscript_Node(
                        name=last_data_ref_node.part_ref,
                        parent=node.parent,
                        type=var_def.type,
                        indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var_def.sizes))

            else:
                return

        if rval_pardecls:
            self.range_nodes.append(node)
        self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class ArrayToLoop(ArrayLoopExpander):
    """
        Transforms the AST by removing expressions arr = <arbitrary-ast-operators>(input-arrays...) a replacing them with loops:

        for i in len(input):
            arr(i) = <arbitrary-ast-operators>(input-array-1(i), input-array-2(i), ...)

        It should only apply when arguments and destination are arrays.
        We assume the Fortran code is well-formed and we do not check carefully if sizes and ranks of arguments match.
    """

    @staticmethod
    def lister_type() -> Type:
        return ArrayLoopNodeLister

    def __init__(self, ast):
        super().__init__(ast)


class ElementalIntrinsicNodeLister(NodeVisitor):
    """
    Finds all elemental operations that have to be transformed to loops in the AST.

    For each binary operator, we first check if RHS is call expression, and the function called is elemental.

    Then, we look if the destination is an array - explicitly or implicitly.
    if it is implicit, e.g., just array name, then we have to replace it with a pardecl operation such that
    the main processing function can replace them with proper accesess dependent on loop iterators.

    We have the following cases:
    - Explicit pardecl in array indices, which we leave as it is.
    - Name node that refers to an array, which we replace with pardecl ALL
    - Data reference, which we have to parse and check if the last element is an array
    """

    def __init__(self, scope_vars, structures, functions: List[ast_internal_classes.Name_Node]):
        self.nodes: List[ast_internal_classes.FNode] = []
        self.range_nodes: List[ast_internal_classes.FNode] = []

        self.scope_vars = scope_vars
        self.structures = structures

        self.ELEMENTAL_INTRINSICS = set(["EXP", "MAX", "MIN"])

        self.ELEMENTAL_FUNCTIONS = set()

        for func in functions:
            if hasattr(func, "elemental") and func.elemental:
                self.ELEMENTAL_FUNCTIONS.add(func.name)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return node

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        is_elemental_intrinsic = node.name.name.startswith('__dace') and node.name.name.split(
            '__dace_')[1] in self.ELEMENTAL_INTRINSICS

        # for functions changed into subroutines, their name is adapted as {name}_srt
        is_elemental = node.name.name.split('_srt')[0] in self.ELEMENTAL_FUNCTIONS

        if not is_elemental_intrinsic and not is_elemental:
            return

        args_pardecls = [
            i for arg in node.args for i in mywalk(arg) if isinstance(i, ast_internal_classes.ParDecl_Node)
        ]

        if len(args_pardecls) > 0:
            self.nodes.append(node)
        else:

            # Handle edge case - args have an array
            # But we don't have a pardecl

            needs_expansion = False

            args = []
            for arg in node.args:

                if isinstance(arg, ast_internal_classes.Name_Node):

                    var = self.scope_vars.get_var(node.parent, arg.name)
                    if var.sizes is None or len(var.sizes) == 0:
                        args.append(arg)
                        continue

                    args.append(
                        ast_internal_classes.Array_Subscript_Node(
                            name=arg,
                            parent=node.parent,
                            type=var.type,
                            indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var.sizes)))

                    needs_expansion = True

                elif isinstance(arg, ast_internal_classes.Data_Ref_Node):

                    _, var_def, last_data_ref_node = self.structures.find_definition(self.scope_vars, arg)

                    if var_def.sizes is None or len(var_def.sizes) == 0:
                        args.append(arg)
                        continue

                    if not isinstance(last_data_ref_node.part_ref, ast_internal_classes.Name_Node):
                        args.append(arg)
                        continue

                    last_data_ref_node.part_ref = ast_internal_classes.Array_Subscript_Node(
                        name=last_data_ref_node.part_ref,
                        parent=node.parent,
                        type=var_def.type,
                        indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var_def.sizes))

                    args.append(arg)

                    needs_expansion = True

                else:
                    args.append(arg)

            if needs_expansion:
                node.args = args
                self.nodes.append(node)

        return

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):

        lval_pardecls = [i for i in mywalk(node.lval) if isinstance(i, ast_internal_classes.ParDecl_Node)]

        # we explicitly ask look for patterns arr = func()
        if not isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
            return

        is_elemental_intrinsic = node.rval.name.name.split('__dace_')[1] in self.ELEMENTAL_INTRINSICS

        # for functions changed into subroutines, their name is adapted as {name}_srt
        is_elemental = node.rval.name.name.split('_srt')[0] in self.ELEMENTAL_FUNCTIONS

        if not is_elemental_intrinsic and not is_elemental:
            return

        if len(lval_pardecls) > 0:
            self.nodes.append(node)
        else:

            # Handle edge case - the left hand side is an array
            # But we don't have a pardecl

            if isinstance(node.lval, ast_internal_classes.Name_Node):

                var = self.scope_vars.get_var(node.lval.parent, node.lval.name)

                # if var.type == 'VOID':
                #    raise NeedsTypeInferenceException(node.rval.name.name, node.line_number)

                if var.sizes is None or len(var.sizes) == 0:
                    return

                node.lval = ast_internal_classes.Array_Subscript_Node(
                    name=node.lval,
                    parent=node.parent,
                    type=var.type,
                    indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var.sizes))
                self.nodes.append(node)

            elif isinstance(node.lval, ast_internal_classes.Data_Ref_Node):

                _, var_def, last_data_ref_node = self.structures.find_definition(self.scope_vars, node.lval)

                if var_def.sizes is None or len(var_def.sizes) == 0:
                    return

                if not isinstance(last_data_ref_node.part_ref, ast_internal_classes.Name_Node):
                    return

                last_data_ref_node.part_ref = ast_internal_classes.Array_Subscript_Node(
                    name=last_data_ref_node.part_ref,
                    parent=node.parent,
                    type=var_def.type,
                    indices=[ast_internal_classes.ParDecl_Node(type='ALL')] * len(var_def.sizes))
                self.nodes.append(node)


class ElementalIntrinsicExpander(ArrayLoopExpander):
    """
        Transforms the AST by removing expressions arr = func(input) a replacing them with loops:

        for i in len(input):
            arr(i) = func(input(i))

        This should only apply under the following conditions:
        - The function is an elemental intrinsic.
        - Arguments are array of the same rank and size.

        Currently, we do not check completely for the size as they can be symbolic.
        We assume that the Fortran code is well-formed, and we check if the arguments are arrays.
    """

    @staticmethod
    def lister_type() -> Type:
        return ElementalIntrinsicNodeLister

    def __init__(self, functions, ast):
        super().__init__(ast, functions)


class ParDeclNonContigArrayExpander(NodeTransformer):
    """
        Variable processer lets the main function know that the following array needs to be processed:
        - Counter to determine iterator name and the name of temporary array.
        - Tuple of (source, main_source) containing original array. For structures, this contains the variable defining array
          and the surrounding data ref node.
        - Type of temporary array.
        - Sizes and offsets of temporary array.
        - Indices for each dimension.
        - For each non-contiguous dimension, we send tuple (idx, main_var, var) - index of dimension, and the two variables defining
          array index. main_var is used to send data ref nodes for structure references
    """
    NonContigArray = namedtuple("NonContigArray", "counter name source type sizes offsets indices noncontig_dims")

    def __init__(self, ast):
        self.ast = ast
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)

        self.structures = ast.structures

        self.nodes_to_process = {}
        self.counter = 0

        self.data_ref_stack = []

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:

            self.nodes_to_process = {}
            n = self.visit(child)

            # Restart type inference to ensure that we reassign size
            if len(self.nodes_to_process) > 0 and isinstance(n, ast_internal_classes.BinOp_Node):
                n.lval.sizes = None

                if isinstance(n.lval, ast_internal_classes.Name_Node):
                    var = self.scope_vars.get_var(node.parent, n.lval.name)
                    var.sizes = None
                elif isinstance(n.lval, ast_internal_classes.Array_Subscript_Node):
                    var = self.scope_vars.get_var(node.parent, n.lval.name.name)
                    var.sizes = None

            for tmp_array_name, tmp_array in self.nodes_to_process.items():
                """
                    For each variable with non-contiguous slices:
                    - Generate the temporary array.
                    - Determine the copy operation b[idx] = a[indices[idx]].
                    - Generate loop, with one iterator for each dimension
                      that needs a copy.

                    We also recursively process indices of arrays to handle cases of nested,
                    non-contiguous arrays.
                    These are handled by visitors, and we expected that nested arrays are added
                    to `nodes_to_process` first.
                """

                node.parent.specification_part.specifications.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Var_Decl_Node(
                            name=tmp_array.name,
                            type=tmp_array.type,
                            sizes=tmp_array.sizes,
                            offsets=tmp_array.offsets,
                            init=None,
                        )
                    ]))

                dest_indices = copy.deepcopy(tmp_array.indices)
                for idx, _, _ in tmp_array.noncontig_dims:
                    iter_var = ast_internal_classes.Name_Node(name=f"tmp_parfor_{tmp_array.counter}_{idx}")
                    dest_indices[idx] = iter_var

                dest = ast_internal_classes.Array_Subscript_Node(
                    name=ast_internal_classes.Name_Node(name=tmp_array.name),
                    type=tmp_array.type,
                    indices=dest_indices,
                    line_numbe=child.line_number)

                source_indices = copy.deepcopy(tmp_array.indices)
                for idx, main_var, var in tmp_array.noncontig_dims:
                    iter_var = ast_internal_classes.Name_Node(name=f"tmp_parfor_{tmp_array.counter}_{idx}")

                    var.indices = [iter_var]
                    source_indices[idx] = main_var

                source, main_source = tmp_array.source
                main_source.indices = source_indices

                body = ast_internal_classes.BinOp_Node(lval=dest,
                                                       op="=",
                                                       rval=source,
                                                       line_number=child.line_number,
                                                       parent=child.parent)

                for idx, _, _ in tmp_array.noncontig_dims:
                    initrange = ast_internal_classes.Int_Literal_Node(value="1")
                    finalrange = tmp_array.sizes[idx]

                    iter_var = ast_internal_classes.Name_Node(name=f"tmp_parfor_{tmp_array.counter}_{idx}")

                    node.parent.specification_part.specifications.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Var_Decl_Node(
                                name=iter_var.name, type='INTEGER', sizes=[], offsets=[1], init=None)
                        ]))

                    init = ast_internal_classes.BinOp_Node(lval=iter_var,
                                                           op="=",
                                                           rval=initrange,
                                                           line_number=child.line_number,
                                                           parent=child.parent)
                    cond = ast_internal_classes.BinOp_Node(lval=iter_var,
                                                           op="<=",
                                                           rval=finalrange,
                                                           line_number=child.line_number,
                                                           parent=child.parent)
                    iter = ast_internal_classes.BinOp_Node(lval=iter_var,
                                                           op="=",
                                                           rval=ast_internal_classes.BinOp_Node(
                                                               lval=iter_var,
                                                               op="+",
                                                               rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                               parent=child.parent),
                                                           line_number=child.line_number,
                                                           parent=child.parent)
                    current_for = ast_internal_classes.Map_Stmt_Node(
                        init=init,
                        cond=cond,
                        iter=iter,
                        body=ast_internal_classes.Execution_Part_Node(execution=[body]),
                        line_number=child.line_number,
                        parent=child.parent)
                    body = current_for

                newbody.append(body)

            newbody.append(n)

        return ast_internal_classes.Execution_Part_Node(execution=newbody)

    # def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

    def _handle_name_node(self, idx, actual_index, var, new_sizes, noncont_sizes, new_offsets):

        if len(var.sizes) == 1:

            needs_process = True
            new_sizes.append(var.sizes[0])
            noncont_sizes.append((idx, actual_index))

        elif len(var.sizes) == 0:
            """
                For scalar-based access, the size of new dimension is just 1.
            """
            new_sizes.append(ast_internal_classes.Int_Literal_Node(value="1"))
        else:
            raise NotImplementedError("Non-contiguous array slices are supported only for 1D indices!")

        new_offsets.append(1)

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):
        """
            When processing a data reference:

            struct%val

            We note down structure reference and take val for further analysis.
            This way, we reuse the same logic without excessive specialization to datarefs.
        """

        struct, variable, last_var = self.structures.find_definition(self.scope_vars, node)

        if not isinstance(last_var.part_ref, ast_internal_classes.Array_Subscript_Node):
            return node

        self.data_ref_stack.append(copy.deepcopy(node))
        node.part_ref = self.visit(node.part_ref)
        self.data_ref_stack.pop()

        if self.needs_copy:
            return node.part_ref
        else:
            return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        indices = []
        self.needs_copy = False
        new_sizes = []
        new_offsets = []
        cont_sizes = []
        noncont_sizes = []

        new_indices = []
        for idx, index in enumerate(node.indices):
            old_data_ref_stack = self.data_ref_stack
            self.data_ref_stack = []
            new_indices.append(self.visit(index))
            self.data_ref_stack = old_data_ref_stack

        for idx, index in enumerate(new_indices):
            """
                For structure references:
                index -> full index var inserted in array node
                actual_index -> the variable that contains actual data

                For other, both are the same.
            """

            if isinstance(index, ast_internal_classes.Data_Ref_Node):

                struct, variable, last_var = self.structures.find_definition(self.scope_vars, index)

                actual_index = last_var.part_ref
            else:
                actual_index = index

            if isinstance(actual_index, ast_internal_classes.Name_Node):

                if actual_index.name in self.nodes_to_process:
                    var_sizes = self.nodes_to_process[actual_index.name].sizes
                else:
                    var_sizes = self.scope_vars.get_var(node.parent, actual_index.name).sizes

                if var_sizes is not None and len(var_sizes) == 1:

                    self.needs_copy = True
                    new_sizes.append(var_sizes[0])
                    """
                        We will later assign actual indices when processing the loop code.
                    """
                    idx_arr = ast_internal_classes.Array_Subscript_Node(
                        name=ast_internal_classes.Name_Node(name=actual_index.name),
                        type=actual_index.type,
                        indices=None,
                        line_number=actual_index.line_number)
                    if isinstance(index, ast_internal_classes.Data_Ref_Node):
                        last_var.part_ref = idx_arr
                        noncont_sizes.append((idx, index, idx_arr))
                    else:
                        noncont_sizes.append((idx, idx_arr, idx_arr))

                elif var_sizes is None or len(var_sizes) == 0:
                    """
                        For scalar-based access, the size of new dimension is just 1.
                    """
                    new_sizes.append(ast_internal_classes.Int_Literal_Node(value="1"))
                else:
                    raise NotImplementedError("Non-contiguous array slices are supported only for 1D indices!")

                new_offsets.append(1)

            elif isinstance(actual_index, ast_internal_classes.ParDecl_Node):

                if node.name.name in self.nodes_to_process:
                    var_sizes = self.nodes_to_process[node.name.name].sizes
                    var_offsets = self.nodes_to_process[node.name.name].offsets
                else:
                    var_sizes = self.scope_vars.get_var(node.parent, node.name.name).sizes
                    var_offsets = self.scope_vars.get_var(node.parent, node.name.name).offsets
                """
                    For range from a:b, we do not generate the code manually.
                    Instead, we will let ArrayToLoop transformation to expand this copy.

                    To make sure that we correctly generate copy indices, the destination
                    array needs to have the same offset as the low range boundary, e.g,
                    tmp_dest(40:45, ...) = source(40:45, ...)

                    This offset will be normalized before translating to SDFG, in IndexExtractor transformation.
                """

                if actual_index.type == 'ALL':

                    new_sizes.append(var_sizes[idx])
                    new_offsets.append(var_offsets[idx])

                elif actual_index.type == 'RANGE':

                    if isinstance(actual_index.range[0], ast_internal_classes.Int_Literal_Node):
                        new_offsets.append(actual_index.range[0].value)
                    else:
                        new_offsets.append(actual_index.range[0])
                    """
                        For range-based offsets, the array size is given as:
                        high - low + 1
                    """
                    new_sizes.append(
                        ast_internal_classes.BinOp_Node(lval=ast_internal_classes.BinOp_Node(
                            lval=actual_index.range[1],
                            op="-",
                            rval=actual_index.range[0],
                            line_number=node.line_number,
                            parent=node.parent),
                                                        op="+",
                                                        rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                        line_number=node.line_number,
                                                        parent=node.parent))
                else:
                    raise NotImplementedError()

            else:
                """
                    For scalar-based access, the size of new dimension is just 1.
                """
                new_sizes.append(ast_internal_classes.Int_Literal_Node(value="1"))
                new_offsets.append(1)

            indices.append(index)

        if self.needs_copy:

            if len(node.indices) > 2:
                raise NotImplementedError("Non-contiguous array slices are supported only for 1D/2D arrays!")
            """
                We will later assign actual indices when processing the loop code.
            """
            source = ast_internal_classes.Array_Subscript_Node(
                name=ast_internal_classes.Name_Node(name=node.name.name),
                type=node.type,
                # will be fixed in further processing
                indices=None,
                line_number=node.line_number)
            main_source = source
            for dataref in self.data_ref_stack[::-1]:
                dataref.part_ref = source
                source = dataref

            name = f"tmp_noncontig_{self.counter}"
            self.nodes_to_process[name] = self.NonContigArray(self.counter, name, (source, main_source), node.type,
                                                              new_sizes, new_offsets, indices, noncont_sizes)
            self.counter += 1

            return ast_internal_classes.Name_Node(name=name, type=node.type, line_number=node.line_number)
        else:
            node.indices = indices
            return node
