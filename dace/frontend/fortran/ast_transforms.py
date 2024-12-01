# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from typing import Dict, List, Optional, Tuple, Set, Union

import sympy as sp

from dace import symbolic as sym
from dace.frontend.fortran import ast_internal_classes, ast_utils


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

    def find_definition(self, scope_vars, node: ast_internal_classes.Data_Ref_Node,
                        variable_name: Optional[ast_internal_classes.Name_Node] = None):

        # we assume starting from the top (left-most) data_ref_node
        # for struct1 % struct2 % struct3 % var
        # we find definition of struct1, then we iterate until we find the var

        struct_type = scope_vars.get_var(node.parent, ast_utils.get_name(node.parent_ref)).type
        struct_def = self.structures[struct_type]
        cur_node = node

        while True:
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
                    return struct_def, struct_def.vars[cur_node.parent_ref.name.name]

                struct_type = struct_def.vars[cur_node.parent_ref.name.name].type
            else:

                if variable_name is not None and cur_node.parent_ref.name == variable_name.name:
                    return struct_def, struct_def.vars[cur_node.parent_ref.name]

                struct_type = struct_def.vars[cur_node.parent_ref.name].type
            struct_def = self.structures[struct_type]

        return struct_def, cur_var


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
        new_node = self.generic_visit(node)
        print("Subroutine: ", node.name.name)
        if self.current_class is not None:
            for i in self.classes:
                if i.is_class is True:
                    if i.name.name == self.current_class.name.name:
                        for j in i.procedure_part.procedures:
                            if j.name.name == node.name.name:
                                return ast_internal_classes.Subroutine_Subprogram_Node(
                                    name=ast_internal_classes.Name_Node(name=i.name.name + "_" + node.name.name,
                                                                        type=node.type),
                                    args=new_node.args,
                                    specification_part=new_node.specification_part,
                                    execution_part=new_node.execution_part,
                                    mandatory_args_count=new_node.mandatory_args_count,
                                    optional_args_count=new_node.optional_args_count,
                                    elemental=new_node.elemental,
                                    line_number=new_node.line_number)
                            elif hasattr(j, "args") and j.args[2] is not None:
                                if j.args[2].name == node.name.name:
                                    return ast_internal_classes.Subroutine_Subprogram_Node(
                                        name=ast_internal_classes.Name_Node(name=i.name.name + "_" + j.name.name,
                                                                            type=node.type),
                                        args=new_node.args,
                                        specification_part=new_node.specification_part,
                                        execution_part=new_node.execution_part,
                                        mandatory_args_count=new_node.mandatory_args_count,
                                        optional_args_count=new_node.optional_args_count,
                                        elemental=new_node.elemental,
                                        line_number=new_node.line_number)
        return new_node

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if self.current_class is not None:
            for i in self.classes:
                if i.is_class is True:
                    if i.name.name == self.current_class.name.name:
                        for j in i.procedure_part.procedures:
                            if j.name.name == node.name.name:
                                return ast_internal_classes.Call_Expr_Node(
                                    name=ast_internal_classes.Name_Node(name=i.name.name + "_" + node.name.name,
                                                                        type=node.type, args=node.args,
                                                                        line_number=node.line_number), args=node.args,
                                    type=node.type, line_number=node.line_number)
        return self.generic_visit(node)


class FindFunctionAndSubroutines(NodeVisitor):
    """
    Finds all function and subroutine names in the AST
    :return: List of names
    """

    def __init__(self):
        self.names: List[ast_internal_classes.Name_Node] = []
        self.module_based_names: Dict[str,List[ast_internal_classes.Name_Node]] = {}
        self.nodes: Dict[str, ast_internal_classes.FNode] = {}
        self.iblocks: Dict[str, List[str]] = {}
        self.current_module = None

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        ret = node.name
        ret.elemental = node.elemental
        self.names.append(ret)
        self.nodes[ret.name] = node
        self.module_based_names[self.current_module].append(ret)

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):
        ret = node.name
        ret.elemental = node.elemental
        self.names.append(ret)
        self.nodes[ret.name] = node
        self.module_based_names[self.current_module].append(ret)

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

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        self.names.append(node.name)


class FindInputs(NodeVisitor):
    """
    Finds all inputs (reads) in the AST node and its children
    :return: List of names
    """

    def __init__(self):

        self.nodes: List[ast_internal_classes.Name_Node] = []

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


class FindFunctionCalls(NodeVisitor):
    """
    Finds all function calls in the AST node and its children
    :return: List of names
    """

    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        self.nodes.append(node)
        for i in node.args:
            self.visit(i)


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
                        # print("Eliminating pointer "+self.pointer_name+" of type "+ k.type +" in struct "+self.parent_struct)
                        continue
                    else:
                        vardecl.append(k)
                if vardecl != []:
                    component_part.component_def_stmts.append(ast_internal_classes.Data_Component_Def_Stmt_Node(
                        vars=ast_internal_classes.Decl_Stmt_Node(vardecl=vardecl, parent=node.parent),
                        parent=node.parent))
            newnode.component_part = component_part
            return newnode
        else:
            return node


class StructConstructorToFunctionCall(NodeTransformer):
    """
    Fortran does not differentiate between structure constructors and functions without arguments.
    We need to go over and convert all structure constructors that are in fact functions and transform them.
    So, we create a closure of all math and defined functions and 
    transform if necessary.
    """

    def __init__(self, funcs=None):
        if funcs is None:
            funcs = []
        self.funcs = funcs

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        self.excepted_funcs = [
            "malloc", "pow", "cbrt", "__dace_sign", "tanh", "atan2",
            "__dace_epsilon", *FortranIntrinsics.function_names()
        ]

    def visit_Structure_Constructor_Node(self, node: ast_internal_classes.Structure_Constructor_Node):
        if isinstance(node.name, str):
            return node
        if node.name is None:
            raise ValueError("Structure name is None")
            return ast_internal_classes.Char_Literal_Node(value="Error!", type="CHARACTER")
        found = False
        for i in self.funcs:
            if i.name == node.name.name:
                found = True
                break
        if node.name.name in self.excepted_funcs or found:
            processed_args = []
            for i in node.args:
                arg = StructConstructorToFunctionCall(self.funcs).visit(i)
                processed_args.append(arg)
            node.args = processed_args
            return ast_internal_classes.Call_Expr_Node(
                name=ast_internal_classes.Name_Node(name=node.name.name, type="VOID", line_number=node.line_number),
                args=node.args, line_number=node.line_number, type="VOID")

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
        self.rename_dict=dict

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        self.excepted_funcs = [
            "malloc", "pow", "cbrt", "__dace_sign", "__dace_allocated", "tanh", "atan2",
            "__dace_epsilon","__dace_exit","surrtpk","surrtab","surrtrf","abor1", *FortranIntrinsics.function_names()
        ]
        #

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if isinstance(node.name, str):
            return node
        assert node.name is not None, f"not a valid call expression, got: {node} / {type(node)}"
        name = node.name.name

        found_in_names= name in [i.name for i in self.funcs.names]
        found_in_renames = False
        if self.rename_dict is not None:
            for k,v in self.rename_dict.items():
                for original_name,replacement_names in v.items():
                    if name in replacement_names:
                        found_in_renames = True
                        module=k
                        original_one=original_name
                        node.name.name=original_name
                        print(f"Found {name} in {module} with original name {original_one}")
                        break

        if name.startswith("__dace_") or  name in self.excepted_funcs or found_in_renames or found_in_names or name in self.funcs.iblocks:
            processed_args = []
            for i in node.args:
                arg = CallToArray(self.funcs).visit(i)
                processed_args.append(arg)
            node.args = processed_args
            return node
        indices = [CallToArray(self.funcs).visit(i) for i in node.args]
        # Array subscript cannot be empty.
        assert indices
        return ast_internal_classes.Array_Subscript_Node(name=node.name, type=node.type, indices=indices,
                                                         line_number=node.line_number)


class ArgumentExtractorNodeLister(NodeVisitor):
    """
    Finds all arguments in function calls in the AST node and its children that have to be extracted into independent expressions
    """

    def __init__(self):
        self.nodes: List[ast_internal_classes.Call_Expr_Node] = []

    def visit_For_Stmt_Node(self, node: ast_internal_classes.For_Stmt_Node):
        return

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        stop = False
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                stop = True

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if not stop and node.name.name not in [
            "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            for i in node.args:
                if isinstance(i, ast_internal_classes.Name_Node) or isinstance(i,
                                                                               ast_internal_classes.Literal) or isinstance(
                        i, ast_internal_classes.Array_Subscript_Node) or isinstance(i,
                                                                                    ast_internal_classes.Data_Ref_Node) or isinstance(
                        i, ast_internal_classes.Actual_Arg_Spec_Node):
                    continue
                else:
                    self.nodes.append(i)
        return self.generic_visit(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class ArgumentExtractor(NodeTransformer):
    """
    Uses the ArgumentExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """

    def __init__(self, program, count=0):
        self.count = count
        self.program = program

        ParentScopeAssigner().visit(program)
        self.scope_vars = ScopeVarsDeclarations(program)
        self.scope_vars.visit(program)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["malloc", "pow", "cbrt", "__dace_epsilon",
                              *FortranIntrinsics.call_extraction_exemptions()]:
            return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                return self.generic_visit(node)
        if not hasattr(self, "count"):
            self.count = 0
        tmp = self.count
        result = ast_internal_classes.Call_Expr_Node(type=node.type,
                                                     name=node.name,
                                                     args=[],
                                                     line_number=node.line_number)
        for i, arg in enumerate(node.args):
            # Ensure we allow to extract function calls from arguments
            if isinstance(arg, ast_internal_classes.Name_Node) or isinstance(arg,
                                                                             ast_internal_classes.Literal) or isinstance(
                    arg, ast_internal_classes.Array_Subscript_Node) or isinstance(arg,
                                                                                  ast_internal_classes.Data_Ref_Node) or isinstance(
                    arg, ast_internal_classes.Actual_Arg_Spec_Node):
                result.args.append(arg)
            else:
                result.args.append(ast_internal_classes.Name_Node(name="tmp_arg_" + str(tmp), type='VOID'))
                tmp = tmp + 1
        self.count = tmp
        return result

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = ArgumentExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            for i in res:
                if i == child:
                    res.pop(res.index(i))

            if res is not None:

                # Variables are counted from 0...end, starting from main node, to all calls nested
                # in main node arguments.
                # However, we need to define nested ones first.
                # We go in reverse order, counting from end-1 to 0.
                temp = self.count + len(res) - 1
                for i in reversed(range(0, len(res))):

                    if isinstance(res[i], ast_internal_classes.Data_Ref_Node):
                        struct_def, cur_var = self.program.structures.find_definition(self.scope_vars, res[i])

                        var_type = cur_var.type
                    else:
                        var_type = res[i].type

                    node.parent.specification_part.specifications.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Var_Decl_Node(
                                name="tmp_arg_" + str(temp),
                                type=var_type,
                                sizes=None,
                                init=None
                            )
                        ])
                    )
                    newbody.append(
                        ast_internal_classes.BinOp_Node(op="=",
                                                        lval=ast_internal_classes.Name_Node(name="tmp_arg_" +
                                                                                                 str(temp),
                                                                                            type=res[i].type),
                                                        rval=res[i],
                                                        line_number=child.line_number))
                    temp = temp - 1

            newbody.append(self.visit(child))

        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class FunctionCallTransformer(NodeTransformer):
    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
            if hasattr(node.rval, "subroutine"):
                if node.rval.subroutine is True:
                    return self.generic_visit(node)
            if node.rval.name.name.find("__dace_") != -1:
                return self.generic_visit(node)
            if node.op != "=":
                return self.generic_visit(node)
            args = node.rval.args
            lval = node.lval
            args.append(lval)
            return (ast_internal_classes.Call_Expr_Node(type=node.rval.type,
                                                        name=ast_internal_classes.Name_Node(
                                                            name=node.rval.name.name + "_srt", type=node.rval.type),
                                                        args=args,
                                                        subroutine=True,
                                                        line_number=node.line_number))

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


class FunctionToSubroutineDefiner(NodeTransformer):
    """
    Transforms all function definitions into subroutine definitions
    """

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):

        if node.ret != None:
            ret = node.ret

        found = False
        if node.specification_part is not None:
            for j in node.specification_part.specifications:

                for k in j.vardecl:
                    if node.ret != None:
                        if k.name == ret.name:
                            j.vardecl[j.vardecl.index(k)].name = node.name.name + "__ret"
                            found = True
                    if k.name == node.name.name:
                        j.vardecl[j.vardecl.index(k)].name = node.name.name + "__ret"
                        found = True
                        break

        if not found:

            var = ast_internal_classes.Var_Decl_Node(
                name=node.name.name + "__ret",
                type='VOID'
            )
            stmt_node = ast_internal_classes.Decl_Stmt_Node(vardecl=[var], line_number=node.line_number)

            if node.specification_part is not None:
                node.specification_part.specifications.append(stmt_node)
            else:
                node.specification_part = ast_internal_classes.Specification_Part_Node(
                    specifications=[stmt_node],
                    symbols=None,
                    interface_blocks=None,
                    uses=None,
                    typedecls=None,
                    enums=None
                )

        execution_part = NameReplacer(node.name.name, node.name.name + "__ret").visit(node.execution_part)
        if node.ret != None:
            execution_part = NameReplacer(ret.name.name, node.name.name + "__ret").visit(node.execution_part)
        args = node.args
        args.append(ast_internal_classes.Name_Node(name=node.name.name + "__ret", type=node.type))
        return ast_internal_classes.Subroutine_Subprogram_Node(
            name=ast_internal_classes.Name_Node(name=node.name.name + "_srt", type=node.type),
            args=args,
            specification_part=node.specification_part,
            execution_part=execution_part,
            subroutine=True,
            line_number=node.line_number,
            elemental=node.elemental)


class CallExtractorNodeLister(NodeVisitor):
    """
    Finds all function calls in the AST node and its children that have to be extracted into independent expressions
    """

    def __init__(self):
        self.nodes: List[ast_internal_classes.Call_Expr_Node] = []

    def visit_For_Stmt_Node(self, node: ast_internal_classes.For_Stmt_Node):
        self.generic_visit(node.init)
        self.generic_visit(node.cond)
        return

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        stop = False
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                stop = True

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if not stop and node.name.name not in [
            "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            self.nodes.append(node)
        return self.generic_visit(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class CallExtractor(NodeTransformer):
    """
    Uses the CallExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """

    def __init__(self, count=0):
        self.count = count

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["malloc", "pow", "cbrt", "__dace_epsilon",
                              *FortranIntrinsics.call_extraction_exemptions()]:
            return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                return self.generic_visit(node)
        if not hasattr(self, "count"):
            self.count = 0
        else:
            self.count = self.count + 1
        tmp = self.count

        for i, arg in enumerate(node.args):
            # Ensure we allow to extract function calls from arguments
            node.args[i] = self.visit(arg)

        return ast_internal_classes.Name_Node(name="tmp_call_" + str(tmp - 1))

    def visit_Specification_Part_Node(self, node: ast_internal_classes.Specification_Part_Node):
        newspec = []

        for i in node.specifications:
            if not isinstance(i, ast_internal_classes.Decl_Stmt_Node):
                newspec.append(self.visit(i))
            else:
                newdecl = []
                for var in i.vardecl:
                    lister = CallExtractorNodeLister()
                    lister.visit(var)
                    res = lister.nodes
                    for j in res:
                        if j == var:
                            res.pop(res.index(j))
                    if len(res) > 0:
                        temp = self.count + len(res) - 1
                        for ii in reversed(range(0, len(res))):
                            newdecl.append(
                                ast_internal_classes.Var_Decl_Node(
                                    name="tmp_call_" + str(temp),
                                    type=res[ii].type,
                                    sizes=None,
                                    line_number=var.line_number,
                                    init=res[ii],
                                )
                            )
                            newdecl.append(
                                ast_internal_classes.Var_Decl_Node(
                                    name="tmp_call_" + str(temp),
                                    type=res[ii].type,
                                    sizes=None,
                                    line_number=var.line_number,
                                    init=res[ii],
                                )
                            )
                            temp = temp - 1
                    newdecl.append(self.visit(var))
                newspec.append(ast_internal_classes.Decl_Stmt_Node(vardecl=newdecl))
        return ast_internal_classes.Specification_Part_Node(specifications=newspec, symbols=node.symbols,
                                                            typedecls=node.typedecls, uses=node.uses, enums=node.enums,
                                                            interface_blocks=node.interface_blocks)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = CallExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            for i in res:
                if i == child:
                    res.pop(res.index(i))
            if res is not None:
                # Variables are counted from 0...end, starting from main node, to all calls nested
                # in main node arguments.
                # However, we need to define nested ones first.
                # We go in reverse order, counting from end-1 to 0.
                temp = self.count + len(res) - 1
                for i in reversed(range(0, len(res))):
                    newbody.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Var_Decl_Node(
                                name="tmp_call_" + str(temp),
                                type=res[i].type,
                                sizes=None,
                                init=None
                            )
                        ]))
                    newbody.append(
                        ast_internal_classes.BinOp_Node(op="=",
                                                        lval=ast_internal_classes.Name_Node(name="tmp_call_" +
                                                                                                 str(temp),
                                                                                            type=res[i].type),
                                                        rval=res[i],
                                                        line_number=child.line_number))
                    temp = temp - 1
            if isinstance(child, ast_internal_classes.Call_Expr_Node):
                new_args = []
                if hasattr(child, "args"):
                    for i in child.args:
                        new_args.append(self.visit(i))
                new_child = ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                name=child.name,
                                                                args=new_args,
                                                                line_number=child.line_number)
                newbody.append(new_child)
            else:
                newbody.append(self.visit(child))

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
            ast_internal_classes.Subroutine_Subprogram_Node,
            ast_internal_classes.Function_Subprogram_Node,
            ast_internal_classes.Main_Program_Node,
            ast_internal_classes.Module_Node
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

                newer_indices.append(ast_internal_classes.Name_Node(name="tmp_index_" + str(tmp)))
                self.replacements["tmp_index_" + str(tmp)] = (i, node.name.name)
                tmp = tmp + 1
        self.count = tmp

        return ast_internal_classes.Array_Subscript_Node(name=node.name, type=node.type, indices=newer_indices,
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
                        else:
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
                                    struct, variable = self.structures.find_definition(
                                        self.scope_vars, parent_node, j.name
                                    )
                                    var_name = j.name.name
                                else:
                                    var_name = j.name.name
                                    variable = self.scope_vars.get_var(child.parent, var_name)
                                offset = variable.offsets[idx]

                                # it can be a symbol - Name_Node - or a value
                                if not isinstance(offset, ast_internal_classes.Name_Node):
                                    offset = ast_internal_classes.Int_Literal_Node(value=str(offset))
                                newbody.append(
                                    ast_internal_classes.BinOp_Node(
                                        op="=",
                                        lval=ast_internal_classes.Name_Node(name=tmp_name),
                                        rval=ast_internal_classes.BinOp_Node(
                                            op="-",
                                            lval=self.replacements[tmp_name][0],
                                            rval=offset,
                                            line_number=child.line_number),
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
                                            line_number=child.line_number),
                                        line_number=child.line_number))
            newbody.append(tmp_child)
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class SignToIf(NodeTransformer):
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
                                                   line_number=node.line_number)
            body_if = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.Call_Expr_Node(
                                                    name=ast_internal_classes.Name_Node(name="abs"),
                                                    type="DOUBLE",
                                                    args=[copy.deepcopy(args[0])],
                                                    line_number=node.line_number),
                                                line_number=node.line_number)
            ])
            body_else = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.UnOp_Node(
                                                    op="-",
                                                    type="VOID",
                                                    lval=ast_internal_classes.Call_Expr_Node(
                                                        name=ast_internal_classes.Name_Node(name="abs"),
                                                        type="DOUBLE",
                                                        args=[copy.deepcopy(args[0])],
                                                        line_number=node.line_number),
                                                    line_number=node.line_number),
                                                line_number=node.line_number)
            ])
            return (ast_internal_classes.If_Stmt_Node(cond=cond,
                                                      body=body_if,
                                                      body_else=body_else,
                                                      line_number=node.line_number))

        else:
            return self.generic_visit(node)


class RenameArguments(NodeTransformer):
    """
    Renames all arguments of a function to the names of the arguments of the function call
    Used when eliminating function statements
    """

    def __init__(self, node_args: list, call_args: list):
        self.node_args = node_args
        self.call_args = call_args

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        for i, j in zip(self.node_args, self.call_args):
            if node.name == j.name:
                return copy.deepcopy(i)
        return node


class ReplaceFunctionStatement(NodeTransformer):
    """
    Replaces a function statement with its content, similar to propagating a macro
    """

    def __init__(self, statement, replacement):
        self.name = statement.name
        self.content = replacement

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if node.name == self.name:
            return ast_internal_classes.Parenthesis_Expr_Node(expr=copy.deepcopy(self.content))
        else:
            return self.generic_visit(node)


class ReplaceFunctionStatementPass(NodeTransformer):
    """
    Replaces a function statement with its content, similar to propagating a macro
    """

    def __init__(self, statefunc: list):
        self.funcs = statefunc

    def visit_Structure_Constructor_Node(self, node: ast_internal_classes.Structure_Constructor_Node):
        for i in self.funcs:
            if node.name.name == i[0].name.name:
                ret_node = copy.deepcopy(i[1])
                ret_node = RenameArguments(node.args, i[0].args).visit(ret_node)
                return ast_internal_classes.Parenthesis_Expr_Node(expr=ret_node)
        return self.generic_visit(node)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        for i in self.funcs:
            if node.name.name == i[0].name.name:
                ret_node = copy.deepcopy(i[1])
                ret_node = RenameArguments(node.args, i[0].args).visit(ret_node)
                return ast_internal_classes.Parenthesis_Expr_Node(expr=ret_node)
        return self.generic_visit(node)


def optionalArgsHandleFunction(func):
    func.optional_args = []
    if func.specification_part is None:
        return 0
    for spec in func.specification_part.specifications:
        for var in spec.vardecl:
            if hasattr(var, "optional") and var.optional:
                func.optional_args.append((var.name, var.type))

    vardecls = []
    new_args = []
    for i in func.args:
        new_args.append(i)
    for arg in func.args:

        found = False
        for opt_arg in func.optional_args:
            if opt_arg[0] == arg.name:
                found = True
                break

        if found:

            name = f'__f2dace_OPTIONAL_{arg.name}'
            already_there = False
            for i in func.args:
                if hasattr(i, "name") and i.name == name:
                    already_there = True
                    break
            if not already_there:
                var = ast_internal_classes.Var_Decl_Node(name=name,
                                                         type='BOOL',
                                                         alloc=False,
                                                         sizes=None,
                                                         offsets=None,
                                                         kind=None,
                                                         optional=False,
                                                         init=None,
                                                         line_number=func.line_number)
                new_args.append(ast_internal_classes.Name_Node(name=name))
                vardecls.append(var)

    if len(new_args) > len(func.args):
        func.args.clear()
        func.args = new_args

    if len(vardecls) > 0:
        specifiers = []
        for i in func.specification_part.specifications:
            specifiers.append(i)
        specifiers.append(
            ast_internal_classes.Decl_Stmt_Node(
                vardecl=vardecls,
                line_number=func.line_number
            )
        )
        func.specification_part.specifications.clear()
        func.specification_part.specifications = specifiers

    return len(new_args)


class OptionalArgsTransformer(NodeTransformer):
    def __init__(self, funcs_with_opt_args):
        self.funcs_with_opt_args = funcs_with_opt_args

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        if node.name.name not in self.funcs_with_opt_args:
            return node

        # Basic assumption for positioanl arguments
        # Optional arguments follow the mandatory ones
        # We use that to determine which optional arguments are missing
        func_decl = self.funcs_with_opt_args[node.name.name]
        optional_args = len(func_decl.optional_args)
        if optional_args == 0:
            return node

        should_be_args = len(func_decl.args)
        mandatory_args = should_be_args - optional_args * 2

        present_args = len(node.args)

        # Remove the deduplicated variable entries acting as flags for optional args
        missing_args_count = should_be_args - present_args
        present_optional_args = present_args - mandatory_args
        new_args = [None] * should_be_args
        print("Func len args: ", len(func_decl.args))
        print("Func: ", func_decl.name.name, "Mandatory: ", mandatory_args, "Optional: ", optional_args, "Present: ",
              present_args, "Missing: ", missing_args_count, "Present Optional: ", present_optional_args)
        print("List: ", node.name.name, len(new_args), mandatory_args)

        if missing_args_count == 0:
            return node

        for i in range(mandatory_args):
            new_args[i] = node.args[i]
        for i in range(mandatory_args, len(node.args)):
            if len(node.args) > i:
                current_arg = node.args[i]
                if not isinstance(current_arg, ast_internal_classes.Actual_Arg_Spec_Node):
                    new_args[i] = current_arg
                else:
                    name = current_arg.arg_name
                    index = 0
                    for j in func_decl.optional_args:
                        if j[0] == name.name:
                            break
                        index = index + 1
                    new_args[mandatory_args + index] = current_arg.arg

        for i in range(mandatory_args, mandatory_args + optional_args):
            relative_position = i - mandatory_args
            if new_args[i] is None:
                dtype = func_decl.optional_args[relative_position][1]
                if dtype == 'INTEGER':
                    new_args[i] = ast_internal_classes.Int_Literal_Node(value='0')
                elif dtype == 'BOOL':
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

        node.args = new_args
        return node


def optionalArgsExpander(node=ast_internal_classes.Program_Node):
    """
    Adds to each optional arg a logical value specifying its status.
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    :note Should only be used on the program node
    """

    modified_functions = {}

    for func in node.subroutine_definitions:
        if optionalArgsHandleFunction(func):
            modified_functions[func.name.name] = func
    for mod in node.modules:
        for func in mod.subroutine_definitions:
            if optionalArgsHandleFunction(func):
                modified_functions[func.name.name] = func

    node = OptionalArgsTransformer(modified_functions).visit(node)

    return node


def functionStatementEliminator(node=ast_internal_classes.Program_Node):
    """
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    :note Should only be used on the program node
    """
    main_program = localFunctionStatementEliminator(node.main_program)
    function_definitions = [localFunctionStatementEliminator(i) for i in node.function_definitions]
    subroutine_definitions = [localFunctionStatementEliminator(i) for i in node.subroutine_definitions]
    modules = []
    for i in node.modules:
        module_function_definitions = [localFunctionStatementEliminator(j) for j in i.function_definitions]
        module_subroutine_definitions = [localFunctionStatementEliminator(j) for j in i.subroutine_definitions]
        modules.append(
            ast_internal_classes.Module_Node(
                name=i.name,
                specification_part=i.specification_part,
                subroutine_definitions=module_subroutine_definitions,
                function_definitions=module_function_definitions,
                interface_blocks=i.interface_blocks,
            ))
    node.main_program = main_program
    node.function_definitions = function_definitions
    node.subroutine_definitions = subroutine_definitions
    node.modules = modules
    return node


def localFunctionStatementEliminator(node: ast_internal_classes.FNode):
    """
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    """
    if node is None:
        return None
    if hasattr(node, "specification_part") and node.specification_part is not None:
        spec = node.specification_part.specifications
    else:
        spec = []
    if hasattr(node, "execution_part"):
        if node.execution_part is not None:
            exec = node.execution_part.execution
        else:
            exec = []
    else:
        exec = []
    new_exec = exec.copy()
    to_change = []
    for i in exec:
        if isinstance(i, ast_internal_classes.BinOp_Node):
            if i.op == "=":
                if isinstance(i.lval, ast_internal_classes.Call_Expr_Node) or isinstance(
                        i.lval, ast_internal_classes.Structure_Constructor_Node):
                    function_statement_name = i.lval.name
                    is_actually_function_statement = False
                    # In Fortran, function statement are defined as scalar values,
                    # but called as arrays, so by identifiying that it is called as
                    # a call_expr or structure_constructor, we also need to match
                    # the specification part and see that it is scalar rather than an array.
                    found = False
                    for j in spec:
                        if found:
                            break
                        for k in j.vardecl:
                            if k.name == function_statement_name.name:
                                if k.sizes is None:
                                    is_actually_function_statement = True
                                    function_statement_type = k.type
                                    j.vardecl.remove(k)
                                    found = True
                                    break
                    if is_actually_function_statement:
                        to_change.append([i.lval, i.rval])
                        new_exec.remove(i)

                    else:
                        # There are no function statements after the first one that isn't a function statement
                        break
    still_changing = True
    while still_changing:
        still_changing = False
        for i in to_change:
            rval = i[1]
            calls = FindFunctionCalls()
            calls.visit(rval)
            for j in to_change:
                for k in calls.nodes:
                    if k.name == j[0].name:
                        calls_to_replace = FindFunctionCalls()
                        calls_to_replace.visit(j[1])
                        # must check if it is recursive and contains other function statements
                        it_is_simple = True
                        for l in calls_to_replace.nodes:
                            for m in to_change:
                                if l.name == m[0].name:
                                    it_is_simple = False
                        if it_is_simple:
                            still_changing = True
                            i[1] = ReplaceFunctionStatement(j[0], j[1]).visit(rval)
    final_exec = []
    for i in new_exec:
        final_exec.append(ReplaceFunctionStatementPass(to_change).visit(i))
    if hasattr(node, "execution_part"):
        if node.execution_part is not None:
            node.execution_part.execution = final_exec
        else:
            node.execution_part = ast_internal_classes.Execution_Part_Node(execution=final_exec)
    else:
        node.execution_part = ast_internal_classes.Execution_Part_Node(execution=final_exec)
        # node.execution_part.execution = final_exec
    if hasattr(node, "specification_part"):
        if node.specification_part is not None:
            node.specification_part.specifications = spec
    # node.specification_part.specifications = spec
    return node


class ArrayLoopNodeLister(NodeVisitor):
    """
    Finds all array operations that have to be transformed to loops in the AST
    """

    def __init__(self):
        self.nodes: List[ast_internal_classes.FNode] = []
        self.range_nodes: List[ast_internal_classes.FNode] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        rval_pardecls = [i for i in mywalk(node.rval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        lval_pardecls = [i for i in mywalk(node.lval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        if not lval_pardecls:
            return
        if rval_pardecls:
            self.range_nodes.append(node)
        self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


def par_Decl_Range_Finder(node: ast_internal_classes.Array_Subscript_Node,
                          ranges: list,
                          rangepos: list,
                          rangeslen: list,
                          count: int,
                          newbody: list,
                          scope_vars: ScopeVarsDeclarations,
                          structures: Structures,
                          declaration=True,
                          main_iterator_ranges: Optional[list] = None
):
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
                offsets = struct_def.vars[cur_node.name].offsets
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
                    if isinstance(offsets[idx], ast_internal_classes.Name_Node):
                        lower_boundary = offsets[idx]
                    else:
                        lower_boundary = ast_internal_classes.Int_Literal_Node(value=str(offsets[idx]))
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
                                                                          name=array_name, type="VOID",
                                                                          line_number=node.line_number),
                                                                      pos=currentindex)
                """
                    When there's an offset, we add MAX_RANGE + offset.
                    But since the generated loop has `<=` condition, we need to subtract 1.
                """
                if offsets[idx] != 1:

                    # support symbols and integer literals
                    if isinstance(offsets[idx], ast_internal_classes.Name_Node):
                        offset = offsets[idx]
                    else:
                        offset = ast_internal_classes.Int_Literal_Node(value=str(offsets[idx]))

                    upper_boundary = ast_internal_classes.BinOp_Node(
                        lval=upper_boundary,
                        op="+",
                        rval=offset
                    )
                    upper_boundary = ast_internal_classes.BinOp_Node(
                        lval=upper_boundary,
                        op="-",
                        rval=ast_internal_classes.Int_Literal_Node(value="1")
                    )

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
                    add = ast_internal_classes.BinOp_Node(
                        lval=start,
                        op="+",
                        rval=ast_internal_classes.Int_Literal_Node(value="1")
                    )
                    substr = ast_internal_classes.BinOp_Node(
                        lval=end,
                        op="-",
                        rval=add
                    )
                    rangeslen.append(substr)

            rangepos.append(currentindex)
            if declaration:
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(
                            name="tmp_parfor_" + str(count + len(rangepos) - 1), type="INTEGER", sizes=None, init=None)
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

                indices.append(
                    ast_internal_classes.Name_Node(name="tmp_parfor_" + str(count + len(rangepos) - 1))
                )
            else:

                """
                    For RHS, we adjust starting array position by taking consideration the initial value
                    of the loop iterator.

                    Offset is handled by always subtracting the lower boundary.
                """
                current_lower_boundary = main_iterator_ranges[currentindex][0]

                indices.append(
                    ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(count + len(rangepos) - 1)),
                        op="+",
                        rval = ast_internal_classes.BinOp_Node(
                            lval=lower_boundary,
                            op="-",
                            rval=current_lower_boundary
                        )
                    )
                )
        else:
            indices.append(i)
        currentindex += 1

    node.indices = indices


class ArrayToLoop(NodeTransformer):
    """
    Transforms the AST by removing array expressions and replacing them with loops
    """

    def __init__(self, ast):
        self.count = 0

        self.ast = ast
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)


    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            lister = ArrayLoopNodeLister()
            lister.visit(child)
            res = lister.nodes
            res_range = lister.range_nodes
            if res is not None and len(res) > 0:

                current = child.lval
                val = child.rval
                ranges = []
                rangepos = []
                par_Decl_Range_Finder(current, ranges, rangepos, [], self.count, newbody, self.scope_vars,
                                      self.ast.structures, True)

                if res_range is not None and len(res_range) > 0:
                    rvals = [i for i in mywalk(val) if isinstance(i, ast_internal_classes.Array_Subscript_Node)]
                    for i in rvals:
                        rangeposrval = []
                        rangesrval = []

                        par_Decl_Range_Finder(i, rangesrval, rangeposrval, [], self.count, newbody, self.scope_vars,
                                              self.ast.structures, False, ranges)

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
                                                #raise NotImplementedError("Ranges must be the same")
                                                continue
                                else:
                                    raise NotImplementedError("Ranges must be identical")

                range_index = 0
                body = ast_internal_classes.BinOp_Node(lval=current, op="=", rval=val, line_number=child.line_number)
                for i in ranges:
                    initrange = i[0]
                    finalrange = i[1]
                    init = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="=",
                        rval=initrange,
                        line_number=child.line_number)
                    cond = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="<=",
                        rval=finalrange,
                        line_number=child.line_number)
                    iter = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="=",
                        rval=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                            op="+",
                            rval=ast_internal_classes.Int_Literal_Node(value="1")),
                        line_number=child.line_number)
                    current_for = ast_internal_classes.Map_Stmt_Node(
                        init=init,
                        cond=cond,
                        iter=iter,
                        body=ast_internal_classes.Execution_Part_Node(execution=[body]),
                        line_number=child.line_number)
                    body = current_for
                    range_index += 1

                newbody.append(body)

                self.count = self.count + range_index
            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


def mywalk(node):
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
        yield node


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
        return ast_internal_classes.Symbol_Decl_Node(name=node.name.replace(self.oldname, self.newname), type=node.type,
                                                     sizes=node.sizes, init=node.init, line_number=node.line_number,
                                                     kind=node.kind, alloc=node.alloc, offsets=node.offsets)


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
                                                               line_number=child.line_number)
                newfor = RenameVar(child.init.lval.name, "_for_it_" + str(self.count)).visit(child)
                self.count += 1
                newfor = self.visit(newfor)
                newbody.append(newfor)

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ElementalFunctionExpander(NodeTransformer):
    "Makes elemental functions into normal functions by creating a loop around thme if they are called with arrays"

    def __init__(self, func_list: list):
        self.func_list = func_list
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Call_Expr_Node):
                arrays = False
                for i in self.func_list:
                    if child.name.name == i.name or child.name.name == i.name + "_srt":
                        if hasattr(i, "elemental"):
                            if i.elemental is True:
                                if len(child.args) > 0:
                                    for j in child.args:
                                        # THIS Needs a proper check
                                        if j.name == "z":
                                            arrays = True

                if not arrays:
                    newbody.append(self.visit(child))
                else:
                    newbody.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Symbol_Decl_Node(
                                name="_for_elem_it_" + str(self.count), type="INTEGER", sizes=None, init=None)
                        ]))
                    newargs = []
                    # The range must be determined! It's currently hard set to 10
                    shape = ["10"]
                    for i in child.args:
                        if isinstance(i, ast_internal_classes.Name_Node):
                            newargs.append(ast_internal_classes.Array_Subscript_Node(name=i, indices=[
                                ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count))],
                                                                                     line_number=child.line_number,
                                                                                     type=i.type))
                            if i.name.startswith("tmp_call_"):
                                for j in newbody:
                                    if isinstance(j, ast_internal_classes.Decl_Stmt_Node):
                                        if j.vardecl[0].name == i.name:
                                            newbody[newbody.index(j)].vardecl[0].sizes = shape
                                            break
                        else:
                            raise NotImplementedError("Only name nodes are supported")

                    newbody.append(ast_internal_classes.For_Stmt_Node(
                        init=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                            op="=",
                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                            line_number=child.line_number),
                        cond=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                            op="<=",
                            rval=ast_internal_classes.Int_Literal_Node(value=shape[0]),
                            line_number=child.line_number),
                        body=ast_internal_classes.Execution_Part_Node(execution=[
                            ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                name=child.name,
                                                                args=newargs,
                                                                line_number=child.line_number)
                        ]), line_number=child.line_number,
                        iter=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                            op="=",
                            rval=ast_internal_classes.BinOp_Node(
                                lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                op="+",
                                rval=ast_internal_classes.Int_Literal_Node(value="1")),
                            line_number=child.line_number)
                    ))
                    self.count += 1


            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class TypeInference(NodeTransformer):
    """
    """

    def __init__(self, ast, assert_voids=True):
        self.assert_voids = assert_voids

        self.ast = ast
        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations(ast)
        self.scope_vars.visit(ast)
        self.structures = ast.structures

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):

        if not hasattr(node, 'type') or node.type == 'VOID' or not hasattr(node, 'dims'):
            try:
                var_def = self.scope_vars.get_var(node.parent, node.name)
                node.type = var_def.type
                node.dims = len(var_def.sizes) if hasattr(var_def, 'sizes') and var_def.sizes is not None else 1
            except Exception as e:
                print(f"Ignore type inference for {node.name}")
                print(e)

        return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        var_def = self.scope_vars.get_var(node.parent, node.name.name)
        node.type = var_def.type
        node.dims = len(var_def.sizes) if var_def.sizes is not None else 1
        return node

    def visit_Parenthesis_Expr_Node(self, node: ast_internal_classes.Parenthesis_Expr_Node):

        self.visit(node.expr)
        node.type = node.expr.type
        if hasattr(node.expr, 'dims'):
            node.dims = node.expr.dims
        return node

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):

        """
            Simple implementation of type promotion in binary ops.
        """

        self.visit(node.lval)
        self.visit(node.rval)

        type_hierarchy = [
            'VOID',
            'BOOL',
            'INTEGER',
            'REAL',
            'DOUBLE'
        ]

        idx_left = type_hierarchy.index(self._get_type(node.lval))
        idx_right = type_hierarchy.index(self._get_type(node.rval))
        idx_void = type_hierarchy.index('VOID')

        # if self.assert_voids:
        #    assert idx_left != idx_void or idx_right != idx_void
        #    #assert self._get_dims(node.lval) == self._get_dims(node.rval)

        node.type = type_hierarchy[max(idx_left, idx_right)]
        if hasattr(node.lval, "dims"):
            node.dims = self._get_dims(node.lval)
        elif hasattr(node.lval, "dims"):
            node.dims = self._get_dims(node.rval)

        if node.op == '=' and idx_left == idx_void and idx_left != idx_void:
            lval_definition = self.scope_vars.get_var(node.parent, node.lval.name)
            lval_definition.type = node.type
            lval_definition.dims = node.dims
            node.lval.type = node.type
            node.lval.dims = node.dims

        return node

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):

        if node.type != 'VOID':
            return node

        struct, variable = self.structures.find_definition(
            self.scope_vars, node
        )
        node.type = variable.type
        node.dims = len(variable.sizes) if variable.sizes is not None else 1
        return node

    def visit_Actual_Arg_Spec_Node(self, node: ast_internal_classes.Actual_Arg_Spec_Node):

        if node.type != 'VOID':
            return node

        self.visit(node.arg)

        func_arg_name_type = self._get_type(node.arg)
        if func_arg_name_type == 'VOID':

            func_arg = self.scope_vars.get_var(node.parent, node.arg.name)
            node.type = func_arg.type
            node.arg.type = func_arg.type
            dims = len(func_arg.sizes) if func_arg.sizes is not None else 1
            node.dims = dims
            node.arg.dims = dims

        else:
            node.type = func_arg_name_type
            node.dims = self._get_dims(node.arg)

        return node

    def _get_type(self, node):

        if isinstance(node, ast_internal_classes.Int_Literal_Node):
            return 'INTEGER'
        elif isinstance(node, ast_internal_classes.Real_Literal_Node):
            return 'REAL'
        elif isinstance(node, ast_internal_classes.Bool_Literal_Node):
            return 'BOOL'
        else:
            return node.type

    def _get_dims(self, node):

        if isinstance(node, ast_internal_classes.Int_Literal_Node):
            return 1
        elif isinstance(node, ast_internal_classes.Real_Literal_Node):
            return 1
        elif isinstance(node, ast_internal_classes.Bool_Literal_Node):
            return 1
        else:
            return node.dims


class ReplaceInterfaceBlocks(NodeTransformer):
    """
    """

    def __init__(self, program, funcs: FindFunctionAndSubroutines):
        self.funcs = funcs

        ParentScopeAssigner().visit(program)
        self.scope_vars = ScopeVarsDeclarations(program)
        self.scope_vars.visit(program)

    def _get_dims(self, node):

        if hasattr(node, "dims"):
            return node.dims

        if isinstance(node, ast_internal_classes.Var_Decl_Node):
            return len(node.sizes) if node.sizes is not None else 1

        raise RuntimeError()

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        # is_func = node.name.name in self.excepted_funcs or node.name in self.funcs.names
        # is_interface_func = not node.name in self.funcs.names and node.name.name in self.funcs.iblocks
        is_interface_func = node.name.name in self.funcs.iblocks

        if is_interface_func:

            available_names = []
            print("Invoke", node.name.name, available_names)
            for name in self.funcs.iblocks[node.name.name]:

                # non_optional_args = len(self.funcs.nodes[name].args) - self.funcs.nodes[name].optional_args_count
                non_optional_args = self.funcs.nodes[name].mandatory_args_count
                print("Check", name, non_optional_args, self.funcs.nodes[name].optional_args_count)

                success = True
                for call_arg, func_arg in zip(node.args[0:non_optional_args],
                                              self.funcs.nodes[name].args[0:non_optional_args]):
                    print("Mandatory arg", call_arg, type(call_arg))
                    if call_arg.type != func_arg.type or self._get_dims(call_arg) != self._get_dims(func_arg):
                        print(f"Ignore function {name}, wrong param type {call_arg.type} instead of {func_arg.type}")
                        success = False
                        break
                    else:
                        print(self._get_dims(call_arg), self._get_dims(func_arg), type(call_arg), call_arg.type,
                              func_arg.name, type(func_arg), func_arg.type)

                optional_args = self.funcs.nodes[name].args[non_optional_args:]
                pos = non_optional_args
                for idx, call_arg in enumerate(node.args[non_optional_args:]):

                    print("Optional arg", call_arg, type(call_arg))
                    if isinstance(call_arg, ast_internal_classes.Actual_Arg_Spec_Node):
                        func_arg_name = call_arg.arg_name
                        try:
                            func_arg = self.scope_vars.get_var(name, func_arg_name.name)
                        except:
                            # this keyword parameter is not available in this function
                            success = False
                            break
                        print('btw', func_arg, type(func_arg), func_arg.type)
                    else:
                        func_arg = optional_args[idx]

                    # if call_arg.type != func_arg.type:
                    if call_arg.type != func_arg.type or self._get_dims(call_arg) != self._get_dims(func_arg):
                        print(f"Ignore function {name}, wrong param type {call_arg.type} instead of {func_arg.type}")
                        success = False
                        break
                    else:
                        print(self._get_dims(call_arg), self._get_dims(func_arg), type(call_arg), call_arg.type,
                              func_arg.name, type(func_arg), func_arg.type)

                if success:
                    available_names.append(name)

            if len(available_names) == 0:
                raise RuntimeError("No matching function calls!")

            if len(available_names) != 1:
                print(node.name.name, available_names)
                raise RuntimeError("Too many matching function calls!")

            print(f"Selected {available_names[0]} as invocation for {node.name}")
            node.name = ast_internal_classes.Name_Node(name=available_names[0])

        return node


class PointerRemoval(NodeTransformer):

    def __init__(self):
        self.nodes = {}

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        if node.name.name in self.nodes:
            original_ref_node = self.nodes[node.name.name]

            cur_ref_node = original_ref_node
            new_ref_node = ast_internal_classes.Data_Ref_Node(
                parent_ref=cur_ref_node.parent_ref,
                part_ref=None
            )
            newer_ref_node = new_ref_node

            while isinstance(cur_ref_node.part_ref, ast_internal_classes.Data_Ref_Node):
                cur_ref_node = cur_ref_node.part_ref
                newest_ref_node = ast_internal_classes.Data_Ref_Node(
                    parent_ref=cur_ref_node.parent_ref,
                    part_ref=None
                )
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

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):

        if node.execution_part is not None:
            execution_part = self.visit(node.execution_part)
        else:
            execution_part = node.execution_part

        if node.specification_part is not None:
            specification_part = self.visit(node.specification_part)
        else:
            specification_part = node.specification_part

        return ast_internal_classes.Subroutine_Subprogram_Node(
            name=node.name,
            args=node.args,
            specification_part=specification_part,
            execution_part=execution_part,
            line_number=node.line_number
        )

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

                        for symbol in var_decl.sizes:
                            symbols_to_remove.add(symbol.name)
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

        return ast_internal_classes.Specification_Part_Node(
            specifications=newspec,
            symbols=new_symbols,
            typedecls=node.typedecls,
            uses=node.uses,
            enums=node.enums
        )


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


class PropagateEnums(NodeTransformer):
    """
    """

    def __init__(self):
        self.parsed_enums = {}

    def _parse_enums(self, enums):

        for j in enums:
            running_count = 0
            for k in j:
                if isinstance(k, list):
                    for l in k:
                        if isinstance(l, ast_internal_classes.Name_Node):
                            self.parsed_enums[l.name] = running_count
                            running_count += 1
                        elif isinstance(l, list):
                            self.parsed_enums[l[0].name] = l[2].value
                            running_count = int(l[2].value) + 1
                        else:

                            raise ValueError("Unknown enum type")
                else:
                    raise ValueError("Unknown enum type")

    def visit_Specification_Part_Node(self, node: ast_internal_classes.Specification_Part_Node):
        self._parse_enums(node.enums)
        return self.generic_visit(node)

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):

        if self.parsed_enums.get(node.name) is not None:
            node.type = 'INTEGER'
            return ast_internal_classes.Int_Literal_Node(value=str(self.parsed_enums[node.name]))

        return node


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
            print("Failed: " + text)
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
        return ast_internal_classes.If_Stmt_Node(line_number=node.line_number, cond=test.cond, body=test.body,
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
                        elif isinstance(old_value, ast_internal_classes.Name_Node) and isinstance(i[0],ast_internal_classes.Name_Node):
                            if old_value.name == i[0].name:
                                old_value = i[1]
                                self.replacements += 1
                                break
                        elif isinstance(old_value,ast_internal_classes.Data_Ref_Node) and isinstance(i[0], ast_internal_classes.Data_Ref_Node):
                            if isinstance(old_value.part_ref,ast_internal_classes.Name_Node) and isinstance(i[0].part_ref,ast_internal_classes.Name_Node) and isinstance(old_value.parent_ref,ast_internal_classes.Name_Node) and isinstance(i[0].parent_ref,ast_internal_classes.Name_Node):
                                if old_value.part_ref.name == i[0].part_ref.name and old_value.parent_ref.name == i[0].parent_ref.name:
                                    old_value = i[1]
                                    self.replacements += 1
                                    break

                    new_node = self.visit(old_value)

                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class getCalls(NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call_Expr_Node(self, node):
        self.calls.append(node.name.name)
        for arg in node.args:
            self.visit(arg)
        return


class FindUnusedFunctions(NodeVisitor):
    def __init__(self, root, parse_order):
        self.root = root
        self.parse_order = parse_order
        self.used_names = {}

    def visit_Subroutine_Subprogram_Node(self, node):
        getacall = getCalls()
        getacall.visit(node.execution_part)
        used_calls = getacall.calls
        self.used_names[node.name.name] = used_calls
        return
