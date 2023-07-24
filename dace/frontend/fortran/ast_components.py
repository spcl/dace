# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from fparser.two.Fortran2008 import Fortran2008 as f08
from fparser.two import Fortran2008
from fparser.two import Fortran2003 as f03
from fparser.two import symbol_table

import copy
from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_internal_classes import FNode, Name_Node
from typing import Any, List, Tuple, Type, TypeVar, Union, overload

#We rely on fparser to provide an initial AST and convert to a version that is more suitable for our purposes

# The following class is used to translate the fparser AST to our own AST of Fortran
# the supported_fortran_types dictionary is used to determine which types are supported by our compiler
# for each entry in the dictionary, the key is the name of the class in the fparser AST and the value is the name of the function that will be used to translate the fparser AST to our AST
# the functions return an object of the class that is the name of the key in the dictionary with _Node appended to it to ensure it is diferentietated from the fparser AST
FASTNode = Any
T = TypeVar('T')


@overload
def get_child(node: Union[FASTNode, List[FASTNode]], child_type: str) -> FASTNode:
    ...


@overload
def get_child(node: Union[FASTNode, List[FASTNode]], child_type: Type[T]) -> T:
    ...


def get_child(node: Union[FASTNode, List[FASTNode]], child_type: Union[str, Type[T], List[Type[T]]]):
    if isinstance(node, list):
        children = node
    else:
        children = node.children

    if not isinstance(child_type, str) and not isinstance(child_type, list):
        child_type = child_type.__name__
        children_of_type = list(filter(lambda child: child.__class__.__name__ == child_type, children))

    elif isinstance(child_type, list):
        if all(isinstance(i, str) for i in child_type):
            child_types = [i for i in child_type]
        else:
            child_types = [i.__name__ for i in child_type]
        children_of_type = list(filter(lambda child: child.__class__.__name__ in child_types, children))

    if len(children_of_type) == 1:
        return children_of_type[0]
    raise ValueError('Expected only one child of type {} but found {}'.format(child_type, children_of_type))


@overload
def get_children(node: Union[FASTNode, List[FASTNode]], child_type: str) -> List[FASTNode]:
    ...


@overload
def get_children(node: Union[FASTNode, List[FASTNode]], child_type: Type[T]) -> List[T]:
    ...


def get_children(node: Union[FASTNode, List[FASTNode]], child_type: Union[str, Type[T], List[Type[T]]]):
    if isinstance(node, list):
        children = node
    else:
        children = node.children

    if not isinstance(child_type, str) and not isinstance(child_type, list):
        child_type = child_type.__name__
        children_of_type = list(filter(lambda child: child.__class__.__name__ == child_type, children))

    elif isinstance(child_type, list):
        child_types = [i.__name__ for i in child_type]
        children_of_type = list(filter(lambda child: child.__class__.__name__ in child_types, children))

    elif isinstance(child_type, str):
        children_of_type = list(filter(lambda child: child.__class__.__name__ == child_type, children))

    return children_of_type


def get_line(node: FASTNode):
    line = None
    if node.item is not None and hasattr(node.item, "span"):
        line = node.item.span
    else:
        tmp = node
        while tmp.parent is not None:
            tmp = tmp.parent
            if tmp.item is not None and hasattr(tmp.item, "span"):
                line = tmp.item.span
                break
    return line


class InternalFortranAst:
    """
    This class is used to translate the fparser AST to our own AST of Fortran
    the supported_fortran_types dictionary is used to determine which types are supported by our compiler
    for each entry in the dictionary, the key is the name of the class in the fparser AST and the value
    is the name of the function that will be used to translate the fparser AST to our AST
    """
    def __init__(self, ast: f03.Program, tables: symbol_table.SymbolTables):
        """
        Initialization of the AST converter
        :param ast: the fparser AST
        :param tables: the symbol table of the fparser AST

        """
        self.ast = ast
        self.tables = tables
        self.functions_and_subroutines = []
        self.symbols = {}
        self.types = {
            "LOGICAL": "BOOL",
            "CHARACTER": "CHAR",
            "INTEGER": "INTEGER",
            "INTEGER4": "INTEGER",
            "REAL4": "REAL",
            "REAL8": "DOUBLE",
            "DOUBLE PRECISION": "DOUBLE",
            "REAL": "REAL",
        }
        self.supported_fortran_syntax = {
            "str": self.str_node,
            "tuple": self.tuple_node,
            "Program": self.program,
            "Main_Program": self.main_program,
            "Program_Stmt": self.program_stmt,
            "End_Program_Stmt": self.end_program_stmt,
            "Subroutine_Subprogram": self.subroutine_subprogram,
            "Function_Subprogram": self.function_subprogram,
            "Subroutine_Stmt": self.subroutine_stmt,
            "Function_Stmt": self.function_stmt,
            "End_Subroutine_Stmt": self.end_subroutine_stmt,
            "End_Function_Stmt": self.end_function_stmt,
            "Module": self.module,
            "Module_Stmt": self.module_stmt,
            "End_Module_Stmt": self.end_module_stmt,
            "Use_Stmt": self.use_stmt,
            "Implicit_Part": self.implicit_part,
            "Implicit_Stmt": self.implicit_stmt,
            "Implicit_None_Stmt": self.implicit_none_stmt,
            "Implicit_Part_Stmt": self.implicit_part_stmt,
            "Declaration_Construct": self.declaration_construct,
            "Declaration_Type_Spec": self.declaration_type_spec,
            "Type_Declaration_Stmt": self.type_declaration_stmt,
            "Entity_Decl": self.entity_decl,
            "Array_Spec": self.array_spec,
            "Ac_Value_List": self.ac_value_list,
            "Array_Constructor": self.array_constructor,
            "Loop_Control": self.loop_control,
            "Block_Nonlabel_Do_Construct": self.block_nonlabel_do_construct,
            "Real_Literal_Constant": self.real_literal_constant,
            "Subscript_Triplet": self.subscript_triplet,
            "Section_Subscript_List": self.section_subscript_list,
            "Explicit_Shape_Spec_List": self.explicit_shape_spec_list,
            "Explicit_Shape_Spec": self.explicit_shape_spec,
            "Type_Attr_Spec": self.type_attr_spec,
            "Attr_Spec": self.attr_spec,
            "Intent_Spec": self.intent_spec,
            "Access_Spec": self.access_spec,
            "Allocatable_Stmt": self.allocatable_stmt,
            "Asynchronous_Stmt": self.asynchronous_stmt,
            "Bind_Stmt": self.bind_stmt,
            "Common_Stmt": self.common_stmt,
            "Data_Stmt": self.data_stmt,
            "Dimension_Stmt": self.dimension_stmt,
            "External_Stmt": self.external_stmt,
            "Intent_Stmt": self.intent_stmt,
            "Intrinsic_Stmt": self.intrinsic_stmt,
            "Optional_Stmt": self.optional_stmt,
            "Parameter_Stmt": self.parameter_stmt,
            "Pointer_Stmt": self.pointer_stmt,
            "Protected_Stmt": self.protected_stmt,
            "Save_Stmt": self.save_stmt,
            "Target_Stmt": self.target_stmt,
            "Value_Stmt": self.value_stmt,
            "Volatile_Stmt": self.volatile_stmt,
            "Execution_Part": self.execution_part,
            "Execution_Part_Construct": self.execution_part_construct,
            "Action_Stmt": self.action_stmt,
            "Assignment_Stmt": self.assignment_stmt,
            "Pointer_Assignment_Stmt": self.pointer_assignment_stmt,
            "Where_Stmt": self.where_stmt,
            "Forall_Stmt": self.forall_stmt,
            "Where_Construct": self.where_construct,
            "Where_Construct_Stmt": self.where_construct_stmt,
            "Masked_Elsewhere_Stmt": self.masked_elsewhere_stmt,
            "Elsewhere_Stmt": self.elsewhere_stmt,
            "End_Where_Stmt": self.end_where_stmt,
            "Forall_Construct": self.forall_construct,
            "Forall_Header": self.forall_header,
            "Forall_Triplet_Spec": self.forall_triplet_spec,
            "Forall_Stmt": self.forall_stmt,
            "End_Forall_Stmt": self.end_forall_stmt,
            "Arithmetic_If_Stmt": self.arithmetic_if_stmt,
            "If_Construct": self.if_construct,
            "If_Stmt": self.if_stmt,
            "If_Then_Stmt": self.if_then_stmt,
            "Else_If_Stmt": self.else_if_stmt,
            "Else_Stmt": self.else_stmt,
            "End_If_Stmt": self.end_if_stmt,
            "Case_Construct": self.case_construct,
            "Select_Case_Stmt": self.select_case_stmt,
            "Case_Stmt": self.case_stmt,
            "End_Select_Stmt": self.end_select_stmt,
            "Do_Construct": self.do_construct,
            "Label_Do_Stmt": self.label_do_stmt,
            "Nonlabel_Do_Stmt": self.nonlabel_do_stmt,
            "End_Do_Stmt": self.end_do_stmt,
            "Interface_Block": self.interface_block,
            "Interface_Stmt": self.interface_stmt,
            "End_Interface_Stmt": self.end_interface_stmt,
            "Generic_Spec": self.generic_spec,
            "Name": self.name,
            "Type_Name": self.type_name,
            "Specification_Part": self.specification_part,
            "Intrinsic_Type_Spec": self.intrinsic_type_spec,
            "Entity_Decl_List": self.entity_decl_list,
            "Int_Literal_Constant": self.int_literal_constant,
            "Logical_Literal_Constant": self.logical_literal_constant,
            "Actual_Arg_Spec_List": self.actual_arg_spec_list,
            "Attr_Spec_List": self.attr_spec_list,
            "Initialization": self.initialization,
            "Procedure_Declaration_Stmt": self.procedure_declaration_stmt,
            "Type_Bound_Procedure_Part": self.type_bound_procedure_part,
            "Contains_Stmt": self.contains_stmt,
            "Call_Stmt": self.call_stmt,
            "Return_Stmt": self.return_stmt,
            "Stop_Stmt": self.stop_stmt,
            "Dummy_Arg_List": self.dummy_arg_list,
            "Part_Ref": self.part_ref,
            "Level_2_Expr": self.level_2_expr,
            "Equiv_Operand": self.level_2_expr,
            "Level_3_Expr": self.level_2_expr,
            "Level_4_Expr": self.level_2_expr,
            "Add_Operand": self.level_2_expr,
            "Or_Operand": self.level_2_expr,
            "And_Operand": self.level_2_expr,
            "Level_2_Unary_Expr": self.level_2_expr,
            "Mult_Operand": self.power_expr,
            "Parenthesis": self.parenthesis_expr,
            "Intrinsic_Name": self.intrinsic_name,
            "Intrinsic_Function_Reference": self.intrinsic_function_reference,
            "Only_List": self.only_list,
            "Structure_Constructor": self.structure_constructor,
            "Component_Spec_List": self.component_spec_list,
            "Write_Stmt": self.write_stmt,
            "Assumed_Shape_Spec_List": self.assumed_shape_spec_list,
            "Allocate_Stmt": self.allocate_stmt,
            "Allocation_List": self.allocation_list,
            "Allocation": self.allocation,
            "Allocate_Shape_Spec": self.allocate_shape_spec,
            "Allocate_Shape_Spec_List": self.allocate_shape_spec_list,
        }

    def list_tables(self):
        for i in self.tables._symbol_tables:
            print(i)

    def create_children(self, node: FASTNode):
        return [self.create_ast(child)
                for child in node] if isinstance(node,
                                                 (list,
                                                  tuple)) else [self.create_ast(child) for child in node.children]

    def create_ast(self, node=None):
        """
        Creates an AST from a FASTNode
        :param node: FASTNode
        :note: this is a recursive function, and relies on the dictionary of supported syntax to call the correct converter functions
        """
        if node is not None:
            if isinstance(node, (list, tuple)):
                return [self.create_ast(child) for child in node]
            return self.supported_fortran_syntax[type(node).__name__](node)
        return None

    def write_stmt(self, node: FASTNode):
        children = self.create_children(node.children[1])
        line = get_line(node)
        return ast_internal_classes.Write_Stmt_Node(args=children, line_number=line)

    def program(self, node: FASTNode):
        children = self.create_children(node)

        main_program = get_child(children, ast_internal_classes.Main_Program_Node)

        function_definitions = [i for i in children if isinstance(i, ast_internal_classes.Function_Subprogram_Node)]

        subroutine_definitions = [i for i in children if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node)]
        modules = [node for node in children if isinstance(node, ast_internal_classes.Module_Node)]

        return ast_internal_classes.Program_Node(main_program=main_program,
                                                 function_definitions=function_definitions,
                                                 subroutine_definitions=subroutine_definitions,
                                                 modules=modules)

    def main_program(self, node: FASTNode):
        children = self.create_children(node)

        name = get_child(children, ast_internal_classes.Program_Stmt_Node)
        specification_part = get_child(children, ast_internal_classes.Specification_Part_Node)
        execution_part = get_child(children, ast_internal_classes.Execution_Part_Node)

        return ast_internal_classes.Main_Program_Node(name=name,
                                                      specification_part=specification_part,
                                                      execution_part=execution_part)

    def program_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, Name_Node)
        return ast_internal_classes.Program_Stmt_Node(name=name, line_number=node.item.span)

    def subroutine_subprogram(self, node: FASTNode):
        children = self.create_children(node)

        name = get_child(children, ast_internal_classes.Subroutine_Stmt_Node)
        specification_part = get_child(children, ast_internal_classes.Specification_Part_Node)
        execution_part = get_child(children, ast_internal_classes.Execution_Part_Node)
        return_type = ast_internal_classes.Void
        return ast_internal_classes.Subroutine_Subprogram_Node(
            name=name.name,
            args=name.args,
            specification_part=specification_part,
            execution_part=execution_part,
            type=return_type,
            line_number=name.line_number,
        )

    def end_program_stmt(self, node: FASTNode):
        return node

    def only_list(self, node: FASTNode):
        children = self.create_children(node)
        names = [i for i in children if isinstance(i, ast_internal_classes.Name_Node)]
        return ast_internal_classes.Only_List_Node(names=names)

    def function_subprogram(self, node: FASTNode):
        raise NotImplementedError("Function subprograms are not supported yet")

    def subroutine_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Arg_List_Node)
        return ast_internal_classes.Subroutine_Stmt_Node(name=name, args=args.args, line_number=node.item.span)

    def ac_value_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Ac_Value_List_Node(value_list=children)

    def power_expr(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        #child 0 is the base, child 2 is the exponent
        #child 1 is "**"
        return ast_internal_classes.Call_Expr_Node(name=ast_internal_classes.Name_Node(name="pow"),
                                                   args=[children[0], children[2]],
                                                   line_number=line)

    def array_constructor(self, node: FASTNode):
        children = self.create_children(node)
        value_list = get_child(children, ast_internal_classes.Ac_Value_List_Node)
        return ast_internal_classes.Array_Constructor_Node(value_list=value_list.value_list)

    def allocate_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Allocate_Stmt_Node(allocation_list=children[1])

    def allocation_list(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def allocation(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        shape = get_child(children, ast_internal_classes.Allocate_Shape_Spec_List)
        return ast_internal_classes.Allocation_Node(name=name, shape=shape)

    def allocate_shape_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Allocate_Shape_Spec_List(shape_list=children)

    def allocate_shape_spec(self, node: FASTNode):
        children = self.create_children(node)
        if len(children) != 2:
            raise NotImplementedError("Only simple allocate shape specs are supported")
        return children[1]

    def structure_constructor(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Type_Name_Node)
        args = get_child(children, ast_internal_classes.Component_Spec_List_Node)
        return ast_internal_classes.Structure_Constructor_Node(name=name, args=args.args, type=None)

    def intrinsic_name(self, node: FASTNode):
        name = node.string
        replacements = {
            "INT": "__dace_int",
            "DBLE": "__dace_dble",
            "SQRT": "sqrt",
            "COSH": "cosh",
            "ABS": "abs",
            "MIN": "min",
            "MAX": "max",
            "EXP": "exp",
            "EPSILON": "__dace_epsilon",
            "TANH": "tanh",
            "SUM": "__dace_sum",
            "SIGN": "__dace_sign",
            "EXP": "exp",
            "SELECTED_INT_KIND": "__dace_selected_int_kind",
            "SELECTED_REAL_KIND": "__dace_selected_real_kind",
        }
        return ast_internal_classes.Name_Node(name=replacements[name])

    def intrinsic_function_reference(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Arg_List_Node)
        if name.name == "__dace_selected_int_kind":
            import math
            return ast_internal_classes.Int_Literal_Node(value=str(
                math.ceil((math.log2(math.pow(10, int(args.args[0].value))) + 1) / 8)),
                                                         line_number=line)
        # This selects the smallest kind that can hold the given number of digits (fp64,fp32 or fp16)
        elif name.name == "__dace_selected_real_kind":
            if int(args.args[0].value) >= 9 or int(args.args[1].value) > 126:
                return ast_internal_classes.Int_Literal_Node(value="8", line_number=line)
            elif int(args.args[0].value) >= 3 or int(args.args[1].value) > 14:
                return ast_internal_classes.Int_Literal_Node(value="4", line_number=line)
            else:
                return ast_internal_classes.Int_Literal_Node(value="2", line_number=line)

        func_types = {
            "__dace_int": "INT",
            "__dace_dble": "DOUBLE",
            "sqrt": "DOUBLE",
            "cosh": "DOUBLE",
            "abs": "DOUBLE",
            "min": "DOUBLE",
            "max": "DOUBLE",
            "exp": "DOUBLE",
            "__dace_epsilon": "DOUBLE",
            "tanh": "DOUBLE",
            "__dace_sum": "DOUBLE",
            "__dace_sign": "DOUBLE",
            "exp": "DOUBLE",
            "__dace_selected_int_kind": "INT",
            "__dace_selected_real_kind": "INT",
        }
        call_type = func_types[name.name]
        return ast_internal_classes.Call_Expr_Node(name=name, type=call_type, args=args.args, line_number=line)

    def function_stmt(self, node: FASTNode):
        raise NotImplementedError(
            "Function statements are not supported yet - at least not if defined this way. Not encountered in code yet."
        )

    def end_subroutine_stmt(self, node: FASTNode):
        return node

    def end_function_stmt(self, node: FASTNode):
        return node

    def parenthesis_expr(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Parenthesis_Expr_Node(expr=children[1])

    def module(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Module_Stmt_Node)
        specification_part = get_child(children, ast_internal_classes.Specification_Part_Node)

        function_definitions = [i for i in children if isinstance(i, ast_internal_classes.Function_Subprogram_Node)]

        subroutine_definitions = [i for i in children if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node)]
        return ast_internal_classes.Module_Node(
            name=name.name,
            specification_part=specification_part,
            function_definitions=function_definitions,
            subroutine_definitions=subroutine_definitions,
            line_number=name.line_number,
        )

    def module_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        return ast_internal_classes.Module_Stmt_Node(name=name, line_number=node.item.span)

    def end_module_stmt(self, node: FASTNode):
        return node

    def use_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        only_list = get_child(children, ast_internal_classes.Only_List_Node)
        return ast_internal_classes.Use_Stmt_Node(name=name.name, list=only_list.names)

    def implicit_part(self, node: FASTNode):
        return node

    def implicit_stmt(self, node: FASTNode):
        return node

    def implicit_none_stmt(self, node: FASTNode):
        return node

    def implicit_part_stmt(self, node: FASTNode):
        return node

    def declaration_construct(self, node: FASTNode):
        raise NotImplementedError("Declaration constructs are not supported yet")
        return node

    def declaration_type_spec(self, node: FASTNode):
        raise NotImplementedError("Declaration type spec is not supported yet")
        return node

    def assumed_shape_spec_list(self, node: FASTNode):
        return node

    def parse_shape_specification(self, dim: f03.Explicit_Shape_Spec, size: List[FASTNode], offset: List[int]):

        dim_expr = [i for i in dim.children if i is not None]

        # handle size definition
        if len(dim_expr) == 1:
            dim_expr = dim_expr[0]
            #now to add the dimension to the size list after processing it if necessary
            size.append(self.create_ast(dim_expr))
            offset.append(1)
        # Here we support arrays that have size declaration - with initial offset.
        elif len(dim_expr) == 2:
            # extract offets
            for expr in dim_expr:
                if not isinstance(expr, f03.Int_Literal_Constant):
                    raise TypeError("Array offsets must be constant expressions!")
            offset.append(int(dim_expr[0].tostr()))

            fortran_size = int(dim_expr[1].tostr()) - int(dim_expr[0].tostr()) + 1
            fortran_ast_size = f03.Int_Literal_Constant(str(fortran_size))

            size.append(self.create_ast(fortran_ast_size))
        else:
            raise TypeError("Array dimension must be at most two expressions")

    def type_declaration_stmt(self, node: FASTNode):

        #decide if its a intrinsic variable type or a derived type

        type_of_node = get_child(node, [f03.Intrinsic_Type_Spec, f03.Declaration_Type_Spec])

        if isinstance(type_of_node, f03.Intrinsic_Type_Spec):
            derived_type = False
            basetype = type_of_node.items[0]
        elif isinstance(type_of_node, f03.Declaration_Type_Spec):
            derived_type = True
            basetype = type_of_node.items[1].string
        else:
            raise TypeError("Type of node must be either Intrinsic_Type_Spec or Declaration_Type_Spec")
        kind = None
        if len(type_of_node.items) >= 2:
            if type_of_node.items[1] is not None:
                if not derived_type:
                    kind = type_of_node.items[1].items[1].string
                    if self.symbols[kind] is not None:
                        if basetype == "REAL":
                            if self.symbols[kind].value == "8":
                                basetype = "REAL8"
                        elif basetype == "INTEGER":
                            if self.symbols[kind].value == "4":
                                basetype = "INTEGER"
                        else:
                            raise TypeError("Derived type not supported")
                    else:
                        raise TypeError("Derived type not supported")
                if derived_type:
                    raise TypeError("Derived type not supported")
        if not derived_type:
            testtype = self.types[basetype]
        else:

            testtype = basetype

        # get the names of the variables being defined
        names_list = get_child(node, ["Entity_Decl_List", "Component_Decl_List"])

        #get the names out of the name list
        names = get_children(names_list, [f03.Entity_Decl, f03.Component_Decl])

        #get the attributes of the variables being defined
        # alloc relates to whether it is statically (False) or dynamically (True) allocated
        # parameter means its a constant, so we should transform it into a symbol
        attributes = get_children(node, "Attr_Spec_List")

        alloc = False
        symbol = False
        attr_size = None
        attr_offset = None
        for i in attributes:
            if i.string.lower() == "allocatable":
                alloc = True
            if i.string.lower() == "parameter":
                symbol = True

            if isinstance(i, Fortran2008.Attr_Spec_List):

                attr_size = []
                attr_offset = []
                sizes = get_child(get_child(i, ["Dimension_Attr_Spec"]), ["Explicit_Shape_Spec_List"])
                
                for shape_spec in get_children(sizes, [f03.Explicit_Shape_Spec]):
                    print(shape_spec)
                    self.parse_shape_specification(shape_spec, attr_size, attr_offset)
                print(sizes.children)
                print(type(sizes))
                #print(sizes.children)

                #if len(i.children) > 0 and isinstance(i.children[0], f03.Dimension_Attr_Spec):
                #    print(i, dir(i), type(i.children[0]), dir(i.children[0]))

        #sizes = get_child(attributes, ["Attr_Spec_List"])
        #print(sizes)

        vardecls = []

        for var in names:
            #first handle dimensions
            size = None
            offset = None
            var_components = self.create_children(var)
            array_sizes = get_children(var, "Explicit_Shape_Spec_List")
            actual_name = get_child(var_components, ast_internal_classes.Name_Node)
            if len(array_sizes) == 1:
                array_sizes = array_sizes[0]
                size = []
                offset = []
                for dim in array_sizes.children:
                    #sanity check
                    if isinstance(dim, f03.Explicit_Shape_Spec):
                        self.parse_shape_specification(dim, size, offset)
            #handle initializiation
            init = None

            initialization = get_children(var, f03.Initialization)
            if len(initialization) == 1:
                initialization = initialization[0]
                #if there is an initialization, the actual expression is in the second child, with the first being the equals sign
                if len(initialization.children) < 2:
                    raise ValueError("Initialization must have an expression")
                raw_init = initialization.children[1]
                init = self.create_ast(raw_init)

            print('t', symbol, size, attr_size)
            print(offset, attr_offset)
            if symbol == False:

                if attr_size is None:
                    vardecls.append(
                        ast_internal_classes.Var_Decl_Node(name=actual_name.name,
                                                        type=testtype,
                                                        alloc=alloc,
                                                        sizes=size,
                                                        offsets=offset,
                                                        kind=kind,
                                                        line_number=node.item.span))
                else:
                    vardecls.append(
                        ast_internal_classes.Var_Decl_Node(name=actual_name.name,
                                                        type=testtype,
                                                        alloc=alloc,
                                                        sizes=attr_size,
                                                        offsets=attr_offset,
                                                        kind=kind,
                                                        line_number=node.item.span))
            else:
                if size is None and attr_size is None:
                    self.symbols[actual_name.name] = init
                    vardecls.append(
                        ast_internal_classes.Symbol_Decl_Node(name=actual_name.name,
                                                              type=testtype,
                                                              alloc=alloc,
                                                              init=init,
                                                              line_number=node.item.span))
                elif attr_size is not None:
                    vardecls.append(
                        ast_internal_classes.Symbol_Array_Decl_Node(name=actual_name.name,
                                                                    type=testtype,
                                                                    alloc=alloc,
                                                                    sizes=attr_size,
                                                                    offsets=attr_offset,
                                                                    kind=kind,
                                                                    init=init,
                                                                    line_number=node.item.span))
                else:
                    vardecls.append(
                        ast_internal_classes.Symbol_Array_Decl_Node(name=actual_name.name,
                                                                    type=testtype,
                                                                    alloc=alloc,
                                                                    sizes=size,
                                                                    offsets=offset,
                                                                    kind=kind,
                                                                    init=init,
                                                                    line_number=node.item.span))
        #print(vardecls[0].sizes)
        #print(vardecls[0].offsets)
        return ast_internal_classes.Decl_Stmt_Node(vardecl=vardecls, line_number=node.item.span)

    def entity_decl(self, node: FASTNode):
        raise NotImplementedError("Entity decl is not supported yet")

    def array_spec(self, node: FASTNode):
        raise NotImplementedError("Array spec is not supported yet")
        return node

    def explicit_shape_spec_list(self, node: FASTNode):
        return node

    def explicit_shape_spec(self, node: FASTNode):
        return node

    def type_attr_spec(self, node: FASTNode):
        return node

    def attr_spec(self, node: FASTNode):
        return node

    def intent_spec(self, node: FASTNode):
        raise NotImplementedError("Intent spec is not supported yet")
        return node

    def access_spec(self, node: FASTNode):
        raise NotImplementedError("Access spec is not supported yet")
        return node

    def allocatable_stmt(self, node: FASTNode):
        raise NotImplementedError("Allocatable stmt is not supported yet")
        return node

    def asynchronous_stmt(self, node: FASTNode):
        raise NotImplementedError("Asynchronous stmt is not supported yet")
        return node

    def bind_stmt(self, node: FASTNode):
        raise NotImplementedError("Bind stmt is not supported yet")
        return node

    def common_stmt(self, node: FASTNode):
        raise NotImplementedError("Common stmt is not supported yet")
        return node

    def data_stmt(self, node: FASTNode):
        raise NotImplementedError("Data stmt is not supported yet")
        return node

    def dimension_stmt(self, node: FASTNode):
        raise NotImplementedError("Dimension stmt is not supported yet")
        return node

    def external_stmt(self, node: FASTNode):
        raise NotImplementedError("External stmt is not supported yet")
        return node

    def intent_stmt(self, node: FASTNode):
        return node

    def intrinsic_stmt(self, node: FASTNode):
        return node

    def optional_stmt(self, node: FASTNode):
        return node

    def parameter_stmt(self, node: FASTNode):
        return node

    def pointer_stmt(self, node: FASTNode):
        return node

    def protected_stmt(self, node: FASTNode):
        return node

    def save_stmt(self, node: FASTNode):
        return node

    def target_stmt(self, node: FASTNode):
        return node

    def value_stmt(self, node: FASTNode):
        return node

    def volatile_stmt(self, node: FASTNode):
        return node

    def execution_part(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Execution_Part_Node(execution=children)

    def execution_part_construct(self, node: FASTNode):
        return node

    def action_stmt(self, node: FASTNode):
        return node

    def level_2_expr(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        if len(children) == 3:
            return ast_internal_classes.BinOp_Node(lval=children[0], op=children[1], rval=children[2], line_number=line)
        else:
            return ast_internal_classes.UnOp_Node(lval=children[1], op=children[0], line_number=line)

    def assignment_stmt(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)

        if len(children) == 3:
            return ast_internal_classes.BinOp_Node(lval=children[0], op=children[1], rval=children[2], line_number=line)
        else:
            return ast_internal_classes.UnOp_Node(lval=children[1], op=children[0], line_number=line)

    def pointer_assignment_stmt(self, node: FASTNode):
        return node

    def where_stmt(self, node: FASTNode):
        return node

    def forall_stmt(self, node: FASTNode):
        return node

    def where_construct(self, node: FASTNode):
        return node

    def where_construct_stmt(self, node: FASTNode):
        return node

    def masked_elsewhere_stmt(self, node: FASTNode):
        return node

    def elsewhere_stmt(self, node: FASTNode):
        return node

    def end_where_stmt(self, node: FASTNode):
        return node

    def forall_construct(self, node: FASTNode):
        return node

    def forall_header(self, node: FASTNode):
        return node

    def forall_triplet_spec(self, node: FASTNode):
        return node

    def forall_stmt(self, node: FASTNode):
        return node

    def end_forall_stmt(self, node: FASTNode):
        return node

    def arithmetic_if_stmt(self, node: FASTNode):
        return node

    def if_stmt(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        cond = children[0]
        body = children[1:]
        return ast_internal_classes.If_Stmt_Node(cond=cond,
                                                 body=ast_internal_classes.Execution_Part_Node(execution=body),
                                                 body_else=ast_internal_classes.Execution_Part_Node(execution=[]),
                                                 line_number=line)

    def if_construct(self, node: FASTNode):
        children = self.create_children(node)
        cond = children[0]
        body = []
        body_else = []
        else_mode = False
        line = get_line(node)
        if line is None:
            line = cond.line_number
        toplevelIf = ast_internal_classes.If_Stmt_Node(cond=cond, line_number=line)
        currentIf = toplevelIf
        for i in children[1:-1]:
            if isinstance(i, ast_internal_classes.Else_If_Stmt_Node):
                newif = ast_internal_classes.If_Stmt_Node(cond=i.cond, line_number=i.line_number)
                currentIf.body = ast_internal_classes.Execution_Part_Node(execution=body)
                currentIf.body_else = ast_internal_classes.Execution_Part_Node(execution=[newif])
                currentIf = newif
                body = []
                continue
            if isinstance(i, ast_internal_classes.Else_Separator_Node):
                else_mode = True
                continue
            if else_mode:
                body_else.append(i)
            else:
                body.append(i)
        currentIf.body = ast_internal_classes.Execution_Part_Node(execution=body)
        currentIf.body_else = ast_internal_classes.Execution_Part_Node(execution=body_else)
        return toplevelIf

    def if_then_stmt(self, node: FASTNode):
        children = self.create_children(node)
        if len(children) != 1:
            raise ValueError("If statement must have a condition")
        return_value = children[0]
        return_value.line_number = node.item.span
        return return_value

    def else_if_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Else_If_Stmt_Node(cond=children[0], line_number=get_line(node))

    def else_stmt(self, node: FASTNode):
        return ast_internal_classes.Else_Separator_Node(line_number=node.item.span)

    def end_if_stmt(self, node: FASTNode):
        return node

    def case_construct(self, node: FASTNode):
        return node

    def select_case_stmt(self, node: FASTNode):
        return node

    def case_stmt(self, node: FASTNode):
        return node

    def end_select_stmt(self, node: FASTNode):
        return node

    def do_construct(self, node: FASTNode):
        return node

    def label_do_stmt(self, node: FASTNode):
        return node

    def nonlabel_do_stmt(self, node: FASTNode):
        children = self.create_children(node)
        loop_control = get_child(children, ast_internal_classes.Loop_Control_Node)
        return ast_internal_classes.Nonlabel_Do_Stmt_Node(iter=loop_control.iter,
                                                          cond=loop_control.cond,
                                                          init=loop_control.init,
                                                          line_number=node.item.span)

    def end_do_stmt(self, node: FASTNode):
        return node

    def interface_block(self, node: FASTNode):
        return node

    def interface_stmt(self, node: FASTNode):
        return node

    def end_interface_stmt(self, node: FASTNode):
        return node

    def generic_spec(self, node: FASTNode):
        return node

    def procedure_declaration_stmt(self, node: FASTNode):
        return node

    def type_bound_procedure_part(self, node: FASTNode):
        return node

    def contains_stmt(self, node: FASTNode):
        return node

    def call_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Arg_List_Node)
        return ast_internal_classes.Call_Expr_Node(name=name, args=args.args, type=None, line_number=node.item.span)

    def return_stmt(self, node: FASTNode):
        return node

    def stop_stmt(self, node: FASTNode):
        return node

    def dummy_arg_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Arg_List_Node(args=children)

    def component_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Component_Spec_List_Node(args=children)

    def attr_spec_list(self, node: FASTNode):
        return node

    def part_ref(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Section_Subscript_List_Node)
        return ast_internal_classes.Call_Expr_Node(
            name=name,
            args=args.list,
            line=line,
        )

    def loop_control(self, node: FASTNode):
        children = self.create_children(node)
        #Structure of loop control is:
        # child[1]. Loop control variable
        # child[1][0] Loop start
        # child[1][1] Loop end
        iteration_variable = children[1][0]
        loop_start = children[1][1][0]
        loop_end = children[1][1][1]
        if len(children[1][1]) == 3:
            loop_step = children[1][1][2]
        else:
            loop_step = ast_internal_classes.Int_Literal_Node(value="1")
        init_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable, op="=", rval=loop_start)
        if isinstance(loop_step, ast_internal_classes.UnOp_Node):
            if loop_step.op == "-":
                cond_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable, op=">=", rval=loop_end)
        else:
            cond_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable, op="<=", rval=loop_end)
        iter_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable,
                                                    op="=",
                                                    rval=ast_internal_classes.BinOp_Node(lval=iteration_variable,
                                                                                         op="+",
                                                                                         rval=loop_step))
        return ast_internal_classes.Loop_Control_Node(init=init_expr, cond=cond_expr, iter=iter_expr)

    def block_nonlabel_do_construct(self, node: FASTNode):
        children = self.create_children(node)
        do = get_child(children, ast_internal_classes.Nonlabel_Do_Stmt_Node)
        body = children[1:-1]
        return ast_internal_classes.For_Stmt_Node(init=do.init,
                                                  cond=do.cond,
                                                  iter=do.iter,
                                                  body=ast_internal_classes.Execution_Part_Node(execution=body),
                                                  line_number=do.line_number)

    def real_literal_constant(self, node: FASTNode):
        return node

    def subscript_triplet(self, node: FASTNode):
        if node.string == ":":
            return ast_internal_classes.ParDecl_Node(type="ALL")
        children = self.create_children(node)
        return ast_internal_classes.ParDecl_Node(type="RANGE", range=children)

    def section_subscript_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Section_Subscript_List_Node(list=children)

    def specification_part(self, node: FASTNode):
        #TODO this can be refactored to consider more fortran declaration options. Currently limited to what is encountered in code.
        others = [self.create_ast(i) for i in node.children if not isinstance(i, f08.Type_Declaration_Stmt)]

        decls = [self.create_ast(i) for i in node.children if isinstance(i, f08.Type_Declaration_Stmt)]

        uses = [self.create_ast(i) for i in node.children if isinstance(i, f08.Use_Stmt)]
        tmp = [self.create_ast(i) for i in node.children]
        typedecls = [i for i in tmp if isinstance(i, ast_internal_classes.Type_Decl_Node)]
        symbols = []
        for i in others:
            if isinstance(i, list):
                symbols.extend(j for j in i if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
            if isinstance(i, ast_internal_classes.Decl_Stmt_Node):
                symbols.extend(j for j in i.vardecl if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
        for i in decls:
            if isinstance(i, list):
                symbols.extend(j for j in i if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
            if isinstance(i, ast_internal_classes.Decl_Stmt_Node):
                symbols.extend(j for j in i.vardecl if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
        names_filtered = []
        for j in symbols:
            for i in decls:
                names_filtered.extend(ii.name for ii in i.vardecl if j.name == ii.name)
        decl_filtered = []
        for i in decls:
            # NOTE: Assignment/named expressions (walrus operator) works with Python 3.8 and later.
            # if vardecl_filtered := [ii for ii in i.vardecl if ii.name not in names_filtered]:
            vardecl_filtered = [ii for ii in i.vardecl if ii.name not in names_filtered]
            if vardecl_filtered:
                decl_filtered.append(ast_internal_classes.Decl_Stmt_Node(vardecl=vardecl_filtered))
        return ast_internal_classes.Specification_Part_Node(specifications=decl_filtered,
                                                            symbols=symbols,
                                                            uses=uses,
                                                            typedecls=typedecls)

    def intrinsic_type_spec(self, node: FASTNode):
        return node

    def entity_decl_list(self, node: FASTNode):
        return node

    def int_literal_constant(self, node: FASTNode):
        return ast_internal_classes.Int_Literal_Node(value=node.string)

    def logical_literal_constant(self, node: FASTNode):
        if node.string in [".TRUE.", ".true.", ".True."]:
            return ast_internal_classes.Bool_Literal_Node(value="True")
        if node.string in [".FALSE.", ".false.", ".False."]:
            return ast_internal_classes.Bool_Literal_Node(value="False")
        raise ValueError("Unknown logical literal constant")

    def real_literal_constant(self, node: FASTNode):
        return ast_internal_classes.Real_Literal_Node(value=node.string)

    def actual_arg_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Arg_List_Node(args=children)

    def initialization(self, node: FASTNode):
        return node

    def name(self, node: FASTNode):
        return ast_internal_classes.Name_Node(name=node.string)

    def type_name(self, node: FASTNode):
        return ast_internal_classes.Type_Name_Node(name=node.string)

    def tuple_node(self, node: FASTNode):
        return node

    def str_node(self, node: FASTNode):
        return node
