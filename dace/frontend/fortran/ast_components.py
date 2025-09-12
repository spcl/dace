# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, List, Optional, Type, TypeVar, Union, overload, TYPE_CHECKING, Dict, Tuple

import fparser
import networkx as nx
from fparser.two import Fortran2003 as f03
from fparser.two import Fortran2008 as f08
from fparser.two.Fortran2003 import Function_Subprogram, Function_Stmt, Prefix, Intrinsic_Type_Spec, \
    Assignment_Stmt, Logical_Literal_Constant, Real_Literal_Constant, Signed_Real_Literal_Constant, \
    Int_Literal_Constant, Signed_Int_Literal_Constant, Hex_Constant, Function_Reference, Length_Selector, Kind_Selector
from fparser.two.utils import Base

from dace.frontend.fortran import ast_internal_classes
from dace.frontend.fortran.ast_internal_classes import Name_Node, Program_Node, Decl_Stmt_Node, Var_Decl_Node
from dace.frontend.fortran.ast_transforms import StructLister, StructDependencyLister, Structures
from dace.frontend.fortran.ast_utils import singular

if TYPE_CHECKING:
    from dace.frontend.fortran.intrinsics import FortranIntrinsics

# We rely on fparser to provide an initial AST and convert to a version that is more suitable for our purposes

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
    # Temporary workaround to allow feature list to be generated
    return None
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


def get_line(node: Base) -> Tuple[int, int]:
    line = None
    if node.item:
        line = node.item.span
    if not line and node.parent:
        line = get_line(node.parent)
    if not line:
        line = (0, 0)
    return line


class InternalFortranAst:
    """
    This class is used to translate the fparser AST to our own AST of Fortran
    the supported_fortran_types dictionary is used to determine which types are supported by our compiler
    for each entry in the dictionary, the key is the name of the class in the fparser AST and the value
    is the name of the function that will be used to translate the fparser AST to our AST
    """

    def __init__(self):
        """
        Initialization of the AST converter
        """
        self.to_parse_list = {}
        self.unsupported_fortran_syntax = {}
        self.current_ast = None
        self.functions_and_subroutines = []
        self.symbols = {}
        self.intrinsics_list = []
        self.placeholders = {}
        self.placeholders_offsets = {}
        self.types = {
            "LOGICAL": "LOGICAL",
            "CHARACTER": "CHAR",
            "INTEGER": "INTEGER",
            "INTEGER4": "INTEGER",
            "INTEGER8": "INTEGER8",
            "REAL4": "REAL",
            "REAL8": "DOUBLE",
            "DOUBLE PRECISION": "DOUBLE",
            "REAL": "REAL",
            "CLASS": "CLASS",
            "Unknown": "REAL",
        }
        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        self.intrinsic_handler = FortranIntrinsics()
        self.supported_fortran_syntax = {
            "str": self.str_node,
            "tuple": self.tuple_node,
            "Program": self.program,
            "Main_Program": self.main_program,
            "Program_Stmt": self.program_stmt,
            "End_Program_Stmt": self.end_program_stmt,
            "Subroutine_Subprogram": self.subroutine_subprogram,
            "Function_Subprogram": self.function_subprogram,
            "Module_Subprogram_Part": self.module_subprogram_part,
            "Internal_Subprogram_Part": self.internal_subprogram_part,
            "Subroutine_Stmt": self.subroutine_stmt,
            "Function_Stmt": self.function_stmt,
            "Prefix": self.prefix_stmt,
            "End_Subroutine_Stmt": self.end_subroutine_stmt,
            "End_Function_Stmt": self.end_function_stmt,
            "Rename": self.rename,
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
            "Signed_Real_Literal_Constant": self.real_literal_constant,
            "Char_Literal_Constant": self.char_literal_constant,
            "Subscript_Triplet": self.subscript_triplet,
            "Section_Subscript_List": self.section_subscript_list,
            "Explicit_Shape_Spec_List": self.explicit_shape_spec_list,
            "Explicit_Shape_Spec": self.explicit_shape_spec,
            "Type_Attr_Spec": self.type_attr_spec,
            "Attr_Spec": self.attr_spec,
            "Intent_Spec": self.intent_spec,
            "Access_Spec": self.access_spec,
            "Access_Stmt": self.access_stmt,
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
            "Procedure_Name_List": self.procedure_name_list,
            "Procedure_Stmt": self.procedure_stmt,
            "End_Interface_Stmt": self.end_interface_stmt,
            "Generic_Spec": self.generic_spec,
            "Name": self.name,
            "Type_Name": self.type_name,
            "Specification_Part": self.specification_part,
            "Intrinsic_Type_Spec": self.intrinsic_type_spec,
            "Entity_Decl_List": self.entity_decl_list,
            "Int_Literal_Constant": self.int_literal_constant,
            "Signed_Int_Literal_Constant": self.int_literal_constant,
            "Hex_Constant": self.hex_constant,
            "Logical_Literal_Constant": self.logical_literal_constant,
            "Actual_Arg_Spec_List": self.actual_arg_spec_list,
            "Actual_Arg_Spec": self.actual_arg_spec,
            "Attr_Spec_List": self.attr_spec_list,
            "Initialization": self.initialization,
            "Procedure_Declaration_Stmt": self.procedure_declaration_stmt,
            "Type_Bound_Procedure_Part": self.type_bound_procedure_part,
            "Data_Pointer_Object": self.data_pointer_object,
            "Contains_Stmt": self.contains_stmt,
            "Call_Stmt": self.call_stmt,
            "Return_Stmt": self.return_stmt,
            "Stop_Stmt": self.stop_stmt,
            "Dummy_Arg_List": self.dummy_arg_list,
            "Dummy_Arg_Name_List": self.dummy_arg_list,
            "Part_Ref": self.part_ref,
            "Level_2_Expr": self.level_2_expr,
            "Equiv_Operand": self.level_2_expr,
            "Level_3_Expr": self.level_2_expr,
            "Level_4_Expr": self.level_2_expr,
            "Level_5_Expr": self.level_2_expr,
            "Add_Operand": self.level_2_expr,
            "Or_Operand": self.level_2_expr,
            "And_Operand": self.level_2_expr,
            "Level_2_Unary_Expr": self.level_2_expr,
            "Mult_Operand": self.power_expr,
            "Parenthesis": self.parenthesis_expr,
            "Intrinsic_Name": self.intrinsic_handler.replace_function_name,
            "Suffix": self.suffix,
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
            "Derived_Type_Def": self.derived_type_def,
            "Derived_Type_Stmt": self.derived_type_stmt,
            "Component_Part": self.component_part,
            "Data_Component_Def_Stmt": self.data_component_def_stmt,
            "End_Type_Stmt": self.end_type_stmt,
            "Data_Ref": self.data_ref,
            "Cycle_Stmt": self.cycle_stmt,
            "Deferred_Shape_Spec": self.deferred_shape_spec,
            "Deferred_Shape_Spec_List": self.deferred_shape_spec_list,
            "Component_Initialization": self.component_initialization,
            "Case_Selector": self.case_selector,
            "Case_Value_Range_List": self.case_value_range_list,
            "Procedure_Designator": self.procedure_designator,
            "Specific_Binding": self.specific_binding,
            "Enum_Def_Stmt": self.enum_def_stmt,
            "Enumerator_Def_Stmt": self.enumerator_def_stmt,
            "Enumerator_List": self.enumerator_list,
            "Enumerator": self.enumerator,
            "End_Enum_Stmt": self.end_enum_stmt,
            "Exit_Stmt": self.exit_stmt,
            "Enum_Def": self.enum_def,
            "Connect_Spec": self.connect_spec,
            "Namelist_Stmt": self.namelist_stmt,
            "Namelist_Group_Object_List": self.namelist_group_object_list,
            "Open_Stmt": self.open_stmt,
            "Connect_Spec_List": self.connect_spec_list,
            "Association": self.association,
            "Association_List": self.association_list,
            "Associate_Stmt": self.associate_stmt,
            "End_Associate_Stmt": self.end_associate_stmt,
            "Associate_Construct": self.associate_construct,
            "Subroutine_Body": self.subroutine_body,
            "Function_Reference": self.function_reference,
            "Binding_Name_List": self.binding_name_list,
            "Generic_Binding": self.generic_binding,
            "Private_Components_Stmt": self.private_components_stmt,
            "Stop_Code": self.stop_code,
            "Error_Stop_Stmt": self.error_stop_stmt,
            "Pointer_Object_List": self.pointer_object_list,
            "Nullify_Stmt": self.nullify_stmt,
            "Deallocate_Stmt": self.deallocate_stmt,
            "Proc_Component_Ref": self.proc_component_ref,
            "Component_Spec": self.component_spec,
            "Allocate_Object_List": self.allocate_object_list,
            "Read_Stmt": self.read_stmt,
            "Close_Stmt": self.close_stmt,
            "Io_Control_Spec": self.io_control_spec,
            "Io_Control_Spec_List": self.io_control_spec_list,
            "Close_Spec_List": self.close_spec_list,
            "Close_Spec": self.close_spec,

            # "Component_Decl_List": self.component_decl_list,
            # "Component_Decl": self.component_decl,
        }
        self.type_arbitrary_array_variable_count = 0

    def fortran_intrinsics(self) -> "FortranIntrinsics":
        return self.intrinsic_handler

    def data_pointer_object(self, node: FASTNode):
        children = self.create_children(node)
        if node.children[1] == "%":
            return ast_internal_classes.Data_Ref_Node(parent_ref=children[0], part_ref=children[2], type="VOID")
        else:
            raise NotImplementedError("Data pointer object not supported yet")

    def create_children(self, node: FASTNode):
        return [self.create_ast(child) for child in node] \
            if isinstance(node, (list, tuple)) else [self.create_ast(child) for child in node.children]

    def cycle_stmt(self, node: FASTNode):
        line = get_line(node)
        return ast_internal_classes.Continue_Node(line_number=line)

    def create_ast(self, node=None):
        """
        Creates an AST from a FASTNode
        :param node: FASTNode
        :note: this is a recursive function, and relies on the dictionary of supported syntax to call the correct converter functions
        """
        if not node:
            return None
        if isinstance(node, (list, tuple)):
            return [self.create_ast(child) for child in node]
        if type(node).__name__ in self.supported_fortran_syntax:
            handler = self.supported_fortran_syntax[type(node).__name__]
            return handler(node)

        if type(node).__name__ == "Intrinsic_Name":
            if node not in self.intrinsics_list:
                self.intrinsics_list.append(node)
        if self.unsupported_fortran_syntax.get(self.current_ast) is None:
            self.unsupported_fortran_syntax[self.current_ast] = []
        if type(node).__name__ not in self.unsupported_fortran_syntax[self.current_ast]:
            if type(node).__name__ not in self.unsupported_fortran_syntax[self.current_ast]:
                self.unsupported_fortran_syntax[self.current_ast].append(type(node).__name__)
        for i in node.children:
            self.create_ast(i)
        print("Unsupported syntax: ", type(node).__name__, node.string)
        return None

    def finalize_ast(self, prog: Program_Node):
        structs_lister = StructLister()
        structs_lister.visit(prog)
        struct_dep_graph = nx.DiGraph()
        for i, name in zip(structs_lister.structs, structs_lister.names):
            if name not in struct_dep_graph.nodes:
                struct_dep_graph.add_node(name)
            struct_deps_finder = StructDependencyLister(structs_lister.names)
            struct_deps_finder.visit(i)
            struct_deps = struct_deps_finder.structs_used
            # print(struct_deps)
            for j, pointing, point_name in zip(struct_deps, struct_deps_finder.is_pointer,
                                               struct_deps_finder.pointer_names):
                if j not in struct_dep_graph.nodes:
                    struct_dep_graph.add_node(j)
                struct_dep_graph.add_edge(name, j, pointing=pointing, point_name=point_name)
        prog.structures = Structures(structs_lister.structs)
        prog.placeholders = self.placeholders
        prog.placeholders_offsets = self.placeholders_offsets

    def suffix(self, node: FASTNode):
        children = self.create_children(node)
        name = children[0]
        return ast_internal_classes.Suffix_Node(name=name)

    def data_ref(self, node: FASTNode):
        children = self.create_children(node)
        idx = len(children) - 1
        parent = children[idx - 1]
        part_ref = children[idx]
        part_ref.isStructMember = True
        # parent.isStructMember=True
        idx = idx - 1
        current = ast_internal_classes.Data_Ref_Node(parent_ref=parent, part_ref=part_ref, type="VOID")

        while idx > 0:
            parent = children[idx - 1]
            current = ast_internal_classes.Data_Ref_Node(parent_ref=parent, part_ref=current, type="VOID")
            idx = idx - 1
        return current

    def end_type_stmt(self, node: FASTNode):
        return None

    def access_stmt(self, node: FASTNode):
        return None

    def generic_binding(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Generic_Binding_Node(name=children[1], binding=children[2])

    def private_components_stmt(self, node: FASTNode):
        return None

    def deallocate_stmt(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        return ast_internal_classes.Deallocate_Stmt_Node(deallocation_list=children[0].list, line_number=line)

    def proc_component_ref(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Data_Ref_Node(parent_ref=children[0], part_ref=children[2], type="VOID")

    def component_spec(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Actual_Arg_Spec_Node(arg_name=children[0], arg=children[1], type="VOID")

    def allocate_object_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Allocate_Object_List_Node(list=children)

    def read_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Read_Stmt_Node(args=children[0], line_number=get_line(node))

    def close_stmt(self, node: FASTNode):
        children = self.create_children(node)
        if node.item is None:
            line = '-1'
        else:
            line = get_line(node)
        return ast_internal_classes.Close_Stmt_Node(args=children[0], line_number=line)

    def io_control_spec(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.IO_Control_Spec_Node(name=children[0], args=children[1])

    def io_control_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.IO_Control_Spec_List_Node(list=children)

    def close_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Close_Spec_List_Node(list=children)

    def close_spec(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Close_Spec_Node(name=children[0], args=children[1])

    def stop_code(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Stop_Stmt_Node(code=node.string)

    def error_stop_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Error_Stmt_Node(error=children[1])

    def pointer_object_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Pointer_Object_List_Node(list=children)

    def nullify_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Nullify_Stmt_Node(list=children[1].list)

    def binding_name_list(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def connect_spec(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Connect_Spec_Node(type=children[0], args=children[1])

    def connect_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Connect_Spec_List_Node(list=children)

    def open_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Open_Stmt_Node(args=children[1].list, line_number=get_line(node))

    def namelist_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Namelist_Stmt_Node(name=children[0][0], list=children[0][1])

    def namelist_group_object_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Namelist_Group_Object_List_Node(list=children)

    def associate_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Associate_Stmt_Node(args=children[1].list)

    def association(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Association_Node(name=children[0], expr=children[2])

    def association_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Association_List_Node(list=children)

    def subroutine_body(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def function_reference(self, node: Function_Reference):
        name, args = self.create_children(node)
        line = get_line(node)
        return ast_internal_classes.Call_Expr_Node(name=name,
                                                   args=args.args if args else [],
                                                   type="VOID", subroutine=False,
                                                   line_number=line)

    def end_associate_stmt(self, node: FASTNode):
        return None

    def associate_construct(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Associate_Construct_Node(associate=children[0], body=children[1])

    def enum_def_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return None

    def enumerator(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def enumerator_def_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return children[1]

    def enumerator_list(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def end_enum_stmt(self, node: FASTNode):
        return None

    def enum_def(self, node: FASTNode):
        children = self.create_children(node)
        return children[1:-1]

    def exit_stmt(self, node: FASTNode):
        line = get_line(node)
        return ast_internal_classes.Exit_Node(line_number=line)

    def deferred_shape_spec(self, node: FASTNode):
        return ast_internal_classes.Defer_Shape_Node()

    def deferred_shape_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def component_initialization(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Component_Initialization_Node(init=children[1])

    def procedure_designator(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Procedure_Separator_Node(parent_ref=children[0], part_ref=children[2])

    def derived_type_def(self, node: FASTNode):
        children = self.create_children(node)
        name = children[0].name
        component_part = get_child(children, ast_internal_classes.Component_Part_Node)
        procedure_part = get_child(children, ast_internal_classes.Bound_Procedures_Node)
        from dace.frontend.fortran.ast_transforms import PartialRenameVar
        if component_part is not None:
            component_part = PartialRenameVar(oldname="__f2dace_A", newname="__f2dace_SA").visit(component_part)
            component_part = PartialRenameVar(oldname="__f2dace_OA", newname="__f2dace_SOA").visit(component_part)
            new_placeholder = {}
            new_placeholder_offsets = {}
            for k, v in self.placeholders.items():
                if "__f2dace_A" in k:
                    new_placeholder[k.replace("__f2dace_A", "__f2dace_SA")] = self.placeholders[k]
                else:
                    new_placeholder[k] = self.placeholders[k]
            self.placeholders = new_placeholder
            for k, v in self.placeholders_offsets.items():
                if "__f2dace_OA" in k:
                    new_placeholder_offsets[k.replace("__f2dace_OA", "__f2dace_SOA")] = self.placeholders_offsets[k]
                else:
                    new_placeholder_offsets[k] = self.placeholders_offsets[k]
            self.placeholders_offsets = new_placeholder_offsets
        return ast_internal_classes.Derived_Type_Def_Node(name=name, component_part=component_part,
                                                          procedure_part=procedure_part)

    def derived_type_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Type_Name_Node)
        return ast_internal_classes.Derived_Type_Stmt_Node(name=name)

    def component_part(self, node: FASTNode):
        children = self.create_children(node)
        component_def_stmts = [i for i in children if isinstance(i, ast_internal_classes.Data_Component_Def_Stmt_Node)]
        return ast_internal_classes.Component_Part_Node(component_def_stmts=component_def_stmts)

    def data_component_def_stmt(self, node: FASTNode):
        children = self.type_declaration_stmt(node)
        return ast_internal_classes.Data_Component_Def_Stmt_Node(vars=children)

    def component_decl_list(self, node: FASTNode):
        children = self.create_children(node)
        component_decls = [i for i in children if isinstance(i, ast_internal_classes.Component_Decl_Node)]
        return ast_internal_classes.Component_Decl_List_Node(component_decls=component_decls)

    def component_decl(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        return ast_internal_classes.Component_Decl_Node(name=name)

    def write_stmt(self, node: FASTNode):
        # children=[]
        # if node.children[0] is not None:
        #    children = self.create_children(node.children[0])
        # if node.children[1] is not None:
        #    children = self.create_children(node.children[1])    
        line = get_line(node)
        return ast_internal_classes.Write_Stmt_Node(args=node.string, line_number=line)

    def program(self, node: FASTNode):
        children = self.create_children(node)
        main_program = get_child(children, ast_internal_classes.Main_Program_Node)
        function_definitions = [i for i in children if isinstance(i, ast_internal_classes.Function_Subprogram_Node)]
        subroutine_definitions = [i for i in children if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node)]
        modules = [node for node in children if isinstance(node, ast_internal_classes.Module_Node)]
        return ast_internal_classes.Program_Node(main_program=main_program,
                                                 function_definitions=function_definitions,
                                                 subroutine_definitions=subroutine_definitions,
                                                 modules=modules,
                                                 module_declarations={})

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
        return ast_internal_classes.Program_Stmt_Node(name=name, line_number=get_line(node))

    def subroutine_subprogram(self, node: FASTNode):

        children = self.create_children(node)

        name = get_child(children, ast_internal_classes.Subroutine_Stmt_Node)
        specification_part = get_child(children, ast_internal_classes.Specification_Part_Node)
        execution_part = get_child(children, ast_internal_classes.Execution_Part_Node)
        internal_subprogram_part = get_child(children, ast_internal_classes.Internal_Subprogram_Part_Node)
        return_type = ast_internal_classes.Void

        optional_args_count = 0
        if specification_part is not None:
            for j in specification_part.specifications:
                for k in j.vardecl:
                    if k.optional:
                        optional_args_count += 1
        mandatory_args_count = len(name.args) - optional_args_count

        return ast_internal_classes.Subroutine_Subprogram_Node(
            name=name.name,
            args=name.args,
            optional_args_count=optional_args_count,
            mandatory_args_count=mandatory_args_count,
            specification_part=specification_part,
            execution_part=execution_part,
            internal_subprogram_part=internal_subprogram_part,
            type=return_type,
            line_number=name.line_number,
            elemental=name.elemental,

        )

    def end_program_stmt(self, node: FASTNode):
        return node

    def only_list(self, node: FASTNode):
        children = self.create_children(node)
        names = [i for i in children if isinstance(i, ast_internal_classes.Name_Node)]
        renames = [i for i in children if isinstance(i, ast_internal_classes.Rename_Node)]
        return ast_internal_classes.Only_List_Node(names=names, renames=renames)

    def prefix_stmt(self, prefix: Prefix):
        if 'recursive' in prefix.string.lower():
            print("recursive found")
        props: Dict[str, bool] = {
            'elemental': False,
            'recursive': False,
            'pure': False,
        }
        type = 'VOID'
        for c in prefix.children:
            if c.string.lower() in props.keys():
                props[c.string.lower()] = True
            elif isinstance(c, Intrinsic_Type_Spec):
                c_type = _find_typename_of_intrinsic_type(c)
                assert c_type in self.types, f"Unknown type {c.string}/{c_type} in prefix: {prefix}"
                type = self.types[c_type]
        return ast_internal_classes.Prefix_Node(type=type,
                                                elemental=props['elemental'],
                                                recursive=props['recursive'],
                                                pure=props['pure'])

    def function_subprogram(self, node: Function_Subprogram):
        children = self.create_children(node)

        specification_part = get_child(children, ast_internal_classes.Specification_Part_Node)
        execution_part = get_child(children, ast_internal_classes.Execution_Part_Node)
        internal_subprogram_part = get_child(children, ast_internal_classes.Internal_Subprogram_Part_Node)

        name = get_child(children, ast_internal_classes.Function_Stmt_Node)
        return_var: Name_Node = name.ret.name if name.ret else name.name
        return_type: str = name.type
        if name.type == 'VOID':
            assert specification_part
            var_decls: List[Var_Decl_Node] = [v
                                              for c in specification_part.specifications if
                                              isinstance(c, Decl_Stmt_Node)
                                              for v in c.vardecl]
            return_type = singular(v.type for v in var_decls if v.name == return_var.name)

        return ast_internal_classes.Function_Subprogram_Node(
            name=name.name,
            args=name.args,
            ret=return_var,
            specification_part=specification_part,
            execution_part=execution_part,
            internal_subprogram_part=internal_subprogram_part,
            type=return_type,
            line_number=name.line_number,
            elemental=name.elemental,
        )

    def function_stmt(self, node: Function_Stmt):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Arg_List_Node)
        prefix = get_child(children, ast_internal_classes.Prefix_Node)

        ftype, elemental = (prefix.type.upper(), prefix.elemental) if prefix else ('VOID', False)
        if prefix is not None and prefix.recursive:
            print("recursive found " + name.name)

        ret = get_child(children, ast_internal_classes.Suffix_Node)
        ret_args = args.args if args else []
        return ast_internal_classes.Function_Stmt_Node(
            name=name, args=ret_args, line_number=get_line(node), ret=ret, elemental=elemental,
            type=ret if ret else ftype)

    def subroutine_stmt(self, node: FASTNode):
        # print(self.name_list)
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Arg_List_Node)
        prefix = get_child(children, ast_internal_classes.Prefix_Node)
        elemental = prefix.elemental if prefix else False
        if prefix is not None and prefix.recursive:
            print("recursive found " + name.name)
        if args is None:
            ret_args = []
        else:
            ret_args = args.args
        return ast_internal_classes.Subroutine_Stmt_Node(name=name, args=ret_args, line_number=get_line(node),
                                                         elemental=elemental)

    def ac_value_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Ac_Value_List_Node(value_list=children)

    def power_expr(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        # child 0 is the base, child 2 is the exponent
        # child 1 is "**"
        return ast_internal_classes.Call_Expr_Node(name=self.intrinsic_handler.replace_function_name(ast_internal_classes.Name_Node(name="POW")),
                                           args=[children[0], children[2]],
                                           line_number=line, type="VOID", subroutine=False)
        #return ast_internal_classes.Call_Expr_Node(name=ast_internal_classes.Name_Node(name="__dace_POW"),
        #                                           args=[children[0], children[2]],
        #                                           line_number=line, type="REAL", subroutine=False)

    def array_constructor(self, node: FASTNode):
        children = self.create_children(node)
        value_list = get_child(children, ast_internal_classes.Ac_Value_List_Node)
        return ast_internal_classes.Array_Constructor_Node(value_list=value_list.value_list, type="VOID")

    def allocate_stmt(self, node: FASTNode):
        children = self.create_children(node)
        if isinstance(children[0], ast_internal_classes.Name_Node):
            print(children[0].name)
        if isinstance(children[0], ast_internal_classes.Data_Ref_Node):
            print(children[0].parent_ref.name + "." + children[0].part_ref.name)

        line = get_line(node)
        return ast_internal_classes.Allocate_Stmt_Node(name=children[0], allocation_list=children[1], line_number=line)

    def allocation_list(self, node: FASTNode):
        children = self.create_children(node)
        return children

    def allocation(self, node: FASTNode):
        children = self.create_children(node)
        name = children[0]
        # if isinstance(children[0], ast_internal_classes.Name_Node):
        #    print(children[0].name)
        # if isinstance(children[0], ast_internal_classes.Data_Ref_Node):
        #    print(children[0].parent_ref.name+"."+children[0].part_ref.name)
        shape = get_child(children, ast_internal_classes.Allocate_Shape_Spec_List)
        return ast_internal_classes.Allocation_Node(name=children[0], shape=shape)

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
        line = get_line(node)
        name = get_child(children, ast_internal_classes.Type_Name_Node)
        args = get_child(children, ast_internal_classes.Component_Spec_List_Node)
        if args == None:
            ret_args = []
        else:
            ret_args = args.args
        return ast_internal_classes.Structure_Constructor_Node(name=name, args=ret_args, type=None, line_number=line)

    def intrinsic_function_reference(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        args = get_child(children, ast_internal_classes.Arg_List_Node)

        if name is None:
            return Name_Node(name="Error! " + node.children[0].string, type='VOID')
        node = self.intrinsic_handler.replace_function_reference(name, args, line, self.symbols)
        return node

    def end_subroutine_stmt(self, node: FASTNode):
        return node

    def end_function_stmt(self, node: FASTNode):
        return node

    def parenthesis_expr(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Parenthesis_Expr_Node(expr=children[1])

    def module_subprogram_part(self, node: FASTNode):
        children = self.create_children(node)
        function_definitions = [i for i in children if isinstance(i, ast_internal_classes.Function_Subprogram_Node)]
        subroutine_definitions = [i for i in children if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node)]
        return ast_internal_classes.Module_Subprogram_Part_Node(function_definitions=function_definitions,
                                                                subroutine_definitions=subroutine_definitions)

    def internal_subprogram_part(self, node: FASTNode):
        children = self.create_children(node)
        function_definitions = [i for i in children if isinstance(i, ast_internal_classes.Function_Subprogram_Node)]
        subroutine_definitions = [i for i in children if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node)]
        return ast_internal_classes.Internal_Subprogram_Part_Node(function_definitions=function_definitions,
                                                                  subroutine_definitions=subroutine_definitions)

    def interface_block(self, node: FASTNode):
        children = self.create_children(node)

        name = get_child(children, ast_internal_classes.Interface_Stmt_Node)
        stmts = get_children(children, ast_internal_classes.Procedure_Statement_Node)
        subroutines = []

        for i in stmts:

            for child in i.namelists:
                subroutines.extend(child.subroutines)

        # Ignore other implementations of an interface block with overloaded procedures
        if name is None or len(subroutines) == 0:
            return node

        return ast_internal_classes.Interface_Block_Node(name=name.name, subroutines=subroutines)

    def module(self, node: FASTNode):
        children = self.create_children(node)

        name = get_child(children, ast_internal_classes.Module_Stmt_Node)
        module_subprogram_part = get_child(children, ast_internal_classes.Module_Subprogram_Part_Node)
        specification_part = get_child(children, ast_internal_classes.Specification_Part_Node)

        function_definitions = [i for i in children if isinstance(i, ast_internal_classes.Function_Subprogram_Node)]

        subroutine_definitions = [i for i in children if isinstance(i, ast_internal_classes.Subroutine_Subprogram_Node)]

        interface_blocks = {}
        if specification_part is not None:
            for iblock in specification_part.interface_blocks:
                interface_blocks[iblock.name] = [x.name for x in iblock.subroutines]

        # add here to definitions
        if module_subprogram_part is not None:
            for i in module_subprogram_part.function_definitions:
                function_definitions.append(i)
            for i in module_subprogram_part.subroutine_definitions:
                subroutine_definitions.append(i)

        return ast_internal_classes.Module_Node(
            name=name.name,
            specification_part=specification_part,
            function_definitions=function_definitions,
            subroutine_definitions=subroutine_definitions,
            interface_blocks=interface_blocks,
            line_number=name.line_number,
        )

    def module_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        return ast_internal_classes.Module_Stmt_Node(name=name, line_number=get_line(node))

    def end_module_stmt(self, node: FASTNode):
        return node

    def use_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        only_list = get_child(children, ast_internal_classes.Only_List_Node)
        if only_list is None:
            return ast_internal_classes.Use_Stmt_Node(name=name.name, list=[], list_all=True)
        return ast_internal_classes.Use_Stmt_Node(name=name.name, list=only_list.names, list_all=False)

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
        # raise NotImplementedError("Declaration type spec is not supported yet")
        return node

    def assumed_shape_spec_list(self, node: FASTNode):
        return node

    def parse_shape_specification(self, dim: f03.Explicit_Shape_Spec, size: List[FASTNode], offset: List[int]):

        dim_expr = [i for i in dim.children if i is not None]

        # handle size definition
        if len(dim_expr) == 1:
            dim_expr = dim_expr[0]
            # now to add the dimension to the size list after processing it if necessary
            size.append(self.create_ast(dim_expr))
            offset.append(1)

        # Here we support arrays that have size declaration - with initial offset.
        elif len(dim_expr) == 2:
            # extract offets
            if isinstance(dim_expr[0], f03.Int_Literal_Constant):
                # raise TypeError("Array offsets must be constant expressions!")
                offset.append(int(dim_expr[0].tostr()))
            else:
                expr = self.create_ast(dim_expr[0])
                offset.append(expr)

            fortran_size = ast_internal_classes.BinOp_Node(
                lval=self.create_ast(dim_expr[1]),
                rval=self.create_ast(dim_expr[0]),
                op="-",
                type="INTEGER"
            )
            size.append(ast_internal_classes.BinOp_Node(
                lval=fortran_size,
                rval=ast_internal_classes.Int_Literal_Node(value=str(1)),
                op="+",
                type="INTEGER")
            )
        else:
            raise TypeError("Array dimension must be at most two expressions")

    def assumed_array_shape(self, var, array_name: Optional[str], linenumber):

        # We do not know the array size. Thus, we insert symbols
        # to mark its size
        shape = get_children(var, "Assumed_Shape_Spec_List")

        if shape is None or len(shape) == 0:
            shape = get_children(var, "Deferred_Shape_Spec_List")

            if shape is None:
                return None, []

        # this is based on structures observed in Fortran codes
        # I don't know why the shape is an array
        if len(shape) > 0:
            dims_count = len(shape[0].items)
            size = []
            vardecls = []

            processed_array_names = []
            if array_name is not None:
                if isinstance(array_name, str):
                    processed_array_names = [array_name]
                else:
                    processed_array_names = [j.children[0].string for j in array_name]
            else:
                raise NotImplementedError("Assumed array shape not supported yet if array name missing")

            sizes = []
            offsets = []
            for actual_array in processed_array_names:

                size = []
                offset = []
                for i in range(dims_count):
                    name = f'__f2dace_A_{actual_array}_d_{i}_s_{self.type_arbitrary_array_variable_count}'
                    offset_name = f'__f2dace_OA_{actual_array}_d_{i}_s_{self.type_arbitrary_array_variable_count}'
                    self.type_arbitrary_array_variable_count += 1
                    self.placeholders[name] = [actual_array, i, self.type_arbitrary_array_variable_count]
                    self.placeholders_offsets[name] = [actual_array, i, self.type_arbitrary_array_variable_count]

                    var = ast_internal_classes.Symbol_Decl_Node(name=name,
                                                                type='INTEGER',
                                                                alloc=False,
                                                                sizes=None,
                                                                offsets=None,
                                                                init=None,
                                                                kind=None,
                                                                line_number=linenumber)
                    var2 = ast_internal_classes.Symbol_Decl_Node(name=offset_name,
                                                                 type='INTEGER',
                                                                 alloc=False,
                                                                 sizes=None,
                                                                 offsets=None,
                                                                 init=None,
                                                                 kind=None,
                                                                 line_number=linenumber)
                    size.append(ast_internal_classes.Name_Node(name=name))
                    offset.append(ast_internal_classes.Name_Node(name=offset_name))

                    self.symbols[name] = None
                    vardecls.append(var)
                    vardecls.append(var2)
                sizes.append(size)
                offsets.append(offset)

            return sizes, vardecls, offsets
        else:
            return None, [], None

    def type_declaration_stmt(self, node: FASTNode):

        # decide if it's an intrinsic variable type or a derived type

        type_of_node = get_child(node, [f03.Intrinsic_Type_Spec, f03.Declaration_Type_Spec])
        # if node.children[2].children[0].children[0].string.lower() =="BOUNDARY_MISSVAL".lower():
        #    print("found boundary missval")
        if isinstance(type_of_node, f03.Intrinsic_Type_Spec):
            derived_type = False
            basetype = type_of_node.items[0]
        elif isinstance(type_of_node, f03.Declaration_Type_Spec):
            if type_of_node.items[0].lower() == "class":
                basetype = "CLASS"
                basetype = type_of_node.items[1].string
                derived_type = True
            else:
                derived_type = True
                basetype = type_of_node.items[1].string
        else:
            raise TypeError("Type of node must be either Intrinsic_Type_Spec or Declaration_Type_Spec")
        kind = None
        size_later = False
        if len(type_of_node.items) >= 2:
            if type_of_node.items[1] is not None:
                if not derived_type:
                    if basetype == "CLASS":
                        kind = "CLASS"
                    elif basetype == "CHARACTER":
                        kind = type_of_node.items[1].items[1].string.lower()
                        if kind == "*":
                            size_later = True
                    else:
                        if isinstance(type_of_node.items[1].items[1], f03.Int_Literal_Constant):
                            kind = type_of_node.items[1].items[1].string.lower()
                            if basetype == "REAL":
                                if kind == "8":
                                    basetype = "REAL8"
                                else:
                                    raise TypeError("Real kind not supported")
                            elif basetype == "INTEGER":
                                if kind == "4":
                                    basetype = "INTEGER"
                                elif kind == "1":
                                    # TODO: support for 1 byte integers /chars would be useful
                                    basetype = "INTEGER"

                                elif kind == "2":
                                    # TODO: support for 2 byte integers would be useful
                                    basetype = "INTEGER"

                                elif kind == "8":
                                    # TODO: support for 8 byte integers would be useful
                                    basetype = "INTEGER"
                                else:
                                    raise TypeError("Integer kind not supported")
                            else:
                                raise TypeError("Derived type not supported")

                        else:
                            kind = type_of_node.items[1].items[1].string.lower()
                            if self.symbols[kind] is not None:
                                if basetype == "REAL":
                                    while hasattr(self.symbols[kind], "name"):
                                        kind = self.symbols[kind].name
                                    if self.symbols[kind].value == "8":
                                        basetype = "REAL8"
                                elif basetype == "INTEGER":
                                    while hasattr(self.symbols[kind], "name"):
                                        kind = self.symbols[kind].name
                                    if self.symbols[kind].value == "4":
                                        basetype = "INTEGER"
                                else:
                                    raise TypeError("Derived type not supported")

                # if derived_type:
                #    raise TypeError("Derived type not supported")
        if not derived_type:
            testtype = self.types[basetype]
        else:

            testtype = basetype

        # get the names of the variables being defined
        names_list = get_child(node, ["Entity_Decl_List", "Component_Decl_List"])

        # get the names out of the name list
        names = get_children(names_list, [f03.Entity_Decl, f03.Component_Decl])

        # get the attributes of the variables being defined
        # alloc relates to whether it is statically (False) or dynamically (True) allocated
        # parameter means it's a constant, so we should transform it into a symbol
        attributes = get_children(node, "Attr_Spec_List")
        comp_attributes = get_children(node, "Component_Attr_Spec_List")
        if len(attributes) != 0 and len(comp_attributes) != 0:
            raise TypeError("Attributes must be either in Attr_Spec_List or Component_Attr_Spec_List not both")

        alloc = False
        symbol = False
        optional = False
        attr_size = None
        attr_offset = None
        assumed_vardecls = []
        for i in attributes + comp_attributes:

            if i.string.lower() == "allocatable":
                alloc = True
            if i.string.lower() == "parameter":
                symbol = True
            if i.string.lower() == "pointer":
                alloc = True
            if i.string.lower() == "optional":
                optional = True

            if isinstance(i, f08.Attr_Spec_List):

                specification = get_children(i, "Attr_Spec")
                for spec in specification:
                    if spec.string.lower() == "optional":
                        optional = True
                    if spec.string.lower() == "allocatable":
                        alloc = True

                dimension_spec = get_children(i, "Dimension_Attr_Spec")
                if len(dimension_spec) == 0:
                    continue

                attr_size = []
                attr_offset = []
                sizes = get_child(dimension_spec[0], ["Explicit_Shape_Spec_List"])

                if sizes is not None:
                    for shape_spec in get_children(sizes, [f03.Explicit_Shape_Spec]):
                        self.parse_shape_specification(shape_spec, attr_size, attr_offset)
                    # we expect a list of lists, where each element correspond to list of symbols for each array name
                    attr_size = [attr_size] * len(names)
                    attr_offset = [attr_offset] * len(names)
                else:
                    attr_size, assumed_vardecls, attr_offset = self.assumed_array_shape(dimension_spec[0], names,
                                                                                        get_line(node))

                    if attr_size is None:
                        raise RuntimeError("Couldn't parse the dimension attribute specification!")

            if isinstance(i, f08.Component_Attr_Spec_List):

                specification = get_children(i, "Component_Attr_Spec")
                for spec in specification:
                    if spec.string.lower() == "optional":
                        optional = True
                    if spec.string.lower() == "allocatable":
                        alloc = True

                dimension_spec = get_children(i, "Dimension_Component_Attr_Spec")
                if len(dimension_spec) == 0:
                    continue

                attr_size = []
                attr_offset = []
                sizes = get_child(dimension_spec[0], ["Explicit_Shape_Spec_List"])
                # if sizes is None:
                #    sizes = get_child(dimension_spec[0], ["Deferred_Shape_Spec_List"])

                if sizes is not None:
                    for shape_spec in get_children(sizes, [f03.Explicit_Shape_Spec]):
                        self.parse_shape_specification(shape_spec, attr_size, attr_offset)
                    # we expect a list of lists, where each element correspond to list of symbols for each array name
                    attr_size = [attr_size] * len(names)
                    attr_offset = [attr_offset] * len(names)
                else:
                    attr_size, assumed_vardecls, attr_offset = self.assumed_array_shape(dimension_spec[0], names,
                                                                                        get_line(node))
                    if attr_size is None:
                        raise RuntimeError("Couldn't parse the dimension attribute specification!")

        vardecls = [*assumed_vardecls]

        for idx, var in enumerate(names):
            # print(self.name_list)
            # first handle dimensions
            size = None
            offset = None
            var_components = self.create_children(var)
            array_sizes = get_children(var, "Explicit_Shape_Spec_List")
            actual_name = get_child(var_components, ast_internal_classes.Name_Node)
            # if actual_name.name not in self.name_list:
            #    return
            if len(array_sizes) == 1:
                array_sizes = array_sizes[0]
                size = []
                offset = []
                for dim in array_sizes.children:
                    # sanity check
                    if isinstance(dim, f03.Explicit_Shape_Spec):
                        self.parse_shape_specification(dim, size, offset)

            # handle initializiation
            init = None

            initialization = get_children(var, f03.Initialization)
            if len(initialization) == 1:
                initialization = initialization[0]
                # if there is an initialization, the actual expression is in the second child, with the first being the equals sign
                if len(initialization.children) < 2:
                    raise ValueError("Initialization must have an expression")
                raw_init = initialization.children[1]
                init = self.create_ast(raw_init)
            else:
                comp_init = get_children(var, "Component_Initialization")
                if len(comp_init) == 1:
                    raw_init = comp_init[0].children[1]
                    init = self.create_ast(raw_init)
            # if size_later:
            #    size.append(len(init))
            if testtype != "INTEGER": symbol = False
            if symbol == False:

                if attr_size is None:

                    if size is None:

                        size, assumed_vardecls, offset = self.assumed_array_shape(var, actual_name.name, get_line(node))
                        if size is None:
                            size = []
                            offset = [1]
                        else:
                            # only one array
                            size = size[0]
                            offset = offset[0]
                            # offset = [1] * len(size)
                        vardecls.extend(assumed_vardecls)

                    vardecls.append(
                        ast_internal_classes.Var_Decl_Node(name=actual_name.name,
                                                           type=testtype,
                                                           alloc=alloc,
                                                           sizes=size,
                                                           offsets=offset,
                                                           kind=kind,
                                                           init=init,
                                                           optional=optional,
                                                           line_number=get_line(node)))
                else:
                    vardecls.append(
                        ast_internal_classes.Var_Decl_Node(name=actual_name.name,
                                                           type=testtype,
                                                           alloc=alloc,
                                                           sizes=attr_size[idx],
                                                           offsets=attr_offset[idx],
                                                           kind=kind,
                                                           init=init,
                                                           optional=optional,
                                                           line_number=get_line(node)))
            else:
                if size is None and attr_size is None:
                    self.symbols[actual_name.name] = init
                    vardecls.append(
                        ast_internal_classes.Symbol_Decl_Node(name=actual_name.name,
                                                              type=testtype,
                                                              sizes=[],
                                                              offsets=[1],
                                                              alloc=alloc,
                                                              init=init,
                                                              optional=optional))
                elif attr_size is not None:
                    vardecls.append(
                        ast_internal_classes.Symbol_Array_Decl_Node(name=actual_name.name,
                                                                    type=testtype,
                                                                    alloc=alloc,
                                                                    sizes=attr_size,
                                                                    offsets=attr_offset,
                                                                    kind=kind,
                                                                    init=init,
                                                                    optional=optional,
                                                                    line_number=get_line(node)))
                else:
                    vardecls.append(
                        ast_internal_classes.Symbol_Array_Decl_Node(name=actual_name.name,
                                                                    type=testtype,
                                                                    alloc=alloc,
                                                                    sizes=size,
                                                                    offsets=offset,
                                                                    kind=kind,
                                                                    init=init,
                                                                    optional=optional,
                                                                    line_number=get_line(node)))
        return ast_internal_classes.Decl_Stmt_Node(vardecl=vardecls)

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
        print("access spec. Fix me")
        # raise NotImplementedError("Access spec is not supported yet")
        return node

    def allocatable_stmt(self, node: FASTNode):
        raise NotImplementedError("Allocatable stmt is not supported yet")
        return node

    def asynchronous_stmt(self, node: FASTNode):
        raise NotImplementedError("Asynchronous stmt is not supported yet")
        return node

    def bind_stmt(self, node: FASTNode):
        print("bind stmt. Fix me")
        # raise NotImplementedError("Bind stmt is not supported yet")
        return node

    def common_stmt(self, node: FASTNode):
        raise NotImplementedError("Common stmt is not supported yet")
        return node

    def data_stmt(self, node: FASTNode):
        print("data stmt! fix me!")
        # raise NotImplementedError("Data stmt is not supported yet")
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
        raise NotImplementedError("Pointer stmt is not supported yet")
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
        children = [child for child in self.create_children(node) if child is not None]
        return ast_internal_classes.Execution_Part_Node(execution=children)

    def execution_part_construct(self, node: FASTNode):
        return node

    def action_stmt(self, node: FASTNode):
        return node

    def level_2_expr(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        if children[1] == "==":
            type = "LOGICAL"
        else:
            type = "VOID"
            if hasattr(children[0], "type"):
                type = children[0].type
        if len(children) == 3:
            return ast_internal_classes.BinOp_Node(lval=children[0], op=children[1], rval=children[2], line_number=line,
                                                   type=type)
        else:
            return ast_internal_classes.UnOp_Node(lval=children[1], op=children[0], line_number=line,
                                                  type=children[1].type)

    def assignment_stmt(self, node: Assignment_Stmt):
        children = self.create_children(node)
        line = get_line(node)

        if len(children) == 3:
            return ast_internal_classes.BinOp_Node(lval=children[0], op=children[1], rval=children[2], line_number=line,
                                                   type=children[0].type)
        else:
            return ast_internal_classes.UnOp_Node(lval=children[1], op=children[0], line_number=line,
                                                  type=children[1].type)

    def pointer_assignment_stmt(self, node: FASTNode):
        children = self.create_children(node)
        line = get_line(node)
        return ast_internal_classes.Pointer_Assignment_Stmt_Node(name_pointer=children[0],
                                                                 name_target=children[2],
                                                                 line_number=line)

    def where_stmt(self, node: FASTNode):
        return node

    def where_construct(self, node: FASTNode):
        children = self.create_children(node)
        line = children[0].line_number
        cond = children[0]
        body = children[1]
        current = 2
        body_else = None
        elifs_cond = []
        elifs_body = []
        while children[current] is not None:
            if isinstance(children[current], str) and children[current].lower() == "elsewhere":
                body_else = children[current + 1]
                current += 2
            else:
                elifs_cond.append(children[current])
                elifs_body.append(children[current + 1])
                current += 2
        return ast_internal_classes.Where_Construct_Node(body=body, cond=cond, body_else=body_else,
                                                         elifs_cond=elifs_cond, elifs_body=elifs_cond, line_number=line)

    def where_construct_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return children[0]

    def masked_elsewhere_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return children[0]

    def elsewhere_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return children[0]

    def end_where_stmt(self, node: FASTNode):
        return None

    def forall_stmt(self, node: FASTNode):
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
        # !THIS IS HACK
        body = [i for i in body if i is not None]
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
            if i is None:
                continue
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
        return_value.line_number = get_line(node)
        return return_value

    def else_if_stmt(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Else_If_Stmt_Node(cond=children[0], line_number=get_line(node))

    def else_stmt(self, node: FASTNode):
        return ast_internal_classes.Else_Separator_Node(line_number=get_line(node))

    def end_if_stmt(self, node: FASTNode):
        return node

    def case_construct(self, node: FASTNode):
        children = self.create_children(node)
        cond_start = children[0]
        cond_end = children[1]
        body = []
        body_else = []
        else_mode = False
        line = get_line(node)
        if line is None:
            line = "Unknown:TODO"
        cond = ast_internal_classes.BinOp_Node(op=cond_end.op[0], lval=cond_start, rval=cond_end.cond[0],
                                               line_number=line)
        for j in range(1, len(cond_end.op)):
            cond_add = ast_internal_classes.BinOp_Node(op=cond_end.op[j], lval=cond_start, rval=cond_end.cond[j],
                                                       line_number=line)
            cond = ast_internal_classes.BinOp_Node(op=".OR.", lval=cond, rval=cond_add, line_number=line)

        toplevelIf = ast_internal_classes.If_Stmt_Node(cond=cond, line_number=line)
        currentIf = toplevelIf
        for i in children[2:-1]:
            if i is None:
                continue
            if isinstance(i, ast_internal_classes.Case_Cond_Node):
                cond = ast_internal_classes.BinOp_Node(op=i.op[0], lval=cond_start, rval=i.cond[0], line_number=line)
                for j in range(1, len(i.op)):
                    cond_add = ast_internal_classes.BinOp_Node(op=i.op[j], lval=cond_start, rval=i.cond[j],
                                                               line_number=line)
                    cond = ast_internal_classes.BinOp_Node(op=".OR.", lval=cond, rval=cond_add, line_number=line)

                newif = ast_internal_classes.If_Stmt_Node(cond=cond, line_number=line)
                currentIf.body = ast_internal_classes.Execution_Part_Node(execution=body)
                currentIf.body_else = ast_internal_classes.Execution_Part_Node(execution=[newif])
                currentIf = newif
                body = []
                continue
            if isinstance(i, str) and i == "__default__":
                else_mode = True
                continue
            if else_mode:
                body_else.append(i)
            else:

                body.append(i)
        currentIf.body = ast_internal_classes.Execution_Part_Node(execution=body)
        currentIf.body_else = ast_internal_classes.Execution_Part_Node(execution=body_else)
        return toplevelIf

    def select_case_stmt(self, node: FASTNode):
        children = self.create_children(node)
        if len(children) != 1:
            raise ValueError("CASE should have only 1 child")
        return children[0]

    def case_stmt(self, node: FASTNode):
        children = self.create_children(node)
        children = [i for i in children if i is not None]
        if len(children) == 1:
            return children[0]
        elif len(children) == 0:
            return "__default__"
        else:
            raise ValueError("Can't parse case statement")

    def case_selector(self, node: FASTNode):
        children = self.create_children(node)
        if len(children) == 1:
            if children[0] is None:
                return None
            returns = ast_internal_classes.Case_Cond_Node(op=[], cond=[])

            for i in children[0]:
                returns.op.append(i[0])
                returns.cond.append(i[1])
            return returns
        else:
            raise ValueError("Can't parse case selector")

    def case_value_range_list(self, node: FASTNode):
        children = self.create_children(node)
        if len(children) == 1:
            return [[".EQ.", children[0]]]
        if len(children) == 2:
            return [[".EQ.", children[0]], [".EQ.", children[1]]]
        else:
            retlist = []
            for i in children:
                retlist.append([".EQ.", i])
            return retlist
            # else:
        #    raise ValueError("Can't parse case range list")

    def end_select_stmt(self, node: FASTNode):
        return node

    def do_construct(self, node: FASTNode):
        return node

    def label_do_stmt(self, node: FASTNode):
        return node

    def nonlabel_do_stmt(self, node: FASTNode):
        children = self.create_children(node)
        loop_control = get_child(children, ast_internal_classes.Loop_Control_Node)
        if loop_control is None:
            if node.string == "DO":
                return ast_internal_classes.While_True_Control(name=node.item.name, line_number=get_line(node))
            else:
                while_control = get_child(children, ast_internal_classes.While_Control)
                return ast_internal_classes.While_Control(cond=while_control.cond, line_number=get_line(node))
        return ast_internal_classes.Nonlabel_Do_Stmt_Node(iter=loop_control.iter,
                                                          cond=loop_control.cond,
                                                          init=loop_control.init,
                                                          line_number=get_line(node))

    def end_do_stmt(self, node: FASTNode):
        return node

    def interface_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        if name is not None:
            return ast_internal_classes.Interface_Stmt_Node(name=name.name)
        else:
            return node

    def end_interface_stmt(self, node: FASTNode):
        return node

    def procedure_name_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Procedure_Name_List_Node(subroutines=children)

    def procedure_stmt(self, node: FASTNode):
        # ignore the procedure statement - just return the name list
        children = self.create_children(node)
        namelists = get_children(children, ast_internal_classes.Procedure_Name_List_Node)
        if namelists is not None:
            return ast_internal_classes.Procedure_Statement_Node(namelists=namelists)
        else:
            return node

    def generic_spec(self, node: FASTNode):
        children = self.create_children(node)
        return node

    def procedure_declaration_stmt(self, node: FASTNode):
        return node

    def specific_binding(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Specific_Binding_Node(name=children[3], args=children[0:2] + [children[4]])

    def type_bound_procedure_part(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Bound_Procedures_Node(procedures=children[1:])

    def contains_stmt(self, node: FASTNode):
        return node

    def call_stmt(self, node: FASTNode):
        children = self.create_children(node)
        name = get_child(children, ast_internal_classes.Name_Node)
        arg_addition = None
        if name is None:
            proc_ref = get_child(children, ast_internal_classes.Procedure_Separator_Node)
            name = proc_ref.part_ref
            arg_addition = proc_ref.parent_ref

        args = get_child(children, ast_internal_classes.Arg_List_Node)
        if args is None:
            ret_args = []
        else:
            ret_args = args.args
        if arg_addition is not None:
            ret_args.insert(0, arg_addition)
        line_number = get_line(node)
        # if node.item is None:
        #    line_number = 42
        # else:
        #    line_number = get_line(node)
        return ast_internal_classes.Call_Expr_Node(name=name, args=ret_args, type="VOID", subroutine=True,
                                                   line_number=line_number)

    def return_stmt(self, node: FASTNode):
        return None

    def stop_stmt(self, node: FASTNode):
        return ast_internal_classes.Call_Expr_Node(name=ast_internal_classes.Name_Node(name="__dace_exit"), args=[],
                                                   type="VOID", subroutine=False, line_number=get_line(node))

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
        return ast_internal_classes.Array_Subscript_Node(name=name, type="VOID", indices=args.list,
                                                         line_number=line)

    def loop_control(self, node: FASTNode):
        children = self.create_children(node)
        # Structure of loop control is:
        if children[1] is None:
            return ast_internal_classes.While_Control(cond=children[0], line_number=get_line(node.parent))
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
        init_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable, op="=", rval=loop_start, type="INTEGER")
        if isinstance(loop_step, ast_internal_classes.UnOp_Node):
            if loop_step.op == "-":
                cond_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable, op=">=", rval=loop_end,
                                                            type="INTEGER")
        else:
            cond_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable, op="<=", rval=loop_end, type="INTEGER")
        iter_expr = ast_internal_classes.BinOp_Node(lval=iteration_variable,
                                                    op="=",
                                                    rval=ast_internal_classes.BinOp_Node(lval=iteration_variable,
                                                                                         op="+",
                                                                                         rval=loop_step,
                                                                                         type="INTEGER"),
                                                    type="INTEGER")
        return ast_internal_classes.Loop_Control_Node(init=init_expr, cond=cond_expr, iter=iter_expr)

    def block_nonlabel_do_construct(self, node: FASTNode):
        children = self.create_children(node)
        do = get_child(children, ast_internal_classes.Nonlabel_Do_Stmt_Node)
        body = children[1:-1]
        body = [i for i in body if i is not None]
        if do is None:
            while_true_header = get_child(children, ast_internal_classes.While_True_Control)
            if while_true_header is not None:
                return ast_internal_classes.While_Stmt_Node(name=while_true_header.name,
                                                            body=ast_internal_classes.Execution_Part_Node(
                                                                execution=body),
                                                            line_number=while_true_header.line_number)
            while_header = get_child(children, ast_internal_classes.While_Control)
            if while_header is not None:
                return ast_internal_classes.While_Stmt_Node(cond=while_header.cond,
                                                            body=ast_internal_classes.Execution_Part_Node(
                                                                execution=body),
                                                            line_number=while_header.line_number)
        return ast_internal_classes.For_Stmt_Node(init=do.init,
                                                  cond=do.cond,
                                                  iter=do.iter,
                                                  body=ast_internal_classes.Execution_Part_Node(execution=body),
                                                  line_number=do.line_number)

    def subscript_triplet(self, node: FASTNode):
        if node.string == ":":
            return ast_internal_classes.ParDecl_Node(type="ALL")
        children = self.create_children(node)
        return ast_internal_classes.ParDecl_Node(type="RANGE", range=children)

    def section_subscript_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Section_Subscript_List_Node(list=children)

    def specification_part(self, node: FASTNode):

        # TODO this can be refactored to consider more fortran declaration options. Currently limited to what is encountered in code.
        others = [self.create_ast(i) for i in node.children if not isinstance(i, f08.Type_Declaration_Stmt)]

        decls = [self.create_ast(i) for i in node.children if isinstance(i, f08.Type_Declaration_Stmt)]
        enums = [self.create_ast(i) for i in node.children if isinstance(i, f03.Enum_Def)]
        # decls = list(filter(lambda x: x is not None, decls))
        uses = [self.create_ast(i) for i in node.children if isinstance(i, f03.Use_Stmt)]
        tmp = [self.create_ast(i) for i in node.children]
        typedecls = [
            i for i in tmp if isinstance(i, ast_internal_classes.Type_Decl_Node)
                              or isinstance(i, ast_internal_classes.Derived_Type_Def_Node)
        ]
        symbols = []
        iblocks = []
        for i in others:
            if isinstance(i, list):
                symbols.extend(j for j in i if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
            if isinstance(i, ast_internal_classes.Decl_Stmt_Node):
                symbols.extend(j for j in i.vardecl if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
            if isinstance(i, ast_internal_classes.Interface_Block_Node):
                iblocks.append(i)

        for i in decls:
            if isinstance(i, list):
                symbols.extend(j for j in i if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
                symbols.extend(j for j in i if isinstance(j, ast_internal_classes.Symbol_Decl_Node))
            if isinstance(i, ast_internal_classes.Decl_Stmt_Node):
                symbols.extend(j for j in i.vardecl if isinstance(j, ast_internal_classes.Symbol_Array_Decl_Node))
                symbols.extend(j for j in i.vardecl if isinstance(j, ast_internal_classes.Symbol_Decl_Node))
        names_filtered = []
        for j in symbols:
            for i in decls:
                names_filtered.extend(ii.name for ii in i.vardecl if j.name == ii.name)
        decl_filtered = []

        for i in decls:
            if i is None:
                continue
            # NOTE: Assignment/named expressions (walrus operator) works with Python 3.8 and later.
            # if vardecl_filtered := [ii for ii in i.vardecl if ii.name not in names_filtered]:
            vardecl_filtered = [ii for ii in i.vardecl if ii.name not in names_filtered]
            if vardecl_filtered:
                decl_filtered.append(ast_internal_classes.Decl_Stmt_Node(vardecl=vardecl_filtered))
        return ast_internal_classes.Specification_Part_Node(specifications=decl_filtered,
                                                            symbols=symbols,
                                                            interface_blocks=iblocks,
                                                            uses=uses,
                                                            typedecls=typedecls,
                                                            enums=enums)

    def intrinsic_type_spec(self, node: FASTNode):
        return node

    def entity_decl_list(self, node: FASTNode):
        return node

    def int_literal_constant(self, node: Union[Int_Literal_Constant, Signed_Int_Literal_Constant]):
        value = node.string
        if value.find("_") != -1:
            x = value.split("_")
            value = x[0]
        return ast_internal_classes.Int_Literal_Node(value=value, type="INTEGER")

    def hex_constant(self, node: Hex_Constant):
        return ast_internal_classes.Int_Literal_Node(value=str(int(node.string[2:-1], 16)), type="INTEGER")

    def logical_literal_constant(self, node: Logical_Literal_Constant):
        if node.string in [".TRUE.", ".true.", ".True."]:
            return ast_internal_classes.Bool_Literal_Node(value="1")
        if node.string in [".FALSE.", ".false.", ".False."]:
            return ast_internal_classes.Bool_Literal_Node(value="0")
        raise ValueError("Unknown logical literal constant")

    def real_literal_constant(self, node: Union[Real_Literal_Constant, Signed_Real_Literal_Constant]):
        value = node.children[0].lower()
        if len(node.children) == 2 and node.children[1] is not None and node.children[1].lower() == "wp":
            return ast_internal_classes.Double_Literal_Node(value=value, type="DOUBLE")
        if value.find("_") != -1:
            x = value.split("_")
            value = x[0]
            print(x[1])
            # FIXME: This depends on custom type `wp` defined only for ICON.
            if x[1] == "wp":
                return ast_internal_classes.Double_Literal_Node(value=value, type="DOUBLE")
        if value.endswith("d0"):
            return ast_internal_classes.Double_Literal_Node(value=value.split("d0")[0], type="DOUBLE")
        return ast_internal_classes.Real_Literal_Node(value=value, type="REAL")

    def char_literal_constant(self, node: FASTNode):
        return ast_internal_classes.Char_Literal_Node(value=node.string, type="CHAR")

    def actual_arg_spec(self, node: FASTNode):
        children = self.create_children(node)
        if len(children) != 2:
            raise ValueError("Actual arg spec must have two children")
        return ast_internal_classes.Actual_Arg_Spec_Node(arg_name=children[0], arg=children[1], type="VOID")

    def actual_arg_spec_list(self, node: FASTNode):
        children = self.create_children(node)
        return ast_internal_classes.Arg_List_Node(args=children)

    def initialization(self, node: FASTNode):
        return node

    def name(self, node: FASTNode):
        return ast_internal_classes.Name_Node(name=node.string.lower(), type="VOID")

    def rename(self, node: FASTNode):
        return ast_internal_classes.Rename_Node(oldname=node.children[2].string.lower(),
                                                newname=node.children[1].string.lower())

    def type_name(self, node: FASTNode):
        return ast_internal_classes.Type_Name_Node(name=node.string.lower())

    def tuple_node(self, node: FASTNode):
        return node

    def str_node(self, node: FASTNode):
        return node

def _find_typename_of_intrinsic_type(typ: Intrinsic_Type_Spec) -> str:
    ACCEPTED_TYPES = {'INTEGER', 'REAL', 'DOUBLE PRECISION', 'LOGICAL', 'CHARACTER'}
    typ_name, kind = typ.children
    assert typ_name in ACCEPTED_TYPES, typ_name

    # TODO: How should we handle character lengths? Just treat it as an extra dimension?
    if isinstance(kind, Length_Selector):
        assert typ_name == 'CHARACTER'
    elif isinstance(kind, Kind_Selector):
        assert typ_name in {'INTEGER', 'REAL', 'LOGICAL'}
        _, kind, _ = kind.children
        assert isinstance(kind, Int_Literal_Constant), f"{kind} must be a integer literal constant"
        num, kkind = kind.children
        assert kkind is None, f"{kind} must have its 'kind' resolved already, found {kkind}"
        typ_name = f"{typ_name}{num}"
    elif kind is None:
        if typ_name in {'INTEGER', 'REAL'}:
            typ_name = f"{typ_name}4"
        elif typ_name in {'DOUBLE PRECISION'}:
            typ_name = f"REAL8"
    return typ_name
