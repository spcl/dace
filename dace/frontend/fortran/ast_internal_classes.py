# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union, overload

# The node class is the base class for all nodes in the AST. It provides attributes including the line number and fields.
# Attributes are not used when walking the tree, but are useful for debugging and for code generation.
# The fields attribute is a list of the names of the attributes that are children of the node.


class FNode(object):
    def __init__(self, *args, **kwargs):  # real signature unknown
        self.integrity_exceptions = []
        self.read_vars = []
        self.written_vars = []
        self.parent: Optional[
            Union[
                Subroutine_Subprogram_Node,
                Function_Subprogram_Node,
                Main_Program_Node,
                Module_Node
            ]
        ] = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    _attributes = ("line_number", )
    _fields = ()
    integrity_exceptions: List
    read_vars: List
    written_vars: List

    def __eq__(self, o: object) -> bool:
        if type(self) is type(o):
            # check that all fields and attributes match
            self_field_vals = list(map(lambda name: getattr(self, name, None), self._fields))
            self_attr_vals = list(map(lambda name: getattr(self, name, None), self._attributes))
            o_field_vals = list(map(lambda name: getattr(o, name, None), o._fields))
            o_attr_vals = list(map(lambda name: getattr(o, name, None), o._attributes))

            return self_field_vals == o_field_vals and self_attr_vals == o_attr_vals
        return False


class Program_Node(FNode):
    _attributes = ()
    _fields = (
        "main_program",
        "function_definitions",
        "subroutine_definitions",
        "modules",
    )


class BinOp_Node(FNode):
    _attributes = (
        'op',
        'type',
    )
    _fields = (
        'lval',
        'rval',
    )


class UnOp_Node(FNode):
    _attributes = (
        'op',
        'postfix',
        'type',
    )
    _fields = ('lval', )


class Main_Program_Node(FNode):
    _attributes = ("name", )
    _fields = ("execution_part", "specification_part")


class Module_Node(FNode):
    _attributes = ('name', )
    _fields = (
        'specification_part',
        'subroutine_definitions',
        'function_definitions',
        'interface_blocks'
    )

class Module_Subprogram_Part_Node(FNode):
    _attributes = ()
    _fields = (
        'subroutine_definitions',
        'function_definitions',
    )

class Actual_Arg_Spec_Node(FNode):
    _fields = (
        'arg_name'
        'arg',
    )

class Function_Subprogram_Node(FNode):
    _attributes = ('name', 'type', 'ret_name')
    _fields = (
        'args',
        'specification_part',
        'execution_part',
    )


class Subroutine_Subprogram_Node(FNode):
    _attributes = ('name', 'type')
    _fields = (
        'args',
        'mandatory_args_count',
        'optional_args_count',
        'specification_part',
        'execution_part',
    )

class Interface_Block_Node(FNode):
    _attributes = ('name')
    _fields = (
        'subroutines',
    )

class Interface_Stmt_Node(FNode):
    _attributes = ()
    _fields = ()

class Procedure_Name_List_Node(FNode):
    _attributes = ()
    _fields = ('subroutines')

class Procedure_Statement_Node(FNode):
    _attributes = ()
    _fields = ('namelists')

class Module_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('functions')

class Program_Stmt_Node(FNode):
    _attributes = ('name', )
    _fields = ()


class Subroutine_Stmt_Node(FNode):
    _attributes = ('name', )
    _fields = ('args', )


class Function_Stmt_Node(FNode):
    _attributes = ('name', )
    _fields = ('args', 'ret')

class Prefix_Node(FNode):
    _attributes = ('elemental', )
    _fields = ()

class Name_Node(FNode):
    _attributes = ('name', 'type')
    _fields = ()


class Name_Range_Node(FNode):
    _attributes = ('name', 'type', 'arrname', 'pos')
    _fields = ()


class Type_Name_Node(FNode):
    _attributes = ('name', 'type')
    _fields = ()


class Specification_Part_Node(FNode):
    _fields = ('specifications', 'symbols', 'interface_blocks', 'typedecls')


class Execution_Part_Node(FNode):
    _fields = ('execution', )


class Statement_Node(FNode):
    _attributes = ('col_offset', )
    _fields = ()


class Array_Subscript_Node(FNode):
    _attributes = (
        'name',
        'type',
    )
    _fields = ('indices', )


class Type_Decl_Node(Statement_Node):
    _attributes = (
        'name',
        'type',
    )
    _fields = ()


class Allocate_Shape_Spec_Node(FNode):
    _attributes = ()
    _fields = ('sizes', )


class Allocate_Shape_Spec_List(FNode):
    _attributes = ()
    _fields = ('shape_list', )


class Allocation_Node(FNode):
    _attributes = ('name', )
    _fields = ('shape', )


class Allocate_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('allocation_list', )


class Symbol_Decl_Node(Statement_Node):
    _attributes = (
        'name',
        'type',
        'alloc',
    )
    _fields = (
        'sizes',
        'typeref',
        'init',
    )


class Symbol_Array_Decl_Node(Statement_Node):
    _attributes = (
        'name',
        'type',
        'alloc',
    )
    _fields = (
        'sizes',
        'offsets'
        'typeref',
        'init',
    )


class Var_Decl_Node(Statement_Node):
    _attributes = (
        'name',
        'type',
        'alloc',
        'kind',
        'optional'
    )
    _fields = (
        'sizes',
        'offsets',
        'actual_offsets'
        'typeref',
        'init',
    )


class Arg_List_Node(FNode):
    _fields = ('args', )


class Component_Spec_List_Node(FNode):
    _fields = ('args', )


class Decl_Stmt_Node(Statement_Node):
    _attributes = ()
    _fields = ('vardecl', )


class VarType:
    _attributes = ()


class Void(VarType):
    _attributes = ()


class Literal(FNode):
    _attributes = ('value', )
    _fields = ()


class Int_Literal_Node(Literal):
    pass

class Real_Literal_Node(Literal):
    pass

class Double_Literal_Node(Literal):
    pass

class Bool_Literal_Node(Literal):
    pass

class String_Literal_Node(Literal):
    pass

class Char_Literal_Node(Literal):
    pass

class Suffix_Node(FNode):
    _attributes = ()
    _fields = ('name')

class Call_Expr_Node(FNode):
    _attributes = ('type', 'subroutine')
    _fields = (
        'name',
        'args',
    )


class Derived_Type_Stmt_Node(FNode):
    _attributes = ('name', )
    _fields = ('args', )


class Derived_Type_Def_Node(FNode):
    _attributes = ('name', )
    _fields = ('component_part', )


class Component_Part_Node(FNode):
    _attributes = ()
    _fields = ('component_def_stmts', )


class Data_Component_Def_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('vars', )


class Data_Ref_Node(FNode):
    _attributes = ()
    _fields = ('parent_ref', 'part_ref')


class Array_Constructor_Node(FNode):
    _attributes = ()
    _fields = ('value_list', )


class Ac_Value_List_Node(FNode):
    _attributes = ()
    _fields = ('value_list', )


class Section_Subscript_List_Node(FNode):
    _fields = ('list')


class Pointer_Assignment_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('name_pointer', 'name_target')


class For_Stmt_Node(FNode):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'body',
        'iter',
    )


class Map_Stmt_Node(For_Stmt_Node):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'body',
        'iter',
    )


class If_Stmt_Node(FNode):
    _attributes = ()
    _fields = (
        'cond',
        'body',
        'body_else',
    )


class Else_Separator_Node(FNode):
    _attributes = ()
    _fields = ()


class Parenthesis_Expr_Node(FNode):
    _attributes = ()
    _fields = ('expr', 'type')


class Nonlabel_Do_Stmt_Node(FNode):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'iter',
    )

class While_True_Control(FNode):
    _attributes = ()
    _fields = (
        'name',
    )


class While_Control(FNode):
    _attributes = ()
    _fields = (
        'cond',
    )

class While_Stmt_Node(FNode):
    _attributes = ('name')
    _fields = (
        'body',
        'cond',
    )

class Loop_Control_Node(FNode):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'iter',
    )


class Else_If_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('cond', )


class Only_List_Node(FNode):
    _attributes = ()
    _fields = ('names','renames', )

class Rename_Node(FNode):
    _attributes = ()
    _fields = ('oldname', 'newname', )

class ParDecl_Node(FNode):
    _attributes = ('type', )
    _fields = ('range', )


class Structure_Constructor_Node(FNode):
    _attributes = ('type', )
    _fields = ('name', 'args')


class Use_Stmt_Node(FNode):
    _attributes = ('name','list_all' )
    _fields = ('list', )


class Write_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('args', )

class Break_Node(FNode):
    _attributes = ()
    _fields = ()
