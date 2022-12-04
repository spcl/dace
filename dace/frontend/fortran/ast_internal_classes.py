from typing import Any, List, Tuple, Type, TypeVar, Union, overload

# The node class is the base class for all nodes in the AST. It provides attributes including the line number and fields.
# Attributes are not used when walking the tree, but are useful for debugging and for code generation.
# The fields attribute is a list of the names of the attributes that are children of the node.


class Node(object):
    def __init__(self, *args, **kwargs):  # real signature unknown
        self.integrity_exceptions = []
        self.read_vars = []
        self.written_vars = []
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
            self_field_vals = list(
                map(lambda name: getattr(self, name, None), self._fields))
            self_attr_vals = list(
                map(lambda name: getattr(self, name, None), self._attributes))
            o_field_vals = list(
                map(lambda name: getattr(o, name, None), o._fields))
            o_attr_vals = list(
                map(lambda name: getattr(o, name, None), o._attributes))

            return self_field_vals == o_field_vals and self_attr_vals == o_attr_vals
        return False


class Program_Node(Node):
    _attributes = ()
    _fields = (
        "main_program",
        "function_definitions",
        "subroutine_definitions",
        "modules",
    )


class BinOp_Node(Node):
    _attributes = (
        'op',
        'type',
    )
    _fields = (
        'lval',
        'rval',
    )


class UnOp_Node(Node):
    _attributes = (
        'op',
        'postfix',
        'type',
    )
    _fields = ('lval', )


class Main_Program_Node(Node):
    _attributes = ("name", )
    _fields = ("execution_part", "specification_part")


class Module_Node(Node):
    _attributes = ('name', )
    _fields = (
        'specification_part',
        'subroutine_definitions',
        'function_definitions',
    )


class Function_Subprogram_Node(Node):
    _attributes = ('name', 'type', 'ret_name')
    _fields = (
        'args',
        'specification_part',
        'execution_part',
    )


class Subroutine_Subprogram_Node(Node):
    _attributes = ('name', 'type')
    _fields = (
        'args',
        'specification_part',
        'execution_part',
    )


class Module_Stmt_Node(Node):
    _attributes = ('name', )
    _fields = ()


class Program_Stmt_Node(Node):
    _attributes = ('name', )
    _fields = ()


class Subroutine_Stmt_Node(Node):
    _attributes = ('name', )
    _fields = ('args', )


class Function_Stmt_Node(Node):
    _attributes = ('name', )
    _fields = ('args', 'return')


class Name_Node(Node):
    _attributes = ('name', 'type')
    _fields = ()


class Name_Range_Node(Node):
    _attributes = ('name', 'type', 'arrname', 'pos')
    _fields = ()


class Type_Name_Node(Node):
    _attributes = ('name', 'type')
    _fields = ()


class Specification_Part_Node(Node):
    _fields = ('specifications', 'symbols', 'typedecls')


class Execution_Part_Node(Node):
    _fields = ('execution', )


class Statement_Node(Node):
    _attributes = ('col_offset', )
    _fields = ()


class Array_Subscript_Node(Node):
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
        'typeref',
        'init',
    )


class Var_Decl_Node(Statement_Node):
    _attributes = (
        'name',
        'type',
        'alloc',
        'kind',
    )
    _fields = (
        'sizes',
        'typeref',
        'init',
    )


class Arg_List_Node(Node):
    _fields = ('args', )


class Component_Spec_List_Node(Node):
    _fields = ('args', )


class Decl_Stmt_Node(Statement_Node):
    _attributes = ()
    _fields = ('vardecl', )


class VarType:
    _attributes = ()


class Void(VarType):
    _attributes = ()


class Literal(Node):
    _attributes = ('value', )
    _fields = ()


class Int_Literal_Node(Literal):
    _attributes = ()
    _fields = ()


class Real_Literal_Node(Literal):
    _attributes = ()
    _fields = ()


class Bool_Literal_Node(Literal):
    _attributes = ()
    _fields = ()


class String_Literal_Node(Literal):
    _attributes = ()
    _fields = ()


class Char_Literal_Node(Literal):
    _attributes = ()
    _fields = ()


class Call_Expr_Node(Node):
    _attributes = ('type', 'subroutine')
    _fields = (
        'name',
        'args',
    )


class Array_Constructor_Node(Node):
    _attributes = ()
    _fields = ('value_list', )


class Ac_Value_List_Node(Node):
    _attributes = ()
    _fields = ('value_list', )


class Section_Subscript_List_Node(Node):
    _fields = ('list')


class For_Stmt_Node(Node):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'body',
        'iter',
    )


class Map_Stmt_Node(For_Stmt_Node):
    _attributes = ()
    _fields = ()


class If_Stmt_Node(Node):
    _attributes = ()
    _fields = (
        'cond',
        'body',
        'body_else',
    )


class Else_Separator_Node(Node):
    _attributes = ()
    _fields = ()


class Parenthesis_Expr_Node(Node):
    _attributes = ()
    _fields = ('expr', )


class Nonlabel_Do_Stmt_Node(Node):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'iter',
    )


class Loop_Control_Node(Node):
    _attributes = ()
    _fields = (
        'init',
        'cond',
        'iter',
    )


class Else_If_Stmt_Node(Node):
    _attributes = ()
    _fields = ('cond', )


class Only_List_Node(Node):
    _attributes = ()
    _fields = ('names', )


class ParDecl_Node(Node):
    _attributes = ('type', )
    _fields = ('range', )


class Structure_Constructor_Node(Node):
    _attributes = ('type', )
    _fields = ('name', 'args')


class Use_Stmt_Node(Node):
    _attributes = ('name', )
    _fields = ('list', )


class Write_Stmt_Node(Node):
    _attributes = ()
    _fields = ('args', )