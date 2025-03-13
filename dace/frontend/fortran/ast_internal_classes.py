# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List, Optional, Tuple, Union, Dict, Any


# The node class is the base class for all nodes in the AST. It provides attributes including the line number and fields.
# Attributes are not used when walking the tree, but are useful for debugging and for code generation.
# The fields attribute is a list of the names of the attributes that are children of the node.


class FNode(object):
    def __init__(self,
                 line_number: Tuple[int, int] = (0, 0),
                 parent: Union[
                     None, 'Subroutine_Subprogram_Node', 'Function_Subprogram_Node', 'Main_Program_Node',
                     'Module_Node'] = None,
                 **kwargs):  # real signature unknown
        self.line_number = line_number
        self.parent = parent
        for k, v in kwargs.items():
            setattr(self, k, v)

    _attributes: Tuple[str, ...] = ("line_number",)
    _fields: Tuple[str, ...] = ()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, type(self)):
            return False
        # check that all fields and attributes match
        self_field_vals = list(map(lambda name: getattr(self, name, None), self._fields))
        self_attr_vals = list(map(lambda name: getattr(self, name, None), self._attributes))
        o_field_vals = list(map(lambda name: getattr(o, name, None), o._fields))
        o_attr_vals = list(map(lambda name: getattr(o, name, None), o._attributes))
        return self_field_vals == o_field_vals and self_attr_vals == o_attr_vals

    def __str__(self) -> str:
        """Project the node object to a readable string, which may not necessarily be a full representation."""

        def _indent(txt: str) -> str:
            INDENT_BY = 2
            lns = txt.strip().split('\n')
            lns = [f"{' ' * INDENT_BY}{l.rstrip()}" for l in lns]
            return '\n'.join(lns)

        def _fieldstr(fnode) -> str:
            if isinstance(fnode, (list, tuple)):
                if not fnode:
                    return "[]"
                fstrs = ',\n'.join([str(f) for f in fnode])
                return f"[\n{_indent(fstrs)}\n]"
            return str(fnode)

        clsname = type(self).__name__.removesuffix('_Node')
        objname = self.name if hasattr(self, 'name') else (self.op if hasattr(self, 'op') else '?')
        objtype = f"/{self.type}" if hasattr(self, 'type') else ''
        fieldstrs = {f: _fieldstr(getattr(self, f)) for f in self._fields
                     if hasattr(self, f) and f not in {'name', 'type'}}
        fieldstrs = [f"{k}:{_indent(v)}" for k, v in fieldstrs.items()]
        if fieldstrs:
            fieldstrs = '\n'.join(fieldstrs)
            return f"{clsname} '{objname}{objtype}':\n{_indent(fieldstrs)}"
        else:
            return f"{clsname} '{objname}{objtype}'"


class Program_Node(FNode):
    def __init__(self,
                 main_program: 'Main_Program_Node',
                 function_definitions: List['Function_Subprogram_Node'],
                 subroutine_definitions: List['Subroutine_Subprogram_Node'],
                 modules: List,
                 module_declarations: Dict,
                 placeholders: Optional[List] = None,
                 placeholders_offsets: Optional[List] = None,
                 structures: Optional['Structures'] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.main_program = main_program
        self.function_definitions = function_definitions
        self.subroutine_definitions = subroutine_definitions
        self.modules = modules
        self.module_declarations = module_declarations
        self.structures = structures
        self.placeholders = placeholders
        self.placeholders_offsets = placeholders_offsets

    _attributes = ()
    _fields = (
        'main_program',
        'function_definitions',
        'subroutine_definitions',
        'modules',
    )


class BinOp_Node(FNode):
    def __init__(self, op: str, lval: FNode, rval: FNode, type: str = 'VOID', **kwargs):
        super().__init__(**kwargs)
        assert rval is not None
        self.op = op
        self.lval = lval
        self.rval = rval
        self.type = type

    _attributes = ('op', 'type')
    _fields = ('lval', 'rval')


class UnOp_Node(FNode):
    _attributes = (
        'op',
        'postfix',
        'type',
    )
    _fields = ('lval',)


class Exit_Node(FNode):
    _attributes = ()
    _fields = ()


class Main_Program_Node(FNode):
    _attributes = ("name",)
    _fields = ("execution_part", "specification_part")


class Module_Node(FNode):
    def __init__(self,
                 name: 'Name_Node',
                 specification_part: 'Specification_Part_Node',
                 subroutine_definitions: List['Subroutine_Subprogram_Node'],
                 function_definitions: List['Function_Subprogram_Node'],
                 interface_blocks: Dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.specification_part = specification_part
        self.subroutine_definitions = subroutine_definitions
        self.function_definitions = function_definitions
        self.interface_blocks = interface_blocks

    _attributes = ('name',)
    _fields = (
        'specification_part',
        'subroutine_definitions',
        'function_definitions',
        'interface_blocks'
    )


class Module_Subprogram_Part_Node(FNode):
    def __init__(self,
                 subroutine_definitions: List['Subroutine_Subprogram_Node'],
                 function_definitions: List['Function_Subprogram_Node'],
                 **kwargs):
        super().__init__(**kwargs)
        self.subroutine_definitions = subroutine_definitions
        self.function_definitions = function_definitions

    _attributes = ()
    _fields = (
        'subroutine_definitions',
        'function_definitions',
    )


class Internal_Subprogram_Part_Node(FNode):
    def __init__(self,
                 subroutine_definitions: List['Subroutine_Subprogram_Node'],
                 function_definitions: List['Function_Subprogram_Node'],
                 **kwargs):
        super().__init__(**kwargs)
        self.subroutine_definitions = subroutine_definitions
        self.function_definitions = function_definitions

    _attributes = ()
    _fields = (
        'subroutine_definitions',
        'function_definitions',
    )


class Actual_Arg_Spec_Node(FNode):
    def __init__(self, arg_name: 'Name_Node', arg: FNode, **kwargs):
        super().__init__(**kwargs)
        self.arg_name = arg_name
        self.arg = arg

    _fields = ('arg_name', 'arg')


class Function_Subprogram_Node(FNode):
    def __init__(self,
                 name: 'Name_Node',
                 args: List,
                 ret: 'Name_Node',
                 specification_part: Optional['Specification_Part_Node'],
                 execution_part: Optional['Execution_Part_Node'],
                 internal_subprogram_part: Optional[Internal_Subprogram_Part_Node],
                 type: str,
                 elemental: bool,
                 **kwargs):
        super().__init__(**kwargs)
        assert type != 'VOID', f"A Fortran function must have a return type; got VOID for {name.name}"
        self.name = name
        self.type = type
        self.ret = ret
        self.args = args
        self.specification_part = specification_part
        self.execution_part = execution_part
        self.internal_subprogram_part = internal_subprogram_part
        self.elemental = elemental

    _attributes = ('name', 'type', 'ret')
    _fields = (
        'args',
        'specification_part',
        'execution_part',
    )


class Subroutine_Subprogram_Node(FNode):
    def __init__(self,
                 name: 'Name_Node',
                 args: List,
                 specification_part: Optional['Specification_Part_Node'],
                 execution_part: Optional['Execution_Part_Node'],
                 internal_subprogram_part: Optional[Internal_Subprogram_Part_Node],
                 mandatory_args_count: int = -1,
                 optional_args_count: int = -1,
                 type: Any = None,
                 elemental: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.type = type
        self.args = args
        self.mandatory_args_count = mandatory_args_count
        self.optional_args_count = optional_args_count
        self.elemental = elemental
        self.specification_part = specification_part
        self.execution_part = execution_part
        self.internal_subprogram_part = internal_subprogram_part

    _attributes = ('name', 'type', 'elemental')
    _fields = (
        'args',
        'mandatory_args_count',
        'optional_args_count',
        'specification_part',
        'execution_part',
        'internal_subprogram_part',
    )


class Interface_Block_Node(FNode):
    _attributes = ('name',)
    _fields = (
        'subroutines',
    )


class Interface_Stmt_Node(FNode):
    _attributes = ()
    _fields = ()


class Procedure_Name_List_Node(FNode):
    _attributes = ()
    _fields = ('subroutines',)


class Procedure_Statement_Node(FNode):
    _attributes = ()
    _fields = ('namelists',)


class Module_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('functions',)


class Program_Stmt_Node(FNode):
    _attributes = ('name',)
    _fields = ()


class Subroutine_Stmt_Node(FNode):
    _attributes = ('name',)
    _fields = ('args',)


class Function_Stmt_Node(FNode):
    def __init__(self, name: 'Name_Node', args: List[FNode], ret: Optional['Suffix_Node'], elemental: bool, type: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.args = args
        self.ret = ret
        self.elemental = elemental
        self.type = type

    _attributes = ('name', 'elemental', 'type')
    _fields = ('args', 'ret',)


class Prefix_Node(FNode):
    def __init__(self, type: str, elemental: bool, recursive: bool, pure: bool, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.elemental = elemental
        self.recursive = recursive
        self.pure = pure

    _attributes = ('elemental', 'recursive', 'pure',)
    _fields = ()


class Name_Node(FNode):
    def __init__(self, name: str, type: str = 'VOID', **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.type = type

    _attributes = ('name', 'type',)
    _fields = ()

    def __str__(self) -> str:
        return self.name


class Name_Range_Node(FNode):
    _attributes = ('name', 'type', 'arrname', 'pos',)
    _fields = ()


class Where_Construct_Node(FNode):
    _attributes = ()
    _fields = ('main_body', 'main_cond', 'else_body', 'elifs_body', 'elifs_cond',)


class Type_Name_Node(FNode):
    _attributes = ('name', 'type',)
    _fields = ()


class Generic_Binding_Node(FNode):
    _attributes = ()
    _fields = ('name', 'binding',)


class Specification_Part_Node(FNode):
    _fields = ('specifications', 'symbols', 'interface_blocks', 'typedecls', 'enums',)


class Stop_Stmt_Node(FNode):
    _attributes = ('code',)


class Error_Stmt_Node(FNode):
    _fields = ('error',)


class Execution_Part_Node(FNode):
    _fields = ('execution',)


class Statement_Node(FNode):
    _attributes = ('col_offset',)
    _fields = ()


class Array_Subscript_Node(FNode):
    def __init__(self, name: Name_Node, type: str, indices: List[FNode], **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.type = type
        self.indices = indices

    _attributes = ('type',)
    _fields = ('name', 'indices',)


class Type_Decl_Node(Statement_Node):
    _attributes = (
        'name',
        'type',
    )
    _fields = ()


class Allocate_Shape_Spec_Node(FNode):
    _attributes = ()
    _fields = ('sizes',)


class Allocate_Shape_Spec_List(FNode):
    _attributes = ()
    _fields = ('shape_list',)


class Allocation_Node(FNode):
    def __init__(self, name: Name_Node, shape: Allocate_Shape_Spec_List, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.shape = shape

    _attributes = ('name',)
    _fields = ('shape',)


class Continue_Node(FNode):
    _attributes = ()
    _fields = ()


class Allocate_Stmt_Node(FNode):
    def __init__(self, allocation_list: List[Allocation_Node], **kwargs):
        super().__init__(**kwargs)
        self.allocation_list = allocation_list

    _attributes = ()
    _fields = ('allocation_list',)


class Symbol_Decl_Node(Statement_Node):
    def __init__(self, name: str, type: str,
                 alloc: Optional[bool] = None, sizes: Optional[List] = None,
                 init: Optional[FNode] = None, typeref: Optional[Any] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.type = type
        self.alloc = alloc
        self.sizes = sizes
        self.typeref = typeref
        self.init = init

    _attributes = (
        'name',
        'type',
        'alloc',
    )
    _fields = (
        'sizes',
        'typeref',
        'init',
        'offsets',
    )


class Symbol_Array_Decl_Node(Statement_Node):
    def __init__(self, name: str, type: str,
                 alloc: Optional[bool] = None, sizes: Optional[List] = None, offsets: Optional[List] = None,
                 init: Optional[FNode] = None, typeref: Optional[Any] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.type = type
        self.alloc = alloc
        self.sizes = sizes
        self.offsets = offsets
        self.typeref = typeref
        self.init = init

    _attributes = (
        'name',
        'type',
        'alloc',
    )
    _fields = (
        'sizes',
        'offsets',
        'typeref',
        'init',
    )


class Var_Decl_Node(Statement_Node):
    def __init__(self, name: str, type: str,
                 alloc: Optional[bool] = None, optional: Optional[bool] = None,
                 sizes: Optional[List] = None, offsets: Optional[List[Union[int, Name_Node]]] = None,
                 init: Optional[FNode] = None, actual_offsets: Optional[List] = None,
                 typeref: Optional[Any] = None, kind: Optional[Any] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.type = type
        self.alloc = alloc
        self.kind = kind
        self.optional = optional
        self.sizes = sizes
        self.offsets = offsets
        self.actual_offsets = actual_offsets
        self.typeref = typeref
        self.init = init

    _attributes = ('name', 'type', 'alloc', 'kind', 'optional')
    _fields = ('sizes', 'offsets', 'actual_offsets', 'typeref', 'init')


class Arg_List_Node(FNode):
    _fields = ('args',)


class Component_Spec_List_Node(FNode):
    _fields = ('args',)


class Allocate_Object_List_Node(FNode):
    _fields = ('list',)


class Deallocate_Stmt_Node(FNode):
    def __init__(self, deallocation_list: List[Name_Node], **kwargs):
        super().__init__(**kwargs)
        self.deallocation_list = deallocation_list

    _fields = ('deallocation_list',)


class Decl_Stmt_Node(Statement_Node):
    def __init__(self, vardecl: List[Var_Decl_Node], **kwargs):
        super().__init__(**kwargs)
        self.vardecl = vardecl

    _attributes = ()
    _fields = ('vardecl',)


class VarType:
    _attributes = ()


class Void(VarType):
    _attributes = ()


class Literal(FNode):
    def __init__(self, value: str, type: str, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.type = type

    _attributes = ('value', 'type')
    _fields = ()

    def __str__(self) -> str:
        return f"{self.type}({self.value})"


class Int_Literal_Node(Literal):
    def __init__(self, value: str, type='INTEGER', **kwargs):
        super().__init__(value, type, **kwargs)


class Real_Literal_Node(Literal):
    def __init__(self, value: str, type='REAL', **kwargs):
        super().__init__(value, type, **kwargs)


class Double_Literal_Node(Literal):
    def __init__(self, value: str, type='DOUBLE', **kwargs):
        super().__init__(value, type, **kwargs)


class Bool_Literal_Node(Literal):
    def __init__(self, value: str, type='LOGICAL', **kwargs):
        assert value in {'0', '1'},\
            f"`{value}` is not a valid respresentation: use `0` for falsey values, and `1` for truthy values."
        super().__init__(value, type, **kwargs)


class Char_Literal_Node(Literal):
    def __init__(self, value: str, type='CHAR', **kwargs):
        super().__init__(value, type, **kwargs)


class Suffix_Node(FNode):
    def __init__(self, name: 'Name_Node', **kwargs):
        super().__init__(**kwargs)
        self.name = name

    _attributes = ()
    _fields = ('name',)


class Call_Expr_Node(FNode):
    def __init__(self, name: 'Name_Node', args: List[FNode], subroutine: bool, type: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.args = args
        self.subroutine = subroutine
        self.type = type

    _attributes = ('type', 'subroutine',)
    _fields = ('name', 'args',)


class Derived_Type_Stmt_Node(FNode):
    _attributes = ('name',)
    _fields = ('args',)


class Derived_Type_Def_Node(FNode):
    def __init__(self, name: Type_Name_Node,
                 component_part: 'Component_Part_Node', procedure_part: 'Bound_Procedures_Node',
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.component_part = component_part
        self.procedure_part = procedure_part

    _attributes = ('name',)
    _fields = ('component_part', 'procedure_part',)


class Component_Part_Node(FNode):
    _attributes = ()
    _fields = ('component_def_stmts',)


class Data_Component_Def_Stmt_Node(FNode):
    def __init__(self, vars: Decl_Stmt_Node, **kwargs):
        super().__init__(**kwargs)
        self.vars = vars

    _attributes = ()
    _fields = ('vars',)


class Data_Ref_Node(FNode):
    def __init__(self, parent_ref: FNode, part_ref: FNode, type: str = 'VOID', **kwargs):
        super().__init__(**kwargs)
        self.parent_ref = parent_ref
        self.part_ref = part_ref
        self.type = type

    _attributes = ('type',)
    _fields = ('parent_ref', 'part_ref')


class Array_Constructor_Node(FNode):
    _attributes = ()
    _fields = ('value_list',)


class Ac_Value_List_Node(FNode):
    _attributes = ()
    _fields = ('value_list',)


class Section_Subscript_List_Node(FNode):
    _fields = ('list',)


class Pointer_Assignment_Stmt_Node(FNode):
    def __init__(self, name_pointer: Name_Node, name_target: FNode, **kwargs):
        super().__init__(**kwargs)
        self.name_pointer = name_pointer
        self.name_target = name_target

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


class Defer_Shape_Node(FNode):
    _attributes = ()
    _fields = ()


class Component_Initialization_Node(FNode):
    _attributes = ()
    _fields = ('init',)


class Case_Cond_Node(FNode):
    _fields = ('cond', 'op')
    _attributes = ()


class Else_Separator_Node(FNode):
    _attributes = ()
    _fields = ()


class Procedure_Separator_Node(FNode):
    _attributes = ()
    _fields = ('parent_ref', 'part_ref')


class Pointer_Object_List_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class Read_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('args',)


class Close_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('args',)


class Open_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('args',)


class Associate_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('args',)


class Associate_Construct_Node(FNode):
    _attributes = ()
    _fields = ('associate', 'body')


class Association_List_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class Association_Node(FNode):
    _attributes = ()
    _fields = ('name', 'expr')


class Connect_Spec_Node(FNode):
    _attributes = ('type',)
    _fields = ('args',)


class Close_Spec_Node(FNode):
    _attributes = ('type',)
    _fields = ('args',)


class Close_Spec_List_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class IO_Control_Spec_Node(FNode):
    _attributes = ('type',)
    _fields = ('args',)


class IO_Control_Spec_List_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class Connect_Spec_List_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class Nullify_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class Namelist_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('list', 'name')


class Namelist_Group_Object_List_Node(FNode):
    _attributes = ()
    _fields = ('list',)


class Bound_Procedures_Node(FNode):
    _attributes = ()
    _fields = ('procedures',)


class Specific_Binding_Node(FNode):
    _attributes = ()
    _fields = ('name', 'args')


class Parenthesis_Expr_Node(FNode):
    def __init__(self, expr: FNode, **kwargs):
        super().__init__(**kwargs)
        assert hasattr(expr, 'type')
        self.expr = expr
        self.type = expr.type

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
    _fields = ('cond',)


class Only_List_Node(FNode):
    _attributes = ()
    _fields = ('names', 'renames',)


class Rename_Node(FNode):
    _attributes = ()
    _fields = ('oldname', 'newname',)


class ParDecl_Node(FNode):
    _attributes = ('type',)
    _fields = ('range',)


class Structure_Constructor_Node(FNode):
    _attributes = ('type',)
    _fields = ('name', 'args')


class Use_Stmt_Node(FNode):
    _attributes = ('name', 'list_all')
    _fields = ('list',)


class Write_Stmt_Node(FNode):
    _attributes = ()
    _fields = ('args',)


class Break_Node(FNode):
    _attributes = ()
    _fields = ()
