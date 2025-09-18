# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy
from typing import Union, Tuple, Optional, List, Iterable

from fparser.api import get_reader
from fparser.two.Fortran2003 import Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt, \
    Component_Decl, Entity_Decl, Specific_Binding, Generic_Binding, Interface_Stmt, Main_Program, Subroutine_Subprogram, \
    Function_Subprogram, Name, Module, Loop_Control, Module_Subprogram_Part, Specification_Part, Execution_Part, \
    Proc_Component_Def_Stmt, Proc_Decl, \
    Stmt_Function_Stmt, Interface_Block, Subroutine_Body, Derived_Type_Def, Function_Body
from fparser.two.utils import Base, BlockBase

from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one

# Type Aliases for common node groupings
ENTRY_POINT_OBJECT_TYPES = Union[Main_Program, Subroutine_Subprogram, Function_Subprogram]
ENTRY_POINT_OBJECT_CLASSES = (Main_Program, Subroutine_Subprogram, Function_Subprogram)
SCOPE_OBJECT_TYPES = Union[Main_Program, Module, Function_Subprogram, Subroutine_Subprogram, Derived_Type_Def,
                           Interface_Block, Subroutine_Body, Function_Body, Stmt_Function_Stmt]
SCOPE_OBJECT_CLASSES = (Main_Program, Module, Function_Subprogram, Subroutine_Subprogram, Derived_Type_Def,
                        Interface_Block, Subroutine_Body, Function_Body, Stmt_Function_Stmt)
NAMED_STMTS_OF_INTEREST_TYPES = Union[Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt,
                                      Component_Decl, Entity_Decl, Specific_Binding, Generic_Binding, Interface_Stmt,
                                      Stmt_Function_Stmt, Proc_Component_Def_Stmt, Proc_Decl]
NAMED_STMTS_OF_INTEREST_CLASSES = (Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt,
                                   Component_Decl, Entity_Decl, Specific_Binding, Generic_Binding, Interface_Stmt,
                                   Stmt_Function_Stmt, Proc_Component_Def_Stmt, Proc_Decl)


def find_name_of_stmt(node: NAMED_STMTS_OF_INTEREST_TYPES) -> Optional[str]:
    """Find the name of the statement if it has one. For anonymous blocks, return `None`."""
    if isinstance(node, Specific_Binding):
        # Ref: https://github.com/stfc/fparser/blob/8c870f84edbf1a24dfbc886e2f7226d1b158d50b/src/fparser/two/Fortran2003.py#L2504
        _, _, _, bname, _ = node.children
        name = bname
    elif isinstance(node, Generic_Binding):
        _, bname, _ = node.children
        name = bname
    elif isinstance(node, Interface_Stmt):
        name, = node.children
        if name == 'ABSTRACT':
            return None
    elif isinstance(node, Proc_Component_Def_Stmt):
        tgt, attrs, plist = node.children
        assert len(plist.children) == 1, \
            f"Only one procedure per statement is accepted due to Fparser bug. Break down the line: {node}"
        name = singular(children_of_type(plist, Name))
    else:
        # TODO: Test out other type specific ways of finding names.
        name = singular(children_of_type(node, Name))
    if name:
        name = f"{name}"
    return name


def find_name_of_node(node: Base) -> Optional[str]:
    """Find the name of the general node if it has one. For anonymous blocks, return `None`."""
    if isinstance(node, NAMED_STMTS_OF_INTEREST_CLASSES):
        return find_name_of_stmt(node)
    stmt = atmost_one(children_of_type(node, NAMED_STMTS_OF_INTEREST_CLASSES))
    if not stmt:
        return None
    return find_name_of_stmt(stmt)


def find_scope_ancestor(node: Base) -> Optional[SCOPE_OBJECT_TYPES]:
    anc = node.parent
    while anc and not isinstance(anc, SCOPE_OBJECT_CLASSES):
        anc = anc.parent
    return anc


def find_named_ancestor(node: Base) -> Optional[NAMED_STMTS_OF_INTEREST_TYPES]:
    anc = find_scope_ancestor(node)
    if not anc:
        return None
    return atmost_one(children_of_type(anc, NAMED_STMTS_OF_INTEREST_CLASSES))


def lineage(anc: Base, des: Base) -> Optional[Tuple[Base, ...]]:
    if anc is des:
        return (anc, )
    if not des.parent:
        return None
    lin = lineage(anc, des.parent)
    if not lin:
        return None
    return lin + (des, )


def _reparent_children(node: Base):
    """Make `node` a parent of all its children, in case it isn't already."""
    for c in node.children:
        if isinstance(c, Base):
            c.parent = node


def set_children(par: Base, children: Iterable[Union[Base, str]]):
    assert hasattr(par, 'content') != hasattr(par, 'items')
    if hasattr(par, 'items'):
        par.items = tuple(children)
    elif hasattr(par, 'content'):
        if not children:
            remove_self(par)
        else:
            par.content = list(children)
    if children:
        _reparent_children(par)


def remove_self(nodes: Union[Base, List[Base]]):
    if isinstance(nodes, Base):
        nodes = [nodes]
    for n in nodes:
        remove_children(n.parent, n)


def replace_node(node: Base, subst: Union[None, Base, Iterable[Base]]):
    # A lot of hacky stuff to make sure that the new nodes are not just the same objects over and over.
    par = node.parent
    repls = []
    for c in par.children:
        if c is not node:
            repls.append(c)
            continue
        if subst is None or isinstance(subst, Base):
            subst = [subst]
        repls.extend(subst)
    if isinstance(par, Loop_Control) and isinstance(subst, Base):
        _, cntexpr, _, _ = par.children
        if cntexpr:
            loopvar, looprange = cntexpr
            for i in range(len(looprange)):
                if looprange[i] is node:
                    looprange[i] = subst
                    subst.parent = par
    set_children(par, repls)


def append_children(par: Base, children: Union[Base, List[Base]]):
    if isinstance(children, Base):
        children = [children]
    set_children(par, list(par.children) + children)


def prepend_children(par: Base, children: Union[Base, List[Base]]):
    if isinstance(children, Base):
        children = [children]
    set_children(par, children + list(par.children))


def remove_children(par: Base, children: Union[Base, List[Base]]):
    if isinstance(children, Base):
        children = [children]
    cids = {id(c) for c in children}
    repl = [c for c in par.children if id(c) not in cids]
    set_children(par, repl)


def copy_fparser_node(n: Base) -> Base:
    try:
        nstr = n.tofortran()
        if isinstance(n, BlockBase):
            x = Base.__new__(type(n), get_reader(nstr))
        else:
            x = Base.__new__(type(n), nstr)
        assert x is not None
        return x
    except (RuntimeError, AssertionError):
        return deepcopy(n)


def _get_module_or_program_parts(mod: Union[Module, Main_Program]) \
        -> Tuple[
            Union[Module_Stmt, Program_Stmt],
            Optional[Specification_Part],
            Optional[Execution_Part],
            Optional[Module_Subprogram_Part],
        ]:
    # There must exist a module statment.
    stmt = singular(children_of_type(mod, Module_Stmt if isinstance(mod, Module) else Program_Stmt))
    # There may or may not exist a specification part.
    spec = list(children_of_type(mod, Specification_Part))
    assert len(spec) <= 1, f"A module/program cannot have more than one specification parts, found {spec} in {mod}"
    spec = spec[0] if spec else None
    # There may or may not exist an execution part.
    expart = list(children_of_type(mod, Execution_Part))
    assert len(expart) <= 1, f"A module/program cannot have more than one execution parts, found {spec} in {mod}"
    expart = expart[0] if expart else None
    # There may or may not exist a subprogram part.
    subp = list(children_of_type(mod, Module_Subprogram_Part))
    assert len(subp) <= 1, f"A module/program cannot have more than one subprogram parts, found {subp} in {mod}"
    subp = subp[0] if subp else None
    return stmt, spec, expart, subp
