# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy
from typing import Union, Tuple, Optional, List, Iterable

from fparser.api import get_reader
import fparser.two.Fortran2003 as f03
from fparser.two.utils import Base, BlockBase

from dace.frontend.fortran import ast_utils

# Type Aliases for common node groupings
# Represents program entry points like the main program, subroutines, and functions.
ENTRY_POINT_OBJECT_TYPES = Union[f03.Main_Program, f03.Subroutine_Subprogram, f03.Function_Subprogram]
ENTRY_POINT_OBJECT_CLASSES = (f03.Main_Program, f03.Subroutine_Subprogram, f03.Function_Subprogram)
# Represents nodes that define a new scope (e.g., modules, functions, derived types).
SCOPE_OBJECT_TYPES = Union[f03.Main_Program, f03.Module, f03.Function_Subprogram, f03.Subroutine_Subprogram,
                           f03.Derived_Type_Def, f03.Interface_Block, f03.Subroutine_Body, f03.Function_Body,
                           f03.Stmt_Function_Stmt]
SCOPE_OBJECT_CLASSES = (f03.Main_Program, f03.Module, f03.Function_Subprogram, f03.Subroutine_Subprogram,
                        f03.Derived_Type_Def, f03.Interface_Block, f03.Subroutine_Body, f03.Function_Body,
                        f03.Stmt_Function_Stmt)
# Represents statements that have a name and are of interest for analysis.
NAMED_STMTS_OF_INTEREST_TYPES = Union[f03.Program_Stmt, f03.Module_Stmt, f03.Function_Stmt, f03.Subroutine_Stmt,
                                      f03.Derived_Type_Stmt, f03.Component_Decl, f03.Entity_Decl, f03.Specific_Binding,
                                      f03.Generic_Binding, f03.Interface_Stmt, f03.Stmt_Function_Stmt,
                                      f03.Proc_Component_Def_Stmt, f03.Proc_Decl]
NAMED_STMTS_OF_INTEREST_CLASSES = (f03.Program_Stmt, f03.Module_Stmt, f03.Function_Stmt, f03.Subroutine_Stmt,
                                   f03.Derived_Type_Stmt, f03.Component_Decl, f03.Entity_Decl, f03.Specific_Binding,
                                   f03.Generic_Binding, f03.Interface_Stmt, f03.Stmt_Function_Stmt,
                                   f03.Proc_Component_Def_Stmt, f03.Proc_Decl)


def find_name_of_stmt(node: NAMED_STMTS_OF_INTEREST_TYPES) -> Optional[str]:
    """
    Finds the name of a statement node if it has one.

    :param node: The fparser statement node.
    :return: The name of the statement as a string, or `None` for anonymous blocks.
    """
    if isinstance(node, f03.Specific_Binding):
        # Ref: https://github.com/stfc/fparser/blob/8c870f84edbf1a24dfbc886e2f7226d1b158d50b/src/fparser/two/Fortran2003.py#L2504
        _, _, _, bname, _ = node.children
        name = bname
    elif isinstance(node, f03.Generic_Binding):
        _, bname, _ = node.children
        name = bname
    elif isinstance(node, f03.Interface_Stmt):
        name, = node.children
        if name == 'ABSTRACT':
            return None
    elif isinstance(node, f03.Proc_Component_Def_Stmt):
        tgt, attrs, plist = node.children
        assert len(plist.children) == 1, \
            f"Only one procedure per statement is accepted due to Fparser bug. Break down the line: {node}"
        name = ast_utils.singular(ast_utils.children_of_type(plist, f03.Name))
    else:
        # TODO: Test out other type specific ways of finding names.
        name = ast_utils.singular(ast_utils.children_of_type(node, f03.Name))
    if name:
        name = f"{name}"
    return name


def find_name_of_node(node: Base) -> Optional[str]:
    """
    Finds the name of a general fparser node if it contains a named statement.

    :param node: The fparser node.
    :return: The name as a string, or `None` if no named statement is found.
    """
    if isinstance(node, NAMED_STMTS_OF_INTEREST_CLASSES):
        return find_name_of_stmt(node)
    stmt = ast_utils.atmost_one(ast_utils.children_of_type(node, NAMED_STMTS_OF_INTEREST_CLASSES))
    if not stmt:
        return None
    return find_name_of_stmt(stmt)


def find_scope_ancestor(node: Base) -> Optional[SCOPE_OBJECT_TYPES]:
    """
    Traverses up the AST from a given node to find the nearest ancestor that defines a scope.

    :param node: The starting fparser node.
    :return: The scope-defining ancestor node, or `None` if not found.
    """
    anc = node.parent
    while anc and not isinstance(anc, SCOPE_OBJECT_CLASSES):
        anc = anc.parent
    return anc


def find_named_ancestor(node: Base) -> Optional[NAMED_STMTS_OF_INTEREST_TYPES]:
    """
    Finds the nearest ancestor that is a named statement of interest.

    :param node: The starting fparser node.
    :return: The named ancestor statement node, or `None` if not found.
    """
    anc = find_scope_ancestor(node)
    if not anc:
        return None
    return ast_utils.atmost_one(ast_utils.children_of_type(anc, NAMED_STMTS_OF_INTEREST_CLASSES))


def lineage(anc: Base, des: Base) -> Optional[Tuple[Base, ...]]:
    """
    Constructs the path (lineage) from an ancestor node to a descendant node.

    :param anc: The ancestor node.
    :param des: The descendant node.
    :return: A tuple of nodes representing the path from ancestor to descendant, or `None` if `des` is not a descendant of `anc`.
    """
    if anc is des:
        return (anc, )
    if not des.parent:
        return None
    lin = lineage(anc, des.parent)
    if not lin:
        return None
    return lin + (des, )


def _reparent_children(node: Base):
    """
    Sets the `parent` attribute of all children of a node to be the node itself.
    This is a utility to fix up the AST after manual modifications.

    :param node: The parent node.
    """
    for c in node.children:
        if isinstance(c, Base):
            c.parent = node


def set_children(par: Base, children: Iterable[Union[Base, str]]):
    """
    Replaces the children of a node with a new set of children.

    :param par: The parent node to modify.
    :param children: An iterable of new children (nodes or strings).
    """
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
    """
    Removes one or more nodes from their parent's children list.

    :param nodes: A single node or a list of nodes to remove.
    """
    if isinstance(nodes, Base):
        nodes = [nodes]
    for n in nodes:
        remove_children(n.parent, n)


def replace_node(node: Base, subst: Union[None, Base, Iterable[Base]]):
    """
    Replaces a node in the AST with one or more other nodes (or nothing).

    :param node: The node to be replaced.
    :param subst: The substitution, which can be `None` (for deletion), a single node, or an iterable of nodes.
    """
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
    if isinstance(par, f03.Loop_Control) and isinstance(subst, Base):
        _, cntexpr, _, _ = par.children
        if cntexpr:
            loopvar, looprange = cntexpr
            for i in range(len(looprange)):
                if looprange[i] is node:
                    looprange[i] = subst
                    subst.parent = par
    set_children(par, repls)


def append_children(par: Base, children: Union[Base, List[Base]]):
    """
    Appends one or more children to a parent node.

    :param par: The parent node.
    :param children: A single child node or a list of child nodes to append.
    """
    if isinstance(children, Base):
        children = [children]
    set_children(par, list(par.children) + children)


def prepend_children(par: Base, children: Union[Base, List[Base]]):
    """
    Prepends one or more children to a parent node.

    :param par: The parent node.
    :param children: A single child node or a list of child nodes to prepend.
    """
    if isinstance(children, Base):
        children = [children]
    set_children(par, children + list(par.children))


def remove_children(par: Base, children: Union[Base, List[Base]]):
    """
    Removes specific children from a parent node.

    :param par: The parent node.
    :param children: A single child node or a list of child nodes to remove.
    """
    if isinstance(children, Base):
        children = [children]
    cids = {id(c) for c in children}
    repl = [c for c in par.children if id(c) not in cids]
    set_children(par, repl)


def copy_fparser_node(n: Base) -> Base:
    """
    Creates a copy of an fparser node. It tries to re-parse from the Fortran string representation
    for a clean copy, otherwise falls back to a deep copy.

    :param n: The fparser node to copy.
    :return: A new fparser node instance.
    """
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


def _get_module_or_program_parts(mod: Union[f03.Module, f03.Main_Program]) \
        -> Tuple[
            Union[f03.Module_Stmt, f03.Program_Stmt],
            Optional[f03.Specification_Part],
            Optional[f03.Execution_Part],
            Optional[f03.Module_Subprogram_Part],
        ]:
    """
    Deconstructs a Module or Main_Program node into its constituent parts.

    :param mod: The Module or Main_Program node.
    :return: A tuple containing the statement, specification part, execution part, and subprogram part.
    """
    # There must exist a module statment.
    stmt = ast_utils.singular(
        ast_utils.children_of_type(mod, f03.Module_Stmt if isinstance(mod, f03.Module) else f03.Program_Stmt))
    # There may or may not exist a specification part.
    spec = list(ast_utils.children_of_type(mod, f03.Specification_Part))
    assert len(spec) <= 1, f"A module/program cannot have more than one specification parts, found {spec} in {mod}"
    spec = spec[0] if spec else None
    # There may or may not exist an execution part.
    expart = list(ast_utils.children_of_type(mod, f03.Execution_Part))
    assert len(expart) <= 1, f"A module/program cannot have more than one execution parts, found {spec} in {mod}"
    expart = expart[0] if expart else None
    # There may or may not exist a subprogram part.
    subp = list(ast_utils.children_of_type(mod, f03.Module_Subprogram_Part))
    assert len(subp) <= 1, f"A module/program cannot have more than one subprogram parts, found {subp} in {mod}"
    subp = subp[0] if subp else None
    return stmt, spec, expart, subp
