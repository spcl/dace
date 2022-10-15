# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from ast import AST, AnnAssign, Assign, Call, Constant, If, Name, For, While, Load, Store, copy_location, NodeVisitor
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union, List, Tuple


def is_simple_expr(node: AST) -> bool:
    return isinstance(node, (Constant, Name, UniqueName, ast.UnaryOp))


class _DefiniteReturn(NodeVisitor):
    def make_definite_return(self, block: List[AST]) -> List[AST]:

        block_returns = self.visit_block(block)

        if not block_returns:
            return_stmt = ast.Return()
            block.append(return_stmt)

        return block

    def generic_visit(self, node):
        return False

    def visit_block(self, block: List[AST]) -> bool:

        returns = False
        num_stmts = len(block)

        for i in range(num_stmts):
            stmt_returns = self.visit(block[i])
            if stmt_returns:
                returns = True
                block[:] = block[:i + 1]
                break

        return returns

    def visit_Return(self, node: ast.Return) -> bool:
        return True

    def visit_IfCFG(self, node: ast.If) -> bool:
        body_returns = self.visit_block(node.body)
        else_returns = self.visit_block(node.orelse)
        return body_returns and else_returns

    def visit_While(self, node: ast.While) -> bool:
        body_returns = self.visit_block(node.body)
        else_returns = self.visit_block(node.orelse)
        return body_returns and else_returns

    def visit_For(self, node: ast.For) -> bool:
        body_returns = self.visit_block(node.body)
        else_returns = self.visit_block(node.orelse)
        return body_returns and else_returns


class UniqueName(AST):
    _fields = ()
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    def to_Name(self) -> Name:

        return Name(id=self.id, ctx=self.ctx)


class SimpleFunction(AST):
    _fields = ('args', 'body', 'returns')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    @classmethod
    def from_FunctionDef(cls, func_def: ast.FunctionDef) -> 'SimpleFunction':

        args = func_def.args
        body = func_def.body

        # Check certain "simple" attributes about the function
        simple_defaults = [
            is_simple_expr(default) or default is None for default in chain(args.defaults, args.kw_defaults)
        ]

        # 1) Only simple default values
        if not all(simple_defaults):
            raise ValueError('All function defaults must be simple!')

        # 2) no decorators
        if func_def.decorator_list:
            raise ValueError('Decorator list must be empty!')

        # 3) implicit return is added
        body = _DefiniteReturn().make_definite_return(body)

        new_node = cls()
        new_node.name = func_def.name
        new_node.args = args
        new_node.body = body
        new_node.returns = func_def.returns
        new_node.type_comment = func_def.type_comment
        copy_location(new_node, func_def)

        return new_node

    def to_FunctionDef(self) -> ast.FunctionDef:

        new_node = ast.FunctionDef()
        new_node.name = self.name
        new_node.args = self.args
        new_node.body = self.body
        new_node.returns = self.returns
        new_node.type_comment = self.type_comment

        new_node.decorator_list = []
        copy_location(new_node, self)

        return new_node


class UniqueClass(AST):
    _fields = ('bases', 'keywords', 'body', 'decorator_list')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    @classmethod
    def from_ClassDef(cls, func_def: ast.ClassDef) -> 'UniqueClass':

        new_node = cls()
        new_node.name = func_def.name
        new_node.bases = func_def.bases
        new_node.keywords = func_def.keywords
        new_node.body = func_def.body
        new_node.decorator_list = func_def.decorator_list

        return new_node

    def to_ClassDef(self) -> ast.ClassDef:

        new_node = ast.ClassDef()
        new_node.name = self.name
        new_node.bases = self.bases
        new_node.keywords = self.keywords
        new_node.body = self.body
        new_node.decorator_list = self.decorator_list

        copy_location(new_node, self)
        return new_node


class SingleAssign(AST):
    _fields = ('target', 'value', 'annotation')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    @classmethod
    def create(cls, target: AST, value: Optional[AST] = None, annotation: Optional[AST] = None) -> 'SingleAssign':

        assert value or annotation, "Either value or annotation must be set!"

        return cls(target=target, value=value, annotation=annotation)

    def to_assignment(self) -> Union[AnnAssign, Assign]:

        if self.annotation:
            new_node = AnnAssign(target=self.target, annotation=self.annotation, value=self.value, simple=1)
        else:
            new_node = Assign(targets=[self.target], value=self.value)

        copy_location(new_node, self)
        return new_node


class PhiAssign(AST):
    _fields = ()
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    # Debug print all phi nodes:
    always_active = False

    @property
    def operand_names(self) -> List[str]:
        return [uid for uid, _ in self.operands if uid is not None]

    @property
    def has_undefined(self) -> bool:
        all_defined = all([(uid is not None) for uid, _ in self.operands])
        return not all_defined

    @classmethod
    def create(cls, target: str, variable_name: str, operands: List[str], active: bool = False) -> 'PhiAssign':

        assert all([isinstance(op, tuple) and len(op) == 2 for op in operands])
        assert all([isinstance(uid, str) or (uid is None) for uid, _ in operands])

        operands = list(operands)

        return cls(target=target, variable_name=variable_name, annotation=None, operands=operands, active=active)

    def to_Assign(self) -> Union[Assign, AnnAssign]:

        operands = [Name(id=uid, ctx=Load()) if uid is not None else Constant(value=None) for uid, _ in self.operands]

        if len(operands) == 1:
            value = operands[0]
        else:
            value = Call(func=Name(id='__phi__', ctx=Load()), args=operands, keywords=[])

        target = Name(id=self.target, ctx=Store())

        if self.annotation is None:
            assign_stmt = Assign(targets=[target], value=value)
        else:
            assign_stmt = AnnAssign(target=target, annotation=self.annotation, value=value, simple=1)

        copy_location(assign_stmt, self)

        return assign_stmt

    def cleaned(self) -> Union[None, SingleAssign, 'PhiAssign']:

        # inactive phi node gets removed
        if not (self.active or self.always_active):
            return None

        # phi node with 1 (not None) operand is simple assignment
        elif len(self.operands) == 1 and self.operands[0][0] is not None:
            uid, _ = self.operands[0]
            value = Name(id=uid, ctx=Load())
            target = Name(id=self.target, ctx=Store())
            assign_stmt = SingleAssign.create(target=target, value=value)
            copy_location(assign_stmt, self)
            return assign_stmt

        # else keep phi node
        else:
            return self

    def activate(self) -> None:

        if self.active:
            return

        self.active = True
        for uid, value in self.operands:
            if isinstance(value, PhiAssign):
                value.activate()

    def add_operand(self, new_uid: str, definition: AST) -> None:

        assert isinstance(new_uid, str) or (new_uid is None)

        if any([new_uid == uid for uid, _ in self.operands]):
            return
        else:
            self.operands.append((new_uid, definition))

            # Activate newly added Phi operands
            if self.active and isinstance(definition, PhiAssign):
                definition.activate()


class IfCFG(AST):
    _fields = ('test', 'body', 'orelse', 'exit')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    def __init__(self, *, test: AST, body: List[AST], orelse: List[AST]) -> None:

        self.test = test
        self.body = body
        self.orelse = orelse
        self.exit = []

        self.entry_phis = []
        self.exit_phis = []

    def to_If(self) -> Union[If, List[AST]]:

        if_stmt = If(
            test=self.test,
            body=self.body,
            orelse=self.orelse,
        )
        copy_location(if_stmt, self)

        if self.exit:
            return [if_stmt] + self.exit
        else:
            return if_stmt


@dataclass
class WhileCFG(AST):
    _fields = ('head', 'ifelse', 'exit')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    head: List[AST] = field(default_factory=list)
    ifelse: If = None
    exit: List[AST] = field(default_factory=list)
    entry_phi_lookup: dict = field(default_factory=dict)
    exit_phi_lookup: dict = field(default_factory=dict)

    # def __init__(self, *, head: List[AST], ifelse: If):

    #     super().__init__()

    #     self.head = head
    #     self.ifelse = ifelse
    #     self.exit = []

    #     self.entry_phi_lookup = {}
    #     self.exit_phi_lookup = {}

    def to_While(self) -> List[AST]:

        if self.head:
            return self.to_complex()
        else:
            return self.to_simple()

    def to_complex(self) -> List[AST]:

        while_body = self.head + [self.ifelse]

        new_node = While(
            test=Constant(value=True),
            body=while_body,
            orelse=[],
        )
        copy_location(new_node, self)

        return [new_node] + self.exit

    def to_simple(self) -> List[AST]:

        if_stmt = self.ifelse
        while_test = if_stmt.test
        while_body = if_stmt.body
        while_else = if_stmt.orelse

        if isinstance(while_body[-1], ast.Continue):
            while_body = while_body[:-1]

        if isinstance(while_else[-1], ast.Break):
            while_else = while_else[:-1]

        new_node = While(
            test=while_test,
            body=while_body,
            orelse=while_else,
        )
        copy_location(new_node, self)

        return [new_node] + self.exit


@dataclass
class SSAFor(AST):
    _fields = ('head', 'target', 'iter', 'body', 'orelse', 'type_comment')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    target: AST
    iter: AST

    head: List[AST] = field(default_factory=list)
    body: List[AST] = field(default_factory=list)
    orelse: List[AST] = field(default_factory=list)

    type_comment: str = None


class ForCFG(AST):
    _fields = ('target', 'iter', 'head', 'ifelse', 'exit')
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    def __init__(self, *, target: AST, iter: AST, head: List[AST], ifelse: If, iter_name: str):

        assert isinstance(ifelse, AST)

        self.target = target
        self.iter = iter
        self.head = head
        self.exit = []

        self.entry_phi_lookup = {}
        self.exit_phi_lookup = {}

        self.ifelse = ifelse
        self.iter_name = iter_name

    def to_For(self) -> List[AST]:

        new_body = self.head + [self.ifelse]

        new_node = For(target=self.target, iter=self.iter, body=new_body, orelse=[])
        copy_location(new_node, self)

        return [new_node] + self.exit

    def to_SSAFor(self) -> List[AST]:

        phi_assigns = self.head

        for el in phi_assigns:
            if not isinstance(el, PhiAssign):
                raise ValueError("ForCFG node has non-PhiAssign statement in head block!")

        target = self.target.elts[0]
        body = self.ifelse.body[:-1]
        orelse = self.ifelse.orelse[:-1]

        new_node = For(head=phi_assigns, target=target, iter=self.iter, body=body, orelse=orelse)
        copy_location(new_node, self)

        return phi_assigns + [new_node] + self.exit

    # def to_While(self) -> List[AST]:

    #     next_call = Call(func=Name(id='next', ctx=Load()), args=[self.iter], keywords=[])
    #     iter_stmt = Assign(targets=[self.target], value=next_call)

    #     new_head = self.head + [iter_stmt]
    #     while_node = WhileCFG(head=new_head, ifelse=self.ifelse)
    #     while_node.exit = self.exit

    #     copy_location(next_call, self)
    #     copy_location(iter_stmt, self)
    #     copy_location(while_node, self)

    #     return while_node.to_While()

    def to_WhileCFG(self) -> List[AST]:

        create_iter = SingleAssign.create(target=Name(id=self.iter_name, ctx=Store()),
                                          value=ast.Call(func=Name(id='for_iter', ctx=Load()),
                                                         args=[self.iter],
                                                         keywords=[]))
        copy_location(create_iter, self)

        next_call = Call(func=Name(id='next', ctx=Load()), args=[Name(id=self.iter_name, ctx=Load())], keywords=[])
        iter_stmt = SingleAssign.create(target=self.target, value=next_call)

        new_head = self.head + [iter_stmt]
        while_node = WhileCFG(head=new_head, ifelse=self.ifelse)
        while_node.exit = self.exit

        copy_location(next_call, self)
        copy_location(iter_stmt, self)
        copy_location(while_node, self)

        return [create_iter, while_node]


@dataclass
class Interface(AST):
    _fields = ()
    _attributes = ('lineno', 'col_offset', 'end_lineno', 'end_col_offset')

    uids: List[str] = field(default_factory=list)
    variables: dict = field(default_factory=dict)

    # def __init__(self) -> None:

    #     self.uids = ()
    #     self.variables = {}

    def set_uids(self, uids: List[str]) -> None:
        self.uids = [uid for uid in uids]

    def set_variables(self, variables: List[Tuple[str, AST]]) -> None:
        self.variables = {name: uid for name, (uid, _) in variables.items() if uid is not None}

        for _, (_, phi) in variables.items():
            if isinstance(phi, PhiAssign):
                phi.activate()

    def to_assignments(self, include_dels: bool) -> List[AST]:

        if not self.variables:
            return None

        return_stmts = [ast.Expr(value=Constant(value='##### Re-establish the Interface ####'))]

        for var_name, uid in self.variables.items():

            var_assign = SingleAssign.create(
                target=Name(id=var_name, ctx=Store()),
                value=Name(id=uid, ctx=Load()),
            )
            copy_location(var_assign, self)
            return_stmts.append(var_assign)

        if include_dels and self.uids:
            del_targets = [ast.Name(id=uid, ctx=ast.Del()) for uid in self.uids]
            del_stmt = ast.Delete(targets=del_targets)
            copy_location(del_stmt, self)
            return_stmts.append(del_stmt)

        return return_stmts
