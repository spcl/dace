
import ast
from ast import AST, AnnAssign, Assign, Call, Constant, If, Name, For, While, Load, Store
from itertools import chain


def is_simple_expr(node: AST) -> bool:
    return isinstance(node, (Constant, Name, UniqueName))


class UniqueName(AST):
    _fields = ()

    def to_Name(self):

        return Name(id=self.id, ctx=self.ctx)


class SimpleFunction(AST):
    _fields = ('args', 'body', 'returns',)

    @classmethod
    def from_FunctionDef(cls, func_def):

        args = func_def.args
        body = func_def.body

        # Check certain "simple" attributes about the function
        simple_defaults = [is_simple_expr(default) or default is None
                           for default in chain(args.defaults, args.kw_defaults)]

        # 1) Only simple default values
        if not all(simple_defaults):
            raise ValueError('All function defaults must be simple!')

        # 2) no decorators
        if func_def.decorator_list:
            raise ValueError('Decorator list must be empty!')

        # 3) implicit return is added
        if not isinstance(body[-1], ast.Return):
            body.append(ast.Return())

        new_node = cls()
        new_node.name = func_def.name
        new_node.args = args
        new_node.body = body
        new_node.returns = func_def.returns
        new_node.type_comment = func_def.type_comment

        return new_node

    def to_FunctionDef(self):

        new_node = ast.FunctionDef()
        new_node.name = self.name
        new_node.args = self.args
        new_node.body = self.body
        new_node.returns = self.returns
        new_node.type_comment = self.type_comment

        new_node.decorator_list = []

        return new_node


class UniqueClass(AST):
    _fields = ('bases', 'keywords', 'body', 'decorator_list',)
    
    @classmethod
    def from_ClassDef(cls, func_def):

        new_node = cls()
        new_node.name = func_def.name
        new_node.bases = func_def.bases
        new_node.keywords = func_def.keywords
        new_node.body = func_def.body
        new_node.decorator_list = func_def.decorator_list

        return new_node

    def to_ClassDef(self):

        new_node = ast.ClassDef()
        new_node.name = self.name
        new_node.bases = self.bases
        new_node.keywords = self.keywords
        new_node.body = self.body
        new_node.decorator_list = self.decorator_list

        return new_node


class SingleAssign(AST):
    _fields = ('target', 'value', 'annotation')

    @classmethod
    def create(cls, target, value=None, annotation=None):

        assert value or annotation, "Either value or annotation must be set!"

        return cls(target=target, value=value, annotation=annotation)

    def to_assignment(self):

        if self.annotation:
            new_node = AnnAssign(target=self.target, annotation=self.annotation, value=self.value, simple=1)
        else:
            new_node = Assign(targets=[self.target], value=self.value)

        return new_node


class PhiAssign(AST):
    _fields = ()

    # Debug print all phi nodes:
    always_active = False

    @property
    def operand_names(self):
        return [uid for uid, _ in self.operands if uid is not None]

    @property
    def has_undefined(self):
        all_defined = all([(uid is not None) for uid, _ in self.operands])
        return not all_defined

    @classmethod
    def create(cls, target, operands, active=False):

        assert all([isinstance(op, tuple) and len(op) == 2 for op in operands])
        assert all([isinstance(uid, str) or (uid is None) for uid, _ in operands])

        operands = list(operands)

        return cls(target=target, annotation=None, operands=operands, active=active)

    def to_Assign(self):

        operands = [Name(id=uid, ctx=Load()) if uid is not None else Constant(value=None)
                    for uid, _ in self.operands]
        
        if len(operands) == 1:
            value = operands[0]
        else:
            value = Call(func=Name(id='__phi__', ctx=Load()), args=operands, keywords=[])

        target = Name(id=self.target, ctx=Store())
        
        if self.annotation is None:
            assign_stmt = Assign(targets=[target], value=value)
        else:
            assign_stmt = AnnAssign(target=target, annotation=self.annotation, value=value, simple=1)

        return assign_stmt

    def cleaned(self):

        # inactive phi node gets removed
        if not (self.active or self.always_active):
            return None

        # phi node with 1 (not None) operand is simple assignment
        elif len(self.operands) == 1 and self.operands[0][0] is not None:
            uid, _ = self.operands[0]
            value = Name(id=uid, ctx=Load())
            target = Name(id=self.target, ctx=Store())
            assign_stmt = SingleAssign.create(target=target, value=value)
            return assign_stmt
        
        # else keep phi node
        else:
            return self

    def activate(self):

        if self.active:
            return
        
        self.active = True
        for uid, value in self.operands:
            if isinstance(value, PhiAssign):
                value.activate()

    def add_operand(self, new_uid, definition):
        
        assert isinstance(new_uid, str) or (new_uid is None)

        if any([new_uid == uid for uid, _ in self.operands]):
            return
        else:
            self.operands.append((new_uid, definition))


class IfCFG(AST):
    _fields = ('test', 'body', 'orelse', 'exit',)

    def __init__(self, *, test, body, orelse):

        self.test = test
        self.body = body
        self.orelse = orelse
        self.exit = []

        self.entry_phis = []
        self.exit_phis = []

    def to_If(self):

        if_stmt = ast.If(
            test = self.test,
            body = self.body,
            orelse = self.orelse,
        )

        if self.exit:
            return [if_stmt] + self.exit
        else:
            return if_stmt


class WhileCFG(AST):
    _fields = ('head', 'ifelse', 'exit')

    def __init__(self, *, head, ifelse):

        self.head = head
        self.ifelse = ifelse
        self.exit = []

        self.entry_phi_lookup = {}
        self.exit_phi_lookup = {}

    def to_While(self):

        if self.head:
            return self.to_complex()
        else:
            return self.to_simple()

    def to_complex(self):

        while_body = self.head + [self.ifelse]

        new_node = While(
            test=Constant(value=True),
            body=while_body,
            orelse=[],
        )

        return [new_node] + self.exit

    def to_simple(self):

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

        return [new_node] + self.exit


class ForCFG(AST):
    _fields = ('target', 'iter', 'head', 'ifelse', 'exit')

    def __init__(self, *, target, iter, head, ifelse):

        assert isinstance(ifelse, AST)

        self.target = target
        self.iter = iter
        self.head = head
        # self.ifelse = ifelse
        self.exit = []

        self.entry_phi_lookup = {}
        self.exit_phi_lookup = {}


        self._x = None
        self.ifelse = ifelse

    @property
    def ifelse(self):
        return self._x

    @ifelse.setter
    def ifelse(self, value):
        assert isinstance(value, AST)

        self._x = value

    def to_For(self):

        new_body = self.head + [self.ifelse]

        new_node = ast.For(
            target=self.target,
            iter=self.iter,
            body=new_body,
            orelse=[]
        )

        return [new_node] + self.exit

    def to_While(self):

        next_call = Call(func=Name(id='next', ctx=Load()), args=[self.iter], keywords=[])
        iter_stmt = Assign(targets=[self.target], value=next_call)

        new_head = self.head + [iter_stmt]

        while_node = WhileCFG(head=new_head, ifelse=self.ifelse)
        while_node.exit = self.exit

        return while_node.to_While()


class Interface(AST):
    _fields = ()

    def __init__(self):

        self.uids = ()
        self.variables = {}

    def set_uids(self, uids):
        self.uids = [uid for uid in uids]

    def set_variables(self, variables):
        self.variables = {name: uid for name, (uid, _) in variables.items() if uid is not None}

        for _, (_, phi) in variables.items():
            if isinstance(phi, PhiAssign):
                phi.activate()

    def to_assignments(self):

        if not self.variables:
            return None

        var_assigns = [ast.Expr(value=Constant(value='##### Re-establish the Interface ####'))]

        for var_name, uid in self.variables.items():

            var_assign = SingleAssign.create(
                target=Name(id=var_name, ctx=Store()),
                value=Name(id=uid, ctx=Load()),
            )
            var_assigns.append(var_assign)

        del_targets = [ast.Name(id=uid, ctx=ast.Del()) for uid in self.uids]
        del_stmt = ast.Delete(targets=del_targets)

        return var_assigns + [del_stmt]
