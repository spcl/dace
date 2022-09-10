
import ast
from typing import Tuple, List, Union, NoReturn, Callable
from ast import AST, Assign, Constant, NodeVisitor, NodeTransformer, Name, Load, Store, copy_location, iter_fields, fix_missing_locations

from .ssa_helpers import Counting_UID
from .ssa_nodes import SingleAssign, PhiAssign, UniqueName, SimpleFunction, UniqueClass, IfCFG, ForCFG, WhileCFG, is_simple_expr


class GeneratorCheck(NodeVisitor):

    def visit_NamedExpr(self, node: ast.NamedExpr) -> NoReturn:
        raise SyntaxError('Assignment expressions in generator'
                          ' expressions are not supported!')


ExprAST = AST    # more specific types due to
StmtAST = AST    # ast.AST not discerning these

class SSA_Preprocessor(NodeTransformer):

    # Return values must be
    # for statement nodes: List[StmtAST]
    # for expression nodes: (ExprAST, List[StmtAST])

    ifexp_varname = "ifexp_var"
    assnexp_varname = "assnexp"

    multi_assign_varname = "multi_assign"
    tuple_assign_varname = "tuple_assign"

    while_varname = "while_cond"
    for_iter_varname = "loop_iter"
    for_vals_varname = "loop_value"
    for_next_varname = "has_value"
    with_manager_varname = "ctx_manager"

    lcomp_varname = "list_comp"
    scomp_varname = "set_comp"
    dcomp_varname = "dict_comp"
    gen_varname = "generator_expr"

    decorator_varname = "decorator"
    decorated_varname = "decorated_func"
    lambda_varname = "lambda_expr"
    fstring_varname = "fstr_val"

    slice_lower_varname = "slice_lower"
    slice_upper_varname = "slice_upper"
    slice_step_varname = "slice_step"
    slice_varname = "index_val"
    attribute_varname = "attr_base"

    class_base_varname = "_base"
    func_default_varname = "_default"
    call_varname = "callee_val"
    return_varname = "_result"

    except_cause_varname = "excp_cause"
    except_exc_varname = "exception_val"

    assert_test_varname = "assert_val"
    assert_msg_varname = "assert_msg"


    def __init__(self, uid_func: Callable[[str], str] = Counting_UID()) -> None:

        self.get_unique_id = uid_func

    def generic_visit(self, node: ExprAST) -> Tuple[ExprAST, List[StmtAST]]:

        ret_stmts = []
        
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, AST):
                        value, stmts = self.visit(value)
                        ret_stmts.extend(stmts)
                        if value is None:
                            continue
                        elif not isinstance(value, AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node, stmts = self.visit(old_value)
                ret_stmts.extend(stmts)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node, ret_stmts

    def visit_block(self, block: List[StmtAST]) -> None:

        new_body = []

        for stmt in block:
            new_stmts = self.visit(stmt)
            new_body.extend(new_stmts)

        block[:] = new_body

    def visit_Module(self, node: ast.Module) -> List[ast.Module]:

        self.visit_block(node.body)
        return [node]
    
    def visit_Constant(self, node: Constant) -> Tuple[Constant, List[ExprAST]]:
        return node, []

    ###########################
    # Unsupported Nodes
    ###########################

    def visit_Global(self, node: ast.Global) -> NoReturn:
        raise SyntaxError("Global statements are not supported!")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> NoReturn:
        raise SyntaxError("Nonlocal statements are not supported!")

    def visit_Match(self, node: 'ast.Match') -> NoReturn:
        raise SyntaxError("Match statements are not supported!")

    ###########################
    # Replaced Expression Nodes
    ###########################

    def visit_Expr(self, node: ast.Expr) -> List[StmtAST]:

        expr, add_stmts = self.visit(node.value)
        node.value = expr

        return add_stmts + [node]

    def visit_IfExp(self, node: ast.IfExp) -> Tuple[ExprAST, List[StmtAST]]:

        ret_stmts = []

        test, test_befores = self.visit(node.test)
        body_exp, body_befores = self.visit(node.body)
        else_exp, else_befores = self.visit(node.orelse)

        ifexp_var = self.ifexp_varname
        uid_body = self.get_unique_id(ifexp_var)
        uid_else = self.get_unique_id(ifexp_var)
        uid_phi_ = self.get_unique_id(ifexp_var)

        body_stmt = SingleAssign.create(target=UniqueName(id=uid_body, ctx=Store()), value=body_exp)
        else_stmt = SingleAssign.create(target=UniqueName(id=uid_else, ctx=Store()), value=else_exp)
        copy_location(body_stmt, node)
        copy_location(else_stmt, node)

        if_stmt = IfCFG(
            test = test,
            body = body_befores + [body_stmt],
            orelse = else_befores + [else_stmt],
        )
        copy_location(if_stmt, node)

        assn_stmt = PhiAssign.create(
            target=uid_phi_,
            variable_name=ifexp_var,
            operands=[(uid_body, body_exp), (uid_else, else_exp)],
            active=True,
        )
        copy_location(assn_stmt, node)

        new_node = UniqueName(id=uid_phi_, ctx=Load())
        copy_location(new_node, node)
        
        statements_before = test_befores + [if_stmt, assn_stmt]

        return new_node, statements_before

    def visit_NamedExpr(self, node: ast.NamedExpr) -> Tuple[ExprAST, List[StmtAST]]:
        target, target_stmts = self.visit(node.target)
        value, value_stmts = self.visit(node.value)

        assn_var = self.assnexp_varname
        uid = self.get_unique_id(assn_var)

        assn_stmt1 = SingleAssign.create(target=UniqueName(id=uid, ctx=Store()), value=value)
        assn_stmt2 = SingleAssign.create(target=target, value=UniqueName(id=uid, ctx=Load()))
        copy_location(assn_stmt1, node)
        copy_location(assn_stmt2, node)

        new_node = UniqueName(id=uid, ctx=Load())
        copy_location(new_node, node)

        statements_before = value_stmts + target_stmts + [assn_stmt1, assn_stmt2]
        
        return new_node, statements_before

    def visit_ListComp(self, node: ast.ListComp) -> Tuple[ExprAST, List[StmtAST]]:

        comp_var = self.lcomp_varname
        uid = self.get_unique_id(comp_var)
        new_node = UniqueName(id=uid, ctx=Load())
        copy_location(new_node, node)

        assn_stmt = SingleAssign.create(
            target=UniqueName(id=uid, ctx=Store()),
            value=ast.List(elts=[], ctx=Load()),
        )
        copy_location(assn_stmt, node)

        element = ast.Expr(ast.Call(
            func=ast.Attribute(value=new_node, attr='append', ctx=Load()),
            args=[node.elt], keywords=[]
        ))
        copy_location(element, node)

        comp_body = self.replace_comprehension([element], node.generators)

        statements_before = [assn_stmt] + comp_body

        return new_node, statements_before

    def visit_SetComp(self, node: ast.SetComp) -> Tuple[ExprAST, List[StmtAST]]:

        comp_var = self.scomp_varname
        uid = self.get_unique_id(comp_var)
        new_node = UniqueName(id=uid, ctx=Load())

        assn_stmt = SingleAssign.create(
            target=UniqueName(id=uid, ctx=Store()),
            value=ast.Call(func=Name(id='set', ctx=Load()), args=[], keywords=[]),
        )
        copy_location(assn_stmt, node)

        element = ast.Expr(ast.Call(
            func=ast.Attribute(value=new_node, attr='add', ctx=Load()),
            args=[node.elt], keywords=[]
        ))
        copy_location(element, node)

        comp_body = self.replace_comprehension([element], node.generators)

        statements_before = [assn_stmt] + comp_body

        return new_node, statements_before

    def visit_DictComp(self, node: ast.DictComp) -> Tuple[ExprAST, List[StmtAST]]:

        comp_var = self.dcomp_varname
        uid = self.get_unique_id(comp_var)
        new_node = UniqueName(id=uid, ctx=Load())

        assn_stmt = SingleAssign.create(
            target=UniqueName(id=uid, ctx=Store()),
            value=ast.Dict(keys=[], values=[]),
        )
        copy_location(assn_stmt, node)

        element = Assign(
            targets=[ast.Subscript(value=new_node, slice=node.key)],
            value=node.value,
        )
        copy_location(element, node)

        comp_body = self.replace_comprehension([element], node.generators)

        statements_before = [assn_stmt] + comp_body

        return new_node, statements_before

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Tuple[ExprAST, List[StmtAST]]:

        # ensure there are no NamedExpr nodes
        GeneratorCheck().visit(node)

        generator_var = self.gen_varname
        uid = self.get_unique_id(generator_var)
        new_node = ast.Call(func=UniqueName(id=uid, ctx=Load()), args=[], keywords=[])
        copy_location(new_node, node)

        element = ast.Expr(value=ast.Yield(value=node.elt))
        copy_location(element, node)

        comp_body = self.replace_comprehension([element], node.generators)

        generator_def = SimpleFunction(
            name=uid,
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=comp_body, 
            decorator_list=[], returns=None, type_comment=None
        )
        copy_location(generator_def, node)
        fix_missing_locations(new_node)

        return new_node, [generator_def]

    def replace_comprehension(self, body: List[AST], comprehensions: List[ast.comprehension]) -> List[StmtAST]:

        assert len(comprehensions) > 0

        new_body = body

        for comp in reversed(comprehensions):

            for if_exp in comp.ifs:
                new_node = ast.If(
                    test=if_exp,
                    body=new_body,
                    orelse=[]
                )
                copy_location(new_node, body[-1])
                new_body = [new_node]

            new_node = ast.For(
                target=comp.target, 
                iter=comp.iter,
                body=new_body,
                orelse=[],
            )
            copy_location(new_node, body[-1])
            new_body = [new_node]

        self.visit_block(new_body)

        return new_body

    ###########################
    # Simple Statements Nodes
    ###########################

    def visit_Delete(self, node: ast.Delete) -> List[ast.Delete]:
        return [node]

    def visit_Pass(self, node: ast.Pass) -> List[ast.Pass]:
        return [node]

    def visit_Break(self, node: ast.Break) -> List[ast.Break]:
        return [node]

    def visit_Continue(self, node: ast.Continue) -> List[ast.Continue]:
        return [node]
    
    def visit_Import(self, node: ast.Import) -> List[ast.Import]:
        return [node]

    def visit_ImportFrom(self, node: ast.ImportFrom) -> List[ast.ImportFrom]:
        return [node]

    def visit_Raise(self, node: ast.Raise) -> List[StmtAST]:
        add_stmts = []

        if node.cause:
            new_cause, stmts1 = self.visit(node.cause)
            node.cause, stmts2 = self.simplify_value(new_cause, self.except_cause_varname)
            add_stmts.extend(stmts1)
            add_stmts.extend(stmts2)

        if node.exc:
            new_exc, stmts1 = self.visit(node.exc)
            node.exc = new_exc
            add_stmts.extend(stmts1)

        return add_stmts + [node]

    def visit_Assert(self, node: ast.Assert) -> List[StmtAST]:

        add_stmts = []

        new_test, stmts1 = self.visit(node.test)
        node.test, stmts2 = self.simplify_value(new_test, self.assert_test_varname)
        add_stmts.extend(stmts1)
        add_stmts.extend(stmts2)

        if node.msg:
            new_msg, stmts1 = self.visit(node.msg)
            node.msg, stmts2 = self.simplify_value(new_msg, self.assert_msg_varname)
            add_stmts.extend(stmts1)
            add_stmts.extend(stmts2)

        return add_stmts + [node]

    ###########################
    # Replaced Assignment Nodes
    ###########################

    def visit_Assign(self, node: ast.Assign) -> List[StmtAST]:
        # Assign(expr* targets, expr value, string? type_comment)

        targets = node.targets
        value, value_stmts = self.visit(node.value)
        
        if len(targets) == 1:
            target, add_assigns = self.visit_target(targets[0])

            new_node = SingleAssign.create(target=target, value=value)
            copy_location(new_node, node)

            return value_stmts + [new_node] + add_assigns

        assign_var = self.multi_assign_varname
        uid_assign = self.get_unique_id(assign_var)

        new_node = SingleAssign.create(target=UniqueName(id=uid_assign, ctx=Store()), value=value)
        copy_location(new_node, node)
        
        new_assigns = [new_node]
        nested_assigns = []

        for target in targets:
            target, add_assigns = self.visit_target(target)
            new_target = SingleAssign.create(target=target, value=UniqueName(id=uid_assign, ctx=Store()))
            copy_location(new_target, node)
            new_assigns.append(new_target)
            nested_assigns.extend(add_assigns)

        new_assigns.extend(nested_assigns)

        return value_stmts + new_assigns

    def visit_AnnAssign(self, node: ast.AnnAssign) -> List[StmtAST]:
        # AnnAssign(expr target, expr annotation, expr? value, int simple)

        value = node.value

        add_stmts = []

        if value:
            value, value_stmts = self.visit(value)
            add_stmts.extend(value_stmts)

        annot = node.annotation
        # annot, annot_stmts = self.visit(annot)
        # add_stmts.extend(annot_stmts)

        target, add_assigns = self.visit_target(node.target)

        new_node = SingleAssign.create(target=target, value=value, annotation=annot)
        copy_location(new_node, node)

        return add_stmts + [new_node] + add_assigns

    def visit_AugAssign(self, node: ast.AugAssign) -> List[StmtAST]:

        target = node.target
        value, value_stmts = self.visit(node.value)
        target, add_assigns = self.visit_target(target)

        new_node = SingleAssign.create(target=target, value=value)
        copy_location(new_node, node)

        return value_stmts + add_assigns + [new_node]

    def visit_target(self, target: AST) -> Tuple[AST, List[StmtAST]]:

        target, target_stmts = self.visit(target)
        additional_assigns = target_stmts

        # Simple Assignments
        if isinstance(target, (UniqueName, Name, ast.Subscript, ast.Attribute)):
            return target, additional_assigns

        # Tuple/List Assignments
        elif isinstance(target, (ast.Tuple, ast.List)):

            assert isinstance(target.ctx, Store), f"target.ctx is not ast.Store() for target {ast.unparse(target)}"

            assign_var = self.tuple_assign_varname
            uid_assign = self.get_unique_id(assign_var)
            new_target = UniqueName(id=uid_assign, ctx=Store())
            copy_location(new_target, target)

            pos_stmts = []
            neg_stmts = []
            nested_pos = []
            nested_neg = []
            starred = None

            # Positive index targets (before starred)
            i_pos = -1
            for i_pos, element in enumerate(target.elts):

                if isinstance(element, ast.Starred):
                    starred = element.value
                    break

                element, nested_stmts = self.visit_target(element)
                nested_pos.extend(nested_stmts)
                
                subscript = ast.Subscript(value=UniqueName(id=uid_assign, ctx=Load()), 
                                          slice=Constant(value=i_pos), ctx=Load())
                pos_stmts.append(SingleAssign.create(target=element, value=subscript))

            # Negative index targets (after starred)
            for i, element in enumerate(target.elts[i_pos+1:], 1):
                i_neg = -i

                if isinstance(element, ast.Starred):
                    raise SyntaxError('two starred expressions in assignment (SSA_Preprocessor')

                element, nested_stmts = self.visit_target(element)
                nested_neg.extend(reversed(nested_stmts))
                
                subscript = ast.Subscript(value=UniqueName(id=uid_assign, ctx=Load()), 
                                          slice=Constant(value=i_neg), ctx=Load())
                neg_stmts.append(SingleAssign.create(target=element, value=subscript))

            # Starred index: all remaining elements
            if starred is not None:

                try:
                    star_slice = ast.Slice(lower=Constant(value=i_pos), upper=Constant(value=i_neg))
                except UnboundLocalError:
                    star_slice = ast.Slice(lower=Constant(value=i_pos))

                element, nested_stmts = self.visit_target(starred)
                nested_pos.extend(nested_stmts)
                
                subscript = ast.Subscript(value=UniqueName(id=uid_assign, ctx=Load()), slice=star_slice, ctx=Load())
                pos_stmts.append(SingleAssign.create(target=element, value=subscript))

                additional_assigns.extend(pos_stmts)
                additional_assigns.extend(reversed(neg_stmts))
                additional_assigns.extend(nested_pos)
                additional_assigns.extend(reversed(nested_neg))

            else:
                # no Starred thus no negative indexing
                additional_assigns.extend(pos_stmts)
                additional_assigns.extend(nested_pos)

            for new_stmt in additional_assigns:
                copy_location(new_stmt, target)

            return new_target, additional_assigns

        # Illegal or unsupported Assignments
        elif isinstance(target, ast.Starred):
            raise SyntaxError('starred assignment target must be in a list or tuple (in SSA_Preprocessor)')
        else:
            raise NotImplementedError(f'Node type {type(target)} as assignment target')


    ###########################
    # Replaced Loop Nodes
    ###########################

    def visit_Try(self, node: ast.Try) -> List[StmtAST]:

        import warnings
        warnings.warn("Exceptions are disabled! Except clauses are removed.")

        body_block = node.body
        else_block = node.orelse
        final_block = node.finalbody

        self.visit_block(body_block)
        self.visit_block(else_block)
        self.visit_block(final_block)

        return body_block + else_block + final_block

    def visit_If(self, node: ast.If) -> List[StmtAST]:
        """turns an ast.If node into a IfCFG node where header, body and orelse fields are all blocks"""

        test_expr, test_stmts = self.visit(node.test)
        body_block = node.body
        else_block = node.orelse

        self.visit_block(body_block)
        self.visit_block(else_block)

        new_node = IfCFG(test=test_expr, body=body_block, orelse=else_block)
        copy_location(new_node, node)

        return test_stmts + [new_node]

    def visit_While(self, node: ast.While) -> List[StmtAST]:
        """turns an ast.While node into a WhileCFG node"""

        test_expr, test_stmts = self.visit(node.test)
        body_block = node.body
        else_block = node.orelse

        self.visit_block(body_block)
        self.visit_block(else_block)

        head = test_stmts
        if_body = body_block + [ast.Continue()]
        if_else = else_block + [ast.Break()]

        ifelse = ast.If(test=test_expr, body=if_body, orelse=if_else)
        copy_location(ifelse, node)

        new_node = WhileCFG(head=head, ifelse=ifelse)
        copy_location(new_node, node)
        fix_missing_locations(new_node)

        return [new_node]

    def visit_For(self, node: ast.For) -> List[StmtAST]:
        """turns an ast.For node into a ForCFG node"""
        #| For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)

        iter, iter_stmts = self.visit(node.iter)
        target = node.target
        body_block = node.body
        else_block = node.orelse

        self.visit_block(body_block)
        self.visit_block(else_block)

        iter_var = self.for_iter_varname
        value_var = self.for_vals_varname
        next_var = self.for_next_varname
        uid_iter = self.get_unique_id(iter_var)
        uid_value = self.get_unique_id(value_var)
        uid_next = self.get_unique_id(next_var)

        new_target, add_assigns = self.visit_target(target)
        target_assign = SingleAssign.create(target=new_target, value=UniqueName(id=uid_value, ctx=Load()))
        copy_location(target_assign, node)

        setup = iter_stmts
        head = []
        if_body = [target_assign, *add_assigns, *body_block, ast.Continue()]
        if_else = else_block + [ast.Break()]

        test_expr = UniqueName(id=uid_next, ctx=Load())
        ifelse = ast.If(test=test_expr, body=if_body, orelse=if_else)

        for_target = ast.Tuple(elts=[UniqueName(id=uid_value, ctx=Store()), UniqueName(id=uid_next, ctx=Store())], ctx=Store())
        # for_iter = UniqueName(id=iter, ctx=Load())

        new_node = ForCFG(target=for_target, iter=iter, head=head, ifelse=ifelse, iter_name=uid_iter)
        copy_location(new_node, node)
        fix_missing_locations(new_node)

        return setup + [new_node]
    
    def visit_With(self, node: ast.With) -> List[StmtAST]:

        add_stmts = []
        self.visit_block(node.body)

        for item in node.items:
            expr = item.context_expr
            expr, stmts1 = self.visit(expr)
            expr, stmts2 = self.simplify_value(expr, self.with_manager_varname)
            item.context_expr = expr
            add_stmts.extend(stmts1)
            add_stmts.extend(stmts2)

        return add_stmts + [node]

    # Async variants are the same as the normal ones
    visit_AsyncFor = visit_For
    visit_AsyncWhile = visit_While
    visit_AsyncWith = visit_With

    #################################
    # Function, Class & Module Nodes
    #################################

    def visit_Lambda(self, node: ast.Lambda) -> Tuple[ExprAST, List[StmtAST]]:

        lambda_var = self.lambda_varname
        uid_lambda = self.get_unique_id(lambda_var)

        body_expr, body_stmts = self.visit(node.body)

        return_stmt = ast.Return(value=body_expr)
        copy_location(return_stmt, node)

        new_body = body_stmts + [return_stmt]

        lambda_func = SimpleFunction(
            name=uid_lambda,
            args=node.args,
            body=new_body,
            decorator_list=[], returns=None, type_comment=None
        )
        copy_location(lambda_func, node)

        new_node = UniqueName(id=uid_lambda, ctx=Load())
        statements_before = [lambda_func]
        copy_location(new_node, node)

        return new_node, statements_before

    def visit_FunctionDef(self, node: ast.FunctionDef) -> List[StmtAST]:
        # FunctionDef(identifier name, arguments args,
        #             stmt* body, expr* decorator_list, expr? returns,
        #             string? type_comment)

        func_name = node.name
        body = node.body
        
        self.visit_block(body)

        # Remove complex default values
        args = node.args
        default_varname = func_name + self.func_default_varname
        args_stmts = []

        new_defaults = []
        for default in args.defaults:
            new_default, stmts1 = self.visit(default)
            simple_default, stmts2 = self.simplify_value(new_default, default_varname)

            args_stmts.extend(stmts1)
            args_stmts.extend(stmts2)
            new_defaults.append(simple_default)
        args.defaults = new_defaults

        new_kwdefaults = []
        for default in args.kw_defaults:
            if default is not None:
                new_default, stmts1 = self.visit(default)
                simple_default, stmts2 = self.simplify_value(new_default, default_varname)

                args_stmts.extend(stmts1)
                args_stmts.extend(stmts2)
                new_kwdefaults.append(simple_default)
            else:
                new_kwdefaults.append(None)
        args.kw_defaults = new_kwdefaults

        # Remove decorators
        decorators = node.decorator_list
        func_expr = UniqueName(id=func_name, ctx=Load())
        copy_location(func_expr, node)
        new_value, dec_stmts = self.replace_decorators(func_expr, decorators)
        node.decorator_list = []

        # Create SimpleFunction
        new_node = SimpleFunction.from_FunctionDef(node)
        copy_location(new_node, node)

        assn_stmt = SingleAssign.create(
            target=Name(id=func_name, ctx=Store()),
            value=new_value,
        )
        copy_location(assn_stmt, node)

        del_stmt = ast.Delete(targets=[UniqueName(id=func_name, ctx=ast.Del())])
        copy_location(del_stmt, node)

        return args_stmts + [new_node] + dec_stmts + [ assn_stmt, del_stmt]
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_Return(self, node: ast.Return) -> List[StmtAST]:
        new_value, stmts1 = self.visit(node.value)
        node.value, stmts2 = self.simplify_value(new_value, self.return_varname)
        return stmts1 + stmts2 + [node]

    def visit_ClassDef(self, node: ast.ClassDef) -> List[StmtAST]:
        #  ClassDef(identifier name, expr* bases,
        #           keyword* keywords, stmt* body, 
        #           expr* decorator_list)

        class_name = node.name

        self.visit_block(node.body)
        
        # Remove complex bases
        base_varname = class_name + self.class_base_varname
        args_stmts = []

        new_bases = []
        for base in node.bases:
            new_base, stmts1 = self.visit(base)
            simple_base, stmts2 = self.simplify_value(new_base, base_varname)

            args_stmts.extend(stmts1)
            args_stmts.extend(stmts2)
            new_bases.append(simple_base)
        node.bases = new_bases

        for keyword in node.keywords:
            new_val, stmts1 = self.visit(keyword.value)
            keyword.value, stmts2 = self.simplify_value(new_val, base_varname)
            args_stmts.extend(stmts1)
            args_stmts.extend(stmts2)

        # Remove decorators
        decorators = node.decorator_list
        class_expr = UniqueName(id=class_name, ctx=Load())
        copy_location(class_expr, node)
        new_value, dec_stmts = self.replace_decorators(class_expr, decorators)
        node.decorator_list = []

        # Create UniqueClass
        new_node = UniqueClass.from_ClassDef(node)
        copy_location(new_node, node)
        
        assn_stmt = SingleAssign.create(
            target=Name(id=class_name, ctx=Store()),
            value=new_value,
        )
        copy_location(assn_stmt, node)

        del_stmt = ast.Delete(targets=[UniqueName(id=class_name, ctx=ast.Del())])
        copy_location(del_stmt, node)

        return args_stmts + [new_node] + dec_stmts + [ assn_stmt, del_stmt]
    
    def replace_decorators(self, element: ExprAST, decorators: List[ExprAST]) -> Tuple[ExprAST, List[StmtAST]]:

        add_stmts = []
        prev_element = element

        for decorator in reversed(decorators):
            simple_dec, stmts_dec = self.simplify_value(decorator, self.decorator_varname)
            prev_element = ast.Call(func=simple_dec, args=[prev_element], keywords=[])
            copy_location(prev_element, element)
            simple_element, stmts_el = self.simplify_value(prev_element, self.decorated_varname)
            add_stmts.extend(stmts_dec)
            add_stmts.extend(stmts_el)
            prev_element = simple_element
        
        return prev_element, add_stmts

    #################################
    # Simplified Nodes
    #################################

    def visit_Call(self, node: ast.Call) -> Tuple[ExprAST, List[StmtAST]]:
        # Call(expr func, expr* args, keyword* keywords)

        add_stmts = []
        call_target, stmts1 = self.visit(node.func)
        add_stmts.extend(stmts1)
        if not isinstance(call_target, (Name, UniqueName, Constant, ast.Attribute)):
            node.func, stmts2 = self.simplify_value(call_target, self.call_varname)
            add_stmts.extend(stmts2)

        new_args = []
        for arg in node.args:
            new_arg, stmts = self.visit(arg)
            add_stmts.extend(stmts)
            new_args.append(new_arg)
        node.args = new_args

        for keyword in node.keywords:
            new_val, stmts = self.visit(keyword.value)
            add_stmts.extend(stmts)
            keyword.value = new_val

        return node, add_stmts

    def visit_Subscript(self, node: ast.Subscript) -> Tuple[ExprAST, List[StmtAST]]:

        slice = node.slice
        add_stmts = []

        if isinstance(slice, ast.Slice):
            
            if hasattr(slice, 'lower') and (slice.lower is not None):
                lower_slice, stmts1 = self.visit(slice.lower)
                slice.lower, stmts2 = self.simplify_value(lower_slice, self.slice_lower_varname)
                add_stmts.extend(stmts1)
                add_stmts.extend(stmts2)
            if hasattr(slice, 'upper') and (slice.upper is not None):
                upper_slice, stmts1 = self.visit(slice.upper)
                slice.upper, stmts2 = self.simplify_value(upper_slice, self.slice_upper_varname)
                add_stmts.extend(stmts1)
                add_stmts.extend(stmts2)
            if hasattr(slice, 'step') and (slice.step is not None):
                step_slice, stmts1 = self.visit(slice.step)
                slice.step, stmts2 = self.simplify_value(step_slice, self.slice_step_varname)
                add_stmts.extend(stmts1)
                add_stmts.extend(stmts2)
            
        elif not is_simple_expr(slice):
            node_slice, stmts1 = self.visit(node.slice)
            node.slice, stmts2 = self.simplify_value(node_slice, self.slice_varname)
            add_stmts.extend(stmts1)
            add_stmts.extend(stmts2)
        
        return node, add_stmts

    def visit_Attribute(self, node: ast.Attribute) -> Tuple[ExprAST, List[StmtAST]]:

        new_value, stmts1 = self.visit(node.value)
        node.value, stmts2 = self.simplify_value(new_value, self.attribute_varname)

        return node, stmts1 + stmts2

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Tuple[ExprAST, List[StmtAST]]:

        add_stmts = []
        fval_var = self.fstring_varname
        fval_nodes = [v for v in node.values if isinstance(v, ast.FormattedValue)]

        for fval_node in fval_nodes:
            new_value, stmts1 = self.visit(fval_node.value)
            fval_node.value, stmts2 = self.simplify_value(new_value, fval_var)
            add_stmts.extend(stmts1)
            add_stmts.extend(stmts2)

        return node, add_stmts

    def simplify_value(self, old_value: AST, name: str) -> Tuple[ExprAST, List[StmtAST]]:

        add_stmts = []

        if is_simple_expr(old_value):
            return old_value, add_stmts

        else:
            # store old value
            uid = self.get_unique_id(name)
            assn_stmt = SingleAssign.create(target=UniqueName(id=uid, ctx=Store()), value=old_value)
            copy_location(assn_stmt, old_value)
            add_stmts.append(assn_stmt)

            # load new value
            new_value = UniqueName(id=uid, ctx=Load())
            copy_location(new_value, old_value)

            return new_value, add_stmts
