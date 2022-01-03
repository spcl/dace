# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import sys
from ply import yacc
from . import lexer
import copy
import dace

from .ast_node import AST_Node, AST_Statements
from .ast_values import AST_Ident, AST_Constant
from .ast_expression import AST_BinExpression, AST_UnaryExpression
from .ast_matrix import AST_Matrix_Row, AST_Matrix, AST_Transpose
from .ast_assign import AST_Assign
from .ast_function import AST_Argument, AST_BuiltInFunCall, AST_FunCall, AST_Function, AST_EndFunc
from .ast_range import AST_RangeExpression
from .ast_loop import AST_ForLoop
from .ast_nullstmt import AST_NullStmt, AST_Comment, AST_EndStmt

tokens = lexer.tokens

precedence = (
    ("right", "COMMA"),
    ("right", "DOTDIVEQ", "DOTMULEQ", "EQ", "EXPEQ", "MULEQ", "MINUSEQ", "DIVEQ", "PLUSEQ", "OREQ", "ANDEQ"),
    ("nonassoc", "HANDLE"),
    ("left", "COLON"),
    ("left", "ANDAND", "OROR"),
    ("left", "EQEQ", "NE", "GE", "LE", "GT", "LT"),
    ("left", "OR", "AND"),
    ("left", "PLUS", "MINUS"),
    ("left", "MUL", "DIV", "DOTMUL", "DOTDIV", "BACKSLASH"),
    ("right", "UMINUS", "NEG"),
    ("right", "TRANSPOSE"),
    ("right", "EXP", "DOTEXP", "POW"),
    ("nonassoc", "LPAREN", "RPAREN", "RBRACE", "LBRACE"),
    ("left", "FIELD", "DOT", "PLUSPLUS", "MINUSMINUS"),
)


def p_top(p):
    """
    top :
        | top stmt
      """

    if len(p) == 1:
        retval = AST_Statements(None, [])
        p[0] = retval
    else:
        retval = copy.deepcopy(p[1])
        retval.append_statement(p[2])
        p[0] = retval


def p_end(p):
    """
    top : top END_STMT
    """
    retval = copy.deepcopy(p[1])
    retval.append_statement(AST_EndStmt(None))
    p[0] = retval


def p_end_function(p):
    """
    top : top END_FUNCTION
    """
    retval = copy.deepcopy(p[1])
    retval.append_statement(AST_EndFunc(None))
    p[0] = retval


def p_arg1(p):
    """
    arg1 : IDENT
    """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di = dace.dtypes.DebugInfo(startl, startc, endl, endc)
    p[0] = AST_Ident(di, p[1])


def p_arg2(p):
    """
    arg1 : NUMBER
         | STRING
    """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di = dace.dtypes.DebugInfo(startl, startc, endl, endc)
    p[0] = AST_Constant(di, p[1])


def p_global(p):
    """
     arg1 : GLOBAL
    """
    raise NotImplementedError("global not implemented")


def p_arg_list(p):
    """
    arg_list : ident_init_opt
             | arg_list COMMA ident_init_opt
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_args(p):
    """
    args : arg1
         | args arg1
    """
    raise NotImplementedError("args not implemented")


def p_break_stmt(p):
    """ break_stmt : BREAK SEMI """
    raise NotImplementedError("break not implemented")


def p_case_list(p):
    """
    case_list :
              | CASE expr sep stmt_list_opt case_list
              | CASE expr error stmt_list_opt case_list
              | OTHERWISE stmt_list
    """
    raise NotImplementedError("case not implemented")


def p_cellarray(p):
    """
    cellarray : LBRACE RBRACE
              | LBRACE matrix_row RBRACE
              | LBRACE matrix_row SEMI RBRACE
    """
    startl, endl = p.linespan(0)
    startc, endc = p.lexspan(0)
    di = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    if len(p) == 3:
        p[0] = AST_Matrix(di, [])
    else:
        p[0] = AST_Matrix(di, p[2])


def p_cellarray_2(p):
    """
    cellarray : LBRACE expr_list RBRACE
    """
    p[0] = AST_Matrix(di, [AST_Matrix_Row(p[2])])


def p_cellarrayref(p):
    """expr : expr LBRACE expr_list RBRACE
            | expr LBRACE RBRACE
    """
    raise NotImplementedError("cellarrayref not implemented")


def p_command(p):
    """
    command : ident args SEMI
    """
    raise NotImplementedError("commands not implemented")


####################


def p_comment_stmt(p):
    """
    comment_stmt : COMMENT
    """
    di = None
    p[0] = AST_Comment(di, p[1])


def p_concat_list1(p):
    """
    matrix_row : expr_list SEMI expr_list
    """

    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    startl, endl = p.linespan(3)
    startc, endc = p.lexspan(3)
    di3 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = [AST_Matrix_Row(di1, p[1]), AST_Matrix_Row(di3, p[3])]


def p_concat_list2(p):
    """
    matrix_row : matrix_row SEMI expr_list
    """
    startl, endl = p.linespan(3)
    startc, endc = p.lexspan(3)
    di3 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = p[1] + [AST_Matrix_Row(di3, p[3])]


def p_continue_stmt(p):
    "continue_stmt : CONTINUE SEMI"
    raise NotImplementedError("continue needs to be implemented")


def p_elseif_stmt(p):
    """
    elseif_stmt :
                | ELSE stmt_list_opt
                | ELSEIF expr sep stmt_list_opt elseif_stmt
                | ELSEIF LPAREN expr RPAREN stmt_list_opt elseif_stmt
    """
    raise NotImplementedError("elseif needs to be implemented")


def p_error_stmt(p):
    """
    error_stmt : ERROR_STMT SEMI
    """
    raise NotImplementedError("error stmt")


def p_expr(p):
    """expr : ident
            | end
            | number
            | string
            | colon
            | NEG
            | matrix
            | cellarray
            | expr2
            | expr1
            | lambda_expr
    """
    p[0] = p[1]


def p_expr_2(p):
    """expr : expr PLUSPLUS
            | expr MINUSMINUS
    """
    startl, endl = p.linespan(2)
    startc, endc = p.lexspan(2)
    di2 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_UnaryExpression(di2, p[1], p[2], "post")


def p_expr1(p):
    """expr1 : MINUS expr %prec UMINUS
             | PLUS expr %prec UMINUS
             | NEG expr
             | HANDLE ident
             | PLUSPLUS ident
             | MINUSMINUS ident
    """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_UnaryExpression(di1, p[2], p[1], "pre")


def p_expr2(p):
    """expr2 : expr AND expr
             | expr ANDAND expr
             | expr BACKSLASH expr
             | expr COLON expr
             | expr DIV expr
             | expr DOT expr
             | expr DOTDIV expr
             | expr DOTDIVEQ expr
             | expr DOTEXP expr
             | expr DOTMUL expr
             | expr DOTMULEQ expr
             | expr EQEQ expr
             | expr POW expr
             | expr EXP expr
             | expr EXPEQ expr
             | expr GE expr
             | expr GT expr
             | expr LE expr
             | expr LT expr
             | expr MINUS expr
             | expr MUL expr
             | expr NE expr
             | expr OR expr
             | expr OROR expr
             | expr PLUS expr
             | expr EQ expr
             | expr MULEQ expr
             | expr DIVEQ expr
             | expr MINUSEQ expr
             | expr PLUSEQ expr
             | expr OREQ expr
             | expr ANDEQ expr
    """
    startl, endl = p.linespan(2)
    startc, endc = p.lexspan(2)
    di2 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    if p[2] == "=":
        p[0] = AST_Assign(di2, p[1], p[3], p[2])
    elif p[2] == ":":
        p[0] = AST_RangeExpression(di2, p[1], p[3])
    else:
        p[0] = AST_BinExpression(di2, p[1], p[3], p[2])


def p_expr_colon(p):
    """ colon : COLON """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_RangeExpression(di1, None, None)


def p_expr_end(p):
    """ end : END_EXPR """
    raise NotImplementedError("end expression needs to be implemented")


def p_expr_ident(p):
    """ ident : IDENT """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_Ident(di1, p[1])


def p_ident_init_opt(p):
    """
    ident_init_opt : NEG
                   | ident
                   | ident EQ expr
    """
    if len(p) == 1:
        raise NotImplementedError("default args need to be implemented")
    if len(p) == 2:
        p[0] = p[1]
    else:
        raise NotImplementedError("default args need to be implemented")


def p_expr_list(p):
    """
    expr_list : exprs
              | exprs COMMA
    """
    p[0] = p[1]


def p_expr_number(p):
    """ number : NUMBER """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_Constant(di1, p[1])


def p_expr_stmt(p):
    """
    expr_stmt : expr_list SEMI
    """
    p[0] = p[1]


def p_expr_string(p):
    """ string : STRING """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_Constant(di1, p[1])


def p_exprs(p):
    """
    exprs : expr
          | exprs COMMA expr
    """
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 4:
        p[0] = p[1]
        p[0].append(p[3])


def p_field_expr(p):
    """
    expr : expr FIELD
    """
    raise NotImplementedError("field expressions needs to be implemented")


def p_foo_stmt(p):
    """ foo_stmt : expr OROR expr SEMI """
    raise NotImplementedError("foo_stmt needs to be implemented")


def p_for_stmt(p):
    """
    for_stmt : FOR ident  EQ expr SEMI stmt_list END_STMT
             | FOR LPAREN ident EQ expr RPAREN SEMI stmt_list END_STMT
             | FOR matrix EQ expr SEMI stmt_list END_STMT
    """
    di = None
    if len(p) == 8:
        p[0] = AST_ForLoop(di, p[2], p[4], AST_Statements(di, p[6]))
    else:
        p[0] = AST_ForLoop(di, p[3], p[5], AST_Statements(di, p[8]))


def p_func_stmt(p):
    """func_stmt : FUNCTION ident lambda_args SEMI
                 | FUNCTION ret EQ ident lambda_args SEMI
    """
    di = None
    if len(p) == 5:
        p[0] = AST_Function(di, p[2], args=p[3], retvals=[])
    else:
        p[0] = AST_Function(di, p[4], args=p[5], retvals=p[2])


def p_funcall_expr(p):
    """expr : expr LPAREN expr_list RPAREN
            | expr LPAREN RPAREN
    """
    startl, endl = p.linespan(1)
    startc, endc = p.lexspan(1)
    di1 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    if len(p) == 4:
        p[0] = AST_FunCall(di1, p[1], [])
    else:
        p[0] = AST_FunCall(di1, p[1], p[3])


def p_global_list(p):
    """global_list : ident
                   | global_list ident
    """
    raise NotImplementedError("globals need to be implemented")


def p_global_stmt(p):
    """
    global_stmt : GLOBAL global_list SEMI
                | GLOBAL ident EQ expr SEMI
    """
    raise NotImplementedError("globals need to be implemented")


def p_if_stmt(p):
    """
    if_stmt : IF expr sep stmt_list_opt elseif_stmt END_STMT
            | IF LPAREN expr RPAREN stmt_list_opt elseif_stmt END_STMT
    """
    raise NotImplementedError("If/else needs to be implemented")


def p_lambda_args(p):
    """lambda_args : LPAREN RPAREN
                   | LPAREN arg_list RPAREN
    """
    if len(p) == 3:
        p[0] = []
    else:
        p[0] = p[2]


def p_lambda_expr(p):
    """lambda_expr : HANDLE lambda_args expr
    """
    raise NotImplementedError("lambda needs to be implemented")


def p_matrix(p):
    """matrix : LBRACKET RBRACKET
              | LBRACKET matrix_row RBRACKET
              | LBRACKET matrix_row SEMI RBRACKET
    """
    startl, endl = p.linespan(0)
    startc, endc = p.lexspan(0)
    di0 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    if len(p) == 3:
        p[0] = AST_Matrix(di0, [])
    else:
        p[0] = AST_Matrix(di0, p[2])


def p_matrix_2(p):
    """matrix : LBRACKET expr_list RBRACKET
              | LBRACKET expr_list SEMI RBRACKET
    """
    startl, endl = p.linespan(0)
    startc, endc = p.lexspan(0)
    di0 = dace.dtypes.DebugInfo(startl, startc, endl, endc)
    startl, endl = p.linespan(2)
    startc, endc = p.lexspan(2)
    di2 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_Matrix(di0, [AST_Matrix_Row(di2, p[2])])


def p_null_stmt(p):
    """
    null_stmt : SEMI
              | COMMA
    """
    di = None
    p[0] = AST_NullStmt(di)


def p_parens_expr(p):
    """
    expr :  LPAREN expr RPAREN
    """
    p[0] = p[2]


def p_persistent_stmt(p):
    """
    persistent_stmt :  PERSISTENT global_list SEMI
                    |  PERSISTENT ident EQ expr SEMI
    """
    raise NotImplementedError("persistent needs to be implemented")


def p_ret(p):
    """
    ret : ident
        | LBRACKET RBRACKET
        | LBRACKET expr_list RBRACKET
    """
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = []
    else:
        p[0] = p[2]


def p_return_stmt(p):
    """ return_stmt : RETURN SEMI """
    raise NotImplementedError("return needs to be implemented")


def p_semi_opt(p):
    """
    semi_opt :
             | semi_opt SEMI
             | semi_opt COMMA
    """
    p[0] = AST_NullStmt(None)


def p_separator(p):
    """
    sep : COMMA
        | SEMI
    """
    p[0] = p[1]


def p_stmt(p):
    """
    stmt : continue_stmt
         | comment_stmt
         | func_stmt
         | break_stmt
         | expr_stmt
         | global_stmt
         | persistent_stmt
         | error_stmt
         | command
         | for_stmt
         | if_stmt
         | null_stmt
         | return_stmt
         | switch_stmt
         | try_catch
         | while_stmt
         | foo_stmt
         | unwind
    """
    # END_STMT is intentionally left out
    p[0] = copy.deepcopy(p[1])


def p_stmt_list(p):
    """
    stmt_list : stmt
              | stmt_list stmt
    """
    if len(p) == 2:
        if p[1] is None:
            p[0] = []
        if isinstance(p[1], list):
            p[0] = copy.deepcopy(p[1])
    elif len(p) == 3:
        p[0] = copy.deepcopy(p[1])
        if p[2] is not None:
            if isinstance(p[2], list):
                p[0] = p[0] + p[2]
            else:
                p[0].append(p[2])
    else:
        assert 0


def p_stmt_list_opt(p):
    """
    stmt_list_opt :
                  | stmt_list
    """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = p[1]


def p_switch_stmt(p):
    """
    switch_stmt : SWITCH expr semi_opt case_list END_STMT
    """
    raise NotImplementedError("switch needs to be implemented")


def p_transpose_expr(p):
    # p[2] contains the exact combination of plain and conjugate
    # transpose operators, such as "'.''.''''".
    """ expr : expr TRANSPOSE """
    startl, endl = p.linespan(2)
    startc, endc = p.lexspan(2)
    di2 = dace.dtypes.DebugInfo(startl, startc, endl, endc)

    p[0] = AST_Transpose(di2, p[1], p[2])


def p_try_catch(p):
    """
    try_catch : TRY stmt_list CATCH stmt_list END_STMT
    """
    raise NotImplementedError("try/catch needs to be implemented")


def p_unwind(p):
    """
    unwind : UNWIND_PROTECT stmt_list UNWIND_PROTECT_CLEANUP stmt_list END_UNWIND_PROTECT
    """
    raise NotImplementedError("unwind needs to be implemented")


def p_while_stmt(p):
    """
    while_stmt : WHILE expr SEMI stmt_list END_STMT
    """
    raise NotImplementedError("while needs to be implemented")


def p_error(p):
    raise ValueError("Unexpected EOF")


parser = yacc.yacc(start="top")


def parse(buf, debug=False):
    new_lexer = lexer.new()
    p = parser.parse(buf, tracking=1, debug=debug, lexer=new_lexer)
    return p


if __name__ == "__main__":
    buf = open(sys.argv[1]).read()
    p = parse(buf, debug=False)
