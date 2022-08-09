
from .coverage import get_builtins_definitons, get_builtins_types

from .ssa_helpers import print_exp, print_stmt
from .ssa_preprocess import SSA_Preprocessor
from .ssa_transpiler import SSA_Transpiler
from .ssa_postprocess import SSA_Postprocessor, SSA_Finisher

from .type_inferrer_stmt import TypeInferrer

from ast import parse, unparse, fix_missing_locations


def ssa_prepare(code):

    tree = parse(code)

    SSA_Preprocessor().visit(tree)
    SSA_Postprocessor().visit(tree)
    SSA_Finisher().visit(tree)
    fix_missing_locations(tree)
    
    return unparse(tree)


def ssa_convert(code):

    tree = parse(code)

    SSA_Preprocessor().visit(tree)
    SSA_Transpiler().visit(tree, get_builtins_definitons())
    SSA_Postprocessor().visit(tree)
    SSA_Finisher().visit(tree)
    fix_missing_locations(tree)

    return unparse(tree)


def typed_ssa_convert(code):

    tree = parse(code)

    # SSA
    SSA_Preprocessor().visit(tree)
    SSA_Transpiler().visit(tree, get_builtins_definitons())
    SSA_Postprocessor().visit(tree)

    # Type Inference
    TypeInferrer().visit(tree, get_builtins_types())

    # Cleanup
    SSA_Finisher().visit(tree)
    fix_missing_locations(tree)

    return unparse(tree)
