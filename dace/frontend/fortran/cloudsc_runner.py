# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
from dace import SDFG, nodes, dtypes
from dace.frontend.fortran import fortran_parser
import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
from fparser.two.symbol_table import SymbolTable
import os



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--testname', type=str, nargs='?', default='cloudscexp2')
    argparser.add_argument('-f', '--folder', type=str, nargs='?', default=None)
    argparser.add_argument('-x', '--extension', type=str, nargs='?', default='f90')
    args = vars(argparser.parse_args())
    testname = args['testname']
    folder = args['folder'] or os.getcwd()
    extension = args['extension']

    parser = ParserFactory().create(std="f2008")
    reader = FortranFileReader(os.path.realpath(os.path.join(folder, f'{testname}.{extension}')))
    ast = parser(reader)
    tables = SymbolTable
    own_ast = ast_components.InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)
    functions_and_subroutines_builder = ast_transforms.FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.nodes
    program = ast_transforms.functionStatementEliminator(program)
    program = ast_transforms.CallToArray(functions_and_subroutines_builder.nodes).visit(program)
    program = ast_transforms.CallExtractor().visit(program)
    program = ast_transforms.SignToIf().visit(program)
    program = ast_transforms.ArrayToLoop().visit(program)
    program = ast_transforms.SumToLoop().visit(program)
    program = ast_transforms.ForDeclarer().visit(program)
    program = ast_transforms.IndexExtractor().visit(program)
    ast2sdfg = fortran_parser.AST_translator(own_ast, __file__)
    sdfg = SDFG("top_level")
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if 'CLOUDSCOUTER' in node.sdfg.name:
                sdfg = node.sdfg
                break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    sdfg.validate()
    sdfg.save(os.path.join(folder, f'{testname}_initial.sdfg'))
    from dace.transformation.passes import ScalarToSymbolPromotion
    for sd in sdfg.all_sdfgs_recursive():
        promoted = ScalarToSymbolPromotion().apply_pass(sd, {})
        print(f"Promoted the following scalars: {promoted}")
    sdfg.save(os.path.join(folder, f'{testname}_promoted.sdfg'))
    from dace.sdfg import utils
    utils.normalize_offsets(sdfg)
    sdfg.save(os.path.join(folder, f'{testname}_normalized.sdfg'))
    sdfg.simplify(verbose=True)
    sdfg.save(os.path.join(folder, f'{testname}_simplified.sdfg'))
    from dace.transformation.auto import auto_optimize as aopt
    aopt.auto_optimize(sdfg, dtypes.DeviceType.Generic)
    sdfg.save(os.path.join(folder, f'{testname}_optimized.sdfg'))
    sdfg.compile()