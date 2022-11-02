from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import *
import sys, os
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from fcdc import *


def test_fortran_frontend(test_string: str,
                          test_name: str,
                          simplify: bool = True):
    parser = ParserFactory().create(std="f2008")
    reader = FortranStringReader(test_string)
    ast = parser(reader)
    tables = SYMBOL_TABLES
    own_ast = InternalFortranAst(ast, tables)
    program = own_ast.create_ast(ast)
    functions_and_subroutines_builder = FindFunctionAndSubroutines()
    functions_and_subroutines_builder.visit(program)
    own_ast.functions_and_subroutines = functions_and_subroutines_builder.nodes
    program = functionStatementEliminator(program)
    program = CallToArray(
        functions_and_subroutines_builder.nodes).visit(program)
    program = CallExtractor().visit(program)
    program = SignToIf().visit(program)
    program = ArrayToLoop().visit(program)
    program = SumToLoop().visit(program)
    program = ForDeclarer().visit(program)
    program = IndexExtractor().visit(program)
    ast2sdfg = AST_translator(own_ast, __file__)
    sdfg = SDFG(test_name)
    ast2sdfg.top_level = program
    ast2sdfg.globalsdfg = sdfg
    ast2sdfg.translate(program, sdfg)

    sdfg.validate()
    if simplify:
        sdfg.simplify(verbose=True)
    sdfg.compile()


if __name__ == "__main__":
    test_string = """
                    PROGRAM simplest_test
                    implicit none
                    integer a
                    real d(2,3),h(10)

                    a=10
                    d(:,:)=0.0
                    d(1,2)=5
                    d(1,1)=1.0
                    d(2,1)=2.0
                    d(2,3)=3.0
                    h(3)=4
                    write (*,*) d(1,2)
                    write (*,*) d(1,1)
                    write (*,*) d(2,1)
                    end"""
    test_string2 = """
                    PROGRAM symbol_test
                    implicit none
                    integer a,b
                    real d(2,3)

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=5
                    write (*,*) d(1,2)
                    write (*,*) d(1,1)
                    write (*,*) d(2,1)
                    end"""
    #test_fortran_frontend(test_string, "simplest_test", simplify=False)
    test_fortran_frontend(test_string2, "symbol_test", simplify=True)
