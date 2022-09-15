#fparser imports
from dataclasses import astuple
from venv import create
from fparser.api import parse
from fparser.two.Fortran2003 import *
from fparser.two.Fortran2008 import *
from fparser.two.parser import *
from fparser.two.utils import *
from fparser.two.symbol_table import *
import os
from fparser.common.readfortran import FortranStringReader, FortranFileReader

#dace imports
import dace
from dace.sdfg import *
from dace.data import Scalar
from dace.properties import CodeBlock

import ast_components
from ast_components import *

if __name__ == "__main__":
    parser = ParserFactory().create(std="f2008")
    testname = "loop3"
    reader = FortranFileReader(
        os.path.realpath("/mnt/c/Users/Alexwork/Desktop/Git/f2dace/tests/" +
                         testname + ".f90"))
    ast = parser(reader)
    tables = SYMBOL_TABLES
    table = tables.lookup("CLOUDPROGRAM")
    own_ast = InternalFortranAst(ast, tables)
    #own_ast.list_tables()
    own_ast.create_ast(ast)
    # node_list = walk(ast)
    # node_types = []
    # for i in node_list:
    #     if type(i).__name__ not in node_types and i is not None and type(
    #             i) != type("string"):
    #         if type(i) == tuple:
    #             print(i)
    #         node_types.append(type(i).__name__)

    #print(ast)
