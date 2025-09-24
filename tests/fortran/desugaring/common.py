# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, Optional, Iterable

from fparser.two.Fortran2003 import Program
from fparser.two.parser import ParserFactory

from dace.frontend.fortran.ast_desugaring import cleanup, types
from dace.frontend.fortran.fortran_parser import construct_full_ast


def parse_and_improve(sources: Dict[str, str], entry_points: Optional[Iterable[types.SPEC]] = None):
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(sources, parser, entry_points=entry_points)
    ast = cleanup.correct_for_function_calls(ast)
    assert isinstance(ast, Program)
    return ast
