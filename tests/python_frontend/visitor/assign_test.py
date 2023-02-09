# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import data
from dace.frontend.python.newast import ProgramVisitor
from typing import Any


def _test_program_variable(visitor: ProgramVisitor, program_name: str, expected_type: str, expected_value: Any) -> bool:
    if not program_name in visitor.variable_p2i_names:
        return False
    internal_name = visitor.variable_p2i_names[program_name]
    variable = visitor.variables[internal_name]
    if variable.type != expected_type:
        return False
    if expected_value and variable.value != expected_value:
            return False
    return True


def test_assign_constant_bool():

    @dace.program
    def func():
        a = True
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'BoolConstant', True)


def test_assign_constant_num():

    @dace.program
    def func():
        a = 5
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'NumConstant', 5)


def test_assign_constant_str():

    @dace.program
    def func():
        a = "Hello World!"
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'StrConstant', 'Hello World!')


def test_assign_list():

    @dace.program
    def func():
        a = [1, 3.14, 'Hello World!']
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'List', None)


def test_assign_tuple():

    @dace.program
    def func():
        a = (1, 3.14, 'Hello World!')
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'Tuple', None)


def test_assign_set():

    @dace.program
    def func():
        a = {1, 3.14, 'Hello World!'}
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'Set', None)


def test_assign_dict():

    @dace.program
    def func():
        a = {0:1, 'a':3.14, 1.23:'Hello World!'}
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'Dict', None)


def test_assign_data_scalar():

    @dace.program
    def func(a: dace.int32):
        b = a
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'b', 'Data', data.Scalar(dace.int32))


def test_assign_data_array():

    @dace.program
    def func(a: dace.int32[10]):
        b = a
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'b', 'Data', data.Array(dace.int32, (10,)))


def test_assign_sliced_data_array():

    @dace.program
    def func(a: dace.int32[10]):
        b = a[1:-1]
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'b', 'SlicedData', None)


def test_assign_named_expr_constant_num():

    @dace.program
    def func():
        a = (b := 3)
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'b', 'NumConstant', 3)
    assert _test_program_variable(visitor, 'a', 'NumConstant', 3)


def test_assign_lambda():

    @dace.program
    def func():
        a = lambda x, y: x + y
    
    visitor = func.to_visitor(simplify=False)
    assert _test_program_variable(visitor, 'a', 'Lambda', None)
    

# def test_assign_if_exp():

#     @dace.program
#     def func(a: dace.int32):
#         b = 3 if a < 0 else 5
    
#     visitor = func.to_visitor(simplify=False)
#     assert _test_program_variable(visitor, 'b', 'IfExp', None)
  

if __name__ == "__main__":
    test_assign_constant_bool()
    test_assign_constant_num()
    test_assign_constant_str()
    test_assign_list()
    test_assign_tuple()
    test_assign_set()
    test_assign_dict()
    test_assign_data_scalar()
    test_assign_data_array()
    test_assign_sliced_data_array()
    test_assign_named_expr_constant_num()
    test_assign_lambda()
    # test_assign_if_exp()

