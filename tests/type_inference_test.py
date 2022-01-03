# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
import numpy as np
import sympy as sp
from dace.config import Config
from dace.codegen.tools import type_inference
from dace import dtypes
import ast


class TestTypeInference(unittest.TestCase):
    def testSimpleAssignment(self):
        # simple assignment tests

        #bool
        code_str = "value=True"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(bool))

        # int
        code_str = "value = 1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(int))

        # float
        code_str = "value = 1.1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(float))

        # string: should return a char*
        code_str = "value = 'hello'"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.pointer(dtypes.int8))

        # assignment with previous symbols
        prev_symbols = {"char_num": dtypes.typeclass(np.int8)}
        code_str = "value =  char_num"
        inf_symbols = type_inference.infer_types(code_str, prev_symbols)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.int8))

        # aug assignment
        code_str = "value += 1.1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(float))

        # annotated assignments
        code_str = "value : int  = 1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(int))

        code_str = "value : dace.int32  = 1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.int32)

        code_str = "value : numpy.float64  = 1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.float64)

        code_str = "value : str"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.pointer(dtypes.int8))

        # type conversion
        # in this case conversion is stricter (int-> int32)
        code_str = "value = int(1.1)"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.int))

        code_str = "value = int32(1.1)"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.int32)

        code_str = "value = dace.float64(1.1)"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.float64)

        code_str = "value = float(1)"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.float))

    def testInferExpr(self):

        code_str = "5 + 3.5"
        inf_type = type_inference.infer_expr_type(code_str)
        self.assertEqual(inf_type, dtypes.typeclass(float))

        prev_symbols = {"n": dtypes.typeclass(int)}
        code_str = "5 + n"
        inf_type = type_inference.infer_expr_type(code_str, prev_symbols)
        self.assertEqual(inf_type, dtypes.typeclass(int))

        #invalid code
        code_str = "a = 5 + 3.5"
        self.assertRaises(TypeError, lambda: type_inference.infer_expr_type(code_str))

    def testExpressionAssignment(self):

        code_str = "res = 5 + 3.1"
        symbols = type_inference.infer_types(code_str)
        self.assertEqual(symbols["res"], dtypes.typeclass(float))

        code_str = "res = 5 + 1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(int))

        # use already defined symbol
        code_str = "res2 = 1 + res"
        symbols = type_inference.infer_types(code_str, symbols)
        self.assertEqual(symbols["res2"], dtypes.typeclass(float))

        code_str = "res3 = 1 + int(res*res2)"
        symbols = type_inference.infer_types(code_str, symbols)
        self.assertEqual(symbols["res3"], dtypes.typeclass(int))

    def testArrayAccess(self):
        code_str = "tmp = array[i]"
        symbols = type_inference.infer_types(code_str, {"array": dtypes.typeclass(float)})
        self.assertEqual(symbols["tmp"], dtypes.typeclass(float))

    def testAssignmentIf(self):
        code_str = "res = 5 if x > 10 else 3.1"
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(float))

    def testIf(self):
        code_str = """if cond1:
    a = 1*2
elif cond2:
    b = 1.2*3.4
else:
    c = True"""
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["a"], dtypes.typeclass(int))
        self.assertEqual(inf_symbols["b"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["c"], dtypes.typeclass(bool))

    def testFunction(self):
        code_str = """def f():
    x = 2
    y = 7.5
        """
        inf_symbols = type_inference.infer_types(code_str)
        self.assertEqual(inf_symbols["x"], dtypes.typeclass(int))
        self.assertEqual(inf_symbols["y"], dtypes.typeclass(float))

    def testInputAST(self):
        # infer input parameter is an AST

        code_str = """var1 = int(in_x)
var2: int = in_y
var3 = 2.1 if (i>1 and i<10) else 2.1 # A comment
res = var1 + var3 * var2
        """

        #create AST
        tree = ast.parse(code_str)
        defined_symbols = {"in_x": dtypes.typeclass(np.float32), "in_y": dtypes.typeclass(np.float32)}
        inf_symbols = type_inference.infer_types(code_str, defined_symbols)
        self.assertEqual(inf_symbols["var1"], dtypes.typeclass(int))
        self.assertEqual(inf_symbols["var2"], dtypes.typeclass(int))
        self.assertEqual(inf_symbols["var3"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(float))

    def testInputList(self):
        # infer input parameter is a list of code_string
        code1 = "var1 = x + 1.1"
        code2 = "var2 = var1 + 2"
        defined_symbols = {"x": dtypes.typeclass(int)}
        inf_symbols = type_inference.infer_types([code1, code2], defined_symbols)
        self.assertEqual(inf_symbols["var1"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["var2"], dtypes.typeclass(float))

    def testSymbolic(self):
        # Define some sympy symbols to work with
        n = sp.Symbol('n')
        m = sp.Symbol('m')

        defined_symbols = {'n': dtypes.typeclass(np.float64)}
        inf_symbol = type_inference.infer_expr_type(n + 5, defined_symbols)
        self.assertEqual(inf_symbol, dtypes.typeclass(np.float64))

        defined_symbols = {'n': dtypes.typeclass(np.int8)}
        inf_symbol = type_inference.infer_expr_type(n * 5, defined_symbols)
        self.assertEqual(inf_symbol, dtypes.typeclass(int))

        defined_symbols = {'n': dtypes.typeclass(np.int8)}
        inf_symbol = type_inference.infer_expr_type(n * 5.0, defined_symbols)
        self.assertEqual(inf_symbol, dtypes.typeclass(float))

        defined_symbols = {'n': dtypes.typeclass(np.int8)}
        inf_symbol = type_inference.infer_expr_type(n * 5.01, defined_symbols)
        self.assertEqual(inf_symbol, dtypes.typeclass(float))

        defined_symbols = {'n': dtypes.typeclass(np.int8), 'm': dtypes.typeclass(np.float32)}
        inf_symbol = type_inference.infer_expr_type(n * m + n, defined_symbols)
        self.assertEqual(inf_symbol, dtypes.typeclass(np.float32))

    def testForLoop(self):
        # for loop
        for_loop_code = """for x in range(6):
            res += 1"""
        inf_symbols = type_inference.infer_types(for_loop_code)
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(int))

        #It is not possible to annotate the type of the variable in the loop guard
        # But it is ok to do so outside of the loop
        # https://stackoverflow.com/questions/41641449/how-do-i-annotate-types-in-a-for-loop/41641489#41641489
        for_loop_code = """i: int
for i in range(5):
    x += i"""
        inf_symbols = type_inference.infer_types(for_loop_code)
        self.assertEqual(inf_symbols["x"], dtypes.typeclass(int))
        self.assertEqual(inf_symbols["i"], dtypes.typeclass(int))

    def testVarious(self):
        # code snippets that contains constructs not directly involved in type inference
        # (borrowed by astunparse tests)

        while_code = """def g():
    while True:
        break
    z = 3
"""
        inf_symbols = type_inference.infer_types(while_code)
        self.assertEqual(inf_symbols["z"], dtypes.typeclass(int))

        raise_from_code = """try:
    1 / 0
except ZeroDivisionError as e:
    raise ArithmeticError from e
"""
        inf_symbols = type_inference.infer_types(raise_from_code)

        try_except_finally_code = """try:
    suite1
except ex1:
    suite2
except ex2:
    suite3
else:
    suite4
finally:
    suite5
"""
        inf_symbols = type_inference.infer_types(try_except_finally_code)

        #function def with arguments
        function_def_return_code = """def f(arg : float):
    res = 5 + arg
    return res
        """
        inf_symbols = type_inference.infer_types(function_def_return_code)
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["arg"], dtypes.typeclass(float))

    def testDefaultDataTypes(self):
        # check that configuration about defult data types is enforced
        config_data_types = Config.get('compiler', 'default_data_types')

        code_str = """value1 = 10
value2=3.14
value3=5000000000"""
        inf_symbols = type_inference.infer_types(code_str)
        if config_data_types.lower() == "python":
            self.assertEqual(inf_symbols["value1"], dtypes.typeclass(np.int64))
            self.assertEqual(inf_symbols["value2"], dtypes.typeclass(np.float64))
        elif config_data_types.lower() == "c":
            self.assertEqual(inf_symbols["value1"], dtypes.typeclass(np.int32))
            self.assertEqual(inf_symbols["value2"], dtypes.typeclass(np.float32))

        # in any case, value3 needs uint64
        self.assertEqual(inf_symbols["value3"], dtypes.typeclass(np.uint64))

    def testCCode(self):
        # tests for situations that could arise from C/C++ tasklet codes

        ###############################################################################################
        # Pointer: this is a situation that could happen in FPGA backend due to OpenCL Keyword Remover:
        # if in a tasklet, there is an assignment in which the right hand side is a connector and the
        # corresponding memlet is dynamic, the OpenCL Keyword Remover, will update the code to be "target = *rhs"
        # In a similar situation the type inference should not consider the leading *
        stmt = ast.parse("value = float_var")
        stmt.body[0].value.id = "*float_var"  # effect of OpenCL Keyword Remover
        prev_symbols = {"float_var": dtypes.typeclass(float)}
        inf_symbols = type_inference.infer_types(stmt, prev_symbols)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(float))


if __name__ == "__main__":
    unittest.main()
