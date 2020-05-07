import unittest
import numpy as np
from dace.codegen.tools import type_inference
from dace import dtypes
import ast

class TestTypeInference(unittest.TestCase):

    def testSimpleAssignement(self):
        # simple assignment tests

        #bool
        code_str = "value=True"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(bool))

        # int
        code_str = "value = 1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(int))

        # float
        code_str = "value = 1.1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(float))

        # assignment with previous symbols
        prev_symbols = {"char_num": dtypes.typeclass(np.int8)}
        code_str = "value =  char_num"
        inf_symbols = type_inference.infer(code_str, prev_symbols)
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.int8))

        # aug assignment
        code_str = "value += 1.1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(float))

        # annotated assignments
        # in this case conversion is stricter (int-> int32)
        code_str = "value : int  = 1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.int32))

        # type conversion
        # in this case conversion is stricter (int-> int32)
        code_str = "value = int(1.1)"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.int32))

        code_str = "value = float(1)"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(np.float32))

    def testExpressionAssignment(self):

        code_str = "res = 5 + 3.1"
        symbols = type_inference.infer(code_str, {})
        self.assertEqual(symbols["res"], dtypes.typeclass(float))

        code_str = "res = 5 + 1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(int))

        # use already defined symbol
        code_str = "res2 = 1 + res"
        symbols = type_inference.infer(code_str, symbols)
        self.assertEqual(symbols["res2"], dtypes.typeclass(float))

        code_str = "res3 = 1 + int(res*res2)"
        symbols = type_inference.infer(code_str, symbols)
        self.assertEqual(symbols["res3"], dtypes.typeclass(int))



    def testAssignmentIf(self):
        code_str = "res = 5 if(x>10) else 3.1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(float))

    def testIf(self):
        code_str = """if cond1:
    a = 1*2
elif cond2:
    b = 1.2*3.4
else:
    c = True"""
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["a"], dtypes.typeclass(int))
        self.assertEqual(inf_symbols["b"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["c"], dtypes.typeclass(bool))


    def testFunction(self):
        code_str = """def f():
    x = 2
    y = 7.5
        """
        inf_symbols = type_inference.infer(code_str, {})
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
        inf_symbols = type_inference.infer(code_str, defined_symbols)
        self.assertEqual(inf_symbols["var1"], dtypes.typeclass(np.int32))
        self.assertEqual(inf_symbols["var2"], dtypes.typeclass(np.int32))
        self.assertEqual(inf_symbols["var3"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(float))


    def testInputList(self):
        # infer input parameter is a list of code_string
        code1 = "var1 = x + 1.1"
        code2 = "var2 = var1 + 2"
        defined_symbols = {"x": dtypes.typeclass(int)}
        inf_symbols = type_inference.infer([code1, code2], defined_symbols)
        self.assertEqual(inf_symbols["var1"], dtypes.typeclass(float))
        self.assertEqual(inf_symbols["var2"], dtypes.typeclass(float))





if __name__ == "__main__":
    unittest.main()

