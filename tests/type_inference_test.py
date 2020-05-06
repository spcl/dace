import unittest
from dace.codegen.tools import type_inference
from dace import dtypes



class TestAssignment(unittest.TestCase):

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
        # int
        code_str = "value = 1.1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["value"], dtypes.typeclass(float))

    def testExpressionAssignment(self):

        code_str = "res = 5 + 3.1"
        inf_symbols = type_inference.infer(code_str, {})
        self.assertEqual(inf_symbols["res"], dtypes.typeclass(float))


    def testExpressionWithVariables(self):
        pass




if __name__ == "__main__":
    unittest.main()

