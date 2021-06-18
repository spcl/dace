# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import unittest

class SymbolTest(unittest.TestCase):
    def test_symbol_eqivalence(self):
        n1 = dace.symbol('n', positive=True)
        n2 = dace.symbol('n')
        self.assertTrue(n1 == n2)

    def test_symbol_sub(self):
        n1 = dace.symbol('n', positive=True)
        n2 = dace.symbol('n')
        self.assertTrue(n1.subs(n2, 0) == 0)
        self.assertTrue((n1 + 1).subs(n2, 0) == 1)

if __name__ == '__main__':
    unittest.main()
