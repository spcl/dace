# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests constant folding with globals. """
import dace
import numpy as np

from dace.frontend.python import astutils
from dace.frontend.python.preprocessing import (GlobalResolver,
                                                ConditionalCodeResolver,
                                                DeadCodeEliminator)
from dace.frontend.python.parser import DaceProgram


class MyConfiguration:
    def __init__(self, parameter):
        self.p = parameter * 2

    @property
    def q(self):
        return self.p * 2

    def get_parameter(self):
        return self.p // 2

    @staticmethod
    def get_random_number():
        return 4

    @property
    def cloned(self):
        return MyConfiguration(self.get_parameter())


N = 2
cfg = MyConfiguration(N)
val = 5

# Confuse AST parser with global of the same name as array
A = 5


@dace.program
def instantiated_global(A):
    A[cfg.q] = (A[cfg.p // 2] * 4 + cfg.p) + val


def test_instantiated_global():
    """
    Tests constant/symbolic values with predetermined global values.
    """
    A = np.random.rand(10)
    reg_A = np.copy(A)
    reg_A[cfg.q] = (reg_A[cfg.p // 2] * 4 + cfg.p) + val

    instantiated_global(A)

    assert np.allclose(A, reg_A)


@dace.program(constant_functions=True)
def instantiated_global_with_funcs(A):
    A[cfg.q] = (A[cfg.get_parameter()] * MyConfiguration.get_random_number() +
                cfg.p) + val


def test_instantiated_global_resolve_functions():
    """
    Tests constant/symbolic values with predetermined global values.
    """
    A = np.random.rand(10)
    reg_A = np.copy(A)
    reg_A[cfg.q] = (reg_A[cfg.p // 2] * 4 + cfg.p) + val

    instantiated_global_with_funcs(A)

    assert np.allclose(A, reg_A)


def test_nested_globals():
    """
    Tests constant/symbolic values with multiple nesting levels.
    """
    @dace.program
    def instantiated_global2(A):
        A[cfg.q] = cfg.cloned.p

    A = np.random.rand(10)
    reg_A = np.copy(A)
    reg_A[cfg.q] = cfg.cloned.p

    instantiated_global2(A)

    assert np.allclose(A, reg_A)


def _analyze_and_unparse_code(func: DaceProgram) -> str:
    src_ast, _, _, _ = astutils.function_to_ast(func.f)
    resolved = {
        k: v
        for k, v in func.global_vars.items() if k not in func.argnames
    }
    src_ast = GlobalResolver(resolved).visit(src_ast)
    src_ast = ConditionalCodeResolver(resolved).visit(src_ast)
    src_ast = DeadCodeEliminator().visit(src_ast)

    return astutils.unparse(src_ast)


def test_dead_code_elimination_if():
    """
    Tests dead code elimination with compile-time if conditions.
    """
    sym = dace.symbol('sym', positive=True)
    cfg_symbolic = MyConfiguration(sym)

    @dace.program
    def test(A):
        if cfg_symbolic.q > sym:
            return 2 * A
        else:
            return 4 * A

    parsed_code = _analyze_and_unparse_code(test)
    assert '4' not in parsed_code
    assert '2' in parsed_code


def test_dead_code_elimination_ifexp():
    """
    Tests dead code elimination with compile-time ternary expressions.
    """
    sym = dace.symbol('sym', positive=True)
    cfg_symbolic = MyConfiguration(sym)

    @dace.program
    def test(A):
        return 2 * A if cfg_symbolic.q > sym else 4 * A

    parsed_code = _analyze_and_unparse_code(test)
    assert '4' not in parsed_code
    assert '2' in parsed_code


def test_dead_code_elimination_noelse():
    """
    Tests dead code elimination with compile-time if conditions (without else).
    """
    scale = None

    @dace.program
    def test(A):
        if scale is None:
            return 2 * A
        return scale * A

    parsed_code = _analyze_and_unparse_code(test)
    assert 'scale' not in parsed_code
    assert '2' in parsed_code


def test_dead_code_elimination_unreachable():
    """
    Tests dead code elimination with unreachable code.
    """
    @dace.program
    def test(A):
        if A[5] > 1:
            return 3 * A
            return 6 * A
        return 2 * A
        return 4 * A

    parsed_code = _analyze_and_unparse_code(test)
    assert '6' not in parsed_code and '4' not in parsed_code  # Dead code
    assert '5' in parsed_code and '1' in parsed_code  # Condition
    assert '3' in parsed_code and '2' in parsed_code  # Reachable code


if __name__ == '__main__':
    test_instantiated_global()
    test_instantiated_global_resolve_functions()
    test_nested_globals()
    test_dead_code_elimination_if()
    test_dead_code_elimination_ifexp()
    test_dead_code_elimination_noelse()
    test_dead_code_elimination_unreachable()
