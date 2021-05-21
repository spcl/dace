# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests constant folding with globals. """
import dace
import numpy as np


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
    A[cfg.q] = (A[cfg.get_parameter()] * MyConfiguration.get_random_number() +
                cfg.p) + val


def test_instantiated_global():
    """
    Tests constant/symbolic values with predetermined global values.
    """
    A = np.random.rand(10)
    reg_A = np.copy(A)
    reg_A[cfg.q] = (reg_A[cfg.get_parameter()] *
                    MyConfiguration.get_random_number() + cfg.p) + val

    instantiated_global(A)

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


# TODO: dace.constant should signal that argument evaluation is deferred to
#       (nested) call time
# dace.constant = lambda x: None
# def test_constant_parameter():
#     """
#     Tests nested functions with constant parameters passed in as arguments.
#     """
#     @dace.program
#     def nested_func(cfg: dace.constant(MyConfiguration), A: dace.float64[20]):
#         return A[cfg.p]

#     @dace.program
#     def constant_parameter(
#             cfg: dace.constant(MyConfiguration),
#             cfg2: dace.constant(MyConfiguration), A: dace.float64[20]):
#         A[cfg.q] = nested_func(cfg, A)
#         A[MyConfiguration.get_random_number()] = nested_func(cfg2, A)

#     cfg1 = MyConfiguration(3)
#     cfg2 = MyConfiguration(4)
#     A = np.random.rand(20)
#     reg_A = np.copy(A)
#     reg_A[12] = reg_A[6]
#     reg_A[4] = reg_A[8]

#     constant_parameter(cfg1, cfg2, A)
#     assert np.allclose(A, reg_A)

if __name__ == '__main__':
    test_instantiated_global()
    test_nested_globals()
    # test_constant_parameter()
