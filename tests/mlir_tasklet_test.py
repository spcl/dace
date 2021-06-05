# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
from dace.dtypes import Language
import numpy as np

@pytest.mark.mlir
def test_mlir_tasklet_explicit():
    sdfg = dace.SDFG('mlir_tasklet_explicit')
    state = sdfg.add_state()
    sdfg.add_array('A', [1], dtype=dace.int32)
    sdfg.add_array('B', [1], dtype=dace.int32)
    sdfg.add_array('C', [1], dtype=dace.int32)

    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a', 'b'},
                                outputs={'c'},
                                code='''
                                    module  {
                                        func @mlir_entry(%a: i32, %b: i32) -> i32 {
                                            %0 = addi %b, %a  : i32
                                            return %0 : i32
                                        }
                                    } 
                                    ''',
                                language=dace.Language.MLIR)

    A = state.add_read('A')
    B = state.add_read('B')
    C = state.add_write('C')

    state.add_edge(A, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(B, None, tasklet, 'b', dace.Memlet('B[0]'))
    state.add_edge(tasklet, 'c', C, None, dace.Memlet('C[0]'))
    sdfg.validate()

    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.random.randint(0, 100, 1).astype(np.int32)
    c = np.random.randint(0, 100, 1).astype(np.int32)

    sdfg(A=a, B=b, C=c)
    assert c == a+b

@dace.program
def mlir_tasklet_implicit(A: dace.uint32[3], B: dace.uint32[2], C: dace.uint32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0] 

        """
        module  {
            func @mlir_entry(%a: i32, %b: i32) -> i32 {
                %0 = addi %b, %a  : i32
                return %0 : i32
            }
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_implicit():
    A = dace.ndarray((1, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)
    C = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    mlir_tasklet_implicit(A, B, C)
    assert C[0] == 7

@dace.program
def mlir_tasklet_swapped(A: dace.uint32[3], B: dace.uint32[2], C: dace.uint32[1]):
    @dace.tasklet('MLIR')
    def add():
        b << A[0]
        a << B[0]
        c >> C[0] 

        """
        module  {
            func @mlir_entry(%a: i32, %b: i32) -> i32 {
                %0 = addi %b, %a  : i32
                return %0 : i32
            }
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_swapped():
    A = dace.ndarray((1, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)
    C = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    mlir_tasklet_swapped(A, B, C)
    assert C[0] == 7

@dace.program
def mlir_tasklet_no_entry(A: dace.uint32[3], B: dace.uint32[2], C: dace.uint32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0] 

        """
        module  {
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_no_entry():
    A = dace.ndarray((1, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)
    C = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    with pytest.raises(RuntimeError):
        mlir_tasklet_no_entry(A, B, C)

@dace.program
def mlir_tasklet_double_entry(A: dace.uint32[3], B: dace.uint32[2], C: dace.uint32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0] 

        """
        module  {
            func @mlir_entry(%a: i32, %b: i32) -> i32 {
                %0 = addi %b, %a  : i32
                return %0 : i32
            }

            func @mlir_entry(%a: i32, %b: i32) -> i32 {
                %0 = addi %b, %a  : i32
                return %0 : i32
            }
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_double_entry():
    A = dace.ndarray((1, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)
    C = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    with pytest.raises(RuntimeError):
        mlir_tasklet_double_entry(A, B, C)

@dace.program
def mlir_tasklet_llvm_dialect(A: dace.uint32[3], B: dace.uint32[2], C: dace.uint32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0] 

        """
        module  {
            llvm.func @mlir_entry(%a: i32, %b: i32) -> i32 {
                %0 = llvm.add %b, %a  : i32
                llvm.return %0 : i32
            }
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_llvm_dialect():
    A = dace.ndarray((1, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)
    C = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    #mlir_tasklet_llvm_dialect(A, B, C)
    #assert C[0] == 7

@dace.program
def mlir_tasklet_float(A: dace.float32[3], B: dace.float32[2], C: dace.float32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0] 

        """
        module  {
            func @mlir_entry(%a: f32, %b: f32) -> f32 {
                %0 = addf %b, %a  : f32
                return %0 : f32
            }
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_float():
    A = dace.ndarray((1, ), dace.float32)
    B = dace.ndarray((1, ), dace.float32)
    C = dace.ndarray((1, ), dace.float32)

    A[:] = 5.5
    B[:] = 2.2
    C[:] = 15.15

    mlir_tasklet_float(A, B, C)
    assert (C[0] - 7.7) < 0.00000001 #precision?

@dace.program
def mlir_tasklet_double_return(A: dace.uint32[4], B: dace.uint32[3], C: dace.uint32[2], D: dace.uint32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0]
        d >> D[0] 

        """
        module  {
            func @mlir_entry(%a: i32, %b: i32) -> (i32, i32) {
                %0 = addi %b, %a : i32
                %1 = addi %0, constant 1 : i32
                return %0, %1 : i32, i32
            }
        }
        """

@pytest.mark.mlir
def test_mlir_tasklet_double_return():
    A = dace.ndarray((1, ), dace.uint32)
    B = dace.ndarray((1, ), dace.uint32)
    C = dace.ndarray((1, ), dace.uint32)
    D = dace.ndarray((1, ), dace.uint32)

    A[:] = 5
    B[:] = 2
    C[:] = 15
    D[:] = 12

    #mlir_tasklet_double_return(A, B, C, D)
    #assert C[0] == 7
    #assert D[0] == 17

if __name__ == "__main__":
    test_mlir_tasklet_explicit()
    test_mlir_tasklet_implicit()
    test_mlir_tasklet_swapped()
    test_mlir_tasklet_no_entry()
    test_mlir_tasklet_double_entry()
    test_mlir_tasklet_llvm_dialect()
    test_mlir_tasklet_float()
    test_mlir_tasklet_double_return()
    