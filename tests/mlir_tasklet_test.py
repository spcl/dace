# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
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
    assert c == a + b


@pytest.mark.mlir
def test_mlir_tasklet_explicit_vec():
    sdfg = dace.SDFG('mlir_tasklet_explicit_vec')
    state = sdfg.add_state()
    sdfg.add_array('A', [4], dace.vector(dace.int32, 4))
    sdfg.add_array('B', [4], dace.vector(dace.int32, 4))
    sdfg.add_array('C', [4], dace.vector(dace.int32, 4))

    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a', 'b'},
                                outputs={'c'},
                                code='''
                                    module  {
                                        func @mlir_entry(%a: vector<4xi32>, %b: vector<4xi32>) -> vector<4xi32> {
                                            %0 = addi %b, %a  : vector<4xi32>
                                            return %0 : vector<4xi32>
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

    a = np.random.rand(4).astype(np.int32)
    b = np.random.rand(4).astype(np.int32)
    c = np.random.rand(4).astype(np.int32)

    sdfg(A=a, B=b, C=c)
    assert (c == a + b).all()


@dace.program
def mlir_tasklet_implicit(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
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
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)
    C = dace.ndarray((1, ), dace.int32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    mlir_tasklet_implicit(A, B, C)
    assert C[0] == 7


@pytest.mark.mlir
def test_mlir_tasklet_inference():
    sdfg = dace.SDFG('mlir_tasklet_explicit_vec')
    state = sdfg.add_state()

    # Test Vectors
    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a', 'b'},
                                outputs={'c'},
                                code='''
                                    module  {
                                        func @mlir_entry(%a: vector<4xi32>, %b: vector<4xi32>) -> vector<4xi32> {
                                            %0 = addi %b, %a  : vector<4xi32>
                                            return %0 : vector<4xi32>
                                        }
                                    }
                                    ''',
                                language=dace.Language.MLIR)
    tasklet.infer_connector_types(sdfg, state)
    assert isinstance(tasklet.in_connectors['a'], dace.dtypes.vector)
    assert tasklet.in_connectors['a'].veclen == 4
    assert isinstance(tasklet.in_connectors['a'].base_type, dace.dtypes.typeclass)
    assert tasklet.in_connectors['a'].base_type.ctype == "int"

    assert isinstance(tasklet.in_connectors['b'], dace.dtypes.vector)
    assert tasklet.in_connectors['b'].veclen == 4
    assert isinstance(tasklet.in_connectors['b'].base_type, dace.dtypes.typeclass)
    assert tasklet.in_connectors['b'].base_type.ctype == "int"

    assert isinstance(tasklet.out_connectors['c'], dace.dtypes.vector)
    assert tasklet.out_connectors['c'].veclen == 4
    assert isinstance(tasklet.out_connectors['c'].base_type, dace.dtypes.typeclass)
    assert tasklet.out_connectors['c'].base_type.ctype == "int"

    # Test ints
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

    tasklet.infer_connector_types(sdfg, state)
    assert isinstance(tasklet.in_connectors['a'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['a'].ctype == "int"

    assert isinstance(tasklet.in_connectors['b'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['b'].ctype == "int"

    assert isinstance(tasklet.out_connectors['c'], dace.dtypes.typeclass)
    assert tasklet.out_connectors['c'].ctype == "int"

    # Test floats
    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a', 'b'},
                                outputs={'c'},
                                code='''
                                module  {
                                    func @mlir_entry(%a: f32, %b: f32) -> f32 {
                                        %0 = addf %b, %a  : f32
                                        return %0 : f32
                                    }
                                }
                                ''',
                                language=dace.Language.MLIR)

    tasklet.infer_connector_types(sdfg, state)
    assert isinstance(tasklet.in_connectors['a'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['a'].ctype == "float"

    assert isinstance(tasklet.in_connectors['b'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['b'].ctype == "float"

    assert isinstance(tasklet.out_connectors['c'], dace.dtypes.typeclass)
    assert tasklet.out_connectors['c'].ctype == "float"

    # Test generic
    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a', 'b'},
                                outputs={'c'},
                                code='''
                                "module"() ( {
                                "func"() ( {
                                ^bb0(%a: i32, %b: i32):  // no predecessors
                                    %0 = "std.addi"(%b, %a) : (i32, i32) -> i32
                                    "std.return"(%0) : (i32) -> ()
                                }) {sym_name = "mlir_entry", type = (i32, i32) -> i32} : () -> ()
                                }) : () -> ()
                                ''',
                                language=dace.Language.MLIR)

    tasklet.infer_connector_types(sdfg, state)
    assert isinstance(tasklet.in_connectors['a'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['a'].ctype == "int"

    assert isinstance(tasklet.in_connectors['b'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['b'].ctype == "int"

    assert isinstance(tasklet.out_connectors['c'], dace.dtypes.typeclass)
    assert tasklet.out_connectors['c'].ctype == "int"

    # Test signed int
    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a'},
                                outputs={'c'},
                                code='''
                                module  {
                                    func @mlir_entry(%a: si32) -> si32 {
                                        return %0 : si32
                                    }
                                }
                                ''',
                                language=dace.Language.MLIR)

    tasklet.infer_connector_types(sdfg, state)
    assert isinstance(tasklet.in_connectors['a'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['a'].ctype == "int"

    assert isinstance(tasklet.out_connectors['c'], dace.dtypes.typeclass)
    assert tasklet.out_connectors['c'].ctype == "int"

    # Test unsigned int
    tasklet = state.add_tasklet(name='mlir_tasklet',
                                inputs={'a'},
                                outputs={'c'},
                                code='''
                                module  {
                                    func @mlir_entry(%a: ui32) -> ui32 {
                                        return %0 : ui32
                                    }
                                }
                                ''',
                                language=dace.Language.MLIR)

    tasklet.infer_connector_types(sdfg, state)
    assert isinstance(tasklet.in_connectors['a'], dace.dtypes.typeclass)
    assert tasklet.in_connectors['a'].ctype == "unsigned int"

    assert isinstance(tasklet.out_connectors['c'], dace.dtypes.typeclass)
    assert tasklet.out_connectors['c'].ctype == "unsigned int"


@dace.program
def mlir_tasklet_swapped(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
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
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)
    C = dace.ndarray((1, ), dace.int32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    mlir_tasklet_swapped(A, B, C)
    assert C[0] == 7


@dace.program
def mlir_tasklet_no_entry(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0]
        """
        module  {
        }
        """


@dace.program
def mlir_tasklet_no_entry_generic(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0]
        """
        "module"() ( {
        }) : () -> ()
        """


@pytest.mark.mlir
def test_mlir_tasklet_no_entry():
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)
    C = dace.ndarray((1, ), dace.int32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    with pytest.raises(SyntaxError):
        mlir_tasklet_no_entry(A, B, C)

    with pytest.raises(SyntaxError):
        mlir_tasklet_no_entry_generic(A, B, C)


@dace.program
def mlir_tasklet_double_entry(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
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
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)
    C = dace.ndarray((1, ), dace.int32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    with pytest.raises(SyntaxError):
        mlir_tasklet_double_entry(A, B, C)


@dace.program
def mlir_tasklet_double_return(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0]
        """
        module  {
            func @mlir_entry(%a: i32, %b: i32) -> (i32, i32) {
                %0 = addi %b, %a  : i32
                return %0, %0 : i32, i32
            }
        }
        """


@dace.program
def mlir_tasklet_double_return_generic(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0]
        """
        "module"() ( {
        "func"() ( {
        ^bb0(%a: i32, %b: i32):  // no predecessors
            %0 = "std.addi"(%b, %a) : (i32, i32) -> i32
            "std.return"(%0, %0) : (i32, i32) -> ()
        }) {sym_name = "mlir_entry", type = (i32, i32) -> (i32, i32)} : () -> ()
        }) : () -> ()
        """


@pytest.mark.mlir
def test_mlir_tasklet_double_return():
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)
    C = dace.ndarray((1, ), dace.int32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    with pytest.raises(SyntaxError):
        mlir_tasklet_double_return(A, B, C)

    with pytest.raises(SyntaxError):
        mlir_tasklet_double_return_generic(A, B, C)


@dace.program
def mlir_tasklet_llvm_dialect_opt(A: dace.int32[3], B: dace.int32[2], C: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        b << B[0]
        c >> C[0]
        """
        "builtin.module"() ( {
        "builtin.func"() ( {
        ^bb0(%a: i32, %b: i32):  // no predecessors
            %0 = "std.addi"(%b, %a) : (i32, i32) -> i32
            "std.return"(%0) : (i32) -> ()
        }) {sym_name = "mlir_entry", type = (i32, i32) -> i32} : () -> ()
        }) : () -> ()
        """


@pytest.mark.mlir
def test_mlir_tasklet_llvm_dialect():
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)
    C = dace.ndarray((1, ), dace.int32)

    A[:] = 5
    B[:] = 2
    C[:] = 15

    mlir_tasklet_llvm_dialect_opt(A, B, C)
    assert C[0] == 7


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
    assert np.allclose(C[0], 7.7)


@dace.program
def mlir_tasklet_recursion(A: dace.int32[2], B: dace.int32[1]):
    @dace.tasklet('MLIR')
    def fib():
        a << A[0]
        b >> B[0]
        """
        "builtin.module"() ( {
        "builtin.func"() ( {
        ^bb0(%a: i32):  // no predecessors
            %c0_i32 = "std.constant"() {value = 0 : i32} : () -> i32
            %c1_i32 = "std.constant"() {value = 1 : i32} : () -> i32
            %0 = "std.cmpi"(%a, %c0_i32) {predicate = 0 : i64} : (i32, i32) -> i1
            %1 = "std.cmpi"(%a, %c1_i32) {predicate = 0 : i64} : (i32, i32) -> i1
            "std.cond_br"(%0)[^bb2, ^bb1] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (i1) -> ()
        ^bb1:  // pred: ^bb0
            "std.cond_br"(%1)[^bb3, ^bb4] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (i1) -> ()
        ^bb2:  // pred: ^bb0
            "std.return"(%c0_i32) : (i32) -> ()
        ^bb3:  // pred: ^bb1
            "std.return"(%c1_i32) : (i32) -> ()
        ^bb4:  // pred: ^bb1
            %2 = "std.subi"(%a, %c1_i32) : (i32, i32) -> i32
            %3 = "std.subi"(%2, %c1_i32) : (i32, i32) -> i32
            %4 = "std.call"(%2) {callee = @mlir_entry} : (i32) -> i32
            %5 = "std.call"(%3) {callee = @mlir_entry} : (i32) -> i32
            %6 = "std.addi"(%4, %5) : (i32, i32) -> i32
            "std.return"(%6) : (i32) -> ()
        }) {sym_name = "mlir_entry", type = (i32) -> i32} : () -> ()
        }) : () -> ()
        """


@pytest.mark.mlir
def test_mlir_tasklet_recursion():
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)

    A[:] = 10
    B[:] = 2

    mlir_tasklet_recursion(A, B)
    assert B[0] == 55


@dace.program
def mlir_tasklet_long_name(A: dace.int32[2], B: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        a << A[0]
        longName >> B[0]
        """
        module  {
            func @mlir_entry(%a: i32) -> i32 {
                return %a : i32
            }
        }
        """


@pytest.mark.mlir
def test_mlir_tasklet_long_name():
    A = dace.ndarray((1, ), dace.int32)
    B = dace.ndarray((1, ), dace.int32)

    A[:] = 10
    B[:] = 2

    mlir_tasklet_long_name(A, B)
    assert B[0] == 10


@dace.program
def mlir_tasklet_no_input(A: dace.int32[1]):
    @dace.tasklet('MLIR')
    def add():
        c >> A[0]
        """
        module  {
            func @mlir_entry() -> i32 {
                %5 = constant 5 : i32
                return %5 : i32
            }
        }
        """


@pytest.mark.mlir
def test_mlir_tasklet_no_input():
    A = dace.ndarray((1, ), dace.int32)

    A[:] = 10

    mlir_tasklet_no_input(A)
    assert A[0] == 5


if __name__ == "__main__":
    test_mlir_tasklet_explicit()
    test_mlir_tasklet_explicit_vec()
    test_mlir_tasklet_implicit()
    test_mlir_tasklet_inference()
    test_mlir_tasklet_swapped()
    test_mlir_tasklet_no_entry()
    test_mlir_tasklet_double_entry()
    test_mlir_tasklet_double_return()
    test_mlir_tasklet_llvm_dialect()
    test_mlir_tasklet_float()
    test_mlir_tasklet_recursion()
    test_mlir_tasklet_long_name()
    test_mlir_tasklet_no_input()
