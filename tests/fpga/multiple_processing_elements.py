# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Tests for kernel detections, both among and within connected components

import dace
import numpy as np
import pytest
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG, GPUTransformSDFG, NestSDFG


def test_PEs_inside_component_0():
    '''
    Tests for PEs detection inside a single Component.
    It computes z =(x+y) + (v+w)

    High-level overview:
     ┌───────────┐        ┌───────────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_2 │◄───┘
                └───────────┘
    Map_0 and Map_1 should belong to two distinct PEs
    :return:
    '''
    @dace.program
    def PEs_inside_component_0(x: dace.float32[8], y: dace.float32[8],
                               v: dace.float32[8], w: dace.float32[8]):
        tmp1 = x + y
        tmp2 = v + w
        return tmp1 + tmp2

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)

    sdfg = PEs_inside_component_0.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    sdfg.save('/tmp/out.sdfg')
    program = sdfg.compile()
    for node, state in program.sdfg.all_nodes_recursive():
        if hasattr(node, '_kernel'):
            print(node, node._kernel)

    z = program(x=x, y=y, v=v, w=w)
    assert np.allclose(z, x + y + v + w)


def test_PEs_inside_component_1():
    '''
    Tests for PEs detection inside a single Component.
    It computes
    - z = alpha*((x+y) + (v+w))
    - t = beta*((x+y) + (v+w))

    High-level overview:
     ┌───────────┐        ┌───────────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_2 │◄───┘
            ────└───────────┘────
            │                   │
     ┌──────v────┐        ┌─────v─────┐
     │   Mul_3   │        │   Mul_4   │
     └───────────┘        └───────────┘


    :return:
    '''
    @dace.program
    def PEs_inside_component_1(x: dace.float32[8], y: dace.float32[8],
                               v: dace.float32[8], w: dace.float32[8],
                               z: dace.float32[8], t: dace.float32[8],
                               alpha: dace.float32, beta: dace.float32):
        tmp1 = x + y
        tmp2 = v + w
        tmp3 = tmp1 + tmp2
        z[:] = alpha * tmp3
        t[:] = beta * tmp3

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)
    z = np.random.rand(8).astype(np.float32)
    t = np.random.rand(8).astype(np.float32)
    alpha = 1.0
    beta = 2.0

    sdfg = PEs_inside_component_1.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    sdfg.save('/tmp/out.sdfg')
    program = sdfg.compile()
    for node, state in program.sdfg.all_nodes_recursive():
        if hasattr(node, '_kernel'):
            print(node, node._kernel)

    program(x=x, y=y, v=v, w=w, z=z, t=t, alpha=alpha, beta=beta)
    ref_z = alpha * (x + y + v + w)
    ref_t = beta * (x + y + v + w)
    assert np.allclose(z, ref_z)
    assert np.allclose(t, ref_t)


def test_PEs_inside_component_2():
    '''
    Tests for PEs detection inside a single Component.
    It computes z =(x+y) and t = (y+v)


    High-level overview:

        x            y         v
        │            │         │
     ┌──V────────<───┘────>────V──────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └───────────┘        └───────────┘

    Map_0 and Map_1 should belong to two distinct PEs
    NOTE: this kind of graph was already executed in parallel

    TODO: let the result is then used again?

    :return:
    '''
    @dace.program
    def PEs_inside_component_2(x: dace.float32[8], y: dace.float32[8],
                               v: dace.float32[8], z: dace.float32[8],
                               t: dace.float32[8]):
        z[:] = x + y
        t[:] = y + v

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    z = np.random.rand(8).astype(np.float32)
    t = np.random.rand(8).astype(np.float32)

    sdfg = PEs_inside_component_2.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    sdfg.save('/tmp/out.sdfg')
    program = sdfg.compile()
    for node, state in program.sdfg.all_nodes_recursive():
        if hasattr(node, '_kernel'):
            print(node, node._kernel)

    program(x=x, y=y, v=v, t=t, z=z)
    assert np.allclose(z, x + y)
    assert np.allclose(t, v + y)


def test_PEs_LNs_inside_component():
    '''
    Tests for PEs detection inside a single Component where we
    have multiple LNs.
    It computes z =(x+y) + (v+w)

    High-level overview:
     ┌───────────┐        ┌───────────┐
     │  Matmul_0 │        │  Matmul_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │   Dot_2   │◄───┘
                └───────────┘

    :return:
    '''
    @dace.program
    def PEs_LNs_inside_component(A: dace.float32[8, 8], x: dace.float32[8],
                                 B: dace.float32[8, 8], y: dace.float32[8]):
        tmp1 = A @ x
        tmp2 = B @ y
        return np.dot(tmp1, tmp2)

    A = np.random.rand(8, 8).astype(np.float32)
    B = np.random.rand(8, 8).astype(np.float32)
    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)

    sdfg = PEs_LNs_inside_component.to_sdfg()
    sdfg.save('/tmp/pre.sdfg')
    from dace.transformation.interstate import GPUTransformSDFG
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    sdfg.save('/tmp/out.sdfg')
    # sdfg.expand_library_nodes()
    # sdfg.save('/tmp/expanded.sdfg')
    program = sdfg.compile()
    for node, state in program.sdfg.all_nodes_recursive():
        if hasattr(node, '_kernel'):
            print(node, node._kernel)

    z = program(A=A, x=x, B=B, y=y)

    ref = np.dot(A @ x, B @ y)
    assert np.allclose(z, ref)


def test_kernel_LN(flatten):
    '''
    A single NSDFG originated by a LibNode expansion (Matmul)
    :return:
    '''
    @dace.program
    def test_kernel_LN(A: dace.float32[32, 32], B: dace.float32[32, 32]):
        return A @ B

    A = np.random.rand(32, 32).astype(np.float32)
    B = np.random.rand(32, 32).astype(np.float32)

    sdfg = test_kernel_LN.to_sdfg()
    sdfg.save('/tmp/pre.sdfg')
    from dace.transformation.interstate import GPUTransformSDFG

    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    from dace.libraries.blas import Gemm
    Gemm.default_implementation = "FPGA1DSystolic"
    sdfg.save('/tmp/out.sdfg')
    if flatten:
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])
        sdfg.save('/tmp/expanded.sdfg')

    # sdfg.save('/tmp/expanded.sdfg')
    program = sdfg.compile()
    for node, state in program.sdfg.all_nodes_recursive():
        if hasattr(node, '_kernel'):
            print(node, node._kernel)

    C = program(A=A, B=B)

    assert np.allclose(C, A @ B)


if __name__ == "__main__":
    # test_PEs_inside_component_0()
    # test_PEs_inside_component_1()
    # test_PEs_inside_component_2()
    # test_PEs_LNs_inside_component()

    # TODO: for this, we should see if we are able to find cuts or not
    # If this is inlined we have 2 possibilities:
    # - either we detect the right number of kernels (4)
    # - or we detect that for each indipendent component there is no split....then we just
    #   use the previous version.
    # THEREFORE, one way of doing this is 1) we split into indipendent compoenents 2) for each of them
    # we see if they can be further split 3) PEs remains only for systolic arrays
    # IN GENERALE, DOVREMMO RISOLVERE QUESTA SITUAZIONE E DETERMINARE IL NUMERO DI KERNEL GIUSTI
    # anche se eper questo particolare caso ci possiamo girare intorno
    # Un altro probelma che c'e' qui e' il seguente: quando andiamo a leggere la mappa B, per via
    # dei dfs_edges mi segue prima il path dal source node, e quindi mi va ad usare un nuovo kernel id
    test_kernel_LN(True)
