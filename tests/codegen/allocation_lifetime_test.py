# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests different allocation lifetimes. """
import dace
from dace.codegen.targets import framecode
from dace.sdfg import infer_types
import numpy as np

N = dace.symbol('N')


def _test_determine_alloc(lifetime: dace.AllocationLifetime,
                          unused: bool = False) -> dace.SDFG:
    """ Creates an SDFG playground for determining allocation. """
    sdfg = dace.SDFG('lifetimetest')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('unused', [N], dace.float64, lifetime=lifetime)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:N'))

    #########################################################################
    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('A', [N], dace.float64)
    nsdfg.add_array('B', [N], dace.float64)
    nsdfg.add_transient('tmp', [N],
                        dace.float64,
                        dace.StorageType.GPU_Global,
                        lifetime=lifetime)
    nsdfg.add_transient('tmp2', [1],
                        dace.float64,
                        dace.StorageType.Register,
                        lifetime=lifetime)
    nstate = nsdfg.add_state()
    ime, imx = nstate.add_map('m2',
                              dict(i='0:20'),
                              schedule=dace.ScheduleType.GPU_Device)
    t1 = nstate.add_access('tmp')
    t2 = nstate.add_access('tmp2')
    nstate.add_nedge(t1, t2, dace.Memlet('tmp[0]'))
    nstate.add_memlet_path(nstate.add_read('A'),
                           ime,
                           t1,
                           memlet=dace.Memlet('A[i]'))
    nstate.add_memlet_path(t2,
                           imx,
                           nstate.add_write('B'),
                           memlet=dace.Memlet('B[0]', wcr='lambda a,b: a+b'))
    #########################################################################
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A'}, {'B'})
    state.add_memlet_path(state.add_read('A'),
                          me,
                          nsdfg_node,
                          dst_conn='A',
                          memlet=dace.Memlet('A[0:N]'))
    state.add_memlet_path(nsdfg_node,
                          mx,
                          state.add_write('B'),
                          src_conn='B',
                          memlet=dace.Memlet('B[0:N]'))

    # Set default storage/schedule types in SDFG
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    return sdfg, (sdfg, state, me, nsdfg, nstate, ime)


def test_determine_alloc_scope():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.Scope)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)

    # tmp cannot be allocated within the inner scope because it is GPU_Global
    assert (1, 'tmp') in codegen.to_allocate[scopes[-2]]
    assert (1, 'tmp2') in codegen.to_allocate[scopes[-1]]


def test_determine_alloc_state():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.State,
                                         unused=True)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)

    # Ensure that unused transients are not allocated
    assert not any('__0_unused' in field for field in codegen.statestruct)

    assert (1, 'tmp') in codegen.to_allocate[scopes[-2]]
    assert (1, 'tmp2') in codegen.to_allocate[scopes[-2]]


def test_determine_alloc_sdfg():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.SDFG)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)

    assert (1, 'tmp') in codegen.to_allocate[scopes[-3]]
    assert (1, 'tmp2') in codegen.to_allocate[scopes[-3]]


def test_determine_alloc_global():
    sdfg, scopes = _test_determine_alloc(dace.AllocationLifetime.Global)
    codegen = framecode.DaCeCodeGenerator()
    codegen.determine_allocation_lifetime(sdfg)
    assert any('__1_tmp' in field for field in codegen.statestruct)
    assert any('__1_tmp2' in field for field in codegen.statestruct)
    assert (1, 'tmp') in codegen.to_allocate[sdfg]
    assert (1, 'tmp2') in codegen.to_allocate[sdfg]
def test_alloc_persistent_register():
    """ Tries to allocate persistent register array. Should fail. """
    @dace.program
    def lifetimetest(input: dace.float64[N]):
        tmp = dace.ndarray([1], input.dtype)
        return tmp + 1

    sdfg: dace.SDFG = lifetimetest.to_sdfg()
    sdfg.arrays['tmp'].storage = dace.StorageType.Register
    sdfg.arrays['tmp'].lifetime = dace.AllocationLifetime.Persistent

    try:
        sdfg.validate()
        raise AssertionError('SDFG should not be valid')
    except dace.sdfg.InvalidSDFGError:
        print('Exception caught, test passed')


if __name__ == '__main__':
    test_determine_alloc_scope()
    test_determine_alloc_state()
    test_determine_alloc_sdfg()
    test_determine_alloc_global()
    test_alloc_persistent_register()
