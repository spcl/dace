# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``DistributedDescriptor`` is the one ``Data`` subtree that allocates nothing.

It is stored in ``sdfg.arrays`` as a transient, so the transient-allocation checks in
``validate_sdfg`` reach it and must skip the checks that do not apply to a communicator. It declares
that by defining ``strides`` and ``total_size`` as constant stubs (``[]`` and ``0``) rather than by
lacking them.
"""

import inspect

import pytest

import dace
from dace import data as dt


def sdfg_with(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_state('s', is_start_block=True)
    return sdfg


def test_process_grid_descriptor_validates():
    sdfg = sdfg_with('pgrid')
    grid = sdfg.add_pgrid(shape=[2, 2])
    assert isinstance(sdfg.arrays[grid], dt.DistributedDescriptor)
    assert sdfg.arrays[grid].transient, 'the checks under test only run for transients'
    sdfg.validate()


def test_subarray_descriptor_validates():
    sdfg = sdfg_with('subarray')
    sub = sdfg.add_subarray(dace.float64, [16, 16], [8, 8])
    assert isinstance(sdfg.arrays[sub], dt.DistributedDescriptor)
    sdfg.validate()


def test_allocated_kinds_are_exactly_the_ones_with_strides():
    """validate_sdfg checks strides/total_size for (Array, Scalar, Stream, Structure). If a new
    Data subclass gains strides, it must join that tuple or its allocation goes unchecked.

    The ``DistributedDescriptor`` subtree is the one exemption, and having the two attributes is not
    what exempts it -- it is that both are CONSTANT stubs saying there is no allocation to check. A
    subclass that overrides either one is a real buffer and must join the allowlist, so the stub
    values and the fact that nobody overrides them are pinned here too.
    """
    assert dt.DistributedDescriptor.strides.fget(None) == []
    assert dt.DistributedDescriptor.total_size.fget(None) == 0

    allocated = (dt.Array, dt.Scalar, dt.Stream, dt.Structure)
    for cls in {c for c in vars(dt).values() if inspect.isclass(c) and issubclass(c, dt.Data)}:
        if issubclass(cls, dt.DistributedDescriptor):
            assert cls.strides is dt.DistributedDescriptor.strides, f'{cls.__name__} overrides the strides stub'
            assert cls.total_size is dt.DistributedDescriptor.total_size, f'{cls.__name__} overrides total_size'
            continue
        has_strides = 'strides' in dir(cls) and 'total_size' in dir(cls)
        assert has_strides == issubclass(
            cls, allocated), (f'{cls.__name__}: strides/total_size={has_strides} but covered-by-allowlist='
                              f'{issubclass(cls, allocated)}')


def test_allocation_checks_still_run_for_real_descriptors():
    """The skip must be scoped to DistributedDescriptor -- an undefined stride still raises."""
    sdfg = sdfg_with('undef_stride')
    sdfg.add_transient('T', [8], dace.float64)
    sdfg.arrays['T'].strides = [dace.symbolic.UndefinedSymbol()]
    with pytest.raises(dace.sdfg.validation.InvalidSDFGError, match='undefined symbol in stride'):
        sdfg.validate()


if __name__ == '__main__':
    test_process_grid_descriptor_validates()
    test_subarray_descriptor_validates()
    test_allocated_kinds_are_exactly_the_ones_with_strides()
    test_allocation_checks_still_run_for_real_descriptors()
