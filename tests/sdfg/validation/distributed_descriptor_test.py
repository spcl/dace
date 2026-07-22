# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``DistributedDescriptor`` is the one ``Data`` subtree with neither ``strides`` nor ``total_size``.

It is stored in ``sdfg.arrays`` as a transient, so the transient-allocation checks in
``validate_sdfg`` reach it and must skip the checks that do not apply to a communicator.
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
    Data subclass gains strides, it must join that tuple or its allocation goes unchecked."""
    allocated = (dt.Array, dt.Scalar, dt.Stream, dt.Structure)
    for cls in {c for c in vars(dt).values() if inspect.isclass(c) and issubclass(c, dt.Data)}:
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
