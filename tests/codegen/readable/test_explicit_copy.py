# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``compiler.cpu.codegen_params.explicit_copy``: whether the readable CPU code generator
    lifts implicit copies to explicit ``CopyLibraryNode`` instances before emission, so ``ExpandAuto``
    lowers each on its merits (single element -> ``=`` tasklet, contiguous -> ``std::memcpy``) instead
    of the ``dace::CopyND`` template the implicit path emits. """
import numpy
import pytest

import dace
from dace.config import set_temporary

N = 16


@dace.program
def mixed_copies(A: dace.float64[N], B: dace.float64[N], sc_in: dace.float64[1], sc_out: dace.float64[1]):
    B[:] = A  # contiguous, same-layout, multi-element copy -> memcpy when lifted
    sc_out[:] = sc_in  # single-element copy -> '=' tasklet when lifted


def generate(implementation: str, explicit_copy: str) -> str:
    # simplify=False: a copy straight into an output argument is exactly what a redundant-copy
    # simplify pass would try to remove, and the point here is to keep the copy so it reaches codegen.
    sdfg = mixed_copies.to_sdfg(simplify=False)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'explicit_copy', value=explicit_copy):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_on_replaces_copynd_in_readable():
    """ ``on`` (the default) removes the ``dace::CopyND`` template and lowers the contiguous copy to
        ``std::memcpy`` instead. """
    on = generate('experimental_readable', 'on')
    assert 'dace::CopyND' not in on, 'explicit_copy on should leave no dace::CopyND behind'
    assert 'memcpy' in on, 'the contiguous copy should lower to memcpy'


def test_off_keeps_copynd_in_readable():
    """ ``off`` takes the pass out: the copies stay on the implicit ``dace::CopyND`` path. """
    off = generate('experimental_readable', 'off')
    assert 'dace::CopyND' in off, 'explicit_copy off should keep the implicit CopyND lowering'


def test_on_is_the_default():
    """ The schema default is ``on``: generating without touching the key matches an explicit ``on``. """
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        sdfg = mixed_copies.to_sdfg(simplify=False)
        default = '\n'.join(o.code for o in sdfg.generate_code() if o.language == 'cpp')
    assert default == generate('experimental_readable', 'on')


def test_legacy_ignores_the_flag():
    """ Legacy never enters the readable pipeline, so its output is byte-identical across the flag. """
    assert generate('legacy', 'on') == generate('legacy', 'off')


@pytest.mark.parametrize('explicit_copy', ['on', 'off'])
def test_both_settings_compile_and_run(explicit_copy):
    """ Either setting must produce the same, correct numbers. """
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'explicit_copy', value=explicit_copy):
        A = numpy.random.rand(N)
        B = numpy.zeros(N)
        sc_in = numpy.array([3.5])
        sc_out = numpy.zeros(1)
        mixed_copies(A=A, B=B, sc_in=sc_in, sc_out=sc_out)
        assert numpy.array_equal(B, A)
        assert numpy.array_equal(sc_out, sc_in)


if __name__ == '__main__':
    test_on_replaces_copynd_in_readable()
    test_off_keeps_copynd_in_readable()
    test_on_is_the_default()
    test_legacy_ignores_the_flag()
    test_both_settings_compile_and_run('on')
    test_both_settings_compile_and_run('off')
