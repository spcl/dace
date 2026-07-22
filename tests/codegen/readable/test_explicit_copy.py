# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``compiler.cpu.codegen_params.explicit_copy`` (lifts implicit copies to explicit
    ``CopyLibraryNode`` instances before emission). """
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
    # simplify=False: keeps the copy alive so it reaches codegen instead of being simplified away.
    sdfg = mixed_copies.to_sdfg(simplify=False)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'explicit_copy', value=explicit_copy):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_on_replaces_copynd_in_readable():
    """ ``on`` (the default) removes ``dace::CopyND`` and lowers the contiguous copy to ``memcpy``. """
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


def self_copy_sdfg(name: str) -> dace.SDFG:
    """``p[:, 3] = p[:, 4]`` as ONE AccessNode -> AccessNode edge on the SAME array -- the shape
    CloudSC's level-shift flux copies have (``pfsqlf[jk] = pfsqlf[jk-1]``).

    The memlet is src-relative: ``subset`` is the read column, ``other_subset`` the written one. A
    self-copy is the case where the endpoint names cannot disambiguate the two, so lifting it to a
    ``CopyLibraryNode`` by reading the pair positionally silently reverses the copy.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('p', [4, 5], dace.float64)
    state = sdfg.add_state('s')
    state.add_edge(state.add_access('p'), None, state.add_access('p'), None,
                   dace.Memlet(data='p', subset='0:4, 4', other_subset='0:4, 3'))
    return sdfg


def run_self_copy(implementation: str) -> numpy.ndarray:
    """Run :func:`self_copy_sdfg` under ``implementation`` on a seeded buffer; return the buffer."""
    p = numpy.arange(20, dtype=numpy.float64).reshape(4, 5).copy()
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation):
        self_copy_sdfg(f'self_copy_{implementation}')(p=p)
    return p


def test_self_copy_direction_matches_legacy():
    """A same-array copy must move data the same way under both generators.

    Regression: ``InsertExplicitCopies`` (which only the readable leg runs) used to treat a self-copy's
    ``subset`` as the DESTINATION, so the lifted ``CopyLibraryNode`` read and wrote the wrong columns
    and the readable leg computed ``p[:, 4] = p[:, 3]`` -- the copy backwards. On CloudSC that turned
    the flux level-shift ``pfsqlf[jk] = pfsqlf[jk-1]`` into ``pfsqlf[jk-1] = pfsqlf[jk]``, propagating
    uninitialised data down the column instead of the accumulated flux.
    """
    expected = numpy.arange(20, dtype=numpy.float64).reshape(4, 5).copy()
    expected[:, 3] = expected[:, 4]

    legacy = run_self_copy('legacy')
    readable = run_self_copy('experimental_readable')

    numpy.testing.assert_array_equal(legacy,
                                     expected,
                                     err_msg='legacy must copy column 4 (subset) onto column 3 (other_subset)')
    numpy.testing.assert_array_equal(readable,
                                     legacy,
                                     err_msg='readable codegen diverges from legacy on a same-array copy')


if __name__ == '__main__':
    test_on_replaces_copynd_in_readable()
    test_off_keeps_copynd_in_readable()
    test_on_is_the_default()
    test_legacy_ignores_the_flag()
    test_both_settings_compile_and_run('on')
    test_both_settings_compile_and_run('off')
    test_self_copy_direction_matches_legacy()
