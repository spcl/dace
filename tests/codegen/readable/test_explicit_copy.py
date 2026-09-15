# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``compiler.cpu.explicit_copy`` (lifts implicit copies to explicit
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


def generate(implementation: str, explicit_copy: bool) -> str:
    # simplify=False: keeps the copy alive so it reaches codegen instead of being simplified away.
    sdfg = mixed_copies.to_sdfg(simplify=False)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'explicit_copy', value=explicit_copy):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_readable_always_lowers():
    """ The readable generator requires the lowering, so the knob has no effect on it: either value
    removes ``dace::CopyND`` and lowers the contiguous copy to ``memcpy``. """
    for value in (True, False):
        code = generate('experimental_readable', value)
        assert 'dace::CopyND' not in code, f'readable must lower copies regardless of the knob (got {value})'
        assert 'memcpy' in code, 'the contiguous copy should lower to memcpy'


def test_legacy_honours_the_flag_and_defaults_on():
    """ The knob governs only the classic generator, and it is ON by default.

    Turning it off is what recovers the implicit ``dace::CopyND`` emission, byte-identical to
    upstream, which is what makes the classic path an A/B reference for the new generators. On is
    the default because off has no lowering at all for a dtype-converting copy: it reaches the
    compiler as a CopyND template instantiated on one element type holding a pointer of the other.
    """
    on = generate('legacy', True)
    assert 'dace::CopyND' not in on, 'explicit_copy on should leave no dace::CopyND behind on legacy'
    assert 'memcpy' in on, 'the contiguous copy should lower to memcpy on legacy too'
    off = generate('legacy', False)
    assert 'dace::CopyND' in off, 'off should keep the implicit CopyND lowering'
    with set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
        sdfg = mixed_copies.to_sdfg(simplify=False)
        default = '\n'.join(o.code for o in sdfg.generate_code() if o.language == 'cpp')
    assert default == on, 'the schema default must be on'


@pytest.mark.parametrize('implementation', ['experimental_readable', 'legacy'])
@pytest.mark.parametrize('explicit_copy', [True, False])
def test_both_settings_compile_and_run(implementation, explicit_copy):
    """ Either setting must produce the same, correct numbers, on either generator. """
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'explicit_copy', value=explicit_copy):
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
    test_readable_always_lowers()
    test_legacy_honours_the_flag_and_defaults_on()
    for implementation in ('experimental_readable', 'legacy'):
        test_both_settings_compile_and_run(implementation, True)
        test_both_settings_compile_and_run(implementation, False)
    test_self_copy_direction_matches_legacy()
