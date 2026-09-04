# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The MPR entry signature's qualifiers: ``const`` on read-only pointers, C's own ``restrict``.

``Data.as_arg`` builds the signature for every DaCe backend and has no notion of a read-only
parameter, so an MPR rendering used to hand out a mutable pointer to a buffer it only reads. Two
things follow, and both were observed: every C/C++ linter reports it (cppcheck
``constParameterPointer`` on all four read-only arguments of arc_distance), and a binding that
published a ``const`` flag derived from the written-set described a signature that said the
opposite.

``__restrict__`` is the same kind of defect one level down: it is the GNU spelling, emitted because
C++ has no ``restrict`` keyword. C does, and a unit that claims to be C23 should carry it. Neither
gcc nor clang diagnoses ``__restrict__`` under ``-std=c23 -pedantic-errors`` -- the name is reserved
for the implementation -- so nothing but this test holds the spelling.

Adding ``const`` cannot change the ABI: it is a qualifier on the POINTEE, the parameter is still one
pointer, and ``T *`` converts to ``const T *`` implicitly. The numeric case here exists to say so
rather than to assume it.
"""
import ctypes
import re

import numpy as np

import dace
from dace.codegen.mpr import readonly_entry_arrays, render, written_containers

from tests.codegen.mpr.conftest import (assert_standalone, build_standalone, call_standalone, compile_diagnostics)

N = dace.symbol('N')


@dace.program
def scale(src: dace.float64[N], dst: dace.float64[N], factor: dace.float64):
    dst[:] = src * factor


def rendered(name: str, language: str):
    sdfg = scale.to_sdfg(simplify=True)
    sdfg.name = name
    result = render(sdfg, language=language)
    return result.sdfg, result.code


def entry_signature(code: str, name: str) -> str:
    """The text between the entry point's parentheses."""
    match = re.search(r'\bvoid\s+%s\s*\(([^)]*)\)' % re.escape(name), code)
    assert match is not None, code
    return match.group(1)


def test_written_containers_sees_the_destination_only():
    sdfg, _ = rendered('mpr_qual_written', 'c++')
    written = written_containers(sdfg)
    assert 'dst' in written, written
    assert 'src' not in written, written


def test_readonly_entry_arrays_is_the_read_only_pointers():
    """Only ARRAYS, and only the unwritten ones. ``factor`` is a by-value scalar, not a pointer, so
    it must not appear here -- qualifying it would be a different (and pointless) change."""
    sdfg, _ = rendered('mpr_qual_readonly', 'c++')
    assert list(readonly_entry_arrays(sdfg)) == ['src']


def test_cpp_qualifies_the_read_only_pointer():
    _, code = rendered('mpr_qual_cpp', 'c++')
    params = entry_signature(code, 'mpr_qual_cpp')
    assert 'const double * __restrict__ src' in params, params
    # The written buffer must stay mutable; qualifying it would not compile.
    assert 'const double * __restrict__ dst' not in params, params
    assert 'double * __restrict__ dst' in params, params
    assert_standalone(code, 'mpr_qual_cpp', language='c++')


def test_c_qualifies_and_uses_the_c_restrict_keyword():
    _, code = rendered('mpr_qual_c', 'c')
    params = entry_signature(code, 'mpr_qual_c')
    assert 'const double * restrict src' in params, params
    assert 'double * restrict dst' in params, params
    assert 'const double * restrict dst' not in params, params
    # The GNU spelling must be gone from the WHOLE unit, not just the signature.
    assert '__restrict__' not in code, code
    assert_standalone(code, 'mpr_qual_c', language='c')


def test_c_render_compiles_without_a_warning():
    """The qualifier is only worth having if it survives the compiler: a ``const`` pointee that the
    body assigns through would be an error, not a warning."""
    _, code = rendered('mpr_qual_c_warn', 'c')
    assert compile_diagnostics(code, 'mpr_qual_c_warn', language='c') == ''


def test_qualified_signature_still_computes_and_keeps_the_abi():
    """Called through ctypes with plain (non-const) pointers, exactly as before -- ``const`` on the
    pointee changes no argument's size, order or register class."""
    sdfg, code = rendered('mpr_qual_call', 'c')
    library = build_standalone(code, 'mpr_qual_call', language='c')
    n = 64
    src = np.arange(n, dtype=np.float64)
    dst = np.zeros(n, dtype=np.float64)
    call_standalone(library, sdfg, {'src': src, 'dst': dst, 'factor': 2.5, 'N': n})
    assert np.allclose(dst, src * 2.5)
    # The read-only argument really was read-only.
    assert np.array_equal(src, np.arange(n, dtype=np.float64))
    assert isinstance(library, ctypes.CDLL)
