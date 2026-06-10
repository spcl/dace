# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the :data:`dace.symbolic.ONE` broadcast-dim sentinel.

The sentinel marks tile-shape broadcast dimensions in the K-dim
vectorisation track (design section 3.8.1 / 3.8.2). It must:

* Be identity-comparable across imports (one shared object).
* Survive sympy arithmetic / simplify (the firewall is identity-based, so
  any pass that special-cases literal-1 dims must see ``ONE`` as a
  non-literal symbol).
* Round-trip through SDFG save / load without corruption.
* Block :class:`ConvertLengthOneArraysToScalars` from scalarising arrays
  whose shape carries ``ONE``.
"""
import dace
import sympy

from dace.symbolic import ONE
from dace.transformation.passes.length_one_array_scalar_conversion import (ConvertLengthOneArraysToScalars)


def test_identity_across_imports():
    """The sentinel is one shared object; identity comparison must work
    after re-importing from the same module path."""
    from dace.symbolic import ONE as ONE2
    assert ONE is ONE2


def test_shape_with_one_preserves_symbol():
    """A descriptor with shape ``(M, ONE, K)`` keeps ``ONE`` as a free
    symbol of the middle dim. Identity ``ONE in dim.free_symbols`` is the
    firewall check."""
    M = dace.symbol("M")
    K = dace.symbol("K")
    shape = (M, ONE, K)
    assert ONE in shape[1].free_symbols
    assert ONE not in shape[0].free_symbols
    assert ONE not in shape[2].free_symbols


def test_sympy_arithmetic_preserves_one():
    """``ONE`` survives ``Mul`` / ``Add`` / ``simplify`` -- it never
    collapses to literal 1."""
    N = dace.symbol("N")
    assert ONE in (ONE * N).free_symbols
    assert ONE in (ONE + 0).free_symbols
    assert sympy.simplify(ONE * N) != N  # ONE stays opaque


def test_subs_one_to_one():
    """Codegen substitution: ``ONE -> 1`` folds out, leaving a literal."""
    N = dace.symbol("N")
    assert (ONE * N).subs(ONE, 1) == N


def test_convert_length_one_arrays_skips_one_marked():
    """:class:`ConvertLengthOneArraysToScalars` must NOT scalarise an
    array whose shape carries ``ONE``."""
    sdfg = dace.SDFG("one_marked_fixture")
    sdfg.add_array("A", (ONE, ), dace.float64, transient=True)
    pre = type(sdfg.arrays["A"]).__name__
    assert pre == "Array"
    ConvertLengthOneArraysToScalars(recursive=True, transient_only=False).apply_pass(sdfg, {})
    post = type(sdfg.arrays["A"]).__name__
    assert post == "Array", f"ONE-marked array got scalarised: post={post!r}"


def test_convert_length_one_arrays_scalarises_literal_one():
    """Sanity check: an array with literal shape ``(1,)`` IS still
    scalarised (the firewall only skips ``ONE``-marked dims)."""
    sdfg = dace.SDFG("literal_one_fixture")
    sdfg.add_array("B", (1, ), dace.float64, transient=True)
    assert type(sdfg.arrays["B"]).__name__ == "Array"
    ConvertLengthOneArraysToScalars(recursive=True, transient_only=False).apply_pass(sdfg, {})
    assert type(sdfg.arrays["B"]).__name__ == "Scalar", \
        "literal-1 array should still scalarise"


def test_sdfg_roundtrip_with_one_marked_shape():
    """Save -> load -> save: ``ONE`` survives serialisation."""
    sdfg = dace.SDFG("rt_one")
    M = dace.symbol("M")
    K = dace.symbol("K")
    sdfg.add_array("A", (M, ONE, K), dace.float64, transient=False)
    sdfg.add_state("s")
    json1 = sdfg.to_json()
    rebuilt = dace.SDFG.from_json(json1)
    json2 = rebuilt.to_json()
    assert json1 == json2, "SDFG with ONE-marked shape did not roundtrip"


def test_one_emitted_as_constexpr_at_compile():
    """End-to-end: an SDFG with ``ONE`` in a transient shape compiles and the
    generated CPP carries ``constexpr int ONE = 1;`` at file scope.

    Codegen is NOT modified -- whichever transformation introduces ``ONE``
    into a shape also registers it via ``sdfg.add_constant``; DaCe's existing
    ``generate_constants`` does the rest.
    """
    import numpy as np

    N = dace.symbol("N")

    @dace.program
    def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:
            c[i] = a[i] + b[i]

    sdfg = axpy.to_sdfg(simplify=True)
    sdfg.add_array("idx_tile_broadcast", (8, ONE), dace.int64, transient=True, storage=dace.dtypes.StorageType.Register)
    sdfg.add_constant("ONE", 1, dace.data.Scalar(dace.int32))
    compiled = sdfg.compile()
    n_val = 8
    a = np.random.random(n_val)
    b = np.random.random(n_val)
    c = np.zeros(n_val)
    compiled(a=a, b=b, c=c, N=n_val)
    np.testing.assert_allclose(c, a + b, rtol=1e-12)

    import glob
    found = False
    for cpp in glob.glob(f"{sdfg.build_folder}/src/cpu/*.cpp"):
        with open(cpp) as f:
            content = f.read()
        if "constexpr" in content and "ONE = 1" in content:
            found = True
            break
    assert found, "expected 'constexpr int ONE = 1;' in generated CPP"
