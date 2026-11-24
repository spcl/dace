# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The C dialect must lower a tasklet's ``z.real()`` / ``z.imag()`` to ``creal`` / ``cimag``.

dace-fortran spells Fortran's ``DBLE(z)`` / ``AIMAG(z)`` as the ``std::complex`` member calls, which the C++ runtime
compiles. CPF's C text kept them, and ``double _Complex`` has no members: QE ``vexx_bp_k_gpu``'s
``hpsi(ig) = hpsi(ig) - exxalfa * DBLE(deexx(ikb)) * vkbp(ig, ikb)`` rendered as C that failed to compile
("request for member 'real' in something not a structure or union").
"""

import dace
from dace.codegen import cpf

N = dace.symbol("N")


def build_sdfg() -> dace.SDFG:
    """``B[i] = 2 * Z[i].real() - Z[i].imag()`` over a map."""
    sdfg = dace.SDFG("complex_member_accessors_c")
    sdfg.add_array("Z", [N], dace.complex128)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "parts",
        {"i": "0:N"},
        {"__in": dace.Memlet("Z[i]")},
        "__out = 2 * __in.real() - __in.imag()",
        {"__out": dace.Memlet("B[i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_member_accessors_lower_to_creal_and_cimag():
    # ``render`` compiles the unit before returning it (``check_compiles``), so the C must build.
    code = cpf.render(build_sdfg(), language="c").code
    assert "creal(" in code and "cimag(" in code, code
    assert ".real()" not in code and ".imag()" not in code, code


if __name__ == "__main__":
    test_member_accessors_lower_to_creal_and_cimag()
