# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the GF(2)-affine carried-state analysis.

The oracle in every test is the BODY ITSELF, executed: a candidate matrix is only correct if
applying it reproduces what running the Python source does. Comparing against a hand-derived matrix
would test the derivation, not the analysis.
"""
import pytest

from dace.transformation.passes.analysis import carried_state, smt_body

pytestmark = pytest.mark.skipif(not smt_body.HAS_Z3, reason='needs z3')

CRC16_STEP = """
if crc & 0x8000:
    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
else:
    crc = (crc << 1) & 0xFFFF
"""

#: CRC-16/CCITT over one byte, as it looks after ShortLoopUnroll flattens the bit loop.
CRC16_BODY = 'crc = crc ^ ((byte << 8) & 0xFFFF)\n' + CRC16_STEP * 8


def crc16_reference(crc: int, byte: int) -> int:
    """The same body, run."""
    crc = crc ^ ((byte << 8) & 0xFFFF)
    for _ in range(8):
        crc = (((crc << 1) ^ 0x1021) & 0xFFFF) if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


def test_crc16_linear_part_is_extracted_and_data_free():
    """CRC is the shape the lift is for: the feedback network is fixed, only the injected byte
    varies, so the linear part exists and does not read ``byte``."""
    linear = carried_state.extract_gf2_linear(CRC16_BODY, 'crc', ['byte'], state_bits=16)
    assert linear is not None, 'CRC-16 must be recognized as GF(2)-affine with a data-free A'
    assert linear.state_bits == 16
    assert len(linear.columns) == 16
    assert all(0 <= c < (1 << 16) for c in linear.columns), 'columns must stay inside the state width'


def test_crc16_matrix_reproduces_the_body_on_every_basis_state():
    """``state' == A * state XOR c`` numerically, against the executed body -- 16 basis states plus
    a spread of arbitrary ones, at several data values."""
    linear = carried_state.extract_gf2_linear(CRC16_BODY, 'crc', ['byte'], state_bits=16)
    assert linear is not None
    for byte in (0x00, 0x01, 0x5A, 0xFF):
        const = crc16_reference(0, byte)
        states = [1 << j for j in range(16)] + [0, 0xFFFF, 0x1234, 0xBEEF, 0x8000, 0x7FFF]
        for state in states:
            assert linear.apply(state) ^ const == crc16_reference(state, byte), \
                f'A*x XOR c disagrees with the body at crc={state:#06x} byte={byte:#04x}'


def test_block_matrix_folds_a_run_of_iterations():
    """The claim the whole lift rests on: ``A^k`` plus the block's own constant replaces running
    ``k`` iterations. ``c_block`` is obtained the way a parallel block would obtain it -- by running
    the block from the ZERO state, at sequential per-element cost."""
    linear = carried_state.extract_gf2_linear(CRC16_BODY, 'crc', ['byte'], state_bits=16)
    assert linear is not None

    data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39]
    block = linear.power(len(data))

    const_block = 0
    for byte in data:
        const_block = crc16_reference(const_block, byte)

    for seed in (0x0000, 0xFFFF, 0x1D0F, 0xABCD):
        sequential = seed
        for byte in data:
            sequential = crc16_reference(sequential, byte)
        assert block.apply(seed) ^ const_block == sequential, \
            f'A^{len(data)} * seed XOR c_block != the sequential run at seed={seed:#06x}'


def test_data_dependent_linear_part_is_refused():
    """A body whose SHIFT depends on the datum has a per-element matrix. Composing those costs a
    matrix build per element, so the lift loses; the analysis must not offer it."""
    body = """
if d & 1:
    s = (s << 1) & 0xFFFF
else:
    s = s & 0xFFFF
"""
    assert carried_state.extract_gf2_linear(body, 's', ['d'], state_bits=16) is None


def test_non_linear_body_is_refused():
    """``s * s`` is not linear over GF(2) -- and it agrees with the zero map on every basis vector,
    so extraction by evaluation alone would have accepted it. The admissibility proof is what
    rejects it."""
    body = 's = (s * s) & 0xFFFF\n'
    assert carried_state.extract_gf2_linear(body, 's', ['d'], state_bits=16) is None


def test_body_outside_the_fragment_is_refused():
    """A loop is not straight-line and a subscript is not a scalar; the encoder refuses both, so the
    analysis returns None rather than guessing."""
    assert carried_state.extract_gf2_linear('for k in range(8):\n    s = s ^ 1\n', 's', ['d'], 16) is None
    assert carried_state.extract_gf2_linear('s = s ^ table[d]\n', 's', ['d'], 16) is None


def test_pure_xor_of_data_is_the_identity_map():
    """``s ^= d`` is affine with ``A = I``: the state passes through untouched and the datum is all
    of ``c``. Pins that a data-free A is not confused with a trivial one."""
    linear = carried_state.extract_gf2_linear('s = s ^ d\n', 's', ['d'], state_bits=8)
    assert linear == carried_state.identity(8)


def test_is_constant_separates_a_constant_term_from_a_datum():
    """``d & 0`` is the constant 0 and ``d`` is not constant. Substituting zero for the data would
    call both of them 0, which is why the check ends in a proof."""
    d = smt_body.bitvec('d', 32)
    assert carried_state.is_constant(d & smt_body.constant(0, 32), [d]) == 0
    assert carried_state.is_constant(d, [d]) is None
    assert carried_state.is_constant(smt_body.constant(7, 32), [d]) == 7


def test_power_matches_repeated_composition():
    """Repeated squaring against the naive fold, on a map that is NOT symmetric so a swapped
    composition order would show up."""
    linear = carried_state.extract_gf2_linear(CRC16_BODY, 'crc', ['byte'], state_bits=16)
    assert linear is not None
    naive = carried_state.identity(16)
    for k in range(1, 12):
        naive = naive.then(linear)
        assert linear.power(k) == naive, f'power({k}) disagrees with {k} compositions'
    assert linear.power(0) == carried_state.identity(16)


def test_power_rejects_a_negative_exponent():
    with pytest.raises(ValueError):
        carried_state.identity(8).power(-1)
