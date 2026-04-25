"""Tests for PermuteDimensions transient-aware permutation.

Covers:
- transient with map-zero init: no permute_in/permute_out for that array
- transient with non-zero init: permute lands AFTER the init state
- transient with no full-extent writer: pass raises ValueError
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.layout.permute_dimensions import PermuteDimensions


def _zero_init_sdfg() -> dace.SDFG:
    """SDFG with one transient zero-initialized by a map, then read by the consumer."""
    N = 8
    sdfg = dace.SDFG('zero_init')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_transient('T', [N, N], dace.float64)

    init = sdfg.add_state('init')
    body = sdfg.add_state_after(init, 'body')

    # T[i, j] = 0 over full extent
    init.add_mapped_tasklet(
        name='zero_T',
        map_ranges={'i': f'0:{N}', 'j': f'0:{N}'},
        inputs={},
        code='__out = 0.0',
        outputs={'__out': dace.Memlet.simple('T', 'i, j')},
        external_edges=True,
    )

    # B[i, j] = A[i, j] + T[i, j]
    body.add_mapped_tasklet(
        name='use_T',
        map_ranges={'i': f'0:{N}', 'j': f'0:{N}'},
        inputs={
            '__a': dace.Memlet.simple('A', 'i, j'),
            '__t': dace.Memlet.simple('T', 'i, j'),
        },
        code='__out = __a + __t',
        outputs={'__out': dace.Memlet.simple('B', 'i, j')},
        external_edges=True,
    )
    return sdfg


def _nonzero_init_sdfg() -> dace.SDFG:
    """SDFG where T[i,j] = i + j (non-zero map init), consumed in a later state."""
    N = 8
    sdfg = dace.SDFG('nonzero_init')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_transient('T', [N, N], dace.float64)

    init = sdfg.add_state('init')
    body = sdfg.add_state_after(init, 'body')

    init.add_mapped_tasklet(
        name='ramp_T',
        map_ranges={'i': f'0:{N}', 'j': f'0:{N}'},
        inputs={},
        code='__out = i + j',
        outputs={'__out': dace.Memlet.simple('T', 'i, j')},
        external_edges=True,
    )

    body.add_mapped_tasklet(
        name='use_T',
        map_ranges={'i': f'0:{N}', 'j': f'0:{N}'},
        inputs={
            '__a': dace.Memlet.simple('A', 'i, j'),
            '__t': dace.Memlet.simple('T', 'i, j'),
        },
        code='__out = __a + __t',
        outputs={'__out': dace.Memlet.simple('B', 'i, j')},
        external_edges=True,
    )
    return sdfg


def _has_state(sdfg: dace.SDFG, label_prefix: str) -> bool:
    return any(s.label.startswith(label_prefix) for s in sdfg.states())


def test_transient_zero_init_skips_permute_in_out():
    sdfg = _zero_init_sdfg()
    assert sdfg.arrays['T'].transient

    PermuteDimensions(
        permute_map={'T': [1, 0]},
        add_permute_maps=True,
    ).apply_pass(sdfg=sdfg, pipeline_results={})

    sdfg.validate()
    # No permute_in / permute_out wrapper since the only permuted array is a
    # zero-initialized transient.
    assert not _has_state(sdfg, 'permute_in')
    assert not _has_state(sdfg, 'permute_out')
    # And no post-init permute either: the init map itself rezeros over the
    # permuted iteration domain.
    assert not _has_state(sdfg, 'permute_after_T')


def test_transient_nonzero_init_inserts_permute_after_state():
    sdfg = _nonzero_init_sdfg()
    PermuteDimensions(
        permute_map={'T': [1, 0]},
        add_permute_maps=True,
    ).apply_pass(sdfg=sdfg, pipeline_results={})

    sdfg.validate()
    assert _has_state(sdfg, 'permute_after_T')
    assert not _has_state(sdfg, 'permute_in')
    assert not _has_state(sdfg, 'permute_out')


def test_transient_without_full_extent_writer_raises():
    """A transient with no writer at all (or no full-extent writer) is a hard error."""
    N = 8
    sdfg = dace.SDFG('orphan')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_transient('T', [N, N], dace.float64)

    # T is referenced but never written over its full extent: only a single
    # element ever gets written, then the consumer reads the (mostly
    # uninitialized) array.
    init = sdfg.add_state('init')
    init.add_mapped_tasklet(
        name='partial_T',
        map_ranges={'i': '0:1', 'j': '0:1'},  # writes T[0,0] only
        inputs={},
        code='__out = 1.0',
        outputs={'__out': dace.Memlet.simple('T', 'i, j')},
        external_edges=True,
    )
    body = sdfg.add_state_after(init, 'body')
    body.add_mapped_tasklet(
        name='use_T',
        map_ranges={'i': f'0:{N}', 'j': f'0:{N}'},
        inputs={
            '__a': dace.Memlet.simple('A', 'i, j'),
            '__t': dace.Memlet.simple('T', 'i, j'),
        },
        code='__out = __a + __t',
        outputs={'__out': dace.Memlet.simple('B', 'i, j')},
        external_edges=True,
    )

    with pytest.raises(ValueError, match="full-extent writer"):
        PermuteDimensions(
            permute_map={'T': [1, 0]},
            add_permute_maps=True,
        ).apply_pass(sdfg=sdfg, pipeline_results={})


if __name__ == '__main__':
    test_transient_zero_init_skips_permute_in_out()
    test_transient_nonzero_init_inserts_permute_after_state()
    test_transient_without_full_extent_writer_raises()
    print('all transient permutation tests passed')
