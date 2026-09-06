# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF picks the library-node implementation a standalone unit can compile -- not the one named "pure".

The distinction matters twice. It is too NARROW to say "pure": a copy node's ``Auto`` selects one
``std::memcpy`` for a contiguous transfer that runs once and a parallel map past the size where the
map earns its overhead, and pinning ``MappedTasklet`` spent an element-wise loop on every copy. It
is too WIDE to say "declares no environment": ``Reduce``'s ``OpenMP`` declares none and lowers onto
``dace::reduce``, ``FindFirst``'s onto ``dace::find_first_index``, and ``Scan``'s only
environment-free spellings are the CUDA ones -- symbols a unit with no include path cannot resolve.

So the selection is an allowlist that has been checked, and these tests hold both edges of it.
"""
import numpy as np
import pytest

import dace
from dace.codegen.cpf import RENDERABLE_IMPLEMENTATIONS, render
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.helper import is_parallel_cpu_transfer_size
from dace.libraries.standard.nodes.copy.select import select_copy_implementation
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

N = 64


@dace.program
def copy_then_scale(a: dace.float64[N], b: dace.float64[N]):
    t = np.copy(a)
    b[:] = t * 2.0


def prepared_sdfg():
    """``copy_then_scale`` with its implicit copies lifted, unsimplified so they survive."""
    sdfg = copy_then_scale.to_sdfg(simplify=False)
    InsertExplicitCopies().apply_pass(sdfg, {})
    return sdfg


def test_a_small_constant_copy_renders_as_one_memcpy():
    sdfg = prepared_sdfg()
    copies = [(node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, CopyLibraryNode)]
    assert copies, 'the fixture is meant to carry copy library nodes for CPF to choose an expansion for'
    # Left at the node's default, so CPF's choice -- not the fixture's -- is what is under test.
    assert all(node.implementation is None for node, _ in copies)
    assert all(select_copy_implementation(node, state) == 'MemcpyCPU' for node, state in copies)

    code = render(sdfg, language='c++').code
    assert 'memcpy' in code
    # The one branch of Auto that would name a runtime template. It needs a GPU_Shared endpoint,
    # which prepare() has already refused, so reaching it here would mean that guard had moved.
    assert 'CopyND' not in code


@pytest.mark.parametrize('elements', [
    dace.symbol('N'),
    dace.symbol('N') * dace.symbol('N'),
    1 << 20,
])
def test_a_large_or_unknown_copy_stays_a_parallel_map(elements):
    """The half that matters at benchmark sizes, and the reason Auto is preferred over a fixed name.

    A single ``memcpy`` is one thread. Past ``parallel_transfer_min_elements`` the mapped form wins
    because it is a PARALLEL map, so the threshold has to route large copies away from the libc
    call -- and a symbolic size, which is every copy in a kernel with a runtime extent, must be
    ASSUMED large rather than treated as unknown-therefore-small.
    """
    assert is_parallel_cpu_transfer_size(elements)


def test_a_small_constant_copy_is_the_only_one_that_takes_the_single_call():
    threshold = int(dace.Config.get('compiler', 'cpu', 'parallel_transfer_min_elements'))
    assert not is_parallel_cpu_transfer_size(threshold - 1)
    assert is_parallel_cpu_transfer_size(threshold)


@pytest.mark.parametrize('implementation', ['OpenMP', 'vectorized', 'auto', 'GPUAuto', 'CUDA', 'PBLAS', 'cutile'])
def test_implementations_that_lower_onto_a_runtime_are_not_selectable(implementation):
    """Named one by one rather than filtered on ``environments``, which reports none for all of these."""
    assert implementation not in RENDERABLE_IMPLEMENTATIONS


if __name__ == '__main__':
    test_a_small_constant_copy_renders_as_one_memcpy()
