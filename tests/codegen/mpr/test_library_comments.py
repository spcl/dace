# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MPR names the library nodes it rendered away.

A pure expansion turns ``Gemm`` into three nested loops, ``Cholesky`` into a triangular sweep, a
``Reduce`` into an accumulation. The loops are correct and unreadable: nothing in them says what
they used to be, and MPR output exists to be read. So MPR records what each library node computes
when it expands it and writes that line above the code the expansion produced.

The comment is not decoration -- it is the only place the rendering states its own intent. These
tests hold three things: that every library node registered in the process has a description (so a
newly added node cannot render as anonymous loops), that the description reaches the emitted text,
and that it appears once per node rather than once per tasklet.
"""
import importlib
import pkgutil

import pytest

import dace
import dace.libraries
from dace.codegen.mpr import (LIBRARY_NODE_DESCRIPTIONS, QUALIFIED_DESCRIPTIONS, description_of, render)
from dace.sdfg.nodes import LibraryNode

M = dace.symbol('M')
N = dace.symbol('N')


def library_node_classes():
    """Every ``LibraryNode`` subclass reachable after importing all of ``dace.libraries``.

    Walked from the base class rather than from a registry so a node that registers itself in some
    other way is still caught -- the question this answers is "what can appear in an SDFG", and the
    answer is every subclass that has been imported.
    """
    for module in pkgutil.walk_packages(dace.libraries.__path__, 'dace.libraries.'):
        try:
            importlib.import_module(module.name)
        except Exception:  # an optional backend whose dependency is not installed
            continue
    found = set()

    def walk(cls):
        for subclass in cls.__subclasses__():
            found.add(subclass)
            walk(subclass)

    walk(LibraryNode)
    return sorted(found, key=lambda cls: (cls.__module__, cls.__name__))


def test_every_library_node_class_has_a_description():
    """A library node with no description renders as loops that say nothing about themselves."""
    undescribed = [f'{cls.__module__}.{cls.__name__}' for cls in library_node_classes() if description_of(cls) is None]
    assert not undescribed, ('these library nodes have no MPR description, so their expansions would render '
                             'anonymously: ' + ', '.join(undescribed) +
                             '\nAdd one to dace.codegen.mpr.LIBRARY_NODE_DESCRIPTIONS.')


def test_description_lookup_prefers_the_qualified_name():
    """Two libraries share the name ``Reduce``, and they do different things.

    A lookup by bare class name would give MPI_Reduce the array-reduction description (or the other
    way round, depending on import order) -- a comment that is confidently wrong is worse than none.
    """
    assert QUALIFIED_DESCRIPTIONS, 'the qualified table is what disambiguates a shared class name'
    for qualified, description in QUALIFIED_DESCRIPTIONS.items():
        module, _, name = qualified.rpartition('.')
        cls = getattr(importlib.import_module(module), name)
        assert description_of(cls) == description, f'{qualified} did not resolve to its qualified description'
        assert name not in LIBRARY_NODE_DESCRIPTIONS, (
            f'{name} also has a bare-name description, so one of the two meanings could be served the '
            "other's text if a lookup ever missed the qualified table")


def test_matmul_rendering_names_the_library_node():
    """The gemm the rendering came from is named in the rendering."""

    @dace.program
    def matmul(a: dace.float64[M, N], b: dace.float64[N, M], c: dace.float64[M, M]):
        c[:] = a @ b

    sdfg = matmul.to_sdfg(simplify=True)
    sdfg.name = 'mpr_comment_matmul'
    code = render(sdfg).code
    assert '// BLAS gemm' in code, ('the matmul rendered without naming the library node it came from:\n' +
                                    '\n'.join(line for line in code.splitlines() if line.strip().startswith('//')))


def test_reduction_rendering_names_the_library_node():
    """A reduction is the case where the loop nest is least self-explanatory."""
    import numpy as np

    @dace.program
    def total(x: dace.float64[N], out: dace.float64[1]):
        out[0] = np.sum(x)

    sdfg = total.to_sdfg(simplify=True)
    sdfg.name = 'mpr_comment_sum'
    code = render(sdfg).code
    assert '// reduction over' in code, 'the reduction rendered without naming the library node it came from'


def test_a_description_is_written_once_per_node_not_once_per_tasklet():
    """One expansion, one comment -- and two nodes of the same kind, two comments.

    Deduplicating on the description TEXT instead of on the node would collapse the second matmul's
    comment into the first, and the reader would see one gemm where the program has two.
    """

    @dace.program
    def two_products(a: dace.float64[M, N], b: dace.float64[N, M], c: dace.float64[M, M], d: dace.float64[M, M]):
        c[:] = a @ b
        d[:] = c @ c

    sdfg = two_products.to_sdfg(simplify=True)
    sdfg.name = 'mpr_comment_two'
    code = render(sdfg).code
    occurrences = code.count('// BLAS gemm')
    assert occurrences == 2, (f'expected one comment per gemm node, got {occurrences}; a count of 1 means the '
                              'emitter deduplicated on the description text rather than on the node')


def test_ordinary_codegen_carries_no_mpr_comments():
    """The provenance map is scoped to a rendering; ordinary code generation must not see it."""

    @dace.program
    def matmul(a: dace.float64[M, N], b: dace.float64[N, M], c: dace.float64[M, M]):
        c[:] = a @ b

    sdfg = matmul.to_sdfg(simplify=True)
    sdfg.name = 'mpr_comment_leak'
    render(sdfg)  # populate and then discard a provenance scope
    with dace.config.set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        ordinary = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    assert '// BLAS gemm' not in ordinary, ('an MPR provenance comment leaked into ordinary code generation; the '
                                            'scope was not restored')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
