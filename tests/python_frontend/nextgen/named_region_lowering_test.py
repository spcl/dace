# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``with dace.named("label"):`` in the next-generation Python frontend.

A named region groups statements under a label for readability, profiling and
transformation targeting without changing what they mean. Canonicalization
recognizes it where it reads ``with`` statements
(``RecognizeExplicitDataflow``) but treats the body as ordinary program code,
and lowering emits a
:class:`~dace.sdfg.analysis.schedule_tree.treenodes.NamedRegionScope` that
tree-to-SDFG reproduces as a real :class:`~dace.sdfg.state.NamedRegion`.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import NamedRegion
from dace.transformation.passes.simplify import SimplifyPass


def _regions(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.NamedRegionScope)]


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_named_region_with_a_label():
    """The label passed to ``dace.named`` reaches the scope and the SDFG."""

    @dace.program
    def labeled(A: dace.float64[1]):
        with dace.named("my named region"):
            A[0] = 20.0

    A = np.zeros(1)
    tree = nextgen.parse_program(labeled, A)
    assert not _callbacks(tree)
    assert [region.label for region in _regions(tree)] == ['my named region']

    sdfg = tree.as_sdfg(simplify=False)
    SimplifyPass(no_inline_named_regions=True).apply_pass(sdfg, {})
    assert [region.label for region in sdfg.nodes() if isinstance(region, NamedRegion)] == ['my named region']

    sdfg.compile()(A=A)
    assert A[0] == 20.0


def test_unnamed_region_gets_a_generated_label():
    """Bare ``with dace.named:`` evaluates to the class, which carries no name,
    so the label is built from the line number -- as the classic frontend
    builds it (``newast.py::visit_With``)."""

    @dace.program
    def unlabeled(A: dace.float64[1]):
        with dace.named:
            A[0] = 20.0

    A = np.zeros(1)
    tree = nextgen.parse_program(unlabeled, A)
    assert not _callbacks(tree)
    labels = [region.label for region in _regions(tree)]
    assert len(labels) == 1 and labels[0].startswith('Named Region ')

    tree.as_sdfg().compile()(A=A)
    assert A[0] == 20.0


def test_nested_named_regions():
    """Regions nest, and the nesting survives into the SDFG."""

    @dace.program
    def three_deep(A: dace.float64[2]):
        with dace.named("outer region"):
            with dace.named("middle region"):
                with dace.named("inner region"):
                    A[0] = 1.0
            A[1] = 2.0

    A = np.zeros(2)
    tree = nextgen.parse_program(three_deep, A)
    assert not _callbacks(tree)
    assert [region.label for region in _regions(tree)] == ['outer region', 'middle region', 'inner region']

    sdfg = tree.as_sdfg(simplify=False)
    SimplifyPass(no_inline_named_regions=True).apply_pass(sdfg, {})
    outer = next(region for region in sdfg.nodes() if isinstance(region, NamedRegion))
    middle = next(region for region in outer.nodes() if isinstance(region, NamedRegion))
    inner = next(region for region in middle.nodes() if isinstance(region, NamedRegion))
    assert (outer.label, middle.label, inner.label) == ('outer region', 'middle region', 'inner region')

    sdfg.compile()(A=A)
    assert np.allclose(A, [1.0, 2.0])


def test_a_region_introduces_no_binding_scope():
    """Python's ``with`` introduces no scope, so neither does the region: a
    name assigned inside is still bound after it."""

    @dace.program
    def carries(A: dace.float64[4], O: dace.float64[4]):
        with dace.named("phase 1"):
            tmp = A + 1.0
        O[:] = tmp * 2.0

    A, O = np.arange(4, dtype=np.float64), np.zeros(4)
    tree = nextgen.parse_program(carries, A, O)
    assert not _callbacks(tree)
    assert len(_regions(tree)) == 1

    tree.as_sdfg().compile()(A=A, O=O)
    assert np.allclose(O, (A + 1.0) * 2.0)


def test_a_free_form_label_still_yields_a_valid_sdfg():
    """Inlining a region prefixes its label onto every block name inside, and
    those names are validated as identifiers -- so a label containing anything
    but identifier characters used to produce an SDFG that failed validation
    when called (``dace.dtypes.sanitize_name``). Reachable from the classic
    frontend too."""

    @dace.program
    def punctuated(A: dace.float64[1]):
        with dace.named("phase 1: warm-up!"):
            A[0] = 7.0

    A = np.zeros(1)
    tree = nextgen.parse_program(punctuated, A)
    assert [region.label for region in _regions(tree)] == ['phase 1: warm-up!']

    sdfg = tree.as_sdfg()  # Simplify inlines the region, folding the label into block names
    sdfg.validate()
    sdfg.compile()(A=A)
    assert A[0] == 7.0


def test_region_survives_a_tree_roundtrip():
    """An SDFG-derived tree keeps the region too, so the label is not lost by
    passing through the schedule tree (``sdfg_to_tree``)."""

    @dace.program
    def labeled(A: dace.float64[1]):
        with dace.named("round trip"):
            A[0] = 20.0

    sdfg = labeled.to_sdfg(simplify=False)
    SimplifyPass(no_inline_named_regions=True).apply_pass(sdfg, {})
    tree = sdfg.as_schedule_tree()
    assert [region.label for region in _regions(tree)] == ['round trip']


if __name__ == '__main__':
    test_named_region_with_a_label()
    test_unnamed_region_gets_a_generated_label()
    test_nested_named_regions()
    test_a_region_introduces_no_binding_scope()
    test_a_free_form_label_still_yields_a_valid_sdfg()
    test_region_survives_a_tree_roundtrip()
