# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that the next-generation frontend never *copies* the live Python objects
shared preprocessing embeds into ``ast.Constant`` nodes (resolved programs,
``__sdfg__`` objects, arbitrary closure constants). The frontend rewrites ASTs
in place and therefore copies before rewriting, but those copies must be
structural (:func:`~dace.frontend.python.astutils.copy_tree`):
copying the embedded objects is wasteful, breaks the identity the parse cache
and callee resolution rely on, and outright fails for objects that refuse to
be copied -- a compiled program's ``ReloadableDLL`` among them.
"""
import ast

import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.frontend.python import astutils
from dace.sdfg.analysis.schedule_tree import treenodes as tn


class Uncopyable:
    """An object embedded in the closure that refuses to be copied, standing
    in for the compiled artifacts (``ReloadableDLL``, ``DaceModule``) real
    programs put there."""

    def __deepcopy__(self, memo):
        raise AssertionError('the frontend copied an embedded object')

    def __sdfg__(self, *args, **kwargs):

        @dace.program
        def doubler(a: dace.float64[20]):
            return a * 2

        return doubler.to_sdfg()

    def __sdfg_closure__(self, reevaluate=None):
        return {}

    def __sdfg_signature__(self):
        return [['a'], []]


def test_copy_tree_shares_embedded_objects():
    embedded = Uncopyable()
    original = ast.parse('x = f(y)').body
    original[0].value.func = ast.Constant(value=embedded)

    clone = astutils.copy_tree(original)

    assert clone is not original
    assert clone[0] is not original[0]
    assert clone[0].value.func.value is embedded


def test_copy_tree_keeps_constant_annotations():
    """Preprocessing marks the constants it embeds with the qualified name the
    object came from, and callee resolution reads it back off the copy."""
    embedded = Uncopyable()
    node = astutils.create_constant(embedded)
    node.qualname = 'module.embedded'

    clone = astutils.copy_tree(ast.Expr(value=node))

    assert clone.value is not node
    assert clone.value.value is embedded
    assert clone.value.qualname == 'module.embedded'


def test_copy_tree_leaves_the_original_alone():
    original = ast.parse('a[i] = b').body[0]

    clone = astutils.copy_tree(original)

    assert clone is not original
    assert clone.targets is not original.targets
    assert clone.lineno == original.lineno
    assert ast.dump(clone) == ast.dump(original)


def test_copy_tree_copies_hidden_marker_payloads():
    """Canonical markers hide their payload from ``ast.walk`` (``_fields =
    ()``). A copy that shared it would still be an alias, and the callback
    restoration path rewrites what it is handed -- corrupting the cached
    canonical body of every other call site."""
    from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt
    inner = ast.parse('x = f(y)').body[0]
    marker = OpaqueStmt(original=inner, reason='test')

    clone = astutils.copy_tree(marker)

    assert clone is not marker
    assert clone.original is not marker.original
    assert clone.originals[0] is not marker.originals[0]
    assert ast.dump(clone.original) == ast.dump(marker.original)


def test_uncopyable_callee_object():
    """An embedded ``__sdfg__`` object survives canonicalization, which copies
    the statements it appears in (augmented-assignment desugaring, ANF)."""
    callee = Uncopyable()

    @dace.program
    def caller(a: dace.float64[20]):
        a += callee(a)

    tree = nextgen.parse_program(caller)
    assert isinstance(tree, tn.ScheduleTreeRoot)


def test_uncopyable_compiled_callee():
    """The classic frontend's parse of an already-compiled callee attaches a
    ``ReloadableDLL`` to the embedded program object."""

    @dace.program
    def callee(a: dace.float64[20]):
        return a * 2

    callee(np.random.rand(20))  # Compile, so the program object carries a DLL

    @dace.program
    def caller(a: dace.float64[20]):
        return callee(a) + 1

    tree = nextgen.parse_program(caller)
    assert not [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_uncopyable_constant_in_callback():
    """A statement the frontend cannot lower is restored into a callback,
    which copies it -- embedded objects included."""

    class Opaque:

        def __deepcopy__(self, memo):
            raise AssertionError('the frontend copied an embedded object')

        def value(self):
            return 4.0

    opaque = Opaque()

    @dace.program
    def uses_opaque(a: dace.float64[20]):
        a[0] = opaque.value()

    tree = nextgen.parse_program(uses_opaque)
    assert isinstance(tree, tn.ScheduleTreeRoot)


if __name__ == '__main__':
    test_copy_tree_shares_embedded_objects()
    test_copy_tree_keeps_constant_annotations()
    test_copy_tree_leaves_the_original_alone()
    test_copy_tree_copies_hidden_marker_payloads()
    test_uncopyable_callee_object()
    test_uncopyable_compiled_callee()
    test_uncopyable_constant_in_callback()
