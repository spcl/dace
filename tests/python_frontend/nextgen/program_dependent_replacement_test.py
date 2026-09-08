# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for PROGRAM-DEPENDENT registry replacements
(``op_repository.program_dependent``): implementations that read or rewrite the
SDFG built so far, rather than only the containers named in their own call.

The next-generation frontend normally decides whether a replacement can be
deferred to a ``ReplacementCallNode`` by TRIAL-RUNNING it on a scratch SDFG
holding just that call's data arguments. A program-dependent replacement can
only ever fail that trial -- the scratch carries none of the program's history
-- so the mark exempts it and the registry's descriptor inference types the
result instead.

The autodiff family (``dace/frontend/python/replacements/torch_autodiff.py``)
is the motivating case: ``torch.autograd.backward`` runs a dependency analysis
over everything parsed up to its call site, ``x.requires_grad_()`` changes what
``x`` IS, and ``x.grad`` names a buffer the ``backward`` expansion created.
Those tests need PyTorch; the family registered here reproduces the same three
shapes with no optional dependency, over the very same dependency analysis.
"""
import contextlib
import copy
import sys
import types

import numpy as np

import dace
from dace import properties
from dace.autodiff.analysis import dependency_analysis
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python import nextgen
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import from_schedule_tree


@properties.make_properties
class TrackedArray(dace.data.Array):
    """An array with a companion snapshot buffer, mirroring the shape of
    :class:`dace.data.ml.ParameterArray` and its gradient buffer."""
    buffer = properties.Property(dtype=str, desc='The companion buffer', default=None, allow_none=True)


def _make_tracked(sdfg: dace.SDFG, name: str) -> None:
    """Convert an array into a :class:`TrackedArray` in place."""
    descriptor = sdfg.arrays[name]
    if isinstance(descriptor, TrackedArray):
        return
    tracked = copy.deepcopy(descriptor)
    tracked.__class__ = TrackedArray
    tracked.buffer = None
    sdfg.arrays[name] = tracked


def _add_buffer(sdfg: dace.SDFG, name: str) -> str:
    """Find or create the companion buffer of a tracked array."""
    descriptor: TrackedArray = sdfg.arrays[name]
    if descriptor.buffer:
        return descriptor.buffer
    buffer_descriptor = copy.deepcopy(descriptor)
    buffer_descriptor.__class__ = dace.data.Array
    buffer_descriptor.transient = True
    descriptor.buffer = sdfg.add_datadesc('snapshot_' + name, buffer_descriptor, find_new_name=True)
    return descriptor.buffer


# A module rather than a namespace object, so the call site spells a qualified
# name the registry is keyed on the way ``torch.autograd.backward`` is.
trackops = types.ModuleType('nextgen_test_trackops')
trackops.snapshot = lambda tensor: None
sys.modules['nextgen_test_trackops'] = trackops


def _track(pv, sdfg: dace.SDFG, state: dace.SDFGState, self: str) -> None:
    _make_tracked(sdfg, self)


def _infer_track(input_desc, **_kwargs):
    return ()


def _infer_track_self(self_desc, **_kwargs):
    result = copy.deepcopy(self_desc)
    result.__class__ = TrackedArray
    result.buffer = None
    # Deliberately wrong about storage class, exactly as the autodiff entry is:
    # the frontend must keep the container's own ``transient`` flag, or a
    # tracked ARGUMENT stops being an argument.
    result.transient = True
    return result


@oprepo.program_dependent
def _snapshot(pv, sdfg: dace.SDFG, state: dace.SDFGState, tensor: str) -> None:
    """
    Copy ``tensor`` into the companion buffer of every tracked array it was
    computed from.

    Program-dependent through and through: the dependency analysis reads the
    dataflow parsed so far, and on a scratch SDFG holding only ``tensor``
    itself it does not even have an entry for it (which is exactly how
    ``torch.autograd.backward`` fails a trial run).
    """
    for source in dependency_analysis(sdfg)[tensor]:
        if not isinstance(sdfg.arrays[source], TrackedArray):
            continue
        buffer = _add_buffer(sdfg, source)
        state.add_nedge(state.add_read(tensor), state.add_write(buffer), Memlet.from_array(tensor, sdfg.arrays[tensor]))


def _infer_snapshot(input_descs, tensor, **_kwargs):
    return ()


@oprepo.program_dependent
def _snapshot_attribute(pv, sdfg: dace.SDFG, state: dace.SDFGState, arr: str) -> str:
    """The companion buffer's name -- which exists only once ``_snapshot`` has
    run, so no scratch trial can produce it."""
    return sdfg.arrays[arr].buffer


def _infer_snapshot_attribute(self_desc):
    result = copy.deepcopy(self_desc)
    result.__class__ = dace.data.Array
    result.transient = True
    return result


@contextlib.contextmanager
def _registered():
    """Install the family for the duration of one test. The registry is global,
    and a leftover entry shows up as an unpaired registration in
    ``schedule_tree/registry_parity_test.py``."""
    oprepo.replaces_method('Array', 'track_')(_track)
    oprepo.infers_method_descriptor('Array', 'track_')(_infer_track)
    oprepo.infers_method_self_descriptor('Array', 'track_')(_infer_track_self)
    oprepo.replaces('nextgen_test_trackops.snapshot')(_snapshot)
    oprepo.infers_descriptor('nextgen_test_trackops.snapshot')(_infer_snapshot)
    oprepo.replaces_attribute('TrackedArray', 'snapshot')(_snapshot_attribute)
    oprepo.infers_attribute_descriptor('TrackedArray', 'snapshot')(_infer_snapshot_attribute)
    try:
        yield
    finally:
        del oprepo.Replacements._method_rep[('Array', 'track_')]
        del oprepo.Replacements._dtype_method_rep[('Array', 'track_')]
        del oprepo.Replacements._dtype_method_self_rep[('Array', 'track_')]
        del oprepo.Replacements._rep['nextgen_test_trackops.snapshot']
        del oprepo.Replacements._dtype_rep['nextgen_test_trackops.snapshot']
        del oprepo.Replacements._attr_rep[('TrackedArray', 'snapshot')]
        del oprepo.Replacements._dtype_attr_rep[('TrackedArray', 'snapshot')]


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_program_dependent_call_defers_instead_of_falling_back():
    """The whole family lowers with no interpreter fallback, even though every
    one of its trial runs on a scratch SDFG raises."""

    @dace.program
    def snapshot_program(x: dace.float64[6]):
        x.track_()
        y = x + 1.0
        trackops.snapshot(y)
        return x.snapshot

    with _registered():
        tree = nextgen.parse_program(snapshot_program, np.zeros(6))

    reasons = [node.reason for node in _nodes_of_type(tree, tn.PythonCallbackNode)]
    assert not reasons, f'unexpected interpreter fallbacks: {reasons}'
    qualnames = [node.qualname for node in _nodes_of_type(tree, tn.ReplacementCallNode)]
    assert qualnames == ['track_', 'nextgen_test_trackops.snapshot', 'TrackedArray.@snapshot']


def test_program_dependent_family_executes():
    """And the deferred expansions produce the right values: the attribute
    resolves to the buffer the earlier call created."""

    @dace.program
    def snapshot_program(x: dace.float64[6]):
        x.track_()
        y = x + 1.0
        trackops.snapshot(y)
        return x.snapshot

    with _registered():
        tree = nextgen.parse_program(snapshot_program, np.zeros(6))
        sdfg = from_schedule_tree(tree)
        sdfg.validate()
        x = np.random.rand(6)
        assert np.allclose(sdfg(x=x.copy()), x + 1.0)


def test_self_descriptor_side_effect_retypes_the_receiver():
    """``x.track_()`` changes what ``x`` IS. Without applying the registry's
    ``infers_method_self_descriptor`` entry to the container repository, the
    following ``x.snapshot`` has no attribute entry to resolve through and the
    program degrades to a callback."""

    @dace.program
    def snapshot_program(x: dace.float64[6]):
        x.track_()
        y = x + 1.0
        trackops.snapshot(y)
        return x.snapshot

    with _registered():
        tree = nextgen.parse_program(snapshot_program, np.zeros(6))

    assert isinstance(tree.containers['x'], TrackedArray)
    # An argument that gains a companion buffer is still an argument: the
    # inference entry's ``transient = True`` must not reach the repository.
    assert 'x' in tree.arg_names
    assert not tree.containers['x'].transient


def test_untracked_receiver_still_falls_back():
    """The exemption is scoped to the marked implementation: the same attribute
    read on a plain array has no registry entry for its class, so it degrades to
    a callback rather than being deferred on faith."""

    @dace.program
    def untracked(x: dace.float64[6]):
        return x.snapshot

    with _registered():
        tree = nextgen.parse_program(untracked, np.zeros(6))

    assert _nodes_of_type(tree, tn.PythonCallbackNode)
    assert not _nodes_of_type(tree, tn.ReplacementCallNode)


def test_program_dependent_marker_is_off_by_default():
    """An unmarked replacement stays subject to the trial -- the conservative
    answer that keeps a build-time callback from becoming a hard error at
    SDFG-construction time."""

    def _plain(pv, sdfg, state, arr):
        return arr

    assert not oprepo.is_program_dependent(_plain)
    assert not oprepo.is_program_dependent(None)
    assert oprepo.is_program_dependent(_snapshot)
    with _registered():
        assert 'snapshot' in oprepo.Replacements.program_dependent_attributes()
    assert 'snapshot' not in oprepo.Replacements.program_dependent_attributes()


if __name__ == '__main__':
    test_program_dependent_call_defers_instead_of_falling_back()
    test_program_dependent_family_executes()
    test_self_descriptor_side_effect_retypes_the_receiver()
    test_untracked_receiver_still_falls_back()
    test_program_dependent_marker_is_off_by_default()
