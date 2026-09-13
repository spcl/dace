# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Name minting against tasklet connectors.

Validation rejects a tasklet connector that shares its name with a data descriptor, constant or
symbol ("Connector name '%s' is already used as a symbol, constant, or array name"), so a caller
minting a GENERIC name into a graph whose connectors it did not choose has to dodge them --
``symm`` hit exactly this, with a transformation minting an array named ``tmp`` next to a tasklet
that already had a ``tmp`` connector.

Dodging them costs a walk over every state and every node, which is why it is a separate method
rather than part of the ordinary minter: the ordinary one runs thousands of times in a single
parse, once per temporary, and paying the walk there made parsing quadratic.
"""
import dace
from dace.sdfg import nodes
from dace.transformation.interstate import InlineSDFG


def sdfg_with_a_tmp_connector() -> dace.SDFG:
    """A graph holding a tasklet whose connectors are named ``tmp`` and ``out``."""
    sdfg = dace.SDFG('tmp_connector')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'tmp'}, {'out'}, 'out = tmp + 1.0')
    state.add_edge(state.add_read('A'), None, tasklet, 'tmp', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg


def test_the_connector_avoiding_minter_dodges_a_connector_name():
    sdfg = sdfg_with_a_tmp_connector()
    assert sdfg.find_new_name_avoiding_connectors('tmp') != 'tmp'
    # `out` is a connector too, and free in the data/symbol namespace, so it must also be dodged.
    assert sdfg.find_new_name_avoiding_connectors('out') != 'out'
    # A name that collides with nothing comes back untouched.
    assert sdfg.find_new_name_avoiding_connectors('unrelated') == 'unrelated'


def test_minting_that_name_would_have_produced_an_invalid_sdfg():
    """The reason the dodge exists, asserted rather than described: taking the connector's name
    makes the graph fail validation."""
    sdfg = sdfg_with_a_tmp_connector()
    sdfg.validate()  # the premise: it is valid before

    sdfg.add_scalar('tmp', dace.float64, transient=True)
    try:
        sdfg.validate()
    except Exception as exc:  # noqa: BLE001 -- the message is the assertion
        assert 'tmp' in str(exc)
    else:
        raise AssertionError('an array sharing a tasklet connector name must not validate')


def test_the_connector_avoiding_name_is_safe_to_add():
    sdfg = sdfg_with_a_tmp_connector()
    sdfg.add_scalar(sdfg.find_new_name_avoiding_connectors('tmp'), dace.float64, transient=True)
    sdfg.validate()


def test_inlining_lifts_a_nested_transient_past_an_outer_connector_name() -> None:
    sdfg = sdfg_with_a_tmp_connector()
    inner = dace.SDFG('inner')
    inner.add_array('x', [4], dace.float64)
    inner.add_array('y', [4], dace.float64)
    inner.add_scalar('tmp', dace.float64, transient=True)
    body = inner.add_state()
    copy_in = body.add_tasklet('copy_in', {'a'}, {'b'}, 'b = a')
    copy_out = body.add_tasklet('copy_out', {'c'}, {'d'}, 'd = c')
    body.add_edge(body.add_read('x'), None, copy_in, 'a', dace.Memlet('x[1]'))
    tmp_node = body.add_access('tmp')
    body.add_edge(copy_in, 'b', tmp_node, None, dace.Memlet('tmp'))
    body.add_edge(tmp_node, None, copy_out, 'c', dace.Memlet('tmp'))
    body.add_edge(copy_out, 'd', body.add_write('y'), None, dace.Memlet('y[1]'))
    outer = sdfg.add_state_after(sdfg.start_block)
    nested = outer.add_nested_sdfg(inner, {'x': None}, {'y': None})
    outer.add_edge(outer.add_read('A'), None, nested, 'x', dace.Memlet('A'))
    outer.add_edge(nested, 'y', outer.add_write('B'), None, dace.Memlet('B'))

    InlineSDFG.apply_to(sdfg, nested_sdfg=nested)

    assert not any(isinstance(node, nodes.NestedSDFG) for node in outer.nodes())
    assert 'tmp' not in sdfg.arrays
    assert 'tmp_0' in sdfg.arrays
    sdfg.validate()


def test_the_ordinary_minter_stays_out_of_the_graph():
    """The ordinary minter answers from the data/symbol/constant namespace only. This is what keeps
    it constant-time, and what a future change must not quietly turn back into a graph walk."""
    sdfg = sdfg_with_a_tmp_connector()
    assert sdfg._find_new_name('tmp') == 'tmp'
    assert sdfg._find_new_name('A') != 'A'


if __name__ == '__main__':
    test_the_connector_avoiding_minter_dodges_a_connector_name()
    test_minting_that_name_would_have_produced_an_invalid_sdfg()
    test_the_connector_avoiding_name_is_safe_to_add()
    test_the_ordinary_minter_stays_out_of_the_graph()
