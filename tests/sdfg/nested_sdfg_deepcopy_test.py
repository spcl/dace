# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests deepcopying (nested) SDFGs. """
import copy
import dace
import numpy as np


def test_deepcopy_same_state():

    sdfg = dace.SDFG('deepcopy_nested_sdfg')
    state = sdfg.add_state('state')

    nsdfg = dace.SDFG('nested')
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {})

    copy_nsdfg = copy.deepcopy(nsdfg_node)
    assert copy_nsdfg.sdfg.parent_nsdfg_node is copy_nsdfg
    assert copy_nsdfg.sdfg.parent is None
    assert copy_nsdfg.sdfg.parent_sdfg is None

    state.add_node(copy_nsdfg)
    assert copy_nsdfg.sdfg.parent is state
    assert copy_nsdfg.sdfg.parent_sdfg is sdfg


def test_deepcopy_same_state_edge():

    sdfg = dace.SDFG('deepcopy_nested_sdfg')
    state = sdfg.add_state('state')

    nsdfg = dace.SDFG('nested')
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {})

    copy_nsdfg = copy.deepcopy(nsdfg_node)
    assert copy_nsdfg.sdfg.parent_nsdfg_node is copy_nsdfg
    assert copy_nsdfg.sdfg.parent is None
    assert copy_nsdfg.sdfg.parent_sdfg is None

    state.add_edge(nsdfg_node, None, copy_nsdfg, None, dace.Memlet())
    assert copy_nsdfg.sdfg.parent is state
    assert copy_nsdfg.sdfg.parent_sdfg is sdfg


def test_deepcopy_diff_state():

    sdfg = dace.SDFG('deepcopy_nested_sdfg')
    state_0 = sdfg.add_state('state_0')
    state_1 = sdfg.add_state('state_1')

    nsdfg = dace.SDFG('nested')
    nsdfg_node = state_0.add_nested_sdfg(nsdfg, None, {}, {})

    copy_nsdfg = copy.deepcopy(nsdfg_node)
    assert copy_nsdfg.sdfg.parent_nsdfg_node is copy_nsdfg
    assert copy_nsdfg.sdfg.parent is None
    assert copy_nsdfg.sdfg.parent_sdfg is None

    state_1.add_node(copy_nsdfg)
    assert copy_nsdfg.sdfg.parent is state_1
    assert copy_nsdfg.sdfg.parent_sdfg is sdfg


def test_deepcopy_diff_state_edge():

    sdfg = dace.SDFG('deepcopy_nested_sdfg')
    sdfg.add_array('A', [1], dace.int32)
    state_0 = sdfg.add_state('state_0')
    state_1 = sdfg.add_state('state_1')

    nsdfg = dace.SDFG('nested')
    nsdfg_node = state_0.add_nested_sdfg(nsdfg, None, {}, {})

    copy_nsdfg = copy.deepcopy(nsdfg_node)
    assert copy_nsdfg.sdfg.parent_nsdfg_node is copy_nsdfg
    assert copy_nsdfg.sdfg.parent is None
    assert copy_nsdfg.sdfg.parent_sdfg is None

    a = state_1.add_access('A')
    state_1.add_edge(a, None, copy_nsdfg, None, dace.Memlet())
    assert copy_nsdfg.sdfg.parent is state_1
    assert copy_nsdfg.sdfg.parent_sdfg is sdfg


def test_deepcopy_diff_sdfg():

    sdfg_0 = dace.SDFG('deepcopy_nested_sdfg_0')
    state_0 = sdfg_0.add_state('state_0')

    nsdfg = dace.SDFG('nested')
    nsdfg_node = state_0.add_nested_sdfg(nsdfg, None, {}, {})

    copy_nsdfg = copy.deepcopy(nsdfg_node)
    assert copy_nsdfg.sdfg.parent_nsdfg_node is copy_nsdfg
    assert copy_nsdfg.sdfg.parent is None
    assert copy_nsdfg.sdfg.parent_sdfg is None

    sdfg_1 = dace.SDFG('deepcopy_nested_sdfg_1')
    state_1 = sdfg_1.add_state('state_1')

    state_1.add_node(copy_nsdfg)
    assert copy_nsdfg.sdfg.parent is state_1
    assert copy_nsdfg.sdfg.parent_sdfg is sdfg_1


def test_deepcopy_diff_sdfg_edge():

    sdfg_0 = dace.SDFG('deepcopy_nested_sdfg_0')
    state_0 = sdfg_0.add_state('state_0')

    nsdfg = dace.SDFG('nested')
    nsdfg_node = state_0.add_nested_sdfg(nsdfg, None, {}, {})

    copy_nsdfg = copy.deepcopy(nsdfg_node)
    assert copy_nsdfg.sdfg.parent_nsdfg_node is copy_nsdfg
    assert copy_nsdfg.sdfg.parent is None
    assert copy_nsdfg.sdfg.parent_sdfg is None

    sdfg_1 = dace.SDFG('deepcopy_nested_sdfg_1')
    sdfg_1.add_array('A', [1], dace.int32)
    state_1 = sdfg_1.add_state('state_1')

    a = state_1.add_access('A')
    state_1.add_edge(a, None, copy_nsdfg, None, dace.Memlet())
    assert copy_nsdfg.sdfg.parent is state_1
    assert copy_nsdfg.sdfg.parent_sdfg is sdfg_1


if __name__ == '__main__':
    test_deepcopy_same_state()
    test_deepcopy_same_state_edge()
    test_deepcopy_diff_state()
    test_deepcopy_diff_state_edge()
    test_deepcopy_diff_sdfg()
