import dace

from dace.transformation.passes.map_over_free_tasklet import MapOverFreeTasklet

def _add_chain(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    name="ch_1",
    length=2,
    add_scalar=False,
    first_has_inputs=False,
):
    for i in range(length):
        if add_scalar:
            sdfg.add_scalar(
                name=name + f"_scalar_{i}", dtype=dace.dtypes.float32, transient=True
            )
        else:
            sdfg.add_array(
                name=name + f"_array_{i}",
                shape=(2,),
                dtype=dace.dtypes.float32,
                transient=True,
            )

    chain_elements = []
    for i in range(length):
        task = state.add_tasklet(
            name=name + f"_tasklet_{i}",
            inputs={} if i == 0 and not first_has_inputs else {"_in"},
            outputs={"_out"},
            code="_out = _in" if i != 0 or first_has_inputs else "_out = 1.0",
        )
        an = state.add_access(
            name + f"_scalar_{i}" if add_scalar else name + f"_array_{i}"
        )
        chain_elements.append(task)
        chain_elements.append(an)

    for i in range(1, len(chain_elements), 2):
        src = chain_elements[i - 1]
        an = chain_elements[i]
        access_name = (
            name + f"_scalar_{i//2}" if add_scalar else name + f"_array_{i//2}[1]"
        )
        assert isinstance(an, dace.nodes.AccessNode) and isinstance(
            src, dace.nodes.Tasklet
        )
        state.add_edge(src, "_out", an, None, dace.memlet.Memlet(access_name))
        if i < len(chain_elements) - 1:
            dst = chain_elements[i + 1]
            state.add_edge(an, None, dst, "_in", dace.memlet.Memlet(access_name))

    return chain_elements

def _trivial_chain_sdfg():
    sdfg = dace.SDFG("main")
    state = sdfg.add_state("_s")
    _add_chain(sdfg, state, name="ch_1", length=2, add_scalar=True)

    sdfg.validate()
    return sdfg


def _two_trivial_chains_sdfg():
    sdfg = dace.SDFG("main")
    state = sdfg.add_state("_s")
    _add_chain(sdfg, state, name="ch_1", length=2, add_scalar=True)
    _add_chain(sdfg, state, name="ch_2", length=4, add_scalar=False)

    sdfg.validate()
    return sdfg


def _multiple_input_chain_sdfg():
    sdfg = dace.SDFG("main")
    state = sdfg.add_state("_s")
    ft = _add_chain(
        sdfg, state, name="ch_1", length=2, add_scalar=True, first_has_inputs=True
    )
    _add_chain(sdfg, state, name="ch_2", length=4, add_scalar=True)

    # First tasklet ch_1_tasklet_0, second first tasklet ch_2_tasklet_0
    tmps = []
    for i in ["0", "1", "2"]:
        sdfg.add_array(f"tmp{i}", shape=(1,), dtype=dace.float32, transient=True)
        node = state.add_access(f"tmp{i}")
        node.setzero = True
        tmps.append(node)

    doublet = state.add_tasklet(
        name="double",
        inputs={"_in1", "_in2"},
        outputs={"_out"},
        code="_out = _in1 * _in2",
    )

    state.add_edge(tmps[0], None, doublet, "_in1", dace.Memlet("tmp0"))
    state.add_edge(tmps[1], None, doublet, "_in2", dace.Memlet("tmp1"))
    state.add_edge(doublet, "_out", tmps[2], None, dace.Memlet("tmp2"))
    state.add_edge(tmps[2], None, ft[0], "_in", dace.Memlet("tmp2"))

    sdfg.validate()
    return sdfg


def _complex_chain_sdfg():
    sdfg = dace.SDFG("main")
    state = sdfg.add_state("_s")
    ft = _add_chain(
        sdfg, state, name="ch_1", length=2, add_scalar=True
    )
    _add_chain(sdfg, state, name="ch_2", length=4, add_scalar=True)

    # First tasklet ch_1_tasklet_0, second first tasklet ch_2_tasklet_0
    tmps = []
    for i in ["0", "1"]:
        sdfg.add_array(f"tmp{i}", shape=(1,), dtype=dace.float32, transient=True)
        node = state.add_access(f"tmp{i}")
        node.setzero = True
        tmps.append(node)

    doublet = state.add_tasklet(
        name="double",
        inputs={"_in1", "_in2"},
        outputs={"_out"},
        code="_out = _in1 * _in2",
    )

    state.add_edge(tmps[0], None, doublet, "_in1", dace.Memlet("tmp0[0]"))
    state.add_edge(ft[-1], None, doublet, "_in2", dace.Memlet(f"ch_1_scalar_{1}"))
    state.add_edge(doublet, "_out", tmps[1], None, dace.Memlet("tmp1[0]"))

    sdfg.validate()
    return sdfg


def _check_is_in_scope(state, sd, node):
    in_scope = (
        isinstance(node, dace.nodes.EntryNode) or
        isinstance(node, dace.nodes.ExitNode) or
        sd[node] is not None
    )
    if not in_scope:
        srcs = [isinstance(e.src, dace.nodes.ExitNode) for e in state.in_edges(node)]
        dsts = [isinstance(e.dst, dace.nodes.EntryNode)  for e in state.out_edges(node)]
        return any(srcs) or any(dsts)
    return True

def test_trivial_chain():
    sdfg = _trivial_chain_sdfg()
    mapOverFreeTasklet = MapOverFreeTasklet()
    mapOverFreeTasklet.apply_pass(sdfg, {})
    for state in sdfg.states():
        sd = state.scope_dict()
        for node in state.nodes():
            assert( _check_is_in_scope(state, sd, node))
    sdfg.validate()

def test_two_trivial_chains_sdfg():
    sdfg = _two_trivial_chains_sdfg()
    mapOverFreeTasklet = MapOverFreeTasklet()
    mapOverFreeTasklet.apply_pass(sdfg, {})
    for state in sdfg.states():
        sd = state.scope_dict()
        for node in state.nodes():
            assert( _check_is_in_scope(state, sd, node))
    sdfg.validate()

def test_multiple_input_chain_sdfg():
    sdfg = _multiple_input_chain_sdfg()
    mapOverFreeTasklet = MapOverFreeTasklet()
    mapOverFreeTasklet.apply_pass(sdfg, {})
    for state in sdfg.states():
        sd = state.scope_dict()
        for node in state.nodes():
            assert( _check_is_in_scope(state, sd, node))
    sdfg.validate()

def test_complex_chain_sdfg():
    sdfg = _complex_chain_sdfg()
    mapOverFreeTasklet = MapOverFreeTasklet()
    mapOverFreeTasklet.apply_pass(sdfg, {})
    for state in sdfg.states():
        sd = state.scope_dict()
        for node in state.nodes():
            assert( _check_is_in_scope(state, sd, node))
    sdfg.validate()

if __name__ == "__main__":
    test_trivial_chain()
    test_two_trivial_chains_sdfg()
    test_multiple_input_chain_sdfg()
    test_complex_chain_sdfg()
