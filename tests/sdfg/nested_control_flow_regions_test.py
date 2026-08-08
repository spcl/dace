# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace


def test_is_start_state_deprecation():
    sdfg = dace.SDFG('deprecation_test')
    with pytest.deprecated_call():
        sdfg.add_state('state1', is_start_state=True)
    sdfg2 = dace.SDFG('deprecation_test2')
    state = dace.SDFGState('state2')
    with pytest.deprecated_call():
        sdfg2.add_node(state, is_start_state=True)


def test_cfg_list_is_rebuilt_on_deserialization():
    """A round-tripped SDFG must keep `cfg_id` consistent with its `cfg_list`.

    `to_json` rebuilds the CFG list before writing, but it is derived state the JSON does not carry.
    Left stale on the way back in, `cfg_id` no longer indexes `cfg_list`, so `PatternNode` lookups
    resolve against the wrong region and `can_be_applied` raises a bare `NodeNotFoundError` -- which
    the pattern matcher swallows as "declined to apply", silently losing every transformation.
    """
    from dace.transformation.interstate import InlineMultistateSDFG

    N = dace.symbol('N')

    @dace.program
    def inner(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0

    @dace.program
    def outer(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        inner(a, b)
        inner(b, c)

    rebuilt = dace.SDFG.from_json(outer.to_sdfg(simplify=False).to_json())

    regions = list(rebuilt.all_control_flow_regions(recursive=True))
    assert len(regions) > 1, 'the nested SDFGs must survive for this to test anything'
    for region in regions:
        assert rebuilt.cfg_list[region.cfg_id] is region, f'{region.label} does not round-trip'

    # With a stale list this raises inside `can_be_applied`; `match_exception` makes that fail here
    # instead of being printed and swallowed.
    with dace.config.temporary_config():
        dace.Config.set('optimizer', 'match_exception', value=True)
        rebuilt.apply_transformations_repeated(InlineMultistateSDFG, validate=False)


if __name__ == '__main__':
    test_is_start_state_deprecation()
    test_cfg_list_is_rebuilt_on_deserialization()
