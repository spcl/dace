# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy


# Compute collapsed shapes and strides, removing singleton dimensions (length == 1)
def collapse_shape_and_strides(subset, strides):
    collapsed_shape = []
    collapsed_strides = []
    for (b, e, s), stride in zip(subset, strides):
        length = (e + 1 - b) // s
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride)
    return collapsed_shape, collapsed_strides


def add_dynamic_inputs(dynamic_inputs, sdfg: dace.SDFG, subset: dace.subsets.Range, state: dace.SDFGState):
    # Add dynamic inputs
    pre_assignments = dict()
    map_lengths = [dace.symbolic.SymExpr((e + 1 - b) // s) for (b, e, s) in subset]

    for dynamic_input_name, datadesc in dynamic_inputs.items():
        if dynamic_input_name in sdfg.arrays:
            continue

        if dynamic_input_name in sdfg.symbols:
            continue

        sdfg.replace(str(dynamic_input_name), "sym_" + str(dynamic_input_name))
        ndesc = copy.deepcopy(datadesc)
        ndesc.transient = False
        sdfg.add_datadesc(dynamic_input_name, ndesc)
        # Should be scalar
        if isinstance(ndesc, dace.data.Scalar):
            pre_assignments["sym_" + dynamic_input_name] = f"{dynamic_input_name}"
        else:
            assert ndesc.shape == (1, ) or ndesc.shape == [
                1,
            ]
            pre_assignments["sym_" + dynamic_input_name] = f"{dynamic_input_name}[0]"

        new_map_lengths = []
        for ml in map_lengths:
            nml = ml.subs({str(dynamic_input_name): "sym_" + str(dynamic_input_name)})
            new_map_lengths.append(nml)
        map_lengths = new_map_lengths

    if pre_assignments != dict():
        # Add a state for assignments in the beginning
        sdfg.add_state_before(state=state, label="pre_assign", is_start_block=True, assignments=pre_assignments)

    collapsed_map_lengths = [ml for ml in map_lengths if ml != 1]
    return collapsed_map_lengths
