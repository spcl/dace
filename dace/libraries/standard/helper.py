# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions: subset
stride collapsing (used to size nested-SDFG data descriptors from memlet
subsets) and dynamic map-range input promotion.
"""
from typing import List

import dace
import copy


def collapse_shape_and_strides(subset: List[dace.subsets.Range], strides: List[dace.symbolic.SymExpr]):
    """Remove singleton dimensions (length 1) from a subset/stride pair.

    The resulting strides describe the access pattern of the subset as a
    view into the parent array, so each parent stride is scaled by the
    subset's step (``stride * s``).  For unit-step subsets this is a
    no-op; for strided subsets it yields the effective per-element
    distance in the underlying memory.
    """
    collapsed_shape = []
    collapsed_strides = []
    for (b, e, s), stride in zip(subset, strides):
        length = (e + 1 - b) // s
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
    return collapsed_shape, collapsed_strides


def add_dynamic_inputs(dynamic_inputs, sdfg: dace.SDFG, subset: dace.subsets.Range, state: dace.SDFGState):
    """Promote dynamic map-range inputs to SDFG-level data descriptors.

    For each dynamic input not already present in the SDFG (e.g., a
    runtime-determined array dimension), the function adds the descriptor,
    renames existing symbolic references with a ``sym_`` prefix, and
    inserts a pre-assignment state that reads the concrete value into the
    symbol.  If no promotion is needed, the SDFG is left unchanged.
    Returns the collapsed (non-singleton) map lengths after substitution.
    """
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
        sdfg.add_state_before(state=state, label="pre_assign", is_start_block=True, assignments=pre_assignments)

    collapsed_map_lengths = [ml for ml in map_lengths if ml != 1]
    return collapsed_map_lengths
