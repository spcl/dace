# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Union, Tuple, Type, Optional, List

import numpy as np
import os
import dace
import copy
import uuid
import pytest
import gc

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.transformation import dataflow as dftrans

from .map_fusion_vertical_test import count_nodes, unique_name


def _make_parallel_sdfg(common_ancestor: bool, ) -> dace.SDFG:
    """Creates maps that are parallel and can only be handled by parallel map fusion.
    """
    sdfg = dace.SDFG(unique_name("parallel_maps_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    names = ["A", "B", "C", "D", "out"]
    for name in names:
        sdfg.add_array(
            name,
            shape=((10, 4) if name == "out" else (10, )),
            dtype=dace.float64,
            transient=False,
        )

    out = state.add_access("out")

    if common_ancestor:
        input_nodes = {state.add_access("A")}
    else:
        input_nodes = set()

    for i, name in enumerate(["A", "B", "C"]):
        it = f"__{i}"
        state.add_mapped_tasklet(
            f"comp_{i}",
            map_ranges={it: "0:10"},
            inputs={"__in": dace.Memlet(f"{name}[{it}]")},
            code=f"__out = __in + {i}.0",
            outputs={"__out": dace.Memlet(f"out[{it}, {i}]")},
            input_nodes=input_nodes,
            output_nodes={out},
            external_edges=True,
        )

    state.add_mapped_tasklet(
        "comp_4",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("A[__i]"),
            "__in2": dace.Memlet("D[__i]")
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet(f"out[__i, 3]")},
        input_nodes=input_nodes,
        output_nodes={out},
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


def _make_parallel_sdfg_args():
    args_ref = {
        "A": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "B": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "C": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "D": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "out": np.array(np.random.rand(10, 4), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)
    return args_ref, args_res


def test_vertical_map_fusion_does_not_apply():
    sdfg = _make_parallel_sdfg(common_ancestor=False)

    # There is no vertical/serial dependency thus MapFusionVertical will not apply.
    count = sdfg.apply_transformations_repeated(
        [dftrans.MapFusionVertical()],
        validate=True,
        validate_all=True,
    )
    assert count == 0


def test_horizontal_no_comon_ancestor():
    sdfg = _make_parallel_sdfg(common_ancestor=False)
    assert count_nodes(sdfg, nodes.AccessNode) == 6
    assert count_nodes(sdfg, nodes.MapExit) == 4


def test_parallel_map_fusion_simple():
    sdfg = _make_parallel_sdfg()
    args_ref = {
        "A": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "B": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "C": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "D": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "out": np.array(np.random.rand(10, 4), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    run_sdfg(sdfg, **args_ref)

    # In serial map fusion nothing will be done.
    sdfg = apply_fusion(
        sdfg,
        removed_maps=0,
        allow_serial_map_fusion=True,
        allow_parallel_map_fusion=False,
    )

    # Because there is no common ancestor the transformation will not apply.
    sdfg = apply_fusion(
        sdfg,
        removed_maps=0,
        allow_serial_map_fusion=False,
        allow_parallel_map_fusion=True,
        only_if_common_ancestor=True,
    )

    # In parallel map fusion it will be fused away.
    sdfg = apply_fusion(
        sdfg,
        final_maps=1,
        allow_serial_map_fusion=False,
        allow_parallel_map_fusion=True,
        only_if_common_ancestor=False,
    )

    # Now look if the code is the same.
    run_sdfg(sdfg, **args_res)

    assert all(np.allclose(args_ref[arg], args_res[arg]) for arg in args_ref.keys())


if __name__ == '__main__':
    pass
