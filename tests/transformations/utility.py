# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Helper functions for compilation. """
from typing import Any, Union, Tuple, Type, Optional, List, Dict

import copy
import numpy as np
import dace
import uuid

from dace import SDFG, SDFGState, data as dace_data
from dace.sdfg import nodes


def count_nodes(
    graph: Union[SDFG, SDFGState],
    node_type: Union[Tuple[Type, ...], Type],
    return_nodes: bool = False,
) -> Union[int, List[nodes.Node]]:
    """Counts the number of nodes of a particular type in `graph`.

    If `graph` is an SDFGState then only count the nodes inside this state,
    but if `graph` is an SDFG count in all states.

    Args:
        graph: The graph to scan.
        node_type: The type or sequence of types of nodes to look for.
    """

    states = graph.states() if isinstance(graph, dace.SDFG) else [graph]
    found_nodes: list[nodes.Node] = []
    for state_nodes in states:
        for node in state_nodes.nodes():
            if isinstance(node, node_type):
                found_nodes.append(node)
    if return_nodes:
        return found_nodes
    return len(found_nodes)


def unique_name(name: str) -> str:
    """Adds a unique string to `name`."""
    maximal_length = 200
    unique_sufix = str(uuid.uuid1()).replace("-", "_")
    if len(name) > (maximal_length - len(unique_sufix)):
        name = name[:(maximal_length - len(unique_sufix) - 1)]
    return f"{name}_{unique_sufix}"


def make_sdfg_args(sdfg: dace.SDFG, ) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = {
        name: (np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype()) if isinstance(
            desc, dace_data.Array) else np.array(np.random.rand(1), copy=True, dtype=desc.dtype.as_numpy_dtype())[0])
        for name, desc in sdfg.arrays.items() if (not desc.transient) and (not name.startswith("__return"))
    }
    res = copy.deepcopy(ref)
    return ref, res


def compile_and_run_sdfg(
    sdfg: dace.SDFG,
    *args: Any,
    **kwargs: Any,
) -> dace.codegen.CompiledSDFG:
    """This function guarantees that the SDFG is compiled and run.

    This function will modify the name of the SDFG to ensure that the code is
    regenerated and recompiled properly. It will also suppress warnings about
    shared objects that are loaded multiple times.
    """

    with dace.config.set_temporary("compiler.use_cache", value=False):
        sdfg_clone = copy.deepcopy(sdfg)

        sdfg_clone.name = unique_name(sdfg_clone.name)
        sdfg_clone._recompile = True
        sdfg_clone._regenerate_code = True  # TODO(phimuell): Find out if it has an effect.
        csdfg = sdfg_clone.compile()
        csdfg(*args, **kwargs)
    return csdfg


def compare_sdfg_res(
    ref: dict[str, Any],
    res: dict[str, Any],
) -> bool:
    """Compares if `res` and  `ref` are the same."""
    return all(np.allclose(ref[name], res[name]) for name in ref.keys())
