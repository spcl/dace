# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Array-descriptor mutation helpers (add or rebuild ``dace.data.Array`` descriptors on an SDFG)."""

import dace


def demote_connector_views(nsdfg_node: dace.nodes.NestedSDFG) -> int:
    """Demote a NestedSDFG's connector descriptors that are ``View`` s to plain arrays.

    A NestedSDFG connector is fed by a parent-side memlet that already resolves any view: inside
    the body the connector is a plain array parameter, never a ``View`` (a ``View`` is only valid
    with a viewing edge to/from the viewed AccessNode in the same state). The descriptor-copying
    helpers that build / re-widen these connectors -- ``nest_state_subgraph`` and
    ``ExpandNestedSDFGInputs`` -- ``copy.deepcopy`` the parent descriptor inward and only clear
    ``transient``. When the parent array is a frontend reshape/flatten ``View`` (e.g. ``C_0``
    viewing ``C[0:XN, 0:YN]`` as ``C[0:XN*YN]``), the connector copy is a ``View`` too, with no
    viewing edge inside the body -- so ``get_view_edge`` finds the connector node's sole edge leads
    to a code node and ``validate()`` rejects it ("Ambiguous or invalid edge to/from a View access
    node").

    Demote each such connector descriptor to its ``Array`` / ``ContainerArray`` equivalent (shape,
    strides, dtype unchanged, so no connector memlet is perturbed), which makes the body validate.

    :param nsdfg_node: the NestedSDFG whose connector descriptors are normalised in place.
    :returns: number of connector descriptors demoted.
    """
    body = nsdfg_node.sdfg
    demoted = 0
    for name in set(nsdfg_node.in_connectors) | set(nsdfg_node.out_connectors):
        desc = body.arrays.get(name)
        if isinstance(desc, (dace.data.ArrayView, dace.data.ContainerView)):
            body.arrays[name] = desc.as_array()
            demoted += 1
    return demoted
