# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Indexed operands of the matrix-product library nodes (``MatMul``, ``Gemm``, ``Gemv`` and ``BatchedMatMul``).

An operand connector ``X`` may come with an index connector ``X_index``, an array of integers. The node then uses the
operand ``X[X_index]``, as NumPy indexes the first dimension of an array: in the view of the operand that the node
works with (its memlet's subset, squeezed as for a matrix, see ``matrix_view``), the first dimension is replaced by the
dimensions of the index. On an output, ``C[C_index] = result`` writes every row of the result to the row of ``C`` that
the index names, and a sum on the output edge adds it there instead (a scatter-add); an input ``C`` of the same
connector is read through the same index. For example:

* ``x[rows] @ w`` is a ``MatMul`` with ``_a = x`` and ``_a_index = rows``;
* a grouped product, in which tile ``t`` multiplies the rows ``rows[t]`` of ``x`` by the matrix ``w[group[t]]``, is a
  ``BatchedMatMul`` with ``_a = x``, ``_a_index = rows`` (of shape ``[tiles, rows]``), ``_b = w`` and
  ``_b_index = group`` (of shape ``[tiles]``);
* adding the rows of such a product to the rows of ``y`` they belong to is ``_c = y`` and ``_c_index = rows``, with a
  sum on the output edge.

Implementations that read and write the operands through their indices do so in their own expansion. The others expand
an indexed node by copies (see ``expand_by_copies``): the indexed operands are gathered into transients, the node
without index connectors computes the product (with the same implementation), and an indexed output is scattered from a
transient. The same expansion copies operands stored in blocks of columns (see ``dace.libraries.blas.blocked``) into
the usual layout.
"""

import copy
from collections.abc import Collection

from dace import SDFG, Memlet, SDFGState, dtypes, nodes
from dace.libraries.blas.blocked import operand_view
from dace.sdfg.scope import is_devicelevel_gpu

#: The suffix of an index connector
INDEX = "_index"


def index_connector(connector: str) -> str:
    """The index connector of an operand connector."""
    return connector + INDEX


def is_index_connector(connector: str) -> bool:
    return connector.endswith(INDEX)


def indexed_connectors(node: nodes.LibraryNode) -> list[str]:
    """The operand connectors of a node that have an index connector."""
    return sorted(c[: -len(INDEX)] for c in node.in_connectors if is_index_connector(c))


def index_shape(state: SDFGState, node: nodes.LibraryNode, connector: str) -> list | None:
    """The shape of the index of an operand (squeezed), or None if the operand is not indexed."""
    for e in state.in_edges(node):
        if e.dst_conn == index_connector(connector):
            subset = copy.deepcopy(e.data.subset)
            subset.squeeze()
            return subset.size()
    return None


def indexed_size(state: SDFGState, node: nodes.LibraryNode, connector: str, size: list) -> list:
    """The size of an operand as the node uses it: ``size`` (of the operand's view) indexed in its first dimension."""
    index = index_shape(state, node, connector)
    return list(size) if index is None else list(index) + list(size[1:])


def operand_edges(state: SDFGState, node: nodes.LibraryNode) -> list:
    """The input edges of a node, without those of index connectors."""
    return [e for e in state.in_edges(node) if not is_index_connector(e.dst_conn)]


def implementation_of(node: nodes.LibraryNode, expansion: type) -> str:
    """The name under which the class of a node has an expansion (the node's implementation, if it has none)."""
    for name, cls in type(node).implementations.items():
        if cls is expansion:
            return name
    return node.implementation


def without_indices(node: nodes.LibraryNode) -> nodes.LibraryNode:
    """A copy of ``node`` without its index connectors."""
    result = copy.deepcopy(node)
    for connector in list(result.in_connectors):
        if is_index_connector(connector):
            result.remove_in_connector(connector)
    return result


def expand_by_copies(
    node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, implementation: str, blocked: Collection[str] = ()
) -> SDFG:
    """An expansion of a node with index connectors, or with operands stored in blocks of columns, by copies: every
    indexed operand is gathered into a transient, every blocked one copied into a transient in the usual layout, the
    node without index connectors and blocked operands (expanded later with ``implementation``) computes the product,
    and an indexed output is scattered from a transient.

    :param node: The node, with index connectors or blocked operands.
    :param state: The state of the node.
    :param sdfg: The SDFG of the state.
    :param implementation: The implementation of the node without index connectors.
    :param blocked: The blocked operands of the node (inputs, see ``dace.libraries.blas.blocked``).
    :return: The expansion, as a nested SDFG.
    """
    nsdfg = SDFG(f"{node.label}_indexed")
    nstate = nsdfg.add_state(node.label)
    in_host_code = not is_devicelevel_gpu(sdfg, state, node)  # Copies of GPU memory run as kernels

    def schedule_of(desc) -> dtypes.ScheduleType:
        if in_host_code and desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned):
            return dtypes.ScheduleType.GPU_Device
        return dtypes.ScheduleType.Default

    # The connectors, in the view of their operands that the node works with
    for e in state.in_edges(node) + state.out_edges(node):
        connector = e.dst_conn if e.dst is node else e.src_conn
        if connector in nsdfg.arrays:  # An input and output of the same name
            continue
        if connector in blocked and (e.src is node or index_connector(connector) in node.in_connectors):
            raise ValueError(f'"{node.label}": a blocked operand ("{connector}") must be an input without an index')
        desc = sdfg.arrays[e.data.data]
        size, dims = operand_view(e.data.subset, connector in blocked)
        nsdfg.add_array(connector, size, desc.dtype, storage=desc.storage, strides=[desc.strides[d] for d in dims])

    def copy_map(name: str, operand: str, transient: str, gather: bool, access: dict, wcr: str | None = None):
        """A map that copies between an operand, indexed in its first dimension, and a transient (in the shape of the
        indexed operand), using the given access nodes."""
        index = index_connector(operand)
        idesc, odesc, tdesc = nsdfg.arrays[index], nsdfg.arrays[operand], nsdfg.arrays[transient]
        iparams = [f"__i{d}" for d in range(len(idesc.shape))]
        rparams = [f"__r{d}" for d in range(len(odesc.shape) - 1)]
        rest = "".join(f", {p}" for p in rparams)
        schedule = schedule_of(odesc)
        element = Memlet(f"{transient}[{', '.join(iparams + rparams)}]")
        row = Memlet(f"{operand}[0:{odesc.shape[0]}{rest}]", wcr=wcr, dynamic=not gather)
        nstate.add_mapped_tasklet(
            name,
            {p: f"0:{s}" for p, s in zip(iparams + rparams, tdesc.shape)},
            {"__ind": Memlet(f"{index}[{', '.join(iparams)}]"), **({"__arr": row} if gather else {"__val": element})},
            "__out = __arr[__ind]" if gather else "__out[__ind] = __val",
            {"__out": element if gather else row},
            schedule=schedule,
            external_edges=True,
            input_nodes={a: n for a, n in access.items() if a == (operand if gather else transient)},
            output_nodes={a: n for a, n in access.items() if a == (transient if gather else operand)},
        )

    def transient_of(connector: str, suffix: str) -> str:
        """A transient in the shape of an indexed operand."""
        index, operand = nsdfg.arrays[index_connector(connector)], nsdfg.arrays[connector]
        name, _ = nsdfg.add_transient(
            f"{connector}_{suffix}",
            list(index.shape) + list(operand.shape[1:]),
            operand.dtype,
            storage=operand.storage,
            find_new_name=True,
        )
        return name

    def unblock(connector: str) -> nodes.AccessNode:
        """A transient that holds the matrix of a blocked operand, in the usual layout."""
        blocks, rows, width = nsdfg.arrays[connector].shape
        desc = nsdfg.arrays[connector]
        name, _ = nsdfg.add_transient(
            f"{connector}_unblocked", [rows, blocks * width], desc.dtype, storage=desc.storage, find_new_name=True
        )
        access = nstate.add_access(name)
        nstate.add_mapped_tasklet(
            f"unblock{connector}",
            {"__r": f"0:{rows}", "__c": f"0:{blocks * width}"},
            {"__in": Memlet(f"{connector}[__c // ({width}), __r, __c % ({width})]")},
            "__out = __in",
            {"__out": Memlet(f"{name}[__r, __c]")},
            schedule=schedule_of(desc),
            external_edges=True,
            output_nodes={name: access},
        )
        return access

    # The operands of the product: the gathered (or unblocked) transients, or the operands themselves
    indexed = set(indexed_connectors(node))
    reads, writes = {}, {}  # Connector -> (access node, wcr)
    for e in state.in_edges(node):
        connector = e.dst_conn
        if is_index_connector(connector):
            continue
        if connector in indexed:
            source = transient_of(connector, "gathered")
            reads[connector] = nstate.add_access(source)
            copy_map(f"gather{connector}", connector, source, True, {source: reads[connector]})
        elif connector in blocked:
            reads[connector] = unblock(connector)
        else:
            reads[connector] = nstate.add_read(connector)
    for e in state.out_edges(node):
        connector = e.src_conn
        if connector in indexed:
            target = transient_of(connector, "scattered")
            writes[connector] = (nstate.add_access(target), None)
            copy_map(f"scatter{connector}", connector, target, False, {target: writes[connector][0]}, e.data.wcr)
        else:
            writes[connector] = (nstate.add_write(connector), e.data.wcr)

    # The product: the node without indices, in a nested SDFG of its own, whose arrays are not named as the node's
    # connectors (an expansion into a tasklet would shadow them)
    product = SDFG(f"{node.label}_product")
    pstate = product.add_state(node.label)
    inner = without_indices(node)
    inner.implementation = implementation
    if blocked:
        inner.blocked = frozenset()
    pstate.add_node(inner)

    def operand(connector: str, access: nodes.AccessNode) -> str:
        name = f"{connector}_operand"
        if name not in product.arrays:
            desc = copy.deepcopy(nsdfg.arrays[access.data])
            desc.transient = False
            product.add_datadesc(name, desc)
        return name

    for connector, access in reads.items():
        name = operand(connector, access)
        pstate.add_edge(pstate.add_read(name), None, inner, connector, Memlet.from_array(name, product.arrays[name]))
    for connector, (access, wcr) in writes.items():
        name = operand(connector, access)
        memlet = Memlet.from_array(name, product.arrays[name])
        memlet.wcr = wcr
        pstate.add_edge(inner, connector, pstate.add_write(name), None, memlet)
    call = nstate.add_nested_sdfg(
        product, {f"{c}_operand" for c in reads}, {f"{c}_operand" for c in writes}, name=f"{node.label}_product"
    )
    for connector, access in reads.items():
        memlet = Memlet.from_array(access.data, nsdfg.arrays[access.data])
        nstate.add_edge(access, None, call, f"{connector}_operand", memlet)
    for connector, (access, wcr) in writes.items():
        memlet = Memlet.from_array(access.data, nsdfg.arrays[access.data])
        memlet.wcr = wcr
        nstate.add_edge(call, f"{connector}_operand", access, None, memlet)
    return nsdfg


def specialized_index_connectors(node: nodes.LibraryNode, result: nodes.LibraryNode, renames: dict[str, str]) -> None:
    """Adds the index connectors of ``node`` to the node it specializes to, whose operand connectors are named as
    ``renames`` says (the others keep their names)."""
    for connector in indexed_connectors(node):
        result.add_in_connector(index_connector(renames.get(connector, connector)))


__all__ = [
    "INDEX",
    "index_connector",
    "is_index_connector",
    "indexed_connectors",
    "index_shape",
    "indexed_size",
    "operand_edges",
    "implementation_of",
    "without_indices",
    "expand_by_copies",
    "specialized_index_connectors",
]
