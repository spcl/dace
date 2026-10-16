# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``CuPy`` expansions of the common library nodes: one Python tasklet over NumPy or CuPy arrays.

The array module follows the storage of the operands: ``cupy`` when they are on the GPU, ``numpy`` when they are on
the host; a node that mixes the two is refused, except a copy, which moves its data across. The tasklet keeps the
node's connectors. The Python backend binds an operand of several elements as a view of its subset, so the tasklet
copies its result into that view, and an operand of one element as its value. A subset dim of extent 1 is squeezed
away, as the einsum expansion squeezes it.
"""

from collections.abc import Callable

from dace import dtypes, library, symbolic
from dace.codegen.python import CUPY_STORAGES
from dace.frontend.operations import detect_reduction_type
from dace.libraries.blas.nodes.axpy import Axpy
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.gemv import Gemv
from dace.libraries.blas.nodes.ger import Ger
from dace.libraries.linalg.nodes.cholesky import Cholesky
from dace.libraries.linalg.nodes.solve import Solve
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.libraries.standard.nodes.reduce import Reduce
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation.transformation import ExpandTransformation

#: The name every expansion here registers under.
IMPLEMENTATION = "CuPy"

#: The array function of each reduction a ``Reduce`` node's lambda can name.
REDUCTIONS = {
    dtypes.ReductionType.Sum: "sum",
    dtypes.ReductionType.Product: "prod",
    dtypes.ReductionType.Min: "min",
    dtypes.ReductionType.Max: "max",
    dtypes.ReductionType.Logical_And: "all",
    dtypes.ReductionType.Logical_Or: "any",
}


class Operands:
    """The operands of a library node as the expressions its Python tasklet reads and writes."""

    __slots__ = ("node", "state", "sdfg", "inputs", "outputs", "xp")

    def __init__(self, node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG):
        self.node, self.state, self.sdfg = node, state, sdfg
        self.inputs = {e.dst_conn: e for e in state.in_edges(node) if not e.data.is_empty()}
        self.outputs = {e.src_conn: e for e in state.out_edges(node) if not e.data.is_empty()}
        self.xp = self.module([*self.inputs.values(), *self.outputs.values()])

    def on_gpu(self, edge: MultiConnectorEdge[Memlet]) -> bool:
        return self.sdfg.arrays[edge.data.data].storage in CUPY_STORAGES

    def module(self, edges: list[MultiConnectorEdge[Memlet]]) -> str:
        """``cupy`` or ``np``, from where the operands of ``edges`` live."""
        sides = {self.on_gpu(edge) for edge in edges}
        if len(sides) > 1:
            raise ValueError(f"{self.node.label}: operands on both the host and the GPU; copy them to one side first")
        return "cupy" if True in sides else "np"

    def squeezed(self, name: str, edge: MultiConnectorEdge[Memlet]) -> str:
        """The tasklet connector ``name`` with the extent-1 dims of its subset squeezed away."""
        subset = edge.data.subset
        if subset.num_elements() == 1:
            return name
        axes = tuple(i for i, size in enumerate(subset.size()) if size == 1)
        return f"{self.xp}.squeeze({name}, axis={axes})" if axes else name

    def __getitem__(self, conn: str) -> str:
        """The input ``conn`` as the tasklet reads it."""
        return self.squeezed(self.inputs[conn].dst_conn, self.inputs[conn])

    def coefficient(self, value, conn: str | None = None) -> str:
        """A ``value`` property, times the runtime scalar of ``conn`` when it is connected."""
        text = symbolic.symstr(value, cpp_mode=False)
        return f"({text}) * {conn}" if conn in self.inputs else text

    def single(self, conn: str) -> bool:
        """Whether the output ``conn`` is one element, which the tasklet assigns rather than copies into."""
        return self.outputs[conn].data.subset.num_elements() == 1

    def store(self, conn: str, value: str) -> str:
        """The statement that writes ``value`` to the output ``conn``."""
        if self.single(conn):
            return f"{conn} = {value}"
        return f'{self.xp}.copyto({self.squeezed(conn, self.outputs[conn])}, {value}, casting="unsafe")'


def register(node_type: type, body: Callable[[Operands], str], operands_type: type[Operands]) -> None:
    """Register ``body``, the code of the tasklet a node of ``node_type`` expands to, as its ``CuPy`` expansion."""

    def expansion(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        operands = operands_type(node, state, sdfg)
        # A tasklet cannot read and write one connector name, so the read edge of an in-out connector is renamed in
        # place before the edges move over to the tasklet; ``operands`` keeps it under the node's connector name
        for edge in state.in_edges(node):
            if edge.dst_conn in node.out_connectors:
                edge.dst_conn = f"{edge.dst_conn}_in"
        inputs = {f"{conn}_in" if conn in node.out_connectors else conn: None for conn in node.in_connectors}
        return nodes.Tasklet(
            node.label, inputs, dict.fromkeys(node.out_connectors), body(operands), language=dtypes.Language.Python
        )

    name = f"Expand{node_type.__name__}{IMPLEMENTATION}"
    members = {"environments": [], "expansion": staticmethod(expansion)}
    library.register_expansion(node_type, IMPLEMENTATION)(type(name, (ExpandTransformation,), members))


def transposed(operand: str, transpose: bool) -> str:
    return f"{operand}.swapaxes(-1, -2)" if transpose else operand


def gemm(o: Operands) -> str:
    node = o.node
    value = f"{o.coefficient(node.alpha, '_alpha')} * ({transposed(o['_a'], node.transA)} @ {transposed(o['_b'], node.transB)})"
    if "_c" in node.in_connectors:
        value += f" + {o.coefficient(node.beta, '_beta')} * {o['_c']}"
    return o.store("_c", value)


def gemv(o: Operands) -> str:
    value = f"{o.coefficient(o.node.alpha)} * ({transposed(o['_A'], o.node.transA)} @ {o['_x']})"
    if "_y" in o.node.in_connectors:
        value += f" + {o.coefficient(o.node.beta)} * {o['_y']}"
    return o.store("_y", value)


def dot(o: Operands) -> str:
    x = f"{o.xp}.conj({o['_x']})" if o.node.conjugate else o["_x"]
    return o.store("_result", f"{o.xp}.dot({x}, {o['_y']})")


def axpy(o: Operands) -> str:
    return o.store("_res", f"{o.coefficient(o.node.a)} * {o['_x']} + {o['_y']}")


def ger(o: Operands) -> str:
    return o.store("_res", f"{o.coefficient(o.node.alpha)} * {o.xp}.outer({o['_x']}, {o['_y']}) + {o['_A']}")


def batched_matmul(o: Operands) -> str:
    node = o.node
    value = f"{o.coefficient(node.alpha)} * ({transposed(o['_a'], node.transA)} @ {transposed(o['_b'], node.transB)})"
    if symbolic.symstr(node.beta, cpp_mode=False) not in ("0", "0.0"):
        value += f" + {o.coefficient(node.beta)} * {o.squeezed('_c', o.outputs['_c'])}"
    return o.store("_c", value)


def einsum(o: Operands) -> str:
    node = o.node
    inputs = ", ".join(o[conn] for conn in sorted(o.node.in_connectors) if conn not in ("_alpha", "_beta"))
    (output,) = node.out_connectors
    value = f"{o.coefficient(node.alpha, '_alpha')} * {o.xp}.einsum({node.einsum_str!r}, {inputs})"
    if symbolic.symstr(node.beta, cpp_mode=False) not in ("0", "0.0") or "_beta" in o.inputs:
        value += f" + {o.coefficient(node.beta, '_beta')} * {o.squeezed(output, o.outputs[output])}"
    return o.store(output, value)


def cholesky(o: Operands) -> str:
    factor = f"{o.xp}.linalg.cholesky({o['_a']})"
    return o.store("_b", factor if o.node.lower else f"{factor}.conj().T")


def solve(o: Operands) -> str:
    return o.store("_bout", f"{o.xp}.linalg.solve({o['_ain']}, {o['_bin']})")


def reduce(o: Operands) -> str:
    node = o.node
    kind = detect_reduction_type(node.wcr)
    if kind not in REDUCTIONS:
        raise NotImplementedError(f"{node.label}: no array function reduces with {node.wcr}")
    function = f"{o.xp}.{REDUCTIONS[kind]}"
    source, out = node.INPUT_CONNECTOR_NAME, node.OUTPUT_CONNECTOR_NAME
    axes = tuple(range(o.inputs[source].data.subset.dims())) if node.axes is None else tuple(node.axes)
    value = f"{function}({source}, axis={axes})"
    if o.single(out):
        if node.identity is None:
            raise NotImplementedError(f"{node.label}: a reduction into the value one element already holds")
        return f"{out} = {value}"
    value = f"{o.xp}.reshape({value}, {out}.shape)"
    if node.identity is None:
        # Without an identity the reduction folds into what the output holds
        value = f"{function}({o.xp}.stack(({out}, {value})), axis=0)"
    return f'{o.xp}.copyto({out}, {value}, casting="unsafe")'


def copy(o: Operands) -> str:
    src, dst = CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME
    value = src
    src_gpu, dst_gpu = o.on_gpu(o.inputs[src]), o.on_gpu(o.outputs[dst])
    if src_gpu and not dst_gpu:
        value = f"cupy.asnumpy({src})"
    elif dst_gpu and not src_gpu:
        value = f"cupy.asarray({src})"
    if o.single(dst):
        return f"{dst} = {value}"
    xp = "cupy" if dst_gpu else "np"
    return f'{xp}.copyto({dst}, {xp}.reshape({value}, {dst}.shape), casting="unsafe")'


def fill(o: Operands) -> str:
    out = FillLibraryNode.OUTPUT_CONNECTOR_NAME
    value = (
        FillLibraryNode.VALUE_CONNECTOR_NAME if FillLibraryNode.VALUE_CONNECTOR_NAME in o.inputs else repr(o.node.value)
    )
    return f"{out} = {value}" if o.single(out) else f'{o.xp}.copyto({out}, {value}, casting="unsafe")'


class CopyOperands(Operands):
    """A copy may read on one side and write on the other."""

    def module(self, edges: list[MultiConnectorEdge[Memlet]]) -> str:
        return "cupy" if any(self.on_gpu(edge) for edge in edges) else "np"


for node_type, body in (
    (Gemm, gemm),
    (Gemv, gemv),
    (Dot, dot),
    (Axpy, axpy),
    (Ger, ger),
    (BatchedMatMul, batched_matmul),
    (Einsum, einsum),
    (Cholesky, cholesky),
    (Solve, solve),
    (Reduce, reduce),
    (FillLibraryNode, fill),
):
    register(node_type, body, Operands)
register(CopyLibraryNode, copy, CopyOperands)
