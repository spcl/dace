# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``CuPy`` expansions of the common library nodes: one Python tasklet over NumPy or CuPy arrays.

The array module follows the storage of the operands: ``cupy`` when they are on the GPU, ``numpy`` when they are on
the host; a node that mixes the two is refused, except a copy, which moves its data across. The tasklet keeps the
node's connectors. The Python backend binds an operand of several elements as a view of its subset, so the tasklet
copies its result into that view, and an operand of one element as its value. A subset dim of extent 1 is squeezed
away, as the einsum expansion squeezes it.
"""

import ast
import copy as copy_module
from collections.abc import Callable

import numpy as np

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
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.libraries.standard.nodes import scan as scan_node
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.libraries.standard.nodes.find_first import INDEX_NAME, FindFirst
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

    __slots__ = ("node", "state", "sdfg", "inputs", "outputs", "xp", "reads")

    def __init__(self, node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG):
        self.node, self.state, self.sdfg = node, state, sdfg
        #: The outputs whose value before the node runs the tasklet reads (:meth:`current`)
        self.reads: list[str] = []
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

    def connector_type(self, edge: MultiConnectorEdge[Memlet] | None) -> dtypes.pointer | None:
        """A pointer for one GPU element: the CuPy call takes it on the device, the host never dereferences it."""
        if edge is None or edge.data.subset.num_elements() != 1 or not self.on_gpu(edge):
            return None
        return dtypes.pointer(self.sdfg.arrays[edge.data.data].dtype)

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

    def current(self, conn: str) -> str:
        """The value the output ``conn`` holds before the node runs, read through an input connector of its own."""
        self.reads.append(conn)
        return f"{conn}_in"

    def store(self, conn: str, value: str) -> str:
        """The statement that writes ``value`` to the output ``conn``."""
        if self.single(conn):
            return f"{conn} = {value}"
        return f'{self.xp}.copyto({self.squeezed(conn, self.outputs[conn])}, {value}, casting="unsafe")'


def register(node_type: type, body: Callable[[Operands], str], operands_type: type[Operands]) -> None:
    """Register ``body``, the code of the tasklet a node of ``node_type`` expands to, as its ``CuPy`` expansion."""

    def expansion(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        operands = operands_type(node, state, sdfg)
        # The body may refuse the node, so the graph changes only after it returns
        code = body(operands)
        # A tasklet cannot read and write one connector name, so the read edge of an in-out connector is renamed in
        # place before the edges move over to the tasklet; ``operands`` keeps it under the node's connector name
        for edge in state.in_edges(node):
            if edge.dst_conn in node.out_connectors:
                edge.dst_conn = f"{edge.dst_conn}_in"
        inputs = {
            f"{conn}_in" if conn in node.out_connectors else conn: operands.connector_type(operands.inputs.get(conn))
            for conn in node.in_connectors
        }
        for conn in operands.reads:
            read_output(state, node, operands.outputs[conn], f"{conn}_in")
            inputs[f"{conn}_in"] = operands.connector_type(operands.outputs[conn])
        outputs = {conn: operands.connector_type(operands.outputs.get(conn)) for conn in node.out_connectors}
        return nodes.Tasklet(node.label, inputs, outputs, code, language=dtypes.Language.Python)

    name = f"Expand{node_type.__name__}{IMPLEMENTATION}"
    members = {"environments": [], "expansion": staticmethod(expansion)}
    library.register_expansion(node_type, IMPLEMENTATION)(type(name, (ExpandTransformation,), members))


def read_output(state: SDFGState, node: nodes.LibraryNode, edge: MultiConnectorEdge[Memlet], conn: str) -> None:
    """Wire the elements the output ``edge`` writes into ``node`` as the input ``conn``, through the node's scopes."""
    scopes = []
    entry = state.entry_node(node)
    while entry is not None:
        scopes.append(entry)
        entry = state.entry_node(entry)
    node.add_in_connector(conn)
    memlet = Memlet(data=edge.data.data, subset=copy_module.deepcopy(edge.data.subset))
    state.add_memlet_path(state.add_read(edge.data.data), *reversed(scopes), node, dst_conn=conn, memlet=memlet)


def literal(value) -> str:
    """A number property as Python source: an integer as written, any other through ``float`` (``inf`` too)."""
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    return f"float({str(value)!r})"


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
    if not o.single(out):
        value = f"{o.xp}.reshape({value}, {out}.shape)"
    if node.identity is None:
        # Without an identity the reduction folds into what the output holds
        value = f"{function}({o.xp}.stack(({o.xp}.asarray({o.current(out)}), {value})), axis=0)"
    return f"{out} = {value}" if o.single(out) else f'{o.xp}.copyto({out}, {value}, casting="unsafe")'


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


def tensor_transpose(o: Operands) -> str:
    # The axes name the dims of the operand as wired; only an operand of more dims has extent-1 dims to drop
    node, source, target = o.node, o.inputs["_inp_tensor"], o.outputs["_out_tensor"]
    whole = source.data.subset.dims() == len(node.axes)
    operand = "_inp_tensor" if whole else o["_inp_tensor"]
    value = f"{o.coefficient(node.alpha)} * {o.xp}.transpose({operand}, {tuple(node.axes)})"
    if symbolic.symstr(node.beta, cpp_mode=False) not in ("0", "0.0"):
        value += f" + {o.coefficient(node.beta)} * {'_out_tensor' if whole else o.squeezed('_out_tensor', target)}"
    if o.single("_out_tensor") or not whole:
        return o.store("_out_tensor", value)
    return f'{o.xp}.copyto(_out_tensor, {value}, casting="unsafe")'


def scatter_conflict_check(o: Operands) -> str:
    # The flag is a host value whatever side the indices live on
    indices = o[ScatterConflictCheck.INPUT_CONNECTOR_NAME]
    return f"{ScatterConflictCheck.OUTPUT_CONNECTOR_NAME} = int({o.xp}.unique({indices}).size != {indices}.size)"


#: The array function of each scan op that has one; the others fold by doubling.
CUMULATIVE = {scan_node.ScanOp.SUM: "cumsum", scan_node.ScanOp.PRODUCT: "cumprod"}
#: Each scalar scan op as the elementwise function folding two arrays.
SCAN_FOLDS = {
    scan_node.ScanOp.SUM: "{xp}.add",
    scan_node.ScanOp.PRODUCT: "{xp}.multiply",
    scan_node.ScanOp.MIN: "{xp}.minimum",
    scan_node.ScanOp.MAX: "{xp}.maximum",
}
SCAN_IDENTITIES = {scan_node.ScanOp.SUM: "0", scan_node.ScanOp.PRODUCT: "1"}


def doubling(update: str) -> str:
    """A prefix fold in log2(n) whole-array steps (Hillis-Steele), ``update`` run at each shift ``__ss``."""
    return f"__ss = 1\nwhile __ss < __sy.shape[0]:\n    {update}\n    __ss *= 2\n"


def scan_seed(node: scan_node.Scan) -> str | None:
    """The value the scan starts from, in the order :func:`~dace.libraries.standard.nodes.scan.chain_seed` picks it."""
    if scan_node.INIT_CONNECTOR_NAME in node.in_connectors and not node.exclusive:
        return scan_node.INIT_CONNECTOR_NAME
    if node.op is scan_node.ScanOp.AFFINE:
        return "0"
    if node.identity is not None:
        return literal(node.identity)
    return SCAN_IDENTITIES.get(node.op)


def value_scan(node: scan_node.Scan, xp: str, seed: str | None) -> str:
    """The code scanning ``__sy`` in place with one of the scalar ops, from ``seed``."""
    fold = SCAN_FOLDS[node.op].format(xp=xp)
    if node.op in CUMULATIVE:
        code = f"__sy = {xp}.{CUMULATIVE[node.op]}(__sy)\n"
    else:
        code = doubling(f"__sy[__ss:] = {fold}(__sy[__ss:], __sy[:-__ss])")
    if seed is not None:
        code += f"__sy = {fold}({seed}, __sy)\n"
    if node.exclusive:
        if seed is None:
            raise ValueError(f"{node.label}: an exclusive {node.op.value} scan needs an identity")
        code += f"__sy = {xp}.concatenate(({xp}.full(1, {seed}, __sy.dtype), __sy))[:__sy.shape[0]]\n"
    return code


def scan(o: Operands) -> str:
    node, xp = o.node, o.xp
    if node.chains != 1 or not symbolic.equal_valued(1, node.stride) or not symbolic.equal_valued(1, node.segments):
        raise NotImplementedError(f"{node.label}: the CuPy scan takes one unit-stride chain over one segment")
    seed = scan_seed(node)
    # One element binds as its value; the scan walks a one-dim array
    code = f"__sy = {xp}.array({o[scan_node.INPUT_CONNECTOR_NAME]}, ndmin=1)\n"
    if node.op is scan_node.ScanOp.AFFINE:
        if node.exclusive:
            raise NotImplementedError(f"{node.label}: an exclusive affine scan has no CuPy form")
        # The doubling composes the affine maps ``x -> a*x + b``; the seed enters through the composed slope
        code += f"__sa = {xp}.array({o[scan_node.COEF_CONNECTOR_NAME]}, ndmin=1)\n"
        code += doubling(
            "__sy[__ss:] = __sa[__ss:] * __sy[:-__ss] + __sy[__ss:]\n    __sa[__ss:] = __sa[__ss:] * __sa[:-__ss]"
        )
        code += f"__sy = __sy + __sa * ({seed})\n"
    else:
        code += value_scan(node, xp, seed)
    out = scan_node.OUTPUT_CONNECTOR_NAME
    return code + o.store(out, "__sy[0]" if o.single(out) else "__sy")


def arg_reduce(o: Operands) -> str:
    # Both answers are host values whatever side the operand lives on; argmax keeps the first extreme
    node = o.node
    values = o["_in"] if not node.transform else f"{node.transform}({o['_in']})"
    code = f"__arv = {values}\n_out_idx = int({o.xp}.arg{node.op}(__arv))\n"
    return code + ("_out_val = __arv[_out_idx].item()" if "_out_val" in o.outputs else "")


class Elementwise(ast.NodeTransformer):
    """A predicate over a whole index range: ``and`` / ``or`` / ``not`` become the logical array functions, and an
    array subscript is clamped into the array, since a guard such as ``i > 0 and a[i - 1]`` no longer
    short-circuits the read it guards; the guard still decides the lanes it covered."""

    def __init__(self, xp: str, arrays: set[str]) -> None:
        self.xp, self.arrays = xp, arrays

    def call(self, function: str, args: list[ast.expr]) -> ast.Call:
        return ast.Call(ast.Attribute(ast.Name(self.xp, ast.Load()), function, ast.Load()), args, [])

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        function = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        result = node.values[0]
        for value in node.values[1:]:
            result = self.call(function, [result, value])
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        return self.call("logical_not", [node.operand]) if isinstance(node.op, ast.Not) else node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        self.generic_visit(node)
        if not (isinstance(node.value, ast.Name) and node.value.id in self.arrays):
            return node
        index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        clamped = [
            self.call("clip", [i, ast.Constant(0), ast.parse(f"{node.value.id}.shape[{dim}] - 1", mode="eval").body])
            for dim, i in enumerate(index)
        ]
        node.slice = ast.Tuple(clamped, ast.Load()) if len(clamped) > 1 else clamped[0]
        return node


def find_first(o: Operands) -> str:
    # The predicate is tested over the whole range at once; the answer is a host value
    node, xp = o.node, o.xp
    begin, end = (symbolic.symstr(bound, cpp_mode=False) for bound in (node.begin, node.end))
    arrays = {conn for conn, edge in o.inputs.items() if edge.data.subset.num_elements() != 1}
    predicate = ast.unparse(Elementwise(xp, arrays).visit(ast.parse(node.predicate, mode="eval")))
    return (
        f"{INDEX_NAME} = {xp}.arange({begin}, max({begin}, {end}), dtype={xp}.int64)\n"
        f"__ffm = {xp}.broadcast_to({xp}.asarray({predicate}, dtype={xp}.bool_), {INDEX_NAME}.shape)\n"
        f"_out_idx = int({INDEX_NAME}[{xp}.argmax(__ffm)]) if bool(__ffm.any()) else ({end})"
    )


class CrossDeviceOperands(Operands):
    """A copy, a node whose answers stay on the host, or a scan seeded from a host scalar may read on one side and
    write on the other."""

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
    (TensorTranspose, tensor_transpose),
):
    register(node_type, body, Operands)
register(CopyLibraryNode, copy, CrossDeviceOperands)
register(ScatterConflictCheck, scatter_conflict_check, CrossDeviceOperands)
register(scan_node.Scan, scan, CrossDeviceOperands)
register(ArgReduce, arg_reduce, CrossDeviceOperands)
register(FindFirst, find_first, CrossDeviceOperands)
