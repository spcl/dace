# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dataclasses
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from dace import InterstateEdge, Memlet, data as dt, dtypes, properties
from dace.autodiff.backward_pass_generator import BackwardPassGenerator
from dace.autodiff import utils as ad_utils
from dace.autodiff.base_abc import AutoDiffException
from dace.libraries.standard import Reduce

from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutils
from dace.sdfg.utils import inline_control_flow_regions
from dace.sdfg.state import ControlFlowBlock, LoopRegion
from dace.transformation.passes.while_to_for_loop import WhileToForLoop


def add_backward_pass(
    sdfg: SDFG,
    outputs: List[Union[nodes.AccessNode, str]],
    inputs: List[Union[nodes.AccessNode, str]],
    data_forwarding_strategy: str = "store_all",
    data_to_recompute: Optional[List[str]] = None,
    simplify: bool = True,
    separate_sdfgs: bool = False,
) -> Optional[SDFG]:
    """Experimental: Add a backward pass to `state` using reverse-mode automatic differentiation.

    ``inputs``, ``outputs`` and ``grads`` can be provided either as ``AccessNode`` nodes, or as ``str``, in which
    case the graph will be searched for exactly one matching ``AccessNode`` with data matching the ``str``.

    The SDFG may contain the following nodes:

    * Maps
    * AccessNodes
    * Reductions (Sum, Min, Max)
    * ONNXOps
    * Multiple states
    * LoopRegions
    * NestedSDFGs (subject to the same constraints)

    When differentiating an :class:`~dace.libraries.onnx.nodes.onnx_op.ONNXOp`, the ONNXBackward registry will be checked
    for any matching backward pass implementations. If none are found, the ONNXForward registry will be checked for
    matching pure implementations. If one is found, symbolic differentiation of the pure implementation will be
    attempted. If this fails, or no pure forward implementation is found, the method will fail.

    .. note::
        This function modifies the input SDFG in-place. Even if ``separate_sdfgs`` is ``True``, modifications
        such as storing intermediate results and inlining ControlFlowRegions can be applied to the original SDFG.

    :param sdfg: the SDFG to add the backward pass to.
    :param outputs: the forward pass outputs of the function to differentiate.
    :param inputs: the inputs w.r.t. which the gradient will be returned.
    :param data_forwarding_strategy: strategy for forwarding data to the backward pass. Could be one of:
        * "store_all": store all intermediate data (default, uses most memory, fastest).
        * "recompute_all": recompute all intermediate data.
        * "user_defined": store all intermediates except for ones specified in `data_to_recompute`.
    :param data_to_recompute: list of data arrays to recompute instead of storing. Only used if
        `data_forwarding_strategy` is "user_defined".
    :param simplify: whether to apply the simplify pass to the forward and backward SDFGs.
    :param separate_sdfgs: whether to create a separate SDFG for the backward pass (see
                           :func:`make_backward_pass`, which also returns how to call the two SDFGs).
    :return: the backward SDFG if separate_sdfgs is True, the original SDFG (which now also contains the backward pass) otherwise.
    """
    # Validate the SDFG
    sdfg.validate()

    if simplify:
        sdfg.simplify()

    # Inline conditional blocks but keep loops, as for loops
    _prepare_control_flow(sdfg)

    if separate_sdfgs:
        return make_backward_pass(
            sdfg, outputs, inputs, data_forwarding_strategy, data_to_recompute, simplify=simplify, simplified=True
        ).backward
    backward_sdfg = sdfg

    # Add backward pass
    gen = BackwardPassGenerator(
        sdfg=sdfg,
        given_gradients=outputs,
        required_gradients=inputs,
        backward_sdfg=backward_sdfg,
        data_forwarding_strategy=data_forwarding_strategy,
        data_to_recompute=data_to_recompute,
    )
    gen.backward()
    sdfg.validate()

    if simplify:
        sdfg.simplify()
        sdfg.validate()

    return backward_sdfg


@dataclasses.dataclass
class BackwardPass:
    """A forward SDFG and its separate backward SDFG, and the data that passes between them."""

    forward: SDFG
    backward: SDFG
    #: Data of the forward pass that the backward pass reads: forward SDFG argument -> backward SDFG argument. The
    #: caller passes what the forward SDFG wrote into the former to the backward SDFG as the latter.
    forwarded: Dict[str, str]
    #: The gradient arrays (arguments of the backward SDFG) of the differentiated inputs, by input
    input_gradients: Dict[str, str]
    #: The gradient arrays (arguments of the backward SDFG) the caller provides, by forward output
    output_gradients: Dict[str, str]


def make_backward_pass(
    sdfg: SDFG,
    outputs: List[Union[nodes.AccessNode, str]],
    inputs: List[Union[nodes.AccessNode, str]],
    data_forwarding_strategy: str = "store_all",
    data_to_recompute: Optional[List[str]] = None,
    simplify: bool = True,
    simplified: bool = False,
    recompute_forward: bool = False,
) -> BackwardPass:
    """Experimental: Creates a backward SDFG for ``sdfg`` using reverse-mode automatic differentiation, and makes the
    data the backward pass needs outputs of ``sdfg`` (which is modified in place).

    Transients the backward pass reads become non-transient. Views and scalars are copied into arrays at the end
    of the forward pass, which the backward SDFG takes as arrays.

    :param sdfg: the forward SDFG.
    :param outputs: the forward pass outputs of the function to differentiate.
    :param inputs: the inputs w.r.t. which the gradient will be returned.
    :param data_forwarding_strategy: see :func:`add_backward_pass`.
    :param data_to_recompute: see :func:`add_backward_pass`.
    :param simplify: whether to apply the simplify pass to the forward and backward SDFGs.
    :param simplified: whether ``sdfg`` was already validated, simplified, and had its conditional blocks inlined.
    :param recompute_forward: if True, the backward SDFG runs the forward pass again before the backward pass
                              (``sdfg`` differentiated in place, on a copy) and nothing is forwarded: the forward
                              SDFG is left unchanged, and the backward SDFG takes the forward SDFG's arguments.
    :return: the two SDFGs and how to call them.
    """
    if not simplified:
        sdfg.validate()
        if simplify:
            sdfg.simplify()
        _prepare_control_flow(sdfg)

    if recompute_forward:
        joint = copy.deepcopy(sdfg)
        joint.name = sdfg.name + "_backward"
        # In-place differentiation needs one scalar output: differentiate the vector-Jacobian product
        # ``sum_i sum(output_i * cotangent_i)``, whose gradients are the cotangents propagated to the inputs
        _, product, cotangents = _add_vector_jacobian_product(
            joint, [o if isinstance(o, str) else o.data for o in outputs]
        )
        result, _, _ = BackwardPassGenerator(
            sdfg=joint,
            given_gradients=[product],
            required_gradients=inputs,
            backward_sdfg=joint,
            data_forwarding_strategy=data_forwarding_strategy,
            data_to_recompute=data_to_recompute,
        ).backward()
        # The product's own gradient is the constant 1
        seed = result.given_grad_names[product]
        joint.arrays[seed].transient = True
        init = joint.add_state_before(joint.start_block, label="vjp_seed")
        init.add_mapped_tasklet(
            "vjp_seed", {"__i": "0:1"}, {}, "__out = 1", {"__out": Memlet(f"{seed}[__i]")}, external_edges=True
        )
        joint.validate()
        if simplify:
            joint.simplify()
        return BackwardPass(
            sdfg, joint, {}, {k: v for k, v in result.required_grad_names.items() if v is not None}, cotangents
        )

    backward_sdfg = SDFG(sdfg.name + "_backward")
    gen = BackwardPassGenerator(
        sdfg=sdfg,
        given_gradients=outputs,
        required_gradients=inputs,
        backward_sdfg=backward_sdfg,
        data_forwarding_strategy=data_forwarding_strategy,
        data_to_recompute=data_to_recompute,
    )
    result, _, backward_inputs = gen.backward()
    forwarded = _forward_data(sdfg, backward_sdfg, backward_inputs)
    sdfg.validate()
    backward_sdfg.validate()
    if simplify:
        sdfg.simplify()
        sdfg.validate()
    return BackwardPass(
        sdfg,
        backward_sdfg,
        forwarded,
        {k: v for k, v in result.required_grad_names.items() if v is not None},
        {k: v for k, v in result.given_grad_names.items() if v is not None},
    )


#: Values of the phase symbol of a :class:`TwoPhaseBackwardPass`
FORWARD_PHASE, BACKWARD_PHASE = 0, 1


@dataclasses.dataclass
class TwoPhaseBackwardPass:
    """
    One SDFG that runs either the forward pass or the backward pass, depending on a phase symbol. The forward phase
    records what the backward phase reads (the tape) in arrays that the caller provides to both calls.
    """

    sdfg: SDFG
    #: The symbol that selects the phase: ``FORWARD_PHASE`` or ``BACKWARD_PHASE``
    phase: str
    #: Arrays that the forward phase writes and the backward phase reads (intermediate values, control-flow decisions,
    #: and symbols); arguments of both calls
    tape: List[str]
    #: The gradient arrays (arguments of the backward phase) of the differentiated inputs, by input
    input_gradients: Dict[str, str]
    #: The gradient arrays (arguments of the backward phase) the caller provides, by forward output
    output_gradients: Dict[str, str]
    #: Arguments (containers and symbols) that each phase uses; the others may be given as None (arrays) or any value
    forward_arguments: Set[str]
    backward_arguments: Set[str]


def make_two_phase_backward_pass(
    sdfg: SDFG,
    outputs: List[Union[nodes.AccessNode, str]],
    inputs: List[Union[nodes.AccessNode, str]],
    phase: str = "autodiff_phase",
    simplify: bool = True,
    simplified: bool = False,
) -> TwoPhaseBackwardPass:
    """Experimental: Differentiates ``sdfg`` in place with reverse-mode automatic differentiation into one SDFG with
    a forward phase and a backward phase, selected by the symbol ``phase``. The forward phase runs ``sdfg``; the
    backward phase propagates the gradients of the outputs (vector-Jacobian product) to the inputs. Data and
    symbols of the forward phase that the backward phase reads become tape arrays.

    :param sdfg: the forward SDFG, which is modified in place.
    :param outputs: the forward pass outputs of the function to differentiate.
    :param inputs: the inputs w.r.t. which the gradient will be returned.
    :param phase: the name of the phase symbol.
    :param simplify: whether to apply the simplify pass.
    :param simplified: whether ``sdfg`` was already validated, simplified, and had its conditional blocks inlined.
    :return: the SDFG and how to call its phases.
    """
    if not simplified:
        sdfg.validate()
        if simplify:
            sdfg.simplify()
        _prepare_control_flow(sdfg)
    output_names = [o if isinstance(o, str) else o.data for o in outputs]

    vjp_state, product, cotangents = _add_vector_jacobian_product(sdfg, output_names)
    result, _, _ = BackwardPassGenerator(
        sdfg=sdfg, given_gradients=[product], required_gradients=inputs, backward_sdfg=sdfg
    ).backward()
    # The backward phase starts with the vector-Jacobian product, whose own gradient is the constant 1; the product
    # itself is not needed
    seed = result.given_grad_names[product]
    sdfg.arrays[seed].transient = True
    for node in list(vjp_state.nodes()):
        vjp_state.remove_node(node)
    backward_start = sdfg.add_state_before(vjp_state, label="vjp_seed")
    backward_start.add_mapped_tasklet(
        "vjp_seed", {"__i": "0:1"}, {}, "__out = 1", {"__out": Memlet(f"{seed}[__i]")}, external_edges=True
    )

    # Cut the SDFG between the phases
    for edge in sdfg.in_edges(backward_start):
        if edge.data.assignments or not edge.data.is_unconditional():
            raise AutoDiffException("Unexpected edge between the forward and the backward pass")
        sdfg.remove_edge(edge)
    forward_start = sdfg.start_block
    forward_blocks = _reachable(sdfg, forward_start)
    backward_blocks = _reachable(sdfg, backward_start)
    if forward_blocks & backward_blocks or len(forward_blocks) + len(backward_blocks) != sdfg.number_of_nodes():
        raise AutoDiffException("The forward and backward passes are not separable")

    # Tape: forward data that the backward phase reads, and symbols that the forward phase assigns and the backward
    # phase reads
    forward_accessed, forward_written = _accessed_data(sdfg, forward_blocks)
    backward_accessed, _ = _accessed_data(sdfg, backward_blocks)
    tape = []
    saved_scalars = []
    for name in sorted(forward_written & backward_accessed):
        desc = sdfg.arrays[name]
        if not desc.transient or isinstance(desc, dt.View):
            continue
        if isinstance(desc, dt.Scalar):  # Scalars are passed by value: they are copied to and from tape arrays
            saved_scalars.append(name)
        else:
            desc.transient = False
            tape.append(name)
    saved_symbols = sorted(_assigned_symbols(sdfg, forward_blocks) & _used_symbols(sdfg, backward_blocks))
    restore = {}
    symbol_tape = set()
    if saved_symbols or saved_scalars:
        save_state = sdfg.add_state("save_tape")
        for block in [b for b in forward_blocks if sdfg.out_degree(b) == 0]:
            sdfg.add_edge(block, save_state, InterstateEdge())
        forward_blocks.add(save_state)
        for symbol in saved_symbols:
            stype = sdfg.symbols.get(symbol, dtypes.int64)
            array, _ = sdfg.add_array(f"tape_{symbol}", [1], stype, find_new_name=True)
            tasklet = save_state.add_tasklet(f"save_{symbol}", {}, {"__out"}, f"__out = {symbol}")
            save_state.add_edge(tasklet, "__out", save_state.add_write(array), None, Memlet(f"{array}[0]"))
            restore[symbol] = f"{array}[0]"
            symbol_tape.add(array)
            tape.append(array)
        for name in saved_scalars:
            array, _ = sdfg.add_array(f"tape_{name}", [1], sdfg.arrays[name].dtype, find_new_name=True)
            save_state.add_nedge(save_state.add_read(name), save_state.add_write(array), Memlet(f"{array}[0]"))
            backward_start.add_nedge(
                backward_start.add_read(array), backward_start.add_write(name), Memlet(f"{array}[0]")
            )
            symbol_tape.add(array)
            tape.append(array)

    # Dispatch on the phase
    if phase not in sdfg.symbols:
        sdfg.add_symbol(phase, dtypes.int32)
    dispatch = sdfg.add_state_before(forward_start, label="phase_dispatch", is_start_block=True)
    sdfg.edges_between(dispatch, forward_start)[0].data.condition = properties.CodeBlock(f"{phase} == {FORWARD_PHASE}")
    sdfg.add_edge(
        dispatch, backward_start, InterstateEdge(condition=f"{phase} == {BACKWARD_PHASE}", assignments=restore)
    )
    backward_accessed |= symbol_tape

    # Arguments that a phase does not use may be omitted
    arguments = sdfg.arglist()
    forward_accessed |= symbol_tape
    forward_arguments = {name for name in arguments if name in forward_accessed} | _used_symbols(sdfg, forward_blocks)
    backward_arguments = {name for name in arguments if name in backward_accessed} | _used_symbols(
        sdfg, backward_blocks
    )
    for name, desc in arguments.items():
        if isinstance(desc, dt.Array) and (name not in forward_arguments or name not in backward_arguments):
            desc.optional = True
    forward_arguments = (forward_arguments & set(arguments)) | {phase}
    backward_arguments = (backward_arguments & set(arguments)) | {phase}

    sdfg.validate()
    if simplify:
        sdfg.simplify()
        sdfg.validate()
    return TwoPhaseBackwardPass(
        sdfg,
        phase,
        tape,
        {k: v for k, v in result.required_grad_names.items() if v is not None},
        cotangents,
        forward_arguments,
        backward_arguments,
    )


def _reachable(sdfg: SDFG, start: ControlFlowBlock) -> Set[ControlFlowBlock]:
    result = set()
    stack = [start]
    while stack:
        block = stack.pop()
        if block not in result:
            result.add(block)
            stack.extend(edge.dst for edge in sdfg.out_edges(block))
    return result


def _states(blocks: Iterable[ControlFlowBlock]) -> Iterable[SDFGState]:
    for block in blocks:
        if isinstance(block, SDFGState):
            yield block
        else:
            yield from block.all_states()


def _interstate_edges(sdfg: SDFG, blocks: Set[ControlFlowBlock]) -> Iterable[InterstateEdge]:
    for edge in sdfg.edges():
        if edge.src in blocks:
            yield edge.data
    for block in blocks:
        if not isinstance(block, SDFGState):
            for edge in block.all_interstate_edges():
                yield edge.data


def _accessed_data(sdfg: SDFG, blocks: Set[ControlFlowBlock]) -> Tuple[Set[str], Set[str]]:
    """The data that ``blocks`` access (including in conditions and assignments), and the data they write."""
    accessed, written = set(), set()
    for state in _states(blocks):
        for node in state.data_nodes():
            accessed.add(node.data)
            if state.in_degree(node) > 0:
                written.add(node.data)
    for edge in _interstate_edges(sdfg, blocks):
        accessed |= edge.free_symbols & sdfg.arrays.keys()
    for block in blocks:
        if not isinstance(block, SDFGState):
            accessed |= block.used_symbols(all_symbols=True) & sdfg.arrays.keys()
    return accessed, written


def _assigned_symbols(sdfg: SDFG, blocks: Set[ControlFlowBlock]) -> Set[str]:
    assigned = set()
    for edge in _interstate_edges(sdfg, blocks):
        assigned |= edge.assignments.keys()
    for block in blocks:
        if not isinstance(block, SDFGState):
            for region in block.all_control_flow_regions():
                if isinstance(region, LoopRegion) and region.loop_variable:
                    assigned.add(region.loop_variable)
    return assigned


def _used_symbols(sdfg: SDFG, blocks: Set[ControlFlowBlock]) -> Set[str]:
    used = set()
    for block in blocks:
        used |= block.used_symbols(all_symbols=True)
    for edge in _interstate_edges(sdfg, blocks):
        used |= edge.free_symbols
    return used - sdfg.arrays.keys()


def _prepare_control_flow(sdfg: SDFG):
    """
    Inlines conditional blocks but keeps loops, and turns while loops that count into for loops (the backward pass
    reverses for loops).
    """
    inline_control_flow_regions(sdfg, ignore_region_types=[LoopRegion])
    WhileToForLoop().apply_pass(sdfg, {})
    # A for loop that may exit before its end (e.g., depending on data) is reversed from the final value of its loop
    # variable, which the edges after it record
    for loop in list(sdfg.all_control_flow_regions(recursive=True)):
        if isinstance(loop, LoopRegion) and loop.loop_variable and ad_utils.may_exit_early(loop):
            if ad_utils.loop_exit_symbol(loop) is not None:
                continue
            graph = loop.parent_graph
            if graph.out_degree(loop) == 0:
                graph.add_state_after(loop, label=f"{loop.label}_exit")
            name = loop.sdfg.find_new_symbol(f"{loop.loop_variable}_exit")
            loop.sdfg.add_symbol(name, loop.sdfg.symbols.get(loop.loop_variable, dtypes.int64))
            for edge in graph.out_edges(loop):
                edge.data.assignments[name] = loop.loop_variable
    for region in sdfg.all_control_flow_regions(recursive=True):
        if region.has_cycles():
            raise AutoDiffException(
                f"{region.label} contains a loop that is not a loop region (e.g., a loop with a "
                "break, or one that exits depending on data); such loops cannot be differentiated"
            )


def _add_vector_jacobian_product(sdfg: SDFG, outputs: List[str]):
    """Adds ``sum_i sum(output_i * cotangent_i)`` at the end of ``sdfg``, with a new input array per cotangent.

    :return: the state that computes the product, the name of the (transient, shape ``(1,)``) product, and the
             cotangent array of every output.
    """
    sinks = sdfg.sink_nodes()
    if len(sinks) != 1:
        raise AutoDiffException("Cannot differentiate an SDFG with several sink blocks")
    state = sdfg.add_state_after(sinks[0], label="vector_jacobian_product")
    dtype = sdfg.arrays[outputs[0]].dtype
    product, _ = sdfg.add_array("vjp", [1], dtype, transient=True, find_new_name=True)
    cotangents: Dict[str, str] = {}
    partials = []
    for output in outputs:
        desc = sdfg.arrays[output]
        cotangent, _ = sdfg.add_array(f"cotangent_{output}", desc.shape, desc.dtype, find_new_name=True)
        cotangents[output] = cotangent
        terms, _ = sdfg.add_array(f"vjp_terms_{output}", desc.shape, desc.dtype, transient=True, find_new_name=True)
        partial, _ = sdfg.add_array(f"vjp_{output}", [1], dtype, transient=True, find_new_name=True)
        index = ", ".join(f"__i{d}" for d in range(len(desc.shape)))
        terms_node = state.add_access(terms)
        state.add_mapped_tasklet(
            f"{output}_vjp_terms",
            {f"__i{d}": f"0:{s}" for d, s in enumerate(desc.shape)},
            {"__a": Memlet(f"{output}[{index}]"), "__b": Memlet(f"{cotangent}[{index}]")},
            "__out = __a * __b",
            {"__out": Memlet(f"{terms}[{index}]")},
            external_edges=True,
            output_nodes={terms: terms_node},
        )
        reduce = Reduce("sum", wcr="lambda a, b: a + b", axes=None, identity=0)
        state.add_node(reduce)
        state.add_edge(terms_node, None, reduce, None, Memlet.from_array(terms, sdfg.arrays[terms]))
        partial_node = state.add_access(partial)
        state.add_edge(reduce, None, partial_node, None, Memlet(f"{partial}[0]"))
        partials.append(partial_node)
    inputs = {f"__p{k}": Memlet(f"{node.data}[0]") for k, node in enumerate(partials)}
    total = state.add_tasklet("vjp_sum", set(inputs), {"__out"}, "__out = " + " + ".join(inputs))
    for (connector, memlet), node in zip(inputs.items(), partials):
        state.add_edge(node, None, total, connector, memlet)
    state.add_edge(total, "__out", state.add_write(product), None, Memlet(f"{product}[0]"))
    return state, product, cotangents


def _forward_data(forward: SDFG, backward: SDFG, backward_inputs: Dict[str, dt.Data]) -> Dict[str, str]:
    """Makes the forward-pass data the backward SDFG reads available to it (see :func:`make_backward_pass`).

    :return: forward SDFG argument -> backward SDFG argument.
    """
    forwarded: Dict[str, str] = {}
    copies = []
    for name in backward_inputs:
        if name not in forward.arrays:
            raise AutoDiffException(f"The backward pass reads {name}, which the forward pass does not have")
        desc = forward.arrays[name]
        if isinstance(desc, dt.View) or isinstance(desc, dt.Scalar):
            # Views only exist within a state and scalars cannot be returned: copy into an array at the end
            copies.append(name)
        else:
            desc.transient = False
            forwarded[name] = name

    if copies:
        sinks = forward.sink_nodes()
        if len(sinks) != 1:
            raise AutoDiffException("Cannot forward views or scalars from an SDFG with several sink blocks")
        state = forward.add_state_after(sinks[0], label="forward_data")
        for name in copies:
            desc = forward.arrays[name]
            shape = (1,) if isinstance(desc, dt.Scalar) else desc.shape
            # A view keeps its layout (e.g., transposed strides), which nested SDFGs of the backward pass assume
            strides = desc.strides if isinstance(desc, dt.View) else None
            out_name, _ = forward.add_array(
                f"{name}_forwarded",
                shape,
                desc.dtype,
                storage=desc.storage,
                strides=strides,
                total_size=_extent(shape, strides) if strides else None,
                transient=False,
                find_new_name=True,
            )
            source = _reconstruct_view(forward, state, name) if isinstance(desc, dt.View) else state.add_read(name)
            state.add_nedge(
                source,
                state.add_write(out_name),
                Memlet(
                    data=name,
                    subset=", ".join(f"0:{s}" for s in desc.shape) or "0",
                    other_subset=", ".join(f"0:{s}" for s in shape),
                ),
            )
            # The backward pass takes the copy as a plain array
            bwd_desc = backward.arrays[name]
            if isinstance(bwd_desc, dt.View):
                backward.arrays[name] = dt.Array(
                    bwd_desc.dtype,
                    bwd_desc.shape,
                    storage=bwd_desc.storage,
                    strides=bwd_desc.strides,
                    total_size=_extent(bwd_desc.shape, bwd_desc.strides),
                )
                forwarded[out_name] = name
            else:  # A scalar: copy into it at the start of the backward pass
                in_name, _ = backward.add_array(out_name, shape, desc.dtype, storage=desc.storage, transient=False)
                bwd_desc.transient = True
                start = backward.add_state_before(backward.start_block, label="backward_data")
                start.add_nedge(start.add_read(in_name), start.add_write(name), Memlet(f"{in_name}[0]"))
                forwarded[out_name] = in_name
    return forwarded


def _extent(shape, strides):
    """The number of elements a strided array spans."""
    return sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1


def _reconstruct_view(sdfg: SDFG, target: SDFGState, name: str) -> nodes.AccessNode:
    """Adds the view ``name`` (and the views it views) to ``target``, as defined in another state of ``sdfg``."""
    for state in sdfg.states():
        if state is target:
            continue
        for node in state.data_nodes():
            if node.data != name:
                continue
            edge = sdutils.get_view_edge(state, node)
            if edge is None:
                continue
            viewed = edge.src if edge.dst is node else edge.dst
            if isinstance(sdfg.arrays[viewed.data], dt.View):
                source = _reconstruct_view(sdfg, target, viewed.data)
            else:
                source = target.add_read(viewed.data)
            view = target.add_access(name)
            view.add_in_connector("views")
            target.add_edge(source, None, view, "views", copy.deepcopy(edge.data))
            return view
    raise AutoDiffException(f"Cannot find the definition of view {name}")
