# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dataclasses
from typing import Dict, List, Union, Optional

from dace import Memlet, data as dt
from dace.autodiff.backward_pass_generator import BackwardPassGenerator
from dace.autodiff.base_abc import AutoDiffException
from dace.libraries.standard import Reduce

from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutils
from dace.sdfg.utils import inline_control_flow_regions
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.while_to_for_loop import WhileToForLoop


def add_backward_pass(sdfg: SDFG,
                      outputs: List[Union[nodes.AccessNode, str]],
                      inputs: List[Union[nodes.AccessNode, str]],
                      data_forwarding_strategy: str = "store_all",
                      data_to_recompute: Optional[List[str]] = None,
                      simplify: bool = True,
                      separate_sdfgs: bool = False) -> Optional[SDFG]:
    """ Experimental: Add a backward pass to `state` using reverse-mode automatic differentiation.

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
        return make_backward_pass(sdfg,
                                  outputs,
                                  inputs,
                                  data_forwarding_strategy,
                                  data_to_recompute,
                                  simplify=simplify,
                                  simplified=True).backward
    backward_sdfg = sdfg

    # Add backward pass
    gen = BackwardPassGenerator(sdfg=sdfg,
                                given_gradients=outputs,
                                required_gradients=inputs,
                                backward_sdfg=backward_sdfg,
                                data_forwarding_strategy=data_forwarding_strategy,
                                data_to_recompute=data_to_recompute)
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


def make_backward_pass(sdfg: SDFG,
                       outputs: List[Union[nodes.AccessNode, str]],
                       inputs: List[Union[nodes.AccessNode, str]],
                       data_forwarding_strategy: str = "store_all",
                       data_to_recompute: Optional[List[str]] = None,
                       simplify: bool = True,
                       simplified: bool = False,
                       recompute_forward: bool = False) -> BackwardPass:
    """ Experimental: Creates a backward SDFG for ``sdfg`` using reverse-mode automatic differentiation, and makes the
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
        product, cotangents = _add_vector_jacobian_product(joint,
                                                           [o if isinstance(o, str) else o.data for o in outputs])
        result, _, _ = BackwardPassGenerator(sdfg=joint,
                                             given_gradients=[product],
                                             required_gradients=inputs,
                                             backward_sdfg=joint,
                                             data_forwarding_strategy=data_forwarding_strategy,
                                             data_to_recompute=data_to_recompute).backward()
        # The product's own gradient is the constant 1
        seed = result.given_grad_names[product]
        joint.arrays[seed].transient = True
        init = joint.add_state_before(joint.start_block, label='vjp_seed')
        init.add_mapped_tasklet('vjp_seed', {'__i': '0:1'}, {},
                                '__out = 1', {'__out': Memlet(f'{seed}[__i]')},
                                external_edges=True)
        joint.validate()
        if simplify:
            joint.simplify()
        return BackwardPass(sdfg, joint, {}, {
            k: v
            for k, v in result.required_grad_names.items() if v is not None
        }, cotangents)

    backward_sdfg = SDFG(sdfg.name + "_backward")
    gen = BackwardPassGenerator(sdfg=sdfg,
                                given_gradients=outputs,
                                required_gradients=inputs,
                                backward_sdfg=backward_sdfg,
                                data_forwarding_strategy=data_forwarding_strategy,
                                data_to_recompute=data_to_recompute)
    result, _, backward_inputs = gen.backward()
    forwarded = _forward_data(sdfg, backward_sdfg, backward_inputs)
    sdfg.validate()
    backward_sdfg.validate()
    if simplify:
        sdfg.simplify()
        sdfg.validate()
    return BackwardPass(sdfg, backward_sdfg, forwarded, {
        k: v
        for k, v in result.required_grad_names.items() if v is not None
    }, {
        k: v
        for k, v in result.given_grad_names.items() if v is not None
    })


def _prepare_control_flow(sdfg: SDFG):
    """
    Inlines conditional blocks but keeps loops, and turns while loops that count into for loops (the backward pass
    reverses for loops).
    """
    inline_control_flow_regions(sdfg, ignore_region_types=[LoopRegion])
    WhileToForLoop().apply_pass(sdfg, {})
    for region in sdfg.all_control_flow_regions(recursive=True):
        if region.has_cycles():
            raise AutoDiffException(f'{region.label} contains a loop that is not a loop region (e.g., a loop with a '
                                    'break, or one that exits depending on data); such loops cannot be differentiated')


def _add_vector_jacobian_product(sdfg: SDFG, outputs: List[str]):
    """ Adds ``sum_i sum(output_i * cotangent_i)`` at the end of ``sdfg``, with a new input array per cotangent.

        :return: the name of the (transient, shape ``(1,)``) product, and the cotangent array of every output.
    """
    sinks = sdfg.sink_nodes()
    if len(sinks) != 1:
        raise AutoDiffException('Cannot differentiate an SDFG with several sink blocks')
    state = sdfg.add_state_after(sinks[0], label='vector_jacobian_product')
    dtype = sdfg.arrays[outputs[0]].dtype
    product, _ = sdfg.add_array('vjp', [1], dtype, transient=True, find_new_name=True)
    cotangents: Dict[str, str] = {}
    partials = []
    for output in outputs:
        desc = sdfg.arrays[output]
        cotangent, _ = sdfg.add_array(f'cotangent_{output}', desc.shape, desc.dtype, find_new_name=True)
        cotangents[output] = cotangent
        terms, _ = sdfg.add_array(f'vjp_terms_{output}', desc.shape, desc.dtype, transient=True, find_new_name=True)
        partial, _ = sdfg.add_array(f'vjp_{output}', [1], dtype, transient=True, find_new_name=True)
        index = ', '.join(f'__i{d}' for d in range(len(desc.shape)))
        terms_node = state.add_access(terms)
        state.add_mapped_tasklet(f'{output}_vjp_terms', {
            f'__i{d}': f'0:{s}'
            for d, s in enumerate(desc.shape)
        }, {
            '__a': Memlet(f'{output}[{index}]'),
            '__b': Memlet(f'{cotangent}[{index}]')
        },
                                 '__out = __a * __b', {'__out': Memlet(f'{terms}[{index}]')},
                                 external_edges=True,
                                 output_nodes={terms: terms_node})
        reduce = Reduce('sum', wcr='lambda a, b: a + b', axes=None, identity=0)
        state.add_node(reduce)
        state.add_edge(terms_node, None, reduce, None, Memlet.from_array(terms, sdfg.arrays[terms]))
        partial_node = state.add_access(partial)
        state.add_edge(reduce, None, partial_node, None, Memlet(f'{partial}[0]'))
        partials.append(partial_node)
    inputs = {f'__p{k}': Memlet(f'{node.data}[0]') for k, node in enumerate(partials)}
    total = state.add_tasklet('vjp_sum', set(inputs), {'__out'}, '__out = ' + ' + '.join(inputs))
    for (connector, memlet), node in zip(inputs.items(), partials):
        state.add_edge(node, None, total, connector, memlet)
    state.add_edge(total, '__out', state.add_write(product), None, Memlet(f'{product}[0]'))
    return product, cotangents


def _forward_data(forward: SDFG, backward: SDFG, backward_inputs: Dict[str, dt.Data]) -> Dict[str, str]:
    """ Makes the forward-pass data the backward SDFG reads available to it (see :func:`make_backward_pass`).

        :return: forward SDFG argument -> backward SDFG argument.
    """
    forwarded: Dict[str, str] = {}
    copies = []
    for name in backward_inputs:
        if name not in forward.arrays:
            raise AutoDiffException(f'The backward pass reads {name}, which the forward pass does not have')
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
            raise AutoDiffException('Cannot forward views or scalars from an SDFG with several sink blocks')
        state = forward.add_state_after(sinks[0], label='forward_data')
        for name in copies:
            desc = forward.arrays[name]
            shape = (1, ) if isinstance(desc, dt.Scalar) else desc.shape
            # A view keeps its layout (e.g., transposed strides), which nested SDFGs of the backward pass assume
            strides = desc.strides if isinstance(desc, dt.View) else None
            out_name, _ = forward.add_array(f'{name}_forwarded',
                                            shape,
                                            desc.dtype,
                                            storage=desc.storage,
                                            strides=strides,
                                            total_size=_extent(shape, strides) if strides else None,
                                            transient=False,
                                            find_new_name=True)
            source = _reconstruct_view(forward, state, name) if isinstance(desc, dt.View) else state.add_read(name)
            state.add_nedge(
                source, state.add_write(out_name),
                Memlet(data=name,
                       subset=', '.join(f'0:{s}' for s in desc.shape) or '0',
                       other_subset=', '.join(f'0:{s}' for s in shape)))
            # The backward pass takes the copy as a plain array
            bwd_desc = backward.arrays[name]
            if isinstance(bwd_desc, dt.View):
                backward.arrays[name] = dt.Array(bwd_desc.dtype,
                                                 bwd_desc.shape,
                                                 storage=bwd_desc.storage,
                                                 strides=bwd_desc.strides,
                                                 total_size=_extent(bwd_desc.shape, bwd_desc.strides))
                forwarded[out_name] = name
            else:  # A scalar: copy into it at the start of the backward pass
                in_name, _ = backward.add_array(out_name, shape, desc.dtype, storage=desc.storage, transient=False)
                bwd_desc.transient = True
                start = backward.add_state_before(backward.start_block, label='backward_data')
                start.add_nedge(start.add_read(in_name), start.add_write(name), Memlet(f'{in_name}[0]'))
                forwarded[out_name] = in_name
    return forwarded


def _extent(shape, strides):
    """The number of elements a strided array spans."""
    return sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1


def _reconstruct_view(sdfg: SDFG, target: SDFGState, name: str) -> nodes.AccessNode:
    """ Adds the view ``name`` (and the views it views) to ``target``, as defined in another state of ``sdfg``. """
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
            view.add_in_connector('views')
            target.add_edge(source, None, view, 'views', copy.deepcopy(edge.data))
            return view
    raise AutoDiffException(f'Cannot find the definition of view {name}')
