# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Backward (gradient) implementations for the ATen library nodes of the TorchDynamo frontend
(:mod:`dace.libraries.torch.nodes.aten`).

Currently this provides the backward pass of :class:`~dace.libraries.torch.NativeDropout`. Because the forward dropout
draws its mask from a counter-based (Philox) generator keyed by ``seed`` and ``offset``, the backward does not need the
forward mask saved: it recomputes it. Applying dropout with the *same* ``seed``/``offset``/``p`` to the output gradient
reproduces ``mask * grad / (1 - p)``, which is exactly the gradient of the input. The backward is therefore the same
:class:`dace.libraries.ml.Dropout` node applied to the incoming gradient (``_x = grad_output``, ``_y = grad_input``),
with no mask output.
"""

import copy

import dace
import dace.autodiff.utils as butils
from dace import nodes as nd
from dace import symbolic
from dace.autodiff.base_abc import BackwardContext, BackwardImplementation, BackwardResult
from dace.libraries.torch.nodes.aten import NativeDropout, as_symbolic
from dace.registry import autoregister_params

#: Output connector carrying the dropout result (``output_1`` is the mask, which has no gradient)
_OUTPUT = "output_0"
#: Input connector whose gradient is produced
_INPUT = "input"
#: Connector names of the backward nested SDFG (local, so they do not clash with ``__return`` tuple naming)
_OUTPUT_GRAD = "output_grad"
_INPUT_GRAD = "input_grad"


@autoregister_params(node_type=NativeDropout, name="recompute")
class NativeDropoutBackward(BackwardImplementation):
    """
    Backward pass of :class:`~dace.libraries.torch.NativeDropout` that recomputes the mask (see the module docstring).

    With ``train=False`` (or ``p == 0``) the forward is the identity, so the backward copies the output gradient into
    the input gradient. The mask output of the forward node is never required.
    """

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return isinstance(node, NativeDropout)

    @staticmethod
    def backward(
        forward_node: nd.Node,
        context: BackwardContext,
        given_gradients: list[str | None],
        required_gradients: list[str | None],
    ) -> tuple[nd.Node, BackwardResult]:
        if _OUTPUT not in given_gradients:
            raise ValueError(f"NativeDropout backward expects a gradient for '{_OUTPUT}', got {given_gradients}")
        if _INPUT not in required_gradients:
            raise ValueError(f"NativeDropout backward has nothing to differentiate (required {required_gradients})")

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        # Local connector arrays (named so they never match the ``__return_<i>`` tuple-return pattern).
        output_grad_desc = copy.deepcopy(butils.forward_out_desc_with_name(forward_node, context, _OUTPUT))
        output_grad_desc.transient = False
        input_grad_desc = copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, _INPUT))
        input_grad_desc.transient = False
        nsdfg.add_datadesc(_OUTPUT_GRAD, output_grad_desc)
        nsdfg.add_datadesc(_INPUT_GRAD, input_grad_desc)

        result = BackwardResult.empty()
        result.given_grad_names[_OUTPUT] = _OUTPUT_GRAD
        result.required_grad_names[_INPUT] = _INPUT_GRAD

        p = as_symbolic(forward_node.arguments.get("p", 0.5))
        train = forward_node.arguments.get("train", True)
        is_identity = train is not None and not train

        symbol_mapping = butils.backward_symbol_mapping(nsdfg, context.backward_state)

        if is_identity or (not symbolic.issymbolic(p) and float(p) == 0.0):
            # The forward was the identity: the gradient passes straight through.
            nstate.add_nedge(
                nstate.add_read(_OUTPUT_GRAD), nstate.add_write(_INPUT_GRAD), nsdfg.make_array_memlet(_OUTPUT_GRAD)
            )
        else:
            from dace.libraries import ml  # Deferred: only needed to build the backward, not to load SDFGs

            drop = ml.Dropout(forward_node.name + "_backward_dropout")
            drop.p = p
            drop.seed = forward_node.seed
            drop.offset = forward_node.offset
            nstate.add_node(drop)
            drop.add_in_connector("_x")
            drop.add_out_connector("_y")
            nstate.add_edge(nstate.add_read(_OUTPUT_GRAD), None, drop, "_x", nsdfg.make_array_memlet(_OUTPUT_GRAD))
            nstate.add_edge(drop, "_y", nstate.add_write(_INPUT_GRAD), None, nsdfg.make_array_memlet(_INPUT_GRAD))

            # The Philox parameters are symbolic properties of the Dropout node, not yet visible as free symbols of the
            # nested SDFG (it is unexpanded): map each one from the backward SDFG, declaring it there if necessary so the
            # forward and backward share the same ``dace_random_seed`` value.
            for expr in (forward_node.seed, forward_node.offset, p):
                for sym in symbolic.pystr_to_symbolic(expr).free_symbols:
                    name = str(sym)
                    if name in symbol_mapping:
                        continue
                    dtype = context.backward_sdfg.symbols.get(name) or getattr(sym, "dtype", dace.int64)
                    if name not in context.backward_sdfg.symbols:
                        context.backward_sdfg.add_symbol(name, dtype)
                    if name not in nsdfg.symbols:
                        nsdfg.add_symbol(name, dtype)
                    symbol_mapping[name] = butils.connector_symbol(name, dtype)

        node = context.backward_state.add_nested_sdfg(
            nsdfg, {_OUTPUT_GRAD}, {_INPUT_GRAD}, symbol_mapping=symbol_mapping
        )
        return node, result
