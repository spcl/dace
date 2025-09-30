from typing import List, Union, Optional

from dace.autodiff.backward_pass_generator import BackwardPassGenerator

from dace.sdfg import SDFG, nodes
from dace.sdfg.utils import inline_control_flow_regions
from dace.sdfg.state import LoopRegion


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
        * NestedSDFGs (subject to the same constraints)

        When differentiating an :class:`~dace.libraries.onnx.nodes.onnx_op.ONNXOp`, the ONNXBackward registry will be checked
        for any matching backward pass implementations. If none are found, the ONNXForward registry will be checked for
        matching pure implementations. If one is found, symbolic differentiation of the pure implementation will be
        attempted. If this fails, or no pure forward implementation is found, the method will fail.


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
        :param separate_sdfgs: whether to create a separate SDFG for the backward pass.
        :return: the backward SDFG if separate_sdfgs is True, the original SDFG (which now also contains the backward pass) otherwise.
    """
    # Validate the SDFG
    sdfg.validate()

    if simplify:
        sdfg.simplify()

    # Inline conditional blocks but keep loops
    inline_control_flow_regions(sdfg, ignore_region_types=[LoopRegion])

    if separate_sdfgs:
        backward_sdfg = SDFG(sdfg.name + "_backward")
    else:
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
