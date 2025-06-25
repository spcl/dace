import typing

from dace.autodiff.backward_pass_generator import BackwardPassGenerator

from dace.sdfg import SDFG, nodes
from dace.autodiff.optimize_backward_pass_generator import autooptimize_sdfgs_for_ad, preprocess_fwd_sdfg, fuse_states_cav
from dace.sdfg.utils import inline_control_flow_regions
from dace.sdfg.state import LoopRegion


def add_backward_pass(sdfg: SDFG,
                      outputs: typing.List[typing.Union[nodes.AccessNode,
                                                        str]],
                      inputs: typing.List[typing.Union[nodes.AccessNode, str]],
                      overwite_strategy: str = "store_all",
                      data_to_recompute: typing.List[str] = None,
                      autooptimize: bool = False,
                      seprate_sdfgs=False):
    """ Experimental: Add a backward pass to `state` using reverse-mode automatic differentiation.

        ``inputs``, ``outputs`` and ``grads`` can be provided either as ``AccessNode`` nodes, or as ``str``, in which
        case the graph will be searched for exactly one matching ``AccessNode`` with data matching the ``str``.

        The SDFG should not contain any inplace operations. It may contain the following nodes:

        * Maps
        * AccessNodes
        * Reductions (Sum, Min, Max)
        * ONNXOps
        * NestedSDFGs containing a single SDFGState (subject to the same constraints). NestedSDFGs may contain multiple
          states as long as all other states are only used for zero initialization.

        When differentiating an :class:`~dace.libraries.onnx.nodes.onnx_op.ONNXOp`, the ONNXBackward registry will be checked
        for any matching backward pass implementations. If none are found, the ONNXForward registry will be checked for
        matching pure implementations. If one is found, symbolic differentiation of the pure implementation will be
        attempted. If this fails, or no pure forward implementation is found, the method will fail.


        :param sdfg: the parent SDFG of ``state``.
        :param state: the state to add the backward pass to. This is also the state of the forward pass.
        :param outputs: the forward pass outputs of the function to differentiate.
        :param inputs: the inputs w.r.t. which the gradient will be returned.
    """
    # Validate SDFG
    sdfg.validate()

    # preprocess the SDFG
    if "go_fast" in sdfg.name:
        preprocess_fwd_sdfg(sdfg)

    # Simplify and validate
    sdfg.validate()
    sdfg.simplify()

    # Inline conditional blocks but keep loops
    inline_control_flow_regions(sdfg, ignore_region_types=[LoopRegion])

    if seprate_sdfgs:
        backward_sdfg = SDFG(sdfg.name + "_backward")
    else:
        backward_sdfg = sdfg

    # Add backward pass
    gen = BackwardPassGenerator(sdfg=sdfg,
                                given_gradients=outputs,
                                required_gradients=inputs,
                                backward_sdfg=backward_sdfg,
                                overwrite_strategy=overwite_strategy,
                                data_to_recompute=data_to_recompute)
    gen.backward()
    if autooptimize:
        autooptimize_sdfgs_for_ad(gen)
    sdfg.validate()

    if seprate_sdfgs:
        return backward_sdfg
