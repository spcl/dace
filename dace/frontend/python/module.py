""" DaCe Python parsing functionality and entry point to Python frontend. """
from dataclasses import dataclass
import collections
import itertools
import tempfile
import logging
import copy
import os
from typing import Any, Callable, Dict, OrderedDict, List, Optional, Set, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.onnx import TrainingMode

import onnx

import dace
from dace import data
from dace.codegen import compiled_sdfg
from dace.sdfg import SDFG, nodes
from dace.frontend.python import common as pycommon

from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.torch import dispatchers
from dace.autodiff import torch as torch_autodiff
from dace.autodiff.library import library as autodiff_library
from dace.libraries.onnx import ONNXModel
from dace.util import find_str_not_in_set, auto_optimize as auto_opt
from dace.codegen import compiled_sdfg

log = logging.getLogger(__name__)


def _onnx_delete_initializers(model: onnx.ModelProto, names: Set[str]):
    """
    Delete the given initializers from the given onnx model.
    Operates inplace
    """
    to_remove = []
    for i, initializer in enumerate(model.graph.initializer):
        if initializer.name in names:
            to_remove.append(i)

    for i in reversed(to_remove):
        model.graph.initializer.pop(i)


class DaceModule(nn.Module, pycommon.SDFGConvertible):
    """ A wrapper that converts a PyTorch ``nn.Module`` to a PyTorch compatible data-centric ``nn.Module``.

        :param module: the model to wrap.
        :param dummy_inputs: a tuple of tensors to use as input when tracing ``model``.
        :param cuda: if ``True``, the module will execute using CUDA. If ``None``, it will be detected from the
                     ``module``.
        :param training: whether to use train mode when tracing ``model``.
        :param backward: whether to enable the backward pass.
        :param inputs_to_skip: if provided, a list of inputs to skip computing gradients for. 
                               (only relevant when the backward pass is enabled)
        :param onnx_simplify: whether to apply onnx simplification using onnxsim.
        :param simplify: whether to apply simplification transforms after conversion (this generally improves performance,
                         but can be slow).
        :param sdfg_name: the name to give to the sdfg (defaults to ``dace_model``).
        :param auto_optimize: whether to apply automatic optimizations.
        :param compile_torch_extension: if True, a torch C++ extension will be compiled and used for this module.
                                        Otherwise, a python ctypes implementation will be used.
        :param debug_transients: if True, the module will have all transients as outputs.

        :Example:
            >>> from dace.frontend.module import DaceModule
            >>> class MyModule(nn.Module):
            ...     def forward(self, x):
            ...        x = torch.log(x)
            ...        x = torch.sqrt(x)
            ...        return x
            >>> module = MyModule()
            >>> module(torch.ones(2))
            tensor([0., 0.])
            >>> dace_module = DaceModule(module)
            >>> dace_module(torch.ones(2))
            tensor([0., 0.])
    """

    def __init__(self,
                 module: nn.Module,
                 dummy_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
                 cuda: Optional[bool] = None,
                 training: bool = False,
                 backward=False,
                 inputs_to_skip: Optional[List[str]] = None,
                 onnx_simplify: bool = True,
                 simplify: bool = True,
                 auto_optimize: bool = False,
                 debug_transients: bool = False,
                 compile_torch_extension: bool = True,
                 sdfg_name: Optional[str] = None):

        super(DaceModule, self).__init__()

        self.backward = backward
        self.model = module
        self.dace_model: Optional[ONNXModel] = None
        self.training = training
        self.sdfg: Optional[SDFG] = None
        self.use_cuda = cuda
        self.sdfg_name = sdfg_name or type(module).__name__
        self.auto_optimize = auto_optimize
        self.onnx_simplify = onnx_simplify
        self.simplify = simplify
        self.debug_transients = debug_transients
        self.compile_torch_extension = compile_torch_extension
        self.inputs_to_skip = inputs_to_skip or []

        self.function = None

        #: hooks that are executed after onnx graph is imported to an SDFG
        self.post_onnx_hooks: OrderedDict[str, Callable[
            [DaceModule], None]] = collections.OrderedDict()

        #: hooks that are executed after the backpropagation sdfg has been created
        self.post_autodiff_hooks: OrderedDict[str, Callable[
            [SDFG, SDFG], None]] = collections.OrderedDict()

        #: hooks that are executed after the sdfg is compiled
        self.post_compile_hooks: OrderedDict[str, Callable[
            [compiled_sdfg.CompiledSDFG], None]] = collections.OrderedDict()

        if dummy_inputs is not None:
            self.function = self._initialize_sdfg(dummy_inputs)

        # setup debug hook
        if self.debug_transients:

            def transients_outputs(module):
                for state in module.sdfg.nodes():
                    for node in state.nodes():
                        if (isinstance(node, nodes.AccessNode)
                                and node.desc(module.sdfg).transient
                                and not isinstance(node.desc(module.sdfg),
                                                   data.Scalar)):
                            if "mean" not in node.data and "std" not in node.data:
                                module.dace_model.outputs.append(node.data)
                                node.desc(module.sdfg).transient = False

            self.prepend_post_onnx_hook("make_transients_outputs",
                                        transients_outputs)

        # setup optimization hooks
        if self.auto_optimize:
            if self.backward:

                def auto_optimize_backward(fwd_sdfg, bwd_sdfg):
                    auto_opt(fwd_sdfg, self.use_cuda, simplify=self.simplify)
                    auto_opt(bwd_sdfg, self.use_cuda, simplify=self.simplify)

                self.append_post_autodiff_hook("auto_optimize",
                                               auto_optimize_backward)
            else:
                self.append_post_onnx_hook(
                    "auto_optimize",
                    lambda dace_module: auto_opt(dace_module.dace_model.sdfg,
                                                 self.use_cuda,
                                                 simplify=self.simplify))
        elif self.simplify:
            if self.backward:

                def simplify(fwd_sdfg, bwd_sdfg):
                    fwd_sdfg.simplify()
                    bwd_sdfg.simplify()

                self.append_post_autodiff_hook("simplify", simplify)
            else:
                self.append_post_onnx_hook(
                    "simplify",
                    lambda dace_module: dace_module.sdfg.simplify())

    def reset_sdfg(self):
        """ Clear the sdfg so that optimizations are reapplied. """
        self.function = None

    def prepend_post_onnx_hook(self, name: str, func: Callable[["DaceModule"],
                                                               None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_onnx_hooks), name)
        self.post_onnx_hooks[name] = func
        self.post_onnx_hooks.move_to_end(name, last=False)

    def append_post_onnx_hook(self, name: str, func: Callable[["DaceModule"],
                                                              None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_onnx_hooks), name)
        self.post_onnx_hooks[name] = func

    def prepend_post_autodiff_hook(self, name: str,
                                   func: Callable[[SDFG, SDFG], None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_autodiff_hooks), name)
        self.post_autodiff_hooks[name] = func
        self.post_autodiff_hooks.move_to_end(name, last=False)

    def append_post_autodiff_hook(self, name: str, func: Callable[[SDFG, SDFG],
                                                                  None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_autodiff_hooks), name)
        self.post_autodiff_hooks[name] = func

    def prepend_post_compile_hook(
            self,
            name: str,
            func: Callable[
                [compiled_sdfg.CompiledSDFG],  # type: ignore
                None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_compile_hooks), name)
        self.post_compile_hooks[name] = func
        self.post_compile_hooks.move_to_end(name, last=False)

    def append_post_compile_hook(
            self,
            name: str,
            func: Callable[
                [compiled_sdfg.CompiledSDFG],  # type: ignore
                None]):
        if self.function is not None:
            log.warning(
                f"Added a hook after the model was already initialized. This hook "
                f"(with name {name}) will not be executed!")
        name = find_str_not_in_set(set(self.post_compile_hooks), name)
        self.post_compile_hooks[name] = func

    def _initialize_sdfg(self, dummy_inputs):

        # determine whether we are using CUDA
        if self.use_cuda is None:
            try:
                module_is_cuda = next(iter(dummy_inputs)).is_cuda
            except StopIteration:
                module_is_cuda = False

            if not module_is_cuda:
                # check the parameters
                try:
                    module_is_cuda = next(self.model.parameters()).is_cuda
                except StopIteration:
                    module_is_cuda = False
            self.use_cuda = module_is_cuda

        if self.use_cuda:
            self.model = self.model.cuda()

        # TODO change to StringIO if not too big
        with tempfile.TemporaryDirectory() as dir_name:
            export_name = os.path.join(dir_name, "export.onnx")

            # save the state of the model, and restore it after tracing
            state = copy.deepcopy(self.state_dict())
            torch.onnx.export(
                self.model,
                dummy_inputs,
                export_name,
                verbose=logging.root.level <= logging.DEBUG,
                # Some models will require training even when we don't want to train:
                # when training is set to EVAL, pytorch currently performs an optimization pass ("onnx_eval_peephole")
                # that renames weights and thus breaks the model in some settings.
                training=(TrainingMode.TRAINING
                          if self.training else TrainingMode.EVAL),
                # opset_version=12,
                export_params=not self.backward,
                # pytorch constant folding will add new unnamed inputs to the graph and remove some of the
                # named parameters of the model: this means that we can't match with the state dict
                # anymore, so we disable this. Our CF is more flexible.
                do_constant_folding=False,
                keep_initializers_as_inputs=True)
            self.load_state_dict(state)
            onnx_model_exported = onnx.load(export_name)

            # Remove buffers and parameters from initializers
            # they should already be in the inputs (from the pytorch exporter)
            # this prevents onnx tools from messing with parameters
            input_names = set()
            for name, _ in itertools.chain(self.named_parameters(),
                                           self.named_buffers()):
                # pytorch adds a "model." prefix here that isn't in the onnx export;
                # remove it
                if not name.startswith("model."):
                    raise ValueError(
                        "Expected parameter names to start with 'model.'")
                input_names.add(name[6:])

            # save the parameters as they are now for later access
            self._exported_parameters = dict(
                (n, p) for n, p in itertools.chain(
                    self.model.named_parameters(), self.model.named_buffers()))

            _onnx_delete_initializers(onnx_model_exported, input_names)

            # load using importer
            dace_model = ONNXModel(self.sdfg_name,
                                   onnx_model_exported,
                                   onnx_simplify=self.onnx_simplify,
                                   cuda=self.use_cuda,
                                   auto_optimize=self.auto_optimize)
            self.sdfg = dace_model.sdfg
            self.dace_model = dace_model

            self.sdfg.validate()

            for _, hook in self.post_onnx_hooks.items():
                hook(self)

            # choose the backend that will generate the function to call during
            # forward
            if self.compile_torch_extension:
                function_generator = dispatchers.register_and_compile_torch_extension
            else:
                function_generator = dispatchers.get_ctypes_dispatcher

            if self.backward:

                # Determine what grads we need
                # For now: we want gradients for all inputs that are not pytorch buffers
                named_buffers = {n for n, _ in self.model.named_buffers()}
                required_gradients = [
                    clean_onnx_name(name) for name in self.dace_model.inputs
                    if name not in named_buffers
                    and name not in self.inputs_to_skip
                ]
                named_parameters = dict(self.model.named_parameters())
                required_gradients.extend(
                    clean_onnx_name(name)
                    for name, param in named_parameters.items()
                    if param.requires_grad)
                required_gradients = list(set(required_gradients))

                self.forward_sdfg, self.backward_sdfg, self._ad_result, self._ad_inp_arrs = torch_autodiff.make_backward_function(
                    dace_model, required_gradients)

                for _, hook in self.post_autodiff_hooks.items():
                    hook(self.forward_sdfg, self.backward_sdfg)
                self.compiled_function = function_generator(self, dummy_inputs)
            else:
                self.compiled_function = function_generator(self, dummy_inputs)

            # order the parameters
            parameters_to_pass = self._call_params()

            def forward(*args):
                return self.compiled_function.function(
                    *self.compiled_function.ptr, *args, *parameters_to_pass)

            return forward

    def _call_params(self) -> Tuple[Union[Tensor, nn.parameter.Parameter]]:
        """ Get the parameters that we need to pass to the model, in the correct order. """

        # self.dace_model.inputs contains the buffers, parameters and the inputs.
        # We only want the parameters and buffers
        model_inputs = self.dace_model.inputs

        # find the index of the first input that is a parameter or buffer
        start_idx = 0
        while start_idx < len(model_inputs) and model_inputs[
                start_idx] not in self._exported_parameters:
            start_idx += 1

        return tuple(self._exported_parameters[i]
                     for i in model_inputs[start_idx:])

    def forward(self, *actual_inputs):
        """ Execute the forward pass using the traced ``module``."""
        if self.function is None:
            self.function = self._initialize_sdfg(actual_inputs)

        return self.function(*actual_inputs)

    # SDFGConvertible methods:
    # used when the model is called in a DaceProgram.
    #################################################

    def __sdfg__(self, *args):
        if self.sdfg is None:
            raise ValueError("""
            Using a PyTorch model in a DaceProgram requires that the model is initialized first.
            Either call this model using some inputs, or pass 'dummy_inputs' to the constructor.
            """)
        for name, param in self._exported_parameters.items():
            onnx_name = clean_onnx_name(name)
            if param.requires_grad:
                autodiff_library.ParameterArray.make_parameter(
                    self.sdfg, onnx_name)
        return self.sdfg

    def _add_gradient_buffers(self) -> List[str]:
        """
        Allocate gradient buffers for all parameters, and add their descriptors to the SDFG.

        :return: a list of the sdfg array names of the gradient buffers
        """

        assert self.sdfg is not None
        if hasattr(self, '_gradient_buffers'):
            return self._gradient_buffers

        buffers = []
        for name, param in self._exported_parameters.items():
            onnx_name = clean_onnx_name(name)
            desc = self.sdfg.arrays[onnx_name]

            if param.requires_grad:
                # allocate gradient buffer
                param.grad = torch.empty_like(param.data)

                # add gradient buffer descriptor to sdfg
                autodiff_library.ParameterArray.make_parameter(
                    self.sdfg, onnx_name)
                desc: autodiff_library.ParameterArray = self.sdfg.arrays[
                    onnx_name]
                grad_name = desc.add_gradient_buffer(self.sdfg, onnx_name)
                grad_desc = self.sdfg.arrays[grad_name]
                grad_desc.transient = False
                buffers.append(grad_name)
        self._gradient_buffers = buffers
        return buffers

    def __sdfg_signature__(self):

        if self.dace_model is None:
            raise ValueError(
                "Can't determine signature before SDFG is generated.")
        inputs = [clean_onnx_name(name) for name in self.dace_model.inputs]
        grad_buffers = self._add_gradient_buffers()
        inputs.extend(grad_buffers)

        return inputs, []

    @staticmethod
    def _tensor_from_param(param):
        t = param.data
        # Accessing .data on a Parameter resets the requires_grad flag
        t.requires_grad = param.requires_grad
        return t

    def __sdfg_closure__(
            self,
            reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        result = {}
        for name, param in self._exported_parameters.items():
            onnx_name = clean_onnx_name(name)
            result[onnx_name] = self._tensor_from_param(param)
            if param.requires_grad:
                grad_name = self.sdfg.arrays[onnx_name].gradient
                assert grad_name, "Expected gradient descriptor to be present"
                assert param.grad is not None, "Expected gradient buffer to be allocated"
                result[grad_name] = param.grad

        return result

    def closure_resolver(
        self,
        constant_args: Dict[str, Any],
        given_args: Set[str],
        parent_closure: Optional[pycommon.SDFGClosure] = None
    ) -> pycommon.SDFGClosure:
        assert self.sdfg is not None
        result = pycommon.SDFGClosure()
        for name, param in self._exported_parameters.items():
            onnx_name = clean_onnx_name(name)

            desc = self.sdfg.arrays[onnx_name]

            class TensorClosure:

                def __init__(self, t):
                    self.t = t

                def __call__(self):
                    return self.t

            if param.requires_grad:
                # the gradient was already added when __sdfg_signature__ was called earlier
                assert desc.gradient, "Expected gradient descriptor to be present"
                grad_name = desc.gradient
                # also add the gradient to the closure, because we need to write to it
                result.closure_arrays[grad_name] = (
                    grad_name, self.sdfg.arrays[grad_name],
                    TensorClosure(param.grad), False)

            result.closure_arrays[onnx_name] = (
                name, desc, TensorClosure(self._tensor_from_param(param)),
                False)
        return result
