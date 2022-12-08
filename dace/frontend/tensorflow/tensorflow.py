# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# TODO: This code should undergo major refactoring

from functools import partial
import numpy as np
import os
import pickle
import re
import math
from typing import Any, List
import warnings

import dace
from dace.memlet import Memlet
from dace import SDFG, SDFGState, dtypes
from dace.data import Scalar
from dace.sdfg.nodes import Tasklet, NestedSDFG
from dace.symbolic import symstr, SymExpr
from dace.frontend.tensorflow.winograd import winograd_convolution
from dace.frontend.tensorflow.transformations.redundant_array import (TensorflowRedundantArray)

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("Cannot use Tensorflow frontend without Tensorflow, " +
                      "please install: https://www.tensorflow.org/install/")

from tensorflow.python.framework import tensor_util


# http://stackoverflow.com/q/3844948/
def _checkEqualIvo(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def _tensortype(tensor: tf.Tensor):
    """ Returns a numpy type from a given TF tensor. """

    # Heuristics to determine op type
    if isinstance(tensor, tf.Operation):
        if len(tensor.outputs) == 1:
            tensor = tensor.outputs[0]
        elif len(tensor.inputs) == 1:
            tensor = tensor.inputs[0]
        elif _checkEqualIvo([inp.dtype for inp in tensor.inputs]):
            tensor = tensor.inputs[0]
        else:
            try:
                dtype = tensor.get_attr("T")
                if dtype.as_numpy_dtype == object:
                    raise NotImplementedError("Type %s is not a valid numpy type" % str(dtype))
                return dtype.as_numpy_dtype
            except ValueError:
                pass
            raise TypeError("Ambiguous type for operation %s" % tensor)

    try:
        if tensor.dtype.as_numpy_dtype == object:
            raise NotImplementedError("Type %s is not a valid numpy type" % str(tensor.dtype))
    except KeyError:
        raise TypeError("Type %s is not a valid numpy type" % str(tensor.dtype))

    if tensor.dtype.is_bool:
        return np.int32

    return tensor.dtype.as_numpy_dtype


def _tensorshape(tensor: tf.Tensor):
    if tensor.shape.dims is None or tensor.shape.dims == []:
        return 1  # Scalar
    return tensor.shape


def _find_node(state, node_id_or_label):
    """ Finds a node according to its ID (if integer is
        provided) or label (if string is provided).

        :param node_id_or_label  Node ID (if int) or label (if str)
        :return A nodes.Node object
    """

    if isinstance(node_id_or_label, str):
        for n in state.nodes():
            if n.label == node_id_or_label:
                return n
        raise LookupError("Node %s not found" % node_id_or_label)
    elif isinstance(node_id_or_label, int):
        return state.nodes()[node_id_or_label]
    else:
        raise TypeError("node_id_or_label is not an int nor string")


def string_builder(string):
    """ To match DaCe variable naming conventions, replaces all undesired 
        characters with "_".
    """
    newstring = string
    if string[0].isdigit():
        newstring = "_" + string
    out = re.sub("[^a-zA-Z0-9_]", "_", newstring)
    return out


def _name(tensor_or_op):
    if isinstance(tensor_or_op, tf.Operation):
        return None
    return string_builder(tensor_or_op.name)


_LASTSESSION = 0


def _atomic_counter_generator():
    ctr = 0
    while True:
        ctr += 1
        yield ctr


_atomic_count = _atomic_counter_generator()


class TFSession:
    def __init__(self, name: str = "tfsession", seed: int = None, config=None):
        """ Creates a DaCe Tensorflow session.

            :param name: (optional) The name of the resulting SDFG.
            :param seed: (optional) Fix random seed.
        """
        warnings.warn(
            'The TensorFlow DaCe frontend has been deprecated and will be '
            'removed in a future version, please use daceml instead:\n'
            'https://github.com/spcl/daceml', DeprecationWarning)

        self._internal_session = tf.Session(config=config)

        # Set for bookkeeping of already visited nodes
        self.visitedNodes = set()

        # Reinit state only used in training mode
        self.reinitState = None

        # Different input dictionaries
        self.constDict = dict()
        self.varDict = dict()
        self.inpDict = dict()
        self.reinitDict = dict()
        self.initDict = dict()
        self.callbackTypeDict = dict()
        self.callbackFunctionDict = dict()

        self.training = False
        self.iterations = 1
        self.seed = seed
        self.graph = SDFG(name)
        self.kill = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def train(
        self,
        optimizer,
        initializer,
        iterations,
        feed_dict,
        gpu=False,
        nodes=None,
        output_gradients=False,
    ):
        """ Trains a subgraph for the specified number of iterations and 
            returns requested nodes after training.
            
            :param optimizer: A TensorFlow tf.Optimizer node.
            :param initializer: Either a list of global and local initializers
                                or one initializer.
            :param iterations: Number of training steps.
            :param feed_dict: Dictionary representing input values and arrays
                              to feed in to the evaluator.
            :param gpu: This boolean should be set if the session is to be run on a GPU.
            :param nodes: (optional) A TensorFlow node or an iterable
                          (e.g. list) of nodes to evaluate.
            :param output_gradients: A boolean, if set, will output all the gradients passed as the
                                     optimizer arument. This will assume optimizer contains the
                                     list of gradient tensors that will be added to the outputs.
            :return: A 2-tuple of (varDict, values) - the first is a dictionary
                     of all variables used in the network in arbitrary order,
                     and the second is a tuple of values in the same order as
                     `nodes`.
        """

        # Initialize a new SDFG
        self.graph = SDFG(self.graph.name)
        self.graph.propagate = False
        self.state = SDFGState("s0", self.graph)
        self.graph.add_node(self.state)
        self.iterations = iterations
        state = self.state
        sdfg = self.graph
        outputs = []
        output_names = []
        # init state
        s0 = state
        # computational state"
        s1 = sdfg.add_state("s1")
        # emtpy exit state
        s2 = sdfg.add_state("s2")
        # As currently output arrays of conflict resolution do not automaticly
        # get reinitialized in each state iterations, we have to manually do
        # it in this state.
        reinitState = sdfg.add_state("reinitialization")
        self.reinitState = reinitState
        # set training mode

        self.training = True

        # add edges between states
        sdfg.add_edge(s0, s1, dace.sdfg.InterstateEdge(assignments=dict(__dacet1=0)))
        sdfg.add_edge(
            s1,
            reinitState,
            dace.sdfg.InterstateEdge(
                condition=dace.properties.CodeProperty.from_string("__dacet1 <" + str(iterations - 1),
                                                                   dace.dtypes.Language.Python),
                assignments={"__dacet1": "__dacet1+1"},
            ),
        )
        sdfg.add_edge(reinitState, s1, dace.sdfg.InterstateEdge())
        sdfg.add_edge(
            s1, s2,
            dace.sdfg.InterstateEdge(
                condition=dace.properties.CodeProperty.from_string("__dacet1 >= " +
                                                                   str(iterations - 1), dace.dtypes.Language.Python)))

        try:
            iter(initializer)
            initializer = list(initializer)
        except TypeError:
            initializer = [initializer]

        try:
            iter(nodes)
            nodes = list(nodes)
        except TypeError:
            nodes = [nodes]

        try:
            iter(optimizer)
            optimizer = list(optimizer)
        except TypeError:
            optimizer = [optimizer]

        ###########################
        # Prepare subgraph to process
        # If only one node was given, construct a list from it
        if not nodes == [None]:
            ops = [node if isinstance(node, tf.Operation) else node.op for node in nodes]
            output_names = [string_builder(node.name) if not isinstance(node, tf.Operation) else None for node in nodes]

        # Visit initializer and create subgraph for init state
        # If only one node was given, construct a list from it

        init = [i if isinstance(i, tf.Operation) else i.op for i in initializer]
        self.visit_backwards(init)

        # Visit the rest of the nodes
        self.state = s1
        state = s1
        # As we are in a new state, all variable nodes should be revisited
        self.visitedNodes.clear()
        if not nodes == [None]:
            self.visit_backwards(ops)
        optimizer = [opt if isinstance(opt, tf.Operation) else opt.op for opt in optimizer]
        self.visit_backwards(optimizer)
        ############################

        # Remove orphan nodes and register node types
        node_types = {}
        for state in self.graph.nodes():
            for node in state.nodes():
                if state.in_degree(node) + state.out_degree(node) == 0:
                    state.remove_node(node)
                    if node.label in self.constDict:
                        del self.constDict[node.label]
                elif isinstance(node, dace.sdfg.nodes.AccessNode):
                    node_types[node.data] = node.desc(self.graph).dtype.type
        ############################
        # Set up arguments
        sdfg_args = {}
        sdfg_args.update(self.constDict)
        sdfg_args.update(self.varDict)
        sdfg_args.update(self.inpDict)
        sdfg_args.update(self.reinitDict)
        sdfg_args.update(self.initDict)

        sdfg_args.update({(k if isinstance(k, str) else string_builder(k.name + "_Inp")): v
                          for k, v in feed_dict.items()})

        # Set scalar arguments to appropriate arrays of size 1
        sdfg_args.update(
            {k: (v if isinstance(v, np.ndarray) else np.array(v, dtype=node_types[k]))
             for k, v in sdfg_args.items()})

        ############################
        # Create output numpy arrays
        if output_gradients:
            for opt in optimizer:
                if isinstance(opt, tf.Tensor):
                    nodes.append(opt)
                    output_names.append(opt.name)
        if not nodes == [None]:
            outputs = {
                name: np.zeros(_tensorshape(node), dtype=_tensortype(node))
                for node, name in zip(nodes, output_names) if name is not None and name not in sdfg_args
            }
            outputs.update({k: v for k, v in sdfg_args.items() if k in output_names})

            sdfg_args.update(outputs)

        ############################
        # Mark outputs as non-transients
        for output in outputs:
            self.graph.arrays[output].transient = False
        ############################

        print("Adding connectors")
        self.graph.fill_scope_connectors()
        # self.graph.simplify(validate=False)
        if gpu:
            self.graph.apply_gpu_transformations()

        # Compile and call the SDFG
        compiled_sdfg = self.graph.compile()
        compiled_sdfg(**sdfg_args)
        ############################

        # Return the outputs and weights

        return (
            self.varDict,
            tuple(outputs[output] if output is not None else None for output in output_names),
        )

    def compile(self, nodes, gpu, name=None, patterns=[], validate=False, permissive=False):
        """ Compiles a subgraph into a callable function, which is equivalent 
            to calling ``run()``. 

            :param nodes: Node or an iterable (e.g. list) of nodes to evaluate.
            :param name: Name of the SDFG to create, or None for a unique name.
            :param gpu: set this boolean to True if compilation has to be done for GPU.
            :param patterns: A list of list of Transformation(s) that should be applied.
            :param validate: Boolean that decides if validation will take place after
                             transformations.
            :param permissive: Should the transformations be permissive
            :return: A function that receives a feed_dict, evaluates the nodes,
                     and returns a tuple of values in the same order as nodes.
        """
        from dace.config import Config

        # Create a unique name for this session
        if name is None:
            global _LASTSESSION
            _LASTSESSION += 1
            name = "tfsession%d" % _LASTSESSION

        # Prepare subgraph to process
        total_nodes = []
        # Determine output type
        output_type = None
        if not isinstance(nodes, (list, tuple, dict)):  # iter() works in TensorFlow
            output_type = object
            total_nodes.append(nodes)
            output_names = _name(nodes)
        elif isinstance(nodes, dict):
            output_type = type(nodes)
            output_names = {}
            for k, node in nodes.items():
                try:
                    iter(node)
                    if isinstance(node, dict):
                        raise TypeError("Dictionaries of dictionaries unsupported")
                    total_nodes.extend(node)
                    output_names[k] = type(node)(_name(n) for n in node)
                except TypeError:
                    total_nodes.append(node)
                    output_names[k] = _name(node)
        elif isinstance(nodes, (list, tuple)):
            output_type = type(nodes)
            total_nodes.extend(nodes)
            output_names = output_type(_name(node) for node in nodes)
        else:
            raise TypeError("Unsupported type for fetches: " + str(type(nodes)))

        total_output_names = [
            string_builder(node.name) if not isinstance(node, tf.Operation) else None for node in total_nodes
        ]

        # Initialize a new SDFG
        self.graph = SDFG(name)
        self.graph.propagate = False
        self.reinitState = self.graph.add_state("reinitialization")
        self.state = self.graph.add_state_after(self.reinitState, "s0")
        self.visitedNodes.clear()
        ############################

        ops = [node if isinstance(node, tf.Operation) else node.op for node in total_nodes]
        self.kill = False
        self.visit_backwards(ops)
        if self.kill:
            raise NotImplementedError("Nodes listed above are not implemented")
        ############################

        # Remove orphan nodes and register node types
        node_types = {}
        for state in self.graph.nodes():
            for node in state.nodes():
                if state.in_degree(node) + state.out_degree(node) == 0:
                    state.remove_node(node)
                    if node.label in self.constDict:
                        del self.constDict[node.label]
                elif isinstance(node, dace.sdfg.nodes.AccessNode):
                    node_types[node.data] = node.desc(self.graph).dtype.type

        # Remove arrays that were not used by other access nodes
        for name, desc in list(self.graph.arrays.items()):
            if name not in node_types and not isinstance(desc, Scalar):
                del self.graph.arrays[name]

        self.graph.fill_scope_connectors()
        ############################
        # Set up arguments
        sdfg_args = {}
        sdfg_args.update(self.constDict)
        sdfg_args.update(self.varDict)
        sdfg_args.update(self.inpDict)
        sdfg_args.update(self.initDict)
        # Set scalar arguments to appropriate arrays of size 1
        sdfg_args.update(
            {k: (v if isinstance(v, np.ndarray) else np.array(v, dtype=node_types[k]))
             for k, v in sdfg_args.items()})

        ############################
        # Create output numpy arrays
        outputs = {
            name: np.zeros(_tensorshape(node), dtype=_tensortype(node))
            for node, name in zip(total_nodes, total_output_names) if name is not None and name not in sdfg_args
        }
        outputs.update({k: v for k, v in sdfg_args.items() if k in total_output_names})
        sdfg_args.update(outputs)
        ############################
        # Mark outputs as non-transients
        for output in outputs:
            if output in self.graph.arrays:
                self.graph.arrays[output].transient = False
        ############################
        # Compile the SDFG
        if gpu:
            #    self.graph.apply_gpu_transformations()
            for aname, array in self.graph.arrays.items():
                if array is None:
                    continue
                if array.storage in [
                        dace.StorageType.Default,
                        dace.StorageType.CPU_Heap,
                ]:
                    array.storage = dace.StorageType.CPU_Pinned

            # Modify sdfg_args
            # import numba.cuda

            # for aname, arg in sdfg_args.items():
            #    if isinstance(arg, np.ndarray):
            #        sdfg_args[aname] = numba.cuda.pinned_array(
            #            arg.shape, dtype=arg.dtype, strides=arg.strides
            #        )
            #        sdfg_args[aname][:] = arg

        if patterns and len(patterns) > 0:
            self.graph.apply_transformations(patterns, validate=validate, permissive=permissive)
        compiled_sdfg = self.graph.compile()
        sdfg_args.update(self.callbackFunctionDict)

        ############################
        # Create the function that invokes the SDFG
        def call_func(feed_dict=None):
            if feed_dict is not None:
                invoke_args = dict(
                    sdfg_args, **{(k if isinstance(k, str) else string_builder(k.name)): v
                                  for k, v in feed_dict.items()})

                compiled_sdfg(**invoke_args)
            else:
                compiled_sdfg(**sdfg_args)

            # Single output
            if output_type is object:
                return outputs[output_names] if output_names is not None else None
            # Dictionary of lists/single outputs
            elif output_type is dict:
                out_dict = {}
                for k, v in output_names.items():
                    if isinstance(v, (list, tuple)):
                        out_dict[k] = type(v)(outputs[vname] if vname is not None else None for vname in v)
                    else:
                        out_dict[k] = outputs[v] if v is not None else None
                return out_dict
            # List of outputs
            else:
                return output_type(outputs[output] if output is not None else None for output in output_names)

        # Return the function
        return call_func

    def run(
        self,
        nodes,
        feed_dict=None,
        gpu=False,
        transformations=None,
        validate=False,
        permissive=False,
        name=None,
        winograd=False,
    ):
        """ Evaluates a subgraph and returns a tuple of the evaluated nodes
            (behaves similarly to sess.run).

            :param nodes: Node or an iterable (e.g. list) of nodes to evaluate.
            :param feed_dict: Dictionary representing input values and arrays
                              to feed in to the evaluator.
            :param name: Name of the SDFG to create, or None for a unique name.
            :param gpu: This boolean should be set if the session is to be run on a GPU.
            :param patterns: A list of list of Transformation(s) that should be applied. the outer
                             list is just in-case you want the transformations in a certain sequence.
            :param validate: Boolean that decides if validation will take place after
                             transformations.
            :return: Tuple or dictionary of values in the same order as `nodes`.
        """
        self.winograd = winograd
        callfunc = self.compile(
            nodes,
            gpu,
            name=name,
            validate=validate,
            permissive=permissive,
            patterns=transformations,
        )
        return callfunc(feed_dict=feed_dict)

    def dfs_nodes(self, source):
        """ Produce nodes in a depth-first-search (DFS) on a TensorFlow graph.

            :param source: The source node to start from.
            :return: A generator of nodes in the depth-first-search.
            :note: Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
                    by D. Eppstein, July 2004.
        """

        # If source is a list of nodes (or any iterable), start from all
        try:
            iter(source)
            nodes = list(source)
        except TypeError:
            nodes = [source]

        visited = set()

        for start in nodes:
            if start in visited:
                continue
            visited.add(start)
            yield start

            inputSet = [inp.op for inp in start.inputs]
            inputSet.extend(list(start.control_inputs))
            stack = [(start, iter(inputSet))]
            while stack:
                parent, children = stack[-1]
                try:
                    child = next(children)

                    if child not in visited:
                        yield child
                        visited.add(child)

                        inputSet = [inp.op for inp in child.inputs]
                        inputSet.extend(list(child.control_inputs))
                        stack.append((child, iter(inputSet)))
                except StopIteration:
                    stack.pop()

    def visit_backwards(self, node):
        """ Visit a graph from an output node backwards to the inputs. """
        for node in self.dfs_nodes(node):
            if node not in self.visitedNodes:
                self.visit(node)

    def visit(self, node):
        """ Visit a specific node in the graph, creating the SDFG. """
        try:
            func = getattr(self, "visit_" + node.type)
        except AttributeError:
            # Only stop processing after all node types have been visited,
            # so that we know which implementations are missing.
            print("MISSING IMPLEMENTATION:", node.type)
            self.visit_callback(node)
            self.visitedNodes.add(node)
            return

        func(node)
        self.visitedNodes.add(node)

    def visit_callback(self, node):
        node_name = "callback_" + string_builder(node.type)
        inputNodes = []
        inputDims = []
        for inpTensor in node.inputs:
            try:
                inputNode, _, itsdims = self.create_and_add_input_node(inpTensor)
                inputNodes.append(inputNode)
                inputDims.append(itsdims)
            except TypeError:
                print("type is not primitive, as seen in callback")
            except ValueError:
                print("Shape can not be inferred")
        try:
            outputList = self.create_and_add_output_node(node)
        except TypeError:
            return

        outputDims = [self.get_default_dims(outp) for outp in node.outputs]

        num_outputs = 0
        # Add outputs as inputs so that the tasklet can modify them in-place
        for _insertpos, (_outp, _dims) in enumerate(zip(outputList, outputDims)):
            if _dims == ["0:1"]:
                # If the output is a scalar, there should be only one output
                assert len(outputList) == 1
                # In this case, it is a callback that returns something, we don't pass by reference
                break
            inputNodes.insert(_insertpos, self.state.add_read(_outp.data))
            inputDims.insert(_insertpos, _dims)
            num_outputs = num_outputs + 1

        taskletInputs = ["i" + str(index) for index in range(len(inputNodes))]
        taskletOutputs = ["out" + str(index) for index in range(len(outputList))]

        def tensorflow_callback(tf_op, *inputList, num_outputs=0):
            # TODO: Do not recreate session every callback invocation
            # TODO(later): Optimize
            real_inputs = inputList[num_outputs:]

            newGraph = tf.Graph()
            with newGraph.as_default():
                newInputs = [tf.constant(_np_inp) for _np_inp in real_inputs]
                newOp = tf.Operation(tf_op.node_def, newGraph, inputs=newInputs)
            outputs_tf = tf.Session(graph=newGraph).run(newOp.outputs)
            if num_outputs == 0:
                return outputs_tf[0]
            for index in range(num_outputs):
                np.copyto(inputList[index], outputs_tf[index])

        tensorflow_callback = partial(tensorflow_callback, node, num_outputs=num_outputs)

        # We need two dicts, one is the sdfg args which is used to give this python partial object
        # Second is the argtypes dict in the sdfg, used to generate function pointer signature
        callback_input_types = []
        for somenode in inputNodes:
            if somenode.desc(self.graph).shape == (1, ):
                callback_input_types.append(somenode.desc(self.graph).dtype)
            else:
                callback_input_types.append(somenode.desc(self.graph))

        if num_outputs > 0:
            dace_data_scalar = dace.data.Scalar(dace.callback(None, *callback_input_types))
        else:
            dace_data_scalar = dace.data.Scalar(
                dace.callback(outputList[0].desc(self.graph).dtype, *callback_input_types))

        # Register callback in SDFG
        node_name, _ = self.graph.add_scalar(node_name, dace_data_scalar.dtype, find_new_name=True)
        self.callbackTypeDict[node_name] = dace_data_scalar
        self.callbackFunctionDict[node_name] = tensorflow_callback

        callback_tasklet = self.state.add_tasklet(
            node_name,
            {*taskletInputs},
            {*taskletOutputs},
            "out0 = " + node_name + "(" + ",".join(taskletInputs) + ")" if num_outputs == 0 else node_name + "(" +
            ",".join(taskletInputs) + ")",
        )

        for index, (inode, dim) in enumerate(zip(inputNodes, inputDims)):
            self.state.add_edge(
                inode,
                None,
                callback_tasklet,
                "i" + str(index),
                Memlet.simple(inode, ",".join(dim)),
            )
        for index, (outnode, dim) in enumerate(zip(outputList, outputDims)):
            self.state.add_edge(
                callback_tasklet,
                "out" + str(index),
                outputList[index],
                None,
                Memlet.simple(outputList[index], ",".join(outputDims[index])),
            )

    # TODO: Remove in favor of callbacks
    def visit_IteratorGetNext(self, node):
        outputList = self.create_and_add_output_node(node)
        outputDims = [self.get_default_dims(_out_tensor) for _out_tensor in node.outputs]

        def tensorflow_dataloader(tf_session, tf_node, *outputs):
            outputs_tf = [tf_session.run(_out) for _out in tf_node.outputs]
            for _index in range(len(outputs_tf)):
                np.copyto(outputs[_index], outputs_tf[_index])

        call_this = partial(tensorflow_dataloader, tf.Session(), node)
        node_name = string_builder(node.type)
        taskletOutputs = ["out" + str(_index) for _index in range(len(node.outputs))]
        dataloader_tasklet = self.state.add_tasklet(
            node_name,
            {},
            {*taskletOutputs},
            node_name + "(" + ",".join(taskletOutputs) + ")",
        )
        self.callbackFunctionDict[node_name] = call_this
        callback_types = []
        for somenode in outputList:
            callback_types.append(somenode.desc(self.graph))
        self.callbackTypeDict[node_name] = dace.data.Scalar(dace.callback(None, *callback_types))
        for _index, _out_dace in enumerate(outputList):
            self.state.add_edge(
                dataloader_tasklet,
                taskletOutputs[_index],
                _out_dace,
                None,
                Memlet.simple(_out_dace, ",".join(outputDims[_index])),
            )

    ######################################################################
    # Operator (TensorFlow graph node) visitors

    def visit_Add(self, node):
        self.visit_element_wise_op(node, "+")

    def visit_Mul(self, node):
        self.visit_element_wise_op(node, "*")

    def visit_Sub(self, node):
        self.visit_element_wise_op(node, "-")

    def visit_RealDiv(self, node):
        self.visit_element_wise_op(node, "/")

    def visit_Equal(self, node):
        self.visit_element_wise_op(node, "==")

    def visit_Const(self, node):
        state = self.state
        label = string_builder(node.name + "_0")

        # Create DaCe shape
        shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(node.outputs[0])))
        # Create np array from tensor value
        npArray = tensor_util.MakeNdarray(node.get_attr("value")).reshape(shape)

        # Add to constDict so that it can be fed to the program
        self.constDict[label] = npArray.astype(_tensortype(node))

        nodeArray = list(filter(lambda a: a.label == label, self.state.nodes()))

        # If node already present set it non transient, otherwise add node
        if not nodeArray:
            dtype = dace.typeclass(_tensortype(node))
            state.add_array(label, shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG)
        else:
            nodeArray[0].desc(self.graph).transient = False

    def visit_NoOp(self, node):
        # no op case where nothing happens
        pass

    def visit_Pack(self, node):
        # we do nothing with this op
        pass

    def visit_StridedSlice(self, node):
        # we do nothing with this op
        pass

    def visit_VariableV2(self, node):

        state = self.state
        label = string_builder(node.name) + "_0"
        shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(node.outputs[0])))

        try:
            outputNode = _find_node(state, label)
            outputNode.desc(self.graph).transient = False
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node))
            state.add_array(label, shape, dtype)

        # If not already added to the varDict, add a placeholder
        # zero-initialized array to it so a value error is not triggered.
        if label not in self.varDict.keys():
            npArray = np.zeros(shape=shape)
            self.varDict[label] = npArray.astype(_tensortype(node))

    def visit_Assign(self, node):
        # Simple memcopy from input1 to input0 as assign has no outputlist but
        # input0 is the variable we want to assign
        # Modified to rely on only the second argument tensor for shape and
        # dtype.
        state = self.state
        label = string_builder(node.inputs[1].name)
        try:
            fillNode = _find_node(state, label)
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.inputs[1]))
            shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(node.inputs[1])))
            fillNode = state.add_transient(name=label,
                                           shape=shape,
                                           dtype=dtype,
                                           lifetime=dtypes.AllocationLifetime.SDFG)

        label = string_builder(node.inputs[0].name)
        try:
            emptyNode = _find_node(state, string_builder(node.inputs[0].name))
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.inputs[1]))
            shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(node.inputs[1])))
            assert dtype is not None
            assert shape is not None
            emptyNode = state.add_transient(name=label,
                                            shape=shape,
                                            dtype=dtype,
                                            lifetime=dtypes.AllocationLifetime.SDFG)
        dims = self.get_default_dims(node.inputs[1])
        memlet = Memlet.simple(emptyNode, ",".join(dims))
        state.add_edge(fillNode, None, emptyNode, None, memlet)

    def visit_AssignVariableOp(self, node):
        self.visit_Assign(node)

    def visit_Placeholder(self, node):

        outputShape = []
        outputParams = []
        outputDims = []
        inputShape = []
        inputParams = []
        inputDims = []
        outputTensor = node.outputs[0]
        state = self.state
        label = string_builder(node.name + "_0")

        # Check if the node is already in the graph and get as a list
        try:
            outputNode = _find_node(state, label)

        except (LookupError):
            outputNode = self.create_and_add_output_node(node)

        dtype = _tensortype(node)

        # If we are in training mode, we set up another map to reduce the huge
        # (iterations x batchsize x size of input) input to one dimension less
        if self.training:
            # Output dimensions of the map

            outputDims = self.get_default_dims(outputTensor)
            outputParams = self.get_default_params(outputTensor, 1)
            outputShape = list(map(str, _tensorshape(outputTensor)))

            # Prepend the iterations dimension to the input (t1=iterations)
            inputShape.append(str(self.iterations))
            inputShape.extend(outputShape)
            inputParams.append("i0")
            inputParams.extend(outputParams)
            inputDims.append("__dacet1:__dacet1+1")
            inputDims.extend(outputDims)

            # create node for the training examples
            shape = dace.properties.ShapeProperty.from_string(",".join(inputShape))
            dtype = _tensortype(node)
            inputNode = state.add_array(name=label + "_Inp", shape=shape, dtype=dace.typeclass(dtype))

            # create and add map
            mapDict = dict(zip(inputParams, inputDims))
            inMemletDict = dict(j0=Memlet.simple(inputNode, ",".join(inputParams)))
            outMemletDict = dict(out=Memlet.simple(outputNode, ",".join(outputParams)))
            code = "out = j0"
            tasklet, map_entry, map_exit = state.add_mapped_tasklet(label, mapDict, inMemletDict, code, outMemletDict)
            state.add_edge(
                inputNode,
                None,
                map_entry,
                None,
                Memlet.simple(inputNode, ",".join(inputDims)),
            )
            state.add_edge(
                map_exit,
                None,
                outputNode,
                None,
                Memlet.simple(outputNode, ",".join(outputDims)),
            )

            # If training example node is not already in inputDict, add a
            # zero array. This prevents DaCe from raising a key error when
            # trying to call the dace function if we only execute a subgraph
            # where it does not appear. This might not be necessary any longer.
            if label + "_Inp" not in self.inpDict.keys():
                self.inpDict[label + "_Inp"] = np.zeros(tuple(map(int, (inputShape))), dtype=dtype)

            # If we are not training, set the output non transient and add to
            # input dict
        else:
            outputNode.desc(self.graph).transient = False
            self.inpDict[label] = np.zeros(tuple(map(int, (outputNode.desc(self.graph).shape))), dtype=dtype)

    def visit_TruncatedNormal(self, node):
        # Creates a truncated normal array and adds it to initDict
        state = self.state
        label = string_builder(node.name + "_0")
        # Check if already in graph, set non-transient. Otherwise add to graph.
        try:
            outputNode = _find_node(state, label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        seed = 0 if self.seed is None else self.seed

        array = tf.truncated_normal(node.outputs[0].shape, seed=seed).eval(session=self._internal_session)
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_RandomStandardNormal(self, node):

        state = self.state
        label = string_builder(node.name + "_0")

        try:
            outputNode = _find_node(state, label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        array = tf.random_normal(node.outputs[0].shape, seed=self.seed).eval(session=self._internal_session)
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_RandomUniform(self, node):
        # Creates a random uniform array and adds it to initDict
        state = self.state
        label = string_builder(node.name + "_0")
        # Check if already in graph, set non-transient. Otherwise add to graph.
        try:
            outputNode = _find_node(state, label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        seed = 0 if self.seed is None else self.seed

        array = tf.random_uniform(node.outputs[0].shape, seed=seed).eval(session=self._internal_session)
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_RandomUniformInt(self, node):
        # Creates a random uniform array and adds it to initDict
        state = self.state
        label = string_builder(node.name + "_0")
        # Check if already in graph, set non-transient. Otherwise add to graph.
        try:
            outputNode = _find_node(state, label)
            outputNode.desc(self.graph).transient = False

        except (LookupError):
            self.create_and_add_output_node(node)

        seed = 0 if self.seed is None else self.seed

        array = tf.random_uniform(
            node.outputs[0].shape,
            dtype=tf.as_dtype(_tensortype(node)),
            minval=node.inputs[1],
            maxval=node.inputs[2],
            seed=seed,
        ).eval(session=self._internal_session)
        self.initDict[label] = array.astype(_tensortype(node))

    def visit_Fill(self, node):
        # Fills an array with a scalar input value
        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            # Scalar input is at position 1
            if count == 1:
                inp, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inp.desc(self.graph))
                inputNodes.append(inp)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        for out in node.outputs:
            params = self.get_default_params(out, 1)
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0] + outputParams[0]
        mapRange = inputDims[0] + outputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Slice(self, node):
        begin_positions = self._internal_session.run(node.inputs[1])
        sizes = self._internal_session.run(node.inputs[2])
        end_positions = begin_positions + sizes
        inputNode, _, _ = self.create_and_add_input_node(node.inputs[0])
        outputNode = self.create_and_add_output_node(node)[0]
        input_subset = [str(b) + ":" + str(e) for b, e in zip(begin_positions, end_positions)]
        sliceMemlet = Memlet.simple(
            inputNode,
            ",".join(input_subset),
            other_subset_str=",".join(self.get_default_dims(node.outputs[0])),
        )
        self.state.add_edge(inputNode, None, outputNode, None, sliceMemlet)

    def visit_Mean(self, node):
        outputNode = self.create_and_add_output_node(node)[0]
        outputDims = self.get_default_dims(node.outputs[0])
        inputNode, params, dims = self.create_and_add_input_node(node.inputs[0])
        reduction_axes = self._internal_session.run(node.inputs[1])
        if len(reduction_axes.shape) == 0:
            reduction_axes = np.array([reduction_axes])
        reduction_axes.sort()
        norm = 1
        for i in reduction_axes:
            norm *= inputNode.desc(self.graph).shape[i]
        norm = _tensortype(node.outputs[0])(norm)
        mapLabel = string_builder(node.type)
        mapParams = params
        mapDims = dims
        mapEntry, mapExit = self.state.add_map(mapLabel, dict(zip(mapParams, mapDims)))
        tasklet = self.state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0/" + str(norm))
        self.add_in_memlets([inputNode], mapEntry, tasklet, [dims], [params])
        outputShape = _tensorshape(node.outputs[0])
        if node.get_attr("keep_dims"):
            outputParams = [params[i] if outputShape[i] != 1 else "0" for i in range(len(mapParams))]
        else:
            temp = set(mapParams[a] for a in reduction_axes)
            outputParams = list(set(mapParams) - temp)
            outputParams.sort()
        if len(outputParams) == 0:
            outputParams = ["0"]
        self.add_out_memlets(
            [outputNode],
            mapExit,
            tasklet,
            [outputDims],
            [outputParams],
            wcr="lambda a,b: a+b",
            wcr_identity=0,
        )

    # Would reduce all but the last dimension in the input.
    def visit_FusedBatchNorm(self, node):
        local_ctr = str(next(_atomic_count))
        ######### All the nodes and constants ##########
        inpTensorNode, inpTensorParams, inpTensorDims = self.create_and_add_input_node(node.inputs[0])
        scale, _, scaleDims = self.create_and_add_input_node(node.inputs[1])
        offset, _, offsetDims = self.create_and_add_input_node(node.inputs[2])
        epsilon = node.get_attr("epsilon")
        epsilon = float(epsilon)
        # outputs
        outputList = self.create_and_add_output_node(node)
        normalisedTensorNode = outputList[0]
        meanTensorNode = outputList[1]
        meanDims = self.get_default_dims(node.outputs[1])
        varianceTensorNode = outputList[2]
        varianceDims = self.get_default_dims(node.outputs[2])
        rootVarianceTensorNode = outputList[4]
        rootVarianceDims = self.get_default_dims(node.outputs[4])
        normalisationScalar = 1
        assert str(node.get_attr("data_format"))[2:-1] == "NHWC"
        assert node.get_attr("is_training") == True
        for i in inpTensorNode.desc(self.graph).shape[:-1]:
            normalisationScalar *= i
        normalisationScalar = float(normalisationScalar)
        sumInputs = self.state.add_transient(
            "sigma_x_" + local_ctr,
            _tensorshape(node.outputs[2]),
            _tensortype(node.outputs[2]),
        )
        sumSquareInputs = self.state.add_transient(
            "sigma_x2_" + local_ctr,
            _tensorshape(node.outputs[2]),
            _tensortype(node.outputs[2]),
        )
        ######## Maps ###################
        nhwcMapBounds = dict(zip(inpTensorParams, inpTensorDims))
        cMapBounds = dict(zip([inpTensorParams[0]], [str(inpTensorDims[-1])]))
        normalisationMapEntry, normalisationMapExit = self.state.add_map(string_builder("normalisation_map"),
                                                                         nhwcMapBounds)
        meanMapEntry, meanMapExit = self.state.add_map(string_builder("mean_map"), nhwcMapBounds)
        varianceMapEntry, varianceMapExit = self.state.add_map(string_builder("variance_map"), nhwcMapBounds)
        varianceSqrtMapEntry, varianceSqrtMapExit = self.state.add_map(string_builder("variance_sqrt_map"), cMapBounds)
        ######### Tasklets #########
        fbnormTasklet = self.state.add_tasklet(
            "fbn_eltwise_norm",
            {"j0", "j1", "j2", "j3", "j4"},
            {"out"},
            "out=j1*((j0-j3)/j4)+j2",
        )
        meanTasklet = self.state.add_tasklet("mean_computation", {"j0"}, {"out"}, "out=j0/" + str(normalisationScalar))
        varianceTasklet1 = self.state.add_tasklet("variance_part_1", {"j0"}, {"out0", "out1"}, "out0=j0; out1 = j0*j0")
        varianceTasklet2 = self.state.add_tasklet(
            "variance_part_2",
            {"j0", "j1"},  # i0 is sigma(X) and i1 is sigma(X^2)
            {
                "out0",
                "out1",
            },  # out0 is the variance and out1 is the sqrt(variance + epsilon)
            "out0=j1/" + str(normalisationScalar) + " - (j0*j0)/(" + str(normalisationScalar * normalisationScalar) +
            ");out1=math.sqrt(out0 + " + str(epsilon) + ");",
        )
        ########## Common edges ##########
        self.add_in_memlets(
            [inpTensorNode, scale, offset, meanTensorNode, rootVarianceTensorNode],
            normalisationMapEntry,
            fbnormTasklet,
            [inpTensorDims, scaleDims, offsetDims, meanDims, varianceDims],
            [
                inpTensorParams,
                [inpTensorParams[-1]],
                [inpTensorParams[-1]],
                [inpTensorParams[-1]],
                [inpTensorParams[-1]],
            ],
        )
        self.add_out_memlets(
            [normalisedTensorNode],
            normalisationMapExit,
            fbnormTasklet,
            [inpTensorDims],
            [inpTensorParams],
        )
        self.add_in_memlets(
            [inpTensorNode],
            meanMapEntry,
            meanTasklet,
            [inpTensorDims],
            [inpTensorParams],
        )
        self.add_out_memlets(
            [meanTensorNode],
            meanMapExit,
            meanTasklet,
            [meanDims],
            [[inpTensorParams[-1]]],
            wcr="lambda a,b: a+b",
            wcr_identity=0,
        )
        self.add_in_memlets(
            [inpTensorNode],
            varianceMapEntry,
            varianceTasklet1,
            [inpTensorDims],
            [inpTensorParams],
        )
        self.add_out_memlets(
            [sumInputs, sumSquareInputs],
            varianceMapExit,
            varianceTasklet1,
            [varianceDims, varianceDims],
            [[inpTensorParams[-1]], [inpTensorParams[-1]]],
            wcr="lambda a,b: a+b",
            wcr_identity=0,
        )
        self.add_in_memlets(
            [sumInputs, sumSquareInputs],
            varianceSqrtMapEntry,
            varianceTasklet2,
            [varianceDims, varianceDims],
            [[inpTensorParams[0]], [inpTensorParams[0]]],
        )
        self.add_out_memlets(
            [varianceTensorNode, rootVarianceTensorNode],
            varianceSqrtMapExit,
            varianceTasklet2,
            [varianceDims, varianceDims],
            [[inpTensorParams[0]], [inpTensorParams[0]]],
        )

    def visit_FusedBatchNormGrad(self, node):
        local_ctr = str(next(_atomic_count))
        ############################INPUTS##############################################
        backpropGradients, backpropParams, backpropDims = self.create_and_add_input_node(node.inputs[0])
        inputData, inputParams, inputDims = self.create_and_add_input_node(node.inputs[1])
        gammaNode, _, gammaDims = self.create_and_add_input_node(node.inputs[2])
        meanNode, _, meanDims = self.create_and_add_input_node(node.inputs[3])
        stdevNode, _, stdevDims = self.create_and_add_input_node(node.inputs[4])
        #############################OUTPUTS#############################################
        outputList = self.create_and_add_output_node(node)
        imageGrads = outputList[0]
        gammaGrads = outputList[1]
        betaGrads = outputList[2]
        ############################TRANSIENTS##########################################
        gammaPrime = self.state.add_transient(
            "gamma_prime" + local_ctr,
            _tensorshape(node.outputs[1]),
            _tensortype(node.outputs[1]),
        )
        betaPrime = self.state.add_transient(
            "beta_prime" + local_ctr,
            _tensorshape(node.outputs[2]),
            _tensortype(node.outputs[2]),
        )
        ###############################MAPS##############################################
        # channelMapLabel = string_builder(node.type) + "_outer"
        # channelMapEntry, channelMapExit = self.state.add_map(
        #    channelMapLabel, dict(zip([backpropParams[-1]], [backpropDims[-1]]))
        # )
        innerMap1Label = string_builder(node.type) + "_inner1"
        innerMap1Entry, innerMap1Exit = self.state.add_map(innerMap1Label, dict(zip(backpropParams, backpropDims)))
        innerMap2Label = string_builder(node.type) + "_inner2"
        innerMap2Entry, innerMap2Exit = self.state.add_map(innerMap2Label, dict(zip(backpropParams, backpropDims)))
        #############################TASKLETS###########################################
        nhw = 1
        for i in backpropGradients.desc(self.graph).shape[:-1]:
            nhw *= i
        nhw = str(float(nhw))
        auxGradsTasklet = self.state.add_tasklet(
            "linear_grads",
            {"y_prime", "x", "mu", "stdev"},
            {"gamma_prime", "beta_prime"},
            "beta_prime = y_prime; gamma_prime = y_prime * (x - mu) / stdev;",
        )
        # add inconnector beta_prime
        inputGradsTasklet = self.state.add_tasklet(
            "input_grads",
            {"gamma", "gamma_prime", "beta_prime", "y_prime", "x", "mu", "stdev"},
            {"x_prime"},
            "x_prime = float(gamma*(" + nhw + "*y_prime - beta_prime - (gamma_prime*(x - mu)/stdev))/(stdev*" + nhw +
            "));",
        )
        inputs = [backpropGradients, inputData, meanNode, stdevNode]
        dims = [backpropDims, inputDims, meanDims, stdevDims]
        # middleParams = [
        #    backpropDims[:-1] + [backpropParams[-1]],
        #    backpropDims[:-1] + [backpropParams[-1]],
        #    [backpropParams[-1]],
        #    [backpropParams[-1]],
        # ]

        # auxGradTasklet in-edges
        for _dim, _node in zip(dims, inputs):
            self.state.add_edge(_node, None, innerMap1Entry, None, Memlet.simple(_node, ",".join(_dim)))
        # self.add_in_memlets(inputs, channelMapEntry, innerMap1Entry, dims, middleParams)
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "y_prime",
            Memlet.simple(backpropGradients, ",".join(backpropParams)),
        )
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "x",
            Memlet.simple(inputData, ",".join(inputParams)),
        )
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "mu",
            Memlet.simple(meanNode, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap1Entry,
            None,
            auxGradsTasklet,
            "stdev",
            Memlet.simple(stdevNode, ",".join([backpropParams[-1]])),
        )
        # auxGradsTasklet out-edges
        self.add_init(gammaPrime.data, 0.0)
        self.state.add_edge(
            auxGradsTasklet,
            "gamma_prime",
            innerMap1Exit,
            None,
            Memlet.simple(
                gammaPrime,
                ",".join([backpropParams[-1]]),
                wcr_str="lambda a,b: a+b",
            ),
        )
        self.state.add_edge(
            innerMap1Exit,
            None,
            gammaPrime,
            None,
            Memlet.simple(gammaPrime, ",".join(gammaDims), wcr_str="lambda a,b: a+b"),
        )
        self.add_init(betaPrime.data, 0.0)
        self.state.add_edge(
            auxGradsTasklet,
            "beta_prime",
            innerMap1Exit,
            None,
            Memlet.simple(
                betaPrime,
                ",".join([backpropParams[-1]]),
                wcr_str="lambda a, b: a+b",
            ),
        )
        self.state.add_edge(
            innerMap1Exit,
            None,
            betaPrime,
            None,
            Memlet.simple(betaPrime, ",".join(gammaDims), wcr_str="lambda a, b: a+b"),
        )
        # second map in-edges
        # self.add_in_memlets(
        #    [gammaNode],
        #    channelMapEntry,
        #    innerMap2Entry,
        #    [gammaDims],
        #    [[backpropParams[-1]]],
        # )
        self.state.add_edge(
            gammaNode,
            None,
            innerMap2Entry,
            None,
            Memlet.simple(gammaNode, ",".join(gammaDims)),
        )
        for _node, _dim in zip(inputs, dims):
            self.state.add_edge(_node, None, innerMap2Entry, None, Memlet.simple(_node, ",".join(_dim)))
        self.state.add_edge(
            gammaPrime,
            None,
            innerMap2Entry,
            None,
            Memlet.simple(gammaPrime, ",".join(gammaDims)),
        )
        self.state.add_edge(
            betaPrime,
            None,
            innerMap2Entry,
            None,
            Memlet.simple(betaPrime, ",".join(gammaDims)),
        )
        # inputGradsTasklet in-edges
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "gamma",
            Memlet.simple(gammaNode, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "beta_prime",
            Memlet.simple(betaPrime, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "gamma_prime",
            Memlet.simple(gammaPrime, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "y_prime",
            Memlet.simple(backpropGradients, ",".join(backpropParams)),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "mu",
            Memlet.simple(meanNode, ",".join([backpropParams[-1]])),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "x",
            Memlet.simple(inputData, ",".join(inputParams)),
        )
        self.state.add_edge(
            innerMap2Entry,
            None,
            inputGradsTasklet,
            "stdev",
            Memlet.simple(stdevNode, ",".join([backpropParams[-1]])),
        )
        # inputGradsTasklet out-edges
        self.state.add_edge(
            inputGradsTasklet,
            "x_prime",
            innerMap2Exit,
            None,
            Memlet.simple(imageGrads, ",".join(backpropParams)),
        )
        self.state.add_edge(
            innerMap2Exit,
            None,
            imageGrads,
            None,
            Memlet.simple(imageGrads, ",".join(backpropDims)),
        )
        self.state.add_edge(
            betaPrime,
            None,
            betaGrads,
            None,
            Memlet.simple(betaPrime, ",".join(gammaDims)),
        )
        self.state.add_edge(
            gammaPrime,
            None,
            gammaGrads,
            None,
            Memlet.simple(gammaPrime, ",".join(gammaDims)),
        )
        # self.add_out_memlets(
        #    [imageGrads],
        #    channelMapExit,
        #    innerMap2Exit,
        #    [backpropDims],
        #    [backpropDims[:-1] + [backpropParams[-1]]],
        # )
        # Add reads and edges. Can't directly add out memlets.
        # self.add_out_memlets(
        #    [gammaGrads],
        #    channelMapExit,
        #    gammaPrime,
        #    [gammaDims],
        #    [[backpropParams[-1]]],
        # )
        # self.add_out_memlets(
        #    [betaGrads], channelMapExit, betaPrime, [gammaDims], [[backpropParams[-1]]]
        # )

    def visit_Tile(self, node):
        # Replicates input multiple times
        inputList = []
        inputNodes = []

        state = self.state

        for inp in node.inputs:

            label = string_builder(inp.name)
            try:
                inputNode = _find_node(state, label)
            except (LookupError):

                inputNode = self.create_and_add_input_node(inp)[0]

            inputNodes.append(inputNode)
            inputList.append(inputNode.desc(self.graph))

        outputList = self.create_and_add_output_node(node)

        mapLabel = string_builder(node.type)
        outputDims = self.get_default_dims(node.outputs[0])
        outputParams = self.get_default_params(node.outputs[0])
        inputDims = self.get_default_dims(node.inputs[0])
        inputParams = []

        for i, dim in enumerate(inputList[0].shape):
            inputParams.append("i" + str(i) + "%" + str(dim))

        mapDict = dict(zip(outputParams, outputDims))
        inMemletDict = dict(j0=Memlet.simple(inputNodes[0], ",".join(inputParams)))
        outMemletDict = dict(out=Memlet.simple(outputList[0], ",".join(outputParams)))
        code = "out = j0"
        tasklet, map_entry, map_exit = state.add_mapped_tasklet(mapLabel, mapDict, inMemletDict, code, outMemletDict)
        state.add_edge(
            inputNodes[0],
            None,
            map_entry,
            None,
            Memlet.simple(inputNodes[0], ",".join(inputDims)),
        )
        state.add_edge(
            map_exit,
            None,
            outputList[0],
            None,
            Memlet.simple(outputList[0], ",".join(outputDims)),
        )

    def visit_ReadVariableOp(self, node):
        # TODO this should ideally be an add_read on the input name
        state = self.state
        inp = node.inputs[0]
        label = string_builder(inp.name)
        try:
            inputNode = _find_node(state, label)
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.outputs[0]))
            shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(node.outputs[0])))
            inputNode = state.add_transient(name=label, shape=shape, dtype=dtype)

        outputNode = self.create_and_add_output_node(node)[0]
        outputDims = self.get_default_dims(node.outputs[0])
        self.state.add_edge(
            inputNode,
            None,
            outputNode,
            None,
            Memlet.simple(outputNode, ",".join(outputDims)),
        )

    def visit_VarHandleOp(self, node):
        self.create_and_add_output_node(node)

    def visit_PreventGradient(self, node):
        # Just a memcopy, works like visit_assign or visit_identity
        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            # relevant input is at position 0
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        for count, out in enumerate(node.outputs):

            dims = self.get_default_dims(out)
            params = self.get_default_params(out)
            outputParams.append(params)
            outputDims.append(dims)

        memlet = Memlet.simple(inputNodes[0], ",".join(inputDims[0]))
        state.add_edge(inputNodes[0], None, outputList[0], None, memlet)

    def visit_ExpandDims(self, node):
        # Takes an N-dimensional array and adds one dimension to it with a
        # length of 1. Example: (M,K) -> (1,M,K).
        # We can just use DaCe memory copy to do the same
        state = self.state
        inputList = []
        inputNodes = []
        inputDims = []
        inputParams = []

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputDims.append(dims)
                inputParams.append(params)

        outputList = self.create_and_add_output_node(node)
        memlet = Memlet.simple(inputNodes[0], ",".join(inputDims[0]))
        state.add_edge(inputNodes[0], None, outputList[0], None, memlet)

    def visit_ApplyGradientDescent(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        mapParams = []
        mapRange = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):

            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputParams.append(params)
            inputDims.append(dims)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        mapLabel = string_builder(node.type)
        # inputList[1] is learning rate which needs its own parameter
        inputParams[1] = ["i4"]
        # This is the variable which is input and output of this map at the same
        # time. We create the output version of it here
        out = node.inputs[0]
        outName = string_builder(out.name)
        outputNode = self.state.add_write(outName)
        dims = self.get_default_dims(out)
        params = self.get_default_params(out)
        outputList = [outputNode]
        outputParams = [params]
        outputDims = [dims]

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0] + ["i4"]
        mapRange = inputDims[0] + ["0:1"]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0", "j1", "j2"}, {"out"}, "out = j0-(j1*j2)")
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)

    def visit_ResourceApplyGradientDescent(self, node):
        # this is actually the same as above, but the real input has no shape or type.
        # that has to be changed.
        state = self.state
        inputList = []
        inputNodes = []
        inputParams = []
        inputDims = []

        # make the input node using the gradient node, because the input node has type "resource"
        # and no shape information.
        inp = node.inputs[0]
        label = string_builder(inp.name)
        try:
            inputNode = _find_node(state, label)
        except (LookupError):
            dtype = dace.typeclass(_tensortype(node.inputs[2]))
            shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(node.inputs[2])))
            inputNode = state.add_transient(name=label, shape=shape, dtype=dtype)
        inputNodes.append(inputNode)
        inputParams.append(self.get_default_params(node.inputs[2]))
        inputDims.append(self.get_default_dims(node.inputs[2]))

        for count, inp in enumerate(node.inputs):
            if count == 0:
                continue
            else:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputParams.append(params)
                inputDims.append(dims)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)

        # inputList[1] is learning rate which needs its own parameter
        inputParams[1] = ["i4"]
        out = node.inputs[2]
        outName = string_builder(node.inputs[0].name)
        outputNode = state.add_write(outName)
        dims = self.get_default_dims(out)
        params = self.get_default_params(out)
        outputList = [outputNode]
        outputParams = [params]
        outputDims = [dims]
        mapLabel = string_builder(node.type)
        mapParams = inputParams[0] + ["i4"]
        mapRange = inputDims[0] + ["0:1"]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0", "j1", "j2"}, {"out"}, "out = j0-(j1*j2)")
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)

    def visit_MatMul(self, node):
        # 2d Matrix Multiplication
        inputList = []
        inputNodes = []
        state = self.state
        mapParams = []
        outputParams = [[]]
        mapRange = []
        outputDims = [[]]
        inputParams = [[], []]
        inputDims = [[], []]

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)

        ndims = len(outputList[0].desc(self.graph).shape)
        # Params for higher dimensions (not verified)
        # (for 2d it works)
        for i in range(0, ndims + 1):
            if i == ndims:
                mapParams.append("i" + str(i))
                inputParams[1].append("i" + str(i))
                outputParams[0].append("i" + str(i))

            elif i == ndims - 1:
                mapParams.append("i" + str(i))
                inputParams[0].append("i" + str(i))
                inputParams[1].append("i" + str(i))

            elif i == ndims - 2:
                mapParams.append("i" + str(i))
                inputParams[0].append("i" + str(i))
                outputParams[0].append("i" + str(i))

            else:
                mapParams.append("i" + str(i))
                inputParams[0].append("i" + str(i))
                inputParams[1].append("i" + str(i))
                outputParams[0].append("i" + str(i))

        for i in range(0, ndims):
            inputDims[0].append(str(0) + ":" + str(node.inputs[0].shape[i]))
            inputDims[1].append(str(0) + ":" + str(node.inputs[1].shape[i]))
            outputDims[0].append(str(0) + ":" + str(node.outputs[0].shape[i]))
            mapRange.append(str(0) + ":" + str(node.inputs[0].shape[i]))

        mapRange.append(str(0) + ":" + str(node.outputs[0].shape[ndims - 1]))
        # if first input needs to be transposed
        if node.get_attr("transpose_a"):
            mapRange[0], mapRange[1] = mapRange[1], mapRange[0]
            inputParams[0][0], inputParams[0][1] = inputParams[0][1], inputParams[0][0]
        # if second input needs to be transposed
        if node.get_attr("transpose_b"):
            inputParams[1][0], inputParams[1][1] = inputParams[1][1], inputParams[1][0]

        mentry, mexit = state.add_map("matmul_outer", {mapParams[1]: mapRange[1]}, dace.ScheduleType.Sequential)
        minentry, minexit = state.add_map(
            "matmul_inner",
            {
                mapParams[0]: mapRange[0],
                mapParams[2]: mapRange[2]
            },
            dace.ScheduleType.CPU_Multicore,
        )
        tasklet = state.add_tasklet("mm_code", {"j0", "j1"}, {"out"}, "out = j0*j1")

        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(minentry, None, tasklet, name, memlet)

        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(out, ",".join(outputParams[i]), wcr_str="lambda a,b: a+b")
            state.add_edge(tasklet, name, minexit, None, memlet)

        self.reinitCR(outputList[0], outputParams, outputDims, "0")
        self.add_out_memlets(outputList, mexit, minexit, outputDims, outputParams, "lambda a,b: a+b", 0)
        self.add_in_memlets(inputNodes, mentry, minentry, inputDims, inputParams)

    def visit_element_wise_op(self, node, operation):
        """ Handles all the element wise operations, supports broadcasting. """
        inputList = []
        inputNodes = []
        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = []
        inputParams = []
        inputDims = []
        state = self.state

        for inp in node.inputs:

            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)

        outputNodes = self.create_and_add_output_node(node)
        mapLabel = string_builder(node.type)
        # create params
        for inp in inputList:
            inputParamsString = []
            for i, dim in enumerate(inp.shape):
                # scalar case that we want to broadcast
                if str(dim) == "1":
                    inputParamsString.append("0")
                else:
                    inputParamsString.append("i" + str(i))

            inputParams.append(inputParamsString)

        params = self.get_default_params(node.outputs[0])
        dims = self.get_default_dims(node.outputs[0])
        outputParams.append(params)
        outputDims.append(dims)

        mapParams = outputParams[0]
        mapRange = outputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0 " + operation + " j1")
        self.add_out_memlets(outputNodes, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Conv2D(self, node):
        if (7 in _tensorshape(node.inputs[0])[1:3] and 3 in _tensorshape(node.inputs[1])[0:2] and self.winograd):
            winograd_convolution(self, node)
        else:
            local_ctr = str(next(_atomic_count))
            inputList = []
            inputNodes = []
            ndims = 0
            strides = node.get_attr("strides")[1]
            state = self.state

            for inp in node.inputs:
                inputNode = self.create_and_add_input_node(inp)[0]
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)

            outputList = self.create_and_add_output_node(node)
            ndims = len(outputList[0].desc(self.graph).shape)
            mapLabel = string_builder(node.type)
            reduce_node = self.state.add_transient(
                mapLabel + "_wcr_avoid" + local_ctr,
                [1],
                outputList[0].desc(self.graph).dtype,
                storage=dace.StorageType.Register,
            )
            reduce_node.setzero = True

            mapParams = []
            outputParams = []
            mapRange = []
            outputDims = [[]]
            inputParams = []
            inputDims = [[], []]
            # create conv params
            inputParams.append(["i0", "i1*" + str(strides) + "+i5", "i2*" + str(strides) + "+i6", "i4"])
            inputParams.append(["i5", "i6", "i4", "i3"])
            outputParams.append(["i0", "i1", "i2", "i3"])
            # create conv dims
            for i in range(0, ndims):
                inputDims[0].append(str(0) + ":" + str(node.inputs[0].shape[i]))
                inputDims[1].append(str(0) + ":" + str(node.inputs[1].shape[i]))
                outputDims[0].append(str(0) + ":" + str(node.outputs[0].shape[i]))
            # add a padding map for same padding(zero padding so that input and
            # output of convolution have the same size)
            if str(node.get_attr("padding"))[2:-1] == "SAME":
                paddedInput, paddedDims = self.inputPadding(
                    node,
                    inputNodes[0],
                    inputList[0],
                    outputList[0].desc(self.graph).shape[1],
                    inputList[1].shape[0],
                    strides,
                    inputDims[0],
                )
                inputDims[0] = paddedDims
                inputNodes[0] = paddedInput

            mapParams = outputParams[0]
            mapParams2 = inputParams[1][:-1]
            mapRange = outputDims[0]
            mapRange2 = inputDims[1][:-1]

            mapEntry, mapExit = state.add_map(mapLabel + "_outer", dict(zip(mapParams, mapRange)))
            mapEntry2, mapExit2 = state.add_map(mapLabel + "_inner", dict(zip(mapParams2, mapRange2)))
            tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0 * j1;")  # printf(\"%f\\t\", j0);")
            self.add_out_memlets(outputList, mapExit, reduce_node, outputDims, outputParams)
            self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
            # add memlets from inner map to tasklet
            for i, inp in enumerate(inputNodes):
                name = "j" + str(i)
                memlet = Memlet.simple(inp, ",".join(inputParams[i]))
                state.add_edge(mapEntry2, None, tasklet, name, memlet)
            # add memelets from tasklet to cr
            for i, out in enumerate(outputList):
                name = "out"
                memlet = Memlet.simple(reduce_node, "0", wcr_str="lambda a,b: a+b")
                state.add_edge(tasklet, name, mapExit2, None, memlet)
                state.add_edge(mapExit2, None, reduce_node, None, memlet)

    def visit_BiasAdd(self, node):

        inputList = []
        inputNodes = []
        state = self.state

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)
        dims = outputList[0].desc(self.graph).shape

        mapLabel = string_builder(node.type)
        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = []
        inputParams = [[], []]
        inputDims = [[], []]

        params = self.get_default_params(node.outputs[0])
        dims = self.get_default_dims(node.outputs[0])
        outputParams.append(params)
        outputDims.append(dims)

        mapParams = outputParams[0]
        inputParams[0] = outputParams[0]
        # the bias matches the last dimension of input resp. output
        inputParams[1] = [mapParams[-1]]
        mapRange = outputDims[0]
        inputDims[0] = outputDims[0]
        inputDims[1] = ["0:" + str(node.inputs[1].shape[0])]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0 + j1")
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_MaxPool(self, node):
        inputList = []
        inputNodes = []
        dims = []
        inputDims = []
        strides_0 = node.get_attr("strides")[1]
        strides_1 = node.get_attr("strides")[2]
        ksize_0 = node.get_attr("ksize")[1]
        ksize_1 = node.get_attr("ksize")[2]
        state = self.state

        for inp in node.inputs:
            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)
        inputParams = [["i0", "i1*" + str(strides_0) + "+i4", "i2*" + str(strides_1) + "+i5", "i3"]]

        outputParams = []
        outputDims = []
        outputList = self.create_and_add_output_node(node)
        dims = self.get_default_dims(node.outputs[0])
        params = self.get_default_params(node.outputs[0])
        outputDims.append(dims)
        outputParams.append(params)

        if str(node.get_attr("padding"))[2:-1] == "SAME":
            assert ksize_0 == ksize_1
            assert strides_0 == strides_1
            paddedInput, paddedDims = self.inputPadding(
                node,
                inputNodes[0],
                inputList[0],
                outputList[0].desc(self.graph).shape[1],
                ksize_0,
                strides_0,
                inputDims[0],
            )
            inputDims[0] = paddedDims
            inputNodes[0] = paddedInput

        mapLabel = string_builder(node.type)
        mapParams1 = outputParams[0]
        mapRange1 = outputDims[0]
        mapParams2 = ["i4", "i5"]
        mapRange2 = ["0:" + str(ksize_0), "0:" + str(ksize_1)]

        mapEntry, mapExit = state.add_map(mapLabel + "_outer", dict(zip(mapParams1, mapRange1)))
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner",
            dict(zip(mapParams2, mapRange2)),
            schedule=dace.ScheduleType.Sequential,
        )
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.reinitCR(outputList[0], outputParams, outputDims, "-99999999999")
        self.add_out_memlets(
            outputList,
            mapExit,
            mapExit2,
            outputDims,
            outputParams,
            "lambda a, b: max(a,b)",
            -99999999999,
            wcr_conflict=False,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        # add memlets from inner map to tasklet
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)
        # add memelets from tasklet to cr
        for i, out in enumerate(outputList):
            name = "out"
            memlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str="lambda a, b: max(a,b)",
                wcr_conflict=False,
            )
            state.add_edge(tasklet, name, mapExit2, None, memlet)

    # TODO bugfix with padding, fails for cases where padding is on left and
    # right, and up and down. Will have to rewrite expression for
    # normalisationScalar
    def visit_AvgPool(self, node):
        inputList = []
        inputNodes = []
        inputDims = []
        strides_0 = node.get_attr("strides")[1]
        strides_1 = node.get_attr("strides")[2]
        ksize_0 = node.get_attr("ksize")[1]
        ksize_1 = node.get_attr("ksize")[2]
        state = self.state
        local_count = str(next(_atomic_count))
        for inp in node.inputs:
            inputNode, _, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)
        inputParams = [["i0", "i1*" + str(strides_0) + "+i4", "i2*" + str(strides_1) + "+i5", "i3"]]

        outputParams = []
        outputDims = []
        outputList = self.create_and_add_output_node(node)
        dims = self.get_default_dims(node.outputs[0])
        params = self.get_default_params(node.outputs[0])
        outputDims.append(dims)
        outputParams.append(params)

        assert str(node.get_attr("padding"))[2:-1] == "VALID"
        # if str(node.get_attr("padding"))[2:-1] == "SAME":
        #    assert ksize_0 == ksize_1
        #    assert strides_0 == strides_1
        #    paddedInput, paddedDims = self.inputPadding(
        #        node,
        #        inputNodes[0],
        #        inputList[0],
        #        outputList[0].desc(self.graph).shape[1],
        #        ksize_0,
        #        strides_0,
        #        inputDims[0],
        #    )
        #    inputDims[0] = paddedDims
        #    inputNodes[0] = paddedInput

        mapLabel = string_builder(node.type)
        mapParams1 = outputParams[0]
        mapRange1 = outputDims[0]
        mapParams2 = ["i4", "i5"]
        mapRange2 = ["0:" + str(ksize_0), "0:" + str(ksize_1)]

        mapEntry, mapExit = state.add_map(mapLabel + "_outer", dict(zip(mapParams1, mapRange1)))
        mapEntry2, mapExit2 = state.add_map(
            mapLabel + "_inner",
            dict(zip(mapParams2, mapRange2)),
            schedule=dace.ScheduleType.Sequential,
        )
        tasklet = state.add_tasklet(mapLabel + "_sum", {"j0"}, {"out"}, "out = j0")
        imgH = node.inputs[0].shape[1]
        imgW = node.inputs[0].shape[2]
        # normalisationScalar = "max((min({imgH}-1,{affine_Hexp}+{kernH}-1)-{affine_Hexp}+1)*(min({imgW}-1,{affine_Wexp}+{kernW}-1)-{affine_Wexp}+1),1)".format(
        #    imgH=str(imgH),
        #    imgW=str(imgW),
        #    affine_Hexp=str(strides_0) + "*" + str(mapParams1[1]),
        #    affine_Wexp=str(strides_1) + "*" + str(mapParams1[2]),
        #    kernH=str(ksize_0),
        #    kernW=str(ksize_1),
        # )
        normalisationScalar = str(ksize_0 * ksize_1)
        tasklet_norm = state.add_tasklet(mapLabel + "_norm", {"out"}, {"out_n"}, "out_n = out/" + normalisationScalar
                                         # + ';printf("%d",'
                                         # + normalisationScalar
                                         # + ");",
                                         )
        temp_node = self.state.add_scalar(
            "scratch_node" + local_count,
            dace.typeclass(_tensortype(node.outputs[0])),
            transient=True,
            storage=dace.StorageType.Register,
        )
        temp_node.setzero = True
        memletTempNode = Memlet.simple(str(temp_node), "0", wcr_str="lambda a, b: a+b")
        memletTempNode_nocr = Memlet.simple(str(temp_node), "0")
        memletOutputInner = Memlet.simple(outputList[0], ",".join(outputParams[0]))
        memletOutputOuter = Memlet.simple(outputList[0], ",".join(outputDims[0]))
        state.add_edge(mapExit2, None, temp_node, None, memletTempNode)
        state.add_edge(temp_node, None, tasklet_norm, "out", memletTempNode_nocr)
        state.add_edge(tasklet_norm, "out_n", mapExit, None, memletOutputInner)
        state.add_edge(mapExit, None, outputList[0], None, memletOutputOuter)
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        # add memlets from inner map to tasklet
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)
        # add memelets from tasklet to cr
        state.add_edge(tasklet, "out", mapExit2, None, memletTempNode)

    def visit_AvgPoolGrad(self, node):
        assert str(node.get_attr("padding"))[2:-1] == "VALID"
        strides_0 = node.get_attr("strides")[1]
        strides_1 = node.get_attr("strides")[2]
        ksize_0 = node.get_attr("ksize")[1]
        ksize_1 = node.get_attr("ksize")[2]
        backpropGrads, backpropParams, backpropDims = self.create_and_add_input_node(node.inputs[1])
        outputNode = self.create_and_add_output_node(node)[0]
        outputParams = [
            "i0",
            "i1*" + str(strides_0) + "+i4",
            "i2*" + str(strides_1) + "+i5",
            "i3",
        ]
        outputDims = self.get_default_dims(node.outputs[0])
        outerMapLabel = string_builder(node.type) + "_outer"
        outerMapParams = backpropParams
        outerMapDims = backpropDims
        outerMapEntry, outerMapExit = self.state.add_map(outerMapLabel, dict(zip(outerMapParams, outerMapDims)))
        innerMapLabel = string_builder(node.type) + "_inner"
        innerMapParams = ["i4", "i5"]
        innerMapDims = ["0:" + str(ksize_0), "0:" + str(ksize_1)]
        innerMapEntry, innerMapExit = self.state.add_map(
            innerMapLabel,
            dict(zip(innerMapParams, innerMapDims)),
            schedule=dace.ScheduleType.Sequential,
        )
        normalisationScalar = ksize_0 * ksize_1
        tasklet = self.state.add_tasklet(
            string_builder(node.type),
            {"backpropGrad"},
            {"outpGrad"},
            "outpGrad = backpropGrad / " + str(normalisationScalar),
        )
        self.add_in_memlets(
            [backpropGrads],
            outerMapEntry,
            innerMapEntry,
            [backpropDims],
            [backpropParams],
        )
        self.state.add_edge(
            innerMapEntry,
            None,
            tasklet,
            "backpropGrad",
            Memlet.simple(backpropGrads, ",".join(backpropParams)),
        )
        self.state.add_edge(
            tasklet,
            "outpGrad",
            innerMapExit,
            None,
            Memlet.simple(
                outputNode,
                ",".join(outputParams),
                wcr_str="lambda a,b: a+b",
            ),
        )
        self.add_out_memlets(
            [outputNode],
            outerMapExit,
            innerMapExit,
            [outputDims],
            [outputParams],
            "lambda a, b: a+b",
            0,
        )

    def visit_Relu(self, node):

        inputList = []
        inputNodes = []
        state = self.state
        inputParams = []
        inputDims = []

        for inp in node.inputs:

            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)

        mapLabel = string_builder(node.type)
        mapParams = []
        mapRange = []
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = max(dace.float32(0),j0)")
        self.add_out_memlets(outputList, mapExit, tasklet, inputDims, inputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_ShapeN(self, node):
        outputLabels = [string_builder(op.name) for op in node.outputs]

        inputNodes = []
        for n in node.inputs:
            inputNodes.append(self.create_and_add_input_node(n)[0])

        shapes = [
            np.array(input_tensor.shape, dtype=_tensortype(node.outputs[i]))
            for i, input_tensor in enumerate(node.inputs)
        ]

        for label, shape, outputTensor, inputNode in zip(outputLabels, shapes, node.outputs, inputNodes):
            self.constDict[label] = shape
            # Make outputs as non transients
            try:
                outpNode = _find_node(self.state, label)
            except (LookupError):
                outpNode = self.state.add_array(
                    label,
                    _tensorshape(outputTensor),
                    _tensortype(outputTensor),
                    lifetime=dtypes.AllocationLifetime.SDFG,
                )
            outpNode.desc(self.graph).transient = False

    def visit_Reshape(self, node):

        inputNode, params, dims = self.create_and_add_input_node(node.inputs[0])
        outputList = self.create_and_add_output_node(node)
        outputParams = [self.get_default_params(node.outputs[0])]
        outputDims = [self.get_default_dims(node.outputs[0])]
        memlet_reshape = Memlet.simple(inputNode, ",".join(dims), other_subset_str=",".join(outputDims[0]))
        self.state.add_edge(inputNode, None, outputList[0], None, memlet_reshape)

    # CUDNN may have different behaviour!
    def visit_MaxPoolGrad(self, node):
        state = self.state
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []
        inputList = []
        inputNodes = []
        strides = int(node.get_attr("strides")[1])
        ksize = node.get_attr("ksize")[2]
        for count, inp in enumerate(node.inputs):

            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

            # params = []

            # for ndims, dim in enumerate(inp.shape):
            # if (not count == 0) and (ndims == 1 or ndims == 2):
            #    params.append("(i" + str(ndims) + "/2)")

            # else:
            #    params.append("i" + str(ndims))

            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        mapLabel = string_builder(node.type)

        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(str(inputList[0].shape))

        # tempNode = state.add_transient(
        #    string_builder(node.name + "_tmp"), shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG
        # )
        tempNode = outputList[0]
        tempList = [tempNode]

        outputDims = inputDims
        # Copy as we manipulate inputParams but don't want map params/range to
        # change
        mapParams = inputParams[0].copy()
        mapRange = inputDims[1].copy()

        mapEntry, mapExit = state.add_map(mapLabel + "_map1_1", dict(zip(mapParams, mapRange)))

        mapParams_remainder = ["i4", "i5"]
        mapRange_remainder = ["0:" + str(ksize), "0:" + str(ksize)]
        mapEntry_remainder, mapExit_remainder = state.add_map(mapLabel + "_map1_2",
                                                              dict(zip(mapParams_remainder, mapRange_remainder)))
        tasklet = state.add_tasklet(
            mapLabel + "_map1",
            {"j0", "j1", "j2"},
            {"out"},
            "if (j0==j1):\n\tout = j2\nelse:\n\tout = 0",
        )
        innerParams = []
        innerParams.append(["i0", str(strides) + "*i1+i4", str(strides) + "*i2+i5", "i3"])
        innerParams.append(["i0", "i1", "i2", "i3"])
        innerParams.append(["i0", "i1", "i2", "i3"])
        self.add_out_memlets(
            tempList,
            mapExit,
            mapExit_remainder,
            outputDims,
            outputDims,
            wcr="lambda a, b: a+b",
            wcr_identity=0,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry_remainder, inputDims.copy(), inputDims.copy())
        for index, node in enumerate(inputNodes):
            self.state.add_edge(
                mapEntry_remainder,
                None,
                tasklet,
                "j" + str(index),
                Memlet.simple(node, ",".join(innerParams[index])),
            )
        self.state.add_edge(
            tasklet,
            "out",
            mapExit_remainder,
            None,
            Memlet.simple(
                tempList[0],
                ",".join(innerParams[0]),
                wcr_str="lambda a, b: a+b",
            ),
        )

        # Second map:
        # as we don't have the indices of the maxpooling we need to manually
        # figure out which one contributed. If it is ambiguous we break the
        # tie by the following priority k[i,j]<k[i+1,j]...<k[0,j+1]...

    #        newDims = [inputDims[0]] * 4
    #        mapRange = inputDims[1]
    #        mapRange[1] += ":"+str(strides)
    #        mapRange[2] += ":"+str(strides)
    #
    #        newParams = [inputParams[0]]
    #        # 2x2 kernel
    #        newParams = [
    #            ["i0", "i1", "i2", "i3"],
    #            ["i0", "i1+1", "i2", "i3"],
    #            ["i0", "i1", "i2+1", "i3"],
    #            ["i0", "i1+1", "i2+1", "i3"],
    #        ]
    #
    #        string = """
    # if(j0!=0):
    #        out0=j0
    #        out1=0
    #        out2=0
    #        out3=0
    # elif(j1!=0):
    #        out0=j0
    #        out1=j1
    #        out2=0
    #        out3=0
    # elif(j2!=0):
    #        out0=j0
    #        out1=j1
    #        out2=j2
    #        out3=0
    # else:
    #        out0=j0
    #        out1=j1
    #        out2=j2
    #        out3=j3
    # """
    #        tasklet = state.add_tasklet(
    #            mapLabel + "_map2",
    #            {"j0", "j1", "j2", "j3"},
    #            {"out0", "out1", "out2", "out3"},
    #            string,
    #        )
    #        mapEntry, mapExit = state.add_map(
    #            mapLabel + "_map2", dict(zip(mapParams, mapRange))
    #        )
    #        self.add_out_memlets(outputList * 4, mapExit, tasklet, newDims, newParams)
    #        self.add_in_memlets(tempList * 4, mapEntry, tasklet, newDims, newParams)

    def visit_ReluGrad(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for inp in node.inputs:
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            dims = self.get_default_dims(out)
            params = self.get_default_params(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "if (j1>0):\n\tout = j0\nelse:\n\tout = 0")
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_BiasAddGrad(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            outputParams.append([inputParams[0][-1]])
            outputDims.append([inputDims[0][-1]])

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.reinitCR(outputList[0], outputParams, outputDims, "0")
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams, "lambda a,b: a+b", 0)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Conv2DBackpropInput(self, node):
        inputNodes = []
        mapParams = []
        outputParams = []
        mapRange = []
        outputDims = []
        inputParams = []
        inputDims = []
        strides = int(node.get_attr("strides")[1])
        state = self.state

        for count, inp in enumerate(node.inputs):
            if not count == 0:
                inputNode, _, dim = self.create_and_add_input_node(inp)
                inputNodes.append(inputNode)
                inputDims.append(dim)

        outputList = self.create_and_add_output_node(node)
        outputParams = [["i0", "i1", "i2", "i4"]]
        outputDims.append(self.get_default_dims(node.outputs[0]))

        ksize = int(node.inputs[1].shape[0])
        if str(node.get_attr("padding"))[2:-1] == "SAME":
            padding = int(strides * (int(node.inputs[2].shape[1]) - 1) + ksize -
                          int(outputList[0].desc(self.graph).shape[1]))
        else:
            padding = 0

        if padding > 0:
            # If padding is even (padding is on each side the same)
            if padding % 2 == 0:
                paddingUp = padding // 2
                paddingDown = padding // 2
            # If padding is uneven, we pad more on the bottom and on the right side
            # of an image (matching TensorFlow behavior)
            else:
                paddingUp = padding // 2
                paddingDown = paddingUp + 1
            paddedOutputDims = outputDims[0].copy()
            paddedOutputDims[1] += "+" + str(padding)
            paddedOutputDims[2] += "+" + str(padding)
            paddedOutput = state.add_transient(
                string_builder(node.outputs[0].name) + "_padded",
                [
                    paddedOutputDims[0][2:],
                    paddedOutputDims[1][2:],
                    paddedOutputDims[2][2:],
                    paddedOutputDims[3][2:],
                ],
                _tensortype(node.outputs[0]),
            )

        if strides > 1:
            # Dilate and pad the incoming gradients
            newShape = [
                node.inputs[2].shape[0],
                node.inputs[2].shape[1] + (node.inputs[2].shape[1] - 1) * (strides - 1) + 2 * (ksize - 1),
                node.inputs[2].shape[2] + (node.inputs[2].shape[2] - 1) * (strides - 1) + 2 * (ksize - 1),
                node.inputs[2].shape[3],
            ]
            if newShape[1] - ksize + 1 < node.outputs[0].shape[1]:
                newShape[1] = node.outputs[0].shape[1] + ksize - 1
                newShape[2] = node.outputs[0].shape[2] + ksize - 1
            expandedGrads = state.add_transient(
                string_builder(node.inputs[2].name) + "_bigger_strided",
                newShape,
                _tensortype(node.inputs[2]),
            )
            expandedGrads.setzero = True
            mapParams = self.get_default_params(node.inputs[2])
            mapRange = self.get_default_dims(node.inputs[2])
            mapLabel = string_builder(node.type) + "_grad_expansion"
            mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
            tasklet = self.state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
            self.add_in_memlets([inputNodes[1]], mapEntry, tasklet, [mapRange], [mapParams])
            expandedGradParams = [
                "i0",
                str(ksize - 1) + "+" + "i1*" + str(strides),
                str(ksize - 1) + "+" + "i2*" + str(strides),
                "i3",
            ]
            expandedGradDims = ["0:" + str(_shape) for _shape in newShape]
            self.add_out_memlets(
                [expandedGrads],
                mapExit,
                tasklet,
                [expandedGradDims],
                [expandedGradParams],
            )
            inputNodes[1] = expandedGrads
            inputDims[1] = expandedGradDims

        elif ksize > 1:
            newShape = [
                node.inputs[2].shape[0],
                node.inputs[2].shape[1] + 2 * (ksize - 1),
                node.inputs[2].shape[2] + 2 * (ksize - 1),
                node.inputs[2].shape[3],
            ]
            if newShape[1] - ksize + 1 < node.outputs[0].shape[1]:
                newShape[1] = node.outputs[0].shape[1] + ksize - 1
                newShape[2] = node.outputs[0].shape[2] + ksize - 1
            expandedGrads = state.add_transient(
                string_builder(node.inputs[2].name) + "_bigger",
                newShape,
                _tensortype(node.inputs[2]),
            )
            expandedGrads.setzero = True
            expanderMemlet = Memlet.simple(
                inputNodes[1],
                ",".join(inputDims[1]),
                other_subset_str=",".join([
                    inputDims[1][0],
                    str(ksize - 1) + ":" + str(ksize - 1) + "+" + str(node.inputs[2].shape[1]),
                    str(ksize - 1) + ":" + str(ksize - 1) + "+" + str(node.inputs[2].shape[2]),
                    inputDims[1][3],
                ]),
            )
            state.add_edge(inputNodes[1], None, expandedGrads, None, expanderMemlet)
            expandedGradDims = ["0:" + str(_shape) for _shape in newShape]
            inputNodes[1] = expandedGrads
            inputDims[1] = expandedGradDims

        # Kernel params
        inputParams.append(["-1-i5+" + str(ksize), "-1-i6+" + str(ksize), "i4", "i3"])

        # Gradient params
        inputParams.append(["i0", "i1" + "+i5", "i2" + "+i6", "i3"])

        mapLabel = string_builder(node.type)
        mapParams = ["i0", "i1", "i2", "i4"]
        mapParams2 = ["i5", "i6", "i3"]
        mapRange = (paddedOutputDims if padding > 0 else outputDims[0])  # gradient dimensions
        mapRange2 = inputDims[0][:-2] + [inputDims[0][-1]]  # Kernel dimensions
        mapEntry, mapExit = state.add_map(mapLabel + "_outer", dict(zip(mapParams, mapRange)))
        mapEntry2, mapExit2 = state.add_map(mapLabel + "_inner", dict(zip(mapParams2, mapRange2)))

        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0 * j1")

        reduce_node = self.state.add_transient(
            mapLabel + "_wcr_avoid",
            [1],
            outputList[0].desc(self.graph).dtype,
            storage=dace.StorageType.Register,
        )
        reduce_node.setzero = True

        if padding > 0:
            self.add_out_memlets(
                [paddedOutput],
                mapExit,
                reduce_node,
                #mapExit2,
                [paddedOutputDims],
                outputParams,
            )
            nonpaddedsubset = paddedOutputDims.copy()
            nonpaddedsubset[1] = (str(paddingUp) + ":" + str(outputList[0].desc(self.graph).shape[1] + paddingUp))
            nonpaddedsubset[2] = (str(paddingUp) + ":" + str(outputList[0].desc(self.graph).shape[2] + paddingUp))
            self.state.add_edge(
                paddedOutput,
                None,
                outputList[0],
                None,
                Memlet.simple(
                    paddedOutput,
                    ",".join(nonpaddedsubset),
                    other_subset_str=",".join(outputDims[0]),
                ),
            )

        else:
            self.add_out_memlets(
                outputList,
                mapExit,
                reduce_node,
                #mapExit2,
                outputDims,
                outputParams,
            )

        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)
        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)

        memlet = Memlet.simple(
            reduce_node,
            #paddedOutput if padding > 0 else outputList[0],
            "0",
            wcr_str="lambda a,b: a+b",
        )
        state.add_edge(tasklet, "out", mapExit2, None, memlet)
        state.add_edge(mapExit2, None, reduce_node, None, memlet)

    def visit_Conv2DBackpropFilter(self, node):
        # convolve loss over input.
        # may need to dilate loss and may need to pad input (no correlation)

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []
        strides = int(node.get_attr("strides")[1])
        # Input, filtersizes, out_backprop
        ksize = int(node.outputs[0].shape[0])
        if str(node.get_attr("padding"))[2:-1] == "SAME":
            padding = int(strides * (int(node.inputs[2].shape[1]) - 1) + ksize - int(node.inputs[0].shape[1]))
        else:
            padding = 0

        for count, inp in enumerate(node.inputs):
            if count != 1:
                inputNode, _, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputDims.append(dims)
        inputParams.append(["i0", "i1+i5", "i2+i6", "i3"])
        inputParams.append(["i0", "i1", "i2", "i4"])

        # inputNodes looks like [input, out_backprop]

        if padding > 0:
            paddedInput, paddedDims = self.inputPadding(
                node,
                inputNodes[0],
                inputList[0],
                int(node.inputs[2].shape[1]),
                ksize,
                strides,
                inputDims[0],
            )
            inputNodes[0] = paddedInput
            inputDims[0] = paddedDims

        if strides > 1:
            # Dilate and the incoming gradients
            newShape = [
                node.inputs[2].shape[0],
                node.inputs[2].shape[1] + (node.inputs[2].shape[1] - 1) * (strides - 1),
                node.inputs[2].shape[2] + (node.inputs[2].shape[2] - 1) * (strides - 1),
                node.inputs[2].shape[3],
            ]
            expandedGrads = state.add_transient(
                string_builder(node.inputs[2].name) + "_bigger",
                newShape,
                _tensortype(node.inputs[2]),
            )
            expandedGrads.setzero = True
            mapParams = self.get_default_params(node.inputs[2])
            mapRange = self.get_default_dims(node.inputs[2])
            mapLabel = string_builder(node.type) + "_grad_expansion"
            mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
            tasklet = self.state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
            self.add_in_memlets([inputNodes[1]], mapEntry, tasklet, [mapRange], [mapParams])
            expandedGradParams = [
                "i0",
                "i1*" + str(strides),
                "i2*" + str(strides),
                "i3",
            ]
            expandedGradDims = ["0:" + str(_shape) for _shape in newShape]
            self.add_out_memlets(
                [expandedGrads],
                mapExit,
                tasklet,
                [expandedGradDims],
                [expandedGradParams],
            )
            inputNodes[1] = expandedGrads
            inputDims[1] = expandedGradDims

        outputList = self.create_and_add_output_node(node)
        for count, out in enumerate(node.outputs):
            params = ["i5", "i6", "i3", "i4"]
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapParams = outputParams[0]
        mapParams2 = inputParams[1][:-1]
        mapRange = outputDims[0]
        mapRange2 = inputDims[1][:-1]
        mapLabel = string_builder(node.type)
        mapEntry, mapExit = state.add_map(mapLabel + "_outer", dict(zip(mapParams, mapRange)))
        mapEntry2, mapExit2 = state.add_map(mapLabel + "_inner", dict(zip(mapParams2, mapRange2)))

        tasklet = state.add_tasklet(mapLabel, {"j0", "j1"}, {"out"}, "out = j0*j1")

        reduce_node = self.state.add_transient(
            mapLabel + "_wcr_avoid",
            [1],
            outputList[0].desc(self.graph).dtype,
            storage=dace.StorageType.Register,
        )
        reduce_node.setzero = True

        self.add_out_memlets(
            outputList,
            mapExit,
            reduce_node,
            #mapExit2,
            outputDims,
            outputParams,
            #"lambda a,b: a+b",
            #0,
        )
        self.add_in_memlets(inputNodes, mapEntry, mapEntry2, inputDims, inputParams)

        for i, inp in enumerate(inputNodes):
            name = "j" + str(i)
            memlet = Memlet.simple(inp, ",".join(inputParams[i]))
            state.add_edge(mapEntry2, None, tasklet, name, memlet)

        #for i, out in enumerate(outputList):
        memlet = Memlet.simple(
            reduce_node,
            #out,
            "0",
            wcr_str="lambda a,b: a+b",
        )
        state.add_edge(tasklet, "out", mapExit2, None, memlet)
        state.add_edge(mapExit2, None, reduce_node, None, memlet)

    def visit_SparseSoftmaxCrossEntropyWithLogits(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        inputParams = []
        inputDims = []

        for inp in node.inputs:
            inputNode, params, dims = self.create_and_add_input_node(inp)
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            inputDims.append(dims)
            inputParams.append(params)

        for out in node.outputs:
            label = string_builder(out.name)
            try:
                outputNode = _find_node(state, label)
            except (LookupError):
                dtype = dace.typeclass(_tensortype(node))
                shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(out)))
                outputNode = state.add_transient(label, shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG)
            outputList.append(outputNode)

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        # 1st map, get maximum in each batchsize dimension
        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(str(inputList[1].shape))

        temp1Node = state.add_transient(mapLabel + "_max_tmp", shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(
            mapLabel + "_max",
            dict(zip(mapParams, mapRange)),
            schedule=dace.ScheduleType.Sequential,
        )
        tasklet = state.add_tasklet(mapLabel + "_max", {"j0"}, {"out"}, "out = j0")
        self.reinitCR(temp1Node, [inputParams[1]], [inputDims[1]], "-999999999999")
        self.add_in_memlets([inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputParams[0]])
        self.add_out_memlets(
            [temp1Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: max(a,b)",
            -9999999999,
        )

        # 2nd map, calculate the denominator sum
        temp2Node = state.add_transient(mapLabel + "_denominator_tmp",
                                        shape,
                                        dtype,
                                        lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(
            mapLabel + "_denominator",
            dict(zip(mapParams, mapRange)),
            schedule=dace.ScheduleType.Sequential,
        )
        tasklet = state.add_tasklet(mapLabel + "_denominator", {"j0", "j1"}, {"out"},
                                    "out = dace::math::exp(j0-j1);",
                                    language=dace.dtypes.Language.CPP)
        self.reinitCR(temp2Node, [inputParams[1]], [inputDims[1]], "0")
        inList = [inputNodes[0], temp1Node]
        self.add_in_memlets(inList, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(
            [temp2Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: a+b",
            0,
        )

        # 3rd map, calculate the sofmax
        shape = dace.properties.ShapeProperty.from_string(str(inputList[0].shape))
        temp3Node = state.add_transient(mapLabel + "_softmax_tmp",
                                        shape,
                                        dtype,
                                        lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(mapLabel + "_softmax", dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel + "_softmax", {"j0", "j1", "j2"}, {"out"},
                                    "out = (dace::math::exp(j0-j1))/j2;",
                                    language=dace.dtypes.Language.CPP)
        inList = [inputNodes[0], temp1Node, temp2Node]
        paramsList = inputParams + [inputParams[1]]
        dimsList = inputDims + [inputDims[1]]
        self.add_in_memlets(inList, mapEntry, tasklet, dimsList, paramsList)
        self.add_out_memlets([temp3Node], mapExit, tasklet, [inputDims[0]], [inputParams[0]])

        # 4th map, calculate the cross-entropy loss for an optional loss output
        # mapEntry, mapExit = state.add_map(
        #    mapLabel + "_loss",
        #    dict(zip(mapParams, mapRange)),
        #    schedule=dace.ScheduleType.Sequential,
        # )
        # tasklet = state.add_tasklet(
        #    mapLabel + "_loss",
        #    {"j0", "j1"},
        #    {"out"},
        #    "if (int(j1) == i1) {\n\tout=-(dace::math::log(j0));}\nelse{\n\tout=0;}",
        #    language=dace.dtypes.Language.CPP,
        # )
        # self.reinitCR(outputList[0], [inputParams[1]], [inputDims[1]], "0")
        # self.add_in_memlets(
        #    [temp3Node, inputNodes[1]], mapEntry, tasklet, inputDims, inputParams
        # )
        # self.add_out_memlets(
        #    [outputList[0]],
        #    mapExit,
        #    tasklet,
        #    [inputDims[1]],
        #    [inputParams[1]],
        #    "lambda a,b: a+b",
        #    0,
        # )

        # 5th map, gradient of the whole layer
        mapEntry, mapExit = state.add_map(mapLabel + "_gradient", dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(
            mapLabel + "_gradient",
            {"j0", "j1"},
            {"out"},
            "if(int(j1)==i1):\n\tout = j0-1\nelse:\n\tout = j0",
        )
        self.add_out_memlets([outputList[1]], mapExit, tasklet, [inputDims[0]], [inputParams[0]])
        self.add_in_memlets([temp3Node, inputNodes[1]], mapEntry, tasklet, inputDims, inputParams)

    def visit_Identity(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        inputParams = []
        inputDims = []

        # Create input node and its params
        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        memlet = Memlet.simple(inputNodes[0], ",".join(inputDims[0]))
        state.add_edge(inputNodes[0], None, outputList[0], None, memlet)

    def visit_LRNGrad(self, node):

        inputList = []
        inputNodes = []
        outputList = []
        state = self.state

        alpha = str(node.get_attr("alpha"))
        beta = str(node.get_attr("beta"))
        bias = str(node.get_attr("bias"))
        depth_radius = str(node.get_attr("depth_radius"))

        for count, inp in enumerate(node.inputs):
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            if count == 0:
                shortDims = []
                shortAccesses = []
                for dim in inp.shape:
                    shortDims.append("0:" + str(dim))
                    shortAccesses.append(str(dim))
                longDims = []
                longDims = shortDims + ["0:" + depth_radius + "*2+1"]
                paddedDims = []
                paddedDims += shortDims
                paddedDims[-1] += "+" + depth_radius + "*2"

        label = string_builder(node.name)
        outputList = self.create_and_add_output_node(node)
        longParams = ["i0", "i1", "i2", "i3", "i4"]
        shortParams = ["i0", "i1", "i2", "i3"]
        copyParams = ["i0", "i1", "i2", "i3+" + depth_radius]
        normParams = ["i0", "i1", "i2", "i3+i4"]

        paddedShape = []
        paddedShape += shortAccesses
        paddedShape[-1] += "+" + depth_radius
        paddedInput = state.add_transient(
            label + "_paddedInput",
            paddedShape,
            dace.typeclass(_tensortype(node)),
            lifetime=dtypes.AllocationLifetime.SDFG,
        )
        mapEntry, mapExit = state.add_map(label + "_padding", dict(zip(shortParams, shortDims)))
        tasklet = state.add_tasklet(label + "_padding", {"j0"}, {"out"}, "out=j0")
        self.add_in_memlets([inputNodes[2]], mapEntry, tasklet, [shortDims], [shortParams])
        self.add_out_memlets([paddedInput], mapExit, tasklet, [paddedDims], [copyParams])

        sqrsum = state.add_transient(label + "_Sqrsum",
                                     shortAccesses,
                                     _tensortype(node),
                                     lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(label + "_sqrsum", dict(zip(longParams, longDims)))
        tasklet = state.add_tasklet(label + "_sqrsum", {"j0"}, {"out"}, "out=j0*j0")
        self.reinitCR(sqrsum, [shortParams], [shortDims], "0")
        self.add_in_memlets([paddedInput], mapEntry, tasklet, [paddedDims], [normParams])
        self.add_out_memlets([sqrsum], mapExit, tasklet, [shortDims], [shortParams], "lambda a,b: a+b", 0)

        label = string_builder(node.name)
        norm = state.add_transient(label + "_Norm",
                                   shortAccesses,
                                   _tensortype(node),
                                   lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(label + "_norm", dict(zip(shortParams, shortDims)))
        tasklet = state.add_tasklet(label + "_norm", {"j0"}, {"out"}, "out=" + alpha + "*j0+" + bias)
        self.add_in_memlets([sqrsum], mapEntry, tasklet, [shortDims], [shortParams])
        self.add_out_memlets([norm], mapExit, tasklet, [shortDims], [shortParams])

        preOut = state.add_transient(label + "_preOut",
                                     shortAccesses,
                                     _tensortype(node),
                                     lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(label, dict(zip(longParams, longDims)))
        taskletCode = ("if (i4==" + depth_radius + "){\n out = pow(j2," + beta + ")-2*" + alpha + "*" + beta +
                       "*j1*j0/j2;}\n else{\n out = -2*" + alpha + "*" + beta + "*j1*j0/j2;}")
        tasklet = state.add_tasklet(label, {"j0", "j1", "j2"}, {"out"}, taskletCode, language=dace.dtypes.Language.CPP)
        self.reinitCR(preOut, [shortParams], [shortDims], "0")
        inList = [inputNodes[1]]
        inList.append(paddedInput)
        inList.append(norm)
        self.add_in_memlets(
            inList,
            mapEntry,
            tasklet,
            [shortDims, paddedDims, shortDims],
            [shortParams, normParams, shortParams],
        )
        self.add_out_memlets([preOut], mapExit, tasklet, [shortDims], [shortParams], "lambda a,b: a+b", 0)

        mapEntry, mapExit = state.add_map(label + "_out", dict(zip(shortParams, shortDims)))
        tasklet = state.add_tasklet(label + "_out", {"j0", "j1"}, {"out"}, "out=j0*j1")
        self.add_in_memlets(
            [inputNodes[0], preOut],
            mapEntry,
            tasklet,
            [shortDims, shortDims],
            [shortParams, shortParams],
        )
        self.add_out_memlets(outputList, mapExit, tasklet, [shortDims], [shortParams])

    def visit_LRN(self, node):

        inputList = []
        inputNodes = []
        outputList = []
        state = self.state
        alpha = str(node.get_attr("alpha"))
        beta = str(node.get_attr("beta"))
        bias = str(node.get_attr("bias"))
        depth_radius = str(node.get_attr("depth_radius"))

        for count, inp in enumerate(node.inputs):
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)
            if count == 0:
                shortDims = []
                shortAccesses = []
                for dim in inp.shape:
                    shortDims.append("0:" + str(dim))
                    shortAccesses.append(str(dim))
                longDims = []
                longDims = shortDims + ["0:" + depth_radius + "*2+1"]
                paddedDims = []
                paddedDims += shortDims
                paddedDims[-1] += "+" + depth_radius + "*2"

        label = string_builder(node.name)
        outputList = self.create_and_add_output_node(node)
        longParams = ["i0", "i1", "i2", "i3", "i4"]
        shortParams = ["i0", "i1", "i2", "i3"]
        copyParams = ["i0", "i1", "i2", "i3+" + depth_radius]
        normParams = ["i0", "i1", "i2", "i3+i4"]

        paddedShape = []
        paddedShape += shortAccesses
        paddedShape[-1] += "+" + depth_radius
        paddedInput = state.add_transient(
            label + "_paddedInput",
            paddedShape,
            dace.typeclass(_tensortype(node)),
            lifetime=dtypes.AllocationLifetime.SDFG,
        )
        mapEntry, mapExit = state.add_map(label + "_padding", dict(zip(shortParams, shortDims)))
        tasklet = state.add_tasklet(label + "_padding", {"j0"}, {"out"}, "out=j0")
        self.add_in_memlets([inputNodes[0]], mapEntry, tasklet, [shortDims], [shortParams])
        self.add_out_memlets([paddedInput], mapExit, tasklet, [paddedDims], [copyParams])

        sqrsum = state.add_transient(label + "_Sqrsum",
                                     shortAccesses,
                                     _tensortype(node),
                                     lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(label + "_sqrsum", dict(zip(longParams, longDims)))
        tasklet = state.add_tasklet(label + "_sqrsum", {"j0"}, {"out"}, "out=j0*j0")
        self.reinitCR(sqrsum, [shortParams], [shortDims], "0")
        self.add_in_memlets([paddedInput], mapEntry, tasklet, [paddedDims], [normParams])
        self.add_out_memlets([sqrsum], mapExit, tasklet, [shortDims], [shortParams], "lambda a,b: a+b", 0)

        mapEntry, mapExit = state.add_map(label, dict(zip(shortParams, shortDims)))
        tasklet = state.add_tasklet(string_builder(node.name), {"j0", "j1"}, {"out"},
                                    "out = j0/(pow(" + bias + "+" + alpha + "*j1," + beta + "));",
                                    language=dace.dtypes.Language.CPP)

        self.add_in_memlets(
            (inputNodes + [sqrsum]),
            mapEntry,
            tasklet,
            [shortDims, shortDims],
            [shortParams, shortParams],
        )
        self.add_out_memlets(outputList, mapExit, tasklet, [shortDims], [shortParams])

    def visit_ArgMax(self, node):

        state = self.state
        inputList = []
        inputNodes = []

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode = self.create_and_add_input_node(inp)[0]
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)

                inputAccesses = [[], []]
                inputDims = [[], []]
                inputParams = [[], []]
                for i, dim in enumerate(inp.shape):
                    if i == 0:
                        inputAccesses[1].append(str(dim))
                        inputParams[1].append("i" + str(i))
                        inputDims[1].append("0:" + str(dim))
                    inputAccesses[0].append(str(dim))
                    inputParams[0].append("i" + str(i))
                    inputDims[0].append("0:" + str(dim))

        outputList = self.create_and_add_output_node(node)

        mapLabel = string_builder(node.name)
        mapEntry, mapExit = state.add_map(mapLabel + "_max", dict(zip(inputParams[0], inputDims[0])))
        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(",".join(inputAccesses[1]))
        temp1Node = state.add_transient(mapLabel + "_max_tmp", shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG)

        tasklet = state.add_tasklet(mapLabel + "_max", {"j0"}, {"out"}, "out = j0")
        self.reinitCR(temp1Node, [inputParams[1]], [inputDims[1]], "-999999999999")
        self.add_in_memlets([inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputParams[0]])
        self.add_out_memlets(
            [temp1Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: max(a,b)",
            -999999999999,
        )

        mapEntry, mapExit = state.add_map(mapLabel + "_arg", dict(zip(inputParams[0], inputDims[0])))
        outputNode = outputList[0]
        tasklet = state.add_tasklet(mapLabel + "_map2", {"j0", "j1"}, {"out"}, "if (j0==j1):\n\tout=i1")
        self.add_in_memlets([inputNodes[0], temp1Node], mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets([outputNode], mapExit, tasklet, [inputDims[1]], [inputParams[1]])

    def visit_Cast(self, node):

        state = self.state
        inputList = []
        inputNodes = []
        outputList = []
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []
        castType = None

        dtype = node.get_attr("DstT")
        if dtype.as_numpy_dtype == object:
            raise NotImplementedError("Type %s is not a valid numpy type" % str(dtype))
        castType = dace.typeclass(dtype.as_numpy_dtype).ctype

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            params = self.get_default_params(out)
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = " + castType + "(j0)")
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)

    def visit_Print(self, node):
        inputList = []
        inputNodes = []
        outputList = []
        state = self.state
        mapParams = []
        mapRange = []
        outputParams = []
        outputDims = []
        inputParams = []
        inputDims = []

        for count, inp in enumerate(node.inputs):
            if count == 0:
                inputNode, params, dims = self.create_and_add_input_node(inp)
                inputList.append(inputNode.desc(self.graph))
                inputNodes.append(inputNode)
                inputParams.append(params)
                inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        for out in node.outputs:
            params = self.get_default_params(out)
            dims = self.get_default_dims(out)
            outputParams.append(params)
            outputDims.append(dims)

        mapLabel = string_builder(node.type)
        mapParams = inputParams[0]
        mapRange = inputDims[0]
        mapEntry, mapExit = state.add_map(mapLabel, dict(zip(mapParams, mapRange)))

        ifClause = "if ("
        for param in mapParams:
            ifClause += param + "==1 and "

        ifClause = ifClause[:-4] + "):"
        taskletCode = ("out = j0\n" + ifClause + '\n\tprintf("' + inputList[0].label + '")\n')
        taskletCode = 'out = j0\nif(True):\n\tprintf("%f\\n",out)'
        tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, taskletCode)
        self.add_out_memlets(outputList, mapExit, tasklet, outputDims, outputParams)
        self.add_in_memlets(inputNodes, mapEntry, tasklet, inputDims, inputParams)

    def visit_Softmax(self, node):

        inputList = []
        inputNodes = []
        state = self.state

        for inp in node.inputs:
            inputNode = self.create_and_add_input_node(inp)[0]
            inputList.append(inputNode.desc(self.graph))
            inputNodes.append(inputNode)

        outputList = self.create_and_add_output_node(node)

        inputDims = [[], []]
        inputParams = [[], []]

        for i, dim in enumerate(inp.shape):
            if i == 0:
                inputParams[1].append("i" + str(i))
                inputDims[1].append("0:" + str(dim))
            inputParams[0].append("i" + str(i))
            inputDims[0].append("0:" + str(dim))

        mapLabel = string_builder(node.name)
        mapEntry, mapExit = state.add_map(mapLabel + "_map1", dict(zip(inputParams[0], inputDims[0])))
        mapParams = inputParams[0]
        mapRange = inputDims[0]

        # 1st map, get maximum in each batchsize dimension
        dtype = dace.typeclass(_tensortype(node))
        shape = dace.properties.ShapeProperty.from_string(str(node.inputs[0].shape.dims[0]))
        temp1Node = state.add_transient(mapLabel + "_max_tmp", shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(mapLabel + "_max", dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel + "_max", {"j0"}, {"out"}, "out = j0")
        self.reinitCR(temp1Node, [inputParams[1]], [inputDims[1]], "-999999999999")
        self.add_in_memlets([inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputParams[0]])
        self.add_out_memlets(
            [temp1Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: max(a,b)",
            -999999999999,
        )

        # 2nd map, calculate the denominator sum
        temp2Node = state.add_transient(mapLabel + "_denominator_tmp",
                                        shape,
                                        dtype,
                                        lifetime=dtypes.AllocationLifetime.SDFG)
        mapEntry, mapExit = state.add_map(mapLabel + "_denominator", dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel + "_denominator", {"j0", "j1"}, {"out"},
                                    "out = dace::math::exp(j0-j1);",
                                    language=dace.dtypes.Language.CPP)
        self.reinitCR(temp2Node, [inputParams[1]], [inputDims[1]], "0")
        inList = [inputNodes[0], temp1Node]
        self.add_in_memlets(inList, mapEntry, tasklet, inputDims, inputParams)
        self.add_out_memlets(
            [temp2Node],
            mapExit,
            tasklet,
            [inputDims[1]],
            [inputParams[1]],
            "lambda a,b: a+b",
            0,
        )

        # 3rd map, calculate the sofmax
        mapEntry, mapExit = state.add_map(mapLabel + "_softmax", dict(zip(mapParams, mapRange)))
        tasklet = state.add_tasklet(mapLabel + "_softmax", {"j0", "j1", "out"}, {"out"},
                                    "out = (dace::math::exp(j0-j1))/j2;",
                                    language=dace.dtypes.Language.CPP)
        inList = [inputList[0], temp1Node, temp2Node]
        paramsList = inputParams + [inputParams[1]]
        dimsList = inputDims + [inputDims[1]]
        self.add_in_memlets(inList, mapEntry, tasklet, dimsList, paramsList)
        self.add_out_memlets(outputList, mapExit, tasklet, [inputDims[0]], [inputParams[0]])

    def visit_AddN(self, node):
        inputNodes = []
        inputParams = []
        inputDims = []
        for count, inp in enumerate(node.inputs):
            inpNode, params, dims = self.create_and_add_input_node(inp)
            inputNodes.append(inpNode)
            inputParams.append(params)
            inputDims.append(dims)

        outputList = self.create_and_add_output_node(node)
        outputParams = self.get_default_params(node.outputs[0])
        outputDims = self.get_default_dims(node.outputs[0])
        jays = ["j" + str(index) for index in range(len(inputNodes))]
        tasklet, mapEntry, mapExit = self.state.add_mapped_tasklet(
            string_builder(node.type),
            dict(zip(inputParams[0], inputDims[0])),
            dict(zip(
                jays,
                [Memlet.simple(inode, ",".join(params)) for inode, params in zip(inputNodes, inputParams)],
            )),
            "out = " + "+".join(jays),
            dict(out=Memlet.simple(outputList[0], ",".join(outputParams))),
        )
        for inp, dim in zip(inputNodes, inputDims):
            self.state.add_edge(inp, None, mapEntry, None, Memlet.simple(inp, ",".join(dim)))
        self.state.add_edge(
            mapExit,
            None,
            outputList[0],
            None,
            Memlet.simple(outputList[0], ",".join(outputDims)),
        )

    def add_in_memlets(self, inputList, otherNode, tasklet, inputDims, inputParams, identifier="j"):
        """ Convenience function that adds two memlets for each input of the 
            node: external and internal to a given map.

            :param inputList: list of inputNodes (DaCe access node)
            :param otherNode: DaCe node (mostly map_entry)
            :param tasklet: Normally a tasklet node, but it can also be another
                            mapEntry, for example map in map.
            :param inputDims: List of list of strings dimension of the
                              respective input. Example:
                              [["0:5","0:7"],["0:2","0:4"]]  
            :param inputParams: List of list of strings params of respective
                                input. Example: [["i0","i1"],["i2","i3"]]
            :param identifier: This will be used as the base identifier of the
                                input connector to the tasklet. Default is 'j'  
        """
        state = self.state
        connected_nodes = set()
        for i, inp in enumerate(inputList):
            assert isinstance(inputDims[i], list)
            if inp.data not in connected_nodes:
                outerMemlet = Memlet.simple(inp, ",".join(inputDims[i]))
                state.add_edge(inp, None, otherNode, None, outerMemlet)
                connected_nodes.add(inp.data)
            name = identifier + str(i)
            innerMemlet = Memlet.simple(inp, ",".join(inputParams[i]))

            if isinstance(tasklet, (Tasklet, NestedSDFG)):
                state.add_edge(otherNode, None, tasklet, name, innerMemlet)
            else:
                state.add_edge(otherNode, None, tasklet, None, innerMemlet)

    def add_out_memlets(
        self,
        outputList,
        otherNode,
        tasklet,
        outputDims,
        outputParams,
        wcr=None,
        wcr_identity=None,
        identifier="out",
        wcr_conflict=True,
    ):
        """ Convenience function that adds two memlets for each output of the 
            node: external and internal to a given map.

            :param outputList: list of outputNodes (DaCe access node)
            :param otherNode: DaCe node (mostly map_entry)
            :param tasklet: Normally a tasklet node, but it can also be another
                            mapEntry, for example map in map.
            :param outputDims: List of list of strings dimension of the
                               respective output. Example:
                               [["0:5","0:7"],["0:2","0:4"]]  
            :param outputParams: List of list of strings params of respective
                                 output. Example: [["i0","i1"],["i2","i3"]]  
            :param wcr: (optional) Write-conflict resolution function (as
                        string).
            :param wcr_identity: (optional) Identity element for write-conflict
                                 resolution. Will be appended to init state.
            :param identifier: This is the base identifier for the out connector
                                of the tasklet. Default value is "out". If there are
                                multiple out connectors, each is numbered from zero.
            :param wcr_conflict: (optional) If False, specifies that this
                                 write-conflict resolution does not incur an
                                 atomic operation.
        """

        connected_nodes = set()

        state = self.state
        for i, out in enumerate(outputList):
            assert isinstance(outputDims[i], list)
            if len(outputList) > 1:
                name = identifier + str(i)
            else:
                name = identifier

            if out.data not in connected_nodes:
                if wcr_identity is not None:
                    self.add_init(out.data, wcr_identity)
                outerMemlet = Memlet.simple(
                    out,
                    ",".join(outputDims[i]),
                    wcr_str=wcr,
                    wcr_conflict=wcr_conflict,
                )
                state.add_edge(otherNode, None, out, None, outerMemlet)
                connected_nodes.add(out.data)
            innerMemlet = Memlet.simple(
                out,
                ",".join(outputParams[i]),
                wcr_str=wcr,
                wcr_conflict=wcr_conflict,
            )

            if isinstance(tasklet, (Tasklet, NestedSDFG)):
                state.add_edge(tasklet, name, otherNode, None, innerMemlet)
            else:
                state.add_edge(tasklet, None, otherNode, None, innerMemlet)

    def create_and_add_input_node(self, inp):
        """ Creates a DaCe access node for each input of `inp`, adds it to the 
            state, and returns it.
            If the node already exists, returns the pre-existing node.

            :param inp: tf.Operation
            :return: A 3-tuple of (input DaCe access node,
                                   list of parameter strings,
                                   list of dimension strings).
        """

        state = self.state
        # Get DaCe name of the operation
        label = string_builder(inp.name)
        if "?" in str(_tensorshape(inp)):
            raise ValueError  # ("Invalid shape for tensor %s" % label)
        # Try to find node in DaCe graph
        try:
            # If successful, use the existing node
            inputNode = _find_node(state, label)
        except (LookupError):
            # Get type and shape of the input tensor
            try:
                dtype = dace.typeclass(_tensortype(inp))
            except TypeError:
                raise TypeError
            shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(inp)))
            # Create and add array, default is transient allocated in the
            # beginning of the SDFG
            inputNode = state.add_transient(name=label,
                                            shape=shape,
                                            dtype=dtype,
                                            lifetime=dtypes.AllocationLifetime.SDFG)

        params = self.get_default_params(inp)
        dims = self.get_default_dims(inp)

        return inputNode, params, dims

    def create_and_add_output_node(self, node):
        """ Creates a DaCe access node for each output of `node`, adds it to 
            the state, and returns it.
            If the node already exists, returns the pre-existing node.

            :param node: tf.Operation
            :return: List of DaCe access node.
        """
        outputList = []
        state = self.state
        for count, out in enumerate(node.outputs):
            label = string_builder(out.name)
            if "?" in str(_tensorshape(out)):
                raise ValueError("Invalid shape {} for tensor {}".format(_tensorshape(out), label))
            # Iterate over all output nodes
            # Try to find node in DaCe graph
            try:
                # If successful, use the existing node
                outputNode = _find_node(state, label)
            except (LookupError):
                # Get type and shape of the tensor
                dtype = dace.typeclass(_tensortype(out))
                shape = dace.properties.ShapeProperty.from_string(str(_tensorshape(out)))
                outputNode = state.add_transient(label, shape, dtype, lifetime=dtypes.AllocationLifetime.SDFG)
            outputList.append(outputNode)
        return outputList

    def add_init(self, arrname: str, value: Any):
        """ Adds an initialization map for a tensor in the init state.

            :param arrname: The tensor name to initialize.
            :param dims: The range (as a string) to use for initialization.
            :param value: A value to set it to (converted to C++ string).
        """
        state: dace.SDFGState = self.reinitState
        data = self.graph.arrays[arrname]

        if isinstance(data, Scalar):
            state.add_mapped_tasklet('reinit_%s' % arrname, [('unused', '0:1')], {},
                                     'out = %s' % value, {'out': dace.Memlet.simple(arrname, '0')},
                                     external_edges=True)
        else:
            state.add_mapped_tasklet(
                'reinit_%s' % arrname, [('o%d' % i, '0:%s' % symstr(shp)) for i, shp in enumerate(data.shape)], {},
                'out = %s' % value,
                {'out': dace.Memlet.simple(arrname, ','.join('o%d' % i for i in range(len(data.shape))))},
                external_edges=True)

    def reinitCR(self, inp: dace.nodes.AccessNode, params, dims, identity):
        """ Adds a reinitialization map to a `reinit` state, setting inputs
            to their initial values. Only used in training mode.

            :param inp: DaCe access node.
            :param params: List of string parameters to `inp`.
            :param dims: List of strings dimensions of `inp`.
            :param identity: Identity value of the CR node (as a string).
        """
        return self.add_init(inp.data, identity)

    def inputPadding(self, node, inpnode, inp, outputSize, kernelSize, strides, inputDims):
        """ Zero-pads the input to fit the outputSize.
            WARNING: This function assumes the height and width of the output is the
            same (which is reasonable for deep learning).

            :param node: tf.Operation
            :param inpnode: DaCe access node to pad
            :param inp: input node descriptor
            :param outputSize: Output size. (int like)
            :param kernelSize: Kernel size.
            :param strides: Strides.
            :param inputDims: List of strings (e.g.["0:N","0:M"]).
            :return: A 2-tuple (output DaCe access node with padded input,
                                list of dimension strings of the padded data).
        """
        state = self.state
        paddingUp = 0
        paddingDown = 0
        label = inpnode.label
        inputSize = inp.shape[1]
        # Calculate padding according to paper
        padding = strides * (outputSize - 1) + kernelSize - inputSize
        # If padding is even (padding is on each side the same)
        if padding % 2 == 0:
            paddingUp = padding // 2
            paddingDown = padding // 2
        # If padding is uneven, we pad more on the bottom and on the right side
        # of an image (matching TensorFlow behavior)
        else:
            paddingUp = padding // 2
            paddingDown = paddingUp + 1

        # Set up the different padding dimensions, accesses and params.
        outputDims = inputDims.copy()
        outputDims[1] = str(paddingUp) + ":" + str(inp.shape[1]) + "+" + str(paddingUp)
        outputDims[2] = str(paddingUp) + ":" + str(inp.shape[2]) + "+" + str(paddingUp)
        padMemlet = Memlet.simple(inpnode, ",".join(inputDims), other_subset_str=",".join(outputDims))
        outputAccesses = list(map(str, list(inp.shape)))
        outputAccesses[1] += "+" + str(paddingUp) + "+" + str(paddingDown)
        outputAccesses[2] += "+" + str(paddingUp) + "+" + str(paddingDown)
        outputDims = []
        inputParams = []
        for i, dim in enumerate(outputAccesses):
            inputParams.append("i" + str(i))
            outputDims.append("0:" + dim)

        outputParams = inputParams.copy()
        outputParams[1] += "+" + str(paddingUp)
        outputParams[2] += "+" + str(paddingUp)

        # Add the padded input to the graph, set it to zero, and add the map.
        shape = dace.properties.ShapeProperty.from_string(",".join(outputAccesses))
        output = state.add_transient(label + "_padded",
                                     shape=shape,
                                     dtype=inp.dtype,
                                     lifetime=dtypes.AllocationLifetime.SDFG)
        output.setzero = True

        # mapParams = inputParams
        # mapRange = inputDims
        # mapLabel = string_builder(node.type)
        # mapEntry, mapExit = state.add_map(mapLabel,
        #                                 dict(zip(mapParams, mapRange)))
        # tasklet = state.add_tasklet(mapLabel, {"j0"}, {"out"}, "out = j0")
        self.state.add_edge(inpnode, None, output, None, padMemlet)
        # self.add_in_memlets([inpnode], mapEntry, tasklet, [inputDims],
        #                   [inputParams])
        # self.add_out_memlets([output], mapExit, tasklet, [outputDims],
        #                    [outputParams])
        return output, outputDims

    def get_default_params(self, tensor, start=0, identifier="i"):
        """ Returns the default parameters of a tensor starting at `start`,
            e.g., ["i0","i1",...].

            :param tensor: tf.Tensor.
            :param start: Starting position for the iteration.
            :param identifier: The base identifier for the parameters. Default is 'i'
            :return: List of parameters as strings ["i0",i"1",...].
        """
        params = []
        shape = _tensorshape(tensor)
        if shape == 1:
            shape = [1]
        for i, dim in enumerate(shape, start):
            params.append(identifier + str(i))
        return params

    def get_default_dims(self, tensor):
        """ Returns the default dimensions of a tensor e.g., ["0:N","0:M"]
        
            :param tensor: tf.Tensor.
            :return: List of dimensions as strings ["0:N","0:M"]
        """
        dims = []
        shape = _tensorshape(tensor)
        if shape == 1:
            shape = [1]
        for dim in shape:
            dims.append("0:" + str(dim))
        return dims
