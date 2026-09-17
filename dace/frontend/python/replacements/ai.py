# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains the Python frontend replacement for AI-implemented library nodes.

``dace.ai`` places an :class:`~dace.libraries.ai.nodes.ai_node.AINode` into the program being
parsed, so that a part of a data-centric program can be specified in natural language rather than
written out::

    @dace.program
    def matmul(A: dace.float32[N, N], B: dace.float32[N, N]):
        return dace.ai('Write _out[i][j] = sum over k of _a[i][k] * _b[k][j], using AVX2 FMA '
                       'intrinsics rather than a scalar triple loop.',
                       a=A, b=B)

The node is expanded like every other library node -- at compile time, through the reserved
``'ai'`` implementation -- so the model is asked once, and the tasklet it writes becomes part of
the SDFG.

Because the description *is* the specification, it has to be able to name the variables the
generated code will see. Those names are the node's connectors, and this replacement assigns them
so that they are predictable from the call site:

* a keyword input ``a=A`` becomes the connector ``_a``,
* a positional input becomes ``_in0``, ``_in1``, ... in the order given,
* the output is ``_out``, or ``_out0``, ``_out1``, ... when several are written through ``out=``.
"""

from numbers import Number
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Imported for its side effect on the annotations below: ``ProgramVisitor`` is a string at run time, and
# resolving it -- as the documentation build does -- needs ``dace`` in this module's namespace
import dace  # noqa
from dace import data, dtypes, symbolic
from dace import Memlet, SDFG, SDFGState
from dace.sdfg import nodes
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError, StringLiteral
from dace.frontend.python.replacements.utils import ProgramVisitor, Shape


def _is_container(sdfg: SDFG, value: Any) -> bool:
    """
    Tests whether an argument of the call refers to a data container.

    The Python frontend passes a data container to a replacement as its name, so a string that
    names one is a container rather than the text it looks like.

    :param sdfg: The SDFG being built.
    :param value: The argument, as parsed by the frontend.
    :return: True if the argument names a data container of the SDFG.
    """
    return type(value) is str and value in sdfg.arrays


def _description_of(pv: ProgramVisitor, sdfg: SDFG, description: Any) -> str:
    """
    Returns the natural-language description given at the call site.

    The frontend hands a string literal over as a :class:`~dace.frontend.python.common.StringLiteral`
    and a data container as its plain name, so a bare ``str`` that names a container of the program
    is an argument in the wrong position rather than a description.

    :param pv: The program visitor, for error reporting.
    :param sdfg: The SDFG being built, to tell a description from a container name.
    :param description: The first argument of the call, as parsed by the frontend.
    :return: The description as a string.
    :raises DaceSyntaxError: If the description is not a string literal, or is empty.
    """
    if _is_container(sdfg, description):
        given = f'the data container "{description}"'
    elif isinstance(description, (StringLiteral, str)):
        given = None
    else:
        given = f'a value of type "{type(description).__name__}"'

    if given is not None:
        raise DaceSyntaxError(
            pv, None, 'The first argument of dace.ai must be a string describing what the node computes, '
            f'but {given} was given. It has to be a string literal or a compile-time constant string, '
            'since it is the specification handed to the model.')
    text = str(description).strip()
    if not text:
        raise DaceSyntaxError(
            pv, None, 'The description given to dace.ai is empty, so there is nothing to '
            'generate code from.')
    return text


def _container_of(pv: ProgramVisitor, sdfg: SDFG, argument: Any, described: str) -> str:
    """
    Resolves one argument of the call to the data container it refers to.

    :param pv: The program visitor, for error reporting.
    :param sdfg: The SDFG being built.
    :param argument: The argument, as parsed by the frontend.
    :param described: How to refer to the argument in an error message.
    :return: The name of the data container.
    :raises DaceSyntaxError: If the argument is not a data container.
    """
    if _is_container(sdfg, argument):
        return argument

    if isinstance(argument, (Number, StringLiteral)) or symbolic.issymbolic(argument):
        raise DaceSyntaxError(
            pv, None, f'{described} of dace.ai is a constant or a symbol, and only data containers can be '
            'connected to a node. Symbols are already visible to the generated code by name, and a '
            'constant can simply be stated in the description.')
    raise DaceSyntaxError(pv, None, f'{described} of dace.ai is not a data container of this program.')


def _input_connectors(pv: ProgramVisitor, sdfg: SDFG, inputs: Sequence[Any], named_inputs: Dict[str,
                                                                                                Any]) -> Dict[str, str]:
    """
    Maps every input of the call to the connector it is read into.

    :param pv: The program visitor, for error reporting.
    :param sdfg: The SDFG being built.
    :param inputs: Positional inputs, which become ``_in0``, ``_in1``, ...
    :param named_inputs: Keyword inputs, where ``a=A`` becomes ``_a``.
    :return: A mapping from connector name to data container name, in call order.
    :raises DaceSyntaxError: If an input is not a data container, or two inputs claim one connector.
    """
    connectors: Dict[str, str] = {}
    for i, argument in enumerate(inputs):
        connectors[f'_in{i}'] = _container_of(pv, sdfg, argument, f'Positional input {i}')

    for keyword, argument in named_inputs.items():
        connector = f'_{keyword}'
        if connector in connectors:
            raise DaceSyntaxError(
                pv, None, f'Two inputs of dace.ai map to the connector "{connector}": the keyword argument '
                f'"{keyword}" collides with a positional input. Pass it by keyword under a different name.')
        connectors[connector] = _container_of(pv, sdfg, argument, f'Input "{keyword}"')

    return connectors


def _output_connectors(pv: ProgramVisitor, sdfg: SDFG, out: Any) -> Dict[str, str]:
    """
    Maps the containers the node writes to the connectors they are written from.

    :param pv: The program visitor, for error reporting.
    :param sdfg: The SDFG being built.
    :param out: One container name, or a sequence of them.
    :return: A mapping from connector name to data container name, in the given order.
    :raises DaceSyntaxError: If an output is not a data container, or the same one is written twice.
    """
    names = list(out) if isinstance(out, (list, tuple)) else [out]
    if not names:
        raise DaceSyntaxError(
            pv, None, 'The "out" argument of dace.ai is empty. Leave it out to have the '
            'output allocated, or name the containers to write to.')

    containers = [_container_of(pv, sdfg, name, f'Output {i}') for i, name in enumerate(names)]
    if len(set(containers)) != len(containers):
        raise DaceSyntaxError(
            pv, None, 'The same container is given twice in the "out" argument of dace.ai. '
            'A node cannot write to one container through two connectors.')

    if len(containers) == 1:
        return {'_out': containers[0]}
    return {f'_out{i}': name for i, name in enumerate(containers)}


def _allocate_output(pv: ProgramVisitor, sdfg: SDFG, inputs: Dict[str, str], shape: Optional[Shape],
                     dtype: Optional[dtypes.typeclass], storage: Optional[dtypes.StorageType]) -> str:
    """
    Adds the transient the node writes to when the call does not name one.

    What is not given is taken from the first input, which makes the common elementwise case
    (``dace.ai('...', a=A, b=B)``) work without repeating the shape and type of ``A``.

    :param pv: The program visitor, used to name the transient after the assignment target.
    :param sdfg: The SDFG being built.
    :param inputs: The input connectors, used to infer whatever was not given.
    :param shape: Shape of the output, or ``None`` to take the shape of the first input.
    :param dtype: Data type of the output, or ``None`` to take the type of the first input.
    :param storage: Storage of the output, or ``None`` to take the storage of the first input.
    :return: The name of the new transient.
    :raises DaceSyntaxError: If there is no input to infer the missing parts from.
    """
    prototype: Optional[data.Data] = None
    if inputs:
        prototype = sdfg.arrays[next(iter(inputs.values()))]

    if prototype is None and (shape is None or dtype is None):
        raise DaceSyntaxError(
            pv, None, 'dace.ai cannot tell what to allocate for the output of a node that has no inputs to '
            'take it from. Give "shape" and "dtype", or write into an existing container with "out".')

    if dtype is None:
        dtype = prototype.dtype
    if storage is None:
        storage = prototype.storage if prototype is not None else dtypes.StorageType.Default
    if shape is None:
        # A scalar input yields a scalar output, rather than an array of one element
        shape = () if isinstance(prototype, data.Scalar) else prototype.shape

    if not shape:
        name, _ = sdfg.add_scalar(pv.get_target_name(), dtype, storage=storage, transient=True, find_new_name=True)
        return name
    name, _ = pv.add_temp_transient(shape, dtype, storage=storage)
    return name


def _node_name(sdfg: SDFG, base: str = 'ai') -> str:
    """
    Returns a name no other library node of this SDFG carries.

    The name is not decoration: :func:`dace.libraries.ai.session.make_id` identifies a slot's
    conversation by the SDFG and node names, so two nodes sharing a name would share a session, and
    refining one of them would rewrite the other. The names are handed out in program order rather
    than derived from the assignment target, which keeps them short, and keeps the description --
    which is what the model is actually asked about -- out of the node's name.

    :param sdfg: The SDFG being built.
    :param base: The name to use if it is free.
    :return: The unused name closest to ``base``.
    """
    taken = {n.name for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)}
    if base not in taken:
        return base
    index = 1
    while f'{base}_{index}' in taken:
        index += 1
    return f'{base}_{index}'


@oprepo.replaces('dace.ai')
@oprepo.replaces('dace.frontend.python.interface.ai')
def ai_node(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            description: Union[StringLiteral, str],
            *inputs: str,
            out: Optional[Union[str, Sequence[str]]] = None,
            shape: Optional[Shape] = None,
            dtype: Optional[dtypes.typeclass] = None,
            storage: Optional[dtypes.StorageType] = None,
            name: Optional[Union[StringLiteral, str]] = None,
            schedule: Optional[dtypes.ScheduleType] = None,
            **named_inputs: str) -> Union[str, Tuple[str, ...]]:
    """
    Adds a node whose implementation is described in natural language.

    See :func:`dace.frontend.python.interface.ai` for the user-facing documentation of the
    arguments; this is the replacement that the Python frontend calls in its place.

    :param pv: The program visitor parsing the program.
    :param sdfg: The SDFG being built.
    :param state: The state to add the node to.
    :param description: What the node must compute.
    :param inputs: Containers read through the connectors ``_in0``, ``_in1``, ...
    :param out: Container(s) to write to. If not given, a transient is allocated.
    :param shape: Shape of the allocated output.
    :param dtype: Data type of the allocated output.
    :param storage: Storage type of the allocated output.
    :param name: Name of the node, which the model sees along with the description. Defaults to an
                 unused name of the form ``ai``, ``ai_1``, ...
    :param schedule: Schedule of the node, which determines its default device mapping.
    :param named_inputs: Containers read through the connectors named after their keywords.
    :return: The name of the container written, or a tuple of them if several were given in ``out``.
    :raises DaceSyntaxError: If the description or the operands cannot be used to build the node.
    """
    from dace.libraries.ai.nodes import AINode  # Avoid depending on the AI library unless it is used

    text = _description_of(pv, sdfg, description)

    # "name", like the other keyword arguments above, configures the node rather than naming an
    # input connector, so passing a container to it is a mistake worth catching here
    if name is not None and (_is_container(sdfg, name) or not isinstance(name, (StringLiteral, str))):
        raise DaceSyntaxError(
            pv, None, 'The "name" argument of dace.ai is the label of the node and must be a string. It is '
            'one of the keyword arguments that configure the node ("out", "shape", "dtype", "storage", '
            '"name", "schedule"), so it cannot be used to name an input connector.')

    in_connectors = _input_connectors(pv, sdfg, inputs, named_inputs)

    if out is None:
        out_connectors = {'_out': _allocate_output(pv, sdfg, in_connectors, shape, dtype, storage)}
    else:
        if shape is not None or dtype is not None or storage is not None:
            raise DaceSyntaxError(
                pv, None, 'The "shape", "dtype" and "storage" arguments of dace.ai describe the output it '
                'allocates, so they cannot be combined with "out", which writes to a container that '
                'already exists.')
        out_connectors = _output_connectors(pv, sdfg, out)

    node = AINode(str(name) if name is not None else _node_name(sdfg),
                  text,
                  inputs={connector: None
                          for connector in in_connectors},
                  outputs={connector: None
                           for connector in out_connectors},
                  schedule=schedule)
    state.add_node(node)

    for connector, container in in_connectors.items():
        state.add_edge(state.add_read(container), None, node, connector,
                       Memlet.from_array(container, sdfg.arrays[container]))
    for connector, container in out_connectors.items():
        state.add_edge(node, connector, state.add_write(container), None,
                       Memlet.from_array(container, sdfg.arrays[container]))

    written = tuple(out_connectors.values())
    return written[0] if len(written) == 1 else written
