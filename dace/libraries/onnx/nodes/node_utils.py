# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Utility functions for ONNX node operations.

This module provides helper functions for working with ONNX operation nodes
in DaCe SDFGs, including:

- Parsing variadic parameter names
- Validating parameter formats
- Schema utilities for ONNX operations

These utilities support the ONNX node system by handling the complexities
of variadic inputs/outputs and parameter naming conventions.
"""

from typing import Tuple

from dace.libraries.onnx.schema import ONNXParameterType, ONNXSchema


def parse_variadic_param(param: str) -> Tuple[str, int]:
    """
    Parse a variadic parameter name into its base name and index.

    ONNX operations can have variadic inputs/outputs, which are named using
    the convention 'base_name__index' (e.g., 'input__0', 'input__1').
    This function extracts the base name and numeric index.

    Args:
        param: The variadic parameter name in format 'name__number'.

    Returns:
        A tuple of (base_name, index) where base_name is the parameter name
        and index is the variadic position (zero-indexed).

    Raises:
        ValueError: If the parameter format is invalid, has leading zeros
                   in the number, or the number is negative.

    Examples:
        >>> parse_variadic_param("input__0")
        ('input', 0)
        >>> parse_variadic_param("output__5")
        ('output', 5)
        >>> parse_variadic_param("input__01")  # raises ValueError
    """
    split = param.split('__')
    if len(split) != 2:
        raise ValueError("Unable to parse variadic parameter '{}'".format(param))
    name = split[0]
    number = split[1]

    if number[0] == '0' and len(number) > 1:
        raise ValueError("Variadic parameters must not be numbered with leading zeros, got: '{}'".format(number))

    number = int(number)
    if number < 0:
        raise ValueError("Variadic parameter numberings must be greater than zero, got: '{}'".format(number))
    return name, number


def get_position(schema: ONNXSchema, is_input: bool, parameter_name: str):
    """Get the position that the parameter has in the ONNX op.

    Args:
        schema: The ONNX schema containing parameter definitions
        is_input: True if looking for input parameters, False for output parameters
        parameter_name: The name of the parameter to find position for

    Returns:
        The position index of the parameter in the operation signature

    Raises:
        ValueError: If parameter is not found, has incorrect variadic format,
                   or schema validation fails
    """
    if "__" in parameter_name:
        parameter_name, variadic_number = parse_variadic_param(parameter_name)
    else:
        variadic_number = None

    matches = [(i, param) for i, param in enumerate(schema.inputs if is_input else schema.outputs)
               if param.name == parameter_name]
    if len(matches) != 1:
        raise ValueError("Error in schema: found more or less than one parameter with name {}".format(parameter_name))

    index, param = matches[0]

    if variadic_number is not None and param.param_type != ONNXParameterType.Variadic:
        raise ValueError("Got variadic index for non-variadic parameter {}".format(parameter_name))

    if variadic_number is None and param.param_type == ONNXParameterType.Variadic:
        raise ValueError("Did not get variadic index for variadic parameter {}. "
                         "Specify a variadic index by renaming the parameter to {}__i, where i is a number".format(
                             parameter_name, parameter_name))

    if variadic_number is not None:
        return variadic_number + index
    else:
        return index
