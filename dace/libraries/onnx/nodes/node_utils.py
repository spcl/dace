from dace.libraries.onnx.schema import ONNXSchema, ONNXParameterType


def parse_variadic_param(param):
    split = param.split('__')
    if len(split) != 2:
        raise ValueError(
            "Unable to parse variadic parameter '{}'".format(param))
    name = split[0]
    number = split[1]

    if number[0] == '0' and len(number) > 1:
        raise ValueError(
            "Variadic parameters must not be numbered with leading zeroes, got: '{}'"
            .format(number))

    number = int(number)
    if number < 0:
        raise ValueError(
            "Variadic parameters numberings must be greater than zero, got: '{}'"
            .format(number))
    return name, number


def get_position(schema: ONNXSchema, is_input: bool, parameter_name: str):
    """Get the position that the parameter has in the onnx op"""
    if "__" in parameter_name:
        parameter_name, variadic_number = parse_variadic_param(parameter_name)
    else:
        variadic_number = None

    matches = [(i, param) for i, param in enumerate(
        schema.inputs if is_input else schema.outputs)
               if param.name == parameter_name]
    if len(matches) != 1:
        raise ValueError(
            "Error in schema: found more or less than one parameter with name {}"
            .format(parameter_name))

    index, param = matches[0]

    if variadic_number is not None and param.param_type != ONNXParameterType.Variadic:
        raise ValueError(
            "Got variadic index for non variadic parameter {}".format(
                parameter_name))

    if variadic_number is None and param.param_type == ONNXParameterType.Variadic:
        raise ValueError(
            "Did not get variadic index for variadic parameter {}. "
            "Specify a variadic index by renaming the parameter to {}__i, where i is a number"
            .format(parameter_name, parameter_name))

    if variadic_number is not None:
        return variadic_number + index
    else:
        return index
