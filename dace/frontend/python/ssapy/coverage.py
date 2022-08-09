
from . import type_types as _typ
from .type_helpers import type_of_type as _type_of_type


builtins_whitelist = (

    # Constants
    'Ellipsis', 'True', 'False', 'None',

    # Basic Types
    'bool', 'int', 'float', 'str',
    'list', 'tuple', 'dict', 'set',

    # Functions
    'range', 'print',
    'filter', 'all', 'any',

    # ssapy
    'for_iter',
)


class_attrs_blacklist = set([

    # Metaclasses
    '__new__',
    
    # Attribute lookup
    '__setattr__'
    '__getattr__',
    '__getattribute__',

    # Descriptors
    '__get__',
    '__set__',
    '__delete__',
])


builtin_types = {
    'bool': _type_of_type(bool),
    'int': _type_of_type(int),
    'float': _type_of_type(float),
    'str': _type_of_type(str),
}


typing_types = {
    'Any': _type_of_type(_typ.Any),
    'Union': _type_of_type(_typ.Union),
    'List': _type_of_type(_typ.List),
    'Tuple': _type_of_type(_typ.Tuple),
    'Dict': _type_of_type(_typ.Dict),
    'Iterable': _type_of_type(_typ.Iterable),
    'TypeVar': _type_of_type(_typ.TypeVar),
}


global_module_types = {

    'builtins': builtin_types,

    'typing': typing_types,

    'ssapy': {
        **typing_types,
        'FunctionReturn': _type_of_type(_typ.FunctionReturn),
    },

}


global_class_attrs = {}


def get_builtins():

    return [var for var in builtins_whitelist]

def get_builtins_definitons():

    import builtins

    return {var: (var, getattr(builtins, var, None)) for var in get_builtins()}


def get_builtins_types():

    types = {}
    builtins = global_module_types['builtins']

    for var in get_builtins():

        if var in builtins:
            types[var] = builtins[var]
        else:
            types[var] = _typ.TypeRef(_typ.Any)

    return types


# ALL python builtins:


errors = ['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning',
          'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning',
          'EOFError', 'EncodingWarning', 'EnvironmentError', 'Exception', 'FileExistsError', 'FileNotFoundError', 'FloatingPointError', 'IOError',
          'FutureWarning', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'GeneratorExit',
          'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 'NotADirectoryError', 'NotImplemented',
          'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 'RecursionError', 
          'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning',
          'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError',
          'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError', ]

constants = [ 'Ellipsis', 'True', 'False', 'None', ]

illegal = ['globals', 'locals', '__import__', ]

functions = ['abs', 'aiter', 'all', 'anext', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod',
             'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset',
             'getattr', 'hasattr', 'hash', 'hex', 'help', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 
             'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed',
             'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']

ipython_specific = ['display', 'execfile', 'runfile', '__IPYTHON__', 'get_ipython',]

unknown = [ 'copyright', 'credits',  'license', '__build_class__', '__debug__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
