""" A module that contains various DaCe type definitions. """
from __future__ import print_function
import ctypes
import enum
import inspect
import numpy
import pydoc
from functools import wraps


class AutoNumber(enum.Enum):
    """ Backwards-compatible version of Enum's `auto()` """

    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class StorageType(AutoNumber):
    """ Available data storage types in the SDFG. """

    Default = ()  # Scope-default storage location
    Immaterial = ()  # Needs materialize function
    Register = ()  # Tasklet storage location
    CPU_Pinned = ()  # NOTE: Can be DMA accessed from accelerators
    CPU_Heap = ()  # NOTE: Allocated with new[]
    CPU_Stack = ()  # NOTE: Allocated on stack
    GPU_Global = ()  # Global memory
    GPU_Shared = ()  # Shared memory
    GPU_Stack = ()  # GPU registers
    FPGA_Global = ()  # Off-chip global memory (DRAM)
    FPGA_Local = ()  # On-chip memory (bulk storage)
    FPGA_Registers = ()  # On-chip memory (fully partitioned registers)


class ScheduleType(AutoNumber):
    """ Available map schedule types in the SDFG. """

    Default = ()  # Scope-default parallel schedule
    Sequential = ()  # Sequential code (single-core)
    MPI = ()  # MPI processes
    CPU_Multicore = ()  # OpenMP
    GPU_Device = ()  # Kernel
    GPU_ThreadBlock = ()  # Thread-block code
    GPU_ThreadBlock_Dynamic = ()  # Allows rescheduling work within a block
    FPGA_Device = ()


# A subset of GPU schedule types
GPU_SCHEDULES = [
    ScheduleType.GPU_Device,
    ScheduleType.GPU_ThreadBlock,
    ScheduleType.GPU_ThreadBlock_Dynamic,
]


class ReductionType(AutoNumber):
    """ Reduction types natively supported by the SDFG compiler. """

    Custom = ()  # Defined by an arbitrary lambda function
    Min = ()  # Minimum value
    Max = ()  # Maximum value
    Sum = ()  # Sum
    Product = ()  # Product
    Logical_And = ()  # Logical AND (&&)
    Bitwise_And = ()  # Bitwise AND (&)
    Logical_Or = ()  # Logical OR (||)
    Bitwise_Or = ()  # Bitwise OR (|)
    Logical_Xor = ()  # Logical XOR (!=)
    Bitwise_Xor = ()  # Bitwise XOR (^)
    Min_Location = ()  # Minimum value and its location
    Max_Location = ()  # Maximum value and its location


class Language(AutoNumber):
    """ Available programming languages for SDFG tasklets. """

    Python = ()
    CPP = ()


class AccessType(AutoNumber):
    """ Types of access to an `AccessNode`. """

    ReadOnly = ()
    WriteOnly = ()
    ReadWrite = ()


class InstrumentationType(AutoNumber):
    """ Types of instrumentation providers.
        @note: Might be determined automatically in future versions.
    """

    No_Instrumentation = ()
    Timer = ()
    PAPI_Counters = ()
    CUDA_Events = ()


# Maps from ScheduleType to default StorageType
SCOPEDEFAULT_STORAGE = {
    None: StorageType.CPU_Heap,
    ScheduleType.Sequential: StorageType.Register,
    ScheduleType.MPI: StorageType.CPU_Heap,
    ScheduleType.CPU_Multicore: StorageType.CPU_Stack,
    ScheduleType.GPU_Device: StorageType.GPU_Shared,
    ScheduleType.GPU_ThreadBlock: StorageType.GPU_Stack,
    ScheduleType.GPU_ThreadBlock_Dynamic: StorageType.GPU_Stack,
    ScheduleType.FPGA_Device: StorageType.FPGA_Global,
}

# Maps from ScheduleType to default ScheduleType for sub-scopes
SCOPEDEFAULT_SCHEDULE = {
    None: ScheduleType.CPU_Multicore,
    ScheduleType.Sequential: ScheduleType.Sequential,
    ScheduleType.MPI: ScheduleType.CPU_Multicore,
    ScheduleType.CPU_Multicore: ScheduleType.Sequential,
    ScheduleType.GPU_Device: ScheduleType.GPU_ThreadBlock,
    ScheduleType.GPU_ThreadBlock: ScheduleType.Sequential,
    ScheduleType.GPU_ThreadBlock_Dynamic: ScheduleType.Sequential,
    ScheduleType.FPGA_Device: ScheduleType.FPGA_Device,
}

# Identifier for dynamic number of Memlet accesses.
DYNAMIC = -1

# Translation of types to C types
_CTYPES = {
    int: "int",
    float: "float",
    bool: "bool",
    numpy.bool: "bool",
    numpy.int8: "char",
    numpy.int16: "short",
    numpy.int32: "int",
    numpy.int64: "long long",
    numpy.uint8: "unsigned char",
    numpy.uint16: "unsigned short",
    numpy.uint32: "unsigned int",
    numpy.uint64: "unsigned long long",
    numpy.float16: "half",
    numpy.float32: "float",
    numpy.float64: "double",
    numpy.complex64: "dace::complex64",
    numpy.complex128: "dace::complex128",
}

# Translation of types to ctypes types
_FFI_CTYPES = {
    int: ctypes.c_int,
    float: ctypes.c_float,
    bool: ctypes.c_bool,
    numpy.bool: ctypes.c_bool,
    numpy.int8: ctypes.c_int8,
    numpy.int16: ctypes.c_int16,
    numpy.int32: ctypes.c_int32,
    numpy.int64: ctypes.c_int64,
    numpy.uint8: ctypes.c_uint8,
    numpy.uint16: ctypes.c_uint16,
    numpy.uint32: ctypes.c_uint32,
    numpy.uint64: ctypes.c_uint64,
    numpy.float16: ctypes.c_uint16,
    numpy.float32: ctypes.c_float,
    numpy.float64: ctypes.c_double,
    numpy.complex64: ctypes.c_uint64,
    numpy.complex128: ctypes.c_longdouble,
}

# Number of bytes per data type
_BYTES = {
    int: 4,
    float: 4,
    bool: 1,
    numpy.bool: 1,
    numpy.int8: 1,
    numpy.int8: 1,
    numpy.int16: 2,
    numpy.int32: 4,
    numpy.int64: 8,
    numpy.uint8: 1,
    numpy.uint16: 2,
    numpy.uint32: 4,
    numpy.uint64: 8,
    numpy.float16: 2,
    numpy.float32: 4,
    numpy.float64: 8,
    numpy.complex64: 8,
    numpy.complex128: 16,
}


class typeclass(object):
    """ An extension of types that enables their use in DaCe.

        These types are defined for three reasons:
            1. Controlling DaCe types
            2. Enabling declaration syntax: `dace.float32[M,N]`
            3. Enabling extensions such as `dace.struct` and `dace.immaterial`
    """

    def __init__(self, wrapped_type):
        # Convert python basic types
        if isinstance(wrapped_type, str):
            try:
                wrapped_type = getattr(numpy, wrapped_type)
            except AttributeError:
                raise ValueError("Unknown type: {}".format(wrapped_type))
        if wrapped_type is int:
            wrapped_type = numpy.int64
        elif wrapped_type is float:
            wrapped_type = numpy.float64
        elif wrapped_type is complex:
            wrapped_type = numpy.complex128

        self.type = wrapped_type  # Type in Python
        self.ctype = _CTYPES[wrapped_type]  # Type in C
        self.ctype_unaligned = self.ctype  # Type in C (without alignment)
        self.dtype = self  # For compatibility support with numpy
        self.bytes = _BYTES[wrapped_type]  # Number of bytes for this type
        self.materialize_func = None  # Materialize function for immaterial types

    def __hash__(self):
        return hash((self.type, self.ctype, self.materialize_func))

    def to_string(self):
        """ A Numpy-like string-representation of the underlying data type. """
        return self.type.__name__

    def as_ctypes(self):
        """ Returns the ctypes version of the typeclass. """
        return _FFI_CTYPES[self.type]

    def is_complex(self):
        if self.type == numpy.complex64 or self.type == numpy.complex128:
            return True
        return False

    def to_json(self):
        return self.type.__name__

    # Create a new type
    def __call__(self, *args, **kwargs):
        return self.type(*args, **kwargs)

    def __eq__(self, other):
        return other is not None and self.ctype == other.ctype

    def __ne__(self, other):
        return other is not None and self.ctype != other.ctype

    def __getitem__(self, s):
        """ This is syntactic sugar that allows us to define an array type
            with the following syntax: dace.uint32[N,M]
            :return: A data.Array data descriptor.
        """
        from dace import data

        if isinstance(s, list) or isinstance(s, tuple):
            return data.Array(self, tuple(s))
        return data.Array(self, (s, ))

    def __repr__(self):
        return self.ctype


# _CTYPES_RULES: returns the largest between two types (dace.types.typeclass) according to C semantic
_CTYPES_RULES = {

    # Both operands are integers
    # C Integer promotion rules apply:
    # - if the types are the same, return the type
    frozenset((typeclass(numpy.uint8), typeclass(numpy.uint8))):
    typeclass(numpy.uint8),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.uint16))):
    typeclass(numpy.uint16),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.uint32))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.uint))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.uint64))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.int8), typeclass(numpy.int8))):
    typeclass(numpy.int8),
    frozenset((typeclass(numpy.int16), typeclass(numpy.int16))):
    typeclass(numpy.int16),
    frozenset((typeclass(numpy.int32), typeclass(numpy.int32))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int32), typeclass(numpy.int))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int64), typeclass(numpy.int64))):
    typeclass(numpy.int64),

    # - Otherwise if both operands after promotion have the same signedness
    frozenset((typeclass(numpy.uint64), typeclass(numpy.uint32))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.uint))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.int16))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.uint8))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.uint16))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.uint8))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.uint8))):
    typeclass(numpy.uint16),
    frozenset((typeclass(numpy.int64), typeclass(numpy.int32))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int64), typeclass(numpy.int))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int64), typeclass(numpy.int16))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int64), typeclass(numpy.int8))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int32), typeclass(numpy.int16))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int), typeclass(numpy.int16))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int32), typeclass(numpy.int8))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int), typeclass(numpy.int8))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int16), typeclass(numpy.int8))):
    typeclass(numpy.int16),

    # - Otherwise, the signedness is different: If the operand with the unsigned type has
    # conversion rank greater or equal than the rank of the type of the signed operand,
    # then the operand with the signed type is implicitly converted to the unsigned type
    frozenset((typeclass(numpy.uint64), typeclass(numpy.int64))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.int32))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.int16))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint64), typeclass(numpy.int8))):
    typeclass(numpy.uint64),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.int32))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.int16))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.int8))):
    typeclass(numpy.uint32),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.int16))):
    typeclass(numpy.uint16),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.int8))):
    typeclass(numpy.uint16),
    frozenset((typeclass(numpy.uint8), typeclass(numpy.int8))):
    typeclass(numpy.uint8),

    # - Otherwise, the signedness is different and the signed operand's rank is greater than
    # unsigned operand's rank. In this case, if the signed type can represent all values of
    # the unsigned type, then the operand with the unsigned type is implicitly converted to
    # the type of the signed operand.
    frozenset((typeclass(numpy.int64), typeclass(numpy.uint32))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int64), typeclass(numpy.uint16))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int64), typeclass(numpy.uint8))):
    typeclass(numpy.int64),
    frozenset((typeclass(numpy.int32), typeclass(numpy.uint16))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int32), typeclass(numpy.uint8))):
    typeclass(numpy.int32),
    frozenset((typeclass(numpy.int16), typeclass(numpy.uint8))):
    typeclass(numpy.int16),

    # If any of the operand is float then the result will be in float (of the biggest rank
    # if the two operands are of the same type)
    frozenset((typeclass(numpy.float64), typeclass(numpy.float64))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.float64), typeclass(numpy.float32))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.float64), typeclass(numpy.float16))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.float32), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.float32), typeclass(numpy.float16))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.float16), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.int8), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.int16), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.int32), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.uint8), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.float16))):
    typeclass(numpy.float16),
    frozenset((typeclass(numpy.int8), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.int16), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.int32), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.uint8), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.float32))):
    typeclass(numpy.float32),
    frozenset((typeclass(numpy.int8), typeclass(numpy.float64))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.int16), typeclass(numpy.float64))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.int32), typeclass(numpy.float64))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.uint8), typeclass(numpy.float64))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.uint16), typeclass(numpy.float64))):
    typeclass(numpy.float64),
    frozenset((typeclass(numpy.uint32), typeclass(numpy.float64))):
    typeclass(numpy.float64),
}


class pointer(typeclass):
    """ A data type for a pointer to an existing typeclass.

        Example use:
            `dace.pointer(dace.struct(x=dace.float32, y=dace.float32))`. """

    def __init__(self, wrapped_typeclass):
        self._typeclass = wrapped_typeclass
        self.type = wrapped_typeclass.type
        self.bytes = int64.bytes
        self.ctype = wrapped_typeclass.ctype + "*"
        self.ctype_unaligned = wrapped_typeclass.ctype_unaligned + "*"
        self.dtype = self
        self.materialize_func = None

    def to_json(self):
        return {'type': 'pointer', 'dtype': self._typeclass.to_json()}

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'pointer':
            raise TypeError("Invalid type for pointer")

        return pointer(json_to_typeclass(json_obj['dtype']))

    def as_ctypes(self):
        """ Returns the ctypes version of the typeclass. """
        return ctypes.POINTER(_FFI_CTYPES[self.type])


def immaterial(dace_data, materialize_func):
    """ A data type with a materialize/serialize function. Data objects with
        this type do not allocate new memory. Whenever it is accessed, the
        materialize/serialize function is invoked instead. """
    dace_data.materialize_func = materialize_func
    return dace_data


class struct(typeclass):
    """ A data type for a struct of existing typeclasses.

        Example use: `dace.struct(a=dace.int32, b=dace.float64)`.
    """

    def __init__(self, name, **fields_and_types):
        # self._data = fields_and_types
        self.type = ctypes.Structure
        self.name = name
        # TODO: Assuming no alignment! Get from ctypes
        # self.bytes = sum(t.bytes for t in fields_and_types.values())
        self.ctype = name
        self.ctype_unaligned = name
        self.dtype = self
        self.materialize_func = None
        self._parse_field_and_types(**fields_and_types)

    @property
    def fields(self):
        return self._data

    def to_json(self):
        return {
            'type': 'struct',
            'name': self.name,
            'data': {k: v.to_json()
                     for k, v in self._data.items()},
            'length': {k: v
                       for k, v in self._length.items()},
            'bytes': self.bytes
        }

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "struct":
            raise TypeError("Invalid type for struct")

        import dace.serialize  # Avoid import loop

        ret = struct(json_obj['name'])
        ret._data = {
            k: json_to_typeclass(v)
            for k, v in json_obj['data'].items()
        }
        ret._length = {k: v for k, v in json_obj['length'].items()}
        ret.bytes = json_obj['bytes']

        return ret

    def _parse_field_and_types(self, **fields_and_types):
        self._data = dict()
        self._length = dict()
        self.bytes = 0
        for k, v in fields_and_types.items():
            if isinstance(v, tuple):
                t, l = v
                if not isinstance(t, pointer):
                    raise TypeError("Only pointer types may have a length.")
                if l not in fields_and_types.keys():
                    raise ValueError(
                        "Length {} not a field of struct {}".format(
                            l, self.name))
                self._data[k] = t
                self._length[k] = l
                self.bytes += t.bytes
            else:
                if isinstance(v, pointer):
                    raise TypeError("Pointer types must have a length.")
                self._data[k] = v
                self.bytes += v.bytes

    def as_ctypes(self):
        """ Returns the ctypes version of the typeclass. """
        # Populate the ctype fields for the struct class.
        fields = []
        for k, v in self._data.items():
            if isinstance(v, pointer):
                fields.append(
                    (k,
                     ctypes.c_void_p))  # ctypes.POINTER(_FFI_CTYPES[v.type])))
            else:
                fields.append((k, _FFI_CTYPES[v.type]))
        fields = sorted(fields, key=lambda f: f[0])
        # Create new struct class.
        struct_class = type("NewStructClass", (ctypes.Structure, ),
                            {"_fields_": fields})
        return struct_class

    def emit_definition(self):
        return """struct {name} {{
{typ}
}};""".format(
            name=self.name,
            typ='\n'.join([
                "    %s %s;" % (t.ctype, tname)
                for tname, t in sorted(self._data.items())
            ]),
        )


####### Utility function ##############
def ptrtonumpy(ptr, inner_ctype, shape):
    import ctypes
    import numpy as np
    return np.ctypeslib.as_array(
        ctypes.cast(ctypes.c_void_p(ptr), ctypes.POINTER(inner_ctype)), shape)


def _atomic_counter_generator():
    ctr = 0
    while True:
        ctr += 1
        yield ctr


class callback(typeclass):
    """ Looks like dace.callback([None, <some_native_type>], *types)"""

    def __init__(self, return_type, *variadic_args):
        self.uid = next(_atomic_counter_generator())
        from dace import data
        if isinstance(return_type, data.Array):
            raise TypeError("Callbacks that return arrays are "
                            "not supported as per SDFG semantics")
        self.dtype = self
        self.return_type = return_type
        self.input_types = []
        for arg in variadic_args:
            if isinstance(arg, typeclass):
                pass
            elif isinstance(arg, data.Data):
                pass
            elif isinstance(arg, str):
                arg = json_to_typeclass(arg)
            else:
                raise TypeError("Cannot resolve type from: {}".format(arg))
            self.input_types.append(arg)
        self.bytes = int64.bytes
        self.materialize_func = None
        self.type = self
        self.ctype = self

    def as_ctypes(self):
        """ Returns the ctypes version of the typeclass. """
        from dace import data

        return_ctype = (self.return_type.as_ctypes()
                        if self.return_type is not None else None)
        input_ctypes = []
        for some_arg in self.input_types:
            if isinstance(some_arg, data.Array):
                input_ctypes.append(ctypes.c_void_p)
            else:
                input_ctypes.append(some_arg.as_ctypes()
                                    if some_arg is not None else None)
        if input_ctypes == [None]:
            input_ctypes = []
        cf_object = ctypes.CFUNCTYPE(return_ctype, *input_ctypes)
        return cf_object

    def signature(self, name):
        from dace import data

        return_type_cstring = (self.return_type.ctype
                               if self.return_type is not None else "void")
        input_type_cstring = []
        for arg in self.input_types:
            if isinstance(arg, data.Array):
                # const hack needed to prevent error in casting const int* to int*
                input_type_cstring.append(arg.dtype.ctype + " const *")
            else:
                input_type_cstring.append(arg.ctype if arg is not None else "")
        cstring = return_type_cstring + " " + "(*" + name + ")("
        for index, inp_arg in enumerate(input_type_cstring):
            if index > 0:
                cstring = cstring + ","
            cstring = cstring + inp_arg
        cstring = cstring + ")"
        return cstring

    def get_trampoline(self, pyfunc):
        from functools import partial
        from dace import data, symbolic

        arraypos = []
        types_and_sizes = []
        for index, arg in enumerate(self.input_types):
            if isinstance(arg, data.Array):
                arraypos.append(index)
                types_and_sizes.append((arg.dtype.as_ctypes(), arg.shape))
        if len(arraypos) == 0:
            return pyfunc

        def trampoline(orig_function, indices, data_types_and_sizes,
                       *other_inputs):
            list_of_other_inputs = list(other_inputs)
            for i in indices:
                data_type, size = data_types_and_sizes[i]
                non_symbolic_sizes = []
                for s in size:
                    if isinstance(s, symbolic.symbol):
                        non_symbolic_sizes.append(s.get())
                    else:
                        non_symbolic_sizes.append(s)
                list_of_other_inputs[i] = ptrtonumpy(
                    other_inputs[i], data_type, non_symbolic_sizes)
            return orig_function(*list_of_other_inputs)

        return partial(trampoline, pyfunc, arraypos, types_and_sizes)

    def __hash__(self):
        return hash((self.uid, self.return_type, *self.input_types))

    def to_json(self):
        return {
            'type': 'callback',
            'arguments': [i.to_json() for i in self.input_types],
            'returntype': self.return_type.to_json()
            if self.return_type else None
        }

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "callback":
            raise TypeError("Invalid type for callback")

        rettype = json_obj['returntype']

        import dace.serialize  # Avoid import loop

        return callback(
            json_to_typeclass(rettype) if rettype else None,
            *(dace.serialize.from_json(arg) for arg in json_obj['arguments']))

    def __str__(self):
        return "dace.callback"

    def __repr__(self):
        return "dace.callback"

    def __eq__(self, other):
        if not isinstance(other, callback):
            return False
        return self.uid == other.uid

    def __ne__(self, other):
        return not self.__eq__(other)


bool = typeclass(numpy.bool)
int8 = typeclass(numpy.int8)
int16 = typeclass(numpy.int16)
int32 = typeclass(numpy.int32)
int64 = typeclass(numpy.int64)
uint8 = typeclass(numpy.uint8)
uint16 = typeclass(numpy.uint16)
uint32 = typeclass(numpy.uint32)
uint64 = typeclass(numpy.uint64)
float16 = typeclass(numpy.float16)
float32 = typeclass(numpy.float32)
float64 = typeclass(numpy.float64)
complex64 = typeclass(numpy.complex64)
complex128 = typeclass(numpy.complex128)

DTYPE_TO_TYPECLASS = {
    int: int32,
    float: float32,
    bool: uint8,
    numpy.bool: uint8,
    numpy.bool_: bool,
    numpy.int8: int8,
    numpy.int16: int16,
    numpy.int32: int32,
    numpy.int64: int64,
    numpy.uint8: uint8,
    numpy.uint16: uint16,
    numpy.uint32: uint32,
    numpy.uint64: uint64,
    numpy.float16: float16,
    numpy.float32: float32,
    numpy.float64: float64,
    numpy.complex64: complex64,
    numpy.complex128: complex128
}

TYPECLASS_STRINGS = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

#######################################################
# Allowed types

# Helper function to determine whether a global variable is a constant
_CONSTANT_TYPES = [
    int,
    float,
    str,
    numpy.intc,
    numpy.intp,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
    numpy.float16,
    numpy.float32,
    numpy.float64,
    numpy.complex64,
    numpy.complex128,
    typeclass,  # , type
]


def isconstant(var):
    """ Returns True if a variable is designated a constant (i.e., that can be
        directly generated in code). """
    return type(var) in _CONSTANT_TYPES


# Lists allowed modules and maps them to C++ namespaces for code generation
_ALLOWED_MODULES = {
    "builtins": "",
    "dace": "dace::",
    "math": "dace::math::",
    "cmath": "dace::cmath::",
}

# Lists allowed modules and maps them to OpenCL
_OPENCL_ALLOWED_MODULES = {"builtins": "", "dace": "", "math": ""}


def ismodule(var):
    """ Returns True if a given object is a module. """
    return inspect.ismodule(var)


def ismodule(var):
    """ Returns True if a given object is a module. """
    return inspect.ismodule(var)


def ismoduleallowed(var):
    """ Helper function to determine the source module of an object, and
        whether it is allowed in DaCe programs. """
    mod = inspect.getmodule(var)
    try:
        for m in _ALLOWED_MODULES:
            if mod.__name__ == m or mod.__name__.startswith(m + "."):
                return True
    except AttributeError:
        return False
    return False


def ismodule_and_allowed(var):
    """ Returns True if a given object is a module and is one of the allowed
        modules in DaCe programs. """
    if inspect.ismodule(var):
        if var.__name__ in _ALLOWED_MODULES:
            return True
    return False


def isallowed(var):
    """ Returns True if a given object is allowed in a DaCe program. """
    from dace.symbolic import symbol
    return (isconstant(var) or ismodule(var) or isinstance(var, symbol)
            or callable(var))


class _external_function(object):
    def __init__(self, f, alt_imps=None):
        self.func = f
        if alt_imps is None:
            self.alt_imps = {}
        else:
            self.alt_imps = alt_imps

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class DebugInfo:
    """ Source code location identifier of a node/edge in an SDFG. Used for
        IDE and debugging purposes. """

    def __init__(self,
                 start_line,
                 start_column,
                 end_line,
                 end_column,
                 filename=None):
        self.start_line = start_line
        self.end_line = end_line
        self.start_column = start_column
        self.end_column = end_column
        self.filename = filename

    # NOTE: Manually marking as serializable to avoid an import loop
    # The data structure is a property on its own (pointing to a range of code),
    # so it is serialized as a dictionary directly.
    def to_json(self):
        return dict(
            type='DebugInfo',
            start_line=self.start_line,
            end_line=self.end_line,
            start_column=self.start_column,
            end_column=self.end_column,
            filename=self.filename)

    @staticmethod
    def from_json(json_obj, context=None):
        return DebugInfo(json_obj['start_line'], json_obj['start_column'],
                         json_obj['end_line'], json_obj['end_column'],
                         json_obj['filename'])


######################################################
# Static (utility) functions


def json_to_typeclass(obj):
    # TODO: this does two different things at the same time. Should be split
    # into two separate functions.
    from dace.serialize import get_serializer
    if isinstance(obj, str):
        return get_serializer(obj)
    elif isinstance(obj, dict) and "type" in obj:
        return get_serializer(obj["type"]).from_json(obj)
    else:
        raise ValueError("Cannot resolve: {}".format(obj))


def paramdec(dec):
    """ Parameterized decorator meta-decorator. Enables using `@decorator`,
        `@decorator()`, and `@decorator(...)` with the same function. """

    @wraps(dec)
    def layer(*args, **kwargs):

        # Allows the use of @decorator, @decorator(), and @decorator(...)
        if len(kwargs) == 0 and len(args) == 1 and callable(
                args[0]) and not isinstance(args[0], typeclass):
            return dec(*args, **kwargs)

        @wraps(dec)
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


#############################################


def deduplicate(iterable):
    """ Removes duplicates in the passed iterable. """
    return type(iterable)(
        [i for i in sorted(set(iterable), key=lambda x: iterable.index(x))])
