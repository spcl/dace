""" A module that contains various DaCe type definitions. """
from __future__ import print_function
import ctypes
import enum
import inspect
import numpy
import itertools
import numpy.ctypeslib as npct


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
    ScheduleType.GPU_Device, ScheduleType.GPU_ThreadBlock,
    ScheduleType.GPU_ThreadBlock_Dynamic
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
    int: 'int',
    float: 'float',
    bool: 'bool',
    numpy.bool: 'bool',
    numpy.int8: 'char',
    numpy.int16: 'short',
    numpy.int32: 'int',
    numpy.int64: 'long long',
    numpy.uint8: 'unsigned char',
    numpy.uint16: 'unsigned short',
    numpy.uint32: 'unsigned int',
    numpy.uint64: 'unsigned long long',
    numpy.float16: 'half',
    numpy.float32: 'float',
    numpy.float64: 'double',
    numpy.complex64: 'dace::complex64',
    numpy.complex128: 'dace::complex128'
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
    numpy.complex128: ctypes.c_longdouble
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
    numpy.complex128: 16
}


class typeclass(object):
    """ An extension of types that enables their use in DaCe.
        
        These types are defined for three reasons:
            1. Controlling DaCe types
            2. Enabling declaration syntax: `dace.float32[M,N]`
            3. Enabling extensions such as `dace.struct` and `dace.immaterial`
    """

    def __init__(self, wrapped_type):
        self.type = wrapped_type
        self.ctype = _CTYPES[wrapped_type]
        self.ctype_unaligned = self.ctype
        self.dtype = self
        self.bytes = _BYTES[wrapped_type]
        self.materialize_func = None

    def __hash__(self):
        return hash((self.type, self.ctype, self.materialize_func))

    def to_string(self):
        """ A Numpy-like string-representation of the underlying data type. """
        return self.type.__name__

    def is_complex(self):
        if self.type == numpy.complex64 or self.type == numpy.complex128:
            return True
        return False

    # Create a new type
    def __call__(self, *args, **kwargs):
        return self.type(*args, **kwargs)

    def __eq__(self, other):
        return self.ctype == other.ctype

    def __ne__(self, other):
        return self.ctype != other.ctype

    def __getitem__(self, s):
        """ This is syntactic sugar that allows us to define an array type
            with the following syntax: dace.uint32[N,M] 
            @return: A data.Array data descriptor.
        """
        from dace import data
        if isinstance(s, list) or isinstance(s, tuple):
            return data.Array(self, tuple(s))
        return data.Array(self, (s, ))

    def __repr__(self):
        return self.ctype


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

    def _parse_field_and_types(self, **fields_and_types):
        self._data = dict()
        self._length = dict()
        self.bytes = 0
        for k, v in fields_and_types.items():
            if isinstance(v, tuple):
                t, l = v
                if not isinstance(t, pointer):
                    raise TypeError('Only pointer types may have a length.')
                if l not in fields_and_types.keys():
                    raise ValueError(
                        'Length {} not a field of struct {}'.format(
                            l, self.name))
                self._data[k] = t
                self._length[k] = l
                self.bytes += t.bytes
            else:
                if isinstance(v, pointer):
                    raise TypeError('Pointer types must have a length.')
                self._data[k] = v
                self.bytes += v.bytes

    def as_ctypes(self):
        # Populate the ctype fields for the struct class.
        fields = []
        for k, v in self._data.items():
            if isinstance(v, pointer):
                fields.append(
                    (k,
                     ctypes.c_void_p))  #ctypes.POINTER(_FFI_CTYPES[v.type])))
            else:
                fields.append((k, _FFI_CTYPES[v.type]))
        fields = sorted(fields, key=lambda f: f[0])
        # Create new struct class.
        struct_class = type('NewStructClass', (ctypes.Structure, ),
                            {"_fields_": fields})
        return struct_class

    def emit_definition(self):
        return '''struct {name} {{
{types}
}};'''.format(
            name=self.name,
            types='\n'.join([
                '    %s %s;' % (t.ctype, tname)
                for tname, t in sorted(self._data.items())
            ]))


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

TYPECLASS_STRINGS = [
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64', 'complex64', 'complex128'
]

#######################################################
# Allowed types

# Helper function to determine whether a global variable is a constant
_CONSTANT_TYPES = [
    int,
    float,
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
    typeclass  #, type
]


def isconstant(var):
    """ Returns True if a variable is designated a constant (i.e., that can be
        directly generated in code). """
    return type(var) in _CONSTANT_TYPES


# Lists allowed modules and maps them to C++ namespaces for code generation
_ALLOWED_MODULES = {
    "builtins": '',
    "dace": 'dace::',
    "math": 'dace::math::',
    "cmath": 'dace::cmath::'
}


def ismoduleallowed(var):
    """ Helper function to determine the source module of an object, and 
        whether it is allowed in DaCe programs. """
    mod = inspect.getmodule(var)
    try:
        for m in _ALLOWED_MODULES:
            if mod.__name__ == m or mod.__name__.startswith(m + '.'):
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
    return isconstant(var) or ismoduleallowed(var)


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


######################################################
# Static (utility) functions


def deduplicate(iterable):
    """ Removes duplicates in the passed iterable. """
    return type(iterable)(
        [i for i in sorted(set(iterable), key=lambda x: iterable.index(x))])
