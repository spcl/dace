
"""
Utils for the IPU target.
"""

import dace
import dace.codegen.targets


# Convert from DACE Types to IPU Types
TYPE_TO_IPU = {
    dace.bool: 'BOOL',
    dace.int8: 'CHAR',
    dace.int16: 'SHORT',
    dace.int32: 'INT',
    dace.int64: 'LONGLONG', # LONG is not supported in IPU
    dace.uint8: 'UNSIGNED_CHAR',
    dace.uint16: 'UNSIGNED_SHORT',
    dace.uint32: 'UNSINGED_INT',
    dace.uint64: 'UNSINGNED_LONGLONG',
    dace.float16: 'HALF',
    dace.float32: 'FLOAT',
    dace.float64: 'DOUBLE', 
    dace.string: 'char*', # Not sure if this is correct
}
