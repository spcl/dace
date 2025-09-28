from dace.library import register_library, _DACE_REGISTERED_LIBRARIES
from .nodes import *
from .schema import onnx_representation, ONNXAttributeType, ONNXAttribute, ONNXTypeConstraint, ONNXParameterType, ONNXSchema, ONNXParameter
from .onnx_importer import ONNXModel
from .backend import DaCeMLBackend, DaCeMLBackendRep

register_library(__name__, "dace.libraries.onnx")
_DACE_REGISTERED_LIBRARIES["dace.libraries.onnx"].default_implementation = "pure"
