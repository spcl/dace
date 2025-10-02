try:
    from .constant_folding import ConstantFolding
    from .parameter_to_transient import parameter_to_transient
except ImportError:
    # ONNX transformations not available
    ConstantFolding = None
    parameter_to_transient = None
