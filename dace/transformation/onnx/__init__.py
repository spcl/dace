try:
    from .constant_folding import ConstantFolding
    from .parameter_to_transient import parameter_to_transient
    from .optimize import expand_onnx_nodes, auto_optimize_onnx
except ImportError:
    # ONNX transformations not available
    ConstantFolding = None
    parameter_to_transient = None
    expand_onnx_nodes = None
    auto_optimize_onnx = None
