"""
ONNX Backend for DaCeML.

This module provides an ONNX Runtime-compatible backend implementation that
allows DaCe to execute ONNX models through the standard ONNX backend interface.

This backend is primarily used for:
- Running ONNX conformance test suites
- Validating correctness against ONNX specifications
- Benchmarking (note: this interface has overhead and is not optimized for performance)

For production use, prefer using ONNXModel directly for better performance.
"""

from typing import Any, Tuple

from dace import dtypes
from onnx.backend import base

from . import onnx_importer


class DaCeMLBackendRep(base.BackendRep):
    """
    Representation of a prepared ONNX model in the DaCe backend.

    This class wraps an ONNXModel and provides the ONNX Backend API
    interface for running the model.

    Attributes:
        model: The imported and compiled ONNXModel ready for execution.
    """

    def __init__(self, model: onnx_importer.ONNXModel):
        """
        Initialize the backend representation.

        Args:
            model: The ONNXModel to wrap.
        """
        self.model = model

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """
        Execute the model with the given inputs.

        This method copies the inputs before execution to avoid modifying
        the original data.

        Args:
            inputs: Sequence of input arrays for the model.
            **kwargs: Additional keyword arguments (passed through but not used).

        Returns:
            Tuple of output arrays from the model.
        """
        # Copy inputs to avoid in-place modifications
        copied_inputs = [x.copy() for x in inputs]
        results = self.model(*copied_inputs)

        # Ensure we always return a tuple
        if isinstance(results, tuple):
            return results
        else:
            return (results,)


class DaCeMLBackend(base.Backend):
    """
    DaCe backend implementation for ONNX Runtime API.

    This backend converts ONNX models to DaCe SDFGs and provides
    the standard ONNX backend interface for execution.
    """

    @classmethod
    def prepare(cls, model, device: str = 'CPU', **kwargs):
        """
        Prepare an ONNX model for execution on the DaCe backend.

        This method imports the ONNX model and converts it to a DaCe SDFG,
        returning a representation that can be executed.

        Args:
            model: The ONNX ModelProto to prepare.
            device: Target device, either 'CPU' or 'CUDA'. Defaults to 'CPU'.
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            A DaCeMLBackendRep instance ready for execution.
        """
        super().prepare(model, device, **kwargs)

        dace_model = onnx_importer.ONNXModel(
            'backend_model',
            model,
            cuda=device == 'CUDA',
            onnx_simplify=False,
            storage=dtypes.StorageType.Default
        )

        return DaCeMLBackendRep(dace_model)
