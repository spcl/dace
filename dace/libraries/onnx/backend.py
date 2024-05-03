"""
ONNX Backend for daceml

Mainly used to run the ONNX test suites (this is not fast!).
"""

from typing import Any, Tuple
from onnx.backend import base

from dace import dtypes

from . import onnx_importer


class DaCeMLBackendRep(base.BackendRep):
    def __init__(self, model: onnx_importer.ONNXModel):
        self.model = model

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        results = self.model(*map(lambda x: x.copy(), inputs))
        if isinstance(results, tuple):
            return results
        else:
            return (results, )


class DaCeMLBackend(base.Backend):
    """
    ONNX Backend for daceml
    """
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        """
        Prepare the model for execution.
        """
        super().prepare(model, device, **kwargs)

        dace_model = onnx_importer.ONNXModel(
            'backend_model',
            model,
            cuda=device == 'CUDA',
            onnx_simplify=False,
            storage=dtypes.StorageType.Default)

        return DaCeMLBackendRep(dace_model)
