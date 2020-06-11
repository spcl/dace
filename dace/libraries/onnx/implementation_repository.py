# TODO
class ONNXImplementations:
    _implementations = {}

    @staticmethod
    def get(name: str) -> list:
        """ Returns implementations of a ONNX op. """
        if name not in ONNXImplementations._implementations:
            return []
        return ONNXImplementations._implementations[name]