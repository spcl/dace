import dace
from dace.libraries.standard.nodes.code import CodeLibraryNode
from dace.properties import Property, make_properties

@make_properties
class GlobalToTile(CodeLibraryNode):
    stride = Property(dtype=int, default=1, allow_none=False, desc="Stride")

    def __init__(self, name, input_names, output_names, stride):
        super().__init__(name=name, input_names=input_names, output_names=output_names)
        self.stride = stride

    def generate_code(self, inputs, outputs):
        assert len(inputs) == 1
        assert len(outputs) == 1
        glb_matrix = next(iter(inputs), None)
        frag_matrix = next(iter(outputs), None)
        code = f"nvcuda::wmma::load_matrix_sync({frag_matrix}, {glb_matrix}, {self.stride});"
        return code