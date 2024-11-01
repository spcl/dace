import dace
from dace.libraries.standard.nodes.code import CodeLibraryNode
from dace.properties import Property, make_properties

@make_properties
class TileToGlobal(CodeLibraryNode):
    stride = Property(dtype=int, allow_none=False, default=1, desc="Stride")

    def __init__(self, name, input_names, output_names, stride):
        self.stride = stride
        super().__init__(name=name, input_names=input_names, output_names=output_names)

    def generate_code(self, inputs, outputs):
        assert len(inputs) == 1
        assert len(outputs) == 1
        glb_matrix = next(iter(outputs), None)
        frag_matrix = next(iter(inputs), None)
        code = f"nvcuda::wmma::store_matrix_sync({glb_matrix}, {frag_matrix}, {self.stride}, nvcuda::wmma::mem_row_major);"
        return code