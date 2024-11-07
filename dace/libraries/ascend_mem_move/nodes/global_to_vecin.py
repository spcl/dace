import dace
from dace.libraries.standard.nodes.code import CodeLibraryNode
from dace.properties import Property, ListProperty, make_properties

@make_properties
class GlobalToVECIN(CodeLibraryNode):
    queue_length = Property(dtype=int, allow_none=False, default=1, desc="Queue Length")
    input_names = ListProperty(element_type=str, allow_none=False, default=[], desc="Input Names")
    output_names = ListProperty(element_type=str, allow_none=False, default=[], desc="Output Names")
    load_length = Property(dtype=int, allow_none=False, default=1, desc="Load Length")

    def __init__(self, name, input_names, output_names, queue_length, load_length):
        self.queue_length = queue_length
        self.input_names = input_names
        self.output_names = output_names
        self.load_length = load_length
        super().__init__(name=name, input_names=input_names, output_names=output_names)

    def generate_code(self, inputs, outputs):
        assert len(inputs) == 1
        assert len(outputs) == 1
        glb_vec = next(iter(outputs), None)
        frag_vec = next(iter(inputs), None)
        code = f"AscendC::LocalTensor<half> {frag_vec}_local = in_queue_{frag_vec}.AllocTensor<half>();\n" + \
               f"DataCopy({frag_vec}, {glb_vec}, {self.load_length});\n" + \
               f"in_queue_{frag_vec}.EnQue({frag_vec});\n"
        return code

    def generate_required_member_definition(self):
        return f"AscendC::TQue<AscendC::QuePosition::VECIN, {self.queue_length}> in_queue_{self.input_names[0]};"

    def generate_required_member_initialization(self):
        return f"pipe.InitBuffer(in_queue_{self.input_names[0]}, {self.queue_length}, {self.load_length}*sizeof(half));"