import dace
from dace.data import ListProperty, make_properties
# from dace.libraries.ascend_mem_move import GlobalToVECIN, VECOUTToGlobal
from dace.libraries.standard.nodes.code import CodeLibraryNode
from dace.properties import Property


@make_properties
class VecUnit(CodeLibraryNode):
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
        return f"//TODO"

    def generate_required_member_definition(self):
        return f"//TODO"

    def generate_required_member_initialization(self):
        return f"//TODO"


def two():
    sdfg = dace.SDFG("ascendc_test_2")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (256 * 32, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    a_dev = sdfg.add_array("ascend_A", (256 * 32, ),
                           dace.float16,
                           dace.dtypes.StorageType.Ascend_Global,
                           transient=True)
    b_host = sdfg.add_array("B", (256 * 32, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    b_dev = sdfg.add_array("ascend_B", (256 * 32, ),
                           dace.float16,
                           dace.dtypes.StorageType.Ascend_Global,
                           transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    bhc = state.add_access("B")
    bdc = state.add_access("ascend_B")
    frag_a = sdfg.add_array("frag_A", (256, ), dace.float16, dace.dtypes.StorageType.Ascend_VECIN, transient=True)
    frag_b = sdfg.add_array("frag_B", (256, ), dace.float16, dace.dtypes.StorageType.Ascend_VECOUT, transient=True)
    #af = state.add_access("frag_A")
    #bf = state.add_access("frag_B")

    dev_entry, dev_exit = state.add_map(name="copy_map_outer",
                                        ndrange={"i": dace.subsets.Range([(0, 256 * 32 - 1, 256 * 32)])},
                                        schedule=dace.dtypes.ScheduleType.Ascend_Device)
    tblock_entry, tblock_exit = state.add_map(name="copy_map_inner",
                                              ndrange={"ii": dace.subsets.Range([(0, 256 * 32 - 1, 256)])},
                                              schedule=dace.dtypes.ScheduleType.Ascend_AiCoreGroup)

    #glb_to_vecin = GlobalToVECIN(name="glb_to_vecin_a", input_names=["IN_A"], output_names=["OUT_frag_A"], queue_length=1, load_length=256)
    glb_to_vecin = state.add_access("frag_A")
    #vecout_to_glb = VECOUTToGlobal(name="vecout_to_glb_b", input_names=["IN_frag_B"], output_names=["OUT_B"], queue_length=1, load_length=256)
    libnode = state.add_access("frag_B")
    #libnode = VecUnit(name="vecout_to_glb_b", input_names=["IN_frag_A"], output_names=["OUT_frag_B"], queue_length=1, load_length=256)

    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet(f"A[0:{256*32}]"))
    state.add_edge(adc, None, dev_entry, "IN_A", dace.memlet.Memlet(f"ascend_A[0:{256*32}]"))
    state.add_edge(dev_entry, "OUT_A", tblock_entry, "IN_A", dace.memlet.Memlet(f"ascend_A[0:{256*32}]"))
    state.add_edge(tblock_entry, "OUT_A", glb_to_vecin, None, dace.memlet.Memlet(f"ascend_A[i + ii:i + ii + 256]"))
    state.add_edge(glb_to_vecin, None, libnode, None, dace.memlet.Memlet(f"frag_A[0:256]"))
    state.add_edge(libnode, None, tblock_exit, "IN_B", dace.memlet.Memlet(f"ascend_B[i + ii:i + ii + 256]"))
    state.add_edge(tblock_exit, "OUT_B", dev_exit, "IN_B", dace.memlet.Memlet(f"ascend_B[i + ii:i + ii + 256]"))
    state.add_edge(dev_exit, "OUT_B", bdc, None, dace.memlet.Memlet(f"ascend_B[0:{256*32}]"))
    state.add_edge(bdc, None, bhc, None, dace.memlet.Memlet(f"B[0:{256*32}]"))

    for n in [dev_entry, tblock_entry]:
        n.add_in_connector("IN_A")
        n.add_out_connector("OUT_A")

    for n in [dev_exit, tblock_exit]:
        n.add_in_connector("IN_B")
        n.add_out_connector("OUT_B")

    #libnode.add_in_connector("IN_A")
    #libnode.add_out_connector("OUT_B")

    #t = state.add_tasklet(name="assign", inputes={"_in"}, outputs={"_out"}, code="_out = _in")

    sdfg.save("ascend_2.sdfgz")
    return sdfg


s = two()
# s.compile()
