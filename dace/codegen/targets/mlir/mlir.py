# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import registry, dtypes
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpu import CPUCodeGen
from dace.sdfg import nodes


@registry.autoregister_params(name='mlir')
class MLIRCodeGen(TargetCodeGenerator):
    target_name = 'mlir'
    title = 'MLIR'

    def __init__(self, frame_codegen, sdfg):
        self._codeobjects = []
        self._cpu_codegen = frame_codegen.dispatcher.get_generic_node_dispatcher(
        )
        frame_codegen.dispatcher.register_node_dispatcher(
            self, self.node_dispatch_predicate)

    def get_generated_codeobjects(self):
        return self._codeobjects

    def node_dispatch_predicate(self, sdfg, state, node):
        return isinstance(
            node, nodes.Tasklet) and node.language == dtypes.Language.MLIR

    def generate_node(self, sdfg, dfg, state_id, node, function_stream,
                      callsite_stream):
        if self.node_dispatch_predicate(sdfg, dfg, node):
            function_uid = str(sdfg.sdfg_id) + "_" + str(state_id) + "_" + str(
                dfg.node_id(node))
            node.code.code = node.code.code.replace(
                "mlir_entry", "mlir_entry_" + function_uid)
            self._codeobjects.append(
                CodeObject(node.name, node.code.code, "mlir", MLIRCodeGen,
                           node.name + "_Source"))

        self._cpu_codegen.generate_node(sdfg, dfg, state_id, node,
                                        function_stream, callsite_stream)

    @staticmethod
    def cmake_options():
        options = []
        return options
