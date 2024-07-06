# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import TYPE_CHECKING
from dace import registry, dtypes
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpu import CPUCodeGen
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator


@registry.autoregister_params(name='mlir')
class MLIRCodeGen(TargetCodeGenerator):
    target_name = 'mlir'
    title = 'MLIR'

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):
        self._codeobjects = []
        self._cpu_codegen: CPUCodeGen = frame_codegen.dispatcher.get_generic_node_dispatcher()
        frame_codegen.dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)

    def get_generated_codeobjects(self):
        return self._codeobjects

    def node_dispatch_predicate(self, sdfg, state, node):
        return isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.MLIR

    def generate_node(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        if self.node_dispatch_predicate(sdfg, dfg, node):
            function_uid = str(cfg.cfg_id) + "_" + str(state_id) + "_" + str(dfg.node_id(node))
            node.code.code = node.code.code.replace("mlir_entry", "mlir_entry_" + function_uid)
            node.label = node.name + "_" + function_uid
            self._codeobjects.append(CodeObject(node.name, node.code.code, "mlir", MLIRCodeGen, node.name + "_Source"))

        self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    @staticmethod
    def cmake_options():
        options = []
        return options
