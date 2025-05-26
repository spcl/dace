import sys
import pathlib

import dace.transformation

template_for_transformation = """
# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import transformation


@transformation.explicit_cf_compatible
class {ExperimentalEmptyTransformation}(transformation.MultiStateTransformation):
    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index, sdfg: dace.SDFG, permissive=False):
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        return

test_loaded_generic = "{GENERIC_TRANSFORMATION_LOADED}"
"""

TEMPLATE_STR_1 = "This is a generic transformation loaded from the experimental folder"
TEMPLATE_STR_2 = "This is a generic transformation loaded from the experimental folder, but with a different name"


def test_experimental_imports():
    try:
        base_dir = pathlib.Path(__file__).parent.parent / "dace" / "transformation" / "experimental"
        generic_folder = base_dir / "generic_folder"
        generic_folder.mkdir(parents=True, exist_ok=True)

        file1 = generic_folder / "empty_transformation.py"
        file2 = base_dir / "empty_transformation_2.py"

        template1 = template_for_transformation.format(
            ExperimentalEmptyTransformation="ExperimentalEmptyTransformation",
            GENERIC_TRANSFORMATION_LOADED=TEMPLATE_STR_1)
        template2 = template_for_transformation.format(
            ExperimentalEmptyTransformation="ExperimentalEmptyTransformation2",
            GENERIC_TRANSFORMATION_LOADED=TEMPLATE_STR_2)

        file1.write_text(template1)
        file2.write_text(template2)

        sys.modules.pop("dace", None)
        for name in list(sys.modules):
            if name.startswith("dace.transformation.experimental"):
                del sys.modules[name]

        import dace

        mod1_name = "dace.transformation.experimental.generic_folder.empty_transformation"
        mod2_name = "dace.transformation.experimental.empty_transformation_2"

        assert mod1_name in sys.modules, f"{mod1_name} not in sys.modules"
        assert mod2_name in sys.modules, f"{mod2_name} not in sys.modules"

        mod1 = sys.modules[mod1_name]
        mod2 = sys.modules[mod2_name]

        assert getattr(mod1, "test_loaded_generic") == TEMPLATE_STR_1
        assert getattr(mod2, "test_loaded_generic") == TEMPLATE_STR_2

        sdfg = dace.SDFG("test_sdfg")
        state = sdfg.add_state("test_state")

        mod1.ExperimentalEmptyTransformation().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)
        mod2.ExperimentalEmptyTransformation2().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)

        import dace.transformation.experimental as exp
        exp.ExperimentalEmptyTransformation().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)
        exp.ExperimentalEmptyTransformation2().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)

        from dace import library

    finally:
        # Cleanup
        try:
            if file1.exists():
                file1.unlink()
            if file2.exists():
                file2.unlink()
            if generic_folder.exists():
                generic_folder.rmdir()
        except Exception as cleanup_err:
            print(f"Cleanup failed: {cleanup_err}")
