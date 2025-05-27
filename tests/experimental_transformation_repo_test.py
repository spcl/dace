import sys
import pathlib
import shutil
import importlib

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

# Create some dummy experimental transformations, ensure they can be loaded, can call some methods of transformations


def test_experimental_transformation_import():
    # Store original state for restoration
    original_modules = set(sys.modules.keys())
    file1 = None
    file2 = None
    generic_folder = None

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

        assert file1.exists(), f"{file1} does not exist after writing!"
        assert file2.exists(), f"{file2} does not exist after writing!"

        # Clear existing dace modules
        modules_to_clear = [name for name in sys.modules if name.startswith("dace")]
        for name in modules_to_clear:
            del sys.modules[name]

        import dace

        mod1_name = "dace.transformation.experimental.generic_folder.empty_transformation"
        mod2_name = "dace.transformation.experimental.empty_transformation_2"

        assert mod1_name in sys.modules, f"{mod1_name} not in sys.modules"
        assert mod2_name in sys.modules, f"{mod2_name} not in sys.modules"

        sdfg = dace.SDFG("test_sdfg")
        state = sdfg.add_state("test_state")

        import dace.transformation.experimental as exp
        exp.ExperimentalEmptyTransformation().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)
        exp.ExperimentalEmptyTransformation2().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)

        mod1 = sys.modules[mod1_name]
        mod2 = sys.modules[mod2_name]

        assert getattr(mod1, "test_loaded_generic") == TEMPLATE_STR_1
        assert getattr(mod2, "test_loaded_generic") == TEMPLATE_STR_2

        mod1.ExperimentalEmptyTransformation().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)
        mod2.ExperimentalEmptyTransformation2().can_be_applied(sdfg=sdfg, graph=state, expr_index=0)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # Comprehensive cleanup
        cleanup_errors = []

        # 1. Remove files
        try:
            if file1 and file1.exists():
                file1.unlink()
        except Exception as e:
            cleanup_errors.append(f"Failed to remove {file1}: {e}")

        try:
            if file2 and file2.exists():
                file2.unlink()
        except Exception as e:
            cleanup_errors.append(f"Failed to remove {file2}: {e}")

        try:
            if generic_folder and generic_folder.exists():
                shutil.rmtree(generic_folder)
        except Exception as e:
            cleanup_errors.append(f"Failed to remove {generic_folder}: {e}")

        # 2. Remove all modules that weren't there originally
        try:
            current_modules = set(sys.modules.keys())
            modules_to_remove = current_modules - original_modules
            for module_name in modules_to_remove:
                if module_name in sys.modules:
                    del sys.modules[module_name]
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up modules: {e}")

        # 3. Force reimport of core dace modules if they existed originally
        try:
            if "dace" in original_modules:
                # Remove and reimport dace to reset its state
                dace_modules = [name for name in sys.modules if name.startswith("dace")]
                for name in dace_modules:
                    if name in sys.modules:
                        del sys.modules[name]
                import dace
                # Do not import experimental transformations again
        except Exception as e:
            cleanup_errors.append(f"Failed to reload dace: {e}")

        # 4. Clear any class registrations from the experimental module
        try:
            import dace.transformation.experimental as exp_mod
            # Remove any dynamically added attributes
            for attr_name in dir(exp_mod):
                if not attr_name.startswith('_'):
                    attr = getattr(exp_mod, attr_name)
                    if hasattr(attr, '__module__') and ('empty_transformation' in getattr(attr, '__module__', '')
                                                        or 'empty_transformation_2' in getattr(attr, '__module__', '')):
                        delattr(exp_mod, attr_name)
        except Exception as e:
            cleanup_errors.append(f"Failed to clean experimental module: {e}")

        # Report cleanup errors but don't fail the test
        if cleanup_errors:
            print("Cleanup warnings:")
            for error in cleanup_errors:
                print(f"  - {error}")


# A simple test to check DaCe calls function without having and files in the experimental folder
# Other tests should get it too
def test_no_experimental_transformation_import():
    import dace
    sdfg = dace.SDFG("test_sdfg")
    state = sdfg.add_state("test_state")
    a1_name, a1 = sdfg.add_array(name="A", shape=(10, ), dtype=dace.float64, transient=False)
    a2_name, _ = sdfg.add_array(name="B", shape=(10, ), dtype=dace.float64, transient=False)
    an1 = state.add_access(a1_name)
    an2 = state.add_access(a2_name)
    state.add_edge(an1, None, an2, None, dace.Memlet.from_array(dataname=a1_name, datadesc=a1))
    sdfg.compile()


if __name__ == "__main__":
    test_experimental_transformation_import()
    test_no_experimental_transformation_import()
    print("Test completed successfully.")
