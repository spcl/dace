import pathlib
import importlib
import sys


def _recursive_import_transformations():
    from dace.config import Config
    from dace.transformation.transformation import TransformationBase
    from dace.transformation import Pass
    import dace.transformation.experimental as experimental_module

    base_path = pathlib.Path(__file__).parent
    package_root = ".".join(__name__.split("."))  # = "dace.transformation.experimental"
    debug_print = Config.get_bool('debugprint')

    for path in base_path.rglob("*.py"):
        if path.name == "__init__.py":
            continue

        rel_path = path.relative_to(base_path)
        module_name = ".".join([package_root] + list(rel_path.with_suffix("").parts))
        if debug_print:
            print(f"[Experimental] Experimental transformation module: {module_name}")

        try:
            # Check if module is already imported to avoid reimporting
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module = importlib.import_module(module_name)

            # For every subclass of Transformation | Pass (except base class), attach to current module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and (issubclass(attr, TransformationBase) or issubclass(attr, Pass))
                        and attr not in (TransformationBase, Pass) and not hasattr(experimental_module, attr_name)
                        and getattr(attr, "__module__", "").startswith("dace.")):
                    setattr(experimental_module, attr_name, attr)

                    if debug_print:
                        print(f"[Experimental] Registered transformation: {attr_name} to module: {attr.__module__}")

        except Exception as e:
            if debug_print:
                print(f"Failed to import {module_name}: {e}")


_recursive_import_transformations()
