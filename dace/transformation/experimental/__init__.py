import os
import importlib.util
import pkgutil
import pathlib

from dace.config import Config
from dace.transformation.transformation import TransformationBase
from dace.transformation import Pass


def _recursive_import_transformations():
    base_path = pathlib.Path(__file__).parent
    package_root = ".".join(__name__.split("."))  # = "dace.transformation.experimental"
    current_module = importlib.import_module(package_root)

    debug_print = Config.get_bool('debugprint')

    for path in base_path.rglob("*.py"):
        if path.name == "__init__.py":
            continue

        rel_path = path.relative_to(base_path)
        module_name = ".".join([package_root] + list(rel_path.with_suffix("").parts))

        try:
            module = importlib.import_module(module_name)

            # For every subclass of Transformation | Pass (except base class), attach to current module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and (issubclass(attr, TransformationBase) or issubclass(attr, Pass))
                        and attr not in (TransformationBase, Pass) and not hasattr(current_module, attr_name)):
                    setattr(current_module, attr_name, attr)
                    if debug_print:
                        print(f"[Experimental] Registered transformation: {attr_name}")

        except Exception as e:
            if debug_print:
                print(f"Failed to import {module_name}: {e}")


_recursive_import_transformations()
