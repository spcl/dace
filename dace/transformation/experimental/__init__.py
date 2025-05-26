import os
import importlib.util
import pkgutil
import pathlib


def _recursive_import_transformations():
    base_path = pathlib.Path(__file__).parent
    package_root = ".".join(__name__.split("."))  # = "dace.transformation.experimental"
    current_module = importlib.import_module(package_root)

    for path in base_path.rglob("*.py"):
        if path.name == "__init__.py":
            continue

        rel_path = path.relative_to(base_path)
        module_name = ".".join([package_root] + list(rel_path.with_suffix("").parts))

        try:
            module = importlib.import_module(module_name)

            # For every subclass of Transformation (except base class), attach to current module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    setattr(current_module, attr_name, attr)
                    print(f"[Experimental] Registered transformation: {attr_name}")

        except Exception as e:
            print(f"Failed to import {module_name}: {e}")


_recursive_import_transformations()
