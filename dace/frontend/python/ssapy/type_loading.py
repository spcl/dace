
from ast import parse
from tkinter.font import names

from .coverage import global_module_types, global_class_attrs, get_builtins_definitons, get_builtins_types
from .type_types import Type
from .ssa_preprocess import SSA_Preprocessor
from .ssa_transpiler import SSA_Transpiler
from .ssa_postprocess import SSA_Postprocessor
from .type_inferrer_stmt import TypeInferrer


def load_typefile(filename, *, namespace):

    with open(filename, mode="r") as file:
        code = file.read()

    tree = parse(code)

    # SSA
    SSA_Preprocessor().visit(tree)
    SSA_Transpiler(make_interface=True).visit(tree, get_builtins_definitons())
    SSA_Postprocessor().visit(tree)

    # Type Inference
    inferrer = TypeInferrer()
    new_types = inferrer.visit(tree, get_builtins_types())
    new_classes = inferrer.class_attrs

    prev_module_types = global_module_types.get(namespace, {})
    new_module_types = {**prev_module_types}

    for var, typ in new_types.items():
        if var in prev_module_types:
            prev_type = prev_module_types[var]
            if (prev_type.origin is Type) and (typ.origin is Type):
                old_class = prev_type.args[0].nominal
                new_class = typ.args[0].nominal

                new_types[var] = prev_type
                new_class_attrs = new_classes.pop(new_class)
                new_classes[old_class] = new_class_attrs
            else:
                new_types[var] = typ

    global_class_attrs.update(new_classes)
    global_module_types[namespace] = new_types
