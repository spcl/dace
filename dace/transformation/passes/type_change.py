# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, properties
from typing import Optional

from dace.transformation.transformation import explicit_cf_compatible

@properties.make_properties
@explicit_cf_compatible
class TypeChange(ppl.Pass):
    CATEGORY: str = 'Helper'

    def __init__(self, from_type: dace.typeclass = None, to_type: dace.typeclass = None):
        self._from_type = from_type
        self._to_type = to_type
        self._swaps_count = 0


    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing


    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False


    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        if hasattr(sdfg, "orig_sdfg") and sdfg.orig_sdfg:  # apply the pass to orig cpu sdfg
            if hasattr(sdfg.orig_sdfg, "all_sdfgs_recursive"):
                for nested_sdfg in sdfg.orig_sdfg.all_sdfgs_recursive():
                    self._change_sdfg_type(nested_sdfg)

        for nested_sdfg in sdfg.all_sdfgs_recursive():
            self._change_sdfg_type(nested_sdfg)
            
        return self._swaps_count


    def report(self, pass_retval: int) -> str:
        if pass_retval is None:
            return "No arrays found to analyze."
        return f"Analyzed {pass_retval} arrays and printed their types."


    def _change_sdfg_type(self, sdfg: SDFG):
        # Swap nodes
        for node, parent in sdfg.all_nodes_recursive():
            if hasattr(node, "in_connectors"):
                for in_con_name, in_con_type in node.in_connectors.items():
                    if in_con_type == self._from_type:
                        node.in_connectors[in_con_name] = self._to_type
                        self._swaps_count += 1
                    elif type(in_con_type) == dace.dtypes.pointer:
                        base_type = in_con_type.base_type  # int* -> int etc
                        if base_type == self._from_type:
                            node.in_connectors[in_con_name] = dace.dtypes.pointer(self._to_type)
                            self._swaps_count += 1
            

            if hasattr(node, "out_connectors"):
                for out_con_name, out_con_type in node.out_connectors.items():
                    if out_con_type == self._from_type:
                        node.out_connectors[out_con_name] = self._to_type
                        self._swaps_count += 1
                    elif type(out_con_type) == dace.dtypes.pointer:
                        base_type = out_con_type.base_type  # int* -> int etc
                        if base_type == self._from_type:
                            node.out_connectors[out_con_name] = dace.dtypes.pointer(self._to_type)
                            self._swaps_count += 1

        # Swap Symbols
        for sym_name, sym_type in sdfg.symbols.items():
            if sym_type == self._from_type:
                sdfg.symbols[sym_name] = self._to_type
                self._swaps_count += 1

        # Swap array types
        for array_name, array_desc in sdfg.arrays.items():
            if self._from_type is not None and self._to_type is not None:
                if array_desc.dtype == self._from_type:
                    array_desc.dtype = self._to_type
                    self._swaps_count += 1

            
        # Swap dace.data.Structure types and StructureView types
        for array_name, array_desc in sdfg._arrays.items():
            if self._is_structure(array_desc) or self._is_structure_view(array_desc):
                self._change_structure_type(array_desc)
                
            
        # Change dace.data.struct field types
        for array_name, array_desc in sdfg._arrays.items():
            if self._is_structure(array_desc):
                if type(array_desc.dtype) == dace.dtypes.pointer:
                    if type(array_desc.dtype.base_type) == dace.dtypes.struct:
                        for field_name, field_desc in array_desc.dtype.base_type.fields.items():
                            if field_desc == dace.dtypes.pointer(self._from_type):
                                array_desc.dtype.base_type.fields[field_name] = dace.dtypes.pointer(self._to_type)
                                self._swaps_count += 1
    

    def _change_structure_type(self, descriptor: dace.data.Structure):
        # Swap the Structure dtype definition
        if type(descriptor.dtype) == dace.dtypes.pointer:
            if type(descriptor.dtype.base_type) == dace.dtypes.struct:
                self._change_struct_type(descriptor.dtype.base_type)

        self._change_member_types(descriptor)
        

    def _change_struct_type(self, descriptor: dace.dtypes.struct):
        for field_name, field_desc in descriptor.fields.items():
            if type(field_desc) == dace.dtypes.pointer:
                base_type = field_desc.base_type # int* -> int etc

                if type(base_type) == dace.dtypes.struct:
                    self._change_struct_type(field_desc.base_type)
                elif base_type == self._from_type:
                    descriptor.fields[field_name] = dace.dtypes.pointer(self._to_type)
                    self._swaps_count += 1
            elif field_desc == self._from_type:
                    descriptor.fields[field_name] = self._to_type
                    self._swaps_count += 1


    def _change_member_types(self, descriptor):
        if not hasattr(descriptor, "members"):
            raise TypeError(f"Expected type with member attr but got {descriptor}")

        for member_name, member_descriptor in descriptor.members.items():
            if self._is_structure(member_descriptor):
                self._change_structure_type(member_descriptor)
            else:
                if member_descriptor.dtype == self._from_type:
                    member_descriptor.dtype = self._to_type
                    self._swaps_count += 1


    def _is_structure(self, obj):
        return type(obj) == dace.data.Structure


    def _is_pointer(self, obj):
        return type(obj) == dace.dtypes.pointer


    def _is_structure_view(self, obj):
        return type(obj) == dace.data.StructureView
