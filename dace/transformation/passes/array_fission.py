from typing import Any, Dict, Optional, Set

from dace.transformation.passes import variable_fission
from dace import SDFG

class ArrayFission(variable_fission.VariableFission):
    def apply_pass(
            self, sdfg: SDFG,
            pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        fission = variable_fission.VariableFission()
        fission.fission_arrays = True
        return fission.apply_pass(sdfg, pipeline_results)