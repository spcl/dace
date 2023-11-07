from typing import Any, Dict, Optional, Set

from dace.transformation.passes import variable_fission
from dace import SDFG

class ArrayFission(variable_fission.VariableFission):
    fission_arrays = True
    def apply_pass(
            self, sdfg: SDFG,
            pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        super().apply_pass(sdfg, pipeline_results)