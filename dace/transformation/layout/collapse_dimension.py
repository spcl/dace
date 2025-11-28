import dace
from typing import Dict, List, Any, Tuple
from dace.sdfg.graph import Edge, EdgeT
from dace.transformation import pass_pipeline as ppl
from dataclasses import dataclass
import copy
import re
import sympy
from sympy import simplify


@dataclass
class CollapseDimensions(ppl.Pass):

    def _block_maps():
        pass

    def _simplify_memlets():
        pass

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes & ppl.Modifies.Memlets & ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self, collapse_map: Dict[str, Tuple[int, int]], verbose: bool = False):
        self._collapse_map = collapse_map
        self._verbose = verbose

    def _collapse_dimensions(self, sdfg: dace.SDFG, array_name: str, array: dace.data.Array, dimensions_to_collapse: Tuple[int, int]):
        old_shape = copy.deepcopy(array.shape)
        old_strides = copy.deepcopy(array.strides)
        array._old_shape = old_shape
        array._old_strides = old_strides

        new_shape = []
        for i, dim in enumerate(old_shape):
            if i == dimensions_to_collapse[0]:
                new_shape.append(dim * old_shape[i+1])
                continue
            elif i == dimensions_to_collapse[1]:
                continue
            else:
                new_shape.append(dim)
        new_shape = tuple(new_shape)
        array.set_shape(new_shape=new_shape)

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    # If array is passed as a complete subset, collapse in subset too
                    # If a subset is passed, then need to special treatment is needed
                    for ie in state.in_edges(node):
                        if ie.data.data == array_name:
                            if ie.data.subset == dace.subsets.Range([(0, d-1, 1) for d in array.shape]):
                                self._collapse_dimensions(node.sdfg, ie.dst_conn, node.sdfg.arrays[ie.dst_conn], dimensions_to_collapse)
                            else:
                                raise Exception("Partial subsets passed to nestedSDFGs are not supported yet")
                    for oe in state.out_edges(node):
                        # Do not do it twice on inout connectors
                        if oe.data.data == array_name and oe.dst_conn not in node.in_connectors:
                            if oe.data.subset == dace.subsets.Range([(0, d-1, 1) for d in array.shape]):
                                self._collapse_dimensions(node.sdfg, oe.src_conn, node.sdfg.arrays[oe.src_conn], dimensions_to_collapse)
                            else:
                                raise Exception("Partial subsets passed from nestedSDFGs are not supported yet")

    def _collapse_memlets(self, sdfg: dace.SDFG, array_name: str, array: dace.data.Array, dimensions_to_collapse: Tuple[int, int]):
        old_strides = array._old_strides

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    # Partial subset check has been done by collapse dimensions
                    # If array name appears in the inputs/outputs of nested SDFG collapse dimensions in the nested SDFG too
                    if (any(ie.data.data == array_name for ie in state.in_edges(node))):
                        ie = {ie.data.data == array_name for ie in state.in_edges(node)}.pop()
                        self._collapse_memlets(node.sdfg, ie.dst_conn, node.sdfg.arrays[ie.dst_conn], dimensions_to_collapse)
                    elif (any(oe.data.data == array_name for oe in state.in_edges(node))):
                        oe = {oe.data.data == array_name for oe in state.in_edges(node)}.pop()
                        self._collapse_memlets(node.sdfg, oe.src_conn, node.sdfg.arrays[oe.src_conn], dimensions_to_collapse)

            for edge in state.edges():
                if edge.data.subset is not None and edge.data.data == array_name:
                    subset: dace.subsets.Range = edge.data.subset
                    if subset.num_elements_exact() != 1:
                        new_range_list = []
                        for i, (b,e,s) in enumerate(subset):
                            if i == dimensions_to_collapse[0]:
                                next_b,next_e,next_s = subset[i+1]
                                assert s == 1
                                assert next_s == 1
                                new_b = b * old_strides[i] + next_b
                                new_e = e * old_strides[i] + next_e
                                new_s = s
                                new_range_list.append((new_b, new_e, new_s))
                            elif i == dimensions_to_collapse[1]:
                                continue
                            else:
                                new_range_list.append((b,e,s))
                    else:
                        new_range_list = []
                        for i, (b,e,s) in enumerate(subset):
                            if i == dimensions_to_collapse[0]:
                                next_b,next_e,next_s = subset[i+1]
                                assert s == 1
                                assert next_s == 1
                                new_b = b * old_strides[i] + next_b
                                new_e = e * old_strides[i] + next_e
                                new_s = s
                                new_range_list.append((new_b, new_e, new_s))
                            elif i == dimensions_to_collapse[1]:
                                continue
                            else:
                                new_range_list.append((b,e,s))
                    edge.data = dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list))

    def _collapse_indices(self, text: str, array_name: str, collapse_idx: int, keep_idx: int, stride: List[dace.symbolic.SymExpr]):
        """
        Transform all <array>[i0, i1, ..., iN] into a version where:

            new_index = indices[collapse_idx] * stride + indices[keep_idx]

        And the resulting pattern becomes:

            <array>[..., new_index, ..., indices[keep_idx], ...]

        Arguments:
            text          - input text
            array_name    - array name to match
            collapse_idx  - int, the index to multiply by stride
            keep_idx      - int, the index to add and also keep as 2nd dim
            stride        - string or int
        """
        
        # Match array[index_list]
        pattern = re.compile(
            rf'({array_name})\s*\[\s*([^\]]*?)\s*\]'
        )


        def repl(match):
            arr = match.group(1)
            idx_raw = match.group(2)

            # Split indices by commas
            idxs = [x.strip() for x in idx_raw.split(",")]

            # Sanity check
            if collapse_idx >= len(idxs) or keep_idx >= len(idxs):
                raise Exception("?")

            d_c = idxs[collapse_idx]
            d_k = idxs[keep_idx]

            # Construct new collapsed index
            new_idx = f"{d_c} * {stride} + {d_k}"

            # Replace collapse_idx with new index
            idxs[collapse_idx] = new_idx

            # keep_idx stays unchanged (as required)

            new_idx_list = ", ".join(idxs)
            return f"{arr}[{new_idx_list}]"

        return pattern.sub(repl, text)

    def _collapse_interstate_edges(self, sdfg: dace.SDFG, array_name: str, array: dace.data.Array, dimensions_to_collapse: Tuple[int, int]):
        old_strides = array._old_strides

        # All interstate edge accesses neet be of form <array_name>[d0, d1, d2]
        # With strides [s0, s1, s2]
        # New interstate access needs to be of form [d0 * s1 + d2, d2]
        for edge in sdfg.all_interstate_edges():
            new_assignments = dict()
            for k, v in edge.data.assignments:
                new_v = self._collapse_indices(v, dimensions_to_collapse[0], dimensions_to_collapse[1], old_strides[dimensions_to_collapse[0]])
                new_assignments[k] = new_v
            edge.data.assignments = new_assignments

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        # For each array in the list:
        # We have a mask for the dimensions to collapse e.g. we have a rank 3 tensor [M, N, K]
        # We want to collapse M and N dimensions and have a rank 2 tensor of shape [M*N, K]

        # Then we can replace the array with the new dimension, then we need to also 
        # collapse all memlets expressions
        # A memlet expression has the initial form of:
        # [mb:me:ms, nb:ne:ns, kb:ke:ks]
        # To collapsed we need a distinction:
        # If it he subset volume is not 1, for example:
        # [mb:me:ms, nb:ne:ns, kb:ke:ks] (For simplicity we assume ms == 1, ns == 1)
        # We collapsed the start and end subsets:
        # [(mb * nb):(me * ne):1, kb:ke:ks]
        # Consider the access:
        # [z, i:i+4, j:j+2048] if (z, i) are collapsed:
        # New subset wille be:
        # [(z*i):(z*i+4), j:j+2048]
        # Special case of a single element access is slightly different.
        # If we access [m, n, k] with strides (NK, K, 1) then we are actually accessing the 1D offset of:
        # mNK + nK + k
        # To have the same access we need to:
        # [m, n, k] -> [m * N + n, k]
        # Which would evaluate to the same one-dimensional access of:
        # mNK + nK + k

        # The collapse function only supports collapsing neighboring arrays. To collapsed none-neighboring dimensions,
        # forst run index permutation transformation

        # 1. First pad dimensions of all arrays
        for array_name, dimensions_to_collapse in self._collapse_map.items():
            array = sdfg.arrays[array_name]
            self._collapse_dimensions(sdfg, array_name, array, dimensions_to_collapse)
            self._collapse_memlets(sdfg, array_name, array, dimensions_to_collapse)
            self._collapse_interstate_edges(sdfg, array_name, array, dimensions_to_collapse)

        return 0
