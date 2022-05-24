import copy
import ast

from typing import Any, Generator, List, Tuple, Optional

from dace import SDFG, nodes
from dace.sdfg.analysis import cutout as cutter
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.estimator import enumeration as en
from dace.optimization.cutout_tuning import CutoutSpace


class OTFMapFusionSpace(CutoutSpace):
    def name(self) -> str:
        return 'OTFMapFusionSpace'

    def apply_config(self, cutout: SDFG, config: Any, make_copy: bool = True) -> Optional[SDFG]:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        num_trans = cutout_.apply_transformations_repeated(OTFMapFusion)
        if num_trans == 0:
            return None

        return cutout_

    def translate_config(self, cutout: SDFG, sdfg: SDFG, config: Any) -> Any:
        state_id, map_ids = config
        maps = map(lambda m_id: cutout.node(state_id).node(m_id), map_ids)

        sstate_id = int(cutout.name.split("_")[-1])
        smap_ids = list(map(lambda m: sdfg.node(sstate_id).node_id(m), maps))

        return sstate_id, smap_ids

    def encode_config(self, config: Any) -> str:
        return str(config)

    def decode_config(self, config: str) -> Any:
        param, target = ast.literal_eval(config)
        return param, target

    def cutouts(self, sdfg: SDFG) -> Generator[SDFG, None, None]:
        for state in sdfg.nodes():
            cutout = cutter.cutout_state(state, *state.nodes(), make_copy=False)
            yield cutout

    def configurations(self, cutout: SDFG) -> Generator[Tuple[int, List[int]], None, None]:
        state_id = 0
        map_ids = []
        for node in cutout.start_state.nodes():
            if isinstance(node, nodes.MapEntry):
                map_ids.append(cutout.start_state.node_id(node))

        yield state_id, map_ids
