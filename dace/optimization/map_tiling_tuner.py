# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from typing import Generator, Tuple, Dict, List

from dace import dtypes
from dace.optimization import cutout_tuner
from dace.transformation import dataflow as df
from dace.transformation import helpers as xfh
from dace.sdfg.analysis.cutout import SDFGCutout

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class MapTilingTuner(cutout_tuner.CutoutTuner):

    def __init__(self,
                 sdfg: dace.SDFG,
                 measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="MapTiling", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self) -> Generator[Tuple[dace.SDFG, str], None, None]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                if xfh.get_parent_map(state, node) is not None:
                    continue

                node_id = state.node_id(node)
                state_id = self._sdfg.node_id(state)
                subgraph_nodes = state.scope_subgraph(node).nodes()
                cutout = SDFGCutout.singlestate_cutout(state, *subgraph_nodes)
                yield cutout, f"{state_id}.{node_id}.{node.label}"

    def space(self, map_entry: dace.nodes.MapEntry) -> Generator[Tuple[int], None, None]:
        choices = [
            None,
            (64, 8, 1),
        ]

        return choices

    def config_from_key(self, key: str, **kwargs) -> List[int]:
        if key == "None":
            return None

        return list(map(lambda k: int(k), key.split(".")))

    def apply(self, config: List[int], label: str, **kwargs) -> None:
        if config is None:
            return

        state_id, node_id, _ = label.split(".")
        map_entry = self._sdfg.node(int(state_id)).node(int(node_id))
        df.MapTiling.apply_to(self._sdfg, map_entry=map_entry, options={"tile_sizes": config})

    def pre_evaluate(self, cutout: dace.SDFG, measurements: int, **kwargs) -> Dict:
        cutout.start_state.instrument = self.instrument

        map_entry = None
        for node in cutout.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(cutout.start_state, node) is None:
                map_entry = node
                break
        assert map_entry is not None

        new_kwargs = {
            "space_kwargs": {
                "map_entry": map_entry
            },
            "cutout": cutout.to_json(),
            "map_entry_id": cutout.start_state.node_id(map_entry),
            "measurements": measurements,
            "key": lambda point: "None" if point is None else ".".join(map(lambda p: str(p), point))
        }
        return new_kwargs

    def evaluate(self, config, cutout, map_entry_id: int, measurements: int, **kwargs) -> float:
        cutout_ = dace.SDFG.from_json(cutout)
        map_ = cutout_.start_state.node(map_entry_id)
        if config == "None":
            df.MapTiling.apply_to(cutout_, map_entry=map_, options={"tile_sizes": config})

        return self.measure(cutout_, measurements)
