# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import ast
import sympy
import itertools

from typing import Any, Generator, Optional

from dace import SDFG, symbolic, properties
from dace.optimization.cutout_tuning.cutout_space import CutoutSpace


class DataLayoutSpace(CutoutSpace):
    def name(self) -> str:
        return 'DataLayoutSpace'

    def apply_config(self, cutout: SDFG, config: Any, make_copy: bool = True) -> Optional[SDFG]:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        key = self.encode_config(config)
        config = self.decode_config(key)

        for array in cutout_._arrays:
            if array in config:
                cutout_._arrays[array].strides = config[array]

        return cutout_

    def translate_config(self, cutout: SDFG, sdfg: SDFG, config: Any) -> Any:
        return config

    def encode_config(self, config: Any) -> str:
        dict_str = ','.join([f'"{k}": "{v}"' for k, v in config.items()])
        dict_str = '{' + dict_str + '}'
        return dict_str

    def decode_config(self, config: str) -> Any:
        config = ast.literal_eval(config)
        for array in config:
            strides = properties.ShapeProperty.from_string(config[array])
            config[array] = strides
        return config

    def cutouts(self, sdfg: SDFG) -> Generator[SDFG, None, None]:
        yield sdfg

    def configurations(self, cutout: SDFG) -> Generator[Any, None, None]:
        groups = self.group_arrays(cutout)

        group_configs = [itertools.permutations(list(range(dims))) for (_, dims), _ in groups]
        global_configs = itertools.product(*group_configs)

        for config in global_configs:
            new_strides = {}
            for i in range(len(groups)):
                group_config = config[i]
                _, group = groups[i]

                for member in group:
                    desc = cutout._arrays[member]
                    strides, total_size = desc.strides_from_layout(*group_config)
                    new_strides[member] = strides

            yield new_strides

    def group_arrays(self, cutout: SDFG):
        groups = {}

        visited = set()
        for state in cutout.nodes():
            for dnode in state.data_nodes():
                if cutout.arrays[dnode.data].transient or dnode.data in visited:
                    continue

                dims = len(dnode.desc(cutout).shape)
                dims = symbolic.evaluate(dims, cutout.constants)
                if state.in_degree(dnode) == 0:
                    type = "input"
                elif state.out_degree(dnode) == 0:
                    type = "output"
                else:
                    type = dnode.data

                group = (type, dims)
                if group not in groups:
                    groups[group] = []

                groups[group].append(dnode.data)

        return list(groups.items())
