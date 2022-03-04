import dace
import sympy
import itertools

from typing import Union, Dict
from pathlib import Path

from kerncraft import machinemodel as mm
from kerncraft import kernel as kk
from kerncraft.models import roofline, ecm

from dace.performance.analysis.flop_counter import FLOPCounter

class ArgMock(object):
    pass

class KerncraftWrapper():
    def __init__(self, machine_file_path: Union[str, Path], cache_predictor: str, unit: str = "FLOP/s", cores: int = -1):
        self._machine_model = mm.MachineModel(path_to_yaml=machine_file_path)

        self._cache_predictor = cache_predictor
        self._unit = unit

        self._cores = self._machine_model._data["cores per socket"]
        if cores > 0:
            self._cores = min(cores, self._cores)

    def roofline(self, kernel: dace.SDFG, values: Dict[str, int]) -> Dict:
        kernel_desc = KerncraftWrapper.kerncraftify(kernel, values)
        kernel_desc.clear_state()
        for k, v in values.items():
            kernel_desc.set_constant(str(k), v)

        args = ArgMock()
        args.verbose = 0
        args.cores = self._cores
        args.cache_predictor = self._cache_predictor
        args.unit = self._unit
        model = roofline.RooflineFLOP(kernel_desc, self._machine_model, args=args)
        
        model.analyze()

        # TODO: Parse kerncraft report into common format for different backends
        report = model.results
        report["FLOP"] = kernel_desc._flops

        precision = "SP" if kernel_desc.datatype[0] == "float" else "DP"
        flops_per_cycle = self._machine_model._data["FLOPs per cycle"][precision]["total"]
        cores_per_socket = self._machine_model._data["cores per socket"]
        clock = self._machine_model._data["clock"].with_prefix("G").value
        peak_flops = flops_per_cycle * cores_per_socket * clock
        report["Peak FLOP/s"] = peak_flops

        # TODO: PrefixedUnit
        flops = report['min performance']['FLOP/s'].with_prefix("G").value
        # Correct ?
        flops = self._machine_model._data["sockets"] * flops
        clock_speed = self._machine_model._data["clock"].with_prefix("G").value
        report['runtime'] = 0.0
        if flops > 0.0:
            report['runtime'] = clock_speed / flops

        return report

    def ecm(self, kernel: dace.SDFG, values: Dict[str, int]):
        raise NotImplementedError()
        
        kernel_desc = Kerncraft.kerncraftify(kernel, values)
        kernel_desc.clear_state()
        for k, v in values.items():
            kernel_desc.set_constant(str(k), v)

        args = ArgMock()
        args.cores = self._cores
        args.cache_predictor = self._cache_predictor
        args.unit = self._unit

        model = ecm.ECM(kernel_desc, self._machine_model, args=args)
        
        model.analyze()

        # TODO: Parse kerncraft report into common format for different backends
        return model.results

    @staticmethod
    def kerncraftify(kernel: dace.SDFG, values: Dict[str, int]):
        tasklet = None
        maps = []
        for node in kernel.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                maps.append(node.map)
            elif isinstance(node, dace.nodes.Tasklet):
                assert tasklet is None
                # Single tasklet assumption
                tasklet = node

        # TODO: symbolic flop
        flop = FLOPCounter.count(tasklet)
        desc = {}
        desc["flops"] = {}
        desc["flops"]["+"] = int(flop)

        desc["loops"] = []
        offsets = []
        for map in maps[::-1]:
            for i, index in enumerate(map.params):
                start, stop, step = [
                    expr.approx
                    if isinstance(expr, dace.symbolic.SymExpr)
                    else sympy.sympify(expr)
                    for expr in map.range.ranges[i]
                ]

                if len(step.free_symbols.intersection(kernel.free_symbols)) > 0:
                    raise ValueError("Wrong loop format")

                offset = start
                stop = sympy.simplify(stop - offset)
                start = sympy.simplify(start - offset)
                offsets.append((index, offset))
                if (
                    len(start.free_symbols.intersection(kernel.free_symbols)) > 0
                    or len(stop.free_symbols.intersection(kernel.free_symbols)) > 0
                ):
                    raise ValueError("Wrong loop format")

                loop = {
                    "index": index,
                    "start": str(start),
                    "stop": str(stop),
                    "step": str(step),
                }
                desc["loops"].append(loop)

        desc["arrays"] = {}
        desc["data sources"] = {}
        desc["data destinations"] = {}
        all_dtype = "float"
        for edge in kernel.start_state.edges():
            if not (
                isinstance(edge.src, dace.nodes.Tasklet)
                or isinstance(edge.dst, dace.nodes.Tasklet)
            ):
                continue

            memlet = edge.data
            name = memlet.data
            array = kernel.arrays[name]
            
            dtype = array.dtype.ctype
            supported_dtype = ["float", "double"]
            if dtype not in supported_dtype:
                dtype = "float"

            if dtype == "double":
                all_dtype = "double"

            desc["arrays"][name] = {
                "type": (dtype,),
                "dimension": [str(dim) for dim in array.shape],
            }

            index_accesses = []
            for dim in memlet.subset.ranges:
                dim_accesses = []

                start, stop, step = [
                    expr.approx
                    if isinstance(expr, dace.symbolic.SymExpr)
                    else sympy.sympify(expr)
                    for expr in dim
                ]
                if len(step.free_symbols) > 0:
                    raise ValueError("Unsupported access")

                # TODO: native sympy replace
                start = str(start)
                stop = str(stop)
                for index, offset in offsets[::-1]:
                    off = f"({index} + {offset})"
                    start = start.replace(index, off)
                    stop = stop.replace(index, off)

                start = sympy.simplify(sympy.sympify(start))
                stop = sympy.simplify(sympy.sympify(stop))

                stop = stop + step
                stop = sympy.simplify(stop)

                dim_accesses.append(str(start))
                start = start + step
                start = sympy.simplify(start)

                while sympy.simplify(start - stop) != 0:
                    dim_accesses.append(str(start))

                    start = start + step
                    start = sympy.simplify(start)

                index_accesses.append(dim_accesses)

            type = ""
            if isinstance(edge.src, dace.nodes.MapEntry):
                type = "data sources"
            elif isinstance(edge.dst, dace.nodes.MapExit):
                type = "data destinations"
            else:
                raise ValueError()

            desc[type][name] = []
            for access in itertools.product(*index_accesses):
                ac = list(access)
                desc[type][name].append(ac)

        # Only single datatype for all arrays supported by kerncraft
        for array in desc["arrays"]:
            desc["arrays"][array]["type"] = (all_dtype,)

        return kk.KernelDescription(desc)
