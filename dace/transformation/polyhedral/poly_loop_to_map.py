# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" polyhedral Loop to map transformation """

from dace import registry, sdfg as sd
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.polyhedral.poly_builder import PolyhedralBuilder
from dace.properties import Property, make_properties


@registry.autoregister
@make_properties
class PolyLoopToMap(DetectLoop):
    """
    Convert control flow loops into dataflow maps by using the polyhedral model.

    First, converts the SDFG into the polyhedral representation.
    Second, performs dependency analysis on the polyhedral representation.
    Third (optional), optimizes for data-locality and parallelism using Pluto.
    Finally, rebuild the SDFG from the polyhedral representation. In the new
    SDFG parallel loops are turned into maps (optional) and tiled (optional).
    """

    # Options
    use_scheduler = Property(
        dtype=bool,
        default=True,
        desc='If True use Pluto scheduler to optimize the control-flow for '
             'data-locality and parallelism, else control-flow is not changed'
    )

    parallelize_loops = Property(
        dtype=bool,
        default=True,
        desc='If True loops with no loop-carried dependencies are transformed '
             'into dataflow maps, else loops are not parallelized'
    )

    exact_dependency_analysis = Property(
        dtype=bool,
        default=True,
        desc='If True use the exact value dependency analysis, '
             'else use the over-approximating memory dependency analysis'
    )

    tile_size = Property(
        dtype=int,
        default=0,
        desc='Tile all possible loops with this tile size. '
             'If tile size is 0 no tiling is performed '
    )

    use_polytopes = Property(
        dtype=bool,
        default=False,
        desc='If True, generates the new SDFG using Polytopes, else with Ranges'
    )

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        """
        Check if the SDFG is a SCoP (Static Control Part) and therefore can be
        modeled in the polyhedral model.

        For an SDFG to be a SCoP the following has to hold:
        - Structured control flow - only for-loops and if-conditions
          (no break, no continue, no exceptions, no goto, no while)
        - Loop boundaries, array indices and conditionals are affine expressions
          on parameters, constants and outer-loop iteration variables
        - All loops have a constant integer stride
        - All operations are side-effect free
        """

        poly_builder = PolyhedralBuilder(sdfg)
        if poly_builder.get_polyhedral_representation():
            return True
        else:
            return False

    @staticmethod
    def match_to_str(graph, candidate):
        return DetectLoop.match_to_str(graph, candidate)

    def apply(self, sdfg: sd.SDFG):
        # extract the polyhedral representation from the SDFG
        poly_builder = PolyhedralBuilder(sdfg)
        poly_builder.transform(
            exact_dependency_analysis=self.exact_dependency_analysis,
            use_pluto=True,
            tile_size=self.tile_size)

        free_symbols = sdfg.free_symbols.copy()
        input_arrays = {a.data for a in sdfg.input_arrays()}
        output_arrays = {a.data for a in sdfg.output_arrays()}

        for edge in sdfg.edges():
            sdfg.remove_edge(edge)

        for node in sdfg.nodes():
            sdfg.remove_node(node)

        for sym in {s for s in sdfg.symbols.keys()}:
            if sym not in free_symbols:
                sdfg.remove_symbol(sym)

        # rebuild the SDFG from the polyhedral representation
        if self.use_scheduler:
            # use the polyhedral representation with optimized control-flow
            poly_builder.rebuild_optimized_sdfg(sdfg,
                                                input_arrays,
                                                output_arrays,
                                                self.parallelize_loops,
                                                self.use_polytopes)
        else:
            # use the polyhedral representation with original control-flow
            poly_builder.rebuild_original_sdfg(sdfg,
                                               input_arrays,
                                               output_arrays,
                                               self.parallelize_loops,
                                               self.use_polytopes)