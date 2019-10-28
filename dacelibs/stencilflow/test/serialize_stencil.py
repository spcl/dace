import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import stencilflow as sf
import dace

if __name__ == "__main__":

    accesses = [
        sf.Access("a", [-1, 0], "a_west"),
        sf.Access("a", [1, 0], "a_east"),
        sf.Access("b", [0, -1], "b_north"),
        sf.Access("b", [0, 1], "b_south"),
    ]

    code = "0.25 * (a_west + a_east + b_north + b_south)"

    boundary_conditions = {
      "a": {
					"type": "constant",
					"value": 1.0
      },
      "b": {
					"type": "constant",
					"value": 1.0
      }
    }

    sdfg = dace.SDFG("test_sdfg")
    state = sdfg.add_state("test_state")
    stencil = sf.Stencil("test_stencil", accesses, code, boundary_conditions)
    state.add_node(stencil)

    serialized = stencil.toJSON(state)
    deserialized = sf.Stencil.fromJSON_object(serialized)
