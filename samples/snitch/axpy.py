# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace

N = dace.symbol('N')

def find_access_node_by_name(sdfg, name):
  """ Finds the first data node by the given name"""
  return next((n, s) for n, s in sdfg.all_nodes_recursive()
              if isinstance(n, dace.nodes.AccessNode) and name == n.data)
def find_map_by_name(sdfg, name):
    """ Finds the first map entry node by the given name """
    return next((n, s) for n, s in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and name == n.label)

@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y

if __name__ == "__main__":
  sdfg = axpy.to_sdfg()
  sdfg.specialize({ 'N': 1024})
  
  # Load elements of X and Y with SSR streamers
  find_access_node_by_name(sdfg, 'X')[0].desc(sdfg).storage = dace.dtypes.StorageType.Snitch_SSR
  find_access_node_by_name(sdfg, 'Y')[0].desc(sdfg).storage = dace.dtypes.StorageType.Snitch_SSR

  # Execute parallel
  find_map_by_name(sdfg, 'multiplication')[0].schedule = dace.ScheduleType.Snitch_Multicore

  # Generate the code
  from dace.codegen.targets.snitch import SnitchCodeGen
  code, header = SnitchCodeGen.gen_code_snitch(sdfg)

  # Write code to files
  with open(f"axpy.c", "w") as fd:
      fd.write(code)
  with open(f"axpy.h", "w") as fd:
      fd.write(header)
