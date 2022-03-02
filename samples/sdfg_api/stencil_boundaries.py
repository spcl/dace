# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Example of a 7x7 stencil that dynamically generates custom boundary conditions executed in parallel. """

import dace
from dace import subsets
import numpy as np
from scipy import signal

H = dace.symbol('H')
W = dace.symbol('W')

STENCIL_KERNEL = np.random.rand(7, 7).astype(np.float32)


#################
# Helper function
def dirichlet_tasklet(state, B, x0, y0, width, height, initval=0):
    # Set up map that only has a write output, without any input. This generates an empty memlet from the map
    # entry to the tasklet automatically
    _, me, mx = state.add_mapped_tasklet('boundary',
                                         dict(i='%s:%s' % (y0, y0 + height), j='%s:%s' % (x0, x0 + width)), {},
                                         '''b = %f''' % initval,
                                         dict(b=dace.Memlet(data=B.data, subset='i,j')),
                                         external_edges=False)
    # Add the edge directly (i.e, without string parsing) using a Range object.
    # Notice that ranges are internally INCLUSIVE.
    out_subset = subsets.Range([(y0, y0 + height - 1, 1), (x0, x0 + width - 1, 1)])
    state.add_nedge(mx, B, dace.Memlet(data=B.data, subset=out_subset))


#################

sdfg = dace.SDFG('stencilboundaries')

# Add arrays and kernel
sdfg.add_array('A', [H, W], dace.float32)
sdfg.add_array('B', [H, W], dace.float32)
sdfg.add_constant('KERNEL', STENCIL_KERNEL)

mainstate = sdfg.add_state()

# The 7x7 stencil
_, me, mx = mainstate.add_mapped_tasklet('stencil',
                                         dict(i='3:H-3', j='3:W-3'),
                                         dict(a=dace.Memlet(data='A', subset='i-3:i+4, j-3:j+4')),
                                         '''
b = 0
for ky in range(7):
    for kx in range(7):
        b += a[ky, kx] * KERNEL[ky, kx]
                                        ''',
                                         dict(b=dace.Memlet(data='B', subset='i,j')),
                                         external_edges=False)

# Connect arrays (we want them to appear once for the main body and all bounds)
A = mainstate.add_read('A')
B = mainstate.add_write('B')
mainstate.add_nedge(A, me, dace.Memlet(data='A', subset='0:H, 0:W'))
mainstate.add_nedge(mx, B, dace.Memlet(data='B', subset='3:H-3, 3:W-3'))

# Add boundary conditions
dirichlet_tasklet(mainstate, B, 0, 0, 3, H)  # Left
dirichlet_tasklet(mainstate, B, W - 3, 0, 3, H)  # Right
dirichlet_tasklet(mainstate, B, 3, 0, W - 6, 3)  # Top
dirichlet_tasklet(mainstate, B, 3, H - 3, W - 6, 3)  # Bottom

# Fill in missing connectors on map nodes (necessary because `add_nedge` was used above).
# If `add_memlet_path` is used, this is not needed.
sdfg.fill_scope_connectors()

# Validate the correctness of the SDFG
sdfg.validate()

# NOTE: If GPUTransformSDFG is applied, boundary kernels will run on separate streams.
if __name__ == '__main__':
    H, W = 24, 24

    A = np.random.rand(H, W).astype(np.float32)
    B = np.random.rand(H, W).astype(np.float32)

    # Emulate same behavior as SDFG
    reg = np.zeros((H, W), dtype=np.float32)
    for i in range(3, H - 3):
        for j in range(3, W - 3):
            reg[i, j] = (A[i - 3:i + 4, j - 3:j + 4] * STENCIL_KERNEL).sum()

    sdfg(A=A, B=B, H=H, W=W)

    diff = np.linalg.norm(reg - B)
    print('Difference:', diff)
    exit(1 if diff >= 1e-4 else 0)
