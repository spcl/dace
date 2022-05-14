import json
import dace
import numpy as np

from pathlib import Path

import dace.optimization.cutout_tuning as optim

I, J, K = (dace.symbol(s, dtype=dace.int64) for s in ('I', 'J', 'K'))

def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K))
    out_field = rng.random((I, J, K))
    coeff = rng.random((I, J, K))

    return in_field, out_field, coeff

@dace.program
def hdiff(in_field: dace.float64[I + 4, J + 4, K],
          out_field: dace.float64[I, J, K], coeff: dace.float64[I, J, K]):
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res1 = lap_field[1:, 1:J + 1, :] - lap_field[:I + 1, 1:J + 1, :]
    flx_field = np.where(
        (res1 *
         (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res1,
    )

    res2 = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :J + 1, :]
    fly_field = np.where(
        (res2 *
         (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res2,
    )
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])

I.set(256)
J.set(256)
K.set(160)

sdfg = hdiff.to_sdfg()
sdfg.specialize(dict(I=I,J=J,K=K))

in_field, out_field, coeff = initialize(I=256, J=256, K=160)

config_path = Path(__file__).parent / "tuning_config.json"
dreport = optim.CutoutTuner.dry_run(sdfg, in_field=in_field, out_field=out_field, coeff=coeff)

tuner = optim.CutoutTuner(sdfg, dreport, config_path)
tuner.tune()

with open("tuned_hdiff.sdfg", "w") as handle:
    json.dump(tuner._sdfg.to_json(), handle)
