import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 16
poly: dc.uint16 = 0x8408

@dc.program
def crc16(data: dc.uint8[N], S: dc.float64[1]):

    crc: dc.uint16 = 0xFFFF
    for i in range(N):
        b = data[i]
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    S[0] = crc

sdfg = crc16.to_sdfg()

sdfg.save("log_sdfgs/crc16_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["data"], outputs=["S"])

sdfg.save("log_sdfgs/crc16_backward.sdfg")

