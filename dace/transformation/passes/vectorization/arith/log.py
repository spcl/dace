import struct
import math
import dace


# Polynomial P(x) for log
def get_log_px(x: dace.float64) -> dace.float64:
    PX1log = 1.01875663804580931796E-4
    PX2log = 4.97494994976747001425E-1
    PX3log = 4.70579119878881725854E0
    PX4log = 1.44989225341610930846E1
    PX5log = 1.79368678507819816313E1
    PX6log = 7.70838733755885391666E0

    px = PX1log
    px *= x
    px += PX2log
    px *= x
    px += PX3log
    px *= x
    px += PX4log
    px *= x
    px += PX5log
    px *= x
    px += PX6log

    return px

# Polynomial Q(x) for log
def get_log_qx(x: dace.float64) -> dace.float64:
    QX1log = 1.12873587189167450590E1
    QX2log = 4.52279145837532221105E1
    QX3log = 8.29875266912776603211E1
    QX4log = 7.11544750618563894466E1
    QX5log = 2.31251620126765340583E1

    qx = x
    qx += QX1log
    qx *= x
    qx += QX2log
    qx *= x
    qx += QX3log
    qx *= x
    qx += QX4log
    qx *= x
    qx += QX5log

    return qx

# Natural logarithm for double precision
@dace.program
def dace_log_d(x: dace.float64, y: dace.float64):
    SQRTH = 0.70710678118654752440

    # Pack double into 64-bit binary
    u = struct.unpack('Q', struct.pack('d', x))[0]

    # Extract exponent
    exp = (u >> 52) & 0x7ff
    exp -= 1023  # remove bias

    fe = float(exp)

    # Set exponent to 0 (bias = 1023) to get mantissa in [0.5, 1)
    u = (u & 0x800fffffffffffff) | 0x3fe0000000000000
    mant = struct.unpack('d', struct.pack('Q', u))[0]

    # Blending
    if mant > SQRTH:
        fe += 1.0
    else:
        mant += mant
    mant -= 1.0

    # Rational form P(x)/Q(x)
    px = get_log_px(mant)
    x2 = mant * mant
    px *= mant
    px *= x2

    qx = get_log_qx(mant)

    res = px / qx

    res -= fe * 2.121944400546905827679e-4
    res -= 0.5 * x2

    res = mant + res
    res += fe * 0.693359375

    y = res

if __name__ == "__main__":
    sdfg = dace_log_d.to_sdfg()
    sdfg.save("dace_log.sdfg")