import dace
# import numba
import numpy
import time


nptype = numpy.float64
dctype = dace.float64


def pyf(x, y):
    return numpy.exp(x) * numpy.sqrt(y)
 

def pyrk4(x0, y0, x1, N, M):
    vx = numpy.ndarray([N + 1], dtype=numpy.float64)
    vy = numpy.ndarray([M, N + 1], dtype=numpy.float64)
    h = (x1 - x0) / N
    for j in range(M):
        vy[j, 0] = y0 + j * 0.1
        for i in range(N + 1):
            vx[i] = x0 + i * h
    for j in range(M):
        for i in range(1, N + 1):
            x = vx[i - 1]
            y = vy[j, i - 1]
            k1 = h * pyf(x, y)
            k2 = h * pyf(x + 0.5 * h, y + 0.5 * k1)
            k3 = h * pyf(x + 0.5 * h, y + 0.5 * k2)
            k4 = h * pyf(x + h, y + k3)
            y += (k1 + k2 + k2 + k3 + k3 + k4) / 6
            vy[j, i] = y
    return vx, vy


# @numba.jit(nopython=True)
# def numba_f(x, y):
#     return x * np.sqrt(y)


# @numba.jit(nopython=True)
# def numba_rk4(x0, y0, x1, n, vx, vy):
#     # vx = [0] * (n + 1)
#     # vy = [0] * (n + 1)
#     # vx = np.ndarray(n + 1, dtype=np.float64)
#     # vy = np.ndarray(n + 1, dtype=np.float64)
#     h = (x1 - x0) / float(n)
#     vx[0] = x = x0
#     vy[0] = y = y0
#     for i in range(1, n + 1):
#         k1 = h * numba_f(x, y)
#         k2 = h * numba_f(x + 0.5 * h, y + 0.5 * k1)
#         k3 = h * numba_f(x + 0.5 * h, y + 0.5 * k2)
#         k4 = h * numba_f(x + h, y + k3)
#         vx[i] = x = x0 + i * h
#         vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
#     return vx, vy


N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def dcf(x: dace.float64, y: dace.float64):
    return numpy.exp(x) * numpy.sqrt(y)
 

@dace.program
def dcrk4(x0: dace.float64, y0: dace.float64, x1: dace.float64):
    vx = numpy.ndarray([N + 1], dtype=numpy.float64)
    vy = numpy.ndarray([M, N + 1], dtype=numpy.float64)
    h = (x1 - x0) / N
    for j in range(M):
        vy[j, 0] = y0 + j * 0.1
        for i in range(N + 1):
            vx[i] = x0 + i * h
    for j in range(M):
        for i in range(1, N + 1):
            x = vx[i - 1]
            y = vy[j, i - 1]
            k1 = h * dcf(x, y)
            k2 = h * dcf(x + 0.5 * h, y + 0.5 * k1)
            k3 = h * dcf(x + 0.5 * h, y + 0.5 * k2)
            k4 = h * dcf(x + h, y + k3)
            y += (k1 + k2 + k2 + k3 + k3 + k4) / 6
            vy[j, i] = y
    return vx, vy


@dace.program
def dcrk4_map(x0: dace.float64, y0: dace.float64, x1: dace.float64):
    vx = numpy.ndarray([N + 1], dtype=numpy.float64)
    vy = numpy.ndarray([M, N + 1], dtype=numpy.float64)
    h = (x1 - x0) / N
    for j in range(M):
        vy[j, 0] = y0 + j * 0.1
        for i in range(N + 1):
            vx[i] = x0 + i * h
    for j in dace.map[0:M]:
        for i in range(1, N + 1):
            x = vx[i - 1]
            y = vy[j, i - 1]
            k1 = h * dcf(x, y)
            k2 = h * dcf(x + 0.5 * h, y + 0.5 * k1)
            k3 = h * dcf(x + 0.5 * h, y + 0.5 * k2)
            k4 = h * dcf(x + h, y + k3)
            y += (k1 + k2 + k2 + k3 + k3 + k4) / 6
            vy[j, i] = y
    return vx, vy


def benchmark(f, kwargs, label, num=10):
    runtimes = [0] * num
    for i in range(num):
        start = time.time()
        res = f(**kwargs)
        finish = time.time()
        runtimes[i] = finish - start
    print("{n} - mean: {a}s, median: {m}s".format(
        n=label, a=numpy.mean(runtimes), m=numpy.median(runtimes)))
    return res


if __name__ == "__main__":
    # N.set(1000000)
    # # vx, vy = dace_rk4(0.0, 1.0, 10.0, N=N)
    # sdfg = dcrk4.to_sdfg()
    # for datadesc in sdfg.arrays.values():
    #     datadesc.storage = dace.dtypes.StorageType.Register
    # exe =  sdfg.compile()
    # kw = {'x0': 0.0, 'y0': 1.0, 'x1': 10.0, 'N': N}
    # # sdfg = dace_rk4.to_sdfg()
    # times = [0] * 10
    # vx = np.ndarray(N.get() + 1, dtype=np.float64)
    # vy = np.ndarray(N.get() + 1, dtype=np.float64)
    # for i in range(10):
    #     start = time.time()
    #     # vx, vy = rk4(0, 1, 10, 100000)
    #     # numba_rk4(0, 1, 10, 100000, vx, vy)
    #     # vx, vy = sdfg(x0=0.0, y0=1.0, x1=10.0, N=N)
    #     vx, vy = exe(**kw)
    #     finish = time.time()
    #     times[i] = finish - start
    #     print("Runtime {}".format(finish - start))
    # print("Avg {}".format(np.mean(times)))
    # print("Median {}".format(np.median(times)))
    # for x, y in list(zip(vx, vy))[::1000]:
    #     print("%4.1f %10.5f %+12.4e" % (x, y, y - (4 + x * x)**2 / 16))

    N.set(100000)
    M.set(10)

    dcfunc = dcrk4.compile()
    dcfunc_map = dcrk4_map.compile()
    # dcsdfg_map = dcrk4_map.to_sdfg()
    # dcsdfg_map.apply_gpu_transformations()
    # dcsdfg_map.apply_strict_transformations()
    # dcfunc_gpu = dcsdfg_map.compile()

    pyres = benchmark(pyrk4,
                      {'x0': 0.0, 'y0': 1.0, 'x1': 10.0,
                       'N': N.get(), 'M': M.get()},
                      'python')
    dcres = benchmark(dcfunc,
                      {'x0': 0.0, 'y0': 1.0, 'x1': 10.0, 'N': N, 'M': M},
                      'dace')
    dcres_map = benchmark(dcfunc_map,
                          {'x0': 0.0, 'y0': 1.0, 'x1': 10.0, 'N': N, 'M': M},
                          'dace_map')

    print("Relative error for x: {}".format(
        numpy.linalg.norm(dcres[0] - pyres[0]) / numpy.linalg.norm(pyres[0])))
    print("Relative error for y: {}".format(
        numpy.linalg.norm(dcres[1] - pyres[1]) / numpy.linalg.norm(pyres[1])))
