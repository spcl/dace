import dace
from dace.perf.roofline import PerformanceSpec as Spec

# i5-5257U
# dual channel
PERF_CPU_CRAPBOOK = Spec(peak_bandwidth = 1.8667 * 64 * 2 / 8,
                         peak_performance = 2.7 * 4 * 2 * 4,
                         data_type = dace.float32, debug = False )

###################################

# AMD Ryzen 7 2700 8 core
# TODO: FMA / AVX same?
PERF_CPU_DAVINCI = Spec(peak_bandwidth = 2.000 * 64 * 4 / 8,     # ????
                        peak_performance = 3.2 * 4 * 2 * 8,
                        data_type = dace.float32, debug = False )

# Davinci GPU
# 2080 Ti
PERF_GPU_DAVINCI = Spec(peak_bandwidth = 616,
                        peak_performance = 13450,
                        data_type = dace.float32, debug = False )

PERF_GPU_DAVINCI_64 = Spec(peak_bandwidth = 616,
                           peak_performance = 420.2,
                           data_type = dace.float64, debug = False )

###################################

# Xeon W3680, sse4
PERF_CPU_MP = Spec(peak_bandwidth = 1.0667*64 * 2 / 8,
                   peak_performance = 3.2 * 8 * 2 * 2,
                   data_type = dace.float64, debug = False )

# GTX 970
PERF_GPU_MP = Spec(peak_bandwidth = 224,
                   peak_performance = 3920,
                   data_type = dace.float32, debug = False )

PERF_GPU_MP_64 = Spec(peak_bandwidth = 224,
                      peak_performance = 122.5,
                      data_type = dace.float64, debug = False )

###################################

PERF_GPU_AULT_V100 = Spec(peak_bandwidth = 897,
                           peak_performance = 14130,
                           data_type = dace.float32, debug = False )

PERF_GPU_AULT_V100_64 = Spec(peak_bandwidth = 897,
                          peak_performance = 7066,
                          data_type = dace.float64, debug = False )
