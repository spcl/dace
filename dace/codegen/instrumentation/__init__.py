from dace.dtypes import InstrumentationType
from .provider import InstrumentationProvider

from .papi import PAPIInstrumentation
from .timer import TimerProvider
from .cuda_events import CUDAEventProvider

INSTRUMENTATION_PROVIDERS = {
    InstrumentationType.No_Instrumentation: None,
    InstrumentationType.PAPI_Counters: PAPIInstrumentation,
    InstrumentationType.Timer: TimerProvider,
    InstrumentationType.CUDA_Events: CUDAEventProvider
}