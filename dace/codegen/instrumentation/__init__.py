from dace.types import InstrumentationType
from .provider import InstrumentationProvider

from .papi import PAPIInstrumentation
from .timer import TimerProvider

INSTRUMENTATION_PROVIDERS = {
    InstrumentationType.No_Instrumentation: None,
    InstrumentationType.PAPI_Counters: PAPIInstrumentation,
    InstrumentationType.Timer: TimerProvider,
    InstrumentationType.CUDA_Events: None  # TODO
}