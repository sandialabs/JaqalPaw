from .datatypes import ClockCycles, Discrete, Spline, Mixed
from .parameters import CLKFREQ
from ..bytecode.binary_conversion import (
    convert_freq_full,
    convert_phase_full,
    convert_amp_full,
)
from ..emulator.byte_decoding import (
    convert_freq_bytes_to_real,
    convert_phase_bytes_to_real,
    convert_amp_bytes_to_real,
)


def make_list_hashable(param):
    if isinstance(param, list):
        if any(
            map(lambda x: isinstance(x, (list, tuple, Discrete, Spline, Mixed)), param)
        ):
            return Mixed(map(make_list_hashable, param))
        return Discrete(param)
    elif isinstance(param, tuple):
        return Spline(param)
    elif param is None:
        return 0
    else:
        return param


def delist(x):
    """For use in pulse command, just ensures that a single value
    or a list with a single value is simply treated as that value"""
    if isinstance(x, list):
        return x[0]
    else:
        return x


def to_clock_cycles(dur):
    if isinstance(dur, ClockCycles):
        return dur
    return ClockCycles(dur * CLKFREQ)


def to_real_time(dur):
    if isinstance(dur, ClockCycles):
        return dur / CLKFREQ
    return dur


def discretize_amplitude(v):
    return convert_amp_bytes_to_real(convert_amp_full(v))


def discretize_frequency(v):
    return convert_freq_bytes_to_real(convert_freq_full(v))


def discretize_phase(v):
    return convert_phase_bytes_to_real(convert_phase_full(v))


def discretize_time(v):
    return to_real_time(to_clock_cycles(v))
